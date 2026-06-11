"""
Exercise 7: LayerNorm & RMSNorm — Your First Fused Kernel
===========================================================

So far every exercise was one operation per kernel: matmul, softmax, etc.
Now we fuse multiple ops into ONE kernel to eliminate intermediate
global memory round-trips.

Before fusion:                  After fusion:
  load x                        load x
  compute mean → store          compute mean (in SRAM)
  load mean                     compute var  (in SRAM, reusing x)
  compute var → store           normalize    (in SRAM)
  load var                      affine       (weight * x + bias)
  normalize → store             store y      ← ONE write!
  load normalized
  affine → store
  ↑ 6 global mem ops            ↑ 2 global mem ops

The fusion patterns here — reduce → broadcast → element-wise — are the
building blocks for every fused kernel you'll ever write.

LAYERNORM (forward):
  For each row x ∈ R^H:
    μ = (1/H) Σ x_i
    σ² = (1/H) Σ (x_i - μ)²
    x̂_i = (x_i - μ) / √(σ² + ε)
    y_i = γ_i * x̂_i + β_i          ← affine (weight + bias)

RMSNORM (forward, used in LLaMA/Mistral):
  For each row x ∈ R^H:
    rms = √( (1/H) Σ x_i² )
    x̂_i = x_i / (rms + ε)
    y_i = γ_i * x̂_i                 ← no bias in standard RMSNorm

Run:  python exercises/07_layer_norm.py
"""

import torch
import triton
import triton.language as tl


# ═══════════════════════════════════════════════════════════
# TODO 1: Fused LayerNorm forward
# ═══════════════════════════════════════════════════════════
#
# Strategy: each program handles one row.
#   Step 1: Load the row into SRAM
#   Step 2: Compute mean (reduction over H)
#   Step 3: Compute variance (reduction over H, reusing loaded data)
#   Step 4: Normalize + affine in one pass
#   Step 5: Store result
#
# For simplicity, assume hidden_dim fits in one block (BLOCK_SIZE).
# This covers many real cases (hidden_dim=768, 1024, 4096 are all ≤ 4096).

@triton.jit
def layer_norm_fused_kernel(
    x_ptr,          # (N, H) input
    weight_ptr,     # (H,) affine weight γ
    bias_ptr,       # (H,) affine bias β
    y_ptr,          # (N, H) output
    N,              # number of rows (batch * seq_len)
    H,              # hidden dimension
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused LayerNorm: mean → var → normalize → affine, all in one kernel.
    One program per row. H must be ≤ BLOCK_SIZE.
    """
    pid = tl.program_id(0)          # row index
    if pid >= N:
        return

    # ── YOUR CODE: Step 1 — load one full row ──
    # offs = tl.arange(0, BLOCK_SIZE)
    # mask = offs < H
    # x = tl.load(x_ptr + pid * H + offs, mask=mask, other=0.0)

    # ── YOUR CODE: Step 2 — compute mean μ ──
    # μ = tl.sum(x, axis=0) / H    ← axis=0 sums the BLOCK_SIZE vector into a scalar

    # ── YOUR CODE: Step 3 — compute variance σ² ──
    # diff = x - μ                  ← μ is scalar, broadcasts automatically
    # σ² = tl.sum(diff * diff, axis=0) / H

    # ── YOUR CODE: Step 4 — normalize ──
    # x_hat = diff / tl.sqrt(σ² + eps)

    # ── YOUR CODE: Step 5 — affine transform ──
    # w = tl.load(weight_ptr + offs, mask=mask, other=0.0)
    # b = tl.load(bias_ptr + offs, mask=mask, other=0.0)
    # y = x_hat * w + b

    # ── YOUR CODE: Step 6 — store ──
    # tl.store(y_ptr + pid * H + offs, y, mask=mask)

    pass


# ═══════════════════════════════════════════════════════════
# TODO 2: RMSNorm forward
# ═══════════════════════════════════════════════════════════
#
# RMSNorm is simpler: no mean subtraction, just RMS normalization.
# Used in LLaMA, Mistral, and most recent LLMs because:
#   - One less reduction → faster
#   - Empirically, re-centering (mean subtraction) doesn't help training

@triton.jit
def rms_norm_fused_kernel(
    x_ptr, weight_ptr, y_ptr,
    N, H,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused RMSNorm: rms → normalize → affine.
    One program per row.
    """
    pid = tl.program_id(0)
    if pid >= N:
        return

    # ── YOUR CODE ──
    # Steps:
    #   1. Load row x
    #   2. Compute rms = sqrt(sum(x²) / H)
    #   3. Normalize: x_hat = x / (rms + eps)
    #   4. Load weight, apply: y = x_hat * weight
    #   5. Store y
    #
    # Key difference from LayerNorm: only ONE reduction (sum of squares),
    # no mean subtraction needed.

    pass


# ═══════════════════════════════════════════════════════════
# TODO 3 (advanced): Tiled reduction for long H
# ═══════════════════════════════════════════════════════════
#
# What if H > BLOCK_SIZE? You can't load the whole row at once.
# Need TWO passes:
#   Pass 1: tile over H, accumulate partial sums for mean/var
#   Pass 2: tile over H again, normalize + affine using the full-row stats
#
# This is the same "online" pattern from softmax (Ex02):
# loop-carried accumulators that aggregate partial results.

@triton.jit
def layer_norm_tiled_kernel(
    x_ptr, weight_ptr, bias_ptr, y_ptr,
    N, H,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Tiled LayerNorm for arbitrary H. Two passes:
    - First pass: accumulate sum_x and sum_x2
    - Second pass: normalize + affine
    """
    pid = tl.program_id(0)
    if pid >= N:
        return

    # ── YOUR CODE ──
    # Hints:
    #   - Use two loop-carried scalars: sum_x = 0.0, sum_x2 = 0.0
    #   - First loop: for off in range(0, H, BLOCK_SIZE):
    #       load tile, sum_x += tl.sum(tile), sum_x2 += tl.sum(tile * tile)
    #   - μ = sum_x / H, σ² = sum_x2 / H - μ * μ  ← parallel variance formula
    #   - Second loop: for off in range(0, H, BLOCK_SIZE):
    #       load tile, load weight/bias tile, normalize, affine, store
    #
    # The parallel formula σ² = E[X²] - E[X]² avoids needing μ in the first pass.
    # But it's numerically less stable — for now, it's fine for learning.

    pass


# ═══════════════════════════════════════════════════════════
# WRAPPERS — call the kernels
# ═══════════════════════════════════════════════════════════

def layer_norm_fused(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
                     eps: float = 1e-5) -> torch.Tensor:
    """Fused LayerNorm: all ops in one kernel, no intermediate tensors."""
    N, H = x.shape
    y = torch.empty_like(x)

    # ── YOUR CODE: choose BLOCK_SIZE and launch the kernel ──
    # BLOCK_SIZE should be ≥ H. Common choices: 1024, 2048, 4096.
    # grid = (N,)   ← one program per row
    # layer_norm_fused_kernel[grid](x, weight, bias, y, N, H, eps=eps, BLOCK_SIZE=...)

    return y


def rms_norm_fused(x: torch.Tensor, weight: torch.Tensor,
                   eps: float = 1e-5) -> torch.Tensor:
    """Fused RMSNorm."""
    N, H = x.shape
    y = torch.empty_like(x)

    # ── YOUR CODE ──
    # grid = (N,)
    # rms_norm_fused_kernel[grid](x, weight, y, N, H, eps=eps, BLOCK_SIZE=...)

    return y


# ═══════════════════════════════════════════════════════════
# BENCHMARK
# ═══════════════════════════════════════════════════════════

def benchmark():
    print("=" * 70)
    print("  LAYERNORM & RMSNORM BENCHMARK")
    print("=" * 70)
    print()

    # Test configs
    configs = [
        # (batch×seq, hidden_dim)
        (1024, 768),      # BERT-base
        (1024, 1024),     # GPT-2 small
        (2048, 4096),     # LLaMA-7B
        (4096, 4096),     # LLaMA-7B long seq
    ]

    for N, H in configs:
        x = torch.randn(N, H, device="cuda", dtype=torch.float16)
        w = torch.randn(H, device="cuda", dtype=torch.float16)
        b = torch.randn(H, device="cuda", dtype=torch.float16)

        # Reference (PyTorch native, unfused)
        ref_ln = torch.nn.functional.layer_norm(x.float(), (H,), w.float(), b.float(), 1e-5).half()
        ref_rms = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-5) * w.float()
        ref_rms = ref_rms.half()

        # Check correctness (skip if kernel not yet implemented)
        try:
            # LayerNorm
            your_ln = layer_norm_fused(x, w, b)
            ln_ok = torch.allclose(your_ln.float(), ref_ln.float(), rtol=1e-2, atol=1e-1)
            ln_status = "✓" if ln_ok else "✗"
            max_err_ln = (your_ln.float() - ref_ln.float()).abs().max().item()

            # RMSNorm
            your_rms = rms_norm_fused(x, w)
            rms_ok = torch.allclose(your_rms.float(), ref_rms.float(), rtol=1e-2, atol=1e-1)
            rms_status = "✓" if rms_ok else "✗"
            max_err_rms = (your_rms.float() - ref_rms.float()).abs().max().item()

            print(f"  ({N}, {H}):  LN max_err={max_err_ln:.6f} {ln_status}"
                  f"  |  RMS max_err={max_err_rms:.6f} {rms_status}")
        except NotImplementedError:
            print(f"  ({N}, {H}):  Kernel not yet implemented — fill in the TODOs")
        except Exception as e:
            print(f"  ({N}, {H}):  ERROR — {e}")

        # Performance
        try:
            torch_ln_ms = triton.testing.do_bench(lambda: torch.nn.functional.layer_norm(x, (H,), w, b, 1e-5), rep=100)
            triton_ln_ms = triton.testing.do_bench(lambda: layer_norm_fused(x, w, b), rep=100)
            speedup_ln = torch_ln_ms / triton_ln_ms

            torch_rms_ms = triton.testing.do_bench(lambda: torch.nn.functional.rms_norm(x, (H,), w, 1e-5), rep=100)
            triton_rms_ms = triton.testing.do_bench(lambda: rms_norm_fused(x, w), rep=100)
            speedup_rms = torch_rms_ms / triton_rms_ms

            print(f"           LN:  PyTorch {torch_ln_ms*1000:.1f}us  Triton {triton_ln_ms*1000:.1f}us  "
                  f"({speedup_ln:.1f}x)")
            print(f"           RMS: PyTorch {torch_rms_ms*1000:.1f}us  Triton {triton_rms_ms*1000:.1f}us  "
                  f"({speedup_rms:.1f}x)")
        except:
            pass
        print()


# ═══════════════════════════════════════════════════════════
# EXERCISE QUESTIONS
# ═══════════════════════════════════════════════════════════
#
# 1. Why does RMSNorm have one fewer reduction than LayerNorm?
#    How does this affect performance on GPU?
#
# 2. The tiled LayerNorm (TODO 3) uses E[X²] - E[X]² for variance.
#    Look up why this is numerically unstable for certain inputs.
#    How does PyTorch's implementation handle this? (Hint: Welford)
#
# 3. Notice we didn't use tl.dot() here. Why? What kind of operations
#    is LayerNorm dominated by — compute or memory? How do you know?
#
# 4. Try extending the kernel to handle the BACKWARD pass (gradient).
#    What intermediate values from forward do you need to save?
#
# 5. PyTorch's torch.nn.functional.layer_norm already uses a fused CUDA
#    kernel. Why is our Triton version sometimes slower? What's different
#    about how PyTorch schedules the work?


if __name__ == "__main__":
    benchmark()
