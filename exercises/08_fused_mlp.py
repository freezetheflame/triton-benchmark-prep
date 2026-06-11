"""
Exercise 8: Fused MLP — Operator Fusion at Scale
==================================================

The MLP block in a Transformer is:

    x → Linear(W1, b1) → GELU → Linear(W2, b2) → y

Without fusion, this creates a MASSIVE intermediate tensor:
  - Shape: (batch*seq_len × intermediate_dim)
  - For LLaMA-7B (seq=4096): 4096 × 11008 = 45M elements
  - In fp16: 90 MB per layer, 32 layers = 2.9 GB wasted intermediate memory

Fusion eliminates this tensor entirely: the output of the first matmul
never leaves SRAM. It goes straight into GELU, then into the second matmul.

This exercise has TWO parts:
  Part A (warm-up):  Fused Linear + Bias + GELU  (one matmul, easy)
  Part B (real deal): Full Fused MLP              (two matmuls, hard)

Run:  python exercises/08_fused_mlp.py
"""

import torch
import triton
import triton.language as tl


# ═══════════════════════════════════════════════════════════
# PART A: Fused Linear + Bias + GELU
# ═══════════════════════════════════════════════════════════
#
# Standard approach:
#   1. Compute C = A @ B         (matmul, store C to global memory)
#   2. Load C, apply bias, GELU  (element-wise, store result)
#
# Fused approach:
#   Compute a TILE of matmul in SRAM → apply bias + GELU in registers
#   → store the FINAL result (never store intermediate matmul result)

@triton.jit
def fused_linear_gelu_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Fused: matmul(A, B) + bias + GELU.
    Each program computes a BLOCK_M × BLOCK_N tile of output.

    Key difference from Exercise 3 (basic matmul):
    After the K-loop, instead of storing the accumulator directly,
    apply bias and GELU element-wise before storing.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # ── YOUR CODE: K-loop matmul (same as Ex03) ──
    # acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    # for k in range(0, K, BLOCK_K):
    #     rk = k + tl.arange(0, BLOCK_K)
    #     a = tl.load(...)
    #     b = tl.load(...)
    #     acc += tl.dot(a, b)
    #
    # ── YOUR CODE: bias + GELU (the fusion part!) ──
    # bias_tile = tl.load(bias_ptr + rn, mask=rn < N, other=0.0)
    # acc = acc + bias_tile[None, :]     ← broadcast bias over rows
    # acc = gelu(acc)                    ← apply activation in-place on acc
    #
    # ── YOUR CODE: store ──
    # mask = (rm[:, None] < M) & (rn[None, :] < N)
    # tl.store(...)

    pass


# Helper: GELU in Triton
@triton.jit
def gelu(x):
    """GELU activation: x * Φ(x) where Φ is the Gaussian CDF."""
    # tanh approximation (used in GPT-2, BERT)
    # GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
    sqrt_2_over_pi = 0.7978845608028654
    coeff = 0.044715
    x3 = x * x * x
    inner = sqrt_2_over_pi * (x + coeff * x3)
    return 0.5 * x * (1.0 + tl.tanh(inner))


# ═══════════════════════════════════════════════════════════
# PART B: Full Fused MLP — Two Matmuls, Zero Intermediate
# ═══════════════════════════════════════════════════════════
#
# The MLP:  h = GELU(x @ W1 + b1)
#           y = h @ W2 + b2
#
# Without fusion:
#   h (M × I) is stored to global memory → 90 MB waste for LLaMA-7B
#
# With fusion:
#   For each tile of the FINAL output (M × O):
#     1. Loop over blocks of the intermediate dimension I
#     2. Load W1 tile → compute partial h → GELU in registers
#     3. Multiply with W2 tile → accumulate into y tile
#     4. Apply b2 → store
#
# The trick: h never exists as a full tensor. Only TILES of h exist,
# each tile consumed immediately by the W2 matmul and then discarded.

@triton.jit
def fused_mlp_kernel(
    # ── Pointers ──
    x_ptr,          # (M, K) input
    w1_ptr,         # (K, I) first weight
    b1_ptr,         # (I,)  first bias
    w2_ptr,         # (I, O) second weight
    b2_ptr,         # (O,)  second bias
    y_ptr,          # (M, O) output
    # ── Dimensions ──
    M, K, I, O,     # batch*seq_len, input_dim, intermediate_dim, output_dim
    # ── Strides ──
    stride_xm, stride_xk,
    stride_w1k, stride_w1i,
    stride_w2i, stride_w2o,
    stride_ym, stride_yo,
    # ── Block sizes ──
    BLOCK_M: tl.constexpr,
    BLOCK_O: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    """
    Full fused MLP: two matmuls, GELU, both biases — all in one kernel.

    Each program computes a BLOCK_M × BLOCK_O tile of the FINAL output.

    The intermediate dimension I is tiled: for each I-block, we compute a
    partial h (GELU'd in registers), then immediately multiply it into the
    W2 output accumulator. h tiles are discarded after use.
    """
    pid_m = tl.program_id(0)
    pid_o = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    ro = pid_o * BLOCK_O + tl.arange(0, BLOCK_O)

    # Accumulator for FINAL output (M × O tile)
    acc_out = tl.zeros([BLOCK_M, BLOCK_O], dtype=tl.float32)

    # ── YOUR CODE: loop over intermediate dimension I ──
    # for i_start in range(0, I, BLOCK_I):
    #     ri = i_start + tl.arange(0, BLOCK_I)
    #
    #     # Step 1: Load x tile (M×K) — only needed for first matmul
    #     #          Actually, x is the INPUT to W1. For each I-block,
    #     #          we need x[m, :] @ W1[:, i:i+BLOCK_I] → partial h[m, i:i+BLOCK_I]
    #     #
    #     # This requires looping over K too! Double loop: I-outer, K-inner.
    #     #
    #     # Revised structure:
    #     #   h_partial = zeros([BLOCK_M, BLOCK_I])
    #     #   for k in range(0, K, BLOCK_K):
    #     #       load x[m, k:k+BLOCK_K], load w1[k:k+BLOCK_K, i:i+BLOCK_I]
    #     #       h_partial += dot(x_tile, w1_tile)
    #     #   h_partial = h_partial + b1[ri]  → GELU
    #     #   h_gelu = gelu(h_partial)
    #     #
    #     # Step 2: Multiply h_gelu with w2[i:i+BLOCK_I, o] → acc_out
    #     #   load w2[ri, ro]
    #     #   acc_out += dot(h_gelu, w2_tile)
    #
    # # After I-loop: acc_out = acc_out + b2
    # # Store y[m, o] = acc_out

    pass


# ═══════════════════════════════════════════════════════════
# WRAPPERS
# ═══════════════════════════════════════════════════════════

def fused_linear_gelu(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Fused: matmul + bias + GELU."""
    M, K = x.shape
    K2, N = w.shape
    assert K == K2

    c = torch.empty(M, N, device=x.device, dtype=x.dtype)

    # ── YOUR CODE ──
    # grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    # fused_linear_gelu_kernel[grid](x, w, bias, c, M, N, K, ...)

    return c


def fused_mlp(x: torch.Tensor, w1: torch.Tensor, b1: torch.Tensor,
              w2: torch.Tensor, b2: torch.Tensor) -> torch.Tensor:
    """Full fused MLP: x @ w1 + b1 → GELU → @ w2 + b2."""
    M, K = x.shape
    K2, I = w1.shape
    I2, O = w2.shape
    assert K == K2 and I == I2

    y = torch.empty(M, O, device=x.device, dtype=x.dtype)

    # ── YOUR CODE ──
    # grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(O, BLOCK_O))
    # fused_mlp_kernel[grid](x, w1, b1, w2, b2, y, M, K, I, O, ...)

    return y


# ═══════════════════════════════════════════════════════════
# BENCHMARK
# ═══════════════════════════════════════════════════════════

def benchmark():
    print("=" * 70)
    print("  FUSED MLP BENCHMARK")
    print("=" * 70)
    print()

    configs = [
        # (M, K, I, O) — typical Transformer MLP shapes
        (512,  1024, 4096, 1024),   # GPT-2 medium, short seq
        (2048, 4096, 11008, 4096),  # LLaMA-7B, seq=2048
    ]

    for M, K, I, O in configs:
        x = torch.randn(M, K, device="cuda", dtype=torch.float16)
        w1 = torch.randn(K, I, device="cuda", dtype=torch.float16)
        b1 = torch.randn(I, device="cuda", dtype=torch.float16)
        w2 = torch.randn(I, O, device="cuda", dtype=torch.float16)
        b2 = torch.randn(O, device="cuda", dtype=torch.float16)

        # Reference
        h = torch.nn.functional.linear(x, w1.T, b1)
        h = torch.nn.functional.gelu(h)
        ref = torch.nn.functional.linear(h, w2.T, b2)

        print(f"  Shape: M={M}, K={K}, I={I}, O={O}")
        intermed_mb = M * I * 2 / 1e6  # fp16 intermediate in MB
        print(f"  Intermediate tensor: {intermed_mb:.0f} MB (eliminated by fusion)")

        # Test Part A
        try:
            yours_a = fused_linear_gelu(x, w1, b1)
            ok_a = torch.allclose(yours_a.float(), h.float(), rtol=1e-2, atol=1e-1)
            print(f"  Part A (Linear+GELU): {'✓' if ok_a else '✗'}")
        except NotImplementedError:
            print(f"  Part A (Linear+GELU): TODO — fill in the kernel")
        except Exception as e:
            print(f"  Part A (Linear+GELU): ERROR — {e}")

        # Test Part B
        try:
            yours_b = fused_mlp(x, w1, b1, w2, b2)
            ok_b = torch.allclose(yours_b.float(), ref.float(), rtol=1e-2, atol=1e-1)
            print(f"  Part B (Full MLP):   {'✓' if ok_b else '✗'}")
        except NotImplementedError:
            print(f"  Part B (Full MLP):   TODO — fill in the kernel")
        except Exception as e:
            print(f"  Part B (Full MLP):   ERROR — {e}")

        # Performance (only if implemented)
        try:
            def unfused():
                h = torch.nn.functional.linear(x, w1.T, b1)
                h = torch.nn.functional.gelu(h)
                return torch.nn.functional.linear(h, w2.T, b2)

            unfused_ms = triton.testing.do_bench(unfused, rep=50)
            fused_ms = triton.testing.do_bench(lambda: fused_mlp(x, w1, b1, w2, b2), rep=50)
            print(f"  Unfused: {unfused_ms*1000:.0f}us  Fused: {fused_ms*1000:.0f}us  "
                  f"({unfused_ms/fused_ms:.1f}x)")
        except:
            pass
        print()


# ═══════════════════════════════════════════════════════════
# EXERCISE QUESTIONS
# ═══════════════════════════════════════════════════════════
#
# 1. Part B has a DOUBLE loop (over I and K). Why can't we do the
#    matmul in one shot? What's the relationship between BLOCK_I
#    and BLOCK_K in terms of shared memory usage?
#
# 2. The intermediate h tile is in fp32 (accumulator) but GELU
#    expects fp32 input. What precision should h be in when we
#    pass it to the W2 matmul? Should we convert to fp16 first?
#
# 3. Compare the unfused vs fused memory footprint. How many bytes
#    of global memory traffic does each approach generate?
#
# 4. The fused MLP trades MORE computation (re-reading x tiles
#    for each I-block) for LESS memory (no h tensor). When is this
#    trade-off worth it? When is it NOT?
#
# 5. Try extending to SiLU (Swish) activation. What changes?
#
# 6. (Hard) Current approach re-reads x for every I-block. How
#    could you restructure the tiling to avoid this? Hint: think
#    about making the O-loop the outermost instead of the I-loop.


if __name__ == "__main__":
    benchmark()
