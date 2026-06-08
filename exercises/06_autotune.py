"""
Exercise 6: Autotuning Matrix Multiplication
===============================================
Learn to use @triton.autotune to automatically find optimal
BLOCK_M/N/K, num_warps, and num_stages.

Key concepts:
  - Config space: the Cartesian product of parameters to try
  - Pruning: key= parameter filters configs that can't run
  - Warmup + benchmark: triton.testing.do_bench handles this
  - Interpreting results: register pressure, occupancy, shared mem

Before autotune:
  @triton.jit
  def matmul_kernel(..., BLOCK_M: tl.constexpr, ...):
      ...

  matmul_kernel[grid](a, b, c, ..., BLOCK_M=64, BLOCK_N=64, BLOCK_K=32)

After autotune:
  @triton.autotune(configs=[...], key=['M','N','K'])
  @triton.jit
  def matmul_kernel(..., BLOCK_M: tl.constexpr, ...):
      ...

  matmul_kernel[grid](a, b, c, M, N, K, ...)
  # ↑ Notice: BLOCK_M/N/K are NO LONGER passed — autotune provides them

Run:  python exercises/06_autotune.py
      TRITON_PRINT_AUTOTUNING=1 python exercises/06_autotune.py  (see all configs)
"""

import torch
import triton
import triton.language as tl


# ═══════════════════════════════════════════════════════════
# TODO 1: Define the config space
# ═══════════════════════════════════════════════════════════
#
# triton.Config(kwargs, num_warps=..., num_stages=...)
#
# Rules of thumb for RTX 4070 SUPER (56 SMs, 128 threads/warp):
#   - num_warps: 4 or 8 (4 warps × 32 threads = 128 threads = 4 CTAs/SM possible)
#   - num_stages: 2-4 (pipeline stages for shared memory)
#   - BLOCK_M × BLOCK_N should fit in registers (~256KB/SM for fp32 acc)
#   - BLOCK_K: 32 or 64 (wider = more compute per load, less → more tiling)
#
# Fill in 8-12 configs that explore the space:
AUTOTUNE_CONFIGS = [
    # ── YOUR CODE: add configs ──
    # triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
    # triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
    # ...
    # ── END ──
]


# ═══════════════════════════════════════════════════════════
# TODO 2: Add pruning function
# ═══════════════════════════════════════════════════════════
#
# The KEY parameter: for each (M, N, K) triple, autotune calls all configs.
# Pruning eliminates configs that can't run or are known-bad.
#
# Example: if BLOCK_M > M, this config is useless — prune it.
# Example: if BLOCK_K > K, same thing.
#
# Return True to KEEP, False to PRUNE.
def prune_config(configs):
    """Called for each (M, N, K) to filter configs."""
    # ── YOUR CODE ──
    # You might want to keep only configs where BLOCK_M ≤ M, etc.
    # or keep all configs for now and let the benchmark sort them out
    return configs  # no pruning for now
    # ── END ──


# ═══════════════════════════════════════════════════════════
# TODO 3: Add @triton.autotune decorator
# ═══════════════════════════════════════════════════════════
#
# Syntax:
#   @triton.autotune(
#       configs=AUTOTUNE_CONFIGS,
#       key=['M', 'N', 'K'],          # parameters to cache tuning results by
#       prune_configs_by={'early_config_prune': prune_config}
#   )
#   @triton.jit
#   def matmul_kernel(...):
#       ...
#
# NOTE: After adding autotune, do NOT pass BLOCK_M/N/K as grid arguments.
# The decorator injects them from the winning config.

@triton.jit
def matmul_autotuned_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # ── IMPORTANT: constexprs go AFTER dynamic args ──
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Tiled matmul — same algorithm as Exercise 3.
    The only difference: BLOCK sizes come from autotune.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        rk = k + tl.arange(0, BLOCK_K)
        a = tl.load(
            a_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak,
            mask=(rm[:, None] < M) & (rk[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            b_ptr + rk[:, None] * stride_bk + rn[None, :] * stride_bn,
            mask=(rk[:, None] < K) & (rn[None, :] < N),
            other=0.0,
        )
        acc += tl.dot(a, b)

    c_mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(
        c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn,
        acc,
        mask=c_mask,
    )


def matmul_autotuned(a: torch.Tensor, b: torch.Tensor):
    """
    Call the autotuned matmul.
    
    IMPORTANT differences from manual kernel:
    1. grid is (M, N, 1) — the 1 lets Triton use stream-k if available
    2. BLOCK_M/N/K are NOT passed — autotune injects them
    3. M, N, K ARE passed as runtime args (needed for key= matching)
    """
    M, K_input = a.shape
    K2, N = b.shape
    assert K_input == K2, f"K dimension mismatch: {K_input} vs {K2}"

    c = torch.empty(M, N, device=a.device, dtype=a.dtype)

    # ── YOUR CODE: call matmul_autotuned_kernel ──
    # grid = ?
    # matmul_autotuned_kernel[grid](a, b, c, M, N, K_input, ...)
    # ── END ──
    
    return c


# ═══════════════════════════════════════════════════════════
# BENCHMARK (runs your kernel vs PyTorch)
# ═══════════════════════════════════════════════════════════

def benchmark():
    """
    Run autotuned matmul across shapes and compare vs PyTorch (cuBLAS).
    
    On first run, the autotuner tests all configs (slow — ~30s).
    On subsequent runs, it uses cached results.
    """
    shapes = [
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
    ]

    print("=" * 70)
    print("  MATMUL AUTOTUNE BENCHMARK")
    print("=" * 70)
    print()

    for M, N, K in shapes:
        a = torch.randn(M, K, device="cuda", dtype=torch.float16)
        b = torch.randn(K, N, device="cuda", dtype=torch.float16)

        # Correctness check
        ref = a @ b
        try:
            yours = matmul_autotuned(a, b)
            max_err = (yours.float() - ref.float()).abs().max().item()
            status = "✓" if max_err < 0.02 else "✗"
            print(f"  {M}×{K}×{N}: max_err={max_err:.6f} {status}")
        except Exception as e:
            print(f"  {M}×{K}×{N}: ERROR — {e}")
            continue

        # Performance
        torch_ms = triton.testing.do_bench(lambda: a @ b, rep=50)
        triton_ms = triton.testing.do_bench(
            lambda: matmul_autotuned(a, b), rep=50
        )

        gflops = 2 * M * N * K * 1e-9
        triton_tflops = gflops / (triton_ms * 1e-3) / 1000
        torch_tflops = gflops / (torch_ms * 1e-3) / 1000

        print(f"           PyTorch: {torch_ms:.3f}ms ({torch_tflops:.1f} TFLOPS)")
        print(f"           Triton:  {triton_ms:.3f}ms ({triton_tflops:.1f} TFLOPS)")
        print()


# ═══════════════════════════════════════════════════════════
# EXERCISE QUESTIONS
# ═══════════════════════════════════════════════════════════
#
# After running with TRITON_PRINT_AUTOTUNING=1:
#
# 1. Which config won for 1024³? For 4096³? Are they the same?
#    Why would different shapes prefer different configs?
#
# 2. Look at the losing configs. Why did they lose?
#    - Too many registers → low occupancy?
#    - Shared memory too large → fewer CTAs per SM?
#    - BLOCK too small → not enough parallelism?
#
# 3. How does num_warps=8 change things vs num_warps=4?
#    Hint: 4 warps × 4 CTAs = 16 warps/SM. 8 warps × 2 CTAs = also 16.
#    But bigger blocks → more registers → fewer concurrent blocks.
#
# 4. Add GROUP_M swizzling (from 03_matmul_optimized) to the autotuned kernel.
#    Does it help for certain shapes?
#
# 5. Try num_stages=2 vs num_stages=4. What's the trade-off?
#    (More stages = better latency hiding, more shared memory usage)


if __name__ == "__main__":
    benchmark()
