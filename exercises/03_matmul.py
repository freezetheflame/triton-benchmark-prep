"""
Exercise 3: Matrix Multiplication
===================================
Implement matmul in Triton.

Key concepts:
  - Tiling: split matrices into BLOCK_M × BLOCK_K and BLOCK_K × BLOCK_N tiles
  - tl.dot: Tensor Core accelerated matrix multiply
  - Grid: 2D — (cdiv(M, BLOCK_M), cdiv(N, BLOCK_N))
  - Accumulator dtype: always float32, even when inputs are float16

For row-major tensors:
  A[M,K]: stride_am = K, stride_ak = 1
  B[K,N]: stride_bk = N, stride_bn = 1
  C[M,N]: stride_cm = N, stride_cn = 1

Pseudocode:
  pid_m = program_id(0), pid_n = program_id(1)
  rm = pid_m * BLOCK_M + arange(0, BLOCK_M)   # [BLOCK_M]
  rn = pid_n * BLOCK_N + arange(0, BLOCK_N)   # [BLOCK_N]
  acc = zeros[BLOCK_M, BLOCK_N]

  for k in range(0, K, BLOCK_K):
    rk = k + arange(0, BLOCK_K)
    a = load A[rm, rk]                        # [BLOCK_M, BLOCK_K]
    b = load B[rk, rn]                        # [BLOCK_K, BLOCK_N]
    acc += tl.dot(a, b)

  store C[rm, rn] = acc
"""

import torch
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Compute C[M,N] = A[M,K] @ B[K,N]"""
    # ── YOUR CODE HERE ──
    # 1. pid_m = tl.program_id(0), pid_n = tl.program_id(1)
    # 2. rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # 3. rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    # 4. acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    # 5. Loop k from 0 to K step BLOCK_K:
    #      rk = k + tl.arange(0, BLOCK_K)
    #      a = tl.load(a_ptr + rm[:,None]*stride_am + rk[None,:]*stride_ak, ...)
    #      b = tl.load(b_ptr + rk[:,None]*stride_bk + rn[None,:]*stride_bn, ...)
    #      acc += tl.dot(a, b)
    # 6. mask and store to c_ptr
    # ── END ──
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
      rk = k + tl.arange(0, BLOCK_K)
      a_off = a_ptr + (rm[:, None] * stride_am + rk[None, :] * stride_ak)
      b_off = b_ptr + (rk[:, None] * stride_bk + rn[None, :] * stride_bn)
      a_mask = (rm[:, None] < M) & (rk[None, :] < K)
      b_mask = (rk[:, None] < K) & (rn[None, :] < N)
      a = tl.load(a_off, mask=a_mask, other=0.0)
      b = tl.load(b_off, mask=b_mask, other=0.0)
      acc += tl.dot(a, b)

    c_mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(c_ptr + (rm[:, None] * stride_cm + rn[None, :] * stride_cn), acc, mask=c_mask)

def matmul(a: torch.Tensor, b: torch.Tensor, BLOCK_M=64, BLOCK_N=64, BLOCK_K=32):
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    c = torch.empty(M, N, device=a.device, dtype=a.dtype)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    matmul_kernel[grid](a, b, c, M, N, K,
                         a.stride(0), a.stride(1),
                         b.stride(0), b.stride(1),
                         c.stride(0), c.stride(1),
                         BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
    return c


# ─── Test ───────────────────────────────────────────────────
def test_matmul():
    print("Testing correctness...")
    for M, N, K in [(64, 64, 64), (128, 128, 128), (256, 256, 256),
                     (511, 513, 517), (128, 256, 64)]:
        a = torch.randn(M, K, device='cuda', dtype=torch.float32)
        b = torch.randn(K, N, device='cuda', dtype=torch.float32)
        ref = a @ b
        c = matmul(a, b)
        # tl.dot uses TF32 internally, ~2% error is normal
        assert torch.allclose(c, ref, atol=5e-2, rtol=1e-2), \
            f"Failed at {M}x{K}x{N}: max_diff={(c-ref).abs().max():.4f}"
        print(f"  {M}x{K}x{N}: ✓")


if __name__ == "__main__":
    test_matmul()
