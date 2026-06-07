"""
Exercise 4: FlashAttention Forward Pass
=========================================
Implement the FlashAttention forward pass in a single Triton kernel.

This is the hardest exercise. Make sure you understand:
  1. Online softmax (see notes/online-softmax.md)
  2. How Q, K, V are tiled (see notes/flash-attention-theory.md)
  3. Triton blocking: BLOCK_M rows of Q, BLOCK_N rows of K/V

Algorithm (for one Q block):
  m = [-inf] * BLOCK_M       # running max per query row
  d = [0] * BLOCK_M          # running denominator
  acc = zeros[BLOCK_M, head_dim]

  For each K,V block:
    S = Q_block @ K_block^T * sm_scale           # [BLOCK_M, BLOCK_N]
    m_new = max(m, max(S, axis=1))               # [BLOCK_M]
    P = exp(S - m_new[:, None])                  # [BLOCK_M, BLOCK_N]
    d_new = d * exp(m - m_new) + sum(P, axis=1)  # [BLOCK_M]
    acc = acc * (d / d_new)[:, None] * exp(m - m_new)[:, None]
    acc += (P @ V_block) / d_new[:, None]
    m, d = m_new, d_new

  Store acc to output

Input shapes: all [batch, heads, seq_len, head_dim] in fp16
Output: O [batch, heads, seq_len, head_dim]

Important:
  - acc must be float32
  - tl.dot requires both operands same dtype
  - Input is fp16, cast as needed
"""

import torch
import triton
import triton.language as tl


@triton.jit
def flash_attention_fwd_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vk, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    B, H, seq_len,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    sm_scale: tl.constexpr,
):
    """FlashAttention forward — fill in the algorithm above."""
    pid = tl.program_id(0)
    num_m_blocks = tl.cdiv(seq_len, BLOCK_M)

    # Decode pid into (batch, head, block_m)
    bh_id = pid // num_m_blocks
    block_m = pid % num_m_blocks
    b_idx = bh_id // H
    h_idx = bh_id % H

    # Offsets for this Q block
    offs_m = block_m * BLOCK_M + tl.arange(0, BLOCK_M)   # [BLOCK_M]
    offs_n = tl.arange(0, BLOCK_N)                        # [BLOCK_N]
    offs_d = tl.arange(0, BLOCK_D)                        # [head_dim]

    # ── YOUR CODE HERE ──
    # 1. Load Q block [BLOCK_M, head_dim]
    # 2. Initialize m=-inf, d=0, acc=zeros
    # 3. Loop over K,V blocks:
    #    a. Load K block, V block
    #    b. S = tl.dot(Q, K^T) * sm_scale
    #    c. Online softmax update
    #    d. Accumulate output
    # 4. Store result to o_ptr
    # ── END ──
    Q = tl.load(q_ptr + b_idx * stride_qb + h_idx * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd)
    m = tl.full((BLOCK_M,), -float('inf'), dtype=tl.float32)
    d = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    for block_n in range(tl.cdiv(seq_len, BLOCK_N)):
        # Load K and V with block_n offset
        kn_base = b_idx * stride_kb + h_idx * stride_kh + block_n * BLOCK_N * stride_kn
        vn_base = b_idx * stride_vb + h_idx * stride_vh + block_n * BLOCK_N * stride_vk

        K = tl.load(k_ptr + kn_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd,
                    mask=offs_n[:, None] < seq_len, other=0.0)
        V = tl.load(v_ptr + vn_base + offs_n[:, None] * stride_vk + offs_d[None, :] * stride_vd,
                    mask=offs_n[:, None] < seq_len, other=0.0)
        S = tl.dot(Q.to(tl.float32), tl.trans(K).to(tl.float32)) * sm_scale
        m_new = tl.maximum(m, tl.max(S, axis=1))
        P = tl.exp(S - m_new[:, None])
        d_new = d * tl.exp(m - m_new) + tl.sum(P, axis=1)
        acc = acc * (d / d_new)[:, None] * tl.exp(m - m_new)[:, None] + tl.dot(P.to(tl.float32), V.to(tl.float32)) / d_new[:, None]
        m = m_new
        d = d_new
    # Mask for storing output
    mask = offs_m < seq_len
    tl.store(o_ptr + b_idx * stride_ob + h_idx * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od, acc, mask=mask[:, None])



def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                    sm_scale: float = None):
    """Wrapper for FlashAttention forward."""
    B, H, seq_len, head_dim = q.shape
    assert q.shape == k.shape == v.shape
    if sm_scale is None:
        sm_scale = 1.0 / (head_dim ** 0.5)

    o = torch.empty_like(q)
    BLOCK_M, BLOCK_N = 64, 64
    grid = (triton.cdiv(seq_len, BLOCK_M) * B * H,)

    flash_attention_fwd_kernel[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        B, H, seq_len,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=head_dim,
        sm_scale=sm_scale,
    )
    return o


def pytorch_attention(q, k, v, sm_scale=None):
    """Reference implementation."""
    if sm_scale is None:
        sm_scale = 1.0 / (q.shape[-1] ** 0.5)
    attn = torch.matmul(q, k.transpose(-2, -1)) * sm_scale
    attn = torch.softmax(attn, dim=-1)
    return torch.matmul(attn, v)


# ─── Test ───────────────────────────────────────────────────
def test_flash_attention():
    print("Testing FlashAttention correctness...")
    torch.manual_seed(42)

    for seq_len, head_dim in [(128, 64), (256, 64), (512, 64)]:
        B, H = 2, 4
        q = torch.randn(B, H, seq_len, head_dim, device='cuda', dtype=torch.float16)
        k = torch.randn(B, H, seq_len, head_dim, device='cuda', dtype=torch.float16)
        v = torch.randn(B, H, seq_len, head_dim, device='cuda', dtype=torch.float16)

        ref = pytorch_attention(q.float(), k.float(), v.float())
        o = flash_attention(q, k, v).float()

        max_diff = (o - ref).abs().max().item()
        mean_diff = (o - ref).abs().mean().item()
        status = "✓" if max_diff < 0.05 else "✗"
        print(f"  seq={seq_len:>4}, d={head_dim:>3} | "
              f"max_diff={max_diff:.4f}, mean_diff={mean_diff:.4f} {status}")


def bench_attention():
    print("\nBenchmark (seq_len=1024, d=64, B=1, H=16):")
    B, H, seq_len, head_dim = 1, 16, 1024, 64
    q = torch.randn(B, H, seq_len, head_dim, device='cuda', dtype=torch.float16)
    k = torch.randn(B, H, seq_len, head_dim, device='cuda', dtype=torch.float16)
    v = torch.randn(B, H, seq_len, head_dim, device='cuda', dtype=torch.float16)

    for _ in range(10):
        flash_attention(q, k, v)
        pytorch_attention(q.float(), k.float(), v.float())
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    flash_attention(q, k, v)
    end.record()
    torch.cuda.synchronize()
    triton_ms = start.elapsed_time(end)

    start.record()
    pytorch_attention(q.float(), k.float(), v.float())
    end.record()
    torch.cuda.synchronize()
    torch_ms = start.elapsed_time(end)

    s_memory_gb = B * H * seq_len * seq_len * 2 / 1e9
    print(f"  PyTorch attention:     {torch_ms:.3f} ms  "
          f"(stores {s_memory_gb:.1f} GB S matrix)")
    print(f"  Triton FlashAttention: {triton_ms:.3f} ms  (O(1) extra memory)")


if __name__ == "__main__":
    test_flash_attention()
    bench_attention()
