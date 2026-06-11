"""
Exercise 9: Grouped Query Attention (GQA)
==========================================

Standard multi-head attention:
  Each of H query heads has its OWN K, V heads.
  → H KV heads, H query heads. Memory: O(H).

Grouped Query Attention (GQA, LLaMA 2/3, Mistral):
  KV heads are SHARED among groups of query heads.
  → G KV heads (G << H), H query heads. Memory: O(G).

Example: LLaMA 2 70B — H=64 query heads, G=8 KV heads.
  Each KV head is shared by 64/8 = 8 query heads.
  KV cache size: 8× smaller!

This exercise: implement a GQA forward kernel that:
  1. Maps each query head to its KV group
  2. Computes attention with the shared K,V
  3. Handles the replication pattern efficiently

The kernel is FlashAttention with a GROUP INDEX instead of direct head index.

Run:  python exercises/09_gqa.py
"""

import torch
import triton
import triton.language as tl


# ═══════════════════════════════════════════════════════════
# GQA FlashAttention Kernel
# ═══════════════════════════════════════════════════════════
#
# Key difference from Ex04 FlashAttention:
#   Instead of using h_idx for both Q and KV:
#     K = load(K + h_idx * stride_kh + ...)
#
#   We compute the KV group:
#     kv_group = h_idx // num_queries_per_kv   (integer division)
#     K = load(K + kv_group * stride_kh + ...)  ← shared KV!
#
# The stride_kh for KV tensors is smaller because G < H.

@triton.jit
def gqa_flash_attention_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kg, stride_kn, stride_kd,   # note: stride_kg, not stride_kh!
    stride_vb, stride_vg, stride_vk, stride_vd,   # note: stride_vg!
    stride_ob, stride_oh, stride_om, stride_od,
    B, H, num_kv_groups, seq_len,
    sm_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    FlashAttention with GQA: Q heads share KV heads in groups.

    Key insight: kv_group = h_idx * num_kv_groups // H
    (or equivalently: h_idx // (H // num_kv_groups))
    """
    pid = tl.program_id(0)
    num_m_blocks = tl.cdiv(seq_len, BLOCK_M)
    num_h_blocks = H
    total_blocks = num_m_blocks * num_h_blocks

    # pid maps to (h_idx, block_m)
    h_idx = pid // num_m_blocks
    block_m = pid % num_m_blocks

    if h_idx >= H:
        return

    b_idx = 0  # assume batch=1 for simplicity; extend as exercise

    # ── YOUR CODE: compute kv_group index ──
    # Each KV head is shared by (H // num_kv_groups) query heads.
    # kv_group = h_idx * num_kv_groups // H
    #
    # Example: H=8, G=2 → head 0-3 → group 0, head 4-7 → group 1
    #   h_idx=0: 0*2//8 = 0
    #   h_idx=3: 3*2//8 = 0
    #   h_idx=4: 4*2//8 = 1

    offs_m = block_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    # Load Q tile (standard — Q uses its own head)
    q = tl.load(
        q_ptr + b_idx * stride_qb + h_idx * stride_qh
        + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
        mask=offs_m[:, None] < seq_len, other=0.0,
    )

    # ── YOUR CODE: online softmax accumulator (same as Ex04) ──
    # m = tl.full((BLOCK_M,), -float('inf'), dtype=tl.float32)
    # d = tl.zeros((BLOCK_M,), dtype=tl.float32)
    # acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    # ── YOUR CODE: K,V loop — use kv_group, not h_idx ──
    # for block_n in range(tl.cdiv(seq_len, BLOCK_N)):
    #     # Load K tile — indexed by kv_group, not h_idx!
    #     K = tl.load(
    #         k_ptr + b_idx * stride_kb + kv_group * stride_kg
    #         + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd,
    #         mask=offs_n[:, None] < seq_len, other=0.0,
    #     )
    #     # Load V tile — also kv_group
    #     V = tl.load(...)
    #
    #     # Standard online softmax (same as Ex04)
    #     S = tl.dot(q.to(tl.float32), tl.trans(K).to(tl.float32)) * sm_scale
    #     m_new = tl.maximum(m, tl.max(S, axis=1))
    #     P = tl.exp(S - m_new[:, None])
    #     d_new = d * tl.exp(m - m_new) + tl.sum(P, axis=1)
    #     acc = (acc * (d / d_new)[:, None] * tl.exp(m - m_new)[:, None]
    #            + tl.dot(P.to(tl.float32), V.to(tl.float32)) / d_new[:, None])
    #     m, d = m_new, d_new

    # ── YOUR CODE: store output ──
    # tl.store(
    #     o_ptr + b_idx * stride_ob + h_idx * stride_oh
    #     + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od,
    #     acc, mask=offs_m[:, None] < seq_len,
    # )

    pass


# ═══════════════════════════════════════════════════════════
# WRAPPER
# ═══════════════════════════════════════════════════════════

def gqa_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                  sm_scale: float = None) -> torch.Tensor:
    """
    GQA FlashAttention forward.

    q: (B, H, seq_len, head_dim)     — H query heads
    k: (B, G, seq_len, head_dim)     — G KV heads (G ≤ H)
    v: (B, G, seq_len, head_dim)
    """
    B, H, seq_len, head_dim = q.shape
    _, G, _, _ = k.shape
    assert H % G == 0, f"H ({H}) must be divisible by G ({G})"

    if sm_scale is None:
        sm_scale = 1.0 / (head_dim ** 0.5)

    o = torch.empty_like(q)

    # ── YOUR CODE ──
    # BLOCK_M, BLOCK_N = 64, 64
    # grid = (triton.cdiv(seq_len, BLOCK_M) * H,)   ← H programs per sequence block
    # gqa_flash_attention_kernel[grid](
    #     q, k, v, o,
    #     q.stride(0), q.stride(1), q.stride(2), q.stride(3),
    #     k.stride(0), k.stride(1), k.stride(2), k.stride(3),
    #     v.stride(0), v.stride(1), v.stride(2), v.stride(3),
    #     o.stride(0), o.stride(1), o.stride(2), o.stride(3),
    #     B, H, G, seq_len, sm_scale,
    #     BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=head_dim,
    # )

    return o


# ═══════════════════════════════════════════════════════════
# CORRECTNESS TEST
# ═══════════════════════════════════════════════════════════

def test_gqa():
    """Test GQA against a reference (naive) implementation."""
    B, H, G, seq_len, head_dim = 1, 8, 2, 256, 64
    assert H % G == 0
    heads_per_group = H // G

    q = torch.randn(B, H, seq_len, head_dim, device="cuda", dtype=torch.float16)
    k = torch.randn(B, G, seq_len, head_dim, device="cuda", dtype=torch.float16)
    v = torch.randn(B, G, seq_len, head_dim, device="cuda", dtype=torch.float16)

    sm_scale = 1.0 / (head_dim ** 0.5)

    # Reference: expand KV to H heads and do standard attention
    k_expanded = k.repeat_interleave(heads_per_group, dim=1)  # (B, H, seq, d)
    v_expanded = v.repeat_interleave(heads_per_group, dim=1)
    ref = torch.nn.functional.scaled_dot_product_attention(
        q, k_expanded, v_expanded, scale=sm_scale
    )

    # Our fused GQA
    try:
        ours = gqa_attention(q, k, v, sm_scale)
        max_err = (ours.float() - ref.float()).abs().max().item()
        ok = torch.allclose(ours.float(), ref.float(), rtol=1e-2, atol=1e-1)
        status = "✓" if ok else "✗"
        print(f"  GQA (H={H}, G={G}, seq={seq_len}): max_err={max_err:.6f} {status}")
    except NotImplementedError:
        print(f"  GQA: TODO — fill in the kernel")
    except Exception as e:
        print(f"  GQA: ERROR — {e}")

    # Performance
    try:
        # Reference: expand KV (costly!) then SDPA
        def ref_fn():
            ke = k.repeat_interleave(heads_per_group, dim=1)
            ve = v.repeat_interleave(heads_per_group, dim=1)
            return torch.nn.functional.scaled_dot_product_attention(q, ke, ve, scale=sm_scale)

        ref_ms = triton.testing.do_bench(ref_fn, rep=50)
        gqa_ms = triton.testing.do_bench(lambda: gqa_attention(q, k, v, sm_scale), rep=50)
        print(f"  Reference (expand+SDPA): {ref_ms*1000:.0f}us")
        print(f"  GQA fused:               {gqa_ms*1000:.0f}us  ({ref_ms/gqa_ms:.1f}x)")
    except:
        pass
    print()


# ═══════════════════════════════════════════════════════════
# EXERCISE QUESTIONS
# ═══════════════════════════════════════════════════════════
#
# 1. The reference implementation expands KV tensors from (B,G,seq,d)
#    to (B,H,seq,d) using repeat_interleave. How much extra memory does
#    this use for LLaMA 2 70B (H=64, G=8, seq=4096, d=128)?
#
# 2. In the GQA kernel, multiple query heads map to the same KV group.
#    During autotuning/benchmarking, the same K,V tiles are loaded from
#    global memory multiple times (once per query head in the group).
#    How could L2 cache help here? What's the access pattern?
#
# 3. What would happen if we used the standard FA kernel from Ex04
#    with G instead of H query heads, then broadcast the result?
#    Would that be correct? Why or why not?
#
# 4. In LLaMA 3 8B, H=32, G=8. If you're computing attention for
#    head 3 (which maps to KV group 0) and head 4 (also group 0),
#    how could you compute BOTH in the same kernel to avoid loading
#    K,V twice? Think about combining multiple Q heads per CTA.
#
# 5. (Hard) Implement multi-query attention (MQA, G=1) as a special
#    case. How much faster is it than GQA with G=8? What's the quality
#    trade-off observed in practice?


if __name__ == "__main__":
    print("=" * 70)
    print("  GROUPED QUERY ATTENTION (GQA)")
    print("=" * 70)
    print()
    test_gqa()
