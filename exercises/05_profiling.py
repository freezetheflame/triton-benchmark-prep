"""
Exercise 5: GPU Kernel Profiling & Roofline Analysis
======================================================
Profile all 5 Triton kernels, compute arithmetic intensity,
compare performance with PyTorch baselines, and identify bottlenecks.

Learning goals:
  1. GPU Event timing — accurate kernel measurement
  2. Arithmetic Intensity — compute vs memory bandwidth bound
  3. Roofline model — theoretical peak vs achieved performance  
  4. Register / shared memory analysis via autotuner output
  5. Generating a profiling report for your resume

Run:  python exercises/05_profiling.py
      TRITON_PRINT_AUTOTUNING=1 python exercises/05_profiling.py  (for detail)
"""

import torch
import triton
import triton.language as tl
import sys
import os
from dataclasses import dataclass, field
from typing import List, Tuple

# ─── Hardware Specs (RTX 4070 SUPER AD104) ───
PEAK_FP32_TFLOPS = 35.5    # Theoretical FP32 peak
MEM_BANDWIDTH_GBS = 504    # GDDR6X 192-bit @ 21 Gbps
ROOFLINE_CROSSOVER = 70    # FLOP/byte (35500/504 ≈ 70)
SM_COUNT = 56
MAX_WARPS_PER_SM = 48
REGS_PER_SM = 65536
SHMEM_PER_SM_KB = 100      # 100 KB per SM (128 KB total, 28 reserved)


# ═══════════════════════════════════════════════════════════
#  PROFILING UTILITIES
# ═══════════════════════════════════════════════════════════

@dataclass
class KernelProfile:
    """Profiling result for one kernel."""
    name: str
    shape: tuple
    triton_ms: float = 0.0
    torch_ms: float = 0.0
    flops: int = 0                # Total FLOPs
    bytes_read: int = 0            # Bytes read from global memory
    bytes_written: int = 0         # Bytes written to global memory
    ai: float = 0.0               # Arithmetic Intensity (FLOP/byte)
    achieved_tflops: float = 0.0
    achieved_bw_gbs: float = 0.0
    bound: str = ""               # "compute" or "memory"
    notes: str = ""


class GPUTimer:
    """GPU event-based timer with warmup."""

    def __init__(self, warmup_iters: int = 10):
        self.warmup_iters = warmup_iters
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)

    def measure(self, fn) -> float:
        """Run fn() and return elapsed time in milliseconds."""
        # Warmup
        for _ in range(self.warmup_iters):
            fn()
        torch.cuda.synchronize()

        # Timed run
        self.start.record()
        fn()
        self.end.record()
        torch.cuda.synchronize()

        return self.start.elapsed_time(self.end)


def compute_ai(flops: int, bytes_total: int) -> float:
    """Arithmetic Intensity = FLOPs / Bytes moved."""
    if bytes_total == 0:
        return float('inf')
    return flops / bytes_total


def classify_bound(ai: float) -> str:
    """Classify kernel as compute-bound or memory-bound."""
    return "compute" if ai > ROOFLINE_CROSSOVER else "memory"


# ═══════════════════════════════════════════════════════════
#  KERNEL PROFILING FUNCTIONS
# ═══════════════════════════════════════════════════════════

# NOTE: We redefine simplified versions here for clean profiling.
# In production, you'd import from the exercise modules.

@triton.jit
def relu_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, tl.maximum(x, 0.0), mask=mask)


@triton.jit
def gelu_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    """GELU approximate: x * sigmoid(1.702 * x)"""
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    y = x * tl.sigmoid(1.702 * x)
    tl.store(out_ptr + offs, y, mask=mask)


@triton.jit
def softmax_online_kernel(x_ptr, out_ptr, n_rows, n_cols, BLOCK_SIZE: tl.constexpr):
    """Online softmax — single pass over each row."""
    row = tl.program_id(0)
    start = row * n_cols
    m = -float('inf')
    d = 0.0
    for i in range(n_cols):
        x = tl.load(x_ptr + start + i)
        m_new = tl.maximum(m, x)
        d = d * tl.exp(m - m_new) + tl.exp(x - m_new)
        m = m_new
    for i in range(n_cols):
        x = tl.load(x_ptr + start + i)
        tl.store(out_ptr + start + i, tl.exp(x - m) / d)


@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr,
                  M, N, K,
                  stride_am, stride_ak,
                  stride_bk, stride_bn,
                  stride_cm, stride_cn,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """Tiled matrix multiply C[M,N] = A[M,K] @ B[K,N]."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak
    b_ptrs = b_ptr + rk[:, None] * stride_bk + rn[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=rm[:, None] < M and rk[None, :] < K - k)
        b = tl.load(b_ptrs, mask=rk[:, None] < K - k and rn[None, :] < N)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=rm[:, None] < M and rn[None, :] < N)


# ═══════════════════════════════════════════════════════════
#  PROFILE EACH KERNEL
# ═══════════════════════════════════════════════════════════

def profile_elementwise(timer: GPUTimer) -> List[KernelProfile]:
    """Profile ReLU and GELU (memory-bound elementwise ops)."""
    results = []
    N = 2**24  # 16M elements → 64 MB

    for name, kernel in [("ReLU", relu_kernel), ("GELU", gelu_kernel)]:
        x = torch.randn(N, device='cuda', dtype=torch.float32)
        out = torch.empty_like(x)
        BLOCK_SIZE = 1024
        grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)

        triton_ms = timer.measure(lambda: kernel[grid](x, out, N, BLOCK_SIZE=BLOCK_SIZE))

        # PyTorch baseline
        torch_fn = torch.relu if name == "ReLU" else \
                   lambda t: t * torch.sigmoid(1.702 * t)
        torch_ms = timer.measure(lambda: torch_fn(x))

        # FLOPs & Bytes for elementwise: 1 op per element
        flops = N
        bytes_total = N * 4 * 2 + N * 4  # read x + write out (2 inputs for GELU is similar)
        ai = compute_ai(flops, bytes_total)

        results.append(KernelProfile(
            name=name,
            shape=(N,),
            triton_ms=triton_ms,
            torch_ms=torch_ms,
            flops=flops,
            bytes_read=N * 4,
            bytes_written=N * 4,
            ai=ai,
            achieved_tflops=flops / (triton_ms * 1e9),       # FLOPs / (ms→s) / 1e12
            achieved_bw_gbs=bytes_total / (triton_ms * 1e6),  # bytes / (ms→s) / 1e9
            bound=classify_bound(ai),
            notes=f"Mem-bound: {bytes_total / (triton_ms * 1e6):.0f}/{MEM_BANDWIDTH_GBS} GB/s"
        ))
    return results


def profile_softmax(timer: GPUTimer) -> List[KernelProfile]:
    """Profile online softmax (medium arithmetic intensity)."""
    results = []
    shapes = [(4096, 4096), (1024, 16384), (16384, 1024)]

    for rows, cols in shapes:
        x = torch.randn(rows, cols, device='cuda', dtype=torch.float32)
        out = torch.empty_like(x)
        BLOCK_SIZE = min(triton.next_power_of_2(cols), 1024)
        grid = (rows,)

        triton_ms = timer.measure(
            lambda: softmax_online_kernel[grid](x, out, rows, cols, BLOCK_SIZE=BLOCK_SIZE))

        torch_ms = timer.measure(lambda: torch.softmax(x, dim=-1))

        # Softmax: per element: exp + subtract + divide ≈ 3 ops
        # + max reduction (log(N) comparisons) ≈ small
        flops = rows * cols * 4  # ~4 ops per element
        bytes_total = rows * cols * 4 * 3  # read + write + intermediates
        ai = compute_ai(flops, bytes_total)

        results.append(KernelProfile(
            name="Softmax",
            shape=(rows, cols),
            triton_ms=triton_ms,
            torch_ms=torch_ms,
            flops=flops,
            bytes_read=rows * cols * 4,
            bytes_written=rows * cols * 4,
            ai=ai,
            achieved_tflops=flops / (triton_ms * 1e9),
            achieved_bw_gbs=bytes_total / (triton_ms * 1e6),
            bound=classify_bound(ai),
            notes=f"Naive scalar loop — shows why tiling matters"
        ))

    return results


def profile_matmul(timer: GPUTimer) -> List[KernelProfile]:
    """Profile tiled matmul (compute-bound)."""
    results = []
    shapes = [(1024, 1024, 1024), (2048, 2048, 2048), (4096, 4096, 4096)]

    for M, N, K in shapes:
        a = torch.randn(M, K, device='cuda', dtype=torch.float16)
        b = torch.randn(K, N, device='cuda', dtype=torch.float16)
        c = torch.empty(M, N, device='cuda', dtype=torch.float16)

        BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 32
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

        triton_ms = timer.measure(
            lambda: matmul_kernel[grid](
                a, b, c, M, N, K,
                a.stride(0), a.stride(1),
                b.stride(0), b.stride(1),
                c.stride(0), c.stride(1),
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K))

        torch_ms = timer.measure(lambda: torch.matmul(a.float(), b.float()))

        # FLOPs = 2 * M * N * K (multiply + add)
        flops = 2 * M * N * K
        bytes_total = (M*K + K*N + M*N) * 2  # fp16 = 2 bytes
        ai = compute_ai(flops, bytes_total)

        results.append(KernelProfile(
            name="Matmul",
            shape=(M, N, K),
            triton_ms=triton_ms,
            torch_ms=torch_ms,
            flops=flops,
            bytes_read=(M*K + K*N) * 2,
            bytes_written=M*N * 2,
            ai=ai,
            achieved_tflops=flops / (triton_ms * 1e9),
            achieved_bw_gbs=bytes_total / (triton_ms * 1e6),
            bound=classify_bound(ai),
            notes=f"fp16 Tensor Core: {flops / (triton_ms * 1e9):.1f}/{PEAK_FP32_TFLOPS*5:.0f} TFLOPs"
        ))
    return results


# ═══════════════════════════════════════════════════════════
#  TRITON FLASH ATTENTION KERNEL (from exercise 04)
# ═══════════════════════════════════════════════════════════

@triton.jit
def fa_kernel(q_ptr, k_ptr, v_ptr, o_ptr,
              stride_qb, stride_qh, stride_qm, stride_qd,
              stride_kb, stride_kh, stride_kn, stride_kd,
              stride_vb, stride_vh, stride_vk, stride_vd,
              stride_ob, stride_oh, stride_om, stride_od,
              B, H, seq_len,
              BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
              BLOCK_D: tl.constexpr, sm_scale: tl.constexpr):
    """FlashAttention forward — online softmax over K,V tiles."""
    pid = tl.program_id(0)
    num_m_blocks = tl.cdiv(seq_len, BLOCK_M)
    bh_id = pid // num_m_blocks
    block_m = pid % num_m_blocks
    b_idx = bh_id // H
    h_idx = bh_id % H

    offs_m = block_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    Q = tl.load(q_ptr + b_idx * stride_qb + h_idx * stride_qh
                + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd)

    m = tl.full((BLOCK_M,), -float('inf'), dtype=tl.float32)
    d = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    for block_n in range(tl.cdiv(seq_len, BLOCK_N)):
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
        acc = (acc * (d / d_new)[:, None] * tl.exp(m - m_new)[:, None]
               + tl.dot(P.to(tl.float32), V.to(tl.float32)) / d_new[:, None])
        m, d = m_new, d_new

    mask = offs_m < seq_len
    tl.store(o_ptr + b_idx * stride_ob + h_idx * stride_oh
             + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od,
             acc, mask=mask[:, None])


def triton_flash_attention(q, k, v, sm_scale=None):
    """Wrapper for the Triton FlashAttention kernel."""
    B, H, seq_len, head_dim = q.shape
    if sm_scale is None:
        sm_scale = 1.0 / (head_dim ** 0.5)

    o = torch.empty_like(q)
    BLOCK_M, BLOCK_N = 64, 64
    grid = (triton.cdiv(seq_len, BLOCK_M) * B * H,)

    fa_kernel[grid](
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


def profile_attention_all(timer: GPUTimer) -> List[KernelProfile]:
    """Profile three attention implementations:
      1. Naive PyTorch (O(S²) memory, stores full S matrix)
      2. Triton FlashAttention (O(1) memory, online softmax)
      3. PyTorch SDPA (cuDNN FlashAttention, fastest)
    
    Same FLOPs, vastly different memory footprints.
    """
    results = []

    # Medium size — naive attention still fits in VRAM
    B, H, seq_len, head_dim = 2, 8, 2048, 64
    q = torch.randn(B, H, seq_len, head_dim, device='cuda', dtype=torch.float16)
    k = torch.randn(B, H, seq_len, head_dim, device='cuda', dtype=torch.float16)
    v = torch.randn(B, H, seq_len, head_dim, device='cuda', dtype=torch.float16)

    sm_scale = 1.0 / (head_dim ** 0.5)

    # ── 1. Naive attention ──
    def naive_attn():
        qf = q.float()
        kf = k.float()
        vf = v.float()
        S = torch.matmul(qf, kf.transpose(-2, -1)) * sm_scale
        P = torch.softmax(S, dim=-1)
        return torch.matmul(P, vf)

    naive_ms = timer.measure(naive_attn)

    # ── 2. Triton FlashAttention ──
    def triton_fa():
        return triton_flash_attention(q, k, v, sm_scale)

    triton_ms = timer.measure(triton_fa)

    # ── 3. PyTorch SDPA ──
    def torch_sdpa():
        with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.FLASH_ATTENTION]):
            return torch.nn.functional.scaled_dot_product_attention(q, k, v)

    sdpa_ms = timer.measure(torch_sdpa)

    # FLOPs: Q@K^T (2*B*H*S*S*d) + softmax (~5*B*H*S*S) + P@V (2*B*H*S*S*d)
    s2 = seq_len * seq_len
    flops = B * H * (4 * s2 * head_dim + 5 * s2)

    # Bytes: KEY difference between implementations
    # Naive: reads Q,K,V + writes S (S*S fp32) + reads S + writes O
    naive_bytes = B * H * (
        seq_len * head_dim * 2 * 3   # Q,K,V reads (fp16)
        + s2 * 4                      # S write (fp32)
        + s2 * 4                      # S read for softmax
        + s2 * 4                      # P read for P@V
        + seq_len * head_dim * 2      # O write (fp16)
    )

    # FA: reads Q,K,V + writes O — no S matrix in global memory
    fa_bytes = B * H * (
        seq_len * head_dim * 2 * 3   # Q,K,V reads (fp16)
        + seq_len * head_dim * 2     # O write (fp16)
    )

    ai_naive = compute_ai(flops, naive_bytes)
    ai_fa = compute_ai(flops, fa_bytes)

    # S matrix memory
    s_mem_mb = B * H * s2 * 4 / 1e6

    results.append(KernelProfile(
        name="Attention-Naive",
        shape=(B, H, seq_len, head_dim),
        triton_ms=0,
        torch_ms=naive_ms,
        flops=flops,
        bytes_read=B * H * seq_len * head_dim * 2 * 3,
        bytes_written=B * H * (s2 * 4 * 2 + seq_len * head_dim * 2),
        ai=ai_naive,
        achieved_tflops=flops / (naive_ms * 1e9),
        achieved_bw_gbs=naive_bytes / (naive_ms * 1e6),
        bound=classify_bound(ai_naive),
        notes=f"S matrix: {s_mem_mb:.0f} MB (O(S²))"
    ))

    results.append(KernelProfile(
        name="Attention-TritonFA",
        shape=(B, H, seq_len, head_dim),
        triton_ms=triton_ms,
        torch_ms=sdpa_ms,
        flops=flops,
        bytes_read=B * H * seq_len * head_dim * 2 * 3,
        bytes_written=B * H * seq_len * head_dim * 2,
        ai=ai_fa,
        achieved_tflops=flops / (triton_ms * 1e9),
        achieved_bw_gbs=fa_bytes / (triton_ms * 1e6),
        bound=classify_bound(ai_fa),
        notes=f"NO S matrix — SRAM only (vs {s_mem_mb:.0f} MB naive)"
    ))

    results.append(KernelProfile(
        name="Attention-SDPA",
        shape=(B, H, seq_len, head_dim),
        triton_ms=0,
        torch_ms=sdpa_ms,
        flops=flops,
        bytes_read=B * H * seq_len * head_dim * 2 * 3,
        bytes_written=B * H * seq_len * head_dim * 2,
        ai=ai_fa,
        achieved_tflops=flops / (sdpa_ms * 1e9),
        achieved_bw_gbs=fa_bytes / (sdpa_ms * 1e6),
        bound=classify_bound(ai_fa),
        notes="cuDNN FlashAttention (vendor-optimized)"
    ))

    return results


# ═══════════════════════════════════════════════════════════
#  REPORT GENERATION
# ═══════════════════════════════════════════════════════════

def print_header():
    print("=" * 95)
    print("  GPU KERNEL PROFILING REPORT — RTX 4070 SUPER (AD104)")
    print("  Peak FP32: 35.5 TFLOPS | Memory BW: 504 GB/s | Roofline crossover: ~70 FLOP/byte")
    print("=" * 95)


def print_profile(p: KernelProfile):
    shape_str = "×".join(str(s) for s in p.shape)
    print(f"\n{'─'*80}")
    print(f"  {p.name}  ({shape_str})")
    print(f"  {'─'*80}")
    print(f"  Triton:  {p.triton_ms:>8.3f} ms" if p.triton_ms > 0 else f"  Triton:  {'N/A':>8}")
    print(f"  PyTorch: {p.torch_ms:>8.3f} ms")
    ratio = p.torch_ms / p.triton_ms if p.triton_ms > 0 else 0
    if ratio > 0:
        print(f"  Speedup: {ratio:>8.2f}x (vs PyTorch)")
    print(f"  ────────────────────────────────────")
    print(f"  FLOPs:         {p.flops/1e9:>8.2f} GFLOP")
    print(f"  Bytes moved:   {(p.bytes_read+p.bytes_written)/1e6:>8.2f} MB")
    print(f"  AI:            {p.ai:>8.1f} FLOP/byte")
    print(f"  Bound:         {p.bound:>8}")
    if p.achieved_tflops > 0:
        pct = p.achieved_tflops / PEAK_FP32_TFLOPS * 100
        print(f"  Achieved:      {p.achieved_tflops:>8.3f} TFLOPS ({pct:.1f}% peak)")
    if p.achieved_bw_gbs > 0:
        pct = p.achieved_bw_gbs / MEM_BANDWIDTH_GBS * 100
        print(f"  Bandwidth:     {p.achieved_bw_gbs:>8.1f} GB/s ({pct:.1f}% peak)")
    if p.notes:
        print(f"  Note:          {p.notes}")


def print_summary(all_results: List[KernelProfile]):
    print(f"\n\n{'='*95}")
    print("  SUMMARY TABLE")
    print(f"{'='*95}")
    print(f"  {'Kernel':<20} {'Shape':<20} {'Triton(ms)':>10} {'Torch(ms)':>10} {'AI':>8} {'Bound':>8}")
    print(f"  {'─'*20} {'─'*20} {'─'*10} {'─'*10} {'─'*8} {'─'*8}")
    for p in all_results:
        shape_str = "×".join(str(s) for s in p.shape)
        triton_str = f"{p.triton_ms:.3f}" if p.triton_ms > 0 else "N/A"
        print(f"  {p.name:<20} {shape_str:<20} {triton_str:>10} {p.torch_ms:>10.3f} {p.ai:>8.1f} {p.bound:>8}")

    # Roofline analysis
    print(f"\n  Roofline Analysis:")
    print(f"  {'─'*60}")
    mem_kernels = [p for p in all_results if p.bound == "memory"]
    comp_kernels = [p for p in all_results if p.bound == "compute"]
    if mem_kernels:
        print(f"  Memory-bound ({len(mem_kernels)}): {', '.join(p.name for p in mem_kernels)}")
        print(f"    → Optimize: increase BLOCK_SIZE, merge loads, use vectorized access")
    if comp_kernels:
        print(f"  Compute-bound ({len(comp_kernels)}): {', '.join(p.name for p in comp_kernels)}")
        print(f"    → Optimize: use Tensor Core, tune num_warps, reduce register pressure")


def print_kernel_compilation_stats():
    """Extract compilation metadata from compiled kernels: shared memory, num_warps, num_stages.
    
    Note: Per-thread register count is NOT available from PTX-level virtual registers
    (they get heavily reduced by ptxas). To get actual register usage, either:
      A) Use @triton.autotune with TRITON_PRINT_AUTOTUNING=1
      B) Use Nsight Compute (ncu --set full)
    """
    print(f"\n\n{'='*95}")
    print("  KERNEL COMPILATION METADATA (from compiled kernel)")
    print(f"  Note: Register count requires autotune or Nsight — see guide below")
    print(f"{'='*95}")
    
    # Trigger compilation of all kernels by doing a tiny run
    N = 1024
    x = torch.randn(N, device='cuda')
    out = torch.empty_like(x)
    relu_kernel[(1,)](x, out, N, BLOCK_SIZE=1024)
    
    x = torch.randn(N, device='cuda')
    out = torch.empty_like(x)
    gelu_kernel[(1,)](x, out, N, BLOCK_SIZE=1024)
    
    x = torch.randn(4, 256, device='cuda')
    out = torch.empty_like(x)
    softmax_online_kernel[(4,)](x, out, 4, 256, BLOCK_SIZE=256)
    
    a = torch.randn(128, 128, device='cuda', dtype=torch.float16)
    b = torch.randn(128, 128, device='cuda', dtype=torch.float16)
    c = torch.empty(128, 128, device='cuda', dtype=torch.float16)
    matmul_kernel[(1,1)](a, b, c, 128, 128, 128, 128, 1, 128, 1, 128, 1,
                         BLOCK_M=128, BLOCK_N=128, BLOCK_K=32)
    
    kernels_info = [
        ("ReLU", relu_kernel),
        ("GELU", gelu_kernel),
        ("Softmax", softmax_online_kernel),
        ("Matmul", matmul_kernel),
    ]
    
    print(f"\n  {'Kernel':<12} {'shmem(KB)':>10} {'num_warps':>10} {'num_stages':>11} {'threads/block':>14}")
    print(f"  {'─'*12} {'─'*10} {'─'*10} {'─'*11} {'─'*14}")
    
    for name, kernel in kernels_info:
        try:
            for dev, binder in kernel.device_caches.items():
                kernel_cache, _, _, _, _ = binder
                if kernel_cache:
                    compiled = list(kernel_cache.values())[0]
                    md = compiled.metadata
                    shmem_kb = md.shared / 1024 if md.shared else 0
                    threads = md.num_warps * 32
                    print(f"  {name:<12} {shmem_kb:>10.1f} {md.num_warps:>10} {md.num_stages:>11} {threads:>14}")
                break
        except Exception as e:
            print(f"  {name:<12} {'FAILED':>8} — {str(e)[:50]}")
    
    print(f"\n  How to get actual register usage:")
    print(f"    1. Add @triton.autotune to your kernel, then:")
    print(f"       TRITON_PRINT_AUTOTUNING=1 python my_kernel.py")
    print(f"    2. Or use Nsight Compute:")
    print(f"       ncu --set full python my_kernel.py  # shows per-thread reg count")


# ═══════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    from datetime import datetime

    # Setup output file
    os.makedirs("profiles", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"profiles/report_{timestamp}.txt"
    log_file = open(report_path, 'w', encoding='utf-8')

    # Tee: write to both terminal and file
    class Tee:
        def __init__(self, *files):
            self.files = files
        def write(self, text):
            for f in self.files:
                f.write(text)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()

    original_stdout = sys.stdout
    sys.stdout = Tee(original_stdout, log_file)

    try:
        print_header()
        timer = GPUTimer(warmup_iters=10)
        all_results = []

        print("\n[1/4] Profiling Element-wise Ops (ReLU, GELU)...")
        all_results.extend(profile_elementwise(timer))

        print("[2/4] Profiling Softmax...")
        all_results.extend(profile_softmax(timer))

        print("[3/4] Profiling Matmul...")
        all_results.extend(profile_matmul(timer))

        print("[4/4] Profiling Attention (Naive vs Triton FA vs SDPA)...")
        all_results.extend(profile_attention_all(timer))

        for p in all_results:
            print_profile(p)

        print_summary(all_results)
        print_kernel_compilation_stats()

        print(f"\n{'='*95}")
        print("  EXERCISE: Answer these questions")
        print(f"{'='*95}")
        print("""
  1. Which kernels are memory-bound? Why?
  2. Which kernels are compute-bound? Why?
  3. Why does Matmul achieve higher % of peak FLOPS than Softmax?
  4. Why is FlashAttention faster than naive attention even though
     it does the same FLOPs? (Hint: look at bytes moved)
  5. Look at the COMPILATION METADATA section. Why does Matmul need
     shared memory (32 KB) but ReLU/Softmax don't (0 KB)?
  6. Which kernel has the most room for improvement? Propose a change.
  """)

    finally:
        sys.stdout = original_stdout
        log_file.close()

    print(f"\nReport saved to: {report_path}")
    print(f"  cat {report_path}   # view")
    print(f"  ls profiles/        # list all reports")
