"""
Exercise 2: Softmax
====================
Implement row-wise softmax in Triton.

Key challenge: softmax needs the max value of the ENTIRE row before normalization.
How do you compute it when processing row in chunks?

Two approaches:
  A) 2-pass: first pass finds max, second pass normalizes
  B) Online (1-pass): maintain running m and d, rescale when new max is found

Online softmax formula (Milakov & Gimelshein 2018):
  m = -inf, d = 0
  For each block:
    m_new = max(m, max(block))
    d = d * exp(m - m_new) + sum(exp(block - m_new))
    m = m_new
  output = exp(x_i - m) / d

See notes/online-softmax.md for the full derivation.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def softmax_2pass_kernel(x_ptr, output_ptr, n_rows, n_cols, BLOCK_SIZE: tl.constexpr):
    """2-pass stable softmax: first pass max, second pass normalize."""
    row_idx = tl.program_id(0)
    row_start = row_idx * n_cols
    max_num = -float('inf')
    # ── YOUR CODE: Pass 1 — find max ──
    # Loop over chunks of this row, accumulate the max
    # ──

    for col_idx in range(n_cols):
        max_num = tl.maximum(max_num, tl.load(x_ptr + row_start + col_idx))
    # ── YOUR CODE: Pass 2 — compute exp(x-max) / sum(exp(x-max)) ──
    # Loop again, first accumulate sum, then store normalized values
    # ──
    sum_exp = 0.0
    for col_idx in range(n_cols):
        x = tl.load(x_ptr + row_start + col_idx)
        exp_x = tl.exp(x - max_num)
        sum_exp += exp_x

    for col_idx in range(n_cols):
        x = tl.load(x_ptr + row_start + col_idx)
        exp_x = tl.exp(x - max_num)
        tl.store(output_ptr + row_start + col_idx, exp_x / sum_exp)


@triton.jit
def softmax_online_kernel(x_ptr, output_ptr, n_rows, n_cols, BLOCK_SIZE: tl.constexpr):
    """Online softmax: single accumulation pass."""
    row_idx = tl.program_id(0)
    row_start = row_idx * n_cols

    # ── YOUR CODE ──
    # m = -inf, d = 0
    # Loop over chunks:
    #   m_new = max(m, max(block))
    #   d = d * exp(m - m_new) + sum(exp(block - m_new))
    #   m = m_new
    # Then second loop: output = exp(x_i - m) / d
    # ──
    m = -float('inf')
    d = 0.0
    for col_idx in range(n_cols):
        x = tl.load(x_ptr + row_start + col_idx)
        m_new = tl.maximum(m, x)
        d_new = d * tl.exp(m - m_new) + tl.exp(x - m_new)
        m = m_new
        d = d_new
    
    for col_idx in range(n_cols):
        x = tl.load(x_ptr + row_start + col_idx)
        exp_x = tl.exp(x - m)
        tl.store(output_ptr + row_start + col_idx, exp_x / d)

# ─── Test ───────────────────────────────────────────────────
def test_softmax():
    torch.manual_seed(42)
    print("Testing correctness...")
    for shape in [(16, 256), (32, 512), (64, 128), (1, 1000)]:
        x = torch.randn(shape, device='cuda', dtype=torch.float32) * 2.0
        ref = torch.softmax(x, dim=-1)

        out_2pass = torch.empty_like(x)
        out_online = torch.empty_like(x)

        n_rows, n_cols = shape
        BLOCK_SIZE = min(triton.next_power_of_2(n_cols), 1024)
        grid = (n_rows,)

        softmax_2pass_kernel[grid](x, out_2pass, n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE)
        softmax_online_kernel[grid](x, out_online, n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE)

        assert torch.allclose(out_2pass, ref, atol=1e-4), f"2-pass failed at {shape}"
        assert torch.allclose(out_online, ref, atol=1e-4), f"Online failed at {shape}"
        print(f"  shape={shape}: ✓")

    # Numerical stability test
    print("\nNumerical stability (large values)...")
    x_large = torch.tensor([[1000.0, 1001.0, 0.0]], device='cuda')
    ref = torch.softmax(x_large, dim=-1)
    out = torch.empty_like(x_large)
    softmax_online_kernel[(1,)](x_large, out, 1, 3, BLOCK_SIZE=4)
    assert torch.allclose(out, ref, atol=1e-4), "Large values failed"
    print(f"  [1000, 1001, 0] → {out.tolist()}")


def bench_softmax():
    print("\nBenchmark (shape=4096x4096):")
    N, M = 4096, 4096
    x = torch.randn(N, M, device='cuda', dtype=torch.float32)
    BLOCK_SIZE = 1024
    grid = (N,)
    out = torch.empty_like(x)

    for _ in range(5):
        softmax_online_kernel[grid](x, out, N, M, BLOCK_SIZE=BLOCK_SIZE)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    softmax_online_kernel[grid](x, out, N, M, BLOCK_SIZE=BLOCK_SIZE)
    end.record()
    torch.cuda.synchronize()
    triton_ms = start.elapsed_time(end)

    for _ in range(5):
        torch.softmax(x, dim=-1)
    torch.cuda.synchronize()
    start.record()
    torch.softmax(x, dim=-1)
    end.record()
    torch.cuda.synchronize()
    torch_ms = start.elapsed_time(end)

    print(f"  Triton online: {triton_ms:.3f} ms")
    print(f"  PyTorch:       {torch_ms:.3f} ms")


if __name__ == "__main__":
    test_softmax()
    bench_softmax()
