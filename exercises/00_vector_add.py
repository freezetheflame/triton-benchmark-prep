"""
Exercise 0: Hello Triton — Vector Addition
============================================
Goal: Write your first Triton kernel and understand the execution model.

Core concepts:
  - tl.program_id(axis) — which block am I?
  - Grid — how are blocks arranged? (e.g., grid=(N_BLOCKS,))
  - tl.load / tl.store — read/write global memory
  - tl.arange — create index vectors
  - Masking — handle edge cases when N is not divisible by block_size

Execution model:
  Grid = (N_BLOCKS,)  means we launch N_BLOCKS independent "blocks"
  Each block gets a unique program_id(0) = 0, 1, 2, ..., N_BLOCKS-1
  Each block processes BLOCK_SIZE elements in parallel

  BLOCK_SIZE = 256, N = 1000 → N_BLOCKS = ceil(1000/256) = 4
  Block 0: elements 0..255
  Block 1: elements 256..511
  Block 2: elements 512..767
  Block 3: elements 768..999 (last 232 elements, rest masked)

Your task: complete add_kernel below.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(
    x_ptr, y_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Compute output[i] = x[i] + y[i] for all i.

    Hints:
    1. pid = tl.program_id(0)
    2. offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    3. mask = offsets < n_elements
    4. x = tl.load(x_ptr + offsets, mask=mask)
    5. y = tl.load(y_ptr + offsets, mask=mask)
    6. tl.store(output_ptr + offsets, x + y, mask=mask)
    """
    # ── YOUR CODE HERE ──
    pid = tl.program_id(0)
    offsets = pid*BLOCK_SIZE+ tl.arange(0,BLOCK_SIZE)

    mask = offsets < n_elements

    x = tl.load(x_ptr+offsets,mask=mask)
    y = tl.load(y_ptr+ offsets, mask=mask)

    tl.store(output_ptr+offsets,x+y,mask=mask)

    # ── END YOUR CODE ──


# ─── CPU reference ──────────────────────────────────────────
def add_cpu(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x + y


# ─── Test ───────────────────────────────────────────────────
def test_add():
    for N in [128, 256, 1000, 1023, 4096, 10000]:
        x = torch.randn(N, device='cuda', dtype=torch.float32)
        y = torch.randn(N, device='cuda', dtype=torch.float32)
        expected = add_cpu(x.cpu(), y.cpu())

        output = torch.empty_like(x)
        BLOCK_SIZE = 256
        grid = (triton.cdiv(N, BLOCK_SIZE),)

        add_kernel[grid](x, y, output, N, BLOCK_SIZE=BLOCK_SIZE)

        assert torch.allclose(output, expected.cuda(), atol=1e-5), \
            f"FAIL at N={N}"
        print(f"  N={N:>6}: ✓")


if __name__ == "__main__":
    print("Exercise 0: Vector Addition")
    test_add()
    print("\nAll tests passed!")

    # ─── Performance ───
    print("\nPerformance comparison (N=10M):")
    N = 10_000_000
    x = torch.randn(N, device='cuda', dtype=torch.float32)
    y = torch.randn(N, device='cuda', dtype=torch.float32)
    output = torch.empty_like(x)

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)

    for _ in range(10):
        add_kernel[grid](x, y, output, N, BLOCK_SIZE=BLOCK_SIZE)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(100):
        add_kernel[grid](x, y, output, N, BLOCK_SIZE=BLOCK_SIZE)
    end.record()
    torch.cuda.synchronize()
    triton_ms = start.elapsed_time(end) / 100

    start.record()
    for _ in range(100):
        _ = x + y
    end.record()
    torch.cuda.synchronize()
    torch_ms = start.elapsed_time(end) / 100

    print(f"  Triton:  {triton_ms:.4f} ms")
    print(f"  PyTorch: {torch_ms:.4f} ms")
    print(f"  Bandwidth Triton:  {3 * N * 4 / triton_ms / 1e6:.1f} GB/s")
