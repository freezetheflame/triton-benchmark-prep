"""
Exercise 1: Element-wise Activation Functions
===============================================
Write ReLU, GELU, SiLU in Triton

Concepts: tl.maximum, tl.exp, tl.sigmoid, memory-bandwidth-bound ops

Note: Triton 3.x does NOT have tl.math.tanh. Write it manually:
  tanh(z) = (exp(2z) - 1) / (exp(2z) + 1)
"""

import torch
import triton
import triton.language as tl


@triton.jit
def relu_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """ReLU(x) = max(0, x)"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    # ── YOUR CODE ──
    y = tl.maximum(x,0.0)
    # ── END ──
    tl.store(output_ptr + offsets, y, mask=mask)

def tanh(z):
    result = (tl.exp(2*z) - 1) / (tl.exp(2*z) + 1)
    return result


@triton.jit
def gelu_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    # ── YOUR CODE ──
    # Hint: sqrt(2/π) ≈ 0.7978845608
    temp = 0.7978845608*(x+ 0.044715*x*x*x)
    temp2 = (tl.exp(2*temp) - 1) / (tl.exp(2*temp) + 1)
    y = 0.5 * x * (1 + temp2)
    tl.store(output_ptr+offsets,y,mask=mask)
    # ── END ──


@triton.jit
def silu_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """SiLU(x) = x * sigmoid(x)"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    # ── YOUR CODE ──
    # Hint: use tl.sigmoid
    y = x * tl.sigmoid(x)
    tl.store(output_ptr+offsets,y,mask=mask)
    # ── END ──


# ─── Test & Benchmark ───────────────────────────────────────
def benchmark_kernel(kernel_fn, x: torch.Tensor, label: str, BLOCK_SIZE=1024):
    output = torch.empty_like(x)
    N = x.numel()
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    for _ in range(10):
        kernel_fn[grid](x, output, N, BLOCK_SIZE=BLOCK_SIZE)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(100):
        kernel_fn[grid](x, output, N, BLOCK_SIZE=BLOCK_SIZE)
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / 100
    GB = N * x.element_size() / 1e9
    bw = 2 * GB / (ms / 1000)
    print(f"  {label:>8}: {ms:>8.4f} ms  |  {bw:>6.1f} GB/s")
    return ms


def test_and_benchmark():
    N = 10_000_000
    x = torch.randn(N, device='cuda', dtype=torch.float32)

    print("Exercise 1: Activation Functions")
    print(f"  Testing N={N:,} elements\n")

    refs = {
        'relu': torch.relu(x),
        'gelu': torch.nn.functional.gelu(x, approximate='tanh'),
        'silu': torch.nn.functional.silu(x),
    }

    kernels = [('ReLU', relu_kernel), ('GELU', gelu_kernel), ('SiLU', silu_kernel)]
    BS = 1024
    grid = (triton.cdiv(N, BS),)
    out = torch.empty_like(x)

    for name, kernel in kernels:
        kernel[grid](x, out, N, BLOCK_SIZE=BS)
        key = name.lower()
        assert torch.allclose(out, refs[key], atol=1e-4), f"{name} failed!"
        print(f"  {name}: correctness ✓")

    print("\n  Performance:")
    for name, kernel in kernels:
        benchmark_kernel(kernel, x, f"{name} Triton")

    for name, fn in [('ReLU', torch.relu),
                      ('GELU', lambda t: torch.nn.functional.gelu(t, approximate='tanh')),
                      ('SiLU', torch.nn.functional.silu)]:
        for _ in range(10): fn(x)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(100): fn(x)
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end) / 100
        GB = N * x.element_size() / 1e9
        bw = 2 * GB / (ms / 1000)
        print(f"  {name:>8} PyTorch: {ms:>8.4f} ms  |  {bw:>6.1f} GB/s")


if __name__ == "__main__":
    test_and_benchmark()
