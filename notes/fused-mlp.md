# Fused MLP Kernel — 消除中间张量

## 为什么要 fusion？

标准的 Transformer MLP 是三步：

```
x = Linear_1(input)       # [N, D] → [N, 4D]    ← 写回 global memory
x = GELU(x)               # [N, 4D]              ← 读 + 写
x = Linear_2(x)           # [N, 4D] → [N, D]     ← 读 + 写
```

每次 `读 + 写` 都走 global memory（GDDR6X ~504 GB/s vs L1 ~5 TB/s）。
**Fusion = 在 SRAM/寄存器里完成全部计算，只写一次最终结果。**

```
Fused: load → matmul_1 → GELU → matmul_2 → store
         ↑_____只走一次 global memory 来回_____↑
```

## 数据流

```
Input:  x ∈ R^{N, K}        (K = d_model)
W1:     w1 ∈ R^{K, 4K}
W2:     w2 ∈ R^{4K, K}

for each tile of rows:
    acc1 = x_tile @ W1          # [BLOCK_M, 4K]
    acc1 = GELU(acc1)           # element-wise in registers
    acc2 = acc1 @ W2            # [BLOCK_M, K]
    store acc2                   # 只写一次！
```

## 关键设计决策

### 决策 1：W1 和 W2 放在哪里？

```
方案 A（推荐）：W1、W2 留在 global memory，分 tile 加载
  → 代码简单，不需要 shared memory 管理
  → W1 很大 [K, 4K] ≈ 4096×16384×4B = 256 MB，放不进取
  
方案 B：W1、W2 一次加载到 shared memory
  → 需要分 tile 加载 + double buffering
  → 只在 K 很小时可行
```

用方案 A。这和 matmul 是一样的——权重每次从 global memory 加载 tile。

### 决策 2：GELU 在哪执行？

在寄存器中，紧跟在 `acc1` 计算之后：

```python
acc1 = tl.zeros([BLOCK_M, BLOCK_K4], dtype=tl.float32)
for k in range(0, K, BLOCK_K):
    a = load A_tile
    b = load W1_tile
    acc1 += tl.dot(a, b)

# GELU 在寄存器中——不写回 memory
acc1 = gelu_approx(acc1)  # acc1 * sigmoid(1.702 * acc1)

acc2 = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
for k in range(0, K4, BLOCK_K2):
    a = acc1_tile          # 从寄存器复用！
    b = load W2_tile
    acc2 += tl.dot(a, b)
```

### 决策 3：一行能放进寄存器吗？

```
BLOCK_M = 64 行
K4 = 16384 维（4×d_model）

acc1: 64 × 16384 × 4B = 4 MB  ← 远超寄存器（~256KB per SM）
```

**结论：一行装不下**。需要把 K4 也分 tile，每个 tile 做了 GELU 后立即消耗掉：

```python
for k4_start in range(0, K4, BLOCK_K4):
    # Step 1: Compute acc1 tile [BLOCK_M, BLOCK_K4]
    acc1 = zeros(...)
    for k in range(0, K, BLOCK_K):
        acc1 += dot(A_tile, W1_tile)
    
    # Step 2: GELU in-place
    acc1 = gelu(acc1)
    
    # Step 3: Multiply with W2 chunk [BLOCK_K4, K]
    for k2 in range(0, OUT_K, BLOCK_OUT):
        acc2 += dot(acc1, W2_chunk)
```

这是 **tiled-gemm → elementwise → tiled-gemm** 的流水线模式。

## 自测框架

```python
def test_fused_mlp():
    B, N, K = 2, 128, 768
    K4 = 4 * K
    
    x = torch.randn(B, N, K, device='cuda', dtype=torch.float16)
    w1 = torch.randn(K, K4, device='cuda', dtype=torch.float16)
    w2 = torch.randn(K4, K, device='cuda', dtype=torch.float16)
    
    # Reference: unfused PyTorch
    ref = torch.nn.functional.gelu(x @ w1) @ w2
    
    # Your fused version
    yours = fused_mlp(x, w1, w2)
    
    # fp16 tolerance
    assert torch.allclose(yours.float(), ref.float(), atol=1e-2, rtol=1e-2)
    
    # Benchmark: measure memory bandwidth
    # Fused saves ~2 * N * K4 * element_size bytes of traffic
```

## 常见陷阱

1. **GELU 精度**：tanh 近似 vs 精确 erf。对于 MLP fusion，tanh 近似足够
2. **fp16 acc1 → fp16 acc2**：中间值可以保持 fp16，但 tl.dot 累加总是 fp32
3. **W2 的加载顺序**：W2[BLOK_K4, K] 的 stride 要和 acc1 的 tile 对齐
4. **不是所有 MLP 都用 GELU**：LLaMA 用 SiLU，GPT 用 GELU，确认模型
5. **bias**：Linear 可能有 bias，fusion 时要加上（会增加复杂度）

## 延伸

- **Fused QKV projection**：Q、K、V 一次加载 X，三次 dot，写三个输出
- **Fused Attention + MLP**：整个 transformer block 一个 kernel（极端优化）
- **torch.compile 能做到吗？**：`torch.compile(model, mode="max-autotune")` 经常能自动 fuse matmul+gelu，但不一定总能成功
