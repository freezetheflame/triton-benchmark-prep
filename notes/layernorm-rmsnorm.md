# LayerNorm & RMSNorm — LLM 的"无声工作者"

> 每个 Transformer block 至少两个 LayerNorm，一个 7B 模型有 32-80 层。
> 这些 normalization kernel 在端到端推理中占比 5-15%，优化它们直接降低延迟。

## 算法

### LayerNorm

```
Input:  x ∈ R^{N, D}  (N tokens, D hidden dim)
Output: y ∈ R^{N, D}

对于每行（每个 token 独立）：
  μ = mean(x)      # 均值
  σ² = var(x)       # 方差
  x̂ = (x - μ) / sqrt(σ² + ε)
  y = γ * x̂ + β     # affine transform (learnable params)
```

关键：**每行独立**，天然适合并行——每个 program 处理一行。

### RMSNorm（LLaMA 系列用的）

```
RMS(x) = sqrt(mean(x²) + ε)
y = x / RMS(x) * γ       # 没有 β，没有减均值
```

比 LayerNorm 少一次 reduction（不需要 mean），更快。

## Triton 实现要点

### 需要两次遍历（或一次 online 算法）

**两遍法**（最直观）：
```
Pass 1: 计算 mean 和 var（或 RMS）
  - 分块加载，Welford 算法累积
  - Welford: 可以在单次遍历中计算 mean 和 var，不需要存所有值

Pass 2: normalize + affine
  - 加载同一块数据，应用 x̂ = (x - μ) / σ * γ + β
  - 写回
```

**一遍法**（fused）：
```
把两次遍历合并，但需要先完整算出 μ 和 σ² 才能 normalize
→ 必须用 2-pass，或者用 inter-block reduction
```

### Block 内的 Welford 算法

```
# 对单个 block 内的值 [x₁, x₂, ..., x_BLOCK]
m_old = 0, s_old = 0
for x in block:
    m_new = m_old + (x - m_old) / count
    s_new = s_old + (x - m_old) * (x - m_new)
    m_old, s_old = m_new, s_new

# 结果：
mean = m_new
var = s_new / count
```

### 跨 Block 合并（关键难点）

每个 block 有自己的 (count, mean, M2)，需要合并：

```
# 合并两组统计量 (n_a, m_a, s_a) 和 (n_b, m_b, s_b)
n_ab = n_a + n_b
delta = m_b - m_a
m_ab = m_a + delta * n_b / n_ab
s_ab = s_a + s_b + delta² * n_a * n_b / n_ab
```

在 Triton 中，跨 block 的 reduction 需要**两次 kernel launch**：
1. Kernel 1: 每个 block 计算自己的 (count, mean, M2)，存到临时 buffer
2. Kernel 2: 合并临时 buffer，计算全局统计量
3. Kernel 3: 用全局统计量 normalize

**优化**：如果 `D ≤ 1024`（大多数 LLM 的 hidden dim），一行可以放进一个 block，不需要跨 block 合并，单个 kernel 搞定。

## 自测框架

```python
# test_layernorm.py 骨架
import torch

def test_layernorm():
    """你的 LayerNorm kernel 应该通过这些测试"""
    
    # Test 1: 基本正确性
    B, N, D = 2, 128, 768
    x = torch.randn(B, N, D, device='cuda')
    ref = torch.nn.functional.layer_norm(x, (D,))
    yours = your_layernorm(x)
    assert torch.allclose(yours, ref, atol=1e-4), f"Failed: max_diff={(yours-ref).abs().max()}"
    
    # Test 2: 大 token 数（激活值场景）
    x = torch.randn(32, 2048, 512, device='cuda')
    ref = torch.nn.functional.layer_norm(x, (D,))
    # ...

    # Test 3: RMSNorm 对比
    # RMSNorm 等价于 LayerNorm(..., elementwise_affine=False) 
    # 但去掉 mean subtraction
    
    # Test 4: 性能
    # vs PyTorch, vs torch.compile, vs apex

if __name__ == "__main__":
    test_layernorm()
```

## 常见陷阱

1. **ε 值**：PyTorch 默认 1e-5，确保一致
2. **fp16 累加**：必须用 fp32 做 Welford，最后 cast
3. **affine 参数位置**：γ 和 β 存在 global memory，每次都要 load
4. **D 不是 2 的幂**：mask 要处理好最后几个元素
5. **RMSNorm 的去均值**：LLaMA 的 RMSNorm 不做 mean subtraction，测试时别搞混

## 延伸

- **Fused LayerNorm + Dropout**：把 dropout 也融进去
- **GroupNorm**：对 channel group 做 norm，stable diffusion 用
- **torch.compile 能自动 fuse 吗？** 有时能，但手写更可控
