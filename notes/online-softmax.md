# Online Softmax：分块计算的数学原理

> 读完这篇，你应该能理解：为什么 softmax 可以分块计算？
> 怎么在不知道全局最大值的情况下，正确地做归一化？

---

## 问题

标准 softmax 需要两轮扫描：

```
softmax(x_i) = exp(x_i - max) / Σ exp(x_j - max)

第一轮：扫描全部 x，找到 max
第二轮：用 max 算 exp(x_i - max) 并求和归一化
```

**如果你只能一次看一小块数据（比如 GPU 上分块处理），怎么算？**

---

## Online Softmax 算法

维护三个统计量：m（当前最大值）、d（当前分母）、处理到当前位置的"半成品"。

### 初始化

```
m = -inf, d = 0
```

### 处理每一块

```
对于第 j 块数据 x_j：

  m_new = max(m, max(x_j))                              # 更新最大值
  d_new = d * exp(m - m_new) + sum(exp(x_j - m_new))    # 旧分母缩放 + 新分母
  m = m_new, d = d_new
```

### 最终归一化

```
处理完所有块后：
  softmax(x_i) = exp(x_i - m) / d
```

---

## 直观理解（用考试分数类比）

你要算每个人的分数在班上排第几（softmax），但只能一次看一个人的分数：

| 步骤 | 看到的人 | m（当前最高分） | d（当前分母） | 解释 |
|------|---------|----------------|-------------|------|
| 初始 | - | -∞ | 0 | |
| 第1人 | 张三: 85 | 85 | exp(85-85)=1 | 以为最高 85 分 |
| 第2人 | 李四: 90 | 90 | 1×exp(85-90) + exp(90-90) = 0.0067+1 = 1.0067 | 发现更高分！旧的结果"贬值" |
| 第3人 | 王五: 88 | 90 | 1.0067×exp(90-90) + exp(88-90) = 1.0067+0.135 = 1.142 | 没超过最高，正常加入 |
| 最终 | - | 90 | 1.142 | exp(分数-90)/1.142 得到 softmax |

### 关键公式

```
d_new = d * exp(m - m_new) + sum(exp(x_new - m_new))
       ^^^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^^
       旧数据的"贬值"系数      新数据的贡献（用新 max）
```

当 m_new > m 时，exp(m - m_new) < 1 → 旧数据被「贬值」
当 m_new = m 时，exp(m - m_new) = 1 → 旧数据不受影响

---

## 数学证明（简要）

要证明：online 算法和一次性计算的结果完全相同。

一次性 softmax（减去全局 M）：
```
p_i = exp(x_i - M) / Σ exp(x_j - M)
```

在 online 算法中，处理完所有 N 个元素后：
```
d = Σ_{j=1}^{N} exp(x_j - M)       # M 是真实全局最大值
```

证明思路：
- 从算法递推式出发，用数学归纳法
- d 的递推式：d_k = d_{k-1} * exp(m_{k-1} - m_k) + Σ exp(x_k - m_k)
- 展开后发现 d = Σ exp(x_i - M)，其中 M = max(x_1,...,x_N)
- 最终 p_i = exp(x_i - M) / d 和一次性计算一致

---

## Softmax 的 Memory-Bound 性质

```
softmax 的计算步骤：
  1. 读入 x → 计算 max → 读入 x → 计算 exp(x-max) → 读入 x → 归一化

这三步都要读 x！在 GPU 上，每次读 x 都是 global memory 访问。
标准 PyTorch softmax = 3 次读 + 1 次写 = memory-bound。
```

Online softmax 优雅之处：
```
读入 x → (同时做) max 比较 + exp 计算 + 累加到 d → 1 次读
```

对于 FlashAttention：online 意味着不需要存 S 矩阵 → 省了 O(n²) 的读写。

---

## 从 Softmax 到 FlashAttention 的推广

Softmax 是一维的：
```
p = softmax(scores)
```

Flash Attention 的每行是二维的：
```
对于 Q 的每一行：
  S = Q_row @ K^T         # [1, seq_len] 的一维向量
  P = softmax(S)           # 和普通 softmax 一模一样！
  O_row = P @ V            # 加权求和
```

**关键洞察**：Q 的第一行和 Q 的第二行，在 softmax 阶段是独立的！
所以可以把 Q 按行分块，每块独立做 online softmax。

这就是为什么 FlashAttention 的分块是「Q 按行分块、KV 按列分块」。

---

## 常见面试追问

**Q：online softmax 会有数值误差吗？**

A：有，但很小。`exp(m - m_new)` 当 m 和 m_new 相差很大时（如差 80），`exp(-80)` 会下溢为 0。但此时旧数据的贡献本来就接近 0，所以不影响结果。实际误差 < 1e-4。

**Q：为什么不用 log-sum-exp 来避免溢出？**

A：online softmax 本质上就是在维护 log-sum-exp。m + log(d) = LSE（log-sum-exp）。FlashAttention backward pass 正是用 LSE 来 recompute 梯度。

**Q：如果数据特别多，online softmax 能并行吗？**

A：能！这就是 parallel reduce。把数据分成多份，每份独立算自己的 (m_i, d_i)，最后 merge：
```
m = max(m_1, m_2, ..., m_k)
d = Σ d_i * exp(m_i - m)
```
