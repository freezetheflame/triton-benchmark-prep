# FlashAttention 完全理解

> 读完这篇，你应该能回答：为什么标准 attention 有 O(n²) 显存问题？
> FlashAttention 怎么在不存完整 attention matrix 的情况下算出正确结果？

---

## 一、标准 Attention 为什么慢

Transformer 的 scaled dot-product attention：

```
S = Q @ K^T            # [seq_len, seq_len] —— 这就是问题！
P = softmax(S)         # [seq_len, seq_len] —— 存了整个矩阵
O = P @ V              # [seq_len, head_dim]
```

### 时间

- Q@K^T：O(n²d) 次乘法，n = seq_len, d = head_dim
- softmax：O(n²) 次 exp
- P@V：O(n²d) 次乘法
- 总计 O(n²d)，对长序列是灾难（2048² > 400 万）

### 显存（更大的问题）

- S 矩阵大小：seq_len × seq_len × sizeof(dtype)
- seq_len=2048, fp16：2048×2048×2 = 8MB（一个 head）
- seq_len=8192, fp16：128MB（一个 head）
- GPT-3 有 96 个 head：8192 时一个 batch 就要 12GB 只存 S 矩阵！
- **实际瓶颈是显存，不是计算**

GPU 计算 S 比从显存读写 S 快得多——所以 attention 是 **memory-bound**。

---

## 二、FlashAttention 的核心思想

**把 attention 写成一个 GPU kernel，中间结果不写回显存。**

标准做法（PyTorch）：
```
Q,K,V 在显存中
→ kernel 1: S = Q @ K^T，把 S 写回显存     ← 读写 O(n²)
→ kernel 2: P = softmax(S)，读 S，写 P      ← 读写 O(n²)
→ kernel 3: O = P @ V，读 P                 ← 读写 O(n²)
```

FlashAttention：
```
Q,K,V 在显存中
→ 一个 kernel：分块读 Q,K,V → 在 SRAM 里算 softmax → 写 O
  中间没有任何 O(n²) 矩阵写回显存
```

### tiling（分块）

把 Q 按行分成块（每块 BLOCK_M 行），K,V 按行分成块（每块 BLOCK_N 行）：

```
对于 Q 的每一块（BLOCK_M 行）：
    维护一个 running O、running max(m)、running denominator(d)
    对于 K,V 的每一块（BLOCK_N 行）：
        从显存加载 Q_block, K_block, V_block 到 SRAM
        在 SRAM 内计算 S_block = Q_block @ K_block^T  [BLOCK_M × BLOCK_N]
        用 online softmax 更新 running statistics
        累加 O += softmax(S_block) @ V_block
    把最终的 O_block 写回显存
```

关键在于：处理 K,V 的每一块时，只需要当前块的 S（大小 BLOCK_M×BLOCK_N），不需要存整个 S 矩阵。

---

## 三、Online Softmax（核心算法）

这是在整个 FlashAttention 里最精妙的部分。

### 标准 softmax 需要两轮扫描

```
softmax(x_i) = exp(x_i - max(x)) / Σ exp(x_j - max(x))

第一轮：找到 max(x)          ← 必须扫描全部元素
第二轮：算 exp 并求和归一化    ← 又扫描全部元素
```

FlashAttention 按块处理 K,V。如果等所有块都算完才知道 max(S)，就没法分块了。

### Online softmax 怎么做

处理每一块时，维护三个统计量：

- **m**（running max）：当前看到的最大值
- **d**（running denominator）：当前分母的累计值
- **O**（running output）：当前累加的输出

当新来一块时，发现新的最大值 m_new > m，就需要「重新缩放」之前的结果：

```
算法（处理第 j 块）：
  m_new = max(m, max(S_j))
  P_j = exp(S_j - m_new)                  # 当前块的 softmax 分子（未归一化）
  d_new = d * exp(m - m_new) + sum(P_j)   # 旧分母缩放 + 新分子
  O_new = (d * exp(m - m_new) * O + P_j @ V_j) / d_new
  m = m_new, d = d_new, O = O_new
```

### 直观理解

想象你在统计一个班的考试成绩，但只能一次看一个人的成绩：

1. 看到第一个人：分数 85。
   - m=85, d=1（暂时认为全班最高 85 分）, "平均"=85

2. 看到第二个人：分数 90。  ← 新的最高分！
   - 之前以为 85 是最高，现在知道 90 才是。之前的结果需要「贬值」
   - m_new=90
   - 旧的 85 现在要缩小：exp(85-90) = exp(-5) ≈ 0.0067
   - m=90, d = 1×exp(85-90) + exp(90-90) = 0.0067 + 1 = 1.0067

3. 继续处理，始终用当前的 max 做缩放

数学上这和最后一次性算完 softmax 的结果完全等价。

---

## 四、为什么分块不丢精度

关键性质：**exp(a-c) 的比值在任意 c 下保持不变**。

```
exp(x_i) / Σ exp(x_j)
= exp(x_i - M) / Σ exp(x_j - M)           # 减去任意常数 M 不改变结果
= exp(x_i - m_1) / Σ exp(x_j - m_1)       # m_1 是第一块的最大值
```

online softmax 每次更新 m 时，通过 `exp(m_old - m_new)` 对历史结果做正确的缩放，等价于最后一次性计算。

---

## 五、Forward vs Backward

以上只讲了 forward。Backward 更复杂：

- 反向传播需要知道 softmax 的中间值 P = softmax(S)
- 但 FlashAttention forward 没有存 P！
- 解决：**recomputation** —— backward 时重新从 Q,K,V 算 S 和 P
- 用 forward 保存的 m 和 d（logsumexp）加速 recomputation

这就是 **FlashAttention 反向传播的 O(n) 显存** 的来源：
- Forward：存 O（输出），m，d（logsumexp）→ O(n) 显存
- Backward：用 m,d 重新算 S → 再算梯度 → O(n) 显存

---

## 六、FlashAttention v1 vs v2 vs v3 的区别

| 版本 | 年份 | 关键改进 |
|------|------|---------|
| v1 | 2022 | 首次提出 fused kernel + tiling + online softmax |
| v2 | 2023 | 更好的并行策略（按 seq_len 维度分块而非 batch/head），减少非矩阵乘操作 |
| v3 | 2024 | 针对 Hopper 架构（H100），利用异步执行和 TMA 指令进一步提速 |

Triton 实现的 FlashAttention 通常对标 v2 算法。

---

## 七、关键数字

| seq_len | S 矩阵大小 (fp16) | 是否能在 12GB 卡上跑标准 attention |
|---------|-------------------|----------------------------------|
| 1024 | 2MB | ✅ |
| 2048 | 8MB | ✅ |
| 4096 | 32MB | ✅（但已经开始吃力） |
| 8192 | 128MB | ❌ OOM（96 heads 时） |
| 16384 | 512MB | ❌ |
| 32768 | 2GB | ❌ FlashAttention 才能跑 |

FlashAttention 让 **seq_len=32K** 成为可能——GPT-4 的 128K context 就是这样来的。

---

## 参考资料

- FlashAttention 论文：Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness", NeurIPS 2022
- FlashAttention-2：Dao, "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning", 2023
- Online softmax 算法：Milakov & Gimelshein, 2018（NVIDIA 内部技术报告）
