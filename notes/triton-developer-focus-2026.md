# 2026 年 Triton 开发者的关注点

> 如果你去华为做 triton-benchmark，面试官和同事关心的就是这些。

---

## 一、Triton vs 手写 CUDA vs 编译器

### Triton 的定位

```
开发效率:  PyTorch  >>>  Triton  >>  CUDA  >>>  PTX/汇编
性能天花板: PTX/汇编  >  CUDA  >  Triton  >>  PyTorch

Triton 占据的是"90% 的 CUDA 性能，10% 的开发成本"这个甜点区。
```

### 当前争论焦点

**"Triton 真的能替代手写 CUDA 吗？"**

- **正方**：FlashAttention 的 Triton 实现在 A100 上达到 cuDNN 的 95%+ 性能
- **反方**：matmul 的 Triton 实现仍然比 cuBLAS 慢 10-20%
- **现实**：90% 的 kernel 不需要极致优化，Triton 够了。剩下 10% 才需要手写 CUDA

**为什么 matmul 不如 cuBLAS？**

cuBLAS 积累了几十年的手工优化：
- 针对每种 GPU 架构（V100/A100/H100）有独立的手写汇编 kernel
- 针对每种 shape（M,N,K 组合）有最优 tile size 表
- 利用 warp-level 矩阵乘加指令（MMA）做指令级优化
- Tensor Core 的 bank conflict 规避

Triton 的自动调优（autotune）在大多数 shape 上接近但无法超越这个水平。

---

## 二、Triton 3.x 生态现状（2026）

### 关键变化

| 特性 | 2.x | 3.x |
|------|-----|-----|
| 后端 | 仅 NVIDIA | NVIDIA + AMD (ROCm) + Intel |
| 编程模型 | 基于 block | 更细粒度的 warp specialization |
| 调试 | 几乎无法调试 | `tl.device_print` 可以在 kernel 内打印 |
| torch.compile | 独立使用 | 可通过 `torch.compile` 自动生成 Triton kernel |
| 华为 Ascend | 不支持 | **正在适配中**  |

### Ascend NPU 适配的关键问题

1. **指令集不同**：Ascend 的 Cube 单元和 NVIDIA Tensor Core 不同
2. **显存架构不同**：Ascend 的 L1/L2 cache 大小和 NVIDIA 完全不同
3. **最优 BLOCK_SIZE 不同**：NVIDIA 的 128×128 tile 在 Ascend 上可能很差
4. **Dtype 支持差异**：Ascend 的 fp16/bf16/fp8 实现和精度与 NVIDIA 不同
5. **Triton compiler 需要生成 DaVinci 指令而非 PTX**

这就是 triton-benchmark 项目的核心——在 Ascend 上跑通 Triton 算子，找出性能问题和最优配置。

---

## 三、Triton Kernel 优化方向（按重要性排序）

### 1. Tiling 策略（最大性能影响）

```
block 太小 → 计算密度不够，overhead 占比高
block 太大 → 寄存器溢出到 local memory，性能暴跌
block 形状 → 方形的 128×128 不一定最优
```

**关注点**：不同 GPU 架构的最优 BLOCK_M × BLOCK_N × BLOCK_K 组合是什么？

### 2. Memory Coalescing（显存合并访问）

```
GPU 一次读 32/128 字节的 cache line
如果 32 个线程访问的显存地址连续 → 1 次事务
如果 32 个线程访问的显存地址随机 → 32 次事务（慢 32 倍！）
```

**关注点**：Triton 的 `tl.load` 是否总是生成 coalesced 访问？跨 stride 访问怎么优化？

### 3. Bank Conflict（共享内存冲突）

```
Shared memory 有 32 个 bank
同一个 warp 的线程访问同一 bank 的不同地址 → 串行化
```

**关注点**：matmul 的 tile 加载是否触发了 bank conflict？padding 策略（如 BLOCK_K+4）的效果？

### 4. Occupancy（占用率）

```
每个 SM 能同时跑的 block 数量
受限于：寄存器数量、shared memory 大小
如果每个 block 用太多寄存器 → occupancy 低 → 延迟隐藏不好
```

**关注点**：Triton kernel 用了多少寄存器？能否减少以提升 occupancy？

### 5. 数值精度

```
fp16 累加 → 精度不够
fp32 累加 → 显存带宽翻倍
bf16 → 范围和 fp32 一样，精度略低，对 ML 足够
fp8 → 最新的 H100 支持，2x 于 fp16 的吞吐
```

**关注点**：混合精度策略——什么时候用 fp16 计算、什么时候升到 fp32？

### 6. L2 Cache 优化

```
GPU 的 L2 cache 是所有 SM 共享的
连续的 program_id 应该处理相邻的数据块 → 利用 L2 cache locality
这就是 matmul 中 GROUP_M swizzling 的作用
```

---

## 四、当前热门研究方向

### 1. Triton + torch.compile

PyTorch 2.x 的 `torch.compile` 默认后端是 Triton。用户写普通 PyTorch 代码，编译器自动生成 Triton kernel。

**问题**：自动生成的 kernel 质量如何？什么情况下需要手写 Triton？

### 2. 长上下文 Attention

FlashAttention 让 seq_len 能到 128K。但还不够——Llama 4 支持 1M context。

**新算法**：
- Ring Attention：多 GPU 环形通信分块 attention
- Tree Attention：类似 reduce 的并行化
- Mamba/SSM：不用 attention 的替代架构

### 3. FP8 训练和推理

H100 支持 FP8 Tensor Core，2x 吞吐于 FP16。Triton 3.x 开始支持 FP8。

**问题**：FP8 的数值稳定性？哪些层可以用 FP8、哪些不行？

### 4. Sparse Attention / Sparse Matmul

不是所有 token 对都需要 attention。稀疏化后 O(n²)→O(n log n)。

Triton 的 block-sparse 支持：只计算非零 block 的 attention。

### 5. Multi-modality 的算子需求

视频、音频、图像的 token 序列长度远超文本。需要针对极长序列的 attention 优化。

---

## 五、你去华为后可能的日常工作

1. **跑 benchmark**：在 Ascend 上跑 Triton kernel，记录正确性和性能
2. **分析差距**：为什么同一个 Triton kernel 在 Ascend 上比 NVIDIA 慢？
3. **调 BLOCK_SIZE**：找 Ascend 上的最优 tile 配置
4. **修精度 bug**：fp16/bf16 在不同架构上的行为差异
5. **写报告**：「在 Ascend 910B 上，Triton 实现的 FlashAttention 达到 NVIDIA A100 的 X% 性能」

---

## 六、推荐的阅读顺序

1. Triton 官方教程（前 4 个）→ 基本语法
2. FlashAttention 论文 → 理解为什么 tiling
3. 本目录的 `flash-attention-theory.md` → 中文梳理
4. 看 vLLM 的 attention kernel 怎么用 Triton 写的
5. 自己动手写 → 就是你现在在做的
