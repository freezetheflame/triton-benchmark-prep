# Triton Benchmark Prep

从零入门 OpenAI Triton GPU 编程 — 算子优化实战。

硬件：RTX 4070 SUPER 12GB | CUDA 13.0 | Compute Capability 8.9

## 环境

```bash
conda activate base
```

| 包 | 版本 |
|----|------|
| Python | 3.12 |
| PyTorch | 2.11.0 |
| Triton | 3.6.0 |
| CUDA | 13.0 |

验证环境：

```bash
python3 -c "import torch; import triton; print('OK: torch', torch.__version__, 'triton', triton.__version__)"
```

## Exercises

| # | 文件 | 核心概念 |
|---|------|---------|
| 0 | `00_vector_add.py` | program_id, grid, tl.load/store, mask, 内存带宽 |
| 1 | `01_activations.py` | tl.math, ReLU/GELU/SiLU, memory-bound ops |
| 2 | `02_softmax.py` | cross-thread reduction, online stable softmax, 数值稳定性 |
| 3 | `03_matmul.py` | tiling, tl.dot (Tensor Core), 自己写 matmul |
| 4 | `04_flash_attention.py` | fused kernel, online softmax in attention, O(1) memory |
| 5 | `05_profiling.py` | GPU timing, roofline model, arithmetic intensity 分析 |
| 6 | `06_autotune.py` | @triton.autotune, config space 搜索, 寄存器压力 |

### 自测

```bash
# 单个
python3 exercises/05_profiling.py

# 全部
for f in exercises/*.py; do echo "=== $(basename $f) ==="; python3 "$f"; echo; done
```

05_profiling.py 是最完整的——跑一次会输出所有 kernel 的性能表格和 roofline 分析，保存到 `profiles/`。

## 笔记

| 文件 | 内容 |
|------|------|
| `notes/pitfalls.md` | Triton 11 个常见陷阱（写之前先看） |
| `notes/online-softmax.md` | 分块 softmax 的数学原理 |
| `notes/flash-attention-theory.md` | FlashAttention 完全理解 |
| `notes/triton-developer-focus-2026.md` | Triton 社区和开发者聚焦 |
| `notes/layernorm-rmsnorm.md` | LayerNorm / RMSNorm 实现要点 + Welford 算法 |
| `notes/fused-mlp.md` | Fused MLP kernel 设计 + 消除中间张量 |

## 博客

完整系列已发布在 [blog.freezetheflame.cc](https://blog.freezetheflame.cc/posts/) — 10 篇 Triton 文章。

## 目录结构

```
triton-benchmark-prep/
├── README.md
├── exercises/           # 编程练习（6 个）
├── notes/               # 理论知识（6 篇）
├── blog-posts/          # 博客源文件
├── profiles/            # profiling 报告输出
└── benchmarks/          # benchmark 结果
```
