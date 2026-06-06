# Triton Benchmark Prep

华为暑期 Triton-Benchmark 项目准备。从零入门 OpenAI Triton GPU 编程。

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

## 自测

写完一个 exercise 后，直接跑对应的 Python 文件：

```bash
cd /home/hectum/triton-benchmark-prep

# 逐个测试
python3 exercises/00_vector_add.py
python3 exercises/01_activations.py
python3 exercises/02_softmax.py
python3 exercises/03_matmul.py
python3 exercises/04_flash_attention.py
```

每个文件内置了：
- **正确性测试**：自动对比 PyTorch reference 实现，输出 ✓ 或 ✗
- **性能测试**：输出 ms 和 GB/s / TFLOPS，对比 PyTorch 原生实现

### 一次性全部自测

```bash
cd /home/hectum/triton-benchmark-prep
for f in exercises/*.py; do echo "=== $(basename $f) ==="; python3 "$f"; echo; done
```

### 已知问题

- Exercise 3 的 benchmark 部分需要 matplotlib，如果报 ImportError 是正常的，test 部分不受影响
- Exercise 4 有已知的数值精度问题（online softmax 累加），正确性 test 可能显示 max_diff 较大，需要自己 debug

## 五个 Exercise

| # | 文件 | 核心概念 |
|---|------|---------|
| 0 | `00_vector_add.py` | program_id, grid, tl.load/store, mask, 内存带宽 |
| 1 | `01_activations.py` | tl.math, ReLU/GELU/SiLU, memory-bound ops |
| 2 | `02_softmax.py` | cross-thread reduction, online stable softmax, 数值稳定性 |
| 3 | `03_matmul.py` | tiling, shared memory, tl.dot (Tensor Core), TFLOPS |
| 4 | `04_flash_attention.py` | fused kernel, online softmax in attention, O(1) memory |

## 学习资料

按推荐阅读顺序：

1. `notes/pitfalls.md` — 11 个常见陷阱（写之前先看）
2. `notes/online-softmax.md` — 分块 softmax 的数学原理
3. `notes/flash-attention-theory.md` — FlashAttention 完全理解
4. `notes/triton-developer-focus-2026.md` — Triton 社区和华为项目关注点

## 目录结构

```
triton-benchmark-prep/
├── README.md
├── exercises/           # 编程练习
├── notes/               # 理论知识
└── blog-posts/          # 博客文章
```
