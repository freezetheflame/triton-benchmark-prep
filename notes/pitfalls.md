# Triton 常见陷阱与最佳实践

> 来源于社区经验、源码阅读和实际踩坑记录。
> 每个操作 Triton kernel 的人都应该先读一遍。

---

## 陷阱 1：Grid 维度理解错误

```
错误理解：grid=(M, N) 表示 M 行 N 列的 2D grid，每个 block 处理一个 (i,j)
正确理解：grid 是 program_id 的笛卡尔积，program_id(axis=0) 对应第一个维度

grid = (A, B, C)
  → program_id(0) ∈ [0, A)   第一维
  → program_id(1) ∈ [0, B)   第二维
  → program_id(2) ∈ [0, C)   第三维

关键：grid 的语义由 kernel INTERNAL 决定，不是 Triton 强制的。
     grid=(M*N,) 的 1D grid 也可以模拟 2D 分块。
```

### 常见错误
```python
# 错误：把 grid 维度当成 (batch, blocks_per_seq)
grid = (B, triton.cdiv(seq_len, BLOCK))

# 正确：Kernel 内部自己 decode program_id
pid = tl.program_id(0)
batch_idx = pid // num_blocks
block_idx = pid % num_blocks
```

---

## 陷阱 2：tl.arange 的广播语义

```python
offs_m = tl.arange(0, BLOCK_M)     # shape [BLOCK_M]
offs_n = tl.arange(0, BLOCK_N)     # shape [BLOCK_N]

# 二维索引时，broadcast 方向至关重要：
offs_m[:, None]   # shape [BLOCK_M, 1]  → 沿列广播
offs_n[None, :]   # shape [1, BLOCK_N]  → 沿行广播

# 加载 2D tile：
a = tl.load(ptr + offs_m[:, None] * stride_m + offs_n[None, :] * stride_n)
#             形状: [BLOCK_M, BLOCK_N]

# 常见错误：忘记加 None 导致 shape mismatch
a = tl.load(ptr + offs_m * stride_m + offs_n * stride_n)  # 错！形状广播不对
```

---

## 陷阱 3：Mask 必须匹配加载形状

```python
offs = tl.arange(0, BLOCK_SIZE)       # shape [BLOCK_SIZE]
mask = offs < n_elements               # shape [BLOCK_SIZE]

# 对于 2D 加载：
offs_m = tl.arange(0, BLOCK_M)         # [BLOCK_M]
offs_n = tl.arange(0, BLOCK_N)         # [BLOCK_N]

# 需要两个维度的 mask
mask_m = offs_m[:, None] < M           # [BLOCK_M, 1]
mask_n = offs_n[None, :] < N           # [1, BLOCK_N]
combined_mask = mask_m & mask_n        # [BLOCK_M, BLOCK_N]

# 常见错误：只给一个维度 mask
tl.store(ptr, data, mask=offs_m < M)   # 错！mask 形状不匹配 data
```

---

## 陷阱 4：tl.constexpr 不会自动推导

```python
# 错误：期望 Triton 从 shape 推导 BLOCK_SIZE
@triton.jit
def kernel(x_ptr, BLOCK_SIZE):  # ← 这不是 constexpr！
    ...

# 正确：必须显式声明 tl.constexpr
@triton.jit
def kernel(x_ptr, BLOCK_SIZE: tl.constexpr):
    ...

# 且调用时必须作为 keyword argument：
kernel[grid](x, N, BLOCK_SIZE=256)  # ← keyword，不是 positional
```

---

## 陷阱 5：float16/bf16 的累加精度

```python
# 错误：用 float16 做累加 → 精度损失严重
acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float16)  # 不要这样做！

# 正确：累加器用 float32，输出时才 cast
acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)  # 累加用 fp32
# ... 计算 ...
output = acc.to(tl.float16)  # 最后 cast
```

---

## 陷阱 6：循环内重复加载同一数据

```python
# 差：每次循环都从 global memory 加载
for k in range(K):
    a = tl.load(a_ptr + offsets + k)  # K 次 global load → 极慢

# 好：一次加载一个 tile 到寄存器
for k_start in range(0, K, BLOCK_K):
    a_tile = tl.load(a_ptr + offsets + k_start * stride)  # K/BLOCK_K 次
```

---

## 陷阱 7：忽视线程 divergence

```python
# 差：条件判断导致 warp divergence
if tl.program_id(0) % 2 == 0:
    x = expensive_compute(x)   # 一半线程空转
else:
    x = simple_op(x)

# 好：按 block 分组，避免 block 内 divergence
# Triton 的 block 对应 CUDA 的 thread block，
# block 内所有线程执行同一个 kernel instance
```

---

## 陷阱 8：Triton 调试极其困难

```python
# 1. 不能用 print()（JIT 编译后没有 Python runtime）
@triton.jit
def kernel(x):
    print(x)  # ← 这不会工作！

# 2. 调试策略：
# - 先用 PyTorch 写 reference 实现，确保算法正确
# - 用小尺寸（如 BLOCK=32）测试 Triton kernel
# - 用 assert torch.allclose(triton_result, reference)
# - 用 TRITON_INTERPRET=1 环境变量启用解释器模式（慢但可调试）
# - 最近版本支持 tl.device_print("msg", val) 打印中间值
```

---

## 陷阱 9：bank conflict

```python
# Shared memory bank conflict 发生在所有线程访问同一 bank 时
# Triton 通常自动处理，但在某些 stride 访问时仍可能发生

# 避免方法：
# 1. 确保 BLOCK_K 不是 32 的整数倍（对 fp32）
# 2. 添加 padding：BLOCK_K + 4 而不是 BLOCK_K
# 3. 使用 triton.autotune 自动搜索最优配置
```

---

## 陷阱 10：首次运行慢（JIT 编译）

```python
# 第一次调用 Triton kernel 会触发 JIT 编译
# 生产代码必须 warmup

for _ in range(10):
    kernel[grid](...)
torch.cuda.synchronize()  # 等待编译完成

# 然后才计时
start = time.time()
kernel[grid](...)
```

---

## 陷阱 11：Triton 缺少常见数学函数

```python
# Triton 3.x 没有 tanh、erf 等高级数学函数
# 需要手动实现：

# tanh(z) = (e^(2z) - 1) / (e^(2z) + 1)
exp_2z = tl.exp(2.0 * z)
tanh_z = (exp_2z - 1.0) / (exp_2z + 1.0)

# GELU 的 erf 近似：
# GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))

# 可用的基本函数：
# tl.exp, tl.log, tl.sqrt, tl.abs, tl.maximum, tl.minimum
# tl.sin, tl.cos  (部分版本)
# tl.sigmoid (可用)
```

---

## 华为 Triton-Benchmark 项目特有的注意事项

1. **跨架构**：Ascend NPU 的 Triton 可能有不同的 BLOCK_SIZE 最优值
2. **Dtype 支持**：fp16/bf16/int8/fp8，不是所有架构都支持所有 dtype
3. **Triton 版本差异**：不同 Triton 版本生成的 PTX 可能不同
4. **算子覆盖**：matmul、attention、norm、activation、reduction 全覆盖
5. **Benchmark 规范**：warmup + 多次测量 + percentile 报告
