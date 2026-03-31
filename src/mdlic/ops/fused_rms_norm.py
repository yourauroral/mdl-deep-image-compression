"""
Fused RMSNorm Triton Kernel — forward + backward.

RMSNorm(x) = w * x / RMS(x),  RMS(x) = sqrt(mean(x²) + eps)

相比 PyTorch 实现（layers.py RMSNorm），fused kernel 将归一化、缩放
合并为一次 kernel launch，减少 HBM 读写和 kernel launch 开销。

数学推导:
  Forward:
    rrms = 1 / sqrt(mean(x²) + eps)     ... reciprocal RMS
    y = w * x * rrms

  Backward (dy → dx, dw):
    令 N = feature dimension，s = sum(x²)/N + eps，rrms = 1/sqrt(s)
    y_i = w_i * x_i * rrms

    ∂L/∂x_i = dy_i * w_i * rrms + x_i * (-1/(2*s^{3/2})) * (2*x_i/N) * sum_j(dy_j * w_j * x_j)
            = rrms * (dy_i * w_i - x_i * rrms² * mean(dy · w · x))
            = rrms * (dy_i * w_i - x_hat_i * mean(dy · w · x_hat))
      其中 x_hat_i = x_i * rrms（即归一化后的 x）。

    ∂L/∂w_i = dy_i * x_i * rrms      ... 沿 batch 维度累加

参考:
  [1] Zhang & Sennrich, "Root Mean Square Layer Normalization,"
      NeurIPS 2019, arXiv:1910.07467.
  [2] Triton Tutorials — Layer Normalization kernel.
      https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html
  [3] Liger-Kernel, "Efficient Triton Kernels for LLM Training,"
      arXiv:2410.10989, 2024. Fused RMSNorm pattern.
"""

import torch
import triton
import triton.language as tl


# ─── Forward Kernel ──────────────────────────────────────────────
@triton.jit
def _rms_norm_fwd_kernel(
    X,          # 输入指针,  shape (..., N)
    W,          # 权重指针,  shape (N,)
    Y,          # 输出指针,  shape (..., N)
    RRMS,       # 1/RMS 指针, shape (M,)，M = 总行数
    stride_x,   # X 在最后一维上的行 stride (即一行有 N 个元素时的 stride)
    stride_y,   # Y 在最后一维上的行 stride
    N,          # feature dimension（最后一维大小）
    eps,        # epsilon
    BLOCK_N: tl.constexpr,  # block size，覆盖整个 N（要求 BLOCK_N >= N）
):
    """
    每个 program 处理一行（row）。
    program_id(0) = 行索引 m。
    """
    row = tl.program_id(0)

    # 计算当前行在 X / Y 中的起始偏移
    X_row_ptr = X + row * stride_x
    Y_row_ptr = Y + row * stride_y

    # 列偏移: 0, 1, ..., BLOCK_N-1
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    # ── 1. 加载 x 和 w ──
    x = tl.load(X_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)

    # ── 2. 计算 RMS ──
    # mean(x²) = sum(x²) / N
    x_sq = x * x
    var = tl.sum(x_sq, axis=0) / N   # scalar: mean(x²)
    rrms = 1.0 / tl.sqrt(var + eps)   # 1 / RMS

    # ── 3. 归一化并缩放 ──
    y = x * rrms * w

    # ── 4. 写回 ──
    tl.store(Y_row_ptr + cols, y, mask=mask)
    # 保存 rrms 供 backward 使用（每行一个标量）
    tl.store(RRMS + row, rrms)


# ─── Backward Kernel ─────────────────────────────────────────────
@triton.jit
def _rms_norm_bwd_kernel(
    DY,         # 上游梯度指针, shape (..., N)
    X,          # 输入指针,     shape (..., N)
    W,          # 权重指针,     shape (N,)
    RRMS,       # 1/RMS 指针,   shape (M,)
    DX,         # 输出: dx,    shape (..., N)
    DW_PARTIAL, # 输出: dw 部分和, shape (M, N) — 每行一份，后续在 Python 侧 sum
    stride_x,
    stride_dy,
    stride_dx,
    stride_dw_partial,  # DW_PARTIAL 的行 stride
    N,
    BLOCK_N: tl.constexpr,
):
    """
    每个 program 处理一行。

    dx_i = rrms * (dy_i * w_i - x_hat_i * mean(dy · w · x_hat))
      其中 x_hat_i = x_i * rrms

    dw_i（当前行贡献）= dy_i * x_i * rrms
    """
    row = tl.program_id(0)

    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    # 加载数据
    dy = tl.load(DY + row * stride_dy + cols, mask=mask, other=0.0).to(tl.float32)
    x  = tl.load(X  + row * stride_x  + cols, mask=mask, other=0.0).to(tl.float32)
    w  = tl.load(W  + cols, mask=mask, other=0.0).to(tl.float32)
    rrms = tl.load(RRMS + row)

    # x_hat = x * rrms（归一化后的 x）
    x_hat = x * rrms

    # dy · w · x_hat 的均值（用于 dx 计算）
    # mean(dy * w * x_hat) = sum(dy * w * x_hat) / N
    dyw_xhat = dy * w * x_hat
    c = tl.sum(dyw_xhat, axis=0) / N   # scalar

    # dx = rrms * (dy * w - x_hat * c)
    dx = rrms * (dy * w - x_hat * c)

    # dw（当前行贡献）= dy * x_hat
    dw_row = dy * x_hat

    # 写回
    tl.store(DX + row * stride_dx + cols, dx, mask=mask)
    tl.store(DW_PARTIAL + row * stride_dw_partial + cols, dw_row, mask=mask)


# ─── Autograd Function ───────────────────────────────────────────
class FusedRMSNormFunction(torch.autograd.Function):
    """
    torch.autograd.Function 封装，连接 Triton kernel 与 PyTorch autograd。
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, eps: float):
        # x: (..., N)，weight: (N,)
        x_shape = x.shape
        N = x_shape[-1]
        assert weight.ndim == 1 and weight.shape[0] == N, (
            f"weight.shape ({weight.shape}) 与 x 最后一维 ({N}) 不匹配"
        )
        M = x.numel() // N  # 总行数

        # 展平为 (M, N)
        x_flat = x.contiguous().view(M, N)

        # 分配输出
        y_flat = torch.empty_like(x_flat)
        rrms = torch.empty(M, device=x.device, dtype=torch.float32)

        # BLOCK_N: 取 >= N 的最小 2 的幂
        BLOCK_N = triton.next_power_of_2(N)

        # 启动 kernel: 每行一个 program
        _rms_norm_fwd_kernel[(M,)](
            x_flat, weight, y_flat, rrms,
            stride_x=x_flat.stride(0),
            stride_y=y_flat.stride(0),
            N=N,
            eps=eps,
            BLOCK_N=BLOCK_N,
        )

        ctx.save_for_backward(x_flat, weight, rrms)
        ctx.N = N
        ctx.BLOCK_N = BLOCK_N
        ctx.M = M

        return y_flat.view(x_shape)

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        x_flat, weight, rrms = ctx.saved_tensors
        N = ctx.N
        M = ctx.M
        BLOCK_N = ctx.BLOCK_N

        dy_flat = dy.contiguous().view(M, N)

        # 分配输出
        dx_flat = torch.empty_like(x_flat)
        # dw 需要对所有行求和，先存每行的部分和
        dw_partial = torch.empty((M, N), device=x_flat.device, dtype=torch.float32)

        _rms_norm_bwd_kernel[(M,)](
            dy_flat, x_flat, weight, rrms,
            dx_flat, dw_partial,
            stride_x=x_flat.stride(0),
            stride_dy=dy_flat.stride(0),
            stride_dx=dx_flat.stride(0),
            stride_dw_partial=dw_partial.stride(0),
            N=N,
            BLOCK_N=BLOCK_N,
        )

        # dw = sum over all rows
        dw = dw_partial.sum(dim=0).to(weight.dtype)

        return dx_flat.view(dy.shape), dw, None  # None for eps


# ─── 便捷函数 ────────────────────────────────────────────────────
def fused_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-10):
    """函数式接口，直接调用 FusedRMSNormFunction。"""
    return FusedRMSNormFunction.apply(x, weight, eps)


# ─── nn.Module 封装 ──────────────────────────────────────────────
class FusedRMSNorm(torch.nn.Module):
    """
    Fused RMSNorm 模块，可直接替换 layers.py 中的 RMSNorm。

    用法:
      # 替换前
      self.norm = RMSNorm(d_model)
      # 替换后
      from mdlic.ops.fused_rms_norm import FusedRMSNorm
      self.norm = FusedRMSNorm(d_model)
    """

    def __init__(self, features: int, eps: float = 1e-10):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return fused_rms_norm(x, self.weight, self.eps)
