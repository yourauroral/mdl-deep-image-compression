"""
Fused Add+RMSNorm Triton Kernel — 残差加法 + RMSNorm 合并为单次 kernel launch。

用于 OLMo 2 post-norm 模式: y = residual + RMSNorm(sublayer_out)。

标准 PyTorch 实现：
  1. RMSNorm(sublayer_out)  → 读写 sublayer_out 一次
  2. residual + normed      → 读写两个张量一次
  总计 ~4 次 HBM 读写。

Fused kernel：读 residual, sublayer_out, weight → 写 y，~2 次 HBM 读写。

数学:
  rrms = 1 / sqrt(mean(sublayer_out²) + eps)
  y = residual + weight * sublayer_out * rrms

Backward:
  dy → d_residual = dy（直接传递）
  dy → d_sublayer = dy * w * rrms - sublayer_hat * mean(dy * w * sublayer_hat) * rrms
     其中 sublayer_hat = sublayer_out * rrms
  dw = sum_rows(dy * sublayer_hat)

参考:
  [1] Zhang & Sennrich, "Root Mean Square Layer Normalization,"
      NeurIPS 2019, arXiv:1910.07467.
  [2] OLMo 2 Tech Report, arXiv:2501.00656, 2025, Section 3.1.
      Post-norm: x = x + RMSNorm(sublayer(x))。
  [3] Triton Tutorials — Layer Normalization kernel.
  [4] Liger-Kernel, arXiv:2410.10989, 2024. Fused pattern（手写实现）。
"""

import torch
import triton
import triton.language as tl


# ─── Forward Kernel ──────────────────────────────────────────────
@triton.jit
def _fused_add_rms_norm_fwd_kernel(
    RESIDUAL_ptr,    # 残差输入, shape (..., N)
    SUBLAYER_ptr,    # sublayer 输出, shape (..., N)
    W_ptr,           # RMSNorm 权重, shape (N,)
    Y_ptr,           # 输出: y = residual + w * sublayer * rrms, shape (..., N)
    RRMS_ptr,        # 输出: 1/RMS, shape (M,)，供 backward 使用
    stride_r,        # RESIDUAL 行 stride
    stride_s,        # SUBLAYER 行 stride
    stride_y,        # Y 行 stride
    N,               # feature dimension
    eps,             # epsilon
    BLOCK_N: tl.constexpr,
):
    """
    每个 program 处理一行。
    y_i = residual_i + w_i * sublayer_i * rrms
    """
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    # 加载 residual, sublayer, w
    r = tl.load(RESIDUAL_ptr + row * stride_r + cols, mask=mask, other=0.0).to(tl.float32)
    s = tl.load(SUBLAYER_ptr + row * stride_s + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    # 计算 RMS(sublayer)
    s_sq = s * s
    var = tl.sum(s_sq, axis=0) / N
    rrms = 1.0 / tl.sqrt(var + eps)

    # y = residual + w * sublayer * rrms
    y = r + w * s * rrms

    tl.store(Y_ptr + row * stride_y + cols, y, mask=mask)
    tl.store(RRMS_ptr + row, rrms)


# ─── Backward Kernel ─────────────────────────────────────────────
@triton.jit
def _fused_add_rms_norm_bwd_kernel(
    DY_ptr,          # 上游梯度, shape (..., N)
    SUBLAYER_ptr,    # 保存的 sublayer 输出
    W_ptr,           # RMSNorm 权重
    RRMS_ptr,        # 保存的 1/RMS
    D_RESIDUAL_ptr,  # 输出: d_residual = dy
    D_SUBLAYER_ptr,  # 输出: d_sublayer
    DW_PARTIAL_ptr,  # 输出: dw 部分和 (M, N)
    stride_dy,
    stride_s,
    stride_dr,
    stride_ds,
    stride_dw,
    N,
    BLOCK_N: tl.constexpr,
):
    """
    Backward: y = residual + w * sublayer * rrms

    d_residual = dy（残差路径梯度直接传递）
    d_sublayer = rrms * (dy * w - sublayer_hat * mean(dy * w * sublayer_hat))
      其中 sublayer_hat = sublayer * rrms
    dw_row = dy * sublayer_hat
    """
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    dy = tl.load(DY_ptr + row * stride_dy + cols, mask=mask, other=0.0).to(tl.float32)
    s = tl.load(SUBLAYER_ptr + row * stride_s + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    rrms = tl.load(RRMS_ptr + row)

    # sublayer_hat = sublayer * rrms（归一化后的 sublayer）
    s_hat = s * rrms

    # d_residual = dy
    tl.store(D_RESIDUAL_ptr + row * stride_dr + cols, dy, mask=mask)

    # d_sublayer = rrms * (dy * w - s_hat * mean(dy * w * s_hat))
    dyw = dy * w
    c = tl.sum(dyw * s_hat, axis=0) / N
    d_sublayer = rrms * (dyw - s_hat * c)

    # dw_row = dy * s_hat
    dw_row = dy * s_hat

    tl.store(D_SUBLAYER_ptr + row * stride_ds + cols, d_sublayer, mask=mask)
    tl.store(DW_PARTIAL_ptr + row * stride_dw + cols, dw_row, mask=mask)


# ─── Autograd Function ───────────────────────────────────────────
class FusedAddRMSNormFunction(torch.autograd.Function):
    """
    y = residual + RMSNorm(sublayer_out, weight)

    Forward: 一次 kernel 完成 add + norm
    Backward: d_residual = dy, d_sublayer 通过 RMSNorm backward 计算
    """

    @staticmethod
    def forward(ctx, residual, sublayer_out, weight, eps):
        assert residual.shape == sublayer_out.shape, (
            f"residual.shape ({residual.shape}) != sublayer_out.shape ({sublayer_out.shape})"
        )
        assert weight.ndim == 1 and weight.shape[0] == residual.shape[-1], (
            f"weight.shape ({weight.shape}) 与 residual 最后一维 ({residual.shape[-1]}) 不匹配"
        )
        x_shape = residual.shape
        N = x_shape[-1]
        M = residual.numel() // N

        # 展平为 (M, N)
        r_flat = residual.contiguous().view(M, N)
        s_flat = sublayer_out.contiguous().view(M, N)

        y_flat = torch.empty_like(r_flat)
        rrms = torch.empty(M, device=residual.device, dtype=torch.float32)

        BLOCK_N = triton.next_power_of_2(N)

        _fused_add_rms_norm_fwd_kernel[(M,)](
            r_flat, s_flat, weight, y_flat, rrms,
            stride_r=r_flat.stride(0),
            stride_s=s_flat.stride(0),
            stride_y=y_flat.stride(0),
            N=N, eps=eps,
            BLOCK_N=BLOCK_N,
        )

        ctx.save_for_backward(s_flat, weight, rrms)
        ctx.N = N
        ctx.M = M
        ctx.BLOCK_N = BLOCK_N

        return y_flat.view(x_shape)

    @staticmethod
    def backward(ctx, dy):
        s_flat, weight, rrms = ctx.saved_tensors
        N = ctx.N
        M = ctx.M
        BLOCK_N = ctx.BLOCK_N

        dy_flat = dy.contiguous().view(M, N)

        d_residual = torch.empty_like(dy_flat)
        d_sublayer = torch.empty_like(s_flat)
        dw_partial = torch.empty((M, N), device=dy.device, dtype=torch.float32)

        _fused_add_rms_norm_bwd_kernel[(M,)](
            dy_flat, s_flat, weight, rrms,
            d_residual, d_sublayer, dw_partial,
            stride_dy=dy_flat.stride(0),
            stride_s=s_flat.stride(0),
            stride_dr=d_residual.stride(0),
            stride_ds=d_sublayer.stride(0),
            stride_dw=dw_partial.stride(0),
            N=N,
            BLOCK_N=BLOCK_N,
        )

        dw = dw_partial.sum(dim=0).to(weight.dtype)

        return (d_residual.view(dy.shape),
                d_sublayer.view(dy.shape),
                dw,
                None)  # eps 无梯度


# ─── 便捷函数 ────────────────────────────────────────────────────
def fused_add_rms_norm(residual, sublayer_out, weight, eps=1e-10):
    """
    函数式接口: y = residual + RMSNorm(sublayer_out, weight, eps)。

    参数:
      residual:     (..., N) 残差输入
      sublayer_out: (..., N) sublayer 输出（attention 或 FFN）
      weight:       (N,)    RMSNorm 权重
      eps:          float   epsilon
    返回:
      y: (..., N) = residual + weight * sublayer_out / RMS(sublayer_out)
    """
    return FusedAddRMSNormFunction.apply(residual, sublayer_out, weight, eps)
