"""
Fused RoPE Triton Kernel — 将 RoPE 旋转直接应用到 Q/K，避免中间张量。

标准 PyTorch 实现：
  1. rotate_half(q)  → 1 个 (B, h, T, d_k) 中间张量
  2. q*cos + rot*sin → 1 个 (B, h, T, d_k) 中间张量
  Q 和 K 各重复一次，总共 4 个中间张量。

Fused kernel：读 Q/K + cos/sin → 就地写回，零中间分配。

参考:
  [1] Su et al., "RoFormer: Enhanced Transformer with Rotary Position
      Embedding," arXiv:2104.09864, 2021. Eq.(34).
  [2] Hsu et al., "Liger Kernel: Efficient Triton Kernels for LLM Training,"
      arXiv:2410.10989, 2024. Fused RoPE pattern（手写实现）。
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_rope_kernel(
    IN_ptr,       # (B*h, T, d_k) — 输入 Q 或 K
    OUT_ptr,      # (B*h, T, d_k) — 输出（可与 IN_ptr 相同实现就地）
    COS_ptr,      # (T, d_k) — 预计算的 cos 值
    SIN_ptr,      # (T, d_k) — 预计算的 sin 值
    stride_bh: tl.constexpr,    # batch*head 维 stride (= T * d_k)
    stride_t: tl.constexpr,     # seq_len 维 stride (= d_k)
    T: tl.constexpr,            # 序列长度
    HALF_DK: tl.constexpr,      # d_k // 2
):
    """
    RoPE 旋转: 对 QK 的每一行 (d_k,) 应用旋转。

    每个 program 处理一个 (batch*head, position) 对的完整 d_k 维度。
    Grid: (B*h*T,)

    RoPE 公式 [1] Eq.(34):
      q'[..., :half] = q[..., :half] * cos[:half] - q[..., half:] * sin[:half]
      q'[..., half:] = q[..., half:] * cos[:half] + q[..., :half] * sin[:half]

    Backward 时调用方传入 sin' = -sin 即可得到反向旋转。

    注意: RotaryEmbedding 输出的 cos/sin 形状为 (T, d_k)，
    其中 cos[:, :half] == cos[:, half:]（cat 结构），因此只读前半段。
    """
    pid = tl.program_id(0)
    bh = pid // T
    t_pos = pid % T

    base = bh * stride_bh + t_pos * stride_t
    cs_base = t_pos * (HALF_DK * 2)

    half_offsets = tl.arange(0, HALF_DK)

    q_first = tl.load(IN_ptr + base + half_offsets).to(tl.float32)
    q_second = tl.load(IN_ptr + base + HALF_DK + half_offsets).to(tl.float32)

    cos_val = tl.load(COS_ptr + cs_base + half_offsets).to(tl.float32)
    sin_val = tl.load(SIN_ptr + cs_base + half_offsets).to(tl.float32)

    new_first = q_first * cos_val - q_second * sin_val
    new_second = q_second * cos_val + q_first * sin_val

    tl.store(OUT_ptr + base + half_offsets, new_first.to(q_first.dtype))
    tl.store(OUT_ptr + base + HALF_DK + half_offsets, new_second.to(q_second.dtype))


def _apply_rope_kernel(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """对单个 tensor (B, h, T, d_k) 应用 RoPE，返回新张量（不修改输入）。"""
    B, h, T, d_k = x.shape
    HALF_DK = d_k // 2
    out = torch.empty_like(x)
    x_flat = x.view(B * h, T, d_k)
    out_flat = out.view(B * h, T, d_k)
    grid = (B * h * T,)
    _fused_rope_kernel[grid](
        x_flat, out_flat, cos, sin,
        stride_bh=x_flat.stride(0), stride_t=x_flat.stride(1),
        T=T, HALF_DK=HALF_DK,
    )
    return out


class _FusedRoPEFunction(torch.autograd.Function):
    """
    Autograd Function 包装 RoPE，保证 backward 数学正确性。

    Forward:  x' = x · cos + rotate_half(x) · sin
    Backward: dx = dy · cos - rotate_half(dy) · sin   (等价于用 -sin 做正向 RoPE)

    推导:
      旋转矩阵 R(θ) 是正交的，R(θ)^T = R(-θ)。
      forward 是线性变换 y = R(θ) · x，
      backward dx = R(θ)^T · dy = R(-θ) · dy，
      即用 sin' = -sin 调用同一个 forward kernel。
    """

    @staticmethod
    def forward(ctx, q, k, cos, sin):
        assert q.is_contiguous() and k.is_contiguous(), "q, k must be contiguous"
        assert q.shape == k.shape, "q and k must have same shape"
        d_k = q.shape[-1]
        assert d_k % 2 == 0, f"d_k must be even, got {d_k}"
        HALF_DK = d_k // 2
        assert HALF_DK > 0 and (HALF_DK & (HALF_DK - 1)) == 0, (
            f"d_k//2 ({HALF_DK}) must be a power of 2 for Triton kernel"
        )
        T = q.shape[2]
        assert cos.shape == (T, d_k) and sin.shape == (T, d_k)
        cos = cos.contiguous()
        sin = sin.contiguous()

        ctx.save_for_backward(cos, sin)
        q_out = _apply_rope_kernel(q, cos, sin)
        k_out = _apply_rope_kernel(k, cos, sin)
        return q_out, k_out

    @staticmethod
    def backward(ctx, dq, dk):
        cos, sin = ctx.saved_tensors
        # 反向旋转：sin → -sin（等价于转置）
        neg_sin = (-sin).contiguous()
        dq_in = dq.contiguous() if not dq.is_contiguous() else dq
        dk_in = dk.contiguous() if not dk.is_contiguous() else dk
        d_q = _apply_rope_kernel(dq_in, cos, neg_sin)
        d_k = _apply_rope_kernel(dk_in, cos, neg_sin)
        return d_q, d_k, None, None


def fused_apply_rotary_emb(q, k, cos, sin):
    """
    Fused RoPE 便捷接口（autograd-aware，out-of-place）。

    参数:
      q: (B, h, T, d_k) — query，contiguous
      k: (B, h, T, d_k) — key，contiguous
      cos: (T, d_k) — 预计算的 cos(position * freq)
      sin: (T, d_k) — 预计算的 sin(position * freq)
    返回:
      (q', k') — 旋转后的新张量（不修改输入，backward 自动正确）
    """
    return _FusedRoPEFunction.apply(q, k, cos, sin)
