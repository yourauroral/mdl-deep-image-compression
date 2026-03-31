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
    QK_ptr,       # (B*h, T, d_k) — Q 或 K，就地修改
    COS_ptr,      # (T, d_k) — 预计算的 cos 值
    SIN_ptr,      # (T, d_k) — 预计算的 sin 值
    stride_bh: tl.constexpr,    # QK 的 batch*head 维 stride (= T * d_k)
    stride_t: tl.constexpr,     # QK 的 seq_len 维 stride (= d_k)
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

    注意: RotaryEmbedding 输出的 cos/sin 形状为 (T, d_k)，
    其中 cos[:, :half] == cos[:, half:]（cat 结构），因此只读前半段。
    """
    pid = tl.program_id(0)
    bh = pid // T
    t_pos = pid % T

    # QK 中当前 (bh, t_pos) 的起始偏移
    base = bh * stride_bh + t_pos * stride_t
    # cos/sin 中当前 t_pos 的起始偏移（stride = d_k = 2*HALF_DK）
    cs_base = t_pos * (HALF_DK * 2)

    half_offsets = tl.arange(0, HALF_DK)

    # 读取 q 的前半段和后半段
    q_first = tl.load(QK_ptr + base + half_offsets).to(tl.float32)
    q_second = tl.load(QK_ptr + base + HALF_DK + half_offsets).to(tl.float32)

    # 读取 cos, sin（只需前半段，因为 cos/sin 前后对称）
    cos_val = tl.load(COS_ptr + cs_base + half_offsets).to(tl.float32)
    sin_val = tl.load(SIN_ptr + cs_base + half_offsets).to(tl.float32)

    # RoPE 旋转 [1] Eq.(34):
    #   q'_first  = q_first * cos - q_second * sin
    #   q'_second = q_second * cos + q_first * sin
    new_first = q_first * cos_val - q_second * sin_val
    new_second = q_second * cos_val + q_first * sin_val

    # 就地写回（转回原始 dtype）
    tl.store(QK_ptr + base + half_offsets, new_first.to(q_first.dtype))
    tl.store(QK_ptr + base + HALF_DK + half_offsets, new_second.to(q_second.dtype))


def fused_apply_rotary_emb(q, k, cos, sin):
    """
    Fused RoPE 便捷接口，就地修改 Q 和 K。

    参数:
      q: (B, h, T, d_k) — query，contiguous
      k: (B, h, T, d_k) — key，contiguous
      cos: (T, d_k) — 预计算的 cos(position * freq)
      sin: (T, d_k) — 预计算的 sin(position * freq)
    返回:
      (q, k) — 就地修改后的 Q, K

    注意: 此函数就地修改 q 和 k，适用于 forward 场景（q/k 是新分配的 view）。
    """
    assert q.is_contiguous() and k.is_contiguous(), "q, k must be contiguous"
    B, h, T, d_k = q.shape
    assert d_k % 2 == 0, f"d_k must be even, got {d_k}"
    HALF_DK = d_k // 2
    # Triton tl.arange 要求 constexpr 参数为 power-of-2
    assert HALF_DK > 0 and (HALF_DK & (HALF_DK - 1)) == 0, (
        f"d_k//2 ({HALF_DK}) must be a power of 2 for Triton kernel"
    )

    # 将 (B, h, T, d_k) 视为 (B*h, T, d_k)
    q_flat = q.view(B * h, T, d_k)
    k_flat = k.view(B * h, T, d_k)

    # cos/sin 形状: (T, d_k)，前后半段相同（RotaryEmbedding 的 cat 结构）
    assert cos.shape == (T, d_k) and sin.shape == (T, d_k)
    cos = cos.contiguous()
    sin = sin.contiguous()

    stride_bh = q_flat.stride(0)
    stride_t = q_flat.stride(1)

    # Grid: 每个 program 处理一个 (batch*head, position) 对
    grid = (B * h * T,)

    # 对 Q 和 K 分别调用 kernel
    _fused_rope_kernel[grid](
        q_flat, cos, sin,
        stride_bh=stride_bh, stride_t=stride_t,
        T=T, HALF_DK=HALF_DK,
    )
    _fused_rope_kernel[grid](
        k_flat, cos, sin,
        stride_bh=stride_bh, stride_t=stride_t,
        T=T, HALF_DK=HALF_DK,
    )

    return q, k
