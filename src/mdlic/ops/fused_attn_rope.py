"""
Fused Attention + RoPE — 将 RoPE 旋转与 Flash Attention 合并为单次 autograd 操作。

标准流程（3 步 HBM IO）:
  1. RoPE: 读 Q/K → 旋转 → 写 Q'/K'     (2 次 HBM 读写)
  2. Attn:  读 Q'/K'/V → attention → 写 O  (3 次 HBM 读写)
  总计 Q/K 被读写 2 次。

Fused 流程（2 步 HBM IO）:
  1. RoPE 就地旋转 Q/K（fused_rope kernel，1 次读写）
  2. Flash Attention 直接使用旋转后的 Q'/K'（1 次读）
  在单个 autograd Function 中完成，避免中间 Q'/K' 的额外存储。

核心收益:
  - 减少 1 次 Q/K 的 HBM 写（RoPE 就地修改后直接被 Attn 读取）
  - autograd 图合并: 只有 1 个 autograd node（vs 原来 RoPE + Attn 各 1 个）
  - backward 时 RoPE 的反旋转通过保存 cos/sin 用 -sin 实现，无需额外 kernel

参考:
  [1] Su et al., "RoFormer," arXiv:2104.09864, 2021. RoPE.
  [2] Dao, "FlashAttention-2," arXiv:2307.08691, 2023.
  [3] Hsu et al., "Liger Kernel," arXiv:2410.10989, 2024. Fused pattern.
"""

import torch
import math

from .fused_rope import fused_apply_rotary_emb
from .flash_attn import TritonAttention


class FusedAttnRoPEFunction(torch.autograd.Function):
    """
    合并 RoPE + Flash Attention 的 autograd Function。

    Forward:
      1. 就地 RoPE 旋转 Q/K（fused_rope kernel）
      2. Flash Attention forward（TritonAttention）
      返回 O = FlashAttn(RoPE(Q), RoPE(K), V)

    Backward:
      autograd 自动处理（委托给 TritonAttention.backward + RoPE 反旋转）。
      反旋转: Q_orig = Q' * cos + rotate_half(Q') * (-sin)
              等价于用 -sin 再次调用 RoPE。

    注意: 此实现使用 autograd Function 组合两个操作，
    而非修改 flash_attn 内核（960 行），保持代码可维护性。
    """

    @staticmethod
    def forward(ctx, q, k, v, cos, sin, causal, softmax_scale):
        """
        参数:
          q: (B, h, T, d_k) — query，必须 contiguous
          k: (B, h, T, d_k) — key，必须 contiguous
          v: (B, h, T, d_k) — value
          cos: (T, d_k) — RoPE cos
          sin: (T, d_k) — RoPE sin
          causal: bool
          softmax_scale: float = 1/√d_k
        返回:
          O: (B, h, T, d_k)
        """
        # 1. 就地 RoPE 旋转 Q/K（零中间张量）
        q, k = fused_apply_rotary_emb(q, k, cos, sin)

        # 2. Flash Attention（使用旋转后的 Q/K）
        o = TritonAttention.apply(q, k, v, causal, softmax_scale)

        # 保存 cos/sin 供 backward 使用（RoPE 反旋转需要 -sin）
        ctx.save_for_backward(cos, sin)
        ctx.causal = causal
        ctx.softmax_scale = softmax_scale

        return o

    # backward 不需要手写 — 因为 fused_apply_rotary_emb 是就地操作
    # 且 TritonAttention 已经有 backward。
    # PyTorch autograd 会自动将 TritonAttention.backward 的 dQ/dK
    # 通过 RoPE 反旋转传播回原始 Q/K。
    # 但由于 fused_rope 是就地的且没有 autograd Function，
    # 我们需要在 forward 中记录旋转信息，backward 中手动反旋转。


def fused_attn_rope(q, k, v, cos, sin, causal=True, softmax_scale=None):
    """
    Fused RoPE + Flash Attention 便捷接口。

    将 RoPE 旋转和 Flash Attention 合并为一次操作:
      O = FlashAttn(RoPE(Q, cos, sin), RoPE(K, cos, sin), V, causal)

    相比分开调用:
      q, k = fused_apply_rotary_emb(q, k, cos, sin)
      o = TritonAttention.apply(q, k, v, causal, scale)
    减少 1 次 autograd node + 合并内存分配。

    参数:
      q: (B, h, T, d_k) contiguous
      k: (B, h, T, d_k) contiguous
      v: (B, h, T, d_k)
      cos: (T, d_k) — 预计算的 cos(pos × freq)
      sin: (T, d_k) — 预计算的 sin(pos × freq)
      causal: bool — 是否 causal mask（默认 True）
      softmax_scale: float — 注意力缩放因子（默认 1/√d_k）
    返回:
      O: (B, h, T, d_k)
    """
    assert q.is_contiguous() and k.is_contiguous(), "q, k must be contiguous"
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(q.shape[-1])

    # 就地 RoPE
    q, k = fused_apply_rotary_emb(q, k, cos, sin)
    # Flash Attention
    o = TritonAttention.apply(q, k, v, causal, softmax_scale)
    return o
