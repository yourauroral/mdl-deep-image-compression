"""
RoPE-then-FlashAttention pipeline — 两个 Triton kernel 的便捷包装。

实现方式:
  1. fused_apply_rotary_emb(q, k, cos, sin)  → out-of-place 旋转后的 q', k'
  2. TritonAttention.apply(q', k', v, ...)    → flash attention 输出 O

收益（相对手写 PyTorch 的 attention + apply_rotary_emb 路径）:
  - RoPE 的 cos/sin 乘加 + half-permute 在 fused_rope kernel 内完成，
    省去 PyTorch 路径上额外的 chunk/cat 中间张量
  - Attention 直接走 flash_attn kernel，softmax + matmul 不物化 attention scores

注:
  - autograd 图上是两个 Function node（fused_rope + TritonAttention）
  - 旋转后的 q/k 会被实际分配；这不是单 kernel fusion，也不消除两步之间的 HBM 流量
  - q/k 是 out-of-place 旋转，不就地修改输入

参考:
  [1] Su et al., "RoFormer," arXiv:2104.09864, 2021. RoPE.
  [2] Dao, "FlashAttention-2," arXiv:2307.08691, 2023.
"""

import math

from .fused_rope import fused_apply_rotary_emb
from .flash_attn import TritonAttention


def rope_then_flash_attn(q, k, v, cos, sin, causal=True, softmax_scale=None):
    """
    先执行 out-of-place RoPE kernel，再执行 Flash Attention kernel。

    数学操作:
      O = FlashAttn(RoPE(Q, cos, sin), RoPE(K, cos, sin), V, causal)

    等价于:
      q, k = fused_apply_rotary_emb(q, k, cos, sin)
      o = TritonAttention.apply(q, k, v, causal, scale)

    此包装不改变 kernel launch 数、autograd node 数或中间张量分配。

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

    # RoPE（autograd-aware，out-of-place）
    q, k = fused_apply_rotary_emb(q, k, cos, sin)
    # Flash Attention
    o = TritonAttention.apply(q, k, v, causal, softmax_scale)
    return o


# 兼容旧调用；名称保留不代表这是单 kernel fusion。
fused_attn_rope = rope_then_flash_attn
