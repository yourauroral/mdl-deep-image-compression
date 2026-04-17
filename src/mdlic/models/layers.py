import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.checkpoint import checkpoint as torch_checkpoint

# Fused RMSNorm Triton kernel（可选）：
# 将归一化 + 缩放合并为单次 kernel launch，减少 HBM 读写。
# 若 Triton 不可用（如 CPU 环境），自动回退到下方 PyTorch 实现。
# Ref: Liger-Kernel arXiv:2410.10989 的 fused pattern（手写实现）。
try:
    from ..ops.fused_rms_norm import fused_rms_norm as _fused_rms_norm
    _USE_FUSED_RMSNORM = True
except ImportError:
    _fused_rms_norm = None
    _USE_FUSED_RMSNORM = False

# Fused SwiGLU Triton kernel（可选）：
# 将 silu(a) ⊙ b 合并为单次 kernel launch，减少 HBM 读写。
# Backward 采用 activation recomputation，不存 silu(a) 中间结果。
# 若 Triton 不可用（如 CPU 环境），自动回退到 F.silu(a) * b。
# Ref: Liger-Kernel arXiv:2410.10989 的 fused SwiGLU pattern（手写实现）。
try:
    from ..ops.fused_swiglu import fused_swiglu as _fused_swiglu
    _USE_FUSED_SWIGLU = True
except ImportError:
    _fused_swiglu = None
    _USE_FUSED_SWIGLU = False

# Flash Attention Triton kernel（可选）：
# 手写 Flash Attention v2，将 QK^T 计算、softmax、V 加权合并为一次 kernel launch，
# 避免 O(N²) 的中间注意力矩阵存储，SRAM 复杂度 O(1)（相对于 seq_len）。
# 特别适合长序列和 causal mask 场景（自回归语言模型）。
#
# 若 Triton 不可用（如 CPU 环境），自动回退到 F.scaled_dot_product_attention
# （PyTorch >= 2.0 内置 FlashAttention 或 Memory-Efficient Attention 后端）。
#
# 参考:
#   [1] Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention
#       with IO-Awareness," NeurIPS 2022, arXiv:2205.14135.
#   [2] Dao, "FlashAttention-2: Faster Attention with Better Parallelism and
#       Work Partitioning," arXiv:2307.08691, 2023.
#       - Causal backward early termination，2x-4x 加速
#   [3] Milakov & Gimelshein, "Online normalizer calculation for softmax,"
#       arXiv:1805.02867, 2018.
try:
    from ..ops.flash_attn import TritonAttention as _TritonAttention
    _USE_TRITON_ATTN = True
except ImportError:
    _TritonAttention = None
    _USE_TRITON_ATTN = False

# Fused RoPE Triton kernel（可选）：
# 将 RoPE 旋转直接应用到 Q/K，避免 rotate_half 等中间张量。
# 若 Triton 不可用，自动回退到 PyTorch apply_rotary_emb。
# Ref: Su et al., arXiv:2104.09864; Liger-Kernel arXiv:2410.10989（手写实现）。
try:
    from ..ops.fused_rope import fused_apply_rotary_emb as _fused_rope
    _USE_FUSED_ROPE = True
except ImportError:
    _fused_rope = None
    _USE_FUSED_ROPE = False

# Fused Add+RMSNorm Triton kernel（可选）：
# 将 OLMo 2 post-norm 中的 residual + RMSNorm(sublayer_out) 合并为单次 kernel launch。
# 减少一次 HBM 读写。若 Triton 不可用，自动回退到 PyTorch 分步计算。
# Ref: OLMo 2 arXiv:2501.00656 Section 3.1; Liger-Kernel arXiv:2410.10989（手写实现）。
try:
    from ..ops.fused_add_rms_norm import fused_add_rms_norm as _fused_add_rms_norm
    _USE_FUSED_ADD_RMSNORM = True
except ImportError:
    _fused_add_rms_norm = None
    _USE_FUSED_ADD_RMSNORM = False

# Fused Attention + RoPE（可选）：
# 将 RoPE 旋转与 Flash Attention 合并为单次操作，减少 Q/K 的 HBM 读写。
# 若任一依赖不可用（fused_rope 或 flash_attn），自动回退到分步调用。
# Ref: Su et al., arXiv:2104.09864; Dao, arXiv:2307.08691（手写实现）。
try:
    from ..ops.fused_attn_rope import fused_attn_rope as _fused_attn_rope
    _USE_FUSED_ATTN_ROPE = True
except ImportError:
    _fused_attn_rope = None
    _USE_FUSED_ATTN_ROPE = False


def get_fused_kernel_status() -> dict:
    """
    返回各 fused Triton kernel 的可用状态，方便训练启动时打印诊断信息。
    """
    return {
        "fused_rms_norm": _USE_FUSED_RMSNORM,
        "fused_swiglu": _USE_FUSED_SWIGLU,
        "flash_attn": _USE_TRITON_ATTN,
        "fused_rope": _USE_FUSED_ROPE,
        "fused_add_rms_norm": _USE_FUSED_ADD_RMSNORM,
        "fused_attn_rope": _USE_FUSED_ATTN_ROPE,
    }

class RotaryEmbedding(nn.Module):
  """
  Rotary Position Embedding (RoPE).

  参考:
    [1] Su et al., "RoFormer: Enhanced Transformer with Rotary Position
        Embedding," arXiv:2104.09864, 2021. 原始 RoPE，base=10000.
    [2] Meta, "LLaMA 3 Tech Report," 2024.
        将 base frequency 从 10000 提升至 500000，改善长序列位置分辨率。
    [3] OLMo 2 Tech Report, arXiv:2501.00656, 2025, Section 3.1.
        沿用 LLaMA 3 的 base=500000 设置。
    [4] CS336 "Language Models from Scratch," Stanford, Spring 2024.

  base frequency 选择：
    θ_i = base^{-2i/d_k}，i = 0,…,d_k/2-1
    base=10000（原始）在 seq_len≫1000 时高频分量旋转超过 2π，位置分辨率退化。
    base=500000 将有效序列长度上限提升约 50x，对 CIFAR-100 的
    seq_len=3072 (32×32×3) 尤为重要。
  """
  def __init__(self, dim: int, base: int = 500000):
    super().__init__()
    assert dim % 2 == 0
    # θ_i = base^{-2i/d_k}，shape: (dim//2,)  [1] Eq.(15)
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    self.register_buffer('inv_freq', inv_freq)
    # (seq_len, device) → (cos, sin)；避免每次 forward 重算 arange/outer/cat
    self._cache_key = None
    self._cache_cos = None
    self._cache_sin = None

  def forward(self, seq_len: int, device: torch.device):
    key = (seq_len, device)
    if self._cache_key != key:
      t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
      freqs = torch.outer(t, self.inv_freq)
      emb = torch.cat([freqs, freqs], dim=-1)
      self._cache_cos = emb.cos()
      self._cache_sin = emb.sin()
      self._cache_key = key
    return self._cache_cos, self._cache_sin

def rotate_half(x):
  """将向量后半段取负后与前半段拼接，实现 90° 旋转。[1] Eq.(34) 的等价实现。"""
  x1 = x[..., :x.shape[-1] // 2]
  x2 = x[..., x.shape[-1] // 2:]
  return torch.cat([-x2, x1], dim=-1)

def apply_rotary_emb(q, k, cos, sin):
  """
  将 RoPE 应用到 query 和 key。
  q, k: (batch, heads, seq_len, d_k)
  cos, sin: (seq_len, d_k)
  Ref: Su et al., arXiv:2104.09864 [1], Eq.(34).
  """
  cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, d_k)
  sin = sin.unsqueeze(0).unsqueeze(0)
  q = (q * cos) + (rotate_half(q) * sin)
  k = (k * cos) + (rotate_half(k) * sin)
  return q, k

class RMSNorm(nn.Module):
  """
  Root Mean Square Layer Normalization.

  RMSNorm(x) = w * x / RMS(x),  RMS(x) = sqrt(mean(x²) + eps)

  相比 LayerNorm 去掉了 re-centering（减均值），参数量减半、计算更快，
  在现代 LLM 中已成为标准选择。

  参考:
    [1] Zhang & Sennrich, "Root Mean Square Layer Normalization,"
        NeurIPS 2019, arXiv:1910.07467. 原始 RMSNorm。
    [2] Touvron et al., "LLaMA," arXiv:2302.13971, 2023, Section 2.
        采用 RMSNorm 替代 LayerNorm。
    [3] OLMo 2 Tech Report, arXiv:2501.00656, 2025, Section 3.1.
        同样采用 RMSNorm。

  当 Triton 可用且输入在 CUDA 上时，自动委托给 fused_rms_norm 函数
  （forward + backward 合并为单次 kernel launch），否则使用 PyTorch 实现。
  """
  def __init__(self, features: int, eps: float = 1e-10):
    super().__init__()
    self.eps = eps
    self.weight = nn.Parameter(torch.ones(features))

  def forward(self, x):
    if _USE_FUSED_RMSNORM and x.is_cuda:
      return _fused_rms_norm(x, self.weight, self.eps)
    # PyTorch fallback
    rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
    return self.weight * (x / rms)

class FeedForwardBlock(nn.Module):
  """
  SwiGLU Feed-Forward Block.

  替换原 ReLU 双线性 FFN 为 SwiGLU 三线性门控 FFN：
      FFN(x) = W2 · (SiLU(W1·x) ⊙ W3·x)

  参考:
    [1] Shazeer, "GLU Variants Improve Transformers," arXiv:2002.05202, 2020.
        提出 SwiGLU = Swish-Gated Linear Unit，Section 2, Eq.(6).
    [2] Touvron et al., "LLaMA," arXiv:2302.13971, 2023.
        采用 SwiGLU + bias=False，Section 2.
    [3] OLMo 2 Tech Report, arXiv:2501.00656, 2025.
        同样采用 SwiGLU，FFN hidden dim = (8/3) * d_model.
    [4] CS336 "Language Models from Scratch," Stanford, Spring 2024.
        Assignment 1 参考实现。

  参数量与原 ReLU FFN 对齐：
    原 ReLU FFN:  2 × (d_model × d_ff)
    SwiGLU FFN:   3 × (d_model × d_ff_swiglu)
    令 d_ff_swiglu = (2/3) × d_ff 即可保持参数量不变。
    调用方应传入已计算好的 d_ff（即 (8/3)*d_model，取整到 64 的倍数）。

  bias=False: 现代 LLM 标准做法 [2][3]，与 weight decay 更兼容。
  """
  def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
      super().__init__()
      self.w1 = nn.Linear(d_model, d_ff, bias=False)   # gate path
      self.w3 = nn.Linear(d_model, d_ff, bias=False)   # value path
      self.w2 = nn.Linear(d_ff, d_model, bias=False)   # down-projection
      self.dropout = nn.Dropout(dropout)

  def forward(self, x):
      # SwiGLU: SiLU(W1·x) ⊙ (W3·x)，再经 W2 投影回 d_model
      # [1] Eq.(6): FFN_SwiGLU(x, W, V, W2) = (Swish1(xW) ⊙ xV) W2
      a = self.w1(x)  # gate path
      b = self.w3(x)  # value path
      if _USE_FUSED_SWIGLU and x.is_cuda:
          # Fused kernel: silu(a) ⊙ b 一次 HBM 读写
          gate = _fused_swiglu(a, b)
      else:
          # PyTorch fallback
          gate = F.silu(a) * b
      return self.dropout(self.w2(gate))


class ReLUFeedForwardBlock(nn.Module):
  """
  标准 ReLU 双线性 FFN（消融 baseline）。

  FFN(x) = W2 · ReLU(W1·x)

  用于 use_swiglu=False 时的 fallback，对比 SwiGLU 的效果。
  d_ff 使用标准 4×d_model。
  参考: Vaswani et al., "Attention Is All You Need," arXiv:1706.03762, 2017.
  """
  def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
      super().__init__()
      # ReLU FFN: d_ff = 4*d_model（标准设置）
      d_ff_relu = 4 * d_model
      self.w1 = nn.Linear(d_model, d_ff_relu, bias=False)
      self.w2 = nn.Linear(d_ff_relu, d_model, bias=False)
      self.dropout = nn.Dropout(dropout)

  def forward(self, x):
      return self.dropout(self.w2(F.relu(self.w1(x))))


class MultiHeadAttentionBlock(nn.Module):
  """
  Multi-Head Attention with RoPE, QK-Norm, and optional Triton Flash Attention.

  当 Triton 可用且输入在 CUDA 上时，自动使用手写 Flash Attention kernel [1][2]，
  否则回退到 F.scaled_dot_product_attention（PyTorch >= 2.0 内置 FlashAttention/
  Memory-Efficient Attention 后端）。

  架构特点：
    - RoPE (Rotary Position Embedding) [3]: base=500000 支持长序列
    - QK-Norm: 对 Q, K 分别应用 RMSNorm 稳定注意力分布 [4]
    - OLMo2 风格 post-norm: x = x + RMSNorm(attn(x)) [5]
    - Flash Attention: O(1) SRAM，避免 O(N²) 中间矩阵 [1][2]

  参考文献：
    [1] Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention
        with IO-Awareness," NeurIPS 2022, arXiv:2205.14135.
    [2] Dao, "FlashAttention-2: Faster Attention with Better Parallelism and
        Work Partitioning," arXiv:2307.08691, 2023.
        - Causal backward early termination
    [3] Su et al., "RoFormer: Enhanced Transformer with Rotary Position
        Embedding," arXiv:2104.09864, 2021.
        - base=500000 参考 LLaMA 3 和 OLMo 2
    [4] Dehghani et al., "Scaling Vision Transformers," arXiv:2302.05442, 2023.
        - QK-Norm 提升注意力稳定性
    [5] OLMo 2 Tech Report, arXiv:2501.00656, 2025, Section 3.1.
        - Post-norm 架构
  """
  def __init__(self, d_model: int, h: int, dropout: float,
               use_rope: bool = True, use_qk_norm: bool = True) -> None:
      super().__init__()
      self.d_model = d_model # Embedding vector size
      self.h = h # Number of heads
      self.use_rope = use_rope
      self.use_qk_norm = use_qk_norm
      # Make sure d_model is divisible by h
      assert d_model % h == 0, "d_model is not divisible by h"

      self.d_k = d_model // h # Dimension of vector seen by each head
      self.w_q = nn.Linear(d_model, d_model, bias=False) # Wq
      self.w_k = nn.Linear(d_model, d_model, bias=False) # Wk
      self.w_v = nn.Linear(d_model, d_model, bias=False) # Wv
      self.w_o = nn.Linear(d_model, d_model, bias=False) # Wo
      self.dropout = nn.Dropout(dropout)
      if use_qk_norm:
          self.q_norm = RMSNorm(self.d_k)
          self.k_norm = RMSNorm(self.d_k)
      if use_rope:
          self.rope = RotaryEmbedding(self.d_k)

  def forward(self, q, k, v, mask=None, position_ids=None, attn_mask=None):
    """
    参数:
      q, k, v: (batch, seq_len, d_model)
      mask: 可选注意力掩码
      position_ids: 可选 (seq_len,) long tensor，指定每个 token 的 RoPE 位置 ID。
                    用于子像素自回归：同一像素内的 3 个通道 token 共享相同位置 ID，
                    使 RoPE 只编码像素间的空间关系。
                    None 时使用默认 0,1,2,...,seq_len-1。
                    Ref: PixelCNN++ [Salimans 2017] — 通道间条件依赖
      attn_mask: 可选 (seq_len, seq_len) float tensor，显式注意力掩码。
                 0.0 = 允许，-inf = 屏蔽。用于 VAR block-causal attention。
                 提供时绕过 Triton Flash Attention，使用 PyTorch SDPA。
                 Ref: Tian et al., arXiv:2404.02905 — VAR block-causal mask
    """
    batch_size, seq_len, _ = q.shape

    query = self.w_q(q).view(batch_size, seq_len, self.h, self.d_k).transpose(1, 2).contiguous()
    key   = self.w_k(k).view(batch_size, seq_len, self.h, self.d_k).transpose(1, 2).contiguous()
    value = self.w_v(v).view(batch_size, seq_len, self.h, self.d_k).transpose(1, 2).contiguous()

    if self.use_qk_norm:
        query = self.q_norm(query)
        key = self.k_norm(key)

    if self.use_rope:
        # 子像素自回归模式下，使用像素级 position_ids 生成 RoPE
        # 同一像素内的通道 token 共享同一位置编码
        if position_ids is not None:
            # position_ids 由调用方构造（如 igpt 子像素 AR 的 floor-div），
            # 上界可由 tensor.size 推断，避免 .item() 引起 GPU→CPU 同步
            max_pos = int(position_ids.shape[0])
            cos_full, sin_full = self.rope(max_pos, q.device)
            cos = cos_full[position_ids]
            sin = sin_full[position_ids]
        else:
            cos, sin = self.rope(seq_len, q.device)

        # 显式 attn_mask 模式 (VAR block-causal): 绕过 Triton Flash Attention，
        # 使用 Fused/PyTorch RoPE + PyTorch SDPA（支持显式 mask）。
        # Ref: Tian et al., arXiv:2404.02905 — VAR block-causal attention
        if attn_mask is not None:
            if _USE_FUSED_ROPE and query.is_cuda:
                query, key = _fused_rope(query, key, cos, sin)
            else:
                query, key = apply_rotary_emb(query, key, cos, sin)
            attn_output = F.scaled_dot_product_attention(
                query, key, value,
                attn_mask=attn_mask,
                dropout_p=0.0,
                is_causal=False,
            )
        # Full causal attention 路径（原有逻辑不变）
        # Fused Attn+RoPE: 将 RoPE 和 Flash Attention 合并为一次操作，
        # 减少 Q/K 的 HBM 读写（RoPE 就地修改后直接被 Attn 读取）。
        elif _USE_FUSED_ATTN_ROPE and query.is_cuda:
            softmax_scale = 1.0 / math.sqrt(self.d_k)
            attn_output = _fused_attn_rope(query, key, value, cos, sin,
                                            causal=True,
                                            softmax_scale=softmax_scale)
            attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
            attn_output = self.dropout(attn_output)
            return self.w_o(attn_output)
        elif _USE_FUSED_ROPE and query.is_cuda:
            query, key = _fused_rope(query, key, cos, sin)
        else:
            query, key = apply_rotary_emb(query, key, cos, sin)

    # 选择注意力后端（仅非 attn_mask 路径到达此处）：
    # 1. Triton 手写 Flash Attention（CUDA + triton 可用时）
    # 2. F.scaled_dot_product_attention（PyTorch 内置后端）
    if attn_mask is None:
      if _USE_TRITON_ATTN and query.is_cuda:
        softmax_scale = 1.0 / math.sqrt(self.d_k)
        attn_output = _TritonAttention.apply(query, key, value, True, softmax_scale)
      else:
        attn_output = F.scaled_dot_product_attention(
          query, key, value,
          attn_mask=None,
          dropout_p=0.0,
          is_causal=True
        )

    attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
    attn_output = self.dropout(attn_output)

    return self.w_o(attn_output)

class GPTBlock(nn.Module):
  def __init__(self, d_model, h, d_ff, dropout,
               use_post_norm: bool = True, use_swiglu: bool = True,
               use_qk_norm: bool = True, use_rope: bool = True,
               activation_checkpointing: bool = False):
    super().__init__()
    self.use_post_norm = use_post_norm
    self.activation_checkpointing = activation_checkpointing
    self.norm1 = RMSNorm(d_model)
    self.norm2 = RMSNorm(d_model)
    self.attn = MultiHeadAttentionBlock(d_model, h, dropout,
                                         use_rope=use_rope,
                                         use_qk_norm=use_qk_norm)
    if use_swiglu:
        self.ff = FeedForwardBlock(d_model, d_ff, dropout)
    else:
        self.ff = ReLUFeedForwardBlock(d_model, d_ff, dropout)

  def _attn_forward(self, x, mask, position_ids=None, attn_mask=None):
    """Attention sublayer，可被 activation checkpoint 包裹。"""
    return self.attn(x, x, x, mask, position_ids=position_ids, attn_mask=attn_mask)

  def forward(self, x, mask=None, position_ids=None, attn_mask=None):
    if self.use_post_norm:
        # OLMo 2 post-norm: x = x + RMSNorm(sublayer(x))
        if self.activation_checkpointing and self.training:
            # Selective activation checkpointing：只 checkpoint attention（显存瓶颈）
            # Ref: Chen et al., "Training Deep Nets with Sublinear Memory Cost,"
            #      arXiv:1604.06174, 2016.
            attn_out = torch_checkpoint(self._attn_forward, x, mask,
                                        position_ids, attn_mask, use_reentrant=False)
        else:
            attn_out = self._attn_forward(x, mask, position_ids, attn_mask=attn_mask)
        # Fused Add+RMSNorm: residual + RMSNorm(sublayer_out) 一次 kernel
        if _USE_FUSED_ADD_RMSNORM and x.is_cuda:
            x = _fused_add_rms_norm(x, attn_out, self.norm1.weight, self.norm1.eps)
        else:
            x = x + self.norm1(attn_out)
        ff_out = self.ff(x)
        if _USE_FUSED_ADD_RMSNORM and x.is_cuda:
            x = _fused_add_rms_norm(x, ff_out, self.norm2.weight, self.norm2.eps)
        else:
            x = x + self.norm2(ff_out)
    else:
        # Pre-norm（消融 baseline）: x = x + sublayer(RMSNorm(x))
        # Ref: Xiong et al., "On Layer Normalization in the Transformer
        #      Architecture," ICML 2020, arXiv:2002.04745.
        if self.activation_checkpointing and self.training:
            attn_out = torch_checkpoint(self._attn_forward,
                                        self.norm1(x), mask,
                                        position_ids, attn_mask, use_reentrant=False)
        else:
            attn_out = self._attn_forward(self.norm1(x), mask,
                                          position_ids, attn_mask=attn_mask)
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
    return x