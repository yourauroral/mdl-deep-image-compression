import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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
    from .flash_attn import TritonAttention as _TritonAttention
    _USE_TRITON_ATTN = True
except ImportError:
    _TritonAttention = None
    _USE_TRITON_ATTN = False

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
  
  def forward(self, seq_len: int, device: torch.device):
    # positions: (seq_len,)
    t = torch.arange(seq_len, device=device).float()
    # outer product → (seq_len, dim//2)，再 cat 成 (seq_len, dim)  [1] Eq.(34)
    freqs = torch.outer(t, self.inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()

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

class LayerNormalization(nn.Module):
  def __init__(self, features: int, eps:float=10**-6) -> None:
      super().__init__()
      self.eps = eps
      self.alpha = nn.Parameter(torch.ones(features)) # alpha is a learnable parameter
      self.bias = nn.Parameter(torch.zeros(features)) # bias is a learnable parameter
  def forward(self, x):
      # x: (batch, seq_len, hidden_size)
        # Keep the dimension for broadcasting
      mean = x.mean(dim = -1, keepdim = True) # (batch, seq_len, 1)
      # Keep the dimension for broadcasting
      var = x.var(dim = -1, keepdim = True, unbiased=False) # (batch, seq_len, 1)
      # eps is to prevent dividing by zero or when var is very small
      x = (x - mean) / torch.sqrt(var + self.eps) 
      return self.alpha * x + self.bias

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
      return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


def conv(in_ch, out_ch, k=5, s=2):
  p = k // 2 
  return nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p) 

def deconv(in_ch, out_ch, k=5, s=2):
  p = k // 2
  op = s - 1 
  return nn.ConvTranspose2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, output_padding=op) 

class ResidualBlock(nn.Module):
  def __init__(self, ch:int):
    super().__init__()
    self.net = nn.Sequential(
      nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1),
    )
    self.act = nn.ReLU(inplace=True) 
  def forward(self, x):
    return self.act(x + self.net(x)) 

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
  def __init__(self, d_model: int, h: int, dropout: float) -> None:
      super().__init__()
      self.d_model = d_model # Embedding vector size
      self.h = h # Number of heads
      # Make sure d_model is divisible by h
      assert d_model % h == 0, "d_model is not divisible by h"

      self.d_k = d_model // h # Dimension of vector seen by each head
      self.w_q = nn.Linear(d_model, d_model, bias=False) # Wq
      self.w_k = nn.Linear(d_model, d_model, bias=False) # Wk
      self.w_v = nn.Linear(d_model, d_model, bias=False) # Wv
      self.w_o = nn.Linear(d_model, d_model, bias=False) # Wo
      self.dropout = nn.Dropout(dropout)
      self.q_norm = RMSNorm(self.d_k)
      self.k_norm = RMSNorm(self.d_k)
      self.rope = RotaryEmbedding(self.d_k)

  def forward(self, q, k, v, mask=None):
    batch_size, seq_len, _ = q.shape

    query = self.w_q(q).view(batch_size, seq_len, self.h, self.d_k).transpose(1, 2).contiguous()
    key   = self.w_k(k).view(batch_size, seq_len, self.h, self.d_k).transpose(1, 2).contiguous()
    value = self.w_v(v).view(batch_size, seq_len, self.h, self.d_k).transpose(1, 2).contiguous()

    query = self.q_norm(query)
    key = self.k_norm(key)

    cos, sin = self.rope(seq_len, q.device)
    query, key = apply_rotary_emb(query, key, cos, sin)

    # 选择注意力后端：
    # 1. Triton 手写 Flash Attention（CUDA + triton 可用时）
    # 2. F.scaled_dot_product_attention（PyTorch 内置后端）
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

    # 合并多头: (batch, h, seq_len, d_k) -> (batch, seq_len, h, d_k) -> (batch, seq_len, d_model)
    attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
    attn_output = self.dropout(attn_output)

    return self.w_o(attn_output)

class GPTBlock(nn.Module):
  def __init__(self, d_model, h, d_ff, dropout):
    super().__init__()
    self.norm1 = RMSNorm(d_model)
    self.norm2 = RMSNorm(d_model)
    self.attn = MultiHeadAttentionBlock(d_model, h, dropout) 
    self.ff = FeedForwardBlock(d_model, d_ff, dropout)
  
  def forward(self, x, mask=None):
    # OLMo 2 post-norm: x = x + RMSNorm(sublayer(x))
    x = x + self.norm1(self.attn(x, x, x, mask))
    x = x + self.norm2(self.ff(x))
    return x