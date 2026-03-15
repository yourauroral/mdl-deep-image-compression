import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import math 

class RotaryEmbedding(nn.Module):
  def __init__(self, dim: int):
    super().__init__()
    # dim is d_k (head dimension), must be even
    assert dim % 2 == 0
    # precompute the inverse frequencies 
    # shape: (dim // 2, ) 
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim)) 
    self.register_buffer('inv_freq', inv_freq)
  
  def forward(self, seq_len: int, device: torch.device):
    #positions: (seq_len, ) 
    t = torch.arange(seq_len, device=device).float()
    # outer product: (seq_len , dim // 2) 
    freqs = torch.outer(t, self.inv_freq)
    # concat to (seq_len, dim) 
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()

def rotate_half(x):
  # x:(..., dim) 
  # split into two halves and rotate 
  x1 = x[..., :x.shape[-1] // 2] 
  x2 = x[..., x.shape[-1] // 2 :] 
  return torch.cat([-x2, x1], dim=-1)

def apply_rotary_emb(q, k, cos, sin):
  # q, k: (batch, heads, seq_len, d_k) 
  # cos, sin: (seq_len, d_k) 
  cos = cos.unsqueeze(0).unsqueeze(0) # (1, 1, seq_len, d_k) 
  sin = sin.unsqueeze(0).unsqueeze(0) 

  q = (q * cos) + (rotate_half(q) * sin) 
  k = (k * cos) + (rotate_half(k) * sin) 
  return q, k

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
  def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
      super().__init__()
      self.linear_1 = nn.Linear(d_model, d_ff) # w1 and b1
      self.dropout = nn.Dropout(dropout)
      self.linear_2 = nn.Linear(d_ff, d_model) # w2 and b2

  def forward(self, x):
      # (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
      return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


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

from .flash_attn import TritonAttention
class MultiHeadAttentionBlock(nn.Module):
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
      self.rope = RotaryEmbedding(self.d_k) 

  # @staticmethod
  # def attention(query, key, value, mask, dropout: nn.Dropout):
  #     d_k = query.shape[-1]
  #     # Just apply the formula from the paper
  #     # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
  #     attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
  #     if mask is not None:
  #         # Write a very low value (indicating -inf) to the positions where mask == 0
  #         min_val = torch.finfo(query.dtype).min 
  #         attention_scores.masked_fill_(mask == 0, min_val) 
  #     attention_scores = attention_scores.softmax(dim=-1) # (batch, h, seq_len, seq_len) # Apply softmax
  #     if dropout is not None:
  #         attention_scores = dropout(attention_scores)
  #     # (batch, h, seq_len, seq_len) --> (batch, h, seq_len, d_k)
  #     # return attention scores which can be used for visualization
  #     return (attention_scores @ value), attention_scores

  # def forward(self, q, k, v, mask):
  #     query = self.w_q(q) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
  #     key = self.w_k(k) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
  #     value = self.w_v(v) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)

  #     # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
  #     query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
  #     key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
  #     value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

  #     # Calculate attention
  #     x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
      
  #     # Combine all the heads together
  #     # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
  #     x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

  #     # Multiply by Wo
  #     # (batch, seq_len, d_model) --> (batch, seq_len, d_model)  
  #     return self.w_o(x)
  def forward(self, q, k, v, mask=None):
    batch_size, seq_len, _ = q.shape
    
    query = self.w_q(q).view(batch_size, seq_len, self.h, self.d_k).transpose(1, 2).contiguous()
    key   = self.w_k(k).view(batch_size, seq_len, self.h, self.d_k).transpose(1, 2).contiguous()
    value = self.w_v(v).view(batch_size, seq_len, self.h, self.d_k).transpose(1, 2).contiguous()

    cos, sin = self.rope(seq_len, q.device) 
    query, key = apply_rotary_emb(query, key, cos, sin) 
    
    causal = True 
    softmax_scale = 1.0 / (self.d_k ** 0.5)
    
    attn_output = TritonAttention.apply(query, key, value, causal, softmax_scale) 
    
    # 合并多头: (batch, h, seq_len, d_k) -> (batch, seq_len, h, d_k) -> (batch, seq_len, d_model)    
    attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
    attn_output = self.dropout(attn_output)

    return self.w_o(attn_output)

class GPTBlock(nn.Module):
  def __init__(self, d_model, h, d_ff, dropout):
    super().__init__()
    self.norm1 = LayerNormalization(d_model)
    self.norm2 = LayerNormalization(d_model)
    self.attn = MultiHeadAttentionBlock(d_model, h, dropout) 
    self.ff = FeedForwardBlock(d_model, d_ff, dropout)
  
  def forward(self, x, mask=None):
    residual = x 
    x = self.norm1(x) 
    x = self.attn(x, x, x, mask) 
    x = residual + x

    residual = x 
    x = self.norm2(x)
    x = self.ff(x) 
    x = residual + x 
    return x 