import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import math 

from .layers import LayerNormalization, GPTBlock, RMSNorm

def generate_causal_mask(seq_len, device):
  mask = torch.tril(torch.ones(seq_len, seq_len, device=device)) 
  return mask.unsqueeze(0).unsqueeze(0) 

class IGPT(nn.Module):
  def __init__(
    self,
    image_size=32,
    in_channels=3,
    vocab_size=256,
    d_model=256,
    N=4,
    h=4,
    d_ff=1024,
    dropout=0.1
  ):
    super().__init__()
    self.seq_len = image_size * image_size * in_channels
    self.vocab_size = vocab_size 
    self.token_embed = nn.Embedding(vocab_size, d_model) 

    self.blocks = nn.ModuleList([
      GPTBlock(d_model, h, d_ff, dropout)
      for _ in range(N) 
    ])

    self.norm = RMSNorm(d_model) 
    self.head = nn.Linear(d_model, vocab_size) 

    self._init_weights()
  
  def _init_weights(self):
    for module in self.modules():
      if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
          nn.init.zeros_(module.bias)
      elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02) 

  def forward(self, x):
    x = x.clamp(0, 1)
    B = x.size(0) 

    x = (x * 255).long() 
    x = x.reshape(B, -1) 

    input_tokens = x[:, :-1] 
    target_tokens = x[:, 1:]

    x = self.token_embed(input_tokens)

    # mask = generate_causal_mask(input_tokens.size(1), x.device) 

    for block in self.blocks:
      x = block(x, mask=None) 
    
    x = self.norm(x) 
    logits = self.head(x).float() 

    ce_loss = F.cross_entropy(
      logits.reshape(-1, self.vocab_size),
      target_tokens.reshape(-1),
      reduction="mean"  
    )

    log_z = torch.logsumexp(logits, dim=-1) 
    z_loss = (log_z ** 2).mean()
    z_loss_weight = 1e-4

    loss = ce_loss + z_loss_weight * z_loss

    return {
      "loss": loss,
      "logits": logits
    }