import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import GPTBlock, RMSNorm
from .igpt import rgb_to_ycbcr_int

try:
    from ..ops.fused_ce_zloss import fused_cross_entropy_zloss as _fused_ce_zloss
    _USE_FUSED_CE = True
except ImportError:
    _fused_ce_zloss = None
    _USE_FUSED_CE = False

try:
    from ..ops.fused_linear_ce import fused_linear_cross_entropy as _fused_linear_ce
    _USE_FUSED_LINEAR_CE = True
except ImportError:
    _fused_linear_ce = None
    _USE_FUSED_LINEAR_CE = False


class MultiScalePyramid(nn.Module):
  """
  RGB → YCbCr → avgpool 下采样到各 scale → 拼接为 pixel-first 多尺度序列。

  scales = [1, 2, 4, 8, 16, 32]，对 32×32 图像得到:
    S0: 1×1×3    =    3 tokens
    S1: 2×2×3    =   12
    S2: 4×4×3    =   48
    S3: 8×8×3    =  192
    S4: 16×16×3  =  768
    S5: 32×32×3  = 3072  (原图)
    总计: 4095 tokens, vocab=256

  每个 scale 内部 pixel-first 排列 [Y₀,Cb₀,Cr₀, Y₁,Cb₁,Cr₁, ...]，
  causal mask 自然实现通道间条件依赖 p(Cb|Y), p(Cr|Y,Cb)。

  Ref:
    [1] Tian et al., "VAR: Visual Autoregressive Modeling," NeurIPS 2024
    [2] Field, "Relations between the statistics of natural images...," 1987
        (1/f² 功率谱 — 信息集中在粗尺度)
    [3] Burt & Adelson, "The Laplacian Pyramid as a Compact Image Code," 1983
  """
  def __init__(self, image_size: int = 32, in_channels: int = 3,
               num_scales: int = 6, use_ycbcr: bool = True):
    super().__init__()
    self.image_size = image_size
    self.in_channels = in_channels
    self.num_scales = num_scales
    self.use_ycbcr = use_ycbcr
    # scales: [1, 2, 4, ..., image_size]
    self.scales = [image_size // (2 ** (num_scales - 1 - k)) for k in range(num_scales)]
    assert self.scales[-1] == image_size, "最细 scale 必须等于 image_size"
    assert self.scales[0] >= 1

    tokens_per_scale = [s * s * in_channels for s in self.scales]
    self.tokens_per_scale = tokens_per_scale
    self.total_tokens = sum(tokens_per_scale)

    scale_ids = torch.cat([
        torch.full((n,), k, dtype=torch.long)
        for k, n in enumerate(tokens_per_scale)
    ])
    self.register_buffer('scale_ids', scale_ids, persistent=False)

    # Per-scale position ids: 每个 scale 从 0 重新计数（按像素而非 token）
    # 同一像素内的 3 个通道 token 共享同一 position_id
    pos_list = []
    for k, s in enumerate(self.scales):
        n_pixels = s * s
        # 每像素 C 个 token，共享同一位置
        pos = torch.arange(n_pixels, dtype=torch.long).repeat_interleave(in_channels)
        pos_list.append(pos)
    position_ids = torch.cat(pos_list)
    self.register_buffer('position_ids', position_ids, persistent=False)

    # Per-scale channel ids: 每个 scale 内 [0,1,2, 0,1,2, ...]
    ch_list = []
    for k, s in enumerate(self.scales):
        n_pixels = s * s
        ch = torch.arange(in_channels, dtype=torch.long).repeat(n_pixels)
        ch_list.append(ch)
    channel_ids = torch.cat(ch_list)
    self.register_buffer('channel_ids', channel_ids, persistent=False)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    x: (B, 3, H, W) float [0,1]
    return: tokens (B, total_tokens) long [0,255]
    """
    B = x.size(0)
    x = x.clamp(0, 1)
    if self.use_ycbcr:
        # YCbCr 在 [0,1] 域做 avgpool，然后转 int [0,255]
        # 用 float 图做下采样，避免 int avgpool 引入累积量化噪声
        pass

    parts = []
    for s in self.scales:
        if s == self.image_size:
            x_s = x
        else:
            x_s = F.adaptive_avg_pool2d(x, s)  # (B, 3, s, s) float [0,1]
        # YCbCr 量化
        if self.use_ycbcr:
            x_int = rgb_to_ycbcr_int(x_s)  # (B, 3, s, s) long
        else:
            x_int = (x_s * 255).round().long()
        # pixel-first: (B, 3, s, s) → (B, s, s, 3) → (B, s*s*3)
        tok = x_int.permute(0, 2, 3, 1).reshape(B, -1)
        parts.append(tok)
    return torch.cat(parts, dim=1)  # (B, total_tokens)


class MSPA(nn.Module):
  """
  Multi-Scale Pixel Autoregression (MSPA).

  VAR 的 next-scale prediction 思想 + 像素空间无损（vocab=256）。
  标准 causal mask 下做 NTP，联合 NLL 等价于 Σ_k CE_k·N_k。

  BPP = total_loss · (T-1) / ln(2) / (H·W·C)
  其中 T-1 是预测 token 数，H·W·C 是原图像素位数。

  Ref:
    [1] Tian et al., "VAR," NeurIPS 2024
    [2] Equitz & Cover, "Successive Refinement of Information," IT 1991
    [3] Salimans et al., "PixelCNN++," ICLR 2017 (subpixel AR)
  """
  def __init__(
    self,
    image_size: int = 32,
    in_channels: int = 3,
    vocab_size: int = 256,
    d_model: int = 256,
    N: int = 8,
    h: int = 8,
    d_ff: int = 682,
    dropout: float = 0.0,
    num_scales: int = 6,
    use_ycbcr: bool = True,
    use_rope: bool = True,
    use_post_norm: bool = True,
    use_swiglu: bool = True,
    use_qk_norm: bool = True,
    use_depth_scaled_init: bool = True,
    use_zloss: bool = True,
    activation_checkpointing: bool = False,
  ):
    super().__init__()
    self.image_size = image_size
    self.in_channels = in_channels
    self.vocab_size = vocab_size
    self.d_model = d_model
    self.N_layers = N
    self.num_scales = num_scales
    self.use_rope = use_rope
    self.use_post_norm = use_post_norm
    self.use_zloss = use_zloss
    self.use_depth_scaled_init = use_depth_scaled_init

    self.pyramid = MultiScalePyramid(
        image_size=image_size,
        in_channels=in_channels,
        num_scales=num_scales,
        use_ycbcr=use_ycbcr,
    )
    self.seq_len = self.pyramid.total_tokens  # e.g. 4095
    self.pixel_bits = image_size * image_size * in_channels  # 原图像素位数

    self.token_embed = nn.Embedding(vocab_size, d_model)
    self.scale_embed = nn.Embedding(num_scales, d_model)
    self.channel_embed = nn.Embedding(in_channels, d_model)

    if not use_rope:
        self.pos_embed = nn.Embedding(self.seq_len - 1, d_model)

    self.blocks = nn.ModuleList([
        GPTBlock(d_model, h, d_ff, dropout,
                 use_post_norm=use_post_norm,
                 use_swiglu=use_swiglu,
                 use_qk_norm=use_qk_norm,
                 use_rope=use_rope,
                 activation_checkpointing=activation_checkpointing)
        for _ in range(N)
    ])

    if not use_post_norm:
        self.final_norm = RMSNorm(d_model)

    self.head = nn.Linear(d_model, vocab_size, bias=False)
    self.head.weight = self.token_embed.weight  # weight tying

    self._init_weights()

  def _init_weights(self):
    N = self.N_layers
    for name, module in self.named_modules():
      if isinstance(module, nn.Linear):
        if name.endswith('head'):
          continue
        if self.use_depth_scaled_init and any(name.endswith(s) for s in ('w_o', 'w2')):
          std = 0.02 / math.sqrt(2 * N)
        else:
          std = 0.02
        nn.init.normal_(module.weight, mean=0.0, std=std)
        if module.bias is not None:
          nn.init.zeros_(module.bias)
      elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)

  def _init_weights_mup(self, base_width: int):
    N = self.N_layers
    for name, module in self.named_modules():
      if isinstance(module, nn.Linear):
        if name.endswith('head'):
          continue
        fan_in = module.weight.shape[1]
        if any(name.endswith(s) for s in ('w_o', 'w2')):
          std = (1.0 / math.sqrt(fan_in)) / math.sqrt(2 * N)
        else:
          std = 1.0 / math.sqrt(fan_in)
        nn.init.normal_(module.weight, mean=0.0, std=std)
        if module.bias is not None:
          nn.init.zeros_(module.bias)
      elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=1.0)

  def _embed_inputs(self, input_tokens: torch.Tensor):
    """
    input_tokens: (B, T-1) long
    return: hidden (B, T-1, d_model), position_ids (T-1,)
    """
    T_in = input_tokens.shape[1]
    device = input_tokens.device

    hidden = self.token_embed(input_tokens)
    hidden = hidden + self.scale_embed(self.pyramid.scale_ids[:T_in]).unsqueeze(0)
    hidden = hidden + self.channel_embed(self.pyramid.channel_ids[:T_in]).unsqueeze(0)

    if self.use_rope:
        position_ids = self.pyramid.position_ids[:T_in].to(device)
    else:
        position_ids = None
        hidden = hidden + self.pos_embed(torch.arange(T_in, device=device))
    return hidden, position_ids

  def forward(self, x, z_loss_weight: float = 1e-4):
    z_loss_weight = float(z_loss_weight)
    B = x.size(0)

    tokens = self.pyramid(x)  # (B, T) long
    input_tokens = tokens[:, :-1]
    target_tokens = tokens[:, 1:]
    T_in = input_tokens.shape[1]

    hidden, position_ids = self._embed_inputs(input_tokens)

    for block in self.blocks:
      hidden = block(hidden, mask=None, position_ids=position_ids)

    if not self.use_post_norm:
      hidden = self.final_norm(hidden)

    z_w = z_loss_weight if self.use_zloss else 0.0

    # 小 vocab (V < 1024) 下 fused_linear_ce 三层嵌套循环开销远大于 fusion 收益
    _use_fused_linear = _USE_FUSED_LINEAR_CE and self.vocab_size >= 1024
    if _use_fused_linear and hidden.is_cuda and z_w > 0:
        ce_loss, z_loss = _fused_linear_ce(
            hidden.reshape(-1, self.d_model),
            self.head.weight,
            target_tokens.reshape(-1),
            z_loss_weight=z_w,
        )
        loss = ce_loss + z_w * z_loss
        logits = None
    else:
        logits = self.head(hidden).float()
        if _USE_FUSED_CE and logits.is_cuda and z_w > 0:
            ce_loss, z_loss = _fused_ce_zloss(
                logits.reshape(-1, self.vocab_size),
                target_tokens.reshape(-1),
                z_loss_weight=z_w,
            )
            loss = ce_loss + z_w * z_loss
        else:
            ce_loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                target_tokens.reshape(-1),
                reduction="mean",
            )
            if self.use_zloss:
                log_z = torch.logsumexp(logits, dim=-1)
                z_loss = (log_z ** 2).mean()
                loss = ce_loss + z_w * z_loss
            else:
                loss = ce_loss

    # BPP 归一化：预测 T_in 个 token，但原图只有 H·W·C 个像素位
    # mean_ce * T_in / ln(2) 是联合 NLL 的 bit 数，再除以像素位
    bpp = (ce_loss.detach() * T_in / math.log(2.0)) / float(self.pixel_bits)

    return {
        "loss": loss,
        "ce_loss": ce_loss,
        "logits": logits,
        "bpp": bpp,
    }

  @torch.no_grad()
  def encode(self, x, max_layer: int = None):
    """Linear probe 接口：返回每层 block 输出 hidden。"""
    tokens = self.pyramid(x)
    input_tokens = tokens[:, :-1]
    hidden, position_ids = self._embed_inputs(input_tokens)

    if max_layer is None:
        max_layer = len(self.blocks) - 1
    outputs = []
    for i, block in enumerate(self.blocks):
        hidden = block(hidden, mask=None, position_ids=position_ids)
        outputs.append(hidden)
        if i >= max_layer:
            break
    return outputs
