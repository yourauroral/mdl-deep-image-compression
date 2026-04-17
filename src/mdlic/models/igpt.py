import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .layers import GPTBlock, RMSNorm

# Fused CE + z-loss Triton kernel（可选）：
# 将 softmax、cross-entropy、z-loss 合并为一次 kernel launch，
# 避免 O(V) 的中间 softmax 矩阵存储。
# 若 Triton 不可用，自动回退到 PyTorch F.cross_entropy + logsumexp。
# Ref: Liger-Kernel arXiv:2410.10989 的 fused CE pattern（手写实现）。
try:
    from ..ops.fused_ce_zloss import fused_cross_entropy_zloss as _fused_ce_zloss
    _USE_FUSED_CE = True
except ImportError:
    _fused_ce_zloss = None
    _USE_FUSED_CE = False

# Fused Linear + CE + z-loss Triton kernel（可选）：
# 将 output head 线性投影与 CE loss 合并，避免实例化完整 (B*T, V) logits 张量。
# 适合 Weight Tying 场景（W = token_embed.weight）。
# Ref: Liger-Kernel arXiv:2410.10989 的 Fused Linear CE pattern（手写实现）。
try:
    from ..ops.fused_linear_ce import fused_linear_cross_entropy as _fused_linear_ce
    _USE_FUSED_LINEAR_CE = True
except ImportError:
    _fused_linear_ce = None
    _USE_FUSED_LINEAR_CE = False

def rgb_to_ycbcr_int(x: torch.Tensor) -> torch.Tensor:
  """
  将 RGB 图像转换为 YCbCr 色彩空间后量化到 [0,255] 整数。

  参考:
    [1] ITU-R BT.601 标准，定义了 YCbCr 从 RGB 的转换系数。
        Y  = 0.299·R + 0.587·G + 0.114·B
        Cb = -0.168736·R - 0.331264·G + 0.5·B + 0.5
        Cr = 0.5·R - 0.418688·G - 0.081312·B + 0.5
    [2] Wallace, "The JPEG Still Picture Compression Standard,"
        IEEE Trans. Consumer Electronics, 1992.
        JPEG 采用 YCbCr 是因为人眼对亮度(Y)敏感、对色度(Cb,Cr)不敏感，
        Y 通道信息量集中、熵低，有利于自回归压缩。

  参数:
    x: (B, 3, H, W) float tensor，值域 [0, 1]，通道顺序 RGB
  返回:
    (B, 3, H, W) long tensor，值域 [0, 255]，通道顺序 YCbCr
  """
  r, g, b = x[:, 0], x[:, 1], x[:, 2]
  y  =  0.299    * r + 0.587    * g + 0.114    * b
  cb = -0.168736 * r - 0.331264 * g + 0.5      * b + 0.5
  cr =  0.5      * r - 0.418688 * g - 0.081312 * b + 0.5
  ycbcr = torch.stack([y, cb, cr], dim=1).clamp(0.0, 1.0)
  return (ycbcr * 255).round().long()

class IGPT(nn.Module):
  """
  Image GPT 自回归压缩模型。

  标准 next-token prediction (NTP)，将图像展平为 token 序列，
  建模 p(x_t | x_{<t})，CE loss 直接对应 Shannon 最优编码长度。
  """
  def __init__(
    self,
    image_size=32,
    in_channels=3,
    vocab_size=256,
    d_model=256,
    N=4,
    h=4,
    d_ff=1024,
    dropout=0.1,
    # 消融开关
    use_ycbcr: bool = True,
    use_rope: bool = True,
    use_post_norm: bool = True,
    use_swiglu: bool = True,
    use_qk_norm: bool = True,
    use_depth_scaled_init: bool = True,
    use_zloss: bool = True,
    activation_checkpointing: bool = False,
    # 子像素自回归 (Sub-pixel Autoregression):
    # 将序列从 channel-first [Y_all, Cb_all, Cr_all] 改为 pixel-first
    # [Y0,Cb0,Cr0, Y1,Cb1,Cr1, ...]，使同一像素内的通道能相互条件化:
    #   p(Cb_i | Y_i, context)  和  p(Cr_i | Y_i, Cb_i, context)
    # Ref: Salimans et al., "PixelCNN++," ICLR 2017 — 通道间条件依赖
    use_subpixel_ar: bool = False,
  ):
    super().__init__()
    self.seq_len = image_size * image_size * in_channels
    self.in_channels = in_channels
    self.image_size = image_size
    self.vocab_size = vocab_size
    self.d_model = d_model
    self.N_layers = N
    self.use_ycbcr = use_ycbcr
    self.use_rope = use_rope
    self.use_zloss = use_zloss
    self.use_depth_scaled_init = use_depth_scaled_init
    self.use_subpixel_ar = use_subpixel_ar
    self.token_embed = nn.Embedding(vocab_size, d_model)

    # Channel embedding: 子像素自回归模式下，为每个通道位置（0=Y/R, 1=Cb/G, 2=Cr/B）
    # 添加可学习的通道嵌入，帮助模型区分同一像素内的不同通道 token。
    # Ref: van den Oord et al., NeurIPS 2016 — 通道条件化需要通道标识
    if use_subpixel_ar:
        self.channel_embed = nn.Embedding(in_channels, d_model)

    # Learned positional embedding fallback（use_rope=False 时）
    # Ref: Radford et al., "GPT-2," 2019 — learned absolute PE
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

    # Final RMSNorm: 仅在 pre-norm 模式下需要。
    # Post-norm 模式中每个 sublayer 输出已经过 RMSNorm，最后一个 block 输出已归一化。
    # Ref: OLMo 2 arXiv:2501.00656 Section 3.1 — post-norm 不需要 final norm。
    self.use_post_norm = use_post_norm
    if not use_post_norm:
        self.final_norm = RMSNorm(d_model)

    # Output head + Weight Tying
    # embedding 和 output head 共享权重矩阵，减少参数量。
    # Ref: Press & Wolf, "Using the Output Embedding to Improve Language Models," EACL 2017.
    self.head = nn.Linear(d_model, vocab_size, bias=False)
    self.head.weight = self.token_embed.weight

    self._init_weights()

  def _init_weights(self):
    """
    权重初始化：基础 std=0.02，残差通路输出投影用深度缩放（可选）。

    参考:
      [1] Radford et al., "GPT-2," 2019 — 1/√(2·N) 深度缩放。
      [2] OLMo 2 arXiv:2501.00656 Section 3.2。
    """
    N = self.N_layers
    for name, module in self.named_modules():
      if isinstance(module, nn.Linear):
        # Weight tying: head.weight is token_embed.weight，跳过避免重复初始化
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
    """
    muP (Maximal Update Parameterization) 初始化。

    muP 通过 fan_in 缩放使不同宽度的模型在相同超参数下表现一致。
    Ref: Yang et al., arXiv:2203.03466, 2022, Table 8.
    """
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

  def forward(self, x, z_loss_weight: float = 1e-4):
    """
    参数:
      x:             (B, C, H, W) float [0,1]
      z_loss_weight: z-loss 权重，默认 1e-4
                     Ref: PaLM arXiv:2204.02311; OLMo 2 arXiv:2501.00656
    """
    z_loss_weight = float(z_loss_weight)
    x = x.clamp(0, 1)
    B = x.size(0)

    if self.use_ycbcr:
        x = rgb_to_ycbcr_int(x)   # (B, 3, H, W) long [0,255]
    else:
        x = (x * 255).round().long()

    if self.use_subpixel_ar:
        # 子像素自回归: pixel-first 排列
        # (B, C, H, W) → (B, H, W, C) → (B, H*W*C)
        # 序列: [Y0, Cb0, Cr0, Y1, Cb1, Cr1, ...]
        # Ref: Salimans et al., "PixelCNN++," ICLR 2017
        x = x.permute(0, 2, 3, 1).reshape(B, -1)
    else:
        x = x.reshape(B, -1)

    # NTP：输入 x[0..T-1]，预测 x[1..T]
    input_tokens  = x[:, :-1]
    target_tokens = x[:, 1:]

    hidden = self.token_embed(input_tokens)

    # 子像素自回归: 添加 channel embedding + 像素级 RoPE position_ids
    position_ids = None
    if self.use_subpixel_ar:
        T = input_tokens.shape[1]
        C = self.in_channels
        channel_indices = torch.arange(T, device=input_tokens.device) % C
        hidden = hidden + self.channel_embed(channel_indices).unsqueeze(0)
        # 同一像素内的 C 个 token 共享同一位置 ID
        position_ids = torch.arange(T, device=input_tokens.device) // C

    # Learned positional embedding（use_rope=False 时）
    if not self.use_rope:
        T = input_tokens.shape[1]
        positions = torch.arange(T, device=input_tokens.device)
        hidden = hidden + self.pos_embed(positions)

    for block in self.blocks:
      hidden = block(hidden, mask=None, position_ids=position_ids)

    # Final norm: 仅 pre-norm 模式需要
    if not self.use_post_norm:
        hidden = self.final_norm(hidden)

    # --- 主 loss: NTP cross-entropy (+ 可选 fused z-loss) ---
    z_w = z_loss_weight if self.use_zloss else 0.0

    # Fused Linear + CE + z-loss: 将 output head 投影和 CE loss 合并，
    # 避免实例化完整 (B*T, V) logits 张量。
    # Ref: Liger-Kernel arXiv:2410.10989（手写 Fused Linear CE pattern）
    if _USE_FUSED_LINEAR_CE and hidden.is_cuda and z_w > 0:
        ce_loss, z_loss = _fused_linear_ce(
            hidden.reshape(-1, self.d_model),
            self.head.weight,
            target_tokens.reshape(-1),
            z_loss_weight=z_w,
        )
        loss = ce_loss + z_w * z_loss
        logits = None  # fused path 不实例化 logits
    else:
        logits = self.head(hidden).float()

        if _USE_FUSED_CE and logits.is_cuda and z_w > 0:
            # Fused CE + z-loss: 一次 kernel launch 完成 softmax → CE → z-loss
            ce_loss, z_loss = _fused_ce_zloss(
                logits.reshape(-1, self.vocab_size),
                target_tokens.reshape(-1),
                z_loss_weight=z_w,
            )
            loss = ce_loss + z_w * z_loss
        else:
            # PyTorch fallback
            ce_loss = F.cross_entropy(
              logits.reshape(-1, self.vocab_size),
              target_tokens.reshape(-1),
              reduction="mean"
            )
            if self.use_zloss:
                log_z = torch.logsumexp(logits, dim=-1)
                z_loss = (log_z ** 2).mean()
                loss = ce_loss + z_w * z_loss
            else:
                loss = ce_loss

    return {
      "loss": loss,
      "ce_loss": ce_loss,
      "logits": logits
    }

  @torch.no_grad()
  def encode(self, x, max_layer: int = None):
    """
    仅走 embed + blocks 到 max_layer，不计算 head / loss，用于 linear probe 等
    表征提取场景，避免 fused-linear-CE 的额外开销。
    返回: list[Tensor(B, T, d_model)]，长度为 max_layer+1（含 embedding 前的输出）。
    """
    x = x.clamp(0, 1)
    B = x.size(0)
    x = rgb_to_ycbcr_int(x) if self.use_ycbcr else (x * 255).round().long()
    if self.use_subpixel_ar:
      x = x.permute(0, 2, 3, 1).reshape(B, -1)
    else:
      x = x.reshape(B, -1)
    input_tokens = x[:, :-1]
    hidden = self.token_embed(input_tokens)

    position_ids = None
    if self.use_subpixel_ar:
      T = input_tokens.shape[1]
      C = self.in_channels
      channel_indices = torch.arange(T, device=input_tokens.device) % C
      hidden = hidden + self.channel_embed(channel_indices).unsqueeze(0)
      position_ids = torch.arange(T, device=input_tokens.device) // C
    if not self.use_rope:
      T = input_tokens.shape[1]
      positions = torch.arange(T, device=input_tokens.device)
      hidden = hidden + self.pos_embed(positions)

    if max_layer is None:
      max_layer = len(self.blocks) - 1
    outputs = []
    for i, block in enumerate(self.blocks):
      hidden = block(hidden, mask=None, position_ids=position_ids)
      outputs.append(hidden)
      if i >= max_layer:
        break
    return outputs
