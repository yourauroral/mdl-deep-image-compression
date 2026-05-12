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

  注意：BT.601 的系数为实数，最终的 round() 是多对一映射，因此从原始 RGB 进入
  YCbCr-int 域是**有损前端**（相对原 RGB 近无损）；YCbCr-int 域内部建模/编解码
  仍严格无损。若需 RGB-bit-exact 无损，传 use_ycbcr=False。

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

  架构（毕设全部 config 共享，硬编码不可关）：RoPE base=500000、QK-Norm、
  RMSNorm Post-Norm（OLMo 2 风格）、SwiGLU FFN、Weight Tying、z-loss 正则、
  深度缩放初始化、子像素自回归（可选）。
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
    use_ycbcr: bool = True,
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
    self.use_subpixel_ar = use_subpixel_ar
    self.token_embed = nn.Embedding(vocab_size, d_model)

    # Channel embedding: 子像素自回归模式下，为每个通道位置（0=Y/R, 1=Cb/G, 2=Cr/B）
    # 添加可学习的通道嵌入，帮助模型区分同一像素内的不同通道 token。
    # Ref: van den Oord et al., NeurIPS 2016 — 通道条件化需要通道标识
    if use_subpixel_ar:
        self.channel_embed = nn.Embedding(in_channels, d_model)

    self.blocks = nn.ModuleList([
      GPTBlock(d_model, h, d_ff, dropout,
               activation_checkpointing=activation_checkpointing)
      for _ in range(N)
    ])

    # Output head + Weight Tying
    # embedding 和 output head 共享权重矩阵，减少参数量。
    # Ref: Press & Wolf, "Using the Output Embedding to Improve Language Models," EACL 2017.
    self.head = nn.Linear(d_model, vocab_size, bias=False)
    self.head.weight = self.token_embed.weight

    self._init_weights()

  def _init_weights(self):
    """
    权重初始化：基础 std=0.02，残差通路输出投影用 1/√(2·N) 深度缩放。

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
        if any(name.endswith(s) for s in ('w_o', 'w2')):
          std = 0.02 / math.sqrt(2 * N)
        else:
          std = 0.02
        nn.init.normal_(module.weight, mean=0.0, std=std)
        if module.bias is not None:
          nn.init.zeros_(module.bias)
      elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)

  def _tokenize(self, x: torch.Tensor) -> torch.Tensor:
    """RGB float [0,1] → 整数 token 序列 (B, T)。"""
    B = x.size(0)
    x = x.clamp(0, 1)
    x = rgb_to_ycbcr_int(x) if self.use_ycbcr else (x * 255).round().long()
    if self.use_subpixel_ar:
        # pixel-first: [Y0,Cb0,Cr0, Y1,Cb1,Cr1, ...]
        x = x.permute(0, 2, 3, 1).reshape(B, -1)
    else:
        x = x.reshape(B, -1)
    return x

  def _embed_inputs(self, input_tokens: torch.Tensor,
                    coarse_ctx: torch.Tensor = None):
    """Token → hidden 的统一入口（forward / encode 共用）。

    返回 (hidden, position_ids)。处理:
      - token_embed
      - 可选 coarse_ctx additive 注入 (CC-iGPT)
      - 子像素 AR 的 channel_embed 与像素级 position_ids
    """
    hidden = self.token_embed(input_tokens)

    if coarse_ctx is not None:
        assert coarse_ctx.shape == hidden.shape, (
            f"coarse_ctx shape {tuple(coarse_ctx.shape)} != hidden {tuple(hidden.shape)}"
        )
        hidden = hidden + coarse_ctx

    T = input_tokens.shape[1]
    position_ids = None
    if self.use_subpixel_ar:
        C = self.in_channels
        channel_indices = torch.arange(T, device=input_tokens.device) % C
        hidden = hidden + self.channel_embed(channel_indices).unsqueeze(0)
        position_ids = torch.arange(T, device=input_tokens.device) // C

    return hidden, position_ids

  def forward(self, x, z_loss_weight: float = 1e-4, coarse_ctx: torch.Tensor = None):
    """
    参数:
      x:             (B, C, H, W) float [0,1]
      z_loss_weight: z-loss 权重，默认 1e-4
                     Ref: PaLM arXiv:2204.02311; OLMo 2 arXiv:2501.00656
      coarse_ctx:    可选 (B, T-1, d_model) tensor，作为 additive 全局上下文
                     注入到 token embedding 之上。用于 CC-iGPT 的 fine 模型，
                     由 coarse 分支 down→up→quantize 后的 token 经 fine.token_embed
                     得到。形状必须与 token embedding 后的 hidden 一致。
    """
    z_w = float(z_loss_weight)

    tokens = self._tokenize(x)
    # NTP：输入 x[0..T-1]，预测 x[1..T]
    input_tokens  = tokens[:, :-1]
    target_tokens = tokens[:, 1:]

    hidden, position_ids = self._embed_inputs(input_tokens, coarse_ctx=coarse_ctx)

    for block in self.blocks:
      hidden = block(hidden, position_ids=position_ids)

    logits = self.head(hidden).float()

    # Fused CE + z-loss: 一次 kernel launch 完成 softmax → CE → z-loss
    # （V=256 下 Fused Linear+CE kernel 经 roofline 证伪、未采用，详见
    #  experiments/kernel_negative_finding.md）
    if _USE_FUSED_CE and logits.is_cuda and z_w > 0:
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
        if z_w > 0:
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
  def encode(self, x, max_layer: int = None, pool: bool = False, coarse_ctx: torch.Tensor = None):
    """
    仅走 embed + blocks 到 max_layer，不计算 head / loss。

    参数:
      pool:        True 时对每层输出做 GAP 并转 CPU，返回 (B, d_model)，大幅节省显存。
                   False 时返回完整 (B, T, d_model)（兼容旧调用）。
      coarse_ctx:  CC-iGPT fine 分支的可选全局上下文（与 forward 同义）。
    返回:
      list[Tensor]，长度为 max_layer+1。
    """
    tokens = self._tokenize(x)
    input_tokens = tokens[:, :-1]
    hidden, position_ids = self._embed_inputs(input_tokens, coarse_ctx=coarse_ctx)

    if max_layer is None:
      max_layer = len(self.blocks) - 1
    outputs = []
    for i, block in enumerate(self.blocks):
      hidden = block(hidden, position_ids=position_ids)
      if pool:
        outputs.append(hidden.float().mean(dim=1).cpu())
      else:
        outputs.append(hidden)
      if i >= max_layer:
        break
    return outputs
