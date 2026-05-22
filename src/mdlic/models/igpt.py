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


class IGPT(nn.Module):
  """
  Image GPT 自回归压缩模型。

  RGB-bit-exact 域：直接在 RGB uint8 上建模，无色彩前端。
  标准 next-token prediction (NTP)，将图像展平为 token 序列，
  建模 p(x_t | x_{<t})，CE loss 直接对应 Shannon 最优编码长度。

  架构：RoPE base=500000、QK-Norm、RMSNorm Post-Norm（OLMo 2 风格）、
  SwiGLU FFN、Weight Tying（softmax 路径）、z-loss 正则、深度缩放初始化、
  子像素自回归 (pixel-first)。

  output_head：
    "softmax"（默认）— 256-way categorical + weight tying with token_embed
    "dmol"          — DMoLHead1D (K-mix 离散化 logistic), 输入端 token_embed 保留
                      但断开 output tying；fp32 强制；z_loss 自动跳过
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
    activation_checkpointing: bool = False,
    drop_path: float = 0.0,
    output_head: str = "softmax",
    n_mixtures: int = 10,
  ):
    super().__init__()
    self.seq_len = image_size * image_size * in_channels
    self.in_channels = in_channels
    self.image_size = image_size
    self.vocab_size = vocab_size
    self.d_model = d_model
    self.N_layers = N
    self.output_head_type = output_head
    self.token_embed = nn.Embedding(vocab_size, d_model)

    # 子像素自回归 (sub-pixel AR, Salimans et al., PixelCNN++ ICLR 2017):
    # 序列布局 [R0,G0,B0, R1,G1,B1, ...]，使同一像素内的通道能相互条件化:
    #   p(G_i | R_i, context), p(B_i | R_i, G_i, context)
    # channel_embed 给每个通道位置 (0=R, 1=G, 2=B) 学习一个嵌入，
    # 帮助模型区分同一像素内的不同通道 token (van den Oord NeurIPS 2016)
    self.channel_embed = nn.Embedding(in_channels, d_model)

    # DropPath 线性 schedule: 第 i 层 drop_prob = drop_path · i/(N-1)
    # Ref: Huang et al., ECCV 2016 — 深层 drop 更激进，浅层保留信息
    dpr = [drop_path * i / max(N - 1, 1) for i in range(N)]
    self.blocks = nn.ModuleList([
      GPTBlock(d_model, h, d_ff, dropout,
               activation_checkpointing=activation_checkpointing,
               drop_path=dpr[i])
      for i in range(N)
    ])

    # Output head：根据 output_head 分发
    if output_head == "softmax":
      # 256-way categorical + weight tying
      # Ref: Press & Wolf, "Using the Output Embedding to Improve Language Models," EACL 2017.
      self.head = nn.Linear(d_model, vocab_size, bias=False)
      self.head.weight = self.token_embed.weight
    elif output_head == "dmol":
      # 断 output tying：DMoL head 输出 K*3 维（与 vocab_size 不兼容）。
      # 输入端 token_embed 仍是 256-way categorical（cc_igpt._compute_coarse_ctx 复用）。
      from ..losses import DMoLHead1D
      self.head = DMoLHead1D(d_model, n_mixtures=n_mixtures)
      self.n_mixtures = n_mixtures
    else:
      raise ValueError(
        f"未知 output_head: '{output_head}'，支持 softmax/dmol"
      )

    self._init_weights()

  def _init_weights(self):
    """
    权重初始化：基础 std=0.02，残差通路输出投影用 1/√(2·N) 深度缩放。

    Head 子树跳过：
      - softmax 路径下 head.weight 与 token_embed 共享，跳过避免重复初始化
      - dmol 路径下 head 是 DMoLHead1D，自身在 __init__ 已做 std=0.005 + log_scale bias=+2

    参考:
      [1] Radford et al., "GPT-2," 2019 — 1/√(2·N) 深度缩放。
      [2] OLMo 2 arXiv:2501.00656 Section 3.2。
    """
    N = self.N_layers
    for name, module in self.named_modules():
      # 跳过 head 子树（含 head.proj 等所有后代）— softmax tying / dmol 自管
      if name == 'head' or name.startswith('head.'):
        continue
      if isinstance(module, nn.Linear):
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
    """RGB float [0,1] → 整数 token 序列 (B, T)。pixel-first: [R0,G0,B0, R1,...]。"""
    B = x.size(0)
    x = x.clamp(0, 1)
    x = (x * 255).round().long()
    return x.permute(0, 2, 3, 1).reshape(B, -1)

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
    C = self.in_channels
    channel_indices = torch.arange(T, device=input_tokens.device) % C
    hidden = hidden + self.channel_embed(channel_indices).unsqueeze(0)
    position_ids = torch.arange(T, device=input_tokens.device) // C

    return hidden, position_ids

  def forward(self, x, z_loss_weight: float = 1e-4, coarse_ctx: torch.Tensor = None):
    """
    参数:
      x:             (B, C, H, W) float [0,1]
      z_loss_weight: z-loss 权重（仅 softmax 路径生效），默认 1e-4
                     Ref: PaLM arXiv:2204.02311; OLMo 2 arXiv:2501.00656
      coarse_ctx:    可选 (B, T-1, d_model) tensor，作为 additive 全局上下文
                     注入到 token embedding 之上。用于 CC-iGPT 的 fine 模型。

    返回 dict:
      softmax 路径: {loss, ce_loss, logits (B, T-1, V)}
      dmol 路径:    {loss = ce_loss = NLL/N (nat/sub-pixel), logits = None}
    """
    tokens = self._tokenize(x)
    # NTP：输入 x[0..T-1]，预测 x[1..T]
    input_tokens  = tokens[:, :-1]
    target_tokens = tokens[:, 1:]

    hidden, position_ids = self._embed_inputs(input_tokens, coarse_ctx=coarse_ctx)

    for block in self.blocks:
      hidden = block(hidden, position_ids=position_ids)

    if self.output_head_type == "dmol":
      # fp32 强制：DMoL CDF 差分 + logsumexp 在 fp16/bf16 下数值不稳。
      # 复用 cc_igpt._compute_coarse_ctx 同款 autocast(enabled=False) pattern。
      from ..losses import dmol_loss_1d
      device_type = "cuda" if hidden.is_cuda else "cpu"
      with torch.amp.autocast(device_type=device_type, enabled=False):
        params = self.head(hidden.float())   # (B, T-1, K*3)
        ce_loss = dmol_loss_1d(params, target_tokens, self.n_mixtures)
        loss = ce_loss   # z-loss 路径下游已 dispatch 跳过；显式不加 z_loss 项

      return {
        "loss": loss,
        "ce_loss": ce_loss,
        "logits": None,
      }

    # ── softmax 路径（原实现） ──
    z_w = float(z_loss_weight)

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
