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
    [3] Wiegand et al., "Overview of the H.264/AVC Video Coding Standard,"
        IEEE Trans. Circuits and Systems for Video Technology, 2003.
        H.264/HEVC 同样以 YCbCr 4:2:0 作为标准色彩格式。

  动机：
    RGB 三通道之间高度相关（对角线方向的 PCA 主成分即为亮度）。
    YCbCr 解相关后，Y 通道承载 ~90% 的结构信息且像素分布更集中，
    Cb/Cr 通道平滑（低熵），自回归模型更容易学习 p(x_t | x_{<t})，
    理论上可直接降低 cross-entropy loss（即 BPP）。

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
  # round() 而非默认的 floor (long 截断)，减小量化误差
  # 最大误差: floor 可达 0.999 → 0，round 最多 0.5/255
  return (ycbcr * 255).round().long()

class IGPT(nn.Module):
  """
  Image GPT 自回归压缩模型。

  在标准 next-token prediction (NTP) 基础上，可选地加入
  Multi-Token Prediction (MTP) 辅助头：

  MTP 参考:
    [1] DeepSeek-V3 Tech Report, arXiv:2412.19437, 2024, Section 2.3.
        每个位置额外预测第 t+2 个 token，辅助头为一个轻量 GPTBlock + Linear。
        训练时 mtp_loss 加权叠加，权重建议从 0.1 开始实验。
        推理时 MTP 头可复用为 speculative decoding draft head（本实现暂不启用）。
    [2] Gloeckle et al., "Better & Faster Large Language Models via
        Multi-Token Prediction," arXiv:2404.19737, 2024.
        系统验证 MTP 在小模型上的效果，提供更密集的梯度信号。

  MTP 工作方式（本实现）：
    main head:  预测 x[1], x[2], …, x[T]   （标准 NTP，偏移 1）
    mtp head:   预测 x[2], x[3], …, x[T+1]  （额外偏移 1，共偏移 2）
    两者共享 token embedding 和 output head 权重 [1]。
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
    use_mtp: bool = False,
    # 消融开关
    use_ycbcr: bool = True,
    use_rope: bool = True,
    use_post_norm: bool = True,
    use_swiglu: bool = True,
    use_qk_norm: bool = True,
    use_depth_scaled_init: bool = True,
    use_zloss: bool = True,
    activation_checkpointing: bool = False,
    # Logit soft-capping: 防止 logits 幅值爆炸
    # logits = cap * tanh(logits / cap)，平滑限幅
    # Ref: Gemma 2, arXiv:2408.00118 — output logit capping = 30.0
    logit_soft_cap: float = 0.0,  # 0.0 = 禁用；推荐 30.0
    # 子像素自回归 (Sub-pixel Autoregression):
    # 将序列从 channel-first [Y_all, Cb_all, Cr_all] 改为 pixel-first
    # [Y0,Cb0,Cr0, Y1,Cb1,Cr1, ...]，使同一像素内的通道能相互条件化:
    #   p(Cb_i | Y_i, context)  和  p(Cr_i | Y_i, Cb_i, context)
    # 这是 PixelCNN++/PixelSNAIL 取得低 bits/dim 的关键技巧之一。
    # Ref: Salimans et al., "PixelCNN++," ICLR 2017 — 通道间条件依赖
    # Ref: van den Oord et al., "Conditional Image Generation with PixelCNN
    #       Decoders," NeurIPS 2016 — 子像素条件分解
    # Ref: Chen et al., "PixelSNAIL," ICML 2018 — 通道自回归
    use_subpixel_ar: bool = False,
    # Sliding Window Attention: 限制每个 token 只关注最近 W 个 token，
    # 利用图像的空间局部性减少注意力计算量。
    # -1 = full causal attention（默认）; 推荐 512 或 1024。
    # Ref: Beltagy et al., "Longformer," arXiv:2004.05150, 2020
    # Ref: Jiang et al., "Mistral 7B," arXiv:2310.06825, 2023
    sliding_window_size: int = -1,
    # Full attention 穿插频率 (Mistral 风格):
    # 每 N 层有 1 层使用 full causal attention，其余层使用 sliding window。
    # 0 = 全部使用 sliding_window_size; 4 = 每 4 层 1 层 full。
    # Ref: Jiang et al., "Mistral 7B," arXiv:2310.06825, 2023
    full_attn_every_n: int = 0,
  ):
    super().__init__()
    self.seq_len = image_size * image_size * in_channels
    self.in_channels = in_channels
    self.image_size = image_size
    self.vocab_size = vocab_size
    self.d_model = d_model
    self.N_layers = N
    self.use_mtp = use_mtp
    self.use_ycbcr = use_ycbcr
    self.use_rope = use_rope
    self.use_zloss = use_zloss
    self.use_depth_scaled_init = use_depth_scaled_init
    self.logit_soft_cap = logit_soft_cap
    self.use_subpixel_ar = use_subpixel_ar
    self.token_embed = nn.Embedding(vocab_size, d_model)

    # Channel embedding: 子像素自回归模式下，为每个通道位置（0=Y/R, 1=Cb/G, 2=Cr/B）
    # 添加可学习的通道嵌入，帮助模型区分同一像素内的不同通道 token。
    # 标准 RoPE/learned PE 只编码序列位置，无法区分通道语义。
    # Ref: van den Oord et al., "Conditional Image Generation with PixelCNN
    #       Decoders," NeurIPS 2016 — 通道条件化需要通道标识
    if use_subpixel_ar:
        self.channel_embed = nn.Embedding(in_channels, d_model)

    # Learned positional embedding fallback（use_rope=False 时）
    # Ref: Radford et al., "GPT-2," 2019 — learned absolute PE
    # 注意: 输入 token 序列长度 = seq_len - 1（去掉最后一个 token 作为 target）
    if not use_rope:
        self.pos_embed = nn.Embedding(self.seq_len - 1, d_model)

    self.sliding_window_size = sliding_window_size
    self.full_attn_every_n = full_attn_every_n

    self.blocks = nn.ModuleList([
      GPTBlock(d_model, h, d_ff, dropout,
               use_post_norm=use_post_norm,
               use_swiglu=use_swiglu,
               use_qk_norm=use_qk_norm,
               use_rope=use_rope,
               activation_checkpointing=activation_checkpointing,
               window_size=self._get_layer_window_size(layer_idx))
      for layer_idx in range(N)
    ])

    # Final RMSNorm: 仅在 pre-norm 模式下需要。
    # Post-norm 模式中每个 sublayer 输出已经过 RMSNorm（x = x + RMSNorm(sublayer(x))），
    # 最后一个 block 输出已归一化，额外加 norm 会破坏消融对比的公平性。
    # Pre-norm 模式中最后一个 block 输出未归一化，必须加 final norm。
    # Ref: Xiong et al., "On Layer Normalization in the Transformer Architecture,"
    #      ICML 2020, arXiv:2002.04745 — pre-norm 需要 final norm。
    # Ref: OLMo 2 arXiv:2501.00656 Section 3.1 — post-norm 不需要 final norm。
    self.use_post_norm = use_post_norm
    if not use_post_norm:
        self.final_norm = RMSNorm(d_model)

    self.head = nn.Linear(d_model, vocab_size, bias=False)

    # Weight Tying: embedding 和 output head 共享权重矩阵
    # 减少 vocab_size × d_model 个参数，同时提供正则化效果。
    # Ref: Press & Wolf, "Using the Output Embedding to Improve Language Models,"
    #      EACL 2017, arXiv:1608.05859.
    # Ref: Radford et al., "GPT-2," 2019 — GPT-2 使用 weight tying。
    # Ref: OLMo 2 arXiv:2501.00656 Section 3.1 — OLMo 2 使用 weight tying。
    self.head.weight = self.token_embed.weight

    # MTP 辅助头：一个额外的 GPTBlock + 共享 output head
    # 参考: DeepSeek-V3 arXiv:2412.19437 Section 2.3
    if use_mtp:
      self.mtp_block = GPTBlock(d_model, h, d_ff, dropout,
                                 use_post_norm=use_post_norm,
                                 use_swiglu=use_swiglu,
                                 use_qk_norm=use_qk_norm,
                                 use_rope=use_rope)
      # MTP 的 final norm 同样遵循 post-norm/pre-norm 规则
      if not use_post_norm:
          self.mtp_norm = RMSNorm(d_model)
      # 共享 self.head 权重，不新建 Linear，节省参数 [1]

    self._init_weights()

  def _get_layer_window_size(self, layer_idx: int) -> int:
    """
    决定第 layer_idx 层的注意力窗口大小。

    - sliding_window_size <= 0: 所有层 full causal (-1)
    - full_attn_every_n == 0: 所有层使用 sliding_window_size
    - full_attn_every_n > 0: 每 N 层中最后 1 层 full causal，其余 windowed
      例如 N=4: layer 3,7,11 用 full，其余 windowed
      Ref: Jiang et al., "Mistral 7B," arXiv:2310.06825, 2023
    """
    if self.sliding_window_size <= 0:
        return -1  # full causal
    if (self.full_attn_every_n > 0
            and (layer_idx + 1) % self.full_attn_every_n == 0):
        return -1  # 此层 full causal
    return self.sliding_window_size

  def _init_weights(self):
    """
    权重初始化：基础 std=0.02，残差通路输出投影用深度缩放（可选）。

    参考:
      [1] Radford et al., "GPT-2," 2019.
          "Modified initialization which accounts for the accumulation on
           the residual path with model depth. We scale the weights of
           residual layers at initialization by 1/√(2·N)."
      [2] OLMo 2 Tech Report, arXiv:2501.00656, 2025, Section 3.2.
          output projection 用 std = 0.02 / √(2·N_layers).
      [3] CS336 "Language Models from Scratch," Stanford, Spring 2024,
          Assignment 1 参考实现。

    深度缩放原理：
      残差网络中，L 个 block 叠加后残差流方差累积为 O(L·σ²)。
      对残差通路的输出投影（w_o, w2）乘以 1/√(2L) 可保持方差稳定。
      作用层：MultiHeadAttention 的 w_o，以及 SwiGLU FFN 的 w2（down-proj）。
    """
    N = self.N_layers
    for name, module in self.named_modules():
      if isinstance(module, nn.Linear):
        # Weight tying: head.weight is token_embed.weight，跳过避免重复初始化
        if name.endswith('head'):
          continue
        # 残差通路输出投影：深度缩放 [1][2][3]（use_depth_scaled_init 开关）
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

    muP 通过 fan_in 缩放使不同宽度的模型在相同超参数下表现一致，
    允许在小模型上调参后直接迁移到大模型。

    参考:
      [1] Yang et al., "Tensor Programs V: Tuning Large Neural Networks via
          Zero-Shot Hyperparameter Transfer," arXiv:2203.03466, 2022.
          Table 8: SP vs muP 初始化规则。

    初始化规则（muP, Table 8）:
      - Embedding:         std = 1.0（与宽度无关）
      - Hidden layers:     std = 1/√fan_in
      - Output head:       weight tying 后与 embedding 共享，用 embedding std=1.0
      - 残差投影(w_o,w2):  额外 ×1/√(2N)（保持深度方差稳定）
    """
    N = self.N_layers
    d = self.d_model
    for name, module in self.named_modules():
      if isinstance(module, nn.Linear):
        # Weight tying: head.weight is token_embed.weight，跳过避免重复初始化
        if name.endswith('head'):
          continue
        fan_in = module.weight.shape[1]
        if any(name.endswith(s) for s in ('w_o', 'w2')):
          # 残差投影: 1/√fan_in × 1/√(2N)
          std = (1.0 / math.sqrt(fan_in)) / math.sqrt(2 * N)
        else:
          # Hidden layers: std = 1/√fan_in
          std = 1.0 / math.sqrt(fan_in)
        nn.init.normal_(module.weight, mean=0.0, std=std)
        if module.bias is not None:
          nn.init.zeros_(module.bias)
      elif isinstance(module, nn.Embedding):
        # Embedding: std = 1.0 [1] Table 8
        nn.init.normal_(module.weight, mean=0.0, std=1.0)

  def forward(self, x, z_loss_weight: float = 1e-4, mtp_weight: float = 0.1,
              label_smoothing_sigma: float = 0.0):
    """
    参数:
      x:             (B, C, H, W) float [0,1]
      z_loss_weight: z-loss 权重，默认 1e-4（从 config 传入）
                     Ref: PaLM arXiv:2204.02311; OLMo 2 arXiv:2501.00656
      mtp_weight:    MTP 辅助 loss 权重，默认 0.1（use_mtp=True 时生效）
                     Ref: DeepSeek-V3 arXiv:2412.19437 Section 2.3
      label_smoothing_sigma: Gaussian label smoothing σ，默认 0.0（禁用）
                     σ > 0 时训练用 soft targets，验证自动用 hard targets。
                     Ref: Szegedy et al., arXiv:1512.00567
    """
    z_loss_weight = float(z_loss_weight)
    mtp_weight = float(mtp_weight)
    x = x.clamp(0, 1)
    B = x.size(0)

    if self.use_ycbcr:
        # RGB → YCbCr 后展平为 token 序列
        # Ref: ITU-R BT.601 [1]; JPEG [2]; H.264 [3]（见 rgb_to_ycbcr_int 注释）
        x = rgb_to_ycbcr_int(x)   # (B, 3, H, W) long [0,255]
    else:
        # 消融 baseline: 直接量化 RGB 到 [0,255]
        x = (x * 255).round().long()

    if self.use_subpixel_ar:
        # 子像素自回归: pixel-first 排列
        # (B, C, H, W) → (B, H, W, C) → (B, H*W*C)
        # 序列: [Y0, Cb0, Cr0, Y1, Cb1, Cr1, ..., Y_{N-1}, Cb_{N-1}, Cr_{N-1}]
        # 使得 Cb_i 可以看到同像素的 Y_i，Cr_i 可以看到 Y_i 和 Cb_i。
        # 这种排列下 causal mask 自然实现通道间条件依赖:
        #   p(Cb_i | Y_i, Y_{<i}, Cb_{<i}, Cr_{<i})
        #   p(Cr_i | Y_i, Cb_i, Y_{<i}, Cb_{<i}, Cr_{<i})
        # Ref: Salimans et al., "PixelCNN++," ICLR 2017
        # Ref: van den Oord et al., "Conditional Image Generation with PixelCNN
        #       Decoders," NeurIPS 2016
        x = x.permute(0, 2, 3, 1).reshape(B, -1)  # (B, seq_len)
    else:
        # 默认: channel-first 排列 [Y_all, Cb_all, Cr_all]
        x = x.reshape(B, -1)      # (B, seq_len)，seq_len = 3*H*W

    # NTP：输入 x[0..T-1]，预测 x[1..T]
    input_tokens  = x[:, :-1]   # (B, T)
    target_tokens = x[:, 1:]    # (B, T)

    hidden = self.token_embed(input_tokens)   # (B, T, d_model)

    # 子像素自回归: 添加 channel embedding + 像素级 RoPE position_ids
    position_ids = None
    if self.use_subpixel_ar:
        T = input_tokens.shape[1]
        C = self.in_channels
        # Channel indices: [0,1,2, 0,1,2, ...] 循环，标识每个 token 属于哪个通道
        channel_indices = torch.arange(T, device=input_tokens.device) % C
        hidden = hidden + self.channel_embed(channel_indices).unsqueeze(0)
        # Pixel-level position_ids: 同一像素内的 C 个 token 共享同一位置 ID
        # [0,0,0, 1,1,1, 2,2,2, ...]，使 RoPE 只编码像素间空间距离
        # Ref: Su et al., "RoFormer," arXiv:2104.09864 — RoPE 编码相对位置
        position_ids = torch.arange(T, device=input_tokens.device) // C

    # Learned positional embedding（use_rope=False 时）
    if not self.use_rope:
        T = input_tokens.shape[1]
        positions = torch.arange(T, device=input_tokens.device)
        hidden = hidden + self.pos_embed(positions)

    for block in self.blocks:
      hidden = block(hidden, mask=None, position_ids=position_ids)

    # Final norm: 仅 pre-norm 模式需要（post-norm 模式在 block 内已归一化）
    if not self.use_post_norm:
        hidden = self.final_norm(hidden)

    # --- 主 loss: NTP cross-entropy (+ 可选 fused z-loss) ---
    z_w = z_loss_weight if self.use_zloss else 0.0

    # Gaussian Label Smoothing: σ > 0 且训练模式时，绕过 fused kernels，
    # 使用 PyTorch F.cross_entropy 的 soft target 路径。
    # 验证时（model.eval()）自动走 hard target 路径，保证 BPP 可比性。
    # Ref: Szegedy et al., "Rethinking the Inception Architecture," arXiv:1512.00567
    use_label_smoothing = label_smoothing_sigma > 0.0 and self.training
    if use_label_smoothing:
        from ..utils import build_gaussian_targets
        logits = self.head(hidden).float()        # (B, T, vocab_size)
        if self.logit_soft_cap > 0:
            logits = self.logit_soft_cap * torch.tanh(logits / self.logit_soft_cap)
        # 构建高斯 soft targets: (B*T, V)
        soft_targets = build_gaussian_targets(
            target_tokens.reshape(-1), self.vocab_size, label_smoothing_sigma
        )
        # F.cross_entropy 支持 (N, C) float target（PyTorch 1.10+）
        # 等价于 -sum(soft_target * log_softmax(logits), dim=-1).mean()
        ce_loss = F.cross_entropy(
            logits.reshape(-1, self.vocab_size), soft_targets, reduction="mean"
        )
        if self.use_zloss:
            log_z = torch.logsumexp(logits, dim=-1)
            z_loss = (log_z ** 2).mean()
            loss = ce_loss + z_w * z_loss
        else:
            loss = ce_loss
    # Fused Linear + CE + z-loss: 将 output head 投影和 CE loss 合并，
    # 避免实例化完整 (B*T, V) logits 张量。
    # 条件: 无 soft-capping（需要完整 logits）、z-loss 启用、kernel 可用。
    # Ref: Liger-Kernel arXiv:2410.10989（手写 Fused Linear CE pattern）
    elif (_USE_FUSED_LINEAR_CE and hidden.is_cuda and z_w > 0
            and self.logit_soft_cap <= 0):
        ce_loss, z_loss = _fused_linear_ce(
            hidden.reshape(-1, self.d_model),
            self.head.weight,
            target_tokens.reshape(-1),
            z_loss_weight=z_w,
        )
        loss = ce_loss + z_w * z_loss
        logits = None  # fused path 不实例化 logits
    else:
        logits = self.head(hidden).float()        # (B, T, vocab_size)

        # Logit soft-capping: logits = cap * tanh(logits / cap)
        # 平滑地限制 logits 幅值，与 z-loss 互补：
        #   z-loss 通过惩罚 logsumexp 间接约束 logits 分布，
        #   soft-capping 直接约束 logits 幅度上界。
        # Ref: Gemma 2, arXiv:2408.00118, Section 2 — output logit capping = 30.0
        if self.logit_soft_cap > 0:
            logits = self.logit_soft_cap * torch.tanh(logits / self.logit_soft_cap)

        if _USE_FUSED_CE and logits.is_cuda and z_w > 0:
            # Fused CE + z-loss: 一次 kernel launch 完成 softmax → CE → z-loss
            # Ref: Liger-Kernel arXiv:2410.10989（手写实现）
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
                # z-loss: 防止 logits 发散
                # Ref: PaLM arXiv:2204.02311 Section 5; OLMo 2 arXiv:2501.00656
                log_z = torch.logsumexp(logits, dim=-1)
                z_loss = (log_z ** 2).mean()
                loss = ce_loss + z_w * z_loss
            else:
                loss = ce_loss

    # --- MTP 辅助 loss（可选）---
    # 预测 x[2..T+1]，利用同一 hidden state 多偏移一位
    # Ref: DeepSeek-V3 arXiv:2412.19437 Section 2.3
    if self.use_mtp and x.shape[1] > 2:
      # MTP target: x[2..T]（去掉首尾各1个，与 hidden[0..T-2] 对齐）
      mtp_target  = x[:, 2:]                       # (B, T-1)
      mtp_hidden  = self.mtp_block(hidden[:, :-1])  # (B, T-1, d_model)
      if not self.use_post_norm:
          mtp_hidden = self.mtp_norm(mtp_hidden)
      mtp_logits  = self.head(mtp_hidden).float()   # 共享 output head [1]
      mtp_loss    = F.cross_entropy(
        mtp_logits.reshape(-1, self.vocab_size),
        mtp_target.reshape(-1),
        reduction="mean"
      )
      loss = loss + mtp_weight * mtp_loss

    return {
      "loss": loss,
      # 单独返回 ce_loss，用于计算 BPP。
      # BPP = ce_loss / ln(2)，因为 ce_loss = -log p(x) 是 nats 单位的编码长度。
      # z_loss 是正则项（惩罚 logits 幅度），不对应实际编码比特，
      # 混入 BPP 计算会高估真实码率。
      # Ref: Shannon, "A Mathematical Theory of Communication," 1948 —
      #      最优编码长度 = -log₂ p(x) = -ln p(x) / ln(2)
      "ce_loss": ce_loss,
      "logits": logits
    }