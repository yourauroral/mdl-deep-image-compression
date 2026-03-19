import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import math 

from .layers import LayerNormalization, GPTBlock, RMSNorm

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
  return (ycbcr * 255).long()

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
  ):
    super().__init__()
    self.seq_len = image_size * image_size * in_channels
    self.vocab_size = vocab_size
    self.use_mtp = use_mtp
    self.token_embed = nn.Embedding(vocab_size, d_model)

    self.blocks = nn.ModuleList([
      GPTBlock(d_model, h, d_ff, dropout)
      for _ in range(N)
    ])

    self.norm = RMSNorm(d_model)
    self.head = nn.Linear(d_model, vocab_size)

    # MTP 辅助头：一个额外的 GPTBlock + 共享 output head
    # 参考: DeepSeek-V3 arXiv:2412.19437 Section 2.3
    if use_mtp:
      self.mtp_block = GPTBlock(d_model, h, d_ff, dropout)
      self.mtp_norm  = RMSNorm(d_model)
      # 共享 self.head 权重，不新建 Linear，节省参数 [1]

    self._init_weights()
  
  def _init_weights(self):
    """
    权重初始化：基础 std=0.02，残差通路输出投影用深度缩放。

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
    N = len(self.blocks)
    for name, module in self.named_modules():
      if isinstance(module, nn.Linear):
        # 残差通路输出投影：深度缩放 [1][2][3]
        if any(name.endswith(s) for s in ('w_o', 'w2')):
          std = 0.02 / math.sqrt(2 * N)
        else:
          std = 0.02
        nn.init.normal_(module.weight, mean=0.0, std=std)
        if module.bias is not None:
          nn.init.zeros_(module.bias)
      elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)

  def forward(self, x, z_loss_weight: float = 1e-4, mtp_weight: float = 0.1):
    """
    参数:
      x:             (B, C, H, W) float [0,1]
      z_loss_weight: z-loss 权重，默认 1e-4（从 config 传入）
                     Ref: PaLM arXiv:2204.02311; OLMo 2 arXiv:2501.00656
      mtp_weight:    MTP 辅助 loss 权重，默认 0.1（use_mtp=True 时生效）
                     Ref: DeepSeek-V3 arXiv:2412.19437 Section 2.3
    """
    z_loss_weight = float(z_loss_weight)
    mtp_weight = float(mtp_weight)
    x = x.clamp(0, 1)
    B = x.size(0)

    # RGB → YCbCr 后展平为 token 序列
    # 通道顺序：[Y 全图, Cb 全图, Cr 全图]（非交错），使 Y 通道的自回归
    # 条件概率更集中，降低模型难度。
    # Ref: ITU-R BT.601 [1]; JPEG [2]; H.264 [3]（见 rgb_to_ycbcr_int 注释）
    x = rgb_to_ycbcr_int(x)   # (B, 3, H, W) long [0,255]
    x = x.reshape(B, -1)      # (B, seq_len)，seq_len = 3*H*W

    # NTP：输入 x[0..T-1]，预测 x[1..T]
    input_tokens  = x[:, :-1]   # (B, T)
    target_tokens = x[:, 1:]    # (B, T)

    hidden = self.token_embed(input_tokens)   # (B, T, d_model)

    for block in self.blocks:
      hidden = block(hidden, mask=None)

    hidden_norm = self.norm(hidden)           # (B, T, d_model)
    logits = self.head(hidden_norm).float()   # (B, T, vocab_size)

    # --- 主 loss: NTP cross-entropy ---
    ce_loss = F.cross_entropy(
      logits.reshape(-1, self.vocab_size),
      target_tokens.reshape(-1),
      reduction="mean"
    )

    # --- z-loss: 防止 logits 发散 ---
    # Ref: PaLM arXiv:2204.02311 Section 5; OLMo 2 arXiv:2501.00656
    log_z = torch.logsumexp(logits, dim=-1)
    z_loss = (log_z ** 2).mean()

    loss = ce_loss + z_loss_weight * z_loss

    # --- MTP 辅助 loss（可选）---
    # 预测 x[2..T+1]，利用同一 hidden state 多偏移一位
    # Ref: DeepSeek-V3 arXiv:2412.19437 Section 2.3
    if self.use_mtp and x.shape[1] > 2:
      # MTP target: x[2..T]（去掉首尾各1个，与 hidden[0..T-2] 对齐）
      mtp_target  = x[:, 2:]                       # (B, T-1)
      mtp_hidden  = self.mtp_block(hidden[:, :-1])  # (B, T-1, d_model)
      mtp_hidden  = self.mtp_norm(mtp_hidden)
      mtp_logits  = self.head(mtp_hidden).float()   # 共享 output head [1]
      mtp_loss    = F.cross_entropy(
        mtp_logits.reshape(-1, self.vocab_size),
        mtp_target.reshape(-1),
        reduction="mean"
      )
      loss = loss + mtp_weight * mtp_loss

    return {
      "loss": loss,
      "logits": logits
    }