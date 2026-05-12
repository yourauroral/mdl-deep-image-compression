import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .igpt import IGPT, rgb_to_ycbcr_int


class CCIGPT(nn.Module):
    """Coarse-Conditioned iGPT (CC-iGPT)。

    双尺度结构：浅层 coarse iGPT (independently encoded into bitstream) +
    主线 fine iGPT，coarse 经 DOWN→UP→quantize 后通过 fine.token_embed 查表，
    作为 additive embedding (`α · coarse_ctx`) 注入 fine。

    BPP_total = (CE_coarse · N_coarse + CE_fine · N_fine) / ln2 / N_fine

    Refs:
      Burt & Adelson, "The Laplacian Pyramid as a Compact Image Code," 1983
      van den Oord et al., "Conditional PixelCNN," NeurIPS 2016 (additive 条件)
      Tian et al., "VAR," NeurIPS 2024 (multi-scale AR)
    """

    def __init__(
        self,
        image_size=32,
        in_channels=3,
        vocab_size=256,
        pool_factor=4,
        # fine 模型（与 iGPT-S 一致）
        fine_d_model=512, fine_N=24, fine_h=8, fine_d_ff=1376,
        # coarse 模型（小且浅）
        coarse_d_model=256, coarse_N=6, coarse_h=4, coarse_d_ff=688,
        dropout=0.1,
        use_ycbcr: bool = True,
        activation_checkpointing: bool = False,
    ):
        super().__init__()
        assert image_size % pool_factor == 0, (
            f"image_size ({image_size}) 必须能被 pool_factor ({pool_factor}) 整除"
        )

        self.image_size = image_size
        self.in_channels = in_channels
        self.pool_factor = pool_factor
        self.coarse_size = image_size // pool_factor

        shared = dict(
            in_channels=in_channels, vocab_size=vocab_size, dropout=dropout,
            use_ycbcr=use_ycbcr,
            activation_checkpointing=activation_checkpointing,
            use_subpixel_ar=False,  # CC-iGPT 强制 channel-first 对齐 ctx
        )

        self.coarse = IGPT(image_size=self.coarse_size,
                           d_model=coarse_d_model, N=coarse_N,
                           h=coarse_h, d_ff=coarse_d_ff, **shared)
        self.fine = IGPT(image_size=image_size,
                         d_model=fine_d_model, N=fine_N,
                         h=fine_h, d_ff=fine_d_ff, **shared)

        # 可学习注入强度 α，初始 1.0。允许模型自适应 ctx 贡献度，
        # 避免 ctx 过强压制 fine 自身的 token embed。
        self.ctx_alpha = nn.Parameter(torch.ones(1))

        # 暴露给 train.py 日志使用
        self.seq_len = self.fine.seq_len
        self.d_model = fine_d_model
        self.N_layers = fine_N
        self.vocab_size = vocab_size
        self.use_ycbcr = use_ycbcr

    def _compute_coarse_ctx(self, coarse_tokens: torch.Tensor) -> torch.Tensor:
        """coarse 量化 token (B, N_c) → fine 用 additive coarse context (B, T_fine-1, d_model)。

        Bit-exact 一致性约束：bitstream 只携带 coarse 量化 token，decoder 必须
        独立从 token 重建出与 encoder 相同的 fine 条件分布。因此本路径输入是
        **量化后的 coarse token**（不是 float），整个管线 encoder/decoder 共用。

        管线：
          token (YCbCr or RGB int 0-255) → 反量化 float → 若是 YCbCr 则 BT.601 inverse
          → bilinear UP 到 fine 分辨率 → clamp → 再 tokenize (与 fine encoder 同规则)
          → fine.token_embed

        强制 autocast(enabled=False)：encoder/decoder 必须在完全相同的 dtype 下
        跑此函数才能 bit-exact。bilinear interp + round + token_embed 在 bf16/fp16
        下结果会跟 fp32 差 ±1 token，导致 bitstream 不可解。
        """
        # 该 reshape 依赖 channel-first 平铺 (IGPT._tokenize 的 use_subpixel_ar=False
        # 行为)。__init__ 已强制 coarse channel-first；此处再加运行时断言，防止
        # 有人替换 self.coarse 或将 _tokenize 改成 pixel-first 后 view(B,C,S,S)
        # 静默错位（loss 仍下降但 ctx 完全乱套）。
        assert not self.coarse.use_subpixel_ar, (
            "_compute_coarse_ctx 依赖 channel-first token 布局；"
            "self.coarse.use_subpixel_ar 必须为 False"
        )
        with torch.amp.autocast(device_type='cuda', enabled=False):
            B = coarse_tokens.size(0)
            S, C = self.coarse_size, self.in_channels
            rec = coarse_tokens.view(B, C, S, S).float() / 255.0

            if self.use_ycbcr:
                # ITU-R BT.601 inverse: YCbCr [0,1] → RGB [0,1]
                y, cb, cr = rec[:, 0], rec[:, 1], rec[:, 2]
                r = y + 1.402 * (cr - 0.5)
                g = y - 0.344136 * (cb - 0.5) - 0.714136 * (cr - 0.5)
                b = y + 1.772 * (cb - 0.5)
                rec = torch.stack([r, g, b], dim=1)
            rec = rec.clamp(0.0, 1.0)

            x_up = F.interpolate(
                rec, size=(self.image_size, self.image_size),
                mode='bilinear', align_corners=False,
            )
            if self.use_ycbcr:
                x_up_tok = rgb_to_ycbcr_int(x_up)
            else:
                x_up_tok = (x_up.clamp(0, 1) * 255).round().long()
            x_up_tok = x_up_tok.reshape(B, -1)                       # (B, H*W*C)
            coarse_ctx = self.fine.token_embed(x_up_tok)             # (B, T, d_model)
            return coarse_ctx[:, :-1]                                 # AR shift

    def forward(self, x, z_loss_weight: float = 1e-4):
        """
        返回 dict:
          loss / ce_loss (= ce_fine, 与 train.py 主指标兼容) /
          ce_loss_coarse / ce_loss_fine /
          bpp (BPP_total, 按 H·W·C 归一化) /
          ctx_alpha (detached) / logits (fused path 下为 None)
        """
        x = x.clamp(0, 1).to(torch.float32)              # encoder/decoder 一致性
        x_c_float = F.adaptive_avg_pool2d(x, self.coarse_size)

        out_c = self.coarse(x_c_float, z_loss_weight=z_loss_weight)
        # bit-exact: ctx 走 coarse 量化 token 重建路径（decoder 同款）
        coarse_tokens = self.coarse._tokenize(x_c_float)
        coarse_ctx = self._compute_coarse_ctx(coarse_tokens)
        out_f = self.fine(x, z_loss_weight=z_loss_weight,
                          coarse_ctx=self.ctx_alpha * coarse_ctx)

        N_c, N_f = self.coarse.seq_len, self.fine.seq_len
        bpp_total = (out_c["ce_loss"] * N_c + out_f["ce_loss"] * N_f) / math.log(2.0) / N_f

        return {
            "loss": out_c["loss"] + out_f["loss"],
            "ce_loss": out_f["ce_loss"],
            "ce_loss_coarse": out_c["ce_loss"],
            "ce_loss_fine": out_f["ce_loss"],
            "bpp": bpp_total,
            "ctx_alpha": self.ctx_alpha.detach(),
            "logits": out_f["logits"],
        }

    @torch.no_grad()
    def encode(self, x, max_layer: int = None, pool: bool = False,
               use_coarse_ctx: bool = True):
        """对 fine 子模型做 linear probe / 表征提取。

        参数:
          use_coarse_ctx: True (默认) 注入 α·coarse_ctx，得到"条件后表征"；
                          False 跳过 ctx，得到"裸 fine 表征"，用于消融对比
                          (论文里通常两组都报)。
        """
        x = x.clamp(0, 1).to(torch.float32)
        coarse_ctx = None
        if use_coarse_ctx:
            x_c_float = F.adaptive_avg_pool2d(x, self.coarse_size)
            coarse_tokens = self.coarse._tokenize(x_c_float)
            coarse_ctx = self.ctx_alpha * self._compute_coarse_ctx(coarse_tokens)
        return self.fine.encode(x, max_layer=max_layer, pool=pool,
                                coarse_ctx=coarse_ctx)
