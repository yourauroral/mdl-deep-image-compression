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
        # 共享开关
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
        assert image_size % pool_factor == 0, (
            f"image_size ({image_size}) 必须能被 pool_factor ({pool_factor}) 整除"
        )

        self.image_size = image_size
        self.in_channels = in_channels
        self.pool_factor = pool_factor
        self.coarse_size = image_size // pool_factor

        shared = dict(
            in_channels=in_channels, vocab_size=vocab_size, dropout=dropout,
            use_ycbcr=use_ycbcr, use_rope=use_rope, use_post_norm=use_post_norm,
            use_swiglu=use_swiglu, use_qk_norm=use_qk_norm,
            use_depth_scaled_init=use_depth_scaled_init, use_zloss=use_zloss,
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

    def _compute_coarse_ctx(self, x_c_float: torch.Tensor) -> torch.Tensor:
        """coarse float 输入 → fine 用的 additive coarse context (B, T_fine-1, d_model)。

        UP 后量化路径必须与 fine encoder 保持一致 —— fine 内部 use_subpixel_ar=False
        强制 channel-first reshape。
        """
        x_up = F.interpolate(
            x_c_float, size=(self.image_size, self.image_size),
            mode='bilinear', align_corners=False,
        )
        if self.use_ycbcr:
            x_up_tok = rgb_to_ycbcr_int(x_up)
        else:
            x_up_tok = (x_up.clamp(0, 1) * 255).round().long()
        x_up_tok = x_up_tok.reshape(x_up_tok.size(0), -1)        # (B, H*W*C)
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
        coarse_ctx = self._compute_coarse_ctx(x_c_float)
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
    def encode(self, x, max_layer: int = None, pool: bool = False):
        """对 fine 子模型做 linear probe / 表征提取（含 coarse_ctx 条件）。"""
        x = x.clamp(0, 1).to(torch.float32)
        x_c_float = F.adaptive_avg_pool2d(x, self.coarse_size)
        coarse_ctx = self._compute_coarse_ctx(x_c_float)
        return self.fine.encode(x, max_layer=max_layer, pool=pool,
                                coarse_ctx=self.ctx_alpha * coarse_ctx)
