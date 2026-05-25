import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .igpt import IGPT


class CCIGPT(nn.Module):
    """Coarse-Conditioned iGPT (CC-iGPT)。

    双尺度结构：浅层 coarse iGPT (independently encoded into bitstream) +
    主线 fine iGPT，coarse 经 DOWN→UP→quantize 后通过 fine.token_embed 查表，
    作为 additive embedding (`α · coarse_ctx`) 注入 fine。

    bpd_total = (CE_coarse · N_coarse + CE_fine · N_fine) / ln2 / N_fine

    序列布局：coarse / fine 都用 sub-pixel AR (pixel-first)
    [R0,G0,B0,R1,G1,B1,...]，token-level 自然学到 p(G|R), p(B|R,G)。

    `coarse_in_channels` 允许 coarse 只压部分通道（R-only 灰度先验）：默认
    None → 与 in_channels 一致；传 1 → coarse 仅看 R 8×8（64 tokens, ~2%
    overhead），fine 自学 G/B 相对 R 的偏色修正。

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
        activation_checkpointing: bool = False,
        # coarse 通道数；None → 与 in_channels 一致。R-only 灰度先验传 1。
        coarse_in_channels: int = None,
        # DropPath 仅作用于 fine（深 24 层），coarse 浅层 6 层不需要
        drop_path: float = 0.0,
    ):
        super().__init__()
        assert image_size % pool_factor == 0, (
            f"image_size ({image_size}) 必须能被 pool_factor ({pool_factor}) 整除"
        )

        self.image_size = image_size
        self.in_channels = in_channels
        self.pool_factor = pool_factor
        self.coarse_size = image_size // pool_factor

        if coarse_in_channels is None:
            coarse_in_channels = in_channels
        # expand 仅能从 size=1 扩展，因此中间值（如 in_channels=3 时传 2）
        # 在 _compute_coarse_ctx 的 x_up.expand(-1, C_fine, -1, -1) 处会崩。
        # 实际语义只有两种：1（灰度先验）或 in_channels（全通道）。
        assert coarse_in_channels in (1, in_channels), (
            f"coarse_in_channels ({coarse_in_channels}) 必须 ∈ {{1, in_channels={in_channels}}}; "
            f"中间值 expand 不可达。"
        )

        shared = dict(
            vocab_size=vocab_size, dropout=dropout,
            activation_checkpointing=activation_checkpointing,
        )

        self.coarse = IGPT(image_size=self.coarse_size,
                           in_channels=coarse_in_channels,
                           d_model=coarse_d_model, N=coarse_N,
                           h=coarse_h, d_ff=coarse_d_ff,
                           drop_path=0.0, **shared)
        self.fine = IGPT(image_size=image_size,
                         in_channels=in_channels,
                         d_model=fine_d_model, N=fine_N,
                         h=fine_h, d_ff=fine_d_ff,
                         drop_path=drop_path,
                         **shared)

        # 可学习注入强度 α，初始 1.0。允许模型自适应 ctx 贡献度，
        # 避免 ctx 过强压制 fine 自身的 token embed。
        self.ctx_alpha = nn.Parameter(torch.ones(1))

    def _compute_coarse_ctx(self, coarse_tokens: torch.Tensor) -> torch.Tensor:
        """coarse 量化 token (B, N_c) → fine 用 additive coarse context (B, T_fine-1, d_model)。

        Bit-exact 一致性约束：bitstream 只携带 coarse 量化 token，decoder 必须
        独立从 token 重建出与 encoder 相同的 fine 条件分布。因此本路径输入是
        **量化后的 coarse token**（不是 float），整个管线 encoder/decoder 共用。

        管线（RGB-bit-exact, pixel-first）：
          1. token (B, N_c) → view(B,S,S,C) → permute(0,3,1,2) 转回 (B,C,S,S)
          2. 反量化到 RGB float [0,1]：int / 255
          3. bilinear UP 到 fine 分辨率
          4. 用 fine encoder 的 tokenize 规则重新 tokenize：(*255).round().long()
          5. permute(0,2,3,1).reshape(B,-1) 排成 pixel-first，fine.token_embed 查表
          6. 丢掉第一个 token 做 AR shift —— ctx[i] 与 fine 被预测位置 i 对齐
             （PixelCNN++ conditional / VAR multi-scale 标准语义）

        强制 autocast(enabled=False)：encoder/decoder 必须在完全相同的 dtype 下
        跑此函数才能 bit-exact。bilinear interp + round + token_embed 在 bf16/fp16
        下结果会跟 fp32 差 ±1 token，导致 bitstream 不可解。
        """
        device_type = "cuda" if coarse_tokens.is_cuda else "cpu"
        with torch.amp.autocast(device_type=device_type, enabled=False):
            B = coarse_tokens.size(0)
            S = self.coarse_size
            C_coarse = self.coarse.in_channels
            C_fine = self.in_channels

            coarse_chw = coarse_tokens.view(B, S, S, C_coarse).permute(0, 3, 1, 2).contiguous()

            rec = coarse_chw.float() / 255.0                     # (B, C_coarse, S, S)

            x_up = F.interpolate(
                rec, size=(self.image_size, self.image_size),
                mode='bilinear', align_corners=False,
            )
            # R-only / 部分通道 coarse 的"灰度先验"：把 C_coarse 通道复制扩展到
            # fine 的 C_fine 通道，每个像素 R/G/B 三个位置看到同一 coarse 值；fine
            # 自学 G/B 相对 R 的偏色。expand 仅是 stride=0 view（free），实际 materialize
            # 由下一行 (clamp * 255).round().long() 完成 —— 它创建新 storage 是 contiguous 的，
            # 因此后续 permute+reshape 安全，无需在 expand 后调 .contiguous() 多走一次 memcpy。
            if C_coarse < C_fine:
                x_up = x_up.expand(-1, C_fine, -1, -1)

            x_up_tok = (x_up.clamp(0, 1) * 255).round().long()
            x_up_tok = x_up_tok.permute(0, 2, 3, 1).reshape(B, -1)
            coarse_ctx = self.fine.token_embed(x_up_tok)         # (B, T, d_model)
            # AR 对齐：ctx[i] 与 fine 被预测位置 i 同位对齐（PixelCNN++ conditional /
            # VAR multi-scale 标准语义）。不作弊：coarse 走独立 bitstream，decoder 先
            # 解完整段 coarse token 得到完整 ctx 再按序解 fine；ctx 是 lossy 低频先验，
            # 不能反推 fine_tok[i] 的精确 0-255 整数值。
            return coarse_ctx[:, 1:]

    def forward(self, x, z_loss_weight: float = 1e-4):
        """
        返回 dict:
          loss / ce_loss (= ce_fine, 与 train.py 主指标兼容) /
          ce_loss_coarse / ce_loss_fine /
          bpd (bits/dim, 按 H·W·C 子像素数归一化；
               对应 BPD_total = (CE_c·N_c + CE_f·N_f) / ln2 / N_f) /
          ctx_alpha (detached) / logits (fused path 下为 None)
        """
        x = x.clamp(0, 1).to(torch.float32)              # encoder/decoder 一致性
        x_c_full = F.adaptive_avg_pool2d(x, self.coarse_size)
        if self.coarse.in_channels < self.in_channels:
            x_c_float = x_c_full[:, :self.coarse.in_channels]
        else:
            x_c_float = x_c_full

        out_c = self.coarse(x_c_float, z_loss_weight=z_loss_weight)
        # bit-exact: ctx 走 coarse 量化 token 重建路径（decoder 同款）
        coarse_tokens = self.coarse._tokenize(x_c_float)
        coarse_ctx = self._compute_coarse_ctx(coarse_tokens)
        out_f = self.fine(x, z_loss_weight=z_loss_weight,
                          coarse_ctx=self.ctx_alpha * coarse_ctx)

        N_c, N_f = self.coarse.seq_len, self.fine.seq_len
        bpd_total = (out_c["ce_loss"] * N_c + out_f["ce_loss"] * N_f) / math.log(2.0) / N_f

        return {
            "loss": out_c["loss"] + out_f["loss"],
            "ce_loss": out_f["ce_loss"],
            "ce_loss_coarse": out_c["ce_loss"],
            "ce_loss_fine": out_f["ce_loss"],
            "bpd": bpd_total,
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
            x_c_full = F.adaptive_avg_pool2d(x, self.coarse_size)
            if self.coarse.in_channels < self.in_channels:
                x_c_float = x_c_full[:, :self.coarse.in_channels]
            else:
                x_c_float = x_c_full
            coarse_tokens = self.coarse._tokenize(x_c_float)
            coarse_ctx = self.ctx_alpha * self._compute_coarse_ctx(coarse_tokens)
        return self.fine.encode(x, max_layer=max_layer, pool=pool,
                                coarse_ctx=coarse_ctx)
