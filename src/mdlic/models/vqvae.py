"""
Multi-Scale VQVAE — 将图像编码为多尺度离散 token map。

VAR (Visual Autoregressive Modeling) 的 tokenizer 组件。
每个 scale 独立量化残差特征，产生从粗到细的离散表示。

架构:
  Encoder: Conv stack → multi-scale 特征提取
  VectorQuantizer: EMA codebook 更新，各 scale 独立 codebook
  Decoder: 从所有 scale 的量化特征重建原图

尺度层级 (CIFAR-10 32×32):
  Scale 1: 1×1   (1 token)
  Scale 2: 2×2   (4 tokens)
  Scale 3: 4×4   (16 tokens)
  Scale 4: 8×8   (64 tokens)
  Scale 5: 16×16 (256 tokens)
  Scale 6: 32×32 (1024 tokens)
  总计: 1365 tokens

参考:
  [1] van den Oord et al., "Neural Discrete Representation Learning,"
      NeurIPS 2017, arXiv:1711.00937 — 原始 VQ-VAE
  [2] Razavi et al., "Generating Diverse High-Fidelity Images with VQ-VAE-2,"
      NeurIPS 2019, arXiv:1906.00446 — 多尺度 VQ-VAE
  [3] Tian et al., "Visual Autoregressive Modeling: Scalable Image Generation
      via Next-Scale Prediction," NeurIPS 2024, arXiv:2404.02905 — VAR tokenizer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ResBlock(nn.Module):
    """
    残差卷积块: Conv → GroupNorm → SiLU → Conv → GroupNorm + skip。

    参考:
      [1] He et al., "Deep Residual Learning," CVPR 2016, arXiv:1512.03385
      [2] Esser et al., "Taming Transformers for High-Resolution Image Synthesis,"
          CVPR 2021, arXiv:2012.09841 — VQGAN 中使用的 ResBlock 结构
    """
    def __init__(self, channels: int, num_groups: int = 32):
        super().__init__()
        # GroupNorm groups: min(num_groups, channels)，防止 channels < 32 时报错
        ng = min(num_groups, channels)
        self.block = nn.Sequential(
            nn.GroupNorm(ng, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(ng, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
        )

    def forward(self, x):
        return x + self.block(x)


class Encoder(nn.Module):
    """
    多尺度 Encoder: 逐步下采样并提取各 scale 特征。

    输入: (B, 3, 32, 32) 图像
    输出: list of (B, embed_dim, H_s, W_s) 特征，从粗到细

    下采样路径: 32→16→8→4→2→1，每步 strided conv + ResBlock。
    各尺度特征通过 1×1 conv 投影到 embed_dim。
    """
    def __init__(self, in_channels: int = 3, hidden_dim: int = 128,
                 embed_dim: int = 256, num_scales: int = 6):
        super().__init__()
        self.num_scales = num_scales

        # 初始投影: 3 → hidden_dim
        self.stem = nn.Conv2d(in_channels, hidden_dim, 3, padding=1, bias=False)

        # 下采样层: 每层 strided conv + ResBlock
        # 32→16→8→4→2→1 共 5 次下采样（对于 6 scales）
        self.down_blocks = nn.ModuleList()
        self.down_projs = nn.ModuleList()  # 各 scale 的特征投影

        ch = hidden_dim
        for i in range(num_scales - 1):  # 5 次下采样
            # 下采样: strided conv
            next_ch = min(ch * 2, 512)  # 通道数加倍，上限 512
            block = nn.Sequential(
                nn.Conv2d(ch, next_ch, 4, stride=2, padding=1, bias=False),
                ResBlock(next_ch),
            )
            self.down_blocks.append(block)
            ch = next_ch

        # 最粗 scale (1×1) 的特征投影
        self.coarsest_proj = nn.Conv2d(ch, embed_dim, 1, bias=False)

        # 各 scale 的特征投影 (从粗到细: 2×2, 4×4, ..., 32×32)
        # 在 decoder 侧通过 upsample + 残差计算各 scale 特征
        # encoder 只输出最粗特征，decoder 负责多尺度分解

    def forward(self, x):
        """
        参数:
          x: (B, 3, 32, 32) 图像 [0, 1]
        返回:
          h: (B, embed_dim, 1, 1) 最粗尺度特征
          intermediates: list of (B, ch, H, W) 中间特征（从细到粗）
        """
        h = self.stem(x)  # (B, hidden_dim, 32, 32)
        intermediates = [h]  # 保存各尺度中间特征，用于多尺度残差

        for block in self.down_blocks:
            h = block(h)
            intermediates.append(h)

        # intermediates: [32×32, 16×16, 8×8, 4×4, 2×2, 1×1] （从细到粗）
        h_coarse = self.coarsest_proj(h)  # (B, embed_dim, 1, 1)
        return h_coarse, intermediates


class VectorQuantizer(nn.Module):
    """
    Vector Quantizer with EMA codebook 更新。

    将连续特征量化到最近的 codebook embedding。
    使用 EMA 更新 codebook（不需要 codebook loss 的梯度）。

    参考:
      [1] van den Oord et al., "Neural Discrete Representation Learning,"
          NeurIPS 2017, arXiv:1711.00937 — VQ-VAE, Section 3
      [2] Razavi et al., "Generating Diverse High-Fidelity Images with VQ-VAE-2,"
          NeurIPS 2019 — EMA codebook 更新
    """
    def __init__(self, num_embeddings: int = 512, embedding_dim: int = 256,
                 commitment_weight: float = 0.25, decay: float = 0.99,
                 eps: float = 1e-5):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_weight = commitment_weight
        self.decay = decay
        self.eps = eps

        # Codebook embeddings
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / num_embeddings,
                         1.0 / num_embeddings)

        # EMA 统计量
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('_ema_w', self.embedding.weight.data.clone())

    def forward(self, z):
        """
        参数:
          z: (B, D, H, W) 连续特征
        返回:
          z_q: (B, D, H, W) 量化后特征（带 straight-through gradient）
          loss: commitment loss
          indices: (B, H, W) codebook indices
          codebook_usage: float — codebook 利用率（用于监控）
        """
        B, D, H, W = z.shape

        # (B, D, H, W) → (B*H*W, D)
        z_flat = z.permute(0, 2, 3, 1).reshape(-1, D)

        # 计算距离: ||z - e||^2 = ||z||^2 + ||e||^2 - 2*z·e
        # Ref: [1] Section 3, Eq.(2)
        d = (z_flat.pow(2).sum(dim=1, keepdim=True)
             + self.embedding.weight.pow(2).sum(dim=1)
             - 2 * z_flat @ self.embedding.weight.t())

        # 最近邻
        indices = d.argmin(dim=1)  # (B*H*W,)
        z_q = self.embedding(indices)  # (B*H*W, D)

        # EMA codebook 更新（仅训练时）
        if self.training:
            # 统计每个 codebook entry 被选中的次数和对应 z 的加权和
            encodings = F.one_hot(indices, self.num_embeddings).float()  # (N, K)
            self._ema_cluster_size.mul_(self.decay).add_(
                encodings.sum(0), alpha=1 - self.decay
            )
            dw = encodings.t() @ z_flat  # (K, D)
            self._ema_w.mul_(self.decay).add_(dw, alpha=1 - self.decay)

            # Laplace smoothing: 防止某些 codebook entry 永远不被使用
            n = self._ema_cluster_size.sum()
            cluster_size = (
                (self._ema_cluster_size + self.eps)
                / (n + self.num_embeddings * self.eps) * n
            )
            self.embedding.weight.data.copy_(self._ema_w / cluster_size.unsqueeze(1))

        # Commitment loss: 鼓励 encoder 输出靠近 codebook
        # Ref: [1] Eq.(3) — β * ||z_e - sg[e]||^2
        commitment_loss = self.commitment_weight * F.mse_loss(z_flat, z_q.detach())

        # Straight-through estimator: 前向用 z_q，反向梯度传给 z
        # Ref: [1] Section 3.1 — "copy gradients from decoder input to encoder output"
        z_q = z_flat + (z_q - z_flat).detach()

        # Reshape back
        z_q = z_q.reshape(B, H, W, D).permute(0, 3, 1, 2)
        indices = indices.reshape(B, H, W)

        return z_q, commitment_loss, indices


class Decoder(nn.Module):
    """
    多尺度 Decoder: 从量化特征重建图像。

    从最粗 scale (1×1) 开始，逐步上采样并融合各 scale 的量化残差。

    上采样路径: 1→2→4→8→16→32，每步 transposed conv + ResBlock。
    """
    def __init__(self, out_channels: int = 3, hidden_dim: int = 128,
                 embed_dim: int = 256, num_scales: int = 6):
        super().__init__()
        self.num_scales = num_scales

        # 通道数序列（从粗到细）: 与 encoder 对称
        channels = []
        ch = hidden_dim
        for i in range(num_scales - 1):
            ch = min(ch * 2, 512)
            channels.append(ch)
        channels = list(reversed(channels))  # 从粗到细

        # 最粗 scale 输入投影
        self.coarse_proj = nn.Conv2d(embed_dim, channels[0] if channels else hidden_dim,
                                     1, bias=False)

        # 上采样层 + 残差融合
        self.up_blocks = nn.ModuleList()
        self.residual_projs = nn.ModuleList()  # 各 scale 量化残差 → 当前通道数

        for i in range(num_scales - 1):
            in_ch = channels[i] if i < len(channels) else hidden_dim
            out_ch = channels[i + 1] if i + 1 < len(channels) else hidden_dim
            # 上采样 + ResBlock
            block = nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1, bias=False),
                ResBlock(out_ch),
            )
            self.up_blocks.append(block)
            # 残差投影: embed_dim → out_ch
            self.residual_projs.append(
                nn.Conv2d(embed_dim, out_ch, 1, bias=False)
            )

        # 最终输出: hidden_dim → 3
        final_ch = channels[-1] if channels else hidden_dim
        ng = min(32, final_ch)
        self.head = nn.Sequential(
            nn.GroupNorm(ng, final_ch),
            nn.SiLU(),
            nn.Conv2d(final_ch, out_channels, 3, padding=1),
        )

    def forward(self, z_scales):
        """
        参数:
          z_scales: list of (B, embed_dim, H_s, W_s) 量化特征（从粗到细）
                    z_scales[0] = 1×1, z_scales[1] = 2×2, ...
        返回:
          recon: (B, 3, 32, 32) 重建图像
        """
        h = self.coarse_proj(z_scales[0])  # (B, ch, 1, 1)

        for i, (up_block, res_proj) in enumerate(
                zip(self.up_blocks, self.residual_projs)):
            h = up_block(h)
            # 融合对应 scale 的量化残差
            if i + 1 < len(z_scales):
                residual = res_proj(z_scales[i + 1])
                h = h + residual

        recon = self.head(h)
        return recon


class MultiScaleVQVAE(nn.Module):
    """
    Multi-Scale VQ-VAE: 将图像编码为多尺度离散 token，并重建。

    工作流程:
      1. Encoder 提取多尺度特征
      2. 每个 scale 独立量化（各自 codebook）
      3. Decoder 从量化特征重建图像

    多尺度残差量化:
      - 最粗 scale (1×1): 直接量化 encoder 最深层特征
      - 后续 scale: 量化上一 scale 重建与当前 scale 特征的残差
      - 这样每个 scale 只需编码增量信息，降低编码冗余

    参考:
      [1] van den Oord et al., arXiv:1711.00937 — VQ-VAE
      [2] Tian et al., arXiv:2404.02905 — VAR multi-scale tokenizer
    """
    def __init__(
        self,
        image_size: int = 32,
        in_channels: int = 3,
        hidden_dim: int = 128,
        embed_dim: int = 256,
        num_embeddings: int = 512,
        num_scales: int = 6,
        commitment_weight: float = 0.25,
        ema_decay: float = 0.99,
    ):
        super().__init__()
        self.image_size = image_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_scales = num_scales
        self.num_embeddings = num_embeddings

        # scale 分辨率: [1, 2, 4, 8, 16, 32] for num_scales=6, image_size=32
        self.scale_sizes = [2**i for i in range(num_scales)]
        assert self.scale_sizes[-1] == image_size, (
            f"num_scales={num_scales} 的最大 scale ({self.scale_sizes[-1]}) "
            f"必须等于 image_size ({image_size})"
        )

        self.encoder = Encoder(in_channels, hidden_dim, embed_dim, num_scales)
        self.decoder = Decoder(in_channels, hidden_dim, embed_dim, num_scales)

        # 每个 scale 独立 VectorQuantizer
        self.quantizers = nn.ModuleList([
            VectorQuantizer(num_embeddings, embed_dim,
                           commitment_weight, ema_decay)
            for _ in range(num_scales)
        ])

        # 各 scale 的特征投影: 从 encoder 中间特征 → embed_dim
        # encoder intermediates 的通道数从细到粗递增
        ch = hidden_dim
        proj_channels = [hidden_dim]  # 32×32 = hidden_dim
        for i in range(num_scales - 1):
            ch = min(ch * 2, 512)
            proj_channels.append(ch)
        # proj_channels: [hidden_dim, hidden_dim*2, ..., 最粗通道数]
        # 从细到粗排列

        self.scale_projs = nn.ModuleList()
        for i, pc in enumerate(proj_channels):
            self.scale_projs.append(
                nn.Conv2d(pc, embed_dim, 1, bias=False)
            )

    def encode(self, x, compute_usage: bool = False):
        """
        编码图像为多尺度离散 token indices。

        参数:
          x: (B, 3, 32, 32) 图像 [0, 1]
          compute_usage: 是否统计 codebook 利用率（触发 .unique() GPU→CPU 同步，
                         训练循环中默认关闭）
        返回:
          indices_list: list of (B, H_s, W_s) token indices，从粗到细
          z_q_list: list of (B, embed_dim, H_s, W_s) 量化特征
          total_vq_loss: scalar — 所有 scale 的 commitment loss 之和
          avg_usage: float — 平均 codebook 利用率（compute_usage=False 时为 0.0）
        """
        _, intermediates = self.encoder(x)
        intermediates = list(reversed(intermediates))

        indices_list = []
        z_q_list = []
        total_vq_loss = 0.0
        total_usage = 0.0

        for i in range(self.num_scales):
            z = self.scale_projs[self.num_scales - 1 - i](intermediates[i])
            target_size = self.scale_sizes[i]
            if z.shape[2] != target_size or z.shape[3] != target_size:
                z = F.adaptive_avg_pool2d(z, (target_size, target_size))

            z_q, vq_loss, indices = self.quantizers[i](z)

            indices_list.append(indices)
            z_q_list.append(z_q)
            total_vq_loss = total_vq_loss + vq_loss
            if compute_usage:
                total_usage += len(indices.unique()) / self.quantizers[i].num_embeddings

        avg_usage = total_usage / self.num_scales if compute_usage else 0.0
        return indices_list, z_q_list, total_vq_loss, avg_usage

    def decode(self, z_q_list):
        """
        从量化特征重建图像。

        参数:
          z_q_list: list of (B, embed_dim, H_s, W_s) 量化特征（从粗到细）
        返回:
          recon: (B, 3, 32, 32)
        """
        return self.decoder(z_q_list)

    def decode_from_indices(self, indices_list):
        """
        从 token indices 重建图像（推理用）。

        参数:
          indices_list: list of (B, H_s, W_s) token indices（从粗到细）
        返回:
          recon: (B, 3, 32, 32)
        """
        z_q_list = []
        for i, indices in enumerate(indices_list):
            B, H, W = indices.shape
            z_q = self.quantizers[i].embedding(indices)  # (B, H, W, D)
            z_q = z_q.permute(0, 3, 1, 2)  # (B, D, H, W)
            z_q_list.append(z_q)
        return self.decode(z_q_list)

    def forward(self, x, compute_usage: bool = False):
        """
        参数:
          x: (B, 3, 32, 32) 图像 [0, 1]
          compute_usage: 是否统计 codebook 利用率（默认 False，避免训练 hot-path 同步）
        返回:
          dict:
            loss: total loss (recon + vq)
            recon_loss: reconstruction loss (MSE)
            vq_loss: commitment loss
            recon: (B, 3, 32, 32) 重建图像
            indices: list of token indices per scale
            codebook_usage: 平均 codebook 利用率（compute_usage=False 时为 0.0）
        """
        x = x.clamp(0, 1)
        indices_list, z_q_list, vq_loss, avg_usage = self.encode(x, compute_usage=compute_usage)
        recon = self.decode(z_q_list)

        # Reconstruction loss: MSE
        recon_loss = F.mse_loss(recon, x)

        # Total loss
        loss = recon_loss + vq_loss

        return {
            "loss": loss,
            "recon_loss": recon_loss,
            "vq_loss": vq_loss,
            "recon": recon,
            "indices": indices_list,
            "codebook_usage": avg_usage,
        }

    def get_num_tokens(self):
        """返回各 scale 的 token 数量。"""
        return [s * s for s in self.scale_sizes]

    def get_total_tokens(self):
        """返回所有 scale 的总 token 数量。"""
        return sum(self.get_num_tokens())
