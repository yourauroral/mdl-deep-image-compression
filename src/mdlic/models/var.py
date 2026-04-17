"""
VAR Transformer — Next-Scale Prediction 自回归模型。

基于现有 iGPT 的 GPTBlock 组件，实现 VAR (Visual Autoregressive Modeling)
的 next-scale prediction 范式。输入多尺度 VQVAE token，从粗到细逐 scale 预测。

核心区别 vs iGPT:
  - iGPT: next-token prediction，1D causal mask
  - VAR: next-scale prediction，block-causal mask
    - Scale 内: BIDIRECTIONAL（同 scale 所有 token 互相可见）
    - 跨 Scale: CAUSAL（只能看到更粗的 scale）

复用资产:
  - GPTBlock（RMSNorm, SwiGLU, RoPE, QK-Norm, post-norm）
  - Fused Triton kernels: RMSNorm, SwiGLU, Fused RoPE, CE+z-loss, Add+RMSNorm
  - Weight Tying, position_ids 机制

不复用（hardcoded causal）:
  - Triton Flash Attention, Fused Attn+RoPE
  → 通过 attn_mask 参数绕过，走 PyTorch SDPA

参考:
  [1] Tian et al., "Visual Autoregressive Modeling: Scalable Image Generation
      via Next-Scale Prediction," NeurIPS 2024, arXiv:2404.02905
  [2] OLMo 2 Tech Report, arXiv:2501.00656 — post-norm 架构
  [3] Su et al., "RoFormer," arXiv:2104.09864 — RoPE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .layers import GPTBlock, RMSNorm

# Fused CE + z-loss（复用 iGPT 的 kernel）
try:
    from ..ops.fused_ce_zloss import fused_cross_entropy_zloss as _fused_ce_zloss
    _USE_FUSED_CE = True
except ImportError:
    _fused_ce_zloss = None
    _USE_FUSED_CE = False


def build_block_causal_mask(scale_sizes, device):
    """
    构建 VAR block-causal attention mask。

    Scale 内: BIDIRECTIONAL（0.0 允许）
    跨 Scale: 粗→细 CAUSAL（下三角 0.0 允许，上三角 -inf 屏蔽）

    例如 scale_sizes = [1, 4, 16, 64, 256, 1024]:
      token 0       = scale 0 (1×1)
      token 1-4     = scale 1 (2×2)
      token 5-20    = scale 2 (4×4)
      ...

    返回 (total_tokens, total_tokens) float tensor。

    参考:
      [1] Tian et al., arXiv:2404.02905, Section 3.2 — block-causal mask
    """
    total = sum(scale_sizes)
    # 初始化为 -inf（全部屏蔽）
    mask = torch.full((total, total), float('-inf'), device=device)

    # 计算每个 scale 的起止位置
    boundaries = []
    offset = 0
    for s in scale_sizes:
        boundaries.append((offset, offset + s))
        offset += s

    # 填充: scale i 的 token 可以看到 scale j (j <= i) 的所有 token
    for i, (start_i, end_i) in enumerate(boundaries):
        for j, (start_j, end_j) in enumerate(boundaries):
            if j <= i:
                # 允许: scale i 可以看到 scale j（含自身）
                mask[start_i:end_i, start_j:end_j] = 0.0

    return mask


class VARTransformer(nn.Module):
    """
    VAR Transformer: Next-Scale Prediction。

    输入 VQVAE 多尺度 token indices，预测下一个 scale 的 token。
    Scale 0 (1×1) 用 learnable start token 作为输入。

    架构:
      - Token embedding: VQVAE codebook size 作为 vocab
      - Scale embedding: 可学习，标识 token 所属 scale
      - Position embedding: RoPE + position_ids（各 scale 内部独立编号）
      - Transformer: 复用 GPTBlock，attn_mask = block-causal
      - Output head: 预测 codebook index（CE loss）

    MDL 计算:
      L(data|model) = 各 scale CE loss 之和 (bits)
      L(model) = 模型参数量 × 编码代价
      MDL = L(model) + L(data|model)
    """
    def __init__(
        self,
        num_embeddings: int = 512,    # VQVAE codebook 大小 = vocab size
        d_model: int = 512,
        N: int = 12,                  # Transformer 层数
        h: int = 8,                   # 注意力头数
        d_ff: int = 1376,             # SwiGLU hidden dim
        dropout: float = 0.0,
        num_scales: int = 6,          # 尺度数
        image_size: int = 32,         # 目标图像分辨率
        # 消融开关（与 iGPT 一致）
        use_rope: bool = True,
        use_post_norm: bool = True,
        use_swiglu: bool = True,
        use_qk_norm: bool = True,
        use_depth_scaled_init: bool = True,
        use_zloss: bool = True,
        activation_checkpointing: bool = False,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.d_model = d_model
        self.N_layers = N
        self.num_scales = num_scales
        self.use_rope = use_rope
        self.use_zloss = use_zloss
        self.use_post_norm = use_post_norm
        self.use_depth_scaled_init = use_depth_scaled_init

        # Scale 分辨率: [1, 2, 4, 8, 16, 32]
        self.scale_resolutions = [2**i for i in range(num_scales)]
        # 每个 scale 的 token 数: [1, 4, 16, 64, 256, 1024]
        self.scale_sizes = [r * r for r in self.scale_resolutions]
        self.total_tokens = sum(self.scale_sizes)

        # Token embedding: VQVAE codebook index → d_model
        self.token_embed = nn.Embedding(num_embeddings, d_model)

        # Scale embedding: 标识每个 token 属于哪个 scale
        # Ref: [1] Section 3.2 — scale-aware positional encoding
        self.scale_embed = nn.Embedding(num_scales, d_model)

        # Learnable start token: 用于 scale 0 (1×1) 的输入
        # Scale 0 没有前序 scale 可条件化，用 learnable embedding 代替
        self.start_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Learned positional embedding fallback（use_rope=False 时）
        if not use_rope:
            self.pos_embed = nn.Embedding(self.total_tokens, d_model)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            GPTBlock(d_model, h, d_ff, dropout,
                     use_post_norm=use_post_norm,
                     use_swiglu=use_swiglu,
                     use_qk_norm=use_qk_norm,
                     use_rope=use_rope,
                     activation_checkpointing=activation_checkpointing)
            for _ in range(N)
        ])

        # Final norm: 仅 pre-norm 模式需要
        if not use_post_norm:
            self.final_norm = RMSNorm(d_model)

        # Output head: 预测 codebook index
        self.head = nn.Linear(d_model, num_embeddings, bias=False)
        # Weight Tying
        # Ref: Press & Wolf, arXiv:1608.05859
        self.head.weight = self.token_embed.weight

        # 缓存 block-causal mask（避免每次 forward 重建）
        self._mask_cache = None

        self._init_weights()

    def _init_weights(self):
        """权重初始化: 与 iGPT 保持一致。"""
        N = self.N_layers
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if name.endswith('head'):
                    continue  # weight tying, 跳过
                if self.use_depth_scaled_init and any(
                        name.endswith(s) for s in ('w_o', 'w2')):
                    std = 0.02 / math.sqrt(2 * N)
                else:
                    std = 0.02
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _get_block_causal_mask(self, device):
        """获取/缓存 block-causal mask。"""
        if (self._mask_cache is None
                or self._mask_cache.device != device):
            self._mask_cache = build_block_causal_mask(
                self.scale_sizes, device)
        return self._mask_cache

    def _build_position_ids(self, device):
        """
        构建 position_ids: 每个 scale 内部独立编号。

        Scale 0 (1 token):  [0]
        Scale 1 (4 tokens): [0, 1, 2, 3]
        Scale 2 (16 tokens): [0, 1, ..., 15]
        ...

        这样 RoPE 编码的是 scale 内部的空间位置关系。
        """
        ids = []
        for s in self.scale_sizes:
            ids.append(torch.arange(s, device=device))
        return torch.cat(ids)  # (total_tokens,)

    def _build_scale_ids(self, device):
        """构建 scale_ids: 标识每个 token 的 scale 编号。"""
        ids = []
        for i, s in enumerate(self.scale_sizes):
            ids.append(torch.full((s,), i, device=device, dtype=torch.long))
        return torch.cat(ids)  # (total_tokens,)

    def forward(self, indices_list, z_loss_weight: float = 1e-4,
                compute_per_scale: bool = False):
        """
        参数:
          indices_list: list of (B, H_s, W_s) token indices，从粗到细
                        len = num_scales
          z_loss_weight: z-loss 权重
          compute_per_scale: 是否逐 scale 计算 CE（会触发 .item() 同步，
                             训练时仅 log step 传 True）

        返回:
          dict:
            loss: total loss
            ce_loss: CE loss（不含 z-loss，用于 BPP 计算）
            per_scale_loss: list of per-scale CE loss（compute_per_scale=False 时为 None）
        """
        z_loss_weight = float(z_loss_weight)
        device = indices_list[0].device
        B = indices_list[0].shape[0]

        # 1. 拼接所有 scale 的 token indices → (B, total_tokens)
        flat_indices = []
        for idx in indices_list:
            flat_indices.append(idx.reshape(B, -1))  # (B, H_s*W_s)
        all_tokens = torch.cat(flat_indices, dim=1)  # (B, total_tokens)

        # 2. 构建输入 (teacher forcing):
        #    对于 next-scale prediction，scale k 的输入是 scale 0..k-1 的 token。
        #    但在 block-causal mask 下，我们可以一次性输入所有 token，
        #    mask 保证 scale k 只能看到 scale 0..k 的 token。
        #    预测目标: 每个位置预测自身的 token（利用 block-causal mask 的因果性）。
        #
        #    具体来说:
        #    - Scale 0 的 token 输入 start_token（无条件）
        #    - Scale k (k>0) 的 token 看到 scale 0..k-1 的真实 token + scale k 内部互看
        #    - 每个位置预测自身的 codebook index

        # Token embedding
        token_emb = self.token_embed(all_tokens)  # (B, total_tokens, d_model)

        # 替换 scale 0 的输入为 start_token
        # scale 0 有 scale_sizes[0] = 1 个 token
        s0 = self.scale_sizes[0]
        start = self.start_token.expand(B, s0, -1)  # (B, 1, d_model)
        hidden = torch.cat([start, token_emb[:, s0:]], dim=1)  # (B, total_tokens, d_model)

        # Scale embedding
        scale_ids = self._build_scale_ids(device)
        hidden = hidden + self.scale_embed(scale_ids).unsqueeze(0)

        # Position embedding
        position_ids = None
        if self.use_rope:
            position_ids = self._build_position_ids(device)
        else:
            positions = torch.arange(self.total_tokens, device=device)
            hidden = hidden + self.pos_embed(positions)

        # Block-causal mask
        attn_mask = self._get_block_causal_mask(device)

        # Transformer forward
        for block in self.blocks:
            hidden = block(hidden, mask=None, position_ids=position_ids,
                          attn_mask=attn_mask)

        # Final norm (pre-norm 模式)
        if not self.use_post_norm:
            hidden = self.final_norm(hidden)

        # 3. 计算 loss: 每个 scale 的 CE loss
        #    Scale 0: 用 start_token 的输出预测 scale 0 的 token
        #    Scale k: 用 scale k 位置的输出预测 scale k 的 token
        #    （block-causal mask 保证 scale k 只看到了 scale 0..k-1 的信息，
        #     加上 scale k 内部的双向信息，但 scale k 自身 token 作为 target）
        z_w = z_loss_weight if self.use_zloss else 0.0

        logits = self.head(hidden).float()  # (B, total_tokens, num_embeddings)
        targets = all_tokens  # (B, total_tokens)

        # 总 CE loss
        ce_loss = F.cross_entropy(
            logits.reshape(-1, self.num_embeddings),
            targets.reshape(-1),
            reduction="mean"
        )

        if self.use_zloss and z_w > 0:
            log_z = torch.logsumexp(logits, dim=-1)
            z_loss = (log_z ** 2).mean()
            loss = ce_loss + z_w * z_loss
        else:
            loss = ce_loss

        # Per-scale CE loss（.item() 同步较贵，仅在日志步触发）
        per_scale_loss = None
        if compute_per_scale:
            per_scale_loss = []
            offset = 0
            for i, s in enumerate(self.scale_sizes):
                scale_logits = logits[:, offset:offset + s]
                scale_targets = targets[:, offset:offset + s]
                scale_ce = F.cross_entropy(
                    scale_logits.reshape(-1, self.num_embeddings),
                    scale_targets.reshape(-1),
                    reduction="mean"
                )
                per_scale_loss.append(scale_ce.item())
                offset += s

        return {
            "loss": loss,
            "ce_loss": ce_loss,
            "per_scale_loss": per_scale_loss,
            "logits": logits,
        }

    @torch.no_grad()
    def generate(self, batch_size: int = 1, device=None, temperature: float = 1.0,
                 top_k: int = 0):
        """
        自回归生成: 从粗到细逐 scale 采样。

        VAR 的生成方式:
          1. Scale 0: 用 start_token 作为输入，预测 1 个 token
          2. Scale k (k>0): 将前序 scale 的真实 token + 当前 scale 的 placeholder
             一起输入 Transformer。block-causal mask 保证当前 scale 看到
             所有前序 scale + 当前 scale 内部互看。
             从当前 scale 位置的 logits 采样所有 token（parallel）。

        参数:
          batch_size: 生成样本数
          device: 设备
          temperature: 采样温度
          top_k: top-k 采样（0 = 不限制）

        返回:
          indices_list: list of (B, H_s, W_s) 生成的 token indices
        """
        if device is None:
            device = next(self.parameters()).device

        B = batch_size
        generated_indices = []  # list of (B, s) per scale

        for scale_idx in range(self.num_scales):
            s = self.scale_sizes[scale_idx]
            r = self.scale_resolutions[scale_idx]

            # 构建到当前 scale (含) 的完整输入序列
            if scale_idx == 0:
                # 只有 start_token
                hidden = self.start_token.expand(B, 1, -1)  # (B, 1, d_model)
                # Scale/position embedding for start_token
                scale_id = torch.zeros(1, device=device, dtype=torch.long)
                hidden = hidden + self.scale_embed(scale_id).unsqueeze(0)
                pos_ids = torch.zeros(1, device=device, dtype=torch.long) if self.use_rope else None
                # Mask: 1×1，全部允许
                mask = torch.zeros((1, 1), device=device)
            else:
                # 拼接: 前序 scale 真实 token + 当前 scale placeholder (零向量)
                prev_tokens = torch.cat(generated_indices, dim=1)  # (B, prev_total)
                prev_emb = self.token_embed(prev_tokens)  # (B, prev_total, d_model)

                # 当前 scale 用零向量作为 placeholder
                # (block-causal mask 让它们看到前序 scale 和彼此，但不提供自身信息)
                cur_placeholder = torch.zeros(B, s, self.d_model, device=device)
                hidden = torch.cat([prev_emb, cur_placeholder], dim=1)

                # Scale embedding
                all_scale_ids = []
                all_pos_ids = []
                for si in range(scale_idx + 1):
                    ss = self.scale_sizes[si]
                    all_scale_ids.append(
                        torch.full((ss,), si, device=device, dtype=torch.long))
                    all_pos_ids.append(torch.arange(ss, device=device))

                all_scale_ids = torch.cat(all_scale_ids)
                all_pos_ids = torch.cat(all_pos_ids)

                hidden = hidden + self.scale_embed(all_scale_ids).unsqueeze(0)

                if self.use_rope:
                    pos_ids = all_pos_ids
                else:
                    total_len = hidden.shape[1]
                    hidden = hidden + self.pos_embed(
                        torch.arange(total_len, device=device)).unsqueeze(0)
                    pos_ids = None

                # Block-causal mask (含当前 scale)
                cur_scales = self.scale_sizes[:scale_idx + 1]
                mask = build_block_causal_mask(cur_scales, device)

            # Forward through transformer
            h = hidden
            for block in self.blocks:
                h = block(h, mask=None,
                         position_ids=pos_ids if self.use_rope else None,
                         attn_mask=mask)

            if not self.use_post_norm:
                h = self.final_norm(h)

            # 取当前 scale 位置的 logits
            if scale_idx == 0:
                cur_logits = self.head(h).float()  # (B, 1, V)
            else:
                cur_logits = self.head(h[:, -s:]).float()  # (B, s, V)

            # Temperature + top-k sampling
            logits_scaled = cur_logits / max(temperature, 1e-8)
            if top_k > 0:
                topk_vals, _ = logits_scaled.topk(top_k, dim=-1)
                logits_scaled[logits_scaled < topk_vals[..., -1:]] = float('-inf')

            probs = F.softmax(logits_scaled, dim=-1)
            sampled = torch.multinomial(probs.reshape(-1, self.num_embeddings),
                                        num_samples=1)
            sampled = sampled.reshape(B, -1)  # (B, s)
            generated_indices.append(sampled)

        # Reshape 回各 scale 的空间形状
        indices_list = []
        for i, (s, r) in enumerate(zip(self.scale_sizes, self.scale_resolutions)):
            indices_list.append(generated_indices[i].reshape(B, r, r))

        return indices_list
