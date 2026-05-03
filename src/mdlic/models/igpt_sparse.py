import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .layers import GPTBlock, RMSNorm
from .igpt import rgb_to_ycbcr_int

# Fused CE + z-loss（可选）
try:
    from ..ops.fused_ce_zloss import fused_cross_entropy_zloss as _fused_ce_zloss
    _USE_FUSED_CE = True
except ImportError:
    _fused_ce_zloss = None
    _USE_FUSED_CE = False

# Fused Linear + CE + z-loss（可选）
try:
    from ..ops.fused_linear_ce import fused_linear_cross_entropy as _fused_linear_ce
    _USE_FUSED_LINEAR_CE = True
except ImportError:
    _fused_linear_ce = None
    _USE_FUSED_LINEAR_CE = False


class SparseIGPT(nn.Module):
    """
    iGPT + Sparse Transformer 稀疏注意力。

    在 iGPT-S 基础上引入 Child et al. (2019) 的交替稀疏注意力模式：
      - 偶数层（Pattern A）：局部窗口，token i 注意 [i-l, i] 内的前驱
      - 奇数层（Pattern B）：步幅注意力，token i 注意所有 (i-j) % l == 0 的前驱

    两种模式交替覆盖完整因果上下文，使序列前期 token 也能获得全局上下文，
    从而降低早期 token 的预测困难度，改善整体 BPP。

    实现上利用 MultiHeadAttentionBlock 已有的 attn_mask 路径（提供显式 mask 时
    绕过 Triton Flash Attn，走 PyTorch SDPA），mask 在 __init__ 构建一次并注册为
    non-persistent buffer，不存入 checkpoint。

    参考:
      [1] Child et al., "Generating Long Sequences with Sparse Transformers,"
          arXiv:1904.10509, 2019. Section 3.3 — strided attention for images.
          CIFAR-10 结果: 2.80 bits/dim (vs iGPT-S baseline 2.97)
      [2] Chen et al., "Generative Pretraining from Pixels," ICML 2020.
          iGPT 原始架构。
    """

    def __init__(
        self,
        image_size: int = 32,
        in_channels: int = 3,
        vocab_size: int = 256,
        d_model: int = 512,
        N: int = 24,
        h: int = 8,
        d_ff: int = 1376,
        dropout: float = 0.0,
        # 消融开关（与 igpt.py 保持一致）
        use_ycbcr: bool = True,
        use_rope: bool = True,
        use_post_norm: bool = True,
        use_swiglu: bool = True,
        use_qk_norm: bool = True,
        use_depth_scaled_init: bool = True,
        use_zloss: bool = True,
        activation_checkpointing: bool = False,
        use_subpixel_ar: bool = False,
        # 稀疏注意力参数
        sparse_stride: int = 128,
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
        self.use_post_norm = use_post_norm
        self.sparse_stride = sparse_stride

        self.token_embed = nn.Embedding(vocab_size, d_model)

        if use_subpixel_ar:
            self.channel_embed = nn.Embedding(in_channels, d_model)

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

        if not use_post_norm:
            self.final_norm = RMSNorm(d_model)

        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.token_embed.weight

        self._init_weights()

        # 预计算稀疏注意力掩码，注册为 non-persistent buffer（不存入 checkpoint）
        # Ref: Child et al., arXiv:1904.10509, Section 3.3, Figure 3
        T = self.seq_len - 1  # 输入序列长度（NTP: x[0..T-1] → x[1..T]）
        assert 0 < sparse_stride <= T, (
            f"sparse_stride={sparse_stride} 超出有效范围 (0, {T}]"
        )
        local_mask, strided_mask = self._build_sparse_masks(T, sparse_stride)
        self.register_buffer('_sparse_local_mask', local_mask, persistent=False)
        self.register_buffer('_sparse_strided_mask', strided_mask, persistent=False)

    @staticmethod
    def _build_sparse_masks(seq_len: int, stride: int):
        """
        构建稀疏注意力掩码对。

        Pattern A（局部窗口）: token i 注意 [i-stride, i] 内的所有前驱。
        Pattern B（步幅）: token i 注意所有满足 (i-j) % stride == 0 的前驱。

        两种模式均包含因果约束（j <= i），合并后等价于覆盖完整因果上下文。

        返回: (local_mask, strided_mask)，均为 (seq_len, seq_len) float32，
              0.0 = 允许注意，-inf = 屏蔽。

        Ref: Child et al., arXiv:1904.10509, 2019. Section 3.3.
        """
        i = torch.arange(seq_len).unsqueeze(1)   # (T, 1)
        j = torch.arange(seq_len).unsqueeze(0)   # (1, T)
        causal = j <= i                           # 因果约束

        # Pattern A: 局部窗口 [i-stride, i]
        local_window = (i - j) < stride
        local_mask = torch.where(
            causal & local_window,
            torch.zeros(1),
            torch.full((1,), float('-inf'))
        )

        # Pattern B: 步幅，(i-j) 是 stride 的整数倍
        strided_col = (i - j) % stride == 0
        strided_mask = torch.where(
            causal & strided_col,
            torch.zeros(1),
            torch.full((1,), float('-inf'))
        )

        return local_mask, strided_mask

    def _init_weights(self):
        """
        权重初始化：基础 std=0.02，残差输出投影用深度缩放（可选）。
        Ref: Radford et al., "GPT-2," 2019; OLMo 2 arXiv:2501.00656.
        """
        N = self.N_layers
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
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
        muP 初始化。
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

    def _tokenize(self, x: torch.Tensor):
        """RGB [0,1] → token 序列，返回 (B, seq_len) long。"""
        x = x.clamp(0, 1)
        B = x.size(0)
        if self.use_ycbcr:
            x = rgb_to_ycbcr_int(x)
        else:
            x = (x * 255).round().long()
        if self.use_subpixel_ar:
            # pixel-first: [Y0,Cb0,Cr0, Y1,Cb1,Cr1, ...]
            x = x.permute(0, 2, 3, 1).reshape(B, -1)
        else:
            x = x.reshape(B, -1)
        return x

    def _embed(self, input_tokens: torch.Tensor):
        """token → hidden，附加 channel embed 和 position embed。"""
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

        return hidden, position_ids

    def forward(self, x: torch.Tensor, z_loss_weight: float = 1e-4):
        """
        参数:
          x:             (B, C, H, W) float [0,1]
          z_loss_weight: z-loss 权重，默认 1e-4
        """
        z_loss_weight = float(z_loss_weight)
        tokens = self._tokenize(x)
        input_tokens = tokens[:, :-1]
        target_tokens = tokens[:, 1:]

        hidden, position_ids = self._embed(input_tokens)

        # 交替稀疏注意力：偶数层局部窗口，奇数层步幅
        # Ref: Child et al., arXiv:1904.10509, Section 3.3
        for idx, block in enumerate(self.blocks):
            attn_mask = (self._sparse_local_mask if idx % 2 == 0
                         else self._sparse_strided_mask)
            # 与 hidden dtype 对齐，避免 AMP 训练时 SDPA 内部 upcasting
            attn_mask = attn_mask.to(device=hidden.device, dtype=hidden.dtype)
            hidden = block(hidden, mask=None, position_ids=position_ids,
                           attn_mask=attn_mask)

        if not self.use_post_norm:
            hidden = self.final_norm(hidden)

        z_w = z_loss_weight if self.use_zloss else 0.0

        if _USE_FUSED_LINEAR_CE and hidden.is_cuda and z_w > 0:
            ce_loss, z_loss = _fused_linear_ce(
                hidden.reshape(-1, self.d_model),
                self.head.weight,
                target_tokens.reshape(-1),
                z_loss_weight=z_w,
            )
            loss = ce_loss + z_w * z_loss
            logits = None
        else:
            logits = self.head(hidden).float()
            if _USE_FUSED_CE and logits.is_cuda and z_w > 0:
                ce_loss, z_loss = _fused_ce_zloss(
                    logits.reshape(-1, self.vocab_size),
                    target_tokens.reshape(-1),
                    z_loss_weight=z_w,
                )
                loss = ce_loss + z_w * z_loss
            else:
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
            "logits": logits,
        }

    @torch.no_grad()
    def encode(self, x: torch.Tensor, max_layer: int = None, pool: bool = False):
        """
        提取各层表征，用于 linear probe。

        参数:
          pool: True 时对每层输出做 GAP 并转 CPU，返回 (B, d_model)。
        返回:
          list[Tensor]，长度为 max_layer+1。
        """
        tokens = self._tokenize(x)
        input_tokens = tokens[:, :-1]
        hidden, position_ids = self._embed(input_tokens)

        if max_layer is None:
            max_layer = len(self.blocks) - 1

        outputs = []
        for idx, block in enumerate(self.blocks):
            attn_mask = (self._sparse_local_mask if idx % 2 == 0
                         else self._sparse_strided_mask)
            attn_mask = attn_mask.to(device=hidden.device, dtype=hidden.dtype)
            hidden = block(hidden, mask=None, position_ids=position_ids,
                           attn_mask=attn_mask)
            if pool:
                outputs.append(hidden.float().mean(dim=1).cpu())
            else:
                outputs.append(hidden)
            if idx >= max_layer:
                break
        return outputs
