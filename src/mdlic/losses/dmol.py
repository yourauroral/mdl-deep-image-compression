"""离散化 logistic 混合 (Discretized Mixture of Logistics) 损失实现。

每个 sub-pixel token 独立产 K-mix 1D logistic 分布参数 (logit_w, mean, log_scale)，
**无通道耦合**：与本仓库 sub-pixel AR pixel-first 序列布局对齐，通道依赖
通过 attention 而非输出头建模。

域：target ∈ [-1, 1] float (PixelCNN++ 原版同款)。
  半 bin 宽 = 1/255 ≈ 0.0039；mean = tanh(mean_raw) ∈ [-1, 1]。
  下游 IGPT 调用方负责把 long ∈ [0, 255] → float ∈ [-1, 1] 归一化 (t/127.5 - 1)。
  历史 v6 第 1 次尝试用 [0, 255] long 域 (bin=0.5, mean ∈ [0,255], bias=+2)，
  在 Stage 2 80M 主网上 step 650 后 CE_f 仍 6.5 不动；boundary token
  (target-mean)·inv_s 量级 ~127·0.135 ≈ 17 在 sigmoid 完全饱和。

Refs:
  Salimans et al., "PixelCNN++," ICLR 2017 — 离散化 logistic 混合定义 + [-1,1] 域 + 1/255 半 bin
  Chen et al., "Generating Long Sequences with Sparse Transformers," 2019 — DMoL 用于 transformer 输出头
  Chen et al., "Generative Pretraining from Pixels (iGPT)," ICML 2020 — std=0.005 head init
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# 数值护栏：v1-v5 失败的核心教训。详细映射见 future.md §8.3
_LOG_SCALE_MIN = -7.0       # inv_s ≤ exp(7) ≈ 1097（防 sigmoid 完全饱和）
_LOG_SCALE_MAX = +5.0       # inv_s ≥ exp(-5) ≈ 0.0067（防 mean·inv_s 量级失控）
_LOG_PROB_FLOOR = -30.0     # last-resort 单 token log_prob 下限
_CDF_DIFF_FLOOR = 1e-12     # CDF 差分极小值兜底（v1 用 1e-5 太松，1e-12 仅 catch 真溢出）

# 半 bin 宽：256 个离散 bin 均匀映到 [-1, 1] 区间 → bin 宽 = 2/255, 半 bin = 1/255
_HALF_BIN = 1.0 / 255.0
# fp32 中 log(255) 常量，用作 fallback 时的 Jacobian 补偿项 log(bin_width/2)/2 = -log(127.5)
_LOG_BIN_HALF_WIDTH = -math.log(127.5)


class DMoLHead1D(nn.Module):
    """每 sub-pixel token 独立的 K-mix 离散化 logistic head。

    输出维度 = K * 3，对应 (logit_w[K], mean_raw[K], log_scale_raw[K])。
    无 weight tying（output 端独立参数；输入 token_embed 仍是 256-way categorical）。
    """

    def __init__(self, d_model: int, n_mixtures: int = 10):
        super().__init__()
        self.n_mixtures = n_mixtures
        self.proj = nn.Linear(d_model, n_mixtures * 3, bias=True)
        self._init_weights()

    def _init_weights(self):
        # OpenAI iGPT (Chen 2020) 输出 head std=0.005 抑制初始 logits 方差，
        # DMoL CDF 差分对此尤敏感（std=0.02 会让 sigmoid 起步即饱和）。
        nn.init.normal_(self.proj.weight, std=0.005)
        nn.init.zeros_(self.proj.bias)
        # log_scale bias = 0 (PixelCNN++ 原版同款；inv_s = 1)。
        # 在 [-1, 1] 域下 bin·inv_s = 1/255 ≈ 0.0039 与 boundary (t-mean)·inv_s ~ 2
        # 都落在 sigmoid 线性区。v6 第 1 次 (旧版 [0,255] 域) bias=+2 是为补偿 bin=0.5
        # 量级偏大；切到 [-1,1] 后 bias 不再需要预偏置。

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """hidden: (B, T, d_model) → params: (B, T, K*3)。"""
        return self.proj(hidden)


def dmol_loss_1d(params: torch.Tensor,
                 target_long: torch.Tensor,
                 n_mixtures: int,
                 reduction: str = "mean") -> torch.Tensor:
    """1D 离散化 logistic 混合 NLL（[-1, 1] 域内部计算）。

    Args:
        params:       (B, T, K*3) float — DMoLHead1D 输出
        target_long:  (B, T) long ∈ [0, 255] — sub-pixel 整数 target
                      （调用方传 long，内部归一化到 [-1, 1] float）
        n_mixtures:   K
        reduction:    "mean" 返回标量；"none" 返回 (B, T) per-token NLL

    Returns:
        NLL（nat / sub-pixel），与 categorical CE 同口径，下游 bpd = NLL / ln2 不变
    """
    K = n_mixtures
    assert params.size(-1) == 3 * K, (
        f"params last dim {params.size(-1)} != 3*K = {3 * K}"
    )

    # split: (B, T, K) × 3
    logit_w, mean_raw, log_scale_raw = params.split(K, dim=-1)

    # mean ∈ [-1, 1]（PixelCNN++ 原版；无通道耦合，pixel-first AR 已通过 attention 建模 R/G/B 依赖）
    mean = torch.tanh(mean_raw)

    # log_scale 双向 clamp，inv_s 范围 [0.0067, 1097]
    log_scale = log_scale_raw.clamp(_LOG_SCALE_MIN, _LOG_SCALE_MAX)
    inv_s = torch.exp(-log_scale)

    # target long ∈ [0, 255] → float ∈ [-1, 1]，PixelCNN++ 原版同款归一化：
    #   t_float = t_long / 127.5 - 1
    # 广播到 mixture 维: (B, T) → (B, T, 1)
    target = (target_long.to(mean.dtype) / 127.5 - 1.0).unsqueeze(-1)

    # 离散 bin 边界（[-1,1] 域，半 bin 宽 = 1/255）
    plus_in = (target + _HALF_BIN - mean) * inv_s
    minus_in = (target - _HALF_BIN - mean) * inv_s

    # 中间 bin: log(σ(plus_in) - σ(minus_in))
    cdf_plus = torch.sigmoid(plus_in)
    cdf_minus = torch.sigmoid(minus_in)
    cdf_delta = (cdf_plus - cdf_minus).clamp(min=_CDF_DIFF_FLOOR)
    log_prob_mid = torch.log(cdf_delta)

    # 边界 target=0   (t_float=-1):    log σ(plus_in)        （概率全在 (-inf, -1+1/255]）
    # 边界 target=255 (t_float=+1):    log σ(-minus_in)      （概率全在 [+1-1/255, +inf)）
    log_prob_low = F.logsigmoid(plus_in)
    log_prob_high = F.logsigmoid(-minus_in)

    is_low = (target_long == 0).unsqueeze(-1)
    is_high = (target_long == 255).unsqueeze(-1)
    log_prob_per_mix = torch.where(
        is_low, log_prob_low,
        torch.where(is_high, log_prob_high, log_prob_mid),
    )  # (B, T, K)

    # mixture 加权: log_w + log_prob_per_mix → logsumexp over K
    log_w = F.log_softmax(logit_w, dim=-1)
    log_prob = torch.logsumexp(log_w + log_prob_per_mix, dim=-1).clamp(min=_LOG_PROB_FLOOR)
    # log_prob: (B, T)

    nll = -log_prob

    if reduction == "mean":
        return nll.mean()
    elif reduction == "none":
        return nll
    elif reduction == "sum":
        return nll.sum()
    else:
        raise ValueError(f"未知 reduction: {reduction}（支持 mean/none/sum）")
