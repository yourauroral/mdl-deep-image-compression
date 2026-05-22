"""离散化 logistic 混合 (Discretized Mixture of Logistics) 损失实现。

每个 sub-pixel token 独立产 K-mix 1D logistic 分布参数 (logit_w, mean, log_scale)，
**无通道耦合**：与本仓库 sub-pixel AR pixel-first 序列布局对齐，通道依赖
通过 attention 而非输出头建模。

Refs:
  Salimans et al., "PixelCNN++," ICLR 2017 — 离散化 logistic 混合定义
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
_HEAD_BIAS_INIT_LOG_SCALE = 2.0  # log_scale 通道初始 bias，使 inv_s ≈ 0.135 落在 sigmoid 线性区


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
        # log_scale 通道（第 3 段 K 个 bias）置 +2.0 起步：
        # inv_s = exp(-2) ≈ 0.135 → bin·inv_s ≈ 0.135 落在 sigmoid 线性区，
        # CDF 差分非零，绕开 v1-v5 的 fallback 风暴（v1-v5 全部 bias=0 起步，
        # inv_s=1, bin·inv_s=0.004 极窄，sigmoid 饱和致 CDF 差分 ~1e-4 量级）。
        K = self.n_mixtures
        with torch.no_grad():
            self.proj.bias[2 * K : 3 * K].fill_(_HEAD_BIAS_INIT_LOG_SCALE)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """hidden: (B, T, d_model) → params: (B, T, K*3)。"""
        return self.proj(hidden)


def dmol_loss_1d(params: torch.Tensor,
                 target_long: torch.Tensor,
                 n_mixtures: int,
                 reduction: str = "mean") -> torch.Tensor:
    """1D 离散化 logistic 混合 NLL。

    Args:
        params:       (B, T, K*3) float — DMoLHead1D 输出
        target_long:  (B, T) long ∈ [0, 255] — sub-pixel 整数 target
        n_mixtures:   K
        reduction:    "mean" 返回标量；"none" 返回 (B, T) per-token NLL

    Returns:
        NLL（nat / sub-pixel），与 categorical CE 同口径，下游 bpd = NLL / ln2 不变

    数值实现：
        - mean = 127.5 · (1 + tanh(mean_raw))         ∈ [0, 255]
        - log_scale = clamp(log_scale_raw, -7, +5)
        - inv_s = exp(-log_scale)
        - half_bin = 0.5（bin 宽=1，target ∈ [0, 255]）
        - plus_in  = (target + 0.5 - mean) · inv_s
        - minus_in = (target - 0.5 - mean) · inv_s
        - 边界 target=0:    log σ(plus_in)
        - 边界 target=255:  log σ(-minus_in)  =  log(1 - σ(minus_in))
        - 中间:             log((σ(plus_in) - σ(minus_in)).clamp(min=1e-12))
        - log_w = log_softmax(logit_w, dim=-1)
        - log_prob = logsumexp(log_w + log_prob_per_mix, dim=-1).clamp(min=-30)
        - NLL = -log_prob.mean()  # 或 reshape (B, T)
    """
    K = n_mixtures
    assert params.size(-1) == 3 * K, (
        f"params last dim {params.size(-1)} != 3*K = {3 * K}"
    )

    # split: (B, T, K) × 3
    logit_w, mean_raw, log_scale_raw = params.split(K, dim=-1)

    # mean ∈ [0, 255]（无通道耦合，pixel-first AR 已通过 attention 建模 R/G/B 依赖）
    mean = 127.5 * (1.0 + torch.tanh(mean_raw))

    # log_scale 双向 clamp，inv_s 范围 [0.0067, 1097]
    log_scale = log_scale_raw.clamp(_LOG_SCALE_MIN, _LOG_SCALE_MAX)
    inv_s = torch.exp(-log_scale)

    # target 广播到 mixture 维: (B, T) → (B, T, 1)，与 (B, T, K) 广播
    target = target_long.unsqueeze(-1).to(mean.dtype)

    plus_in = (target + 0.5 - mean) * inv_s
    minus_in = (target - 0.5 - mean) * inv_s

    # 中间 bin: log(σ(plus_in) - σ(minus_in))
    # 用 logsigmoid 数值稳定地表达 σ；clamp 防极小差分
    cdf_plus = torch.sigmoid(plus_in)
    cdf_minus = torch.sigmoid(minus_in)
    cdf_delta = (cdf_plus - cdf_minus).clamp(min=_CDF_DIFF_FLOOR)
    log_prob_mid = torch.log(cdf_delta)

    # 边界 target=0:   log σ(plus_in)              （无下界，全部概率落在 (-inf, 0+0.5]）
    # 边界 target=255: log σ(-minus_in) = log(1 - σ(minus_in)) （全部概率落在 [255-0.5, +inf)）
    log_prob_low = F.logsigmoid(plus_in)
    log_prob_high = F.logsigmoid(-minus_in)

    # 选择正确分支
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
