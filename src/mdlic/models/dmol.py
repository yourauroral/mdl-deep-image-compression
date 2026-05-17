"""Discretized Mixture of Logistics (DMoL) output head for RGB-bit-exact AR.

Salimans et al., "PixelCNN++: Improving the PixelCNN with Discretized Logistic
Mixture Likelihood and Other Modifications," ICLR 2017.

为什么 DMoL：256-class softmax 对每个 sub-pixel 独立分类，结构上无法建模 RGB 三
通道间的相关性。DMoL 通过把"一个 pixel"看作连续 logistic 混合分布、再离散化到 256
个 bin，并把 G 的均值线性依赖 R、B 的均值线性依赖 (R, G)，**自带通道间线性 AR**，
等价于学到色彩去相关变换。这是 PixelCNN++ / Image Transformer / PixelSNAIL /
Sparse Transformer 等 RGB-bit-exact ≤3.0 文献的共同选择，不是可选项。

每个 mixture 输出 10 个参数：
  logit_w  — mixture 权重（softmax 归一化前）
  μR, μG, μB  — 三通道均值（μG_eff = μG + tanh(αGR)·(R-1)；μB_eff 同理）
  log_sR, log_sG, log_sB  — 三通道 log-scale（避免 scale 数值下溢）
  αGR, βBR, βBG  — 通道间均值耦合系数（PixelCNN++ §2.2）

DMoL 路径下 token id 在 [0, 255]（RGB int8），与 vocab=256 兼容；输入端仍用
embedding 查表（input embed），仅输出端走 DMoL（不复用 token_embed 权重）。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


_LOG_SCALE_MIN = -5.0


def _split_mixture_params(params: torch.Tensor, n_mix: int):
    """(B, T, n_mix*10) → 拆出 logit_w / μ / log_s / 通道耦合系数 4 组张量。

    返回:
      logit_w:  (B, T, n_mix)
      means:    (B, T, 3, n_mix)        # [R, G, B] × mixture, tanh bound 到 [-1, 1]
      log_scales: (B, T, 3, n_mix)      # clamp 到 >= _LOG_SCALE_MIN
      coeffs:   (B, T, 3, n_mix)        # [αGR, βBR, βBG] × mixture, tanh 压到 (-1,1)

    means 必须 tanh bound 到 target 归一化范围 [-1, 1]，否则 random init 时
    mean 量级在 [-5, 5]，配合 inv_s = exp(-log_s_min) 让 sigmoid 完全饱和，
    CDF 差分变 0 后触发 log_pdf fallback，单 token log-prob 飞至 -10000+ nat
    导致 batch 平均 loss 爆炸。PixelCNN++ 原版关键细节，遗漏会致 ep1 train_bpd
    单调上升。
    """
    B, T, D = params.shape
    assert D == n_mix * 10, f"DMoL 参数维度应为 n_mix*10={n_mix*10}, got {D}"
    params = params.view(B, T, n_mix, 10)
    logit_w = params[..., 0]                                # (B, T, n_mix)
    means = torch.tanh(params[..., 1:4]).permute(0, 1, 3, 2).contiguous()      # (B, T, 3, n_mix)
    log_scales = params[..., 4:7].clamp(min=_LOG_SCALE_MIN).permute(0, 1, 3, 2).contiguous()
    coeffs = torch.tanh(params[..., 7:10]).permute(0, 1, 3, 2).contiguous()
    return logit_w, means, log_scales, coeffs


class DMoLHead(nn.Module):
    """d_model hidden → n_mix*10 DMoL 参数的线性投影。"""

    def __init__(self, d_model: int, n_mixtures: int = 10):
        super().__init__()
        self.n_mixtures = n_mixtures
        self.proj = nn.Linear(d_model, n_mixtures * 10)
        nn.init.normal_(self.proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.proj.bias)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.proj(hidden)


def dmol_log_prob(params: torch.Tensor, target_pixels_int: torch.Tensor,
                  n_mixtures: int) -> torch.Tensor:
    """计算 DMoL 下每个 pixel 的 log-prob。

    参数:
      params:             (B, T, n_mix*10) float
      target_pixels_int:  (B, T, 3) long ∈ [0, 255]，[R, G, B] 整数
      n_mixtures:         混合分量数

    返回:
      log_prob: (B, T) float，per-pixel log p(R, G, B)
    """
    assert target_pixels_int.dtype == torch.long
    assert target_pixels_int.shape[-1] == 3

    logit_w, means, log_scales, coeffs = _split_mixture_params(params, n_mixtures)

    # 离散化前归一到 [-1, 1]，与 PixelCNN++ 一致（让 mixture 在固定数值范围学）
    target = target_pixels_int.float() / 127.5 - 1.0                  # (B, T, 3)

    # 通道间均值耦合（关键的"自带通道去相关"机制）
    R, G, _B = target[..., 0:1], target[..., 1:2], target[..., 2:3]    # (B, T, 1) each
    mean_R = means[..., 0, :]                                          # (B, T, n_mix)
    mean_G = means[..., 1, :] + coeffs[..., 0, :] * R
    mean_B = means[..., 2, :] + coeffs[..., 1, :] * R + coeffs[..., 2, :] * G

    log_s_R = log_scales[..., 0, :]
    log_s_G = log_scales[..., 1, :]
    log_s_B = log_scales[..., 2, :]

    inv_s_R = torch.exp(-log_s_R)
    inv_s_G = torch.exp(-log_s_G)
    inv_s_B = torch.exp(-log_s_B)

    # 离散化 logistic CDF: P(x) = σ((x+0.5/127.5 − μ)/s) − σ((x−0.5/127.5 − μ)/s)
    # bin 宽 1/127.5（因为我们已归一到 [-1, 1] 域，256 个 bin → 步长 2/255 ≈ 1/127.5）
    half_bin = 1.0 / 255.0

    def _log_cdf_diff(x, mean, inv_s, log_s, is_low_edge, is_high_edge):
        """计算 log[σ(plus) − σ(minus)]，并处理 0 / 255 边界。"""
        x = x.unsqueeze(-1)                                # (B, T, 1) → (B, T, 1, 1) broadcastable
        centered = x - mean                                 # (B, T, n_mix)
        plus_in = (centered + half_bin) * inv_s
        minus_in = (centered - half_bin) * inv_s
        cdf_plus = torch.sigmoid(plus_in)
        cdf_minus = torch.sigmoid(minus_in)
        # 中间 bin: log(cdf_plus - cdf_minus)，clamp 防 log(0)
        log_prob_mid = torch.log((cdf_plus - cdf_minus).clamp(min=1e-12))
        # 左边界 (x = 0): log σ(plus_in)
        log_prob_low = F.logsigmoid(plus_in)
        # 右边界 (x = 255): log σ(-minus_in) = log(1 − σ(minus_in))
        log_prob_high = F.logsigmoid(-minus_in)
        # 数值稳定 fallback: bin 极窄时用 PDF · bin_width
        # mid_in = centered * inv_s; pdf_approx ≈ logistic_pdf(mid_in)
        mid_in = centered * inv_s
        log_pdf = mid_in - log_s - 2.0 * F.softplus(mid_in) + math.log(2.0 * half_bin)
        # 主路径：用 mid，边界条件覆盖
        out = torch.where(is_low_edge, log_prob_low,
                          torch.where(is_high_edge, log_prob_high, log_prob_mid))
        # 当 cdf_plus - cdf_minus 数值过小（远离 mean 的尾部），退回 log_pdf 近似
        # 阈值 1e-12 仅捕真正的数值溢出，避免未训练初期大量 token 命中 fallback
        # 引入大噪声（这是 ep1 loss 单调上升的次要诱因之一）
        too_small = (cdf_plus - cdf_minus) < 1e-12
        out = torch.where(too_small & ~is_low_edge & ~is_high_edge, log_pdf, out)
        return out

    is_low_R = target_pixels_int[..., 0:1] == 0
    is_high_R = target_pixels_int[..., 0:1] == 255
    is_low_G = target_pixels_int[..., 1:2] == 0
    is_high_G = target_pixels_int[..., 1:2] == 255
    is_low_B = target_pixels_int[..., 2:3] == 0
    is_high_B = target_pixels_int[..., 2:3] == 255

    log_p_R = _log_cdf_diff(target[..., 0], mean_R, inv_s_R, log_s_R, is_low_R, is_high_R)
    log_p_G = _log_cdf_diff(target[..., 1], mean_G, inv_s_G, log_s_G, is_low_G, is_high_G)
    log_p_B = _log_cdf_diff(target[..., 2], mean_B, inv_s_B, log_s_B, is_low_B, is_high_B)

    log_p_pixel_per_mix = log_p_R + log_p_G + log_p_B                  # (B, T, n_mix)
    log_w = F.log_softmax(logit_w, dim=-1)                              # (B, T, n_mix)
    log_prob = torch.logsumexp(log_w + log_p_pixel_per_mix, dim=-1)     # (B, T)
    return log_prob


def dmol_loss(params: torch.Tensor, target_pixels_int: torch.Tensor,
              n_mixtures: int) -> torch.Tensor:
    """返回 mean NLL（nat / pixel），用作 loss。"""
    log_prob = dmol_log_prob(params, target_pixels_int, n_mixtures)
    return -log_prob.mean()
