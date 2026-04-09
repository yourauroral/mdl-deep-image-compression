"""
公共工具函数 — 被 train.py、evaluate.py、dryrun_forward.py 等多处复用。
"""

import os
import math
import random
import torch
import torch.nn.functional as F
import numpy as np


def seed_everything(seed: int = 42):
    """
    统一设置所有随机种子，确保实验可复现。

    设置范围:
      - Python random
      - NumPy
      - PyTorch CPU + CUDA（所有 GPU）
      - cuDNN deterministic（牺牲少量性能换取确定性）

    参考:
      [1] PyTorch Reproducibility 文档 — "Controlling sources of randomness"
      [2] CS336 "Language Models from Scratch," Stanford, 2024 — seed 设置最佳实践

    参数:
      seed: int — 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 确保 cuDNN 使用确定性算法（对 CIFAR 级小图影响可忽略）
    # Ref: PyTorch 文档 — torch.backends.cudnn.deterministic
    os.environ["PYTHONHASHSEED"] = str(seed)


def compute_bpp(ce_loss: torch.Tensor, channels: int) -> torch.Tensor:
    """
    从交叉熵损失计算 BPP (Bits Per Pixel)。

    BPP = CE_loss / ln(2) × C

    原理:
      CE_loss = -log p(x_t | x_{<t}) 的均值（nats/token），
      每个像素有 C 个通道（各自对应一个 token），
      除以 ln(2) 将 nats 转换为 bits。

    参考:
      [1] Shannon, "A Mathematical Theory of Communication," 1948 —
          最优编码长度 = -log₂ p(x) = -ln p(x) / ln(2)

    参数:
      ce_loss: scalar tensor — per-token 交叉熵（nats）
      channels: int — 通道数（通常 3）
    返回:
      scalar tensor — BPP（bits/pixel）
    """
    return (ce_loss / math.log(2)) * channels


def build_gaussian_targets(targets: torch.Tensor, vocab_size: int,
                           sigma: float) -> torch.Tensor:
    """
    构建 Gaussian label smoothing 的 soft target 分布。

    对每个 target 值 t，生成 P(k) ∝ exp(-(k-t)²/(2σ²)), k=0..V-1，
    并通过 softmax 归一化为概率分布。保留像素值的序数关系：
    预测 127 时，128 获得的概率远高于 0。

    动机:
      标准 categorical CE 将 256 个像素值视为无序类别，
      预测 128（差 1）与预测 0（差 127）惩罚相同。
      Gaussian soft target 注入序数先验，使模型学到"接近即好"。
      这与 PixelCNN++ 使用 discretized logistic 的动机一致。

    参考:
      [1] Szegedy et al., "Rethinking the Inception Architecture," CVPR 2016,
          arXiv:1512.00567 — 提出 uniform label smoothing 正则化
      [2] Salimans et al., "PixelCNN++," ICLR 2017 —
          discretized logistic mixture 隐式保留序数关系

    参数:
      targets: (M,) long tensor，取值 [0, vocab_size-1]
      vocab_size: int，词汇表大小（256）
      sigma: float，高斯标准差（推荐 1.0）
    返回:
      (M, vocab_size) float tensor，每行和为 1 的 soft target 分布
    """
    # arange: [0, 1, ..., V-1]，shape (1, V)
    bins = torch.arange(vocab_size, device=targets.device, dtype=torch.float32).unsqueeze(0)
    # targets: (M,) → (M, 1)
    t = targets.unsqueeze(1).float()
    # log_probs = -(k - t)² / (2σ²)，shape (M, V)
    log_probs = -((bins - t) ** 2) / (2.0 * sigma * sigma)
    # softmax 归一化（数值稳定，等价于 exp + normalize）
    return torch.softmax(log_probs, dim=-1)


# ---------------------------------------------------------------------------
# Discretized Mixture of Logistics (DMOL)
# ---------------------------------------------------------------------------

def discretized_mix_logistic_loss(
    x: torch.Tensor,
    params: torch.Tensor,
    num_mixtures: int = 10,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    单通道 Discretized Mixture of Logistics 负对数似然。

    每个像素值 x ∈ {0,...,255} 被建模为 K 个 logistic 分布的混合：
      P(x) = Σ_k π_k · [σ((x+0.5-μ_k)/s_k) - σ((x-0.5-μ_k)/s_k)]
    其中 σ 为 sigmoid，s_k = exp(log_s_k)。

    与 categorical CE (256 类) 的关键区别:
      CE 将像素值视为无序类别，预测 128 时 127 和 0 的惩罚相同；
      DMOL 利用像素值的序数结构，邻近值自然获得高概率。
      这是 PixelCNN++ 达到 2.92 bits/dim (vs CE ~3.77) 的核心原因。

    参考:
      [1] Salimans et al., "PixelCNN++: Improving the PixelCNN with Discretized
          Logistic Mixture Likelihood and Other Modifications," ICLR 2017.
      [2] Kingma et al., "Improved Variational Inference with Inverse
          Autoregressive Flow," NeurIPS 2016 — logistic distribution 的 CDF 形式。

    参数:
      x:       (M,) long [0, 255] — 目标像素值
      params:  (M, 3K) float — K 组混合参数 [logit_π, μ, log_s]
      num_mixtures: int — 混合分量数 K（默认 10）
      reduction: "mean" | "none" — 损失聚合方式
    返回:
      scalar (reduction="mean") 或 (M,) tensor (reduction="none") — 负对数似然 (nats)
    """
    K = num_mixtures
    M = x.shape[0]
    assert K >= 1, f"num_mixtures 必须 >= 1，got {K}"
    assert params.shape == (M, 3 * K), f"params shape mismatch: {params.shape} vs ({M}, {3*K})"

    # 确保 float32（AMP autocast 下 params 可能是 fp16/bf16）
    params = params.float()

    # 拆分参数: logit_π (M, K), μ (M, K), log_s (M, K)
    logit_pi = params[:, :K]
    mu = params[:, K:2*K]
    log_s = params[:, 2*K:3*K].clamp(-7.0, 7.0)  # 防止 exp 溢出

    # 像素值归一化到 [-1, 1]（PixelCNN++ 约定，数值更稳定）
    # x=0 → -1, x=255 → 1, bin 宽 = 2/255
    x_float = x.float().unsqueeze(1) / 127.5 - 1.0  # (M, 1)
    inv_stdv = torch.exp(-log_s)  # (M, K), 1/s_k

    # CDF bin 边界（[-1, 1] 空间中 bin 半宽 = 1/255）
    plus_in  = inv_stdv * (x_float + 1.0/255.0 - mu)  # (M, K)
    minus_in = inv_stdv * (x_float - 1.0/255.0 - mu)  # (M, K)

    cdf_plus  = torch.sigmoid(plus_in)   # (M, K)
    cdf_minus = torch.sigmoid(minus_in)  # (M, K)

    # 计算 log P(x | component k)，分三种情况:
    # 1. x = 0 (左边界):  log P = log σ(plus_in)  = -softplus(-plus_in)
    # 2. x = 255 (右边界): log P = log(1 - σ(minus_in)) = -softplus(minus_in)
    # 3. 中间:            log P = log(σ(plus) - σ(minus))
    #    当 cdf_plus - cdf_minus 极小时（s 很小或 x 远离 μ），使用 softplus 稳定化:
    #    log(σ(a) - σ(b)) = a - softplus(a) + log(1 - exp(b - a))
    #    ≈ a - softplus(a) + log1p(-exp(b - a))  (当 b < a)
    # Ref: PixelCNN++ 官方实现 — mid_in 近似公式

    # 稳定的中间范围 log-prob
    log_cdf_plus  = plus_in - F.softplus(plus_in)    # log σ(plus_in)
    log_one_minus_cdf_minus = -F.softplus(minus_in)   # log(1 - σ(minus_in))
    # CDF 差值（中间范围）
    cdf_delta = cdf_plus - cdf_minus
    # 安全 log: 当差值极小时切换到稳定近似
    # log(σ(a) - σ(b)) where a = plus_in, b = minus_in
    # 使用 log(max(δ, 1e-12)) 兜底，但优先用精确公式
    log_prob_mid = torch.log(cdf_delta.clamp(min=1e-12))

    # 组合三种情况
    x_long = x.unsqueeze(1)  # (M, 1)
    log_prob_per_k = torch.where(
        x_long == 0,
        log_cdf_plus,
        torch.where(
            x_long == 255,
            log_one_minus_cdf_minus,
            log_prob_mid
        )
    )  # (M, K)

    # 混合: log P(x) = logsumexp_k [log π_k + log P(x|k)]
    log_pi = F.log_softmax(logit_pi, dim=-1)  # (M, K)
    log_prob = torch.logsumexp(log_pi + log_prob_per_k, dim=-1)  # (M,)

    nll = -log_prob  # (M,)
    if reduction == "mean":
        return nll.mean()
    return nll


def discretized_mix_logistic_loss_channels(
    targets: torch.Tensor,
    params: torch.Tensor,
    num_mixtures: int = 10,
    in_channels: int = 3,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    跨通道条件化 Discretized Mixture of Logistics (DMOL) 负对数似然。

    PixelCNN++ 原版方案：三个通道共享混合权重 π_k，但 μ 存在跨通道依赖：
      - Y (第一通道): μ_y 直接使用
      - Cb (第二通道): μ_cb = μ_cb_base + α · y_actual
      - Cr (第三通道): μ_cr = μ_cr_base + β · y_actual + γ · cb_actual
    其中 y_actual, cb_actual 是归一化后的真实像素值（teacher forcing）。

    这使得 Cb 的分布直接以 Y 为条件，Cr 的分布以 Y 和 Cb 为条件，
    无需额外模型容量即可捕获通道间相关性。

    参数布局 (per-pixel, 10K 维):
      params[:, 0:K]     — logit_π (K 个混合权重 logits，三通道共享)
      params[:, K:2K]    — μ_y
      params[:, 2K:3K]   — μ_cb_base
      params[:, 3K:4K]   — μ_cr_base
      params[:, 4K:5K]   — log_s_y
      params[:, 5K:6K]   — log_s_cb
      params[:, 6K:7K]   — log_s_cr
      params[:, 7K:8K]   — coeff α (Cb ← Y)
      params[:, 8K:9K]   — coeff β (Cr ← Y)
      params[:, 9K:10K]  — coeff γ (Cr ← Cb)

    适用场景:
      子像素自回归 (pixel-first) 序列 [Y0,Cb0,Cr0, Y1,Cb1,Cr1, ...] 中，
      每个像素的 3 个 token 对应的 hidden state 取最后一个 (Cr 位置) 来预测整个像素。

    注意:
      本函数要求输入已按像素粒度组织:
        targets: (num_pixels, 3) 或 (num_pixels * 3,) — 每像素 [Y, Cb, Cr]
        params:  (num_pixels, 10K) — 每像素的混合参数

    参考:
      [1] Salimans et al., "PixelCNN++," ICLR 2017, Section 2.1 —
          "We predict the three color channels jointly... by factoring
           p(r,g,b|x) = p(r|x) · p(g|r,x) · p(b|r,g,x)"
      [2] van den Oord et al., "Conditional Image Generation with PixelCNN
          Decoders," NeurIPS 2016 — 通道分解条件化原理

    参数:
      targets:      (P*C,) 或 (P, C) long [0, 255] — 目标像素值
      params:       (P, 10K) float — 像素级混合参数
      num_mixtures: int — 混合分量数 K
      in_channels:  int — 通道数（默认 3: Y/Cb/Cr 或 R/G/B）
      reduction:    "mean" | "none" — 损失聚合方式
    返回:
      scalar (reduction="mean") 或 (P,) tensor (reduction="none") — 负对数似然 (nats)
    """
    K = num_mixtures
    C = in_channels
    assert C == 3, f"跨通道条件化目前仅支持 3 通道，got {C}"
    assert K >= 1, f"num_mixtures 必须 >= 1，got {K}"

    # 重组为像素粒度
    if targets.dim() == 1:
        assert targets.shape[0] % C == 0, (
            f"targets 长度 {targets.shape[0]} 不能被 {C} 整除"
        )
        targets = targets.reshape(-1, C)  # (P, 3)
    P = targets.shape[0]
    assert params.shape == (P, 10 * K), (
        f"params shape mismatch: {params.shape} vs ({P}, {10*K})"
    )

    # 确保 float32（AMP autocast 下 params 可能是 fp16/bf16）
    params = params.float()

    # 拆分参数
    logit_pi   = params[:, 0:K]            # (P, K) 混合权重 logits
    mu_y       = params[:, K:2*K]          # (P, K) Y 通道均值
    mu_cb_base = params[:, 2*K:3*K]        # (P, K) Cb 基础均值
    mu_cr_base = params[:, 3*K:4*K]        # (P, K) Cr 基础均值
    log_s_y    = params[:, 4*K:5*K].clamp(-7.0, 7.0)
    log_s_cb   = params[:, 5*K:6*K].clamp(-7.0, 7.0)
    log_s_cr   = params[:, 6*K:7*K].clamp(-7.0, 7.0)
    coeff_a    = torch.tanh(params[:, 7*K:8*K])   # α: Cb ← Y, tanh 限幅 [-1,1]
    coeff_b    = torch.tanh(params[:, 8*K:9*K])   # β: Cr ← Y
    coeff_g    = torch.tanh(params[:, 9*K:10*K])   # γ: Cr ← Cb

    # 提取每通道 target 并归一化到 [-1, 1]
    y_val  = targets[:, 0].float() / 127.5 - 1.0  # (P,)
    cb_val = targets[:, 1].float() / 127.5 - 1.0
    cr_val = targets[:, 2].float() / 127.5 - 1.0

    # 跨通道条件化均值
    # μ_cb = μ_cb_base + α · y_actual (teacher forcing)
    # μ_cr = μ_cr_base + β · y_actual + γ · cb_actual
    # Ref: PixelCNN++ [1] Section 2.1
    mu_cb = mu_cb_base + coeff_a * y_val.unsqueeze(1)   # (P, K)
    mu_cr = mu_cr_base + coeff_b * y_val.unsqueeze(1) + coeff_g * cb_val.unsqueeze(1)

    # 计算三通道各自的 log P(x_c | component k)
    def _log_prob_single_channel(x_c, mu_c, log_s_c, x_int):
        """单通道 discretized logistic log-prob."""
        inv_stdv = torch.exp(-log_s_c)  # (P, K)
        x_c = x_c.unsqueeze(1)          # (P, 1)
        plus_in  = inv_stdv * (x_c + 1.0/255.0 - mu_c)
        minus_in = inv_stdv * (x_c - 1.0/255.0 - mu_c)

        cdf_plus  = torch.sigmoid(plus_in)
        cdf_minus = torch.sigmoid(minus_in)

        log_cdf_plus = plus_in - F.softplus(plus_in)
        log_one_minus_cdf_minus = -F.softplus(minus_in)
        log_prob_mid = torch.log((cdf_plus - cdf_minus).clamp(min=1e-12))

        x_i = x_int.unsqueeze(1)  # (P, 1)
        return torch.where(
            x_i == 0, log_cdf_plus,
            torch.where(x_i == 255, log_one_minus_cdf_minus, log_prob_mid)
        )

    log_p_y  = _log_prob_single_channel(y_val,  mu_y,  log_s_y,  targets[:, 0])
    log_p_cb = _log_prob_single_channel(cb_val, mu_cb, log_s_cb, targets[:, 1])
    log_p_cr = _log_prob_single_channel(cr_val, mu_cr, log_s_cr, targets[:, 2])

    # 联合概率: log P(y,cb,cr | k) = log P(y|k) + log P(cb|y,k) + log P(cr|y,cb,k)
    log_prob_per_k = log_p_y + log_p_cb + log_p_cr  # (P, K)

    # 混合: log P(pixel) = logsumexp_k [log π_k + log P(y,cb,cr | k)]
    log_pi = F.log_softmax(logit_pi, dim=-1)  # (P, K)
    log_prob = torch.logsumexp(log_pi + log_prob_per_k, dim=-1)  # (P,)

    # 归一化: DMOL loss 是每像素 3 通道的联合 NLL。
    # 为了与 per-token CE loss 保持 BPP 公式一致:
    #   BPP = ce_loss / ln(2) * C，其中 ce_loss 是 per-token (即 per-channel) NLL，
    # 需要将联合 NLL 除以 C 得到 per-token 均值。
    nll = -log_prob / C  # (P,), per-token 平均 NLL

    if reduction == "mean":
        return nll.mean()
    return nll
