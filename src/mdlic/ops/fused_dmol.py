"""
Fused Discretized Mixture of Logistics (DMOL) Triton Kernel — forward + backward.

将 DMOL 的 logistic CDF 计算、mixture 合并、log-prob 求和融合为一次 kernel launch，
避免存储中间 CDF/sigmoid 矩阵，降低 HBM 带宽。

支持两种模式:
  1. 单通道 per-token: 每行 3K 参数 [logit_π, μ, log_s]
  2. 跨通道条件化 per-pixel: 每行 10K 参数 [logit_π, μ_y, μ_cb, μ_cr,
     log_s_y, log_s_cb, log_s_cr, coeff_α, coeff_β, coeff_γ]

数学推导:
  设 K 个混合分量，第 k 个分量: 权重 π_k, 均值 μ_k, 尺度 s_k = exp(log_s_k)

  Forward (单通道):
    1. 归一化: x_c = x / 127.5 - 1.0 ∈ [-1, 1]
    2. CDF 差值: δ_k = σ((x_c + 1/255 - μ_k)/s_k) - σ((x_c - 1/255 - μ_k)/s_k)
       边界: x=0 → log σ(plus_in),  x=255 → log(1 - σ(minus_in))
    3. 混合: log P(x) = logsumexp_k [log π_k + log δ_k]
    4. NLL = -log P(x)

  Forward (跨通道条件化):
    μ_cb = μ_cb_base + α · y_actual
    μ_cr = μ_cr_base + β · y_actual + γ · cb_actual
    log P(pixel) = logsumexp_k [log π_k + log δ_y + log δ_cb + log δ_cr]

  Backward:
    通过 autograd 自动完成（保存 params + targets, 重算 forward）。

参考:
  [1] Salimans et al., "PixelCNN++: Improving the PixelCNN with Discretized
      Logistic Mixture Likelihood and Other Modifications," ICLR 2017.
  [2] Milakov & Gimelshein, "Online normalizer calculation for softmax,"
      arXiv:1805.02867, 2018. log-sum-exp 数值稳定技巧。
  [3] Hsu et al., "Liger-Kernel," arXiv:2410.10989, 2024. Fused kernel pattern.
"""

import torch
import triton
import triton.language as tl


# ─── 辅助函数: Triton sigmoid / softplus ──────────────────────────

@triton.jit
def _sigmoid(x):
    """σ(x) = 1 / (1 + exp(-x))，用 Triton 原语实现。"""
    return 1.0 / (1.0 + tl.exp(-x))


@triton.jit
def _log_sigmoid(x):
    """log σ(x) = x - softplus(x) = x - log(1 + exp(x))，数值稳定。"""
    # 当 x 很大时 softplus(x) ≈ x，log σ(x) ≈ 0
    # 当 x 很小时 softplus(x) ≈ exp(x)，log σ(x) ≈ x
    return tl.where(x > 0, -tl.log(1.0 + tl.exp(-x)), x - tl.log(1.0 + tl.exp(x)))


@triton.jit
def _softplus(x):
    """softplus(x) = log(1 + exp(x))，数值稳定。"""
    return tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))


@triton.jit
def _log_one_minus_sigmoid(x):
    """log(1 - σ(x)) = -softplus(x)。"""
    return -_softplus(x)


@triton.jit
def _dmol_log_prob_single(x_int, x_c, mu, log_s):
    """
    单通道 discretized logistic log-prob。

    参数:
      x_int: 原始整数像素值 [0, 255]
      x_c:   归一化像素值 [-1, 1]
      mu:    logistic 均值 (K 个)
      log_s: log 尺度 (K 个, 已 clamp)
    返回:
      log_prob: log P(x | component k), shape 同 mu
    """
    inv_stdv = tl.exp(-log_s)
    plus_in  = inv_stdv * (x_c + 1.0 / 255.0 - mu)
    minus_in = inv_stdv * (x_c - 1.0 / 255.0 - mu)

    # log P(x|k) 分三种情况:
    # x=0: log σ(plus_in)
    log_cdf_plus = _log_sigmoid(plus_in)
    # x=255: log(1 - σ(minus_in))
    log_one_m_cdf_minus = _log_one_minus_sigmoid(minus_in)
    # 中间: log(σ(plus) - σ(minus)), clamp 防 log(0)
    cdf_delta = _sigmoid(plus_in) - _sigmoid(minus_in)
    # tl.maximum 替代 clamp
    log_prob_mid = tl.log(tl.maximum(cdf_delta, 1e-12))

    # 组合
    log_prob = tl.where(
        x_int == 0, log_cdf_plus,
        tl.where(x_int == 255, log_one_m_cdf_minus, log_prob_mid)
    )
    return log_prob


# ─── Forward Kernel: 单通道 per-token ────────────────────────────
@triton.jit
def _fused_dmol_fwd_kernel(
    PARAMS,      # 输入参数指针, shape (M, 3K)
    TARGETS,     # target 像素值指针, shape (M,) long [0,255]
    NLL_OUT,     # 输出: 每行 NLL, shape (M,)
    stride_params,  # params 行 stride
    M: tl.constexpr,           # 总行数
    K: tl.constexpr,           # 混合分量数
    BLOCK_K: tl.constexpr,     # >= K 的 2 的幂
):
    """
    每个 program 处理一行: 3K 参数 → 1 个 NLL 值。

    算法:
      1. 拆分 [logit_π, μ, log_s]
      2. 归一化 target, 计算 CDF 差值 log-prob
      3. logsumexp(log_π + log_prob) → log P(x)
      4. NLL = -log P(x)
    """
    row = tl.program_id(0)
    if row >= M:
        return

    # 加载 target
    x_int = tl.load(TARGETS + row).to(tl.int32)
    x_c = x_int.to(tl.float32) / 127.5 - 1.0

    # 列偏移
    ks = tl.arange(0, BLOCK_K)
    mask = ks < K

    # 加载参数: [logit_π | μ | log_s], 各 K 维
    base = PARAMS + row * stride_params
    logit_pi = tl.load(base + ks, mask=mask, other=0.0).to(tl.float32)
    mu       = tl.load(base + K + ks, mask=mask, other=0.0).to(tl.float32)
    log_s    = tl.load(base + 2 * K + ks, mask=mask, other=0.0).to(tl.float32)
    # clamp log_s 到 [-7, 7]
    log_s = tl.maximum(tl.minimum(log_s, 7.0), -7.0)

    # 单通道 log-prob per component
    log_prob_k = _dmol_log_prob_single(x_int, x_c, mu, log_s)

    # log_softmax(logit_pi): log_pi = logit_pi - logsumexp(logit_pi)
    pi_max = tl.max(logit_pi, axis=0)
    log_pi = logit_pi - pi_max - tl.log(tl.sum(tl.exp(logit_pi - pi_max), axis=0))

    # logsumexp(log_pi + log_prob_k) → log P(x)
    combined = tl.where(mask, log_pi + log_prob_k, -float('inf'))
    c_max = tl.max(combined, axis=0)
    log_prob = c_max + tl.log(tl.sum(tl.exp(combined - c_max), axis=0))

    # NLL = -log P(x)
    tl.store(NLL_OUT + row, -log_prob)


# ─── Forward Kernel: 跨通道条件化 per-pixel ──────────────────────
@triton.jit
def _fused_dmol_channels_fwd_kernel(
    PARAMS,      # 输入参数指针, shape (P, 10K)
    TARGETS,     # target 像素值指针, shape (P, 3) long [0,255]
    NLL_OUT,     # 输出: 每像素 NLL (已除以 C=3), shape (P,)
    stride_params,
    stride_targets,
    P: tl.constexpr,          # 像素数
    K: tl.constexpr,          # 混合分量数
    BLOCK_K: tl.constexpr,    # >= K 的 2 的幂
):
    """
    每个 program 处理一个像素 (3 通道): 10K 参数 → 1 个 NLL 值。

    参数布局 (10K 维):
      [0:K]     logit_π
      [K:2K]    μ_y
      [2K:3K]   μ_cb_base
      [3K:4K]   μ_cr_base
      [4K:5K]   log_s_y
      [5K:6K]   log_s_cb
      [6K:7K]   log_s_cr
      [7K:8K]   coeff_α (Cb ← Y)
      [8K:9K]   coeff_β (Cr ← Y)
      [9K:10K]  coeff_γ (Cr ← Cb)

    跨通道条件化:
      μ_cb = μ_cb_base + tanh(α) · y_actual
      μ_cr = μ_cr_base + tanh(β) · y_actual + tanh(γ) · cb_actual
    """
    px = tl.program_id(0)
    if px >= P:
        return

    # 加载 3 通道 target
    t_base = TARGETS + px * stride_targets
    y_int  = tl.load(t_base + 0).to(tl.int32)
    cb_int = tl.load(t_base + 1).to(tl.int32)
    cr_int = tl.load(t_base + 2).to(tl.int32)
    y_c  = y_int.to(tl.float32)  / 127.5 - 1.0
    cb_c = cb_int.to(tl.float32) / 127.5 - 1.0
    cr_c = cr_int.to(tl.float32) / 127.5 - 1.0

    # 列偏移
    ks = tl.arange(0, BLOCK_K)
    mask = ks < K

    # 加载全部 10K 参数
    p_base = PARAMS + px * stride_params
    logit_pi   = tl.load(p_base + ks, mask=mask, other=0.0).to(tl.float32)
    mu_y       = tl.load(p_base + K + ks, mask=mask, other=0.0).to(tl.float32)
    mu_cb_base = tl.load(p_base + 2*K + ks, mask=mask, other=0.0).to(tl.float32)
    mu_cr_base = tl.load(p_base + 3*K + ks, mask=mask, other=0.0).to(tl.float32)
    log_s_y    = tl.load(p_base + 4*K + ks, mask=mask, other=0.0).to(tl.float32)
    log_s_cb   = tl.load(p_base + 5*K + ks, mask=mask, other=0.0).to(tl.float32)
    log_s_cr   = tl.load(p_base + 6*K + ks, mask=mask, other=0.0).to(tl.float32)
    coeff_a    = tl.load(p_base + 7*K + ks, mask=mask, other=0.0).to(tl.float32)
    coeff_b    = tl.load(p_base + 8*K + ks, mask=mask, other=0.0).to(tl.float32)
    coeff_g    = tl.load(p_base + 9*K + ks, mask=mask, other=0.0).to(tl.float32)

    # clamp log_s
    log_s_y  = tl.maximum(tl.minimum(log_s_y,  7.0), -7.0)
    log_s_cb = tl.maximum(tl.minimum(log_s_cb, 7.0), -7.0)
    log_s_cr = tl.maximum(tl.minimum(log_s_cr, 7.0), -7.0)

    # tanh 限幅系数
    # tanh(x) = (exp(2x)-1)/(exp(2x)+1)
    coeff_a = tl.extra.cuda.libdevice.tanh(coeff_a)
    coeff_b = tl.extra.cuda.libdevice.tanh(coeff_b)
    coeff_g = tl.extra.cuda.libdevice.tanh(coeff_g)

    # 跨通道条件化均值
    mu_cb = mu_cb_base + coeff_a * y_c
    mu_cr = mu_cr_base + coeff_b * y_c + coeff_g * cb_c

    # 三通道 log-prob
    log_p_y  = _dmol_log_prob_single(y_int,  y_c,  mu_y,  log_s_y)
    log_p_cb = _dmol_log_prob_single(cb_int, cb_c, mu_cb, log_s_cb)
    log_p_cr = _dmol_log_prob_single(cr_int, cr_c, mu_cr, log_s_cr)

    # 联合 log-prob per component
    log_prob_k = log_p_y + log_p_cb + log_p_cr  # (BLOCK_K,)

    # log_softmax(logit_pi)
    pi_max = tl.max(logit_pi, axis=0)
    log_pi = logit_pi - pi_max - tl.log(tl.sum(tl.exp(logit_pi - pi_max), axis=0))

    # logsumexp → log P(pixel)
    combined = tl.where(mask, log_pi + log_prob_k, -float('inf'))
    c_max = tl.max(combined, axis=0)
    log_prob = c_max + tl.log(tl.sum(tl.exp(combined - c_max), axis=0))

    # NLL per pixel, 除以 C=3 得到 per-token 平均
    tl.store(NLL_OUT + px, -log_prob / 3.0)


# ─── Autograd Function: 单通道 ──────────────────────────────────
class FusedDMOLFunction(torch.autograd.Function):
    """
    单通道 Fused DMOL autograd 封装。

    Forward: Triton kernel 计算 per-row NLL, 返回标量 mean。
    Backward: 依赖 PyTorch autograd 对 params 求导（保存 params + targets,
    backward 时回退到 PyTorch 实现重算，确保梯度正确性）。

    注: 与 fused_ce_zloss.py 不同，DMOL 的 backward 梯度公式较复杂
    （涉及 sigmoid 链式法则 + mixture softmax），手写 backward kernel
    的 bug 风险高。采用 "fused forward + PyTorch backward" 策略:
    forward 省显存+带宽，backward 牺牲少量性能换取正确性。
    """

    @staticmethod
    def forward(ctx, params: torch.Tensor, targets: torch.Tensor,
                num_mixtures: int):
        assert params.ndim == 2, f"params 必须为 2D (M, 3K)，got ndim={params.ndim}"
        assert targets.ndim == 1, f"targets 必须为 1D (M,)，got ndim={targets.ndim}"
        M, D = params.shape
        K = num_mixtures
        assert D == 3 * K, f"params 列数 ({D}) != 3*K ({3*K})"
        assert targets.shape[0] == M
        assert M > 0, f"params 必须至少有 1 行，got shape {params.shape}"

        BLOCK_K = triton.next_power_of_2(K)

        nll_out = torch.zeros(M, device=params.device, dtype=torch.float32)

        _fused_dmol_fwd_kernel[(M,)](
            params.contiguous(), targets.contiguous(), nll_out,
            stride_params=params.stride(0),
            M=M, K=K, BLOCK_K=BLOCK_K,
        )

        loss = nll_out.mean()

        # 保存用于 backward (PyTorch 路径)
        ctx.save_for_backward(params, targets)
        ctx.num_mixtures = num_mixtures

        return loss

    @staticmethod
    def backward(ctx, grad_output):
        params, targets = ctx.saved_tensors
        K = ctx.num_mixtures

        # 回退到 PyTorch 实现计算梯度（forward 已由 fused kernel 完成，
        # backward 需要完整 autograd 图，这里重算一次 PyTorch forward）
        params_grad = params.detach().requires_grad_(True)
        params_grad.grad = None  # 确保无残留梯度
        with torch.enable_grad():
            from ..utils import discretized_mix_logistic_loss
            loss_recomputed = discretized_mix_logistic_loss(
                targets, params_grad, num_mixtures=K, reduction="mean"
            )
            loss_recomputed.backward()

        return params_grad.grad * grad_output, None, None


# ─── Autograd Function: 跨通道条件化 ────────────────────────────
class FusedDMOLChannelsFunction(torch.autograd.Function):
    """
    跨通道条件化 Fused DMOL autograd 封装。

    Forward: Triton kernel per-pixel (10K 参数 → 1 NLL)。
    Backward: PyTorch 路径重算（同 FusedDMOLFunction 策略）。
    """

    @staticmethod
    def forward(ctx, params: torch.Tensor, targets: torch.Tensor,
                num_mixtures: int):
        # targets: (P, 3) or flattened (P*3,) → reshape to (P, 3)
        if targets.ndim == 1:
            assert targets.shape[0] % 3 == 0
            targets = targets.reshape(-1, 3)
        assert params.ndim == 2
        P = targets.shape[0]
        K = num_mixtures
        assert P > 0, f"targets 必须至少有 1 个像素，got shape {targets.shape}"
        assert params.shape == (P, 10 * K), (
            f"params shape mismatch: expected ({P}, {10*K}), got {params.shape}"
        )

        BLOCK_K = triton.next_power_of_2(K)

        # 确保 targets 是 contiguous int64
        targets_contig = targets.contiguous().to(torch.int64)
        params_contig = params.contiguous()

        nll_out = torch.zeros(P, device=params.device, dtype=torch.float32)

        _fused_dmol_channels_fwd_kernel[(P,)](
            params_contig, targets_contig, nll_out,
            stride_params=params_contig.stride(0),
            stride_targets=targets_contig.stride(0),
            P=P, K=K, BLOCK_K=BLOCK_K,
        )

        loss = nll_out.mean()

        ctx.save_for_backward(params, targets)
        ctx.num_mixtures = num_mixtures

        return loss

    @staticmethod
    def backward(ctx, grad_output):
        params, targets = ctx.saved_tensors
        K = ctx.num_mixtures

        params_grad = params.detach().requires_grad_(True)
        params_grad.grad = None  # 确保无残留梯度
        with torch.enable_grad():
            from ..utils import discretized_mix_logistic_loss_channels
            loss_recomputed = discretized_mix_logistic_loss_channels(
                targets, params_grad, num_mixtures=K, in_channels=3,
                reduction="mean"
            )
            loss_recomputed.backward()

        return params_grad.grad * grad_output, None, None


# ─── 便捷函数 ───────────────────────────────────────────────────

def fused_dmol_loss(
    params: torch.Tensor,
    targets: torch.Tensor,
    num_mixtures: int = 10,
) -> torch.Tensor:
    """
    Fused 单通道 DMOL loss 计算。

    参数:
      params:       (M, 3K) float tensor — 混合参数
      targets:      (M,) long tensor [0, 255] — 目标像素值
      num_mixtures: int — 混合分量数 K

    返回:
      scalar tensor — mean NLL (nats)
    """
    return FusedDMOLFunction.apply(
        params.contiguous(), targets.contiguous(), num_mixtures
    )


def fused_dmol_loss_channels(
    params: torch.Tensor,
    targets: torch.Tensor,
    num_mixtures: int = 10,
) -> torch.Tensor:
    """
    Fused 跨通道条件化 DMOL loss 计算。

    参数:
      params:       (P, 10K) float tensor — 像素级混合参数
      targets:      (P, 3) 或 (P*3,) long tensor [0, 255] — 目标像素值
      num_mixtures: int — 混合分量数 K

    返回:
      scalar tensor — mean NLL (nats, per-token 平均)
    """
    return FusedDMOLChannelsFunction.apply(
        params.contiguous(), targets.contiguous(), num_mixtures
    )
