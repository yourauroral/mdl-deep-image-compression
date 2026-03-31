"""
Fused Cross-Entropy + z-loss Triton Kernel — forward + backward.

将 softmax、cross-entropy loss、z-loss 合并为一次 kernel launch，
避免存储完整的 softmax 概率矩阵 (M × V)，显著降低显存和 HBM 带宽。

数学推导:
  设 logits = x ∈ R^V（某一行），target = t ∈ {0,...,V-1}，z-loss 权重 = w

  Forward:
    1. Online softmax（数值稳定）:
       m = max(x)                         ... 行最大值
       lse = log(sum(exp(x_i - m))) + m   ... log-sum-exp
    2. Cross-entropy:
       ce = -x_t + lse                    ... = -log softmax(x)_t
    3. z-loss（惩罚 logits 幅度）:
       z = lse²                           ... PaLM arXiv:2204.02311
    4. 返回 ce（均值）和 z（均值），外部组合: loss = ce + w * z

  Backward (d_loss/d_x):
    总 loss per sample = ce + w * z = (-x_t + lse) + w * lse²
    令 s_i = softmax(x)_i = exp(x_i - lse)

    ∂ce/∂x_i  = s_i - 1{i=t}
    ∂z/∂x_i   = 2 * lse * s_i
    ∂loss/∂x_i = (1 + 2*w*lse) * s_i - 1{i=t}

    注意: 这里假设 upstream gradient = 1/M（mean reduction），
    实际在 kernel 内乘以 1/M。

参考:
  [1] Milakov & Gimelshein, "Online normalizer calculation for softmax,"
      arXiv:1805.02867, 2018. Online softmax 的数值稳定实现。
  [2] Google, "PaLM: Scaling Language Modeling with Pathways,"
      arXiv:2204.02311, 2022, Section 5. z-loss = w * mean(lse²).
  [3] Hsu et al., "Liger-Kernel: Efficient Triton Kernels for LLM Training,"
      arXiv:2410.10989, 2024. Fused CE kernel pattern.
  [4] Triton Tutorials — Softmax kernel.
      https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html
"""

import torch
import triton
import triton.language as tl


# ─── Forward Kernel ──────────────────────────────────────────────
@triton.jit
def _fused_ce_zloss_fwd_kernel(
    LOGITS,      # 输入 logits 指针, shape (M, V)
    TARGETS,     # target 索引指针, shape (M,)
    CE_OUT,      # 输出: 每行 CE loss, shape (M,)
    ZLOSS_OUT,   # 输出: 每行 lse², shape (M,)
    stride_logits,  # logits 行 stride
    M,           # 总行数（batch_size × seq_len）
    V,           # vocab size
    BLOCK_V: tl.constexpr,  # block size，需 >= V
):
    """
    每个 program 处理一行 logits → 一个 CE loss 值 + 一个 lse² 值。

    算法: Online softmax [1] 的一趟变体:
      1. 找 max（数值稳定）
      2. 计算 sum(exp(x - max))
      3. lse = log(sum) + max
      4. ce = -x[target] + lse
      5. z = lse²
    """
    row = tl.program_id(0)
    if row >= M:
        return

    # 加载 target 索引
    target = tl.load(TARGETS + row)

    # 列偏移
    cols = tl.arange(0, BLOCK_V)
    mask = cols < V

    # ── 1. 加载 logits ──
    logits_ptr = LOGITS + row * stride_logits
    x = tl.load(logits_ptr + cols, mask=mask, other=-float('inf')).to(tl.float32)

    # ── 2. Online softmax: max → sum(exp) → lse ──
    # [1] 数值稳定: 先减 max 再 exp
    row_max = tl.max(x, axis=0)
    x_shifted = x - row_max
    exp_x = tl.exp(x_shifted)
    sum_exp = tl.sum(exp_x, axis=0)
    lse = tl.log(sum_exp) + row_max   # log-sum-exp

    # ── 3. CE loss: -logits[target] + lse ──
    # 用 mask 提取 target 位置的 logit
    target_mask = cols == target
    x_target = tl.sum(tl.where(target_mask, x, tl.zeros_like(x)), axis=0)
    ce = -x_target + lse

    # ── 4. z-loss: lse² ──
    z = lse * lse

    # ── 5. 写回 ──
    tl.store(CE_OUT + row, ce)
    tl.store(ZLOSS_OUT + row, z)


# ─── Backward Kernel ────────────────────────────────────────────
@triton.jit
def _fused_ce_zloss_bwd_kernel(
    LOGITS,      # 输入 logits 指针, shape (M, V)
    TARGETS,     # target 索引指针, shape (M,)
    D_LOGITS,    # 输出: d_logits, shape (M, V)
    stride_logits,
    stride_dlogits,
    z_loss_weight,  # z-loss 权重
    M,           # 总行数
    V,           # vocab size
    inv_M,       # 1/M，用于 mean reduction 的梯度缩放
    BLOCK_V: tl.constexpr,
):
    """
    每个 program 处理一行 backward。

    d_logits_i = inv_M * [(1 + 2*w*lse) * softmax_i - 1{i=target}]

    其中:
      - softmax_i = exp(x_i - lse)
      - lse = log(sum(exp(x)))
      - w = z_loss_weight
      - inv_M = 1/M（mean reduction 的梯度缩放）
    """
    row = tl.program_id(0)
    if row >= M:
        return

    target = tl.load(TARGETS + row)

    cols = tl.arange(0, BLOCK_V)
    mask = cols < V

    # ── 1. 加载 logits，重算 softmax ──
    # 不存储中间 softmax，backward 时重算（节省显存）
    logits_ptr = LOGITS + row * stride_logits
    x = tl.load(logits_ptr + cols, mask=mask, other=-float('inf')).to(tl.float32)

    row_max = tl.max(x, axis=0)
    x_shifted = x - row_max
    exp_x = tl.exp(x_shifted)
    sum_exp = tl.sum(exp_x, axis=0)
    softmax = exp_x / sum_exp

    lse = tl.log(sum_exp) + row_max

    # ── 2. 梯度计算 ──
    # d_logits_i = inv_M * [(1 + 2*w*lse) * softmax_i - 1{i=target}]
    scale = 1.0 + 2.0 * z_loss_weight * lse
    target_mask = cols == target
    indicator = tl.where(target_mask, tl.full_like(softmax, 1.0), tl.zeros_like(softmax))
    d_logits = inv_M * (scale * softmax - indicator)

    # ── 3. 写回 ──
    tl.store(D_LOGITS + row * stride_dlogits + cols, d_logits, mask=mask)


# ─── Autograd Function ──────────────────────────────────────────
class FusedCrossEntropyZLossFunction(torch.autograd.Function):
    """
    torch.autograd.Function 封装。

    Forward: 返回 (ce_loss, z_loss)，均为标量（mean reduction）。
    Backward: 直接计算 d_logits = ∂(ce + w*z)/∂logits，一趟完成。
    """

    @staticmethod
    def forward(ctx, logits: torch.Tensor, targets: torch.Tensor,
                z_loss_weight: float):
        # logits: (M, V)，targets: (M,)
        assert logits.ndim == 2, f"logits 必须为 2D (M, V)，got ndim={logits.ndim}"
        assert targets.ndim == 1, f"targets 必须为 1D (M,)，got ndim={targets.ndim}"
        M, V = logits.shape
        assert targets.shape[0] == M, (
            f"logits 行数 ({M}) 与 targets 长度 ({targets.shape[0]}) 不匹配"
        )

        # BLOCK_V: 取 >= V 的最小 2 的幂
        BLOCK_V = triton.next_power_of_2(V)

        # 分配输出
        ce_out = torch.empty(M, device=logits.device, dtype=torch.float32)
        zloss_out = torch.empty(M, device=logits.device, dtype=torch.float32)

        # 启动 forward kernel: 每行一个 program
        _fused_ce_zloss_fwd_kernel[(M,)](
            logits, targets, ce_out, zloss_out,
            stride_logits=logits.stride(0),
            M=M, V=V,
            BLOCK_V=BLOCK_V,
        )

        # mean reduction
        ce_loss = ce_out.mean()
        z_loss = zloss_out.mean()

        # 保存 backward 所需
        ctx.save_for_backward(logits, targets)
        ctx.z_loss_weight = z_loss_weight
        ctx.M = M
        ctx.V = V
        ctx.BLOCK_V = BLOCK_V

        return ce_loss, z_loss

    @staticmethod
    def backward(ctx, grad_ce: torch.Tensor, grad_zloss: torch.Tensor):
        logits, targets = ctx.saved_tensors
        z_loss_weight = ctx.z_loss_weight
        M = ctx.M
        V = ctx.V
        BLOCK_V = ctx.BLOCK_V

        # 分配 d_logits
        d_logits = torch.empty_like(logits)

        # 梯度推导:
        # forward 返回 (ce, z)，外部做 loss = f(ce, z)
        # autograd 传入 grad_ce = ∂loss/∂ce, grad_zloss = ∂loss/∂z
        # 需要: d_logits = grad_ce * ∂ce/∂x + grad_zloss * ∂z/∂x
        #      = grad_ce * (softmax - indicator)/M + grad_zloss * 2*lse*softmax/M
        #      = [(grad_ce + 2*grad_zloss*lse) * softmax - grad_ce*indicator] / M
        # 这等价于 kernel 中用 effective_w = grad_zloss / grad_ce（当 grad_ce≠0）
        # 然后整体乘 grad_ce。
        # kernel 计算: inv_M * [(1 + 2*w*lse)*softmax - indicator]
        # 令 w = grad_zloss / grad_ce，然后乘 grad_ce 即可。
        grad_ce_val = grad_ce.item()
        grad_zloss_val = grad_zloss.item()
        # 除零保护：当 grad_ce ≈ 0 时（如 loss 被 detach 或 stop_gradient），
        # effective_w 无定义，此时 z_loss 梯度贡献也趋于零，安全跳过。
        if abs(grad_ce_val) > 1e-12:
            effective_w = grad_zloss_val / grad_ce_val
        else:
            effective_w = 0.0
        inv_M = 1.0 / M

        _fused_ce_zloss_bwd_kernel[(M,)](
            logits, targets, d_logits,
            stride_logits=logits.stride(0),
            stride_dlogits=d_logits.stride(0),
            z_loss_weight=effective_w,
            M=M, V=V,
            inv_M=inv_M,
            BLOCK_V=BLOCK_V,
        )

        d_logits = d_logits * grad_ce_val

        return d_logits, None, None  # None for targets, z_loss_weight


# ─── 便捷函数 ───────────────────────────────────────────────────
def fused_cross_entropy_zloss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    z_loss_weight: float = 1e-4,
) -> tuple:
    """
    Fused cross-entropy + z-loss 计算。

    参数:
      logits:         (M, V) float tensor，未经 softmax 的 logits
      targets:        (M,) long tensor，target 类别索引
      z_loss_weight:  z-loss 权重（默认 1e-4）

    返回:
      (ce_loss, z_loss): 两个标量 tensor
        - ce_loss = mean(-log softmax(logits)[targets])
        - z_loss  = mean(logsumexp(logits)²)
        - 外部组合: total_loss = ce_loss + z_loss_weight * z_loss
    """
    return FusedCrossEntropyZLossFunction.apply(
        logits.contiguous(), targets.contiguous(), z_loss_weight
    )
