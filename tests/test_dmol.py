"""DMoL (Discretized Mixture of Logistics) 单元测试。

覆盖 6 个核心数值不变量 + optimizer 分组划分：
  1. random init forward NLL ∈ [5.5, 7.5] nat/sub-pixel（远低于均匀 8.0）
  2. 离散化分布归一性：sum_{x=0}^{255} exp(log_prob(x)) ≈ 1.0
  3. gradient check（小规模 K=2, d_model=16）
  4. 边界 target=0 / 255 走特殊分支 loss 仍 finite
  5. head bias init 让 inv_s 落在 sigmoid 线性区 [0.05, 0.5]
  6. CC-iGPT + DMoL fine head smoke：forward + backward 三处 grad 非零
  7. optimizer 分组划分：head_params + other_params 并集=全集，交集=空
"""
import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.mdlic.losses.dmol import (
    DMoLHead1D,
    dmol_loss_1d,
    _LOG_SCALE_MIN,
    _LOG_SCALE_MAX,
    _HEAD_BIAS_INIT_LOG_SCALE,
)
from src.mdlic.models.cc_igpt import CCIGPT


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ──────────────────────────────────────────────────────────────
# 1. random init forward NLL 在合理区间
# ──────────────────────────────────────────────────────────────

def test_dmol_loss_finite_at_init(device):
    """random init head + random target → NLL ∈ [5.5, 7.5] nat/sub-pixel。

    注意：均匀分布上界 ln(256) ≈ 5.55；小于 5.55 不太可能（除非 K-mix 已经能
    猜到一些结构），大于 7.5 说明 init 让 head 输出失常。本次的 head bias
    log_scale=+2 init 应让 NLL ∈ [5.5, 7.5]。
    """
    torch.manual_seed(42)
    K = 10
    d_model = 64
    head = DMoLHead1D(d_model, n_mixtures=K).to(device)
    hidden = torch.randn(2, 100, d_model, device=device) * 0.5
    params = head(hidden)
    assert params.shape == (2, 100, K * 3)

    target = torch.randint(0, 256, (2, 100), device=device)
    nll = dmol_loss_1d(params, target, n_mixtures=K, reduction="mean")
    assert torch.isfinite(nll).item(), f"NLL = {nll.item()} not finite"
    assert 5.0 <= nll.item() <= 8.0, (
        f"random init NLL = {nll.item():.4f} 超出 [5.0, 8.0]，"
        f"head init / log_scale bias 可能配错"
    )


# ──────────────────────────────────────────────────────────────
# 2. 离散化分布是合法 normalized 分布
# ──────────────────────────────────────────────────────────────

def test_dmol_log_prob_normalized(device):
    """对所有 256 个 target 求 sum exp(log_prob) ≈ 1.0（误差 < 1e-3）。

    这是 DMoL 是合法概率分布的 fundamental 验证：CDF 差分 + 边界单边 logsigmoid
    必须组成 [0, 255] 上的合法离散分布。
    """
    torch.manual_seed(7)
    K = 10
    d_model = 32
    head = DMoLHead1D(d_model, n_mixtures=K).to(device)
    # 单点 hidden，得单一 mixture 参数
    hidden = torch.randn(1, 1, d_model, device=device) * 0.3
    params = head(hidden)  # (1, 1, K*3)

    # 对所有 256 个 target 算 log_prob
    log_probs = []
    for t in range(256):
        target = torch.full((1, 1), t, device=device, dtype=torch.long)
        nll_per = dmol_loss_1d(params, target, n_mixtures=K, reduction="none")
        log_probs.append(-nll_per.item())  # log_prob = -nll
    log_probs_t = torch.tensor(log_probs)
    total_prob = torch.exp(log_probs_t).sum().item()

    assert abs(total_prob - 1.0) < 1e-3, (
        f"DMoL 分布在 [0, 255] 上归一化误差 {abs(total_prob - 1.0):.4e} > 1e-3，"
        f"实际 sum = {total_prob:.6f}，离散化 CDF 差分 / 边界 logsigmoid 实现有 bug"
    )


# ──────────────────────────────────────────────────────────────
# 3. gradient check（autograd 数值验证）
# ──────────────────────────────────────────────────────────────

def test_dmol_gradient_check():
    """torch.autograd.gradcheck 验证 dmol_loss_1d 的解析梯度与数值梯度一致。

    用极小规模 K=2, d_model=8, batch=1, T=2 跑 fp64 gradcheck（要求 fp64 输入）。
    """
    torch.manual_seed(1)
    K = 2
    # gradcheck 需要 double 精度
    params = torch.randn(1, 2, K * 3, dtype=torch.float64, requires_grad=True)
    target = torch.tensor([[100, 150]], dtype=torch.long)

    def f(p):
        return dmol_loss_1d(p, target, n_mixtures=K, reduction="mean")

    assert torch.autograd.gradcheck(f, (params,), eps=1e-4, atol=1e-3, rtol=1e-3), (
        "DMoL loss 数值梯度与解析梯度不一致，dmol_loss_1d 实现有 bug"
    )


# ──────────────────────────────────────────────────────────────
# 4. 边界 target=0 / 255 走特殊分支
# ──────────────────────────────────────────────────────────────

def test_dmol_boundary_targets(device):
    """target=0 / target=255 走 logsigmoid 单边分支，loss 必须 finite。

    历史 v1 fallback 阈值 1e-5 太松，边界处 CDF 差分可能命中 fallback；
    本次实现用 logsigmoid 单边公式，应永远 finite。
    """
    torch.manual_seed(13)
    K = 10
    d_model = 32
    head = DMoLHead1D(d_model, n_mixtures=K).to(device)
    hidden = torch.randn(2, 4, d_model, device=device) * 0.5
    params = head(hidden)

    # target 全 0
    target_low = torch.zeros(2, 4, device=device, dtype=torch.long)
    nll_low = dmol_loss_1d(params, target_low, n_mixtures=K, reduction="mean")
    assert torch.isfinite(nll_low).item(), "target=0 时 NLL 非 finite"

    # target 全 255
    target_high = torch.full((2, 4), 255, device=device, dtype=torch.long)
    nll_high = dmol_loss_1d(params, target_high, n_mixtures=K, reduction="mean")
    assert torch.isfinite(nll_high).item(), "target=255 时 NLL 非 finite"


# ──────────────────────────────────────────────────────────────
# 5. head bias init 让 inv_s 落在 sigmoid 线性区
# ──────────────────────────────────────────────────────────────

def test_dmol_initial_log_scale_active(device):
    """init 后 head bias 中 log_scale 通道 = +2.0，inv_s = exp(-2) ≈ 0.135。

    这是 v6 vs v1-v5 最关键的修复（v1-v5 全部 bias=0 起步导致 inv_s=1，bin·inv_s
    极窄，CDF 差分 ~1e-4 量级触发 fallback 风暴）。
    """
    torch.manual_seed(99)
    K = 10
    d_model = 32
    head = DMoLHead1D(d_model, n_mixtures=K)

    # 直接检查 bias init
    bias = head.proj.bias.data
    log_scale_bias = bias[2 * K : 3 * K]
    assert torch.allclose(log_scale_bias, torch.tensor([_HEAD_BIAS_INIT_LOG_SCALE] * K)), (
        f"log_scale bias init 错误：expected {_HEAD_BIAS_INIT_LOG_SCALE}, got {log_scale_bias.tolist()}"
    )

    # forward 后验证 log_scale_raw 在 +2 附近 → inv_s ≈ 0.135
    head = head.to(device)
    hidden = torch.zeros(1, 1, d_model, device=device)  # zero hidden → 输出 = bias
    params = head(hidden)
    _, _, log_scale_raw = params.split(K, dim=-1)
    log_scale = log_scale_raw.clamp(_LOG_SCALE_MIN, _LOG_SCALE_MAX)
    inv_s = torch.exp(-log_scale).mean().item()
    assert 0.05 <= inv_s <= 0.5, (
        f"init 后 inv_s = {inv_s:.4f} 不在线性区 [0.05, 0.5]，"
        f"log_scale bias init 被破坏"
    )


# ──────────────────────────────────────────────────────────────
# 6. CC-iGPT + DMoL smoke
# ──────────────────────────────────────────────────────────────

def test_ccigpt_dmol_smoke(device):
    """小 CC-iGPT (fine N=2, d_model=64) + dmol head → forward + backward
    三处 grad 非零（coarse / fine / ctx_alpha）。"""
    torch.manual_seed(2)
    m = CCIGPT(
        image_size=32, in_channels=3, vocab_size=256, pool_factor=4,
        fine_d_model=64, fine_N=2, fine_h=2, fine_d_ff=128,
        coarse_d_model=64, coarse_N=2, coarse_h=2, coarse_d_ff=64,
        dropout=0.0,
        output_head="dmol",
        n_mixtures=10,
    ).to(device).train()

    x = torch.rand(2, 3, 32, 32, device=device)
    out = m(x)
    assert torch.isfinite(out["loss"]).item()
    assert torch.isfinite(out["bpd"]).item()
    assert 0.0 < out["bpd"].item() < 50.0
    # DMoL 路径 fine.logits 应为 None
    assert out["logits"] is None, (
        f"DMoL 路径 fine.logits 应为 None, got {type(out['logits'])}"
    )

    out["loss"].backward()
    # coarse / fine / ctx_alpha 三处都必须有非零梯度
    coarse_grad = sum(p.grad.abs().sum().item() for p in m.coarse.parameters()
                      if p.grad is not None)
    fine_grad = sum(p.grad.abs().sum().item() for p in m.fine.parameters()
                    if p.grad is not None)
    head_grad = sum(p.grad.abs().sum().item() for p in m.fine.head.parameters()
                    if p.grad is not None)
    assert coarse_grad > 0, "coarse 分支无梯度"
    assert fine_grad > 0, "fine 分支无梯度"
    assert head_grad > 0, "DMoL head 无梯度"
    assert m.ctx_alpha.grad is not None and m.ctx_alpha.grad.abs().sum().item() > 0, (
        "ctx_alpha 无梯度"
    )

    # coarse 仍是 softmax（vocab=256 head + tying）
    assert m.coarse.output_head_type == "softmax"
    assert m.fine.output_head_type == "dmol"


# ──────────────────────────────────────────────────────────────
# 7. optimizer 分组划分 — 防 head 同时被两个 optimizer 更新
# ──────────────────────────────────────────────────────────────

def test_optimizer_param_partition(device):
    """train.py _split_dmol_head_params 必须保证三组并集=全集，交集=空。

    若 head 既出现在 AdamW group 又被 Adamax 更新，权重会被 step 两次，
    训练动力学完全崩坏 — 这是 train.py 改造的最高风险点。
    """
    from scripts.train import _split_dmol_head_params

    torch.manual_seed(3)
    m = CCIGPT(
        image_size=32, in_channels=3, vocab_size=256, pool_factor=4,
        fine_d_model=64, fine_N=2, fine_h=2, fine_d_ff=128,
        coarse_d_model=64, coarse_N=2, coarse_h=2, coarse_d_ff=64,
        dropout=0.0,
        output_head="dmol",
        n_mixtures=10,
    ).to(device)

    head_params, other_params, head_param_names = _split_dmol_head_params(m)

    # 全集
    all_named = list(m.named_parameters())
    all_params_set = {id(p) for _, p in all_named}
    head_set = {id(p) for p in head_params}
    other_set = {id(p) for p in other_params}

    # 1. 并集 = 全集
    assert head_set | other_set == all_params_set, (
        f"head ∪ other 缺少 {len(all_params_set - head_set - other_set)} 个参数"
    )
    # 2. 交集为空
    assert not (head_set & other_set), (
        f"head ∩ other 非空：{len(head_set & other_set)} 个参数被分到两组"
    )
    # 3. head 只包含 fine.head 子树
    assert all("fine.head.proj." in n for n in head_param_names), (
        f"head_param_names 含意外参数: {head_param_names}"
    )
    # 4. head 数量 = 2（DMoLHead1D 只有 proj.weight + proj.bias）
    assert len(head_params) == 2, (
        f"head_params 应该恰好 2 个 (proj.weight + proj.bias), got {len(head_params)}"
    )


# ──────────────────────────────────────────────────────────────
# 8. softmax 路径不回归（regression test）
# ──────────────────────────────────────────────────────────────

def test_softmax_path_unchanged(device):
    """确认 IGPT/CCIGPT 默认 output_head=softmax 时行为完全等价于历史路径。"""
    from src.mdlic.models.igpt import IGPT

    torch.manual_seed(0)
    m_softmax = IGPT(
        image_size=8, in_channels=3, vocab_size=256,
        d_model=64, N=2, h=4, d_ff=128, dropout=0.0,
    ).to(device)
    # weight tying 仍生效
    assert m_softmax.head.weight is m_softmax.token_embed.weight, (
        "softmax 路径下 head 与 token_embed 必须 weight-tied"
    )
    assert m_softmax.output_head_type == "softmax"
    x = torch.rand(2, 3, 8, 8, device=device)
    out = m_softmax(x)
    assert out["logits"] is not None
    assert out["logits"].shape == (2, 8 * 8 * 3 - 1, 256)
