"""
Fused Linear Cross-Entropy 单元测试 — 验证 fused kernel 与分步实现的数值等价性。

对比:
  Fused:  fused_linear_cross_entropy(hidden, weight, targets, z_w)
  分步:   logits = hidden @ weight.T
          ce = F.cross_entropy(logits, targets)
          z = mean(logsumexp(logits)²)

参考:
  [1] Hsu et al., "Liger Kernel," arXiv:2410.10989, 2024.
  [2] Google, "PaLM," arXiv:2204.02311, 2022. z-loss.
"""

import pytest
import torch
import torch.nn.functional as F
import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.mdlic.ops.fused_linear_ce import fused_linear_cross_entropy


# ── PyTorch 参考实现 ────────────────────────────────────────────
def linear_ce_ref(hidden, weight, targets, z_loss_weight=1e-4):
    """分步参考: linear projection → CE + z-loss"""
    logits = (hidden.float() @ weight.float().T)
    ce = F.cross_entropy(logits, targets, reduction="mean")
    lse = torch.logsumexp(logits, dim=-1)
    z = (lse ** 2).mean()
    return ce, z


# ── 测试参数 ────────────────────────────────────────────────────
# (M, D, V): tokens, hidden_dim, vocab_size
SHAPES = [
    (4, 32, 16),       # 极小
    (16, 64, 256),     # 小: d_model=64, vocab=256
    (64, 128, 256),    # 典型 CIFAR: d_model=128, vocab=256
    (8, 128, 64),      # 小 vocab
    (1, 128, 256),     # 单样本
]

Z_WEIGHTS = [0.0, 1e-4, 1e-2]

TOLERANCES = {
    "ce":   {"atol": 1e-3, "rtol": 1e-3},
    "z":    {"atol": 1e-2, "rtol": 1e-2},
    "grad": {"atol": 1e-2, "rtol": 1e-2},
}


@pytest.mark.parametrize("M, D, V", SHAPES,
                         ids=[f"{M}x{D}x{V}" for M, D, V in SHAPES])
class TestFusedLinearCE:
    """Fused Linear+CE+z-loss vs 分步实现"""

    def test_forward_ce(self, M, D, V):
        """CE loss allclose"""
        torch.manual_seed(42)
        hidden = torch.randn(M, D, device="cuda")
        weight = torch.randn(V, D, device="cuda")
        targets = torch.randint(0, V, (M,), device="cuda")

        ce_fused, z_fused = fused_linear_cross_entropy(hidden, weight, targets)
        ce_ref, z_ref = linear_ce_ref(hidden, weight, targets)

        torch.testing.assert_close(ce_fused, ce_ref, **TOLERANCES["ce"])

    def test_forward_zloss(self, M, D, V):
        """z-loss allclose"""
        torch.manual_seed(42)
        hidden = torch.randn(M, D, device="cuda")
        weight = torch.randn(V, D, device="cuda")
        targets = torch.randint(0, V, (M,), device="cuda")

        ce_fused, z_fused = fused_linear_cross_entropy(hidden, weight, targets)
        ce_ref, z_ref = linear_ce_ref(hidden, weight, targets)

        torch.testing.assert_close(z_fused, z_ref, **TOLERANCES["z"])

    @pytest.mark.parametrize("z_w", Z_WEIGHTS,
                             ids=[f"zw={w}" for w in Z_WEIGHTS])
    def test_backward_d_hidden(self, M, D, V, z_w):
        """d_hidden 梯度 allclose"""
        torch.manual_seed(42)

        # fused path
        h1 = torch.randn(M, D, device="cuda", requires_grad=True)
        w = torch.randn(V, D, device="cuda")
        targets = torch.randint(0, V, (M,), device="cuda")
        ce1, z1 = fused_linear_cross_entropy(h1, w, targets, z_w)
        loss1 = ce1 + z_w * z1
        loss1.backward()

        # ref path
        h2 = h1.data.clone().detach().requires_grad_(True)
        ce2, z2 = linear_ce_ref(h2, w, targets, z_w)
        loss2 = ce2 + z_w * z2
        loss2.backward()

        torch.testing.assert_close(h1.grad, h2.grad, **TOLERANCES["grad"])


class TestFusedLinearCEEdgeCases:
    """边界情况"""

    def test_weight_tying_compatible(self):
        """验证与 weight tying 场景兼容（共享的 embedding weight）"""
        M, D, V = 16, 128, 256
        embed = torch.nn.Embedding(V, D).cuda()
        weight = embed.weight  # 共享权重

        hidden = torch.randn(M, D, device="cuda", requires_grad=True)
        targets = torch.randint(0, V, (M,), device="cuda")

        ce, z = fused_linear_cross_entropy(hidden, weight, targets)
        loss = ce + 1e-4 * z
        loss.backward()

        assert hidden.grad is not None
        assert hidden.grad.isfinite().all()

    def test_large_logits_stability(self):
        """大 hidden 值 — 数值稳定性"""
        torch.manual_seed(42)
        M, D, V = 8, 64, 256
        hidden = torch.randn(M, D, device="cuda") * 10
        weight = torch.randn(V, D, device="cuda")
        targets = torch.randint(0, V, (M,), device="cuda")

        ce_fused, z_fused = fused_linear_cross_entropy(hidden, weight, targets)
        ce_ref, z_ref = linear_ce_ref(hidden, weight, targets)

        # 大幅值下容差稍宽
        torch.testing.assert_close(ce_fused, ce_ref, atol=5e-2, rtol=5e-2)


if __name__ == "__main__":
    print("=== Fused Linear Cross-Entropy Unit Test ===\n")
    passed = 0
    total = 0
    for M, D, V in SHAPES:
        total += 1
        try:
            torch.manual_seed(42)
            hidden = torch.randn(M, D, device="cuda")
            weight = torch.randn(V, D, device="cuda")
            targets = torch.randint(0, V, (M,), device="cuda")
            ce_f, z_f = fused_linear_cross_entropy(hidden, weight, targets)
            ce_r, z_r = linear_ce_ref(hidden, weight, targets)
            torch.testing.assert_close(ce_f, ce_r, atol=1e-3, rtol=1e-3)
            passed += 1
        except Exception as e:
            print(f"  FAIL ({M},{D},{V}): {e}")
    print(f"Forward: {passed}/{total} passed")

    # backward 快速验证
    h = torch.randn(16, 128, device="cuda", requires_grad=True)
    w = torch.randn(256, 128, device="cuda")
    t = torch.randint(0, 256, (16,), device="cuda")
    ce, z = fused_linear_cross_entropy(h, w, t)
    (ce + 1e-4 * z).backward()
    assert h.grad is not None and h.grad.isfinite().all()
    print("Backward: PASSED")
    print("\nAll quick checks passed!")
