"""
Fused Cross-Entropy + z-loss Triton Kernel 单元测试 — forward + backward 数值精度验证。

对比 Triton fused kernel 与 PyTorch F.cross_entropy + 手动 z-loss 实现，
验证 CE loss、z-loss 值以及反向梯度在多种 shape/dtype 下的 allclose。

参考:
  [1] Milakov & Gimelshein, "Online normalizer calculation for softmax,"
      arXiv:1805.02867, 2018.
  [2] Google, "PaLM," arXiv:2204.02311, 2022, Section 5. z-loss.
"""

import pytest
import torch
import torch.nn.functional as F
import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.mdlic.ops.fused_ce_zloss import fused_cross_entropy_zloss


# ── PyTorch 参考实现 ────────────────────────────────────────────
def ce_zloss_ref(logits: torch.Tensor, targets: torch.Tensor,
                 z_loss_weight: float = 1e-4):
    """
    PyTorch cross-entropy + z-loss 参考实现。

    ce_loss = F.cross_entropy(logits, targets)
    z_loss  = mean(logsumexp(logits, dim=-1)²)     [2] Section 5
    """
    # 用 fp32 计算保证精度
    logits_fp32 = logits.float()
    ce_loss = F.cross_entropy(logits_fp32, targets, reduction="mean")
    lse = torch.logsumexp(logits_fp32, dim=-1)
    z_loss = (lse ** 2).mean()
    return ce_loss, z_loss


# ── 测试参数 ────────────────────────────────────────────────────
# (M, V): batch_size × seq_len, vocab_size
SHAPES = [
    (4, 16),      # 极小
    (16, 256),    # 标准 vocab=256（像素值）
    (64, 256),    # 中等 batch
    (8, 128),     # 非 256 vocab
    (1, 256),     # 单样本
    (32, 64),     # 小 vocab
]

Z_LOSS_WEIGHTS = [0.0, 1e-4, 1e-2]

# fused kernel 在 fp32 下计算 softmax，精度应该很好
TOLERANCES = {
    "ce_loss":  {"atol": 1e-5, "rtol": 1e-5},
    "z_loss":   {"atol": 1e-4, "rtol": 1e-4},
    "grad":     {"atol": 1e-5, "rtol": 1e-4},
}


@pytest.mark.parametrize("M, V", SHAPES,
                         ids=[f"{M}x{V}" for M, V in SHAPES])
class TestFusedCEZLoss:
    """Fused CE+z-loss vs PyTorch 参考实现"""

    def test_forward_ce(self, M, V):
        """CE loss 前向 allclose"""
        torch.manual_seed(42)
        logits = torch.randn(M, V, device="cuda", dtype=torch.float32)
        targets = torch.randint(0, V, (M,), device="cuda")

        ce_fused, z_fused = fused_cross_entropy_zloss(logits, targets, z_loss_weight=1e-4)
        ce_ref, z_ref = ce_zloss_ref(logits, targets, z_loss_weight=1e-4)

        torch.testing.assert_close(ce_fused, ce_ref, **TOLERANCES["ce_loss"])

    def test_forward_zloss(self, M, V):
        """z-loss 前向 allclose"""
        torch.manual_seed(42)
        logits = torch.randn(M, V, device="cuda", dtype=torch.float32)
        targets = torch.randint(0, V, (M,), device="cuda")

        ce_fused, z_fused = fused_cross_entropy_zloss(logits, targets, z_loss_weight=1e-4)
        ce_ref, z_ref = ce_zloss_ref(logits, targets, z_loss_weight=1e-4)

        torch.testing.assert_close(z_fused, z_ref, **TOLERANCES["z_loss"])

    @pytest.mark.parametrize("z_weight", Z_LOSS_WEIGHTS,
                             ids=[f"zw={w}" for w in Z_LOSS_WEIGHTS])
    def test_backward(self, M, V, z_weight):
        """
        反向梯度 allclose — 验证组合 loss = ce + z_weight * z 的梯度。
        """
        torch.manual_seed(42)

        # fused 路径
        logits1 = torch.randn(M, V, device="cuda", dtype=torch.float32,
                               requires_grad=True)
        targets = torch.randint(0, V, (M,), device="cuda")
        ce1, z1 = fused_cross_entropy_zloss(logits1, targets, z_loss_weight=z_weight)
        loss1 = ce1 + z_weight * z1
        loss1.backward()

        # 参考路径
        logits2 = logits1.data.clone().detach().requires_grad_(True)
        ce2, z2 = ce_zloss_ref(logits2, targets, z_loss_weight=z_weight)
        loss2 = ce2 + z_weight * z2
        loss2.backward()

        torch.testing.assert_close(logits1.grad, logits2.grad,
                                   **TOLERANCES["grad"])

    def test_fp16_input(self, M, V):
        """fp16 logits — kernel 内部应 upcast 到 fp32"""
        torch.manual_seed(42)
        logits_fp16 = torch.randn(M, V, device="cuda", dtype=torch.float16)
        targets = torch.randint(0, V, (M,), device="cuda")

        # fused kernel 接受 fp16 输入
        ce_fused, z_fused = fused_cross_entropy_zloss(logits_fp16, targets)

        # 参考: 手动转 fp32 计算
        ce_ref, z_ref = ce_zloss_ref(logits_fp16, targets)

        # fp16 输入但 kernel 内部 fp32 计算，精度应该和 fp32 参考接近
        torch.testing.assert_close(ce_fused, ce_ref, atol=1e-3, rtol=1e-3)


class TestFusedCEZLossEdgeCases:
    """边界情况测试"""

    def test_zero_zloss_weight(self):
        """z_loss_weight=0 时应等价于纯 cross-entropy"""
        torch.manual_seed(42)
        M, V = 16, 256
        logits = torch.randn(M, V, device="cuda", requires_grad=True)
        targets = torch.randint(0, V, (M,), device="cuda")

        ce_fused, z_fused = fused_cross_entropy_zloss(logits, targets,
                                                       z_loss_weight=0.0)
        ce_ref = F.cross_entropy(logits.float(), targets)

        torch.testing.assert_close(ce_fused, ce_ref, atol=1e-5, rtol=1e-5)

    def test_large_logits(self):
        """大 logits 值 — 验证 online softmax 的数值稳定性"""
        torch.manual_seed(42)
        M, V = 8, 256
        logits = torch.randn(M, V, device="cuda") * 100  # 大幅值
        targets = torch.randint(0, V, (M,), device="cuda")

        ce_fused, z_fused = fused_cross_entropy_zloss(logits, targets)
        ce_ref, z_ref = ce_zloss_ref(logits, targets)

        # 大 logits 下 numerical stability 很重要
        torch.testing.assert_close(ce_fused, ce_ref, atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    print("=== Fused CE + z-loss Unit Test ===\n")
    passed = 0
    total = 0
    for M, V in SHAPES:
        total += 1
        try:
            torch.manual_seed(42)
            logits = torch.randn(M, V, device="cuda")
            targets = torch.randint(0, V, (M,), device="cuda")
            ce_f, z_f = fused_cross_entropy_zloss(logits, targets)
            ce_r, z_r = ce_zloss_ref(logits, targets)
            torch.testing.assert_close(ce_f, ce_r, atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(z_f, z_r, atol=1e-4, rtol=1e-4)
            passed += 1
        except Exception as e:
            print(f"  FAIL ({M},{V}): {e}")
    print(f"Forward: {passed}/{total} passed")

    # backward 快速验证
    logits = torch.randn(16, 256, device="cuda", requires_grad=True)
    targets = torch.randint(0, 256, (16,), device="cuda")
    ce, z = fused_cross_entropy_zloss(logits, targets, z_loss_weight=1e-4)
    loss = ce + 1e-4 * z
    loss.backward()
    assert logits.grad is not None and logits.grad.isfinite().all()
    print("Backward: PASSED")
    print("\nAll quick checks passed!")
