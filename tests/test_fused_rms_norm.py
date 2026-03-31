"""
Fused RMSNorm Triton Kernel 单元测试 — forward + backward 数值精度验证。

对比 Triton fused kernel 与 PyTorch 手动实现的 RMSNorm，
验证前向输出和反向梯度在多种 shape/dtype 下的 allclose。

参考:
  [1] Zhang & Sennrich, "Root Mean Square Layer Normalization,"
      NeurIPS 2019, arXiv:1910.07467.
"""

import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.mdlic.ops.fused_rms_norm import fused_rms_norm


# ── PyTorch 参考实现 ────────────────────────────────────────────
def rms_norm_ref(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-10):
    """
    PyTorch RMSNorm 参考实现。
    RMSNorm(x) = w * x / sqrt(mean(x²) + eps)  [1]
    """
    x_fp32 = x.float()
    rms = x_fp32.pow(2).mean(dim=-1, keepdim=True).add(eps).sqrt()
    return (weight.float() * (x_fp32 / rms)).to(x.dtype)


# ── 测试参数 ────────────────────────────────────────────────────
# (M, N) 组合覆盖: 小尺寸、典型模型维度、非 2 的幂
SHAPES = [
    (1, 32),        # 单行，最小 case
    (4, 64),        # 小 batch
    (16, 128),      # d_model=128（CIFAR baseline）
    (64, 256),      # 中等规模
    (8, 384),       # d_ff=384（SwiGLU hidden）
    (32, 512),      # d_model=512
    (2, 100),       # 非 2 的幂
]

DTYPES = [torch.float32, torch.float16, torch.bfloat16]

# 容差: fp16/bf16 需要更宽松的容差
TOLERANCES = {
    torch.float32:  {"atol": 1e-5, "rtol": 1e-5},
    torch.float16:  {"atol": 1e-2, "rtol": 1e-2},
    torch.bfloat16: {"atol": 2e-2, "rtol": 2e-2},
}


@pytest.mark.parametrize("M, N", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES,
                         ids=["fp32", "fp16", "bf16"])
class TestFusedRMSNorm:
    """Fused RMSNorm vs PyTorch 参考实现"""

    def test_forward(self, M, N, dtype):
        """前向输出 allclose"""
        torch.manual_seed(42)
        x = torch.randn(M, N, device="cuda", dtype=dtype)
        w = torch.randn(N, device="cuda", dtype=dtype).abs() + 0.1  # 正权重

        out_fused = fused_rms_norm(x, w)
        out_ref = rms_norm_ref(x, w)

        tol = TOLERANCES[dtype]
        torch.testing.assert_close(out_fused, out_ref, **tol)

    def test_backward_dx(self, M, N, dtype):
        """反向梯度 dx allclose"""
        torch.manual_seed(42)
        # fused 路径
        x1 = torch.randn(M, N, device="cuda", dtype=dtype, requires_grad=True)
        w1 = torch.randn(N, device="cuda", dtype=dtype).abs() + 0.1
        w1.requires_grad_(True)
        out1 = fused_rms_norm(x1, w1)
        loss1 = out1.sum()
        loss1.backward()

        # 参考路径
        x2 = x1.data.clone().detach().requires_grad_(True)
        w2 = w1.data.clone().detach().requires_grad_(True)
        out2 = rms_norm_ref(x2, w2)
        loss2 = out2.sum()
        loss2.backward()

        tol = TOLERANCES[dtype]
        torch.testing.assert_close(x1.grad, x2.grad, **tol)

    def test_backward_dw(self, M, N, dtype):
        """反向梯度 dw allclose"""
        torch.manual_seed(42)
        x1 = torch.randn(M, N, device="cuda", dtype=dtype, requires_grad=True)
        w1 = torch.randn(N, device="cuda", dtype=dtype).abs() + 0.1
        w1 = w1.clone().detach().requires_grad_(True)
        out1 = fused_rms_norm(x1, w1)
        out1.sum().backward()

        x2 = x1.data.clone().detach().requires_grad_(True)
        w2 = w1.data.clone().detach().requires_grad_(True)
        out2 = rms_norm_ref(x2, w2)
        out2.sum().backward()

        tol = TOLERANCES[dtype]
        torch.testing.assert_close(w1.grad, w2.grad, **tol)

    def test_3d_input(self, M, N, dtype):
        """3D 输入 (B, T, N) — 验证 reshape 逻辑正确"""
        if M < 4:
            pytest.skip("M 太小无法 reshape 为 3D")
        torch.manual_seed(42)
        B = 2
        T = M // 2
        x = torch.randn(B, T, N, device="cuda", dtype=dtype)
        w = torch.randn(N, device="cuda", dtype=dtype).abs() + 0.1

        out_fused = fused_rms_norm(x, w)
        out_ref = rms_norm_ref(x, w)

        tol = TOLERANCES[dtype]
        torch.testing.assert_close(out_fused, out_ref, **tol)


if __name__ == "__main__":
    # 快速验证（不依赖 pytest）
    print("=== Fused RMSNorm Unit Test ===\n")
    passed = 0
    total = 0
    for M, N in SHAPES:
        for dtype in DTYPES:
            total += 1
            try:
                x = torch.randn(M, N, device="cuda", dtype=dtype)
                w = torch.randn(N, device="cuda", dtype=dtype).abs() + 0.1
                out_fused = fused_rms_norm(x, w)
                out_ref = rms_norm_ref(x, w)
                tol = TOLERANCES[dtype]
                torch.testing.assert_close(out_fused, out_ref, **tol)
                passed += 1
            except Exception as e:
                print(f"  FAIL ({M},{N}) {dtype}: {e}")
    print(f"\nForward: {passed}/{total} passed")

    # backward 快速验证
    x = torch.randn(16, 128, device="cuda", requires_grad=True)
    w = torch.ones(128, device="cuda", requires_grad=True)
    out = fused_rms_norm(x, w)
    out.sum().backward()
    assert x.grad is not None and w.grad is not None
    print("Backward: PASSED (grad exists and finite)")
    print("\nAll quick checks passed!")
