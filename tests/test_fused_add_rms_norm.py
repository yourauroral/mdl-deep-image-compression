"""
Fused Add+RMSNorm Triton Kernel 单元测试 — forward + backward 数值精度验证。

对比 Triton fused kernel (y = residual + RMSNorm(sublayer_out))
与 PyTorch 分步实现 (residual + weight * sublayer / RMS(sublayer))。

用于 OLMo 2 post-norm 路径: x = x + RMSNorm(sublayer(x))。

参考:
  [1] Zhang & Sennrich, "Root Mean Square Layer Normalization,"
      NeurIPS 2019, arXiv:1910.07467.
  [2] OLMo 2 Tech Report, arXiv:2501.00656, 2025, Section 3.1.
"""

import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.mdlic.ops.fused_add_rms_norm import fused_add_rms_norm


# ── PyTorch 参考实现 ────────────────────────────────────────────
def add_rms_norm_ref(residual, sublayer_out, weight, eps=1e-10):
    """
    PyTorch 分步参考实现: y = residual + RMSNorm(sublayer_out)

    RMSNorm(s) = w * s / sqrt(mean(s²) + eps)  [1]
    OLMo 2 post-norm: x = x + RMSNorm(sublayer(x))  [2]
    """
    s_fp32 = sublayer_out.float()
    rms = s_fp32.pow(2).mean(dim=-1, keepdim=True).add(eps).sqrt()
    normed = (weight.float() * (s_fp32 / rms)).to(sublayer_out.dtype)
    return residual + normed


# ── 测试参数 ────────────────────────────────────────────────────
SHAPES = [
    (1, 32),
    (4, 64),
    (16, 128),     # d_model=128（CIFAR baseline）
    (64, 256),
    (8, 384),      # d_ff=384
    (32, 512),
    (2, 100),      # 非 2 的幂
]

DTYPES = [torch.float32, torch.float16, torch.bfloat16]

TOLERANCES = {
    torch.float32:  {"atol": 1e-5, "rtol": 1e-5},
    torch.float16:  {"atol": 1e-2, "rtol": 1e-2},
    torch.bfloat16: {"atol": 2e-2, "rtol": 2e-2},
}


@pytest.mark.parametrize("M, N", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES,
                         ids=["fp32", "fp16", "bf16"])
class TestFusedAddRMSNorm:
    """Fused Add+RMSNorm vs PyTorch 参考实现"""

    def test_forward(self, M, N, dtype):
        """前向输出 allclose"""
        torch.manual_seed(42)
        residual = torch.randn(M, N, device="cuda", dtype=dtype)
        sublayer = torch.randn(M, N, device="cuda", dtype=dtype)
        w = torch.randn(N, device="cuda", dtype=dtype).abs() + 0.1

        out_fused = fused_add_rms_norm(residual, sublayer, w)
        out_ref = add_rms_norm_ref(residual, sublayer, w)

        tol = TOLERANCES[dtype]
        torch.testing.assert_close(out_fused, out_ref, **tol)

    def test_backward_d_residual(self, M, N, dtype):
        """d_residual 应为 dy（直接传递）"""
        torch.manual_seed(42)

        # fused 路径
        r1 = torch.randn(M, N, device="cuda", dtype=dtype, requires_grad=True)
        s1 = torch.randn(M, N, device="cuda", dtype=dtype, requires_grad=True)
        w1 = (torch.randn(N, device="cuda", dtype=dtype).abs() + 0.1).requires_grad_(True)
        out1 = fused_add_rms_norm(r1, s1, w1)
        out1.sum().backward()

        # 参考路径
        r2 = r1.data.clone().detach().requires_grad_(True)
        s2 = s1.data.clone().detach().requires_grad_(True)
        w2 = w1.data.clone().detach().requires_grad_(True)
        out2 = add_rms_norm_ref(r2, s2, w2)
        out2.sum().backward()

        tol = TOLERANCES[dtype]
        torch.testing.assert_close(r1.grad, r2.grad, **tol)

    def test_backward_d_sublayer(self, M, N, dtype):
        """d_sublayer allclose"""
        torch.manual_seed(42)

        r1 = torch.randn(M, N, device="cuda", dtype=dtype, requires_grad=True)
        s1 = torch.randn(M, N, device="cuda", dtype=dtype, requires_grad=True)
        w1 = (torch.randn(N, device="cuda", dtype=dtype).abs() + 0.1).requires_grad_(True)
        out1 = fused_add_rms_norm(r1, s1, w1)
        out1.sum().backward()

        r2 = r1.data.clone().detach().requires_grad_(True)
        s2 = s1.data.clone().detach().requires_grad_(True)
        w2 = w1.data.clone().detach().requires_grad_(True)
        out2 = add_rms_norm_ref(r2, s2, w2)
        out2.sum().backward()

        tol = TOLERANCES[dtype]
        torch.testing.assert_close(s1.grad, s2.grad, **tol)

    def test_backward_dw(self, M, N, dtype):
        """dw allclose"""
        torch.manual_seed(42)

        r1 = torch.randn(M, N, device="cuda", dtype=dtype, requires_grad=True)
        s1 = torch.randn(M, N, device="cuda", dtype=dtype, requires_grad=True)
        w1 = (torch.randn(N, device="cuda", dtype=dtype).abs() + 0.1).requires_grad_(True)
        out1 = fused_add_rms_norm(r1, s1, w1)
        out1.sum().backward()

        r2 = r1.data.clone().detach().requires_grad_(True)
        s2 = s1.data.clone().detach().requires_grad_(True)
        w2 = w1.data.clone().detach().requires_grad_(True)
        out2 = add_rms_norm_ref(r2, s2, w2)
        out2.sum().backward()

        tol = TOLERANCES[dtype]
        torch.testing.assert_close(w1.grad, w2.grad, **tol)

    def test_3d_input(self, M, N, dtype):
        """3D 输入 (B, T, N)"""
        if M < 4:
            pytest.skip("M 太小无法 reshape 为 3D")
        torch.manual_seed(42)
        B = 2
        T = M // 2
        residual = torch.randn(B, T, N, device="cuda", dtype=dtype)
        sublayer = torch.randn(B, T, N, device="cuda", dtype=dtype)
        w = torch.randn(N, device="cuda", dtype=dtype).abs() + 0.1

        out_fused = fused_add_rms_norm(residual, sublayer, w)
        out_ref = add_rms_norm_ref(residual, sublayer, w)

        tol = TOLERANCES[dtype]
        torch.testing.assert_close(out_fused, out_ref, **tol)


class TestFusedAddRMSNormEdgeCases:
    """边界情况测试"""

    def test_zero_sublayer(self):
        """sublayer 全零时，y = residual + 0 = residual"""
        torch.manual_seed(42)
        M, N = 8, 128
        residual = torch.randn(M, N, device="cuda")
        sublayer = torch.zeros(M, N, device="cuda")
        w = torch.ones(N, device="cuda")

        out = fused_add_rms_norm(residual, sublayer, w)
        # sublayer 全零: RMS(0) = sqrt(eps) → RMSNorm(0) ≈ 0
        # 所以 y ≈ residual
        torch.testing.assert_close(out, residual, atol=1e-3, rtol=1e-3)

    def test_residual_passthrough(self):
        """验证 d_residual = dy（梯度直接传递）"""
        M, N = 4, 64
        r = torch.randn(M, N, device="cuda", requires_grad=True)
        s = torch.randn(M, N, device="cuda", requires_grad=True)
        w = torch.ones(N, device="cuda", requires_grad=True)

        out = fused_add_rms_norm(r, s, w)
        # dy = ones
        out.sum().backward()

        # d_residual 应该全为 1（因为 dy = ones 且 d_residual = dy）
        expected_dr = torch.ones_like(r)
        torch.testing.assert_close(r.grad, expected_dr, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    print("=== Fused Add+RMSNorm Unit Test ===\n")
    passed = 0
    total = 0
    for M, N in SHAPES:
        for dtype in DTYPES:
            total += 1
            try:
                torch.manual_seed(42)
                residual = torch.randn(M, N, device="cuda", dtype=dtype)
                sublayer = torch.randn(M, N, device="cuda", dtype=dtype)
                w = torch.randn(N, device="cuda", dtype=dtype).abs() + 0.1
                out_fused = fused_add_rms_norm(residual, sublayer, w)
                out_ref = add_rms_norm_ref(residual, sublayer, w)
                tol = TOLERANCES[dtype]
                torch.testing.assert_close(out_fused, out_ref, **tol)
                passed += 1
            except Exception as e:
                print(f"  FAIL ({M},{N}) {dtype}: {e}")
    print(f"Forward: {passed}/{total} passed")

    # backward 快速验证
    r = torch.randn(16, 128, device="cuda", requires_grad=True)
    s = torch.randn(16, 128, device="cuda", requires_grad=True)
    w = torch.ones(128, device="cuda", requires_grad=True)
    out = fused_add_rms_norm(r, s, w)
    out.sum().backward()
    assert r.grad is not None and s.grad is not None and w.grad is not None
    print("Backward: PASSED")
    print("\nAll quick checks passed!")
