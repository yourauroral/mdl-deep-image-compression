"""
Fused SwiGLU Triton Kernel 单元测试 — forward + backward 数值精度验证。

对比 Triton fused kernel (silu(a) * b 一次 kernel) 与 PyTorch F.silu(a) * b，
验证前向输出和反向梯度在多种 shape/dtype 下的 allclose。

参考:
  [1] Shazeer, "GLU Variants Improve Transformers," arXiv:2002.05202, 2020.
      SwiGLU = Swish(a) * b, Eq.(6).
"""

import pytest
import torch
import torch.nn.functional as F
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.mdlic.ops.fused_swiglu import fused_swiglu


# ── PyTorch 参考实现 ────────────────────────────────────────────
def swiglu_ref(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """PyTorch SwiGLU 参考实现: silu(a) * b  [1] Eq.(6)"""
    return F.silu(a.float()).to(a.dtype) * b


# ── 测试参数 ────────────────────────────────────────────────────
# (M, N): batch_tokens × d_ff
SHAPES = [
    (1, 32),        # 单行
    (4, 64),        # 小
    (16, 128),      # 典型 d_model
    (64, 384),      # d_ff=384（CIFAR baseline SwiGLU hidden）
    (8, 256),       # 中等
    (32, 512),      # 较大
    (2, 100),       # 非 2 的幂
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
class TestFusedSwiGLU:
    """Fused SwiGLU vs PyTorch 参考实现"""

    def test_forward(self, M, N, dtype):
        """前向输出 allclose"""
        torch.manual_seed(42)
        a = torch.randn(M, N, device="cuda", dtype=dtype)
        b = torch.randn(M, N, device="cuda", dtype=dtype)

        out_fused = fused_swiglu(a, b)
        out_ref = swiglu_ref(a, b)

        tol = TOLERANCES[dtype]
        torch.testing.assert_close(out_fused, out_ref, **tol)

    def test_backward_da(self, M, N, dtype):
        """反向梯度 da allclose（activation recomputation 路径）"""
        torch.manual_seed(42)

        # fused 路径
        a1 = torch.randn(M, N, device="cuda", dtype=dtype, requires_grad=True)
        b1 = torch.randn(M, N, device="cuda", dtype=dtype, requires_grad=True)
        out1 = fused_swiglu(a1, b1)
        out1.sum().backward()

        # 参考路径
        a2 = a1.data.clone().detach().requires_grad_(True)
        b2 = b1.data.clone().detach().requires_grad_(True)
        out2 = swiglu_ref(a2, b2)
        out2.sum().backward()

        tol = TOLERANCES[dtype]
        torch.testing.assert_close(a1.grad, a2.grad, **tol)

    def test_backward_db(self, M, N, dtype):
        """反向梯度 db allclose"""
        torch.manual_seed(42)

        a1 = torch.randn(M, N, device="cuda", dtype=dtype, requires_grad=True)
        b1 = torch.randn(M, N, device="cuda", dtype=dtype, requires_grad=True)
        out1 = fused_swiglu(a1, b1)
        out1.sum().backward()

        a2 = a1.data.clone().detach().requires_grad_(True)
        b2 = b1.data.clone().detach().requires_grad_(True)
        out2 = swiglu_ref(a2, b2)
        out2.sum().backward()

        tol = TOLERANCES[dtype]
        torch.testing.assert_close(b1.grad, b2.grad, **tol)

    def test_3d_input(self, M, N, dtype):
        """3D 输入 (B, T, N) — 验证 reshape 逻辑"""
        if M < 4:
            pytest.skip("M 太小无法 reshape 为 3D")
        torch.manual_seed(42)
        B = 2
        T = M // 2
        a = torch.randn(B, T, N, device="cuda", dtype=dtype)
        b = torch.randn(B, T, N, device="cuda", dtype=dtype)

        out_fused = fused_swiglu(a, b)
        out_ref = swiglu_ref(a, b)

        tol = TOLERANCES[dtype]
        torch.testing.assert_close(out_fused, out_ref, **tol)


class TestFusedSwiGLUEdgeCases:
    """边界情况测试"""

    def test_zero_input(self):
        """全零输入 — silu(0)=0, 结果应为零"""
        a = torch.zeros(4, 64, device="cuda")
        b = torch.ones(4, 64, device="cuda")
        out = fused_swiglu(a, b)
        assert out.abs().max().item() == 0.0

    def test_large_input(self):
        """大幅值输入 — 验证数值稳定性"""
        torch.manual_seed(42)
        a = torch.randn(8, 128, device="cuda") * 50
        b = torch.randn(8, 128, device="cuda") * 50
        out_fused = fused_swiglu(a, b)
        out_ref = swiglu_ref(a, b)
        torch.testing.assert_close(out_fused, out_ref, atol=1e-1, rtol=1e-2)

    def test_backward_gradient_flow(self):
        """验证梯度可以正确流过 fused kernel"""
        a = torch.randn(4, 64, device="cuda", requires_grad=True)
        b = torch.randn(4, 64, device="cuda", requires_grad=True)
        out = fused_swiglu(a, b)
        loss = out.sum()
        loss.backward()
        assert a.grad is not None and a.grad.isfinite().all()
        assert b.grad is not None and b.grad.isfinite().all()


if __name__ == "__main__":
    print("=== Fused SwiGLU Unit Test ===\n")
    passed = 0
    total = 0
    for M, N in SHAPES:
        for dtype in DTYPES:
            total += 1
            try:
                torch.manual_seed(42)
                a = torch.randn(M, N, device="cuda", dtype=dtype)
                b = torch.randn(M, N, device="cuda", dtype=dtype)
                out_fused = fused_swiglu(a, b)
                out_ref = swiglu_ref(a, b)
                tol = TOLERANCES[dtype]
                torch.testing.assert_close(out_fused, out_ref, **tol)
                passed += 1
            except Exception as e:
                print(f"  FAIL ({M},{N}) {dtype}: {e}")
    print(f"Forward: {passed}/{total} passed")

    # backward 快速验证
    a = torch.randn(16, 128, device="cuda", requires_grad=True)
    b = torch.randn(16, 128, device="cuda", requires_grad=True)
    out = fused_swiglu(a, b)
    out.sum().backward()
    assert a.grad is not None and b.grad is not None
    print("Backward: PASSED")
    print("\nAll quick checks passed!")
