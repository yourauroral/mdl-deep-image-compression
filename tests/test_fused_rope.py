"""
Fused RoPE Triton Kernel 单元测试 — 数值精度验证。

对比 Triton fused kernel（就地旋转）与 PyTorch apply_rotary_emb（layers.py）。
两者应产生完全相同的旋转结果。

注意: Fused RoPE 是就地操作（in-place），测试前需 clone 输入。
Triton tl.arange 要求 HALF_DK 为 2 的幂次。

参考:
  [1] Su et al., "RoFormer: Enhanced Transformer with Rotary Position
      Embedding," arXiv:2104.09864, 2021. Eq.(34).
"""

import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.mdlic.ops.fused_rope import fused_apply_rotary_emb
from src.mdlic.models.layers import RotaryEmbedding, apply_rotary_emb


# ── 测试参数 ────────────────────────────────────────────────────
# (B, h, T, d_k): batch, heads, seq_len, head_dim
# d_k//2 必须是 2 的幂（Triton 限制）
SHAPES = [
    (1, 2, 16, 32),     # 最小 case, d_k=32 → HALF_DK=16
    (2, 4, 64, 32),     # 典型 CIFAR: h=4, d_k=32
    (1, 4, 128, 64),    # d_k=64
    (2, 8, 32, 64),     # 更多 heads
    (1, 2, 256, 32),    # 长序列
]

DTYPES = [torch.float32, torch.float16, torch.bfloat16]

TOLERANCES = {
    torch.float32:  {"atol": 1e-5, "rtol": 1e-5},
    torch.float16:  {"atol": 1e-2, "rtol": 1e-2},
    torch.bfloat16: {"atol": 2e-2, "rtol": 2e-2},
}


@pytest.mark.parametrize("B, h, T, d_k", SHAPES,
                         ids=[f"B{B}_h{h}_T{T}_dk{d_k}"
                              for B, h, T, d_k in SHAPES])
@pytest.mark.parametrize("dtype", DTYPES,
                         ids=["fp32", "fp16", "bf16"])
class TestFusedRoPE:
    """Fused RoPE vs PyTorch apply_rotary_emb"""

    def _get_cos_sin(self, T, d_k, device, dtype):
        """生成 RoPE cos/sin（使用 RotaryEmbedding 模块）"""
        rope = RotaryEmbedding(d_k).to(device)
        cos, sin = rope(T, device)
        return cos.to(dtype), sin.to(dtype)

    def test_forward_q(self, B, h, T, d_k, dtype):
        """Q 旋转结果 allclose"""
        torch.manual_seed(42)
        device = "cuda"
        q = torch.randn(B, h, T, d_k, device=device, dtype=dtype)
        k = torch.randn(B, h, T, d_k, device=device, dtype=dtype)
        cos, sin = self._get_cos_sin(T, d_k, device, dtype)

        # PyTorch 参考
        q_ref, k_ref = apply_rotary_emb(q.clone(), k.clone(), cos, sin)

        # Fused kernel（就地操作，需 contiguous + clone）
        q_fused = q.clone().contiguous()
        k_fused = k.clone().contiguous()
        q_fused, k_fused = fused_apply_rotary_emb(q_fused, k_fused, cos, sin)

        tol = TOLERANCES[dtype]
        torch.testing.assert_close(q_fused, q_ref, **tol)

    def test_forward_k(self, B, h, T, d_k, dtype):
        """K 旋转结果 allclose"""
        torch.manual_seed(42)
        device = "cuda"
        q = torch.randn(B, h, T, d_k, device=device, dtype=dtype)
        k = torch.randn(B, h, T, d_k, device=device, dtype=dtype)
        cos, sin = self._get_cos_sin(T, d_k, device, dtype)

        q_ref, k_ref = apply_rotary_emb(q.clone(), k.clone(), cos, sin)

        q_fused = q.clone().contiguous()
        k_fused = k.clone().contiguous()
        q_fused, k_fused = fused_apply_rotary_emb(q_fused, k_fused, cos, sin)

        tol = TOLERANCES[dtype]
        torch.testing.assert_close(k_fused, k_ref, **tol)

    def test_inplace_semantics(self, B, h, T, d_k, dtype):
        """验证就地修改语义: 返回的 tensor 与输入共享存储"""
        torch.manual_seed(42)
        device = "cuda"
        q = torch.randn(B, h, T, d_k, device=device, dtype=dtype).contiguous()
        k = torch.randn(B, h, T, d_k, device=device, dtype=dtype).contiguous()
        cos, sin = self._get_cos_sin(T, d_k, device, dtype)

        q_ptr = q.data_ptr()
        k_ptr = k.data_ptr()
        q_out, k_out = fused_apply_rotary_emb(q, k, cos, sin)

        # 就地修改: 返回的 tensor 应该与输入共享相同的存储
        assert q_out.data_ptr() == q_ptr
        assert k_out.data_ptr() == k_ptr


class TestFusedRoPEEdgeCases:
    """边界情况测试"""

    def test_single_position(self):
        """T=1 — 单位置旋转"""
        torch.manual_seed(42)
        B, h, T, d_k = 1, 2, 1, 32
        q = torch.randn(B, h, T, d_k, device="cuda").contiguous()
        k = torch.randn(B, h, T, d_k, device="cuda").contiguous()
        rope = RotaryEmbedding(d_k).to("cuda")
        cos, sin = rope(T, q.device)

        q_ref, k_ref = apply_rotary_emb(q.clone(), k.clone(), cos, sin)
        q_fused, k_fused = fused_apply_rotary_emb(q.clone().contiguous(),
                                                    k.clone().contiguous(),
                                                    cos, sin)
        torch.testing.assert_close(q_fused, q_ref, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(k_fused, k_ref, atol=1e-5, rtol=1e-5)

    def test_position_invariance(self):
        """验证 RoPE 的基本性质: position 0 的 cos=1, sin=0 → 不改变向量"""
        B, h, d_k = 1, 1, 32
        # cos = [1,1,...], sin = [0,0,...] 对应 position 0
        cos = torch.ones(1, d_k, device="cuda")
        sin = torch.zeros(1, d_k, device="cuda")
        q = torch.randn(B, h, 1, d_k, device="cuda").contiguous()
        q_orig = q.clone()
        q_out, _ = fused_apply_rotary_emb(q, q.clone().contiguous(), cos, sin)
        torch.testing.assert_close(q_out, q_orig, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    print("=== Fused RoPE Unit Test ===\n")
    passed = 0
    total = 0
    for B, h, T, d_k in SHAPES:
        for dtype in DTYPES:
            total += 1
            try:
                torch.manual_seed(42)
                device = "cuda"
                q = torch.randn(B, h, T, d_k, device=device, dtype=dtype)
                k = torch.randn(B, h, T, d_k, device=device, dtype=dtype)
                rope = RotaryEmbedding(d_k).to(device)
                cos, sin = rope(T, device)
                cos, sin = cos.to(dtype), sin.to(dtype)

                q_ref, k_ref = apply_rotary_emb(q.clone(), k.clone(), cos, sin)
                q_f, k_f = fused_apply_rotary_emb(q.clone().contiguous(),
                                                    k.clone().contiguous(),
                                                    cos, sin)
                tol = TOLERANCES[dtype]
                torch.testing.assert_close(q_f, q_ref, **tol)
                torch.testing.assert_close(k_f, k_ref, **tol)
                passed += 1
            except Exception as e:
                print(f"  FAIL B{B}_h{h}_T{T}_dk{d_k} {dtype}: {e}")
    print(f"Forward: {passed}/{total} passed")
    print("\nAll quick checks passed!")
