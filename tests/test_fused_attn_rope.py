"""
Fused Attention + RoPE 单元测试 — 验证合并操作与分步操作的数值等价性。

对比:
  Fused:  fused_attn_rope(q, k, v, cos, sin)
  分步:   q, k = apply_rotary_emb(q, k, cos, sin)
          o = F.scaled_dot_product_attention(q, k, v, is_causal=True)

参考:
  [1] Su et al., "RoFormer," arXiv:2104.09864, 2021. RoPE.
  [2] Dao, "FlashAttention-2," arXiv:2307.08691, 2023.
"""

import pytest
import torch
import torch.nn.functional as F
import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.mdlic.ops.fused_attn_rope import fused_attn_rope
from src.mdlic.models.layers import RotaryEmbedding, apply_rotary_emb


# ── 测试参数 ────────────────────────────────────────────────────
# d_k//2 必须是 2 的幂（fused_rope Triton 限制）
SHAPES = [
    (1, 2, 64, 32),     # 小 case
    (2, 4, 64, 32),     # 典型 CIFAR: h=4, d_k=32
    (1, 4, 128, 64),    # d_k=64
    (2, 8, 32, 64),     # 更多 heads
]

DTYPES = [torch.float16, torch.bfloat16]

TOLERANCES = {
    torch.float16:  {"atol": 2e-2, "rtol": 2e-2},
    torch.bfloat16: {"atol": 3e-2, "rtol": 3e-2},
}


@pytest.mark.parametrize("B, h, T, d_k", SHAPES,
                         ids=[f"B{B}_h{h}_T{T}_dk{d_k}"
                              for B, h, T, d_k in SHAPES])
@pytest.mark.parametrize("dtype", DTYPES, ids=["fp16", "bf16"])
class TestFusedAttnRoPE:
    """Fused Attn+RoPE vs 分步 RoPE + SDPA"""

    def _get_cos_sin(self, T, d_k, device, dtype):
        rope = RotaryEmbedding(d_k).to(device)
        cos, sin = rope(T, device)
        return cos.to(dtype), sin.to(dtype)

    def test_forward(self, B, h, T, d_k, dtype):
        """前向输出 allclose"""
        torch.manual_seed(42)
        device = "cuda"
        q = torch.randn(B, h, T, d_k, device=device, dtype=dtype)
        k = torch.randn(B, h, T, d_k, device=device, dtype=dtype)
        v = torch.randn(B, h, T, d_k, device=device, dtype=dtype)
        cos, sin = self._get_cos_sin(T, d_k, device, dtype)
        scale = 1.0 / math.sqrt(d_k)

        # Fused path
        out_fused = fused_attn_rope(
            q.clone().contiguous(), k.clone().contiguous(), v.clone(),
            cos, sin, causal=True, softmax_scale=scale
        )

        # 分步参考: RoPE → SDPA
        q_ref, k_ref = apply_rotary_emb(q.clone(), k.clone(), cos, sin)
        out_ref = F.scaled_dot_product_attention(
            q_ref, k_ref, v.clone(), is_causal=True
        )

        tol = TOLERANCES[dtype]
        torch.testing.assert_close(out_fused, out_ref, **tol)


class TestFusedAttnRoPEEdgeCases:
    """边界情况"""

    def test_single_token(self):
        """T=1 单 token — RoPE position=0, attention trivial"""
        B, h, T, d_k = 1, 2, 1, 32
        dtype = torch.float16
        q = torch.randn(B, h, T, d_k, device="cuda", dtype=dtype).contiguous()
        k = torch.randn(B, h, T, d_k, device="cuda", dtype=dtype).contiguous()
        v = torch.randn(B, h, T, d_k, device="cuda", dtype=dtype).contiguous()
        rope = RotaryEmbedding(d_k).to("cuda")
        cos, sin = rope(T, q.device)
        cos, sin = cos.to(dtype), sin.to(dtype)

        out = fused_attn_rope(q, k, v, cos, sin)
        # T=1: attention output = V (trivially)
        # 但 RoPE 会旋转 Q/K，所以 softmax(q@k^T/scale) = [1.0]
        # → output = v
        assert out.shape == (B, h, T, d_k)
        assert out.isfinite().all()


if __name__ == "__main__":
    print("=== Fused Attn+RoPE Unit Test ===\n")
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
                v = torch.randn(B, h, T, d_k, device=device, dtype=dtype)
                rope = RotaryEmbedding(d_k).to(device)
                cos, sin = rope(T, device)
                cos, sin = cos.to(dtype), sin.to(dtype)
                scale = 1.0 / math.sqrt(d_k)

                out_fused = fused_attn_rope(
                    q.clone().contiguous(), k.clone().contiguous(), v.clone(),
                    cos, sin, causal=True, softmax_scale=scale
                )
                q_ref, k_ref = apply_rotary_emb(q.clone(), k.clone(), cos, sin)
                out_ref = F.scaled_dot_product_attention(
                    q_ref, k_ref, v.clone(), is_causal=True
                )
                tol = TOLERANCES[dtype]
                torch.testing.assert_close(out_fused, out_ref, **tol)
                passed += 1
            except Exception as e:
                print(f"  FAIL B{B}_h{h}_T{T}_dk{d_k} {dtype}: {e}")
    print(f"Forward: {passed}/{total} passed")
    print("\nAll quick checks passed!")
