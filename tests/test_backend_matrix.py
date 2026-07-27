"""Backend matrix for PyTorch fallbacks and optional CUDA/Triton paths."""

import pytest
import torch

import mdlic.models.layers as layers
from mdlic.models.layers import MultiHeadAttentionBlock, RotaryEmbedding, apply_rotary_emb


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_rope_cpu_fallback_preserves_attention_dtype(dtype):
    torch.manual_seed(1)
    q = torch.randn(2, 2, 7, 8, dtype=dtype, requires_grad=True)
    k = torch.randn(2, 2, 7, 8, dtype=dtype, requires_grad=True)
    cos, sin = RotaryEmbedding(8)(7, torch.device("cpu"))

    q_rotated, k_rotated = apply_rotary_emb(q, k, cos, sin)

    assert q_rotated.dtype == dtype
    assert k_rotated.dtype == dtype
    assert torch.isfinite(q_rotated.float()).all()
    assert torch.isfinite(k_rotated.float()).all()
    (q_rotated.float().sum() + k_rotated.float().sum()).backward()
    assert torch.isfinite(q.grad.float()).all()
    assert torch.isfinite(k.grad.float()).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("is_causal", [True, False], ids=["causal", "bidirectional"])
def test_attention_cpu_pytorch_backend_matrix(dtype, is_causal, monkeypatch):
    for flag in (
        "_USE_FUSED_RMSNORM",
        "_USE_FUSED_SWIGLU",
        "_USE_TRITON_ATTN",
        "_USE_FUSED_ROPE",
        "_USE_ROPE_FLASH_PIPELINE",
    ):
        monkeypatch.setattr(layers, flag, False)

    torch.manual_seed(2)
    block = MultiHeadAttentionBlock(d_model=16, h=2, dropout=0.0).to(dtype=dtype)
    x = torch.randn(2, 7, 16, dtype=dtype, requires_grad=True)
    position_ids = torch.arange(7, dtype=torch.long) // 2

    output = block(
        x, x, x,
        position_ids=position_ids,
        is_causal=is_causal,
    )

    assert output.shape == x.shape
    assert output.dtype == dtype
    assert torch.isfinite(output.float()).all()
    output.float().square().mean().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad.float()).all()


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("triton_enabled", [False, True], ids=["triton_off", "triton_on"])
@pytest.mark.parametrize("is_causal", [True, False], ids=["causal", "bidirectional"])
def test_attention_cuda_backend_matrix(dtype, triton_enabled, is_causal, monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("GPU does not support bfloat16")

    called = {"pipeline": False}
    if triton_enabled:
        if not layers.get_fused_kernel_status()["rope_then_flash_attn"]:
            pytest.skip("Triton RoPE/FlashAttention pipeline is unavailable")
        original = layers._rope_then_flash_attn

        def tracked_pipeline(*args, **kwargs):
            called["pipeline"] = True
            return original(*args, **kwargs)

        monkeypatch.setattr(layers, "_rope_then_flash_attn", tracked_pipeline)
        monkeypatch.setattr(layers, "_USE_ROPE_FLASH_PIPELINE", True)
    else:
        for flag in (
            "_USE_FUSED_RMSNORM",
            "_USE_TRITON_ATTN",
            "_USE_FUSED_ROPE",
            "_USE_ROPE_FLASH_PIPELINE",
        ):
            monkeypatch.setattr(layers, flag, False)

    torch.manual_seed(3)
    block = MultiHeadAttentionBlock(32, 4, dropout=0.0).cuda().to(dtype=dtype)
    seq_len = 16 if is_causal else layers._TRITON_ATTN_ALIGNMENT
    x = torch.randn(2, seq_len, 32, device="cuda", dtype=dtype, requires_grad=True)
    output = block(x, x, x, is_causal=is_causal)
    output.float().square().mean().backward()

    assert output.dtype == dtype
    assert torch.isfinite(output.float()).all()
    assert torch.isfinite(x.grad.float()).all()
    assert called["pipeline"] is triton_enabled
