from scripts import profile_kernels
from mdlic.models.layers import apply_swiglu


class _FakeCudaMemory:
    def synchronize(self):
        pass

    def memory_allocated(self):
        return 100 * 1024 * 1024

    def reset_peak_memory_stats(self):
        pass

    def max_memory_allocated(self):
        return 148 * 1024 * 1024


def test_measure_memory_reports_peak_delta(monkeypatch):
    monkeypatch.setattr(profile_kernels.torch, "cuda", _FakeCudaMemory())

    assert profile_kernels.measure_memory(lambda: None) == 48.0


def test_rope_flash_pipeline_counts_two_kernel_memory_traffic():
    args = dict(M=1, N=64, B=2, h=4, T=16, d_k=8, V=256, dtype_bytes=2)

    _, flash_bytes = profile_kernels.estimate_flops_bytes("flash_attn", **args)
    _, pipeline_bytes = profile_kernels.estimate_flops_bytes(
        "rope_then_flash_attn", **args
    )

    numel = args["B"] * args["h"] * args["T"] * args["d_k"]
    expected_rope_bytes = (
        4 * numel + 2 * args["T"] * args["d_k"]
    ) * args["dtype_bytes"]
    assert pipeline_bytes == flash_bytes + expected_rope_bytes


def test_swiglu_profile_uses_the_model_fallback_exactly():
    assert profile_kernels.pytorch_swiglu_reference is apply_swiglu

    for dtype in (profile_kernels.torch.float32, profile_kernels.torch.bfloat16):
        a = profile_kernels.torch.linspace(-2, 2, 16, dtype=dtype)
        b = profile_kernels.torch.linspace(1, 3, 16, dtype=dtype)
        actual = profile_kernels.pytorch_swiglu_reference(a, b)
        expected = profile_kernels.F.silu(a) * b
        assert actual.dtype == dtype
        assert profile_kernels.torch.equal(actual, expected)
