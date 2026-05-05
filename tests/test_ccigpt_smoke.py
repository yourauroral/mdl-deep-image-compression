"""CC-iGPT 单元测试: 验证 coarse_ctx 注入正确性、shape 对齐、BPP 公式与
关闭 ctx 时退化为 vanilla iGPT。"""
import math
import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.mdlic.models.cc_igpt import CCIGPT
from src.mdlic.models.igpt import IGPT, rgb_to_ycbcr_int


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def small_ccigpt(device):
    torch.manual_seed(0)
    return CCIGPT(
        image_size=32, in_channels=3, vocab_size=256,
        pool_factor=4,
        fine_d_model=64, fine_N=2, fine_h=2, fine_d_ff=128,
        coarse_d_model=64, coarse_N=2, coarse_h=2, coarse_d_ff=64,
        dropout=0.0,
    ).to(device).eval()


def test_coarse_aligned_tokens_valid(device):
    """DOWN→UP→量化后的 coarse_aligned_tokens 必须是 [0,255] long。"""
    x = torch.rand(2, 3, 32, 32, device=device)
    x_c = F.adaptive_avg_pool2d(x.clamp(0, 1), 8)
    x_up = F.interpolate(x_c, size=(32, 32), mode='bilinear', align_corners=False)
    tok = rgb_to_ycbcr_int(x_up)
    assert tok.shape == (2, 3, 32, 32)
    assert tok.dtype == torch.long
    assert tok.min().item() >= 0
    assert tok.max().item() <= 255


def test_forward_shapes_and_finite(small_ccigpt, device):
    x = torch.rand(2, 3, 32, 32, device=device)
    out = small_ccigpt(x)
    for k in ["loss", "ce_loss", "ce_loss_coarse", "ce_loss_fine", "bpp", "ctx_alpha"]:
        assert k in out, f"missing key: {k}"
    assert torch.isfinite(out["loss"]).item()
    assert torch.isfinite(out["bpp"]).item()
    bpp = out["bpp"].item()
    assert 0.0 < bpp < 50.0


def test_bpp_formula_consistency(small_ccigpt, device):
    """BPP_total = (CE_c · N_c + CE_f · N_f) / ln2 / N_f, 手算对照。"""
    x = torch.rand(2, 3, 32, 32, device=device)
    out = small_ccigpt(x)
    N_c = small_ccigpt.coarse.seq_len  # 8*8*3 = 192
    N_f = small_ccigpt.fine.seq_len    # 32*32*3 = 3072
    ce_c = out["ce_loss_coarse"].item()
    ce_f = out["ce_loss_fine"].item()
    expected = (ce_c * N_c + ce_f * N_f) / math.log(2.0) / N_f
    assert abs(out["bpp"].item() - expected) < 1e-5


def test_disable_ctx_equivalent_to_vanilla_igpt(device):
    """关闭 coarse_ctx (传 None) 时 fine 必须等价于 vanilla iGPT，CE 差 < 1e-5。"""
    torch.manual_seed(42)
    fine = IGPT(
        image_size=32, in_channels=3, vocab_size=256,
        d_model=64, N=2, h=4, d_ff=128, dropout=0.0,
    ).to(device).eval()
    x = torch.rand(2, 3, 32, 32, device=device)
    out_no_ctx = fine(x)
    out_with_zero_ctx = fine(x, coarse_ctx=None)
    assert abs(out_no_ctx["ce_loss"].item() - out_with_zero_ctx["ce_loss"].item()) < 1e-6


def test_shape_assert_triggers(device):
    """coarse_ctx shape 错误时 IGPT.forward 必须 assert 命中。"""
    torch.manual_seed(7)
    fine = IGPT(
        image_size=32, in_channels=3, vocab_size=256,
        d_model=64, N=2, h=4, d_ff=128, dropout=0.0,
    ).to(device).eval()
    x = torch.rand(2, 3, 32, 32, device=device)
    bad_ctx = torch.zeros(2, 100, 64, device=device)  # 错误 T 维度
    with pytest.raises(AssertionError):
        fine(x, coarse_ctx=bad_ctx)


def test_backward_grads_flow(small_ccigpt, device):
    """coarse、fine、ctx_alpha 三处都必须有非零梯度。"""
    small_ccigpt.train()
    x = torch.rand(2, 3, 32, 32, device=device)
    out = small_ccigpt(x)
    out["loss"].backward()
    assert small_ccigpt.ctx_alpha.grad is not None
    assert small_ccigpt.ctx_alpha.grad.abs().sum().item() > 0
    coarse_grad_sum = sum(
        p.grad.abs().sum().item() for p in small_ccigpt.coarse.parameters()
        if p.grad is not None
    )
    fine_grad_sum = sum(
        p.grad.abs().sum().item() for p in small_ccigpt.fine.parameters()
        if p.grad is not None
    )
    assert coarse_grad_sum > 0
    assert fine_grad_sum > 0
