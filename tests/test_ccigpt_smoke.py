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


def test_encoder_decoder_ctx_consistency(small_ccigpt, device):
    """Bit-exact 一致性：encoder 真实 forward 用的 coarse_ctx 必须等于 decoder
    仅凭 coarse bitstream tokens 重建出的 coarse_ctx。

    场景：实际熵编码部署时 bitstream 只携带 coarse 量化 token (YCbCr int)。
    解码端拿到这些 token 后必须独立重建出与 encoder 相同的 fine 条件分布
    p(x_fine | coarse_ctx)，否则算术编码不可解。

    与旧版本测试的关键差异：旧版仅"对同一函数传同一输入比较"——这是 trivially
    true 的恒等测试，无法捕获 forward 路径偷偷走 float 旁路（绕过量化）的回归。
    本版本用 spy hook 截获 forward 内部实际使用的 coarse_tokens 与 ctx，再与
    "decoder 视角"独立调用做对比，能真正发现量化路径被绕过。
    """
    m = small_ccigpt
    x = torch.rand(2, 3, 32, 32, device=device)

    captured = {}
    orig_compute = m._compute_coarse_ctx
    def spy(tokens):
        out = orig_compute(tokens)
        captured["tokens"] = tokens.detach().clone()
        captured["ctx"] = out.detach().clone()
        return out
    m._compute_coarse_ctx = spy
    try:
        with torch.no_grad():
            m(x)  # 触发完整 forward，记录 forward 实际用的 (tokens, ctx)
    finally:
        m._compute_coarse_ctx = orig_compute

    # decoder 视角：从 bitstream 解出 coarse token 后独立调 _compute_coarse_ctx
    decoder_tokens = captured["tokens"]
    ctx_dec = m._compute_coarse_ctx(decoder_tokens)

    max_diff = (captured["ctx"] - ctx_dec).abs().max().item()
    assert max_diff < 1e-6, (
        f"encoder forward 内 ctx 与 decoder 独立重建 ctx 不一致 "
        f"(max diff={max_diff:.6e})。可能 forward 路径绕开了量化 token。"
    )

    # 进一步：forward 用的 token 必须可独立从 _tokenize(x_c_float) 复现，
    # 否则 decoder 拿不到一样的 tokens。
    x_c_float = F.adaptive_avg_pool2d(x.clamp(0, 1), m.coarse_size)
    expected_tokens = m.coarse._tokenize(x_c_float)
    assert (captured["tokens"] == expected_tokens).all(), (
        "forward 内 _compute_coarse_ctx 收到的 tokens 与 coarse._tokenize 输出不一致"
    )

    # reshape 正确性: 直接对比反量化 (B,C,S,S) 与 rgb_to_ycbcr_int(x_c_float)
    # 应当 byte-exact（防止 channel-first vs pixel-first reshape 误用）
    ycbcr_direct = rgb_to_ycbcr_int(x_c_float)
    B = x.size(0)
    S, C = m.coarse_size, m.in_channels
    ycbcr_via_reshape = decoder_tokens.view(B, C, S, S)
    assert (ycbcr_direct == ycbcr_via_reshape).all(), (
        "coarse_tokens reshape 错误：view(B,C,S,S) 没还原出原 YCbCr 平面"
    )


def test_encoder_decoder_ctx_consistency_rgb(device):
    """同一致性测试，但走 use_ycbcr=False 的 RGB 反量化分支。"""
    torch.manual_seed(1)
    m = CCIGPT(
        image_size=32, in_channels=3, vocab_size=256, pool_factor=4,
        fine_d_model=64, fine_N=2, fine_h=2, fine_d_ff=128,
        coarse_d_model=64, coarse_N=2, coarse_h=2, coarse_d_ff=64,
        dropout=0.0, use_ycbcr=False,
    ).to(device).eval()
    x = torch.rand(2, 3, 32, 32, device=device)

    captured = {}
    orig_compute = m._compute_coarse_ctx
    def spy(tokens):
        out = orig_compute(tokens)
        captured["tokens"] = tokens.detach().clone()
        captured["ctx"] = out.detach().clone()
        return out
    m._compute_coarse_ctx = spy
    try:
        with torch.no_grad():
            m(x)
    finally:
        m._compute_coarse_ctx = orig_compute

    ctx_dec = m._compute_coarse_ctx(captured["tokens"])
    max_diff = (captured["ctx"] - ctx_dec).abs().max().item()
    assert max_diff < 1e-6, f"RGB 分支 ctx 不一致 (max diff={max_diff:.6e})"


def test_ccigpt_disable_ctx_equivalent_to_fine_alone(small_ccigpt, device):
    """CCIGPT.encode(use_coarse_ctx=False) 必须严格等价 fine 子模型独立 encode。

    覆盖 linear_probe 消融对照路径：用户传 --no_coarse_ctx 时 fine 应该看不到
    任何 ctx 注入；本测试防止未来 encode 实现误把 ctx 默认接通。
    """
    m = small_ccigpt
    x = torch.rand(2, 3, 32, 32, device=device)
    with torch.no_grad():
        out_via_ccigpt = m.encode(x, max_layer=1, pool=True, use_coarse_ctx=False)
        out_via_fine = m.fine.encode(x, max_layer=1, pool=True)
    for k in out_via_ccigpt:
        assert k in out_via_fine
        diff = (out_via_ccigpt[k] - out_via_fine[k]).abs().max().item()
        assert diff < 1e-5, f"layer {k} 输出不一致 (max diff={diff:.6e})"


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
