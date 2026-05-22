"""CC-iGPT 单元测试: 验证 coarse_ctx 注入正确性、shape 对齐、bits/dim 公式与
关闭 ctx 时退化为 vanilla iGPT。RGB-bit-exact 域唯一。"""
import math
import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.mdlic.models.cc_igpt import CCIGPT
from src.mdlic.models.igpt import IGPT


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
    tok = (x_up.clamp(0, 1) * 255).round().long()
    assert tok.shape == (2, 3, 32, 32)
    assert tok.dtype == torch.long
    assert tok.min().item() >= 0
    assert tok.max().item() <= 255


def test_forward_shapes_and_finite(small_ccigpt, device):
    x = torch.rand(2, 3, 32, 32, device=device)
    out = small_ccigpt(x)
    for k in ["loss", "ce_loss", "ce_loss_coarse", "ce_loss_fine", "bpd", "ctx_alpha"]:
        assert k in out, f"missing key: {k}"
    assert torch.isfinite(out["loss"]).item()
    assert torch.isfinite(out["bpd"]).item()
    bpd = out["bpd"].item()
    assert 0.0 < bpd < 50.0


def test_bpd_formula_consistency(small_ccigpt, device):
    """bpd_total = (CE_c · N_c + CE_f · N_f) / ln2 / N_f, 手算对照。"""
    x = torch.rand(2, 3, 32, 32, device=device)
    out = small_ccigpt(x)
    N_c = small_ccigpt.coarse.seq_len  # 8*8*3 = 192
    N_f = small_ccigpt.fine.seq_len    # 32*32*3 = 3072
    ce_c = out["ce_loss_coarse"].item()
    ce_f = out["ce_loss_fine"].item()
    expected = (ce_c * N_c + ce_f * N_f) / math.log(2.0) / N_f
    assert abs(out["bpd"].item() - expected) < 1e-5


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

    场景：实际熵编码部署时 bitstream 只携带 coarse 量化 token (RGB int)。
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
    assert len(out_via_ccigpt) == len(out_via_fine), (
        f"encode 输出层数不一致：{len(out_via_ccigpt)} vs {len(out_via_fine)}"
    )
    for i, (a, b) in enumerate(zip(out_via_ccigpt, out_via_fine)):
        diff = (a - b).abs().max().item()
        assert diff < 1e-5, f"layer {i} 输出不一致 (max diff={diff:.6e})"


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


# ===================== sub-pixel AR + RGB 真无损测试 =====================

def test_ccigpt_subpixel_rgb_forward_finite(device):
    """CC-iGPT forward 必须给出有限的 loss/bpd。"""
    torch.manual_seed(5)
    m = CCIGPT(
        image_size=32, in_channels=3, vocab_size=256, pool_factor=4,
        fine_d_model=64, fine_N=2, fine_h=2, fine_d_ff=128,
        coarse_d_model=64, coarse_N=2, coarse_h=2, coarse_d_ff=64,
        dropout=0.0,
    ).to(device).eval()
    x = torch.rand(2, 3, 32, 32, device=device)
    out = m(x)
    assert torch.isfinite(out["loss"]).item()
    assert torch.isfinite(out["bpd"]).item()
    assert 0.0 < out["bpd"].item() < 50.0


def test_ccigpt_subpixel_rgb_ctx_consistency(device):
    """encoder forward 实际用的 ctx 必须等于 decoder 仅凭 coarse bitstream
    tokens 重建的 ctx (bit-exact 不变量)。"""
    torch.manual_seed(6)
    m = CCIGPT(
        image_size=32, in_channels=3, vocab_size=256, pool_factor=4,
        fine_d_model=64, fine_N=2, fine_h=2, fine_d_ff=128,
        coarse_d_model=64, coarse_N=2, coarse_h=2, coarse_d_ff=64,
        dropout=0.0,
    ).to(device).eval()
    x = torch.rand(2, 3, 32, 32, device=device)

    captured = {}
    orig = m._compute_coarse_ctx
    def spy(tokens):
        out = orig(tokens)
        captured["tokens"] = tokens.detach().clone()
        captured["ctx"] = out.detach().clone()
        return out
    m._compute_coarse_ctx = spy
    try:
        with torch.no_grad():
            m(x)
    finally:
        m._compute_coarse_ctx = orig

    ctx_dec = m._compute_coarse_ctx(captured["tokens"])
    max_diff = (captured["ctx"] - ctx_dec).abs().max().item()
    assert max_diff < 1e-6, (
        f"ctx 不一致 (max diff={max_diff:.6e})"
    )

    # forward 用的 token 必须可独立从 coarse._tokenize(x_c_float) 复现
    x_c_float = F.adaptive_avg_pool2d(x.clamp(0, 1), m.coarse_size)
    expected_tokens = m.coarse._tokenize(x_c_float)
    assert (captured["tokens"] == expected_tokens).all()


def test_ccigpt_subpixel_ctx_pixel_first_layout(device):
    """_compute_coarse_ctx 入口/出口必须走 pixel-first 平铺。手算两端平铺顺序
    与实际输出对比；错位会让 forward 仍能跑但 bitstream 不可解。"""
    torch.manual_seed(7)
    m = CCIGPT(
        image_size=32, in_channels=3, vocab_size=256, pool_factor=4,
        fine_d_model=64, fine_N=2, fine_h=2, fine_d_ff=128,
        coarse_d_model=64, coarse_N=2, coarse_h=2, coarse_d_ff=64,
        dropout=0.0,
    ).to(device).eval()

    x = torch.rand(1, 3, 32, 32, device=device)
    x_c_float = F.adaptive_avg_pool2d(x.clamp(0, 1), 8)
    coarse_tokens = m.coarse._tokenize(x_c_float)            # (1, 192) pixel-first

    # 手算 _compute_coarse_ctx：
    # 入口：(1, 192) pixel-first → (1, 8, 8, 3) → permute → (1, 3, 8, 8)
    B, S, C = 1, 8, 3
    coarse_chw = coarse_tokens.view(B, S, S, C).permute(0, 3, 1, 2).contiguous()
    rec = coarse_chw.float() / 255.0
    x_up = F.interpolate(rec, size=(32, 32), mode='bilinear', align_corners=False)
    x_up_tok = (x_up.clamp(0, 1) * 255).round().long()       # (1, 3, 32, 32)
    # 出口：pixel-first 平铺
    expected_pixel_first = x_up_tok.permute(0, 2, 3, 1).reshape(1, -1)
    expected_ctx = m.fine.token_embed(expected_pixel_first)[:, 1:]

    actual_ctx = m._compute_coarse_ctx(coarse_tokens)
    diff = (actual_ctx - expected_ctx).abs().max().item()
    assert diff < 1e-6, (
        f"_compute_coarse_ctx 未走 pixel-first 双端分支 (max diff={diff:.6e})"
    )


def test_ccigpt_subpixel_backward(device):
    """coarse / fine / ctx_alpha 三处都必须有非零梯度。"""
    torch.manual_seed(8)
    m = CCIGPT(
        image_size=32, in_channels=3, vocab_size=256, pool_factor=4,
        fine_d_model=64, fine_N=2, fine_h=2, fine_d_ff=128,
        coarse_d_model=64, coarse_N=2, coarse_h=2, coarse_d_ff=64,
        dropout=0.0,
    ).to(device).train()
    x = torch.rand(2, 3, 32, 32, device=device)
    out = m(x)
    out["loss"].backward()
    assert m.ctx_alpha.grad is not None
    assert m.ctx_alpha.grad.abs().sum().item() > 0
    coarse_grad_sum = sum(
        p.grad.abs().sum().item() for p in m.coarse.parameters()
        if p.grad is not None
    )
    fine_grad_sum = sum(
        p.grad.abs().sum().item() for p in m.fine.parameters()
        if p.grad is not None
    )
    assert coarse_grad_sum > 0
    assert fine_grad_sum > 0


# ──────────────────────────────────────────────────────────────
# R-only coarse 灰度先验 (coarse_in_channels=1)
# ──────────────────────────────────────────────────────────────

def _build_ronly_ccigpt(device):
    """coarse_in_channels=1 的最小 CCIGPT。"""
    return CCIGPT(
        image_size=32, in_channels=3, vocab_size=256, pool_factor=4,
        fine_d_model=64, fine_N=2, fine_h=2, fine_d_ff=128,
        coarse_d_model=64, coarse_N=2, coarse_h=2, coarse_d_ff=64,
        dropout=0.0,
        coarse_in_channels=1,
    ).to(device)


def test_ccigpt_ronly_forward_finite(device):
    """R-only coarse forward + backward 通过；coarse_seq=64, fine_seq=3072。"""
    torch.manual_seed(20)
    m = _build_ronly_ccigpt(device).train()
    assert m.coarse.in_channels == 1
    assert m.coarse.seq_len == 8 * 8 * 1            # 64
    assert m.fine.seq_len == 32 * 32 * 3            # 3072

    x = torch.rand(2, 3, 32, 32, device=device)
    out = m(x)
    for k in ["loss", "ce_loss_coarse", "ce_loss_fine", "bpd"]:
        assert torch.isfinite(out[k]).item()
    assert 0.0 < out["bpd"].item() < 50.0

    # bpd 公式手算对照
    N_c, N_f = m.coarse.seq_len, m.fine.seq_len
    expected = (
        out["ce_loss_coarse"].item() * N_c +
        out["ce_loss_fine"].item()   * N_f
    ) / math.log(2.0) / N_f
    assert abs(out["bpd"].item() - expected) < 1e-5

    out["loss"].backward()
    coarse_grad = sum(p.grad.abs().sum().item() for p in m.coarse.parameters()
                      if p.grad is not None)
    fine_grad = sum(p.grad.abs().sum().item() for p in m.fine.parameters()
                    if p.grad is not None)
    assert coarse_grad > 0 and fine_grad > 0
    assert m.ctx_alpha.grad is not None and m.ctx_alpha.grad.abs().sum().item() > 0


def test_ccigpt_ronly_ctx_consistency(device):
    """encoder 实际 ctx 必须等于 decoder 仅凭 coarse_tokens 重建的 ctx
    （bit-exact 不变量，R-only 路径下尤其关键：expand contiguous + 部分通道
    截取 都是新出错点）。"""
    torch.manual_seed(21)
    m = _build_ronly_ccigpt(device).eval()
    x = torch.rand(2, 3, 32, 32, device=device)

    captured = {}
    orig = m._compute_coarse_ctx
    def spy(tokens):
        out = orig(tokens)
        captured["tokens"] = tokens.detach().clone()
        captured["ctx"] = out.detach().clone()
        return out
    m._compute_coarse_ctx = spy
    try:
        with torch.no_grad():
            m(x)
    finally:
        m._compute_coarse_ctx = orig

    # forward 用的 token 必须可独立从 (R-only 截通道后的) _tokenize 复现
    x_c_full = F.adaptive_avg_pool2d(x.clamp(0, 1), m.coarse_size)
    x_c_r = x_c_full[:, :1]
    expected_tokens = m.coarse._tokenize(x_c_r)
    assert (captured["tokens"] == expected_tokens).all()
    assert captured["tokens"].shape == (2, 64)       # B, S*S*C_coarse

    # decoder 重建 ctx 必须 bit-exact 等于 encoder 实际 ctx
    ctx_dec = m._compute_coarse_ctx(captured["tokens"])
    max_diff = (captured["ctx"] - ctx_dec).abs().max().item()
    assert max_diff < 1e-6, (
        f"R-only 路径 encoder/decoder ctx 不一致 (max diff={max_diff:.6e})"
    )


def test_ccigpt_ronly_expand_layout(device):
    """R-only 灰度先验：fine 看到的 ctx 在每个像素位置上，R/G/B 三个通道槽
    必须接收到同一个 R coarse 值（expand 复制三份后 token 应相同）。
    错位（比如 expand 后忘 .contiguous、或顺序反了）会让 G/B 槽看到错误的
    像素值，loss 仍能降但 ctx 语义乱套。
    """
    torch.manual_seed(22)
    m = _build_ronly_ccigpt(device).eval()

    x = torch.rand(1, 3, 32, 32, device=device)
    x_c_full = F.adaptive_avg_pool2d(x.clamp(0, 1), 8)
    x_c_r = x_c_full[:, :1]                          # (1, 1, 8, 8)
    coarse_tokens = m.coarse._tokenize(x_c_r)        # (1, 64) pixel-first 单通道

    # 手算：(1,64) → view(1,8,8,1) → permute → (1,1,8,8) → /255
    coarse_chw = coarse_tokens.view(1, 8, 8, 1).permute(0, 3, 1, 2).contiguous()
    rec = coarse_chw.float() / 255.0
    x_up_partial = F.interpolate(rec, size=(32, 32), mode='bilinear', align_corners=False)
    # R 复制三份的灰度先验
    x_up = x_up_partial.expand(-1, 3, -1, -1).contiguous()
    # 三通道值必须严格相同（灰度先验的定义）
    assert torch.equal(x_up[:, 0], x_up[:, 1]) and torch.equal(x_up[:, 1], x_up[:, 2])

    x_up_tok = (x_up.clamp(0, 1) * 255).round().long()
    expected_pixel_first = x_up_tok.permute(0, 2, 3, 1).reshape(1, -1)
    # pixel-first 出口下，每个像素位置 (R_slot, G_slot, B_slot) 三个 token id 相同
    reshaped = expected_pixel_first.view(1, 32 * 32, 3)
    assert torch.equal(reshaped[..., 0], reshaped[..., 1])
    assert torch.equal(reshaped[..., 1], reshaped[..., 2])

    expected_ctx = m.fine.token_embed(expected_pixel_first)[:, 1:]
    actual_ctx = m._compute_coarse_ctx(coarse_tokens)
    diff = (actual_ctx - expected_ctx).abs().max().item()
    assert diff < 1e-6, (
        f"R-only expand 灰度先验布局不一致 (max diff={diff:.6e})"
    )
