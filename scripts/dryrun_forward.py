#!/usr/bin/env python3
"""
Quick forward pass sanity check — 验证模型构建和 forward 是否正常。

包含:
  1. 默认配置 forward + weight tying / post-norm 验证
  2. 子像素自回归 backward
  3. CC-iGPT smoke (coarse + fine + ctx_alpha 梯度检查)
  4. Fused kernel 状态检查
  5. Numerical sanity（loss 有限、bits/dim 合理范围）

Usage:
    python scripts/dryrun_forward.py
    python scripts/dryrun_forward.py --config configs/igpt_cifar10_s.yaml
"""

import os
import sys
import argparse
import yaml
import math
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.mdlic.models.igpt import IGPT
from src.mdlic.models.cc_igpt import CCIGPT
from scripts.train import _build_model_from_config, _build_ccigpt_from_config
from src.mdlic.models.layers import get_fused_kernel_status


def _check_finite(out: dict, tag: str, in_channels: int = 3):
    """检查 loss 和 ce_loss 是否有限（非 NaN/Inf）。

    按 loss_unit 分发 bpd 计算：
      per_subpixel_nat — bpd = ce / ln2（softmax 路径）
      per_pixel_nat    — bpd = ce / ln2 / C（DMoL 路径）
    """
    loss_val = out['loss'].item()
    ce_val = out['ce_loss'].item()
    assert math.isfinite(loss_val), f"[{tag}] loss is {loss_val} (NaN/Inf!)"
    assert math.isfinite(ce_val), f"[{tag}] ce_loss is {ce_val} (NaN/Inf!)"
    unit = out.get("loss_unit", "per_subpixel_nat")
    if unit == "per_pixel_nat":
        bpd = ce_val / math.log(2) / in_channels
    else:
        bpd = ce_val / math.log(2)
    assert 0.0 < bpd < 50.0, f"[{tag}] bits/dim={bpd:.2f} 超出合理范围 (0, 50)"
    logits_info = out['logits'].shape if out['logits'] is not None else "None"
    print(f"  [{tag}] loss={loss_val:.4f}  ce_loss={ce_val:.4f}  bits/dim={bpd:.2f}  "
          f"unit={unit}  logits={logits_info}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/igpt_cifar10_s.yaml')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    mcfg = config["model"]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # ── 0. Fused kernel 状态 ──
    print("\n=== Fused Kernel Status ===")
    kernel_status = get_fused_kernel_status()
    for name, avail in kernel_status.items():
        print(f"  {name}: {'ON' if avail else 'OFF'}")

    x = torch.rand(2, 3, 32, 32, device=device)

    # ── 1. 默认配置 forward（按 config.model.type 分发到 iGPT / CC-iGPT） ──
    model_type = mcfg.get("type", "igpt")
    print(f"\n=== Test 1: Default Config (model.type={model_type}) ===")
    if model_type == "ccigpt":
        model = _build_ccigpt_from_config(mcfg, device)
    else:
        model = _build_model_from_config(mcfg, device)
    out = model(x)
    _check_finite(out, f"default ({model_type})")

    # Weight tying 验证（仅 softmax 路径有；DMoL head 与 token_embed 不共享）
    if model_type == "ccigpt":
        assert model.fine.head.weight is model.fine.token_embed.weight, "Weight tying failed on fine!"
        assert model.coarse.head.weight is model.coarse.token_embed.weight, "Weight tying failed on coarse!"
        print("  [weight tying] OK — fine.head & coarse.head 都共享 embed")
    elif getattr(model, "output_head", "softmax") == "dmol":
        assert isinstance(model.token_embed, torch.nn.ModuleList), (
            "DMoL 路径 token_embed 应为 ModuleList (三通道分别 embed)"
        )
        assert len(model.token_embed) == 3
        assert hasattr(model.head, "proj"), "DMoLHead 应有 .proj Linear"
        print("  [DMoL head] OK — 三通道独立 embed + DMoLHead 投影 (无 weight tying)")
    else:
        assert model.head.weight is model.token_embed.weight, "Weight tying failed!"
        print("  [weight tying] OK — head.weight is token_embed.weight")

    # ── 2. 子像素自回归 (仅 iGPT；CC-iGPT 强制 channel-first，不适用) ──
    if model_type != "ccigpt":
        print("\n=== Test 2: Sub-pixel Autoregression ===")
        model4 = IGPT(
            image_size=32, in_channels=3, vocab_size=256,
            d_model=mcfg["d_model"], N=mcfg["N"], h=mcfg["h"], d_ff=mcfg["d_ff"],
            dropout=0.1, use_subpixel_ar=True,
        ).to(device)
        out4 = model4(x)
        _check_finite(out4, "subpixel-ar")
        assert hasattr(model4, 'channel_embed'), "sub-pixel AR model should have channel_embed"
        assert model4.channel_embed.weight.shape == (3, mcfg["d_model"]), (
            f"channel_embed shape mismatch: {model4.channel_embed.weight.shape}"
        )
        print("  [channel_embed] OK — shape (3, d_model)")
        out4['loss'].backward()
        grad_ok4 = all(p.grad is not None for p in model4.parameters() if p.requires_grad)
        assert grad_ok4, "Some parameters missing gradients after backward (subpixel-ar)"
        print("  [backward] all grads computed: OK")

    # ── 3. CC-iGPT smoke (硬编码 mini 配置，与 Test 1 真实 config 路径互补) ──
    print("\n=== Test 3: CC-iGPT (Coarse-Conditioned iGPT, mini hardcoded) ===")
    model5 = CCIGPT(
        image_size=32, in_channels=3, vocab_size=256,
        pool_factor=4,
        fine_d_model=mcfg["d_model"], fine_N=2,
        fine_h=mcfg["h"], fine_d_ff=mcfg["d_ff"],
        coarse_d_model=128, coarse_N=2, coarse_h=4, coarse_d_ff=344,
        dropout=0.0,
    ).to(device)
    out5 = model5(x)
    bpd5 = out5["bpd"].item()
    loss5 = out5["loss"].item()
    ce5_c = out5["ce_loss_coarse"].item()
    ce5_f = out5["ce_loss_fine"].item()
    alpha5 = out5["ctx_alpha"].item()
    assert math.isfinite(loss5), "CC-iGPT loss NaN/Inf"
    assert 0.0 < bpd5 < 50.0, f"CC-iGPT bits/dim={bpd5:.2f} 超出合理范围"
    print(f"  [ccigpt] loss={loss5:.4f}  ce_coarse={ce5_c:.4f}  ce_fine={ce5_f:.4f}  "
          f"bpd_total={bpd5:.4f}  α={alpha5:.3f}")
    out5["loss"].backward()
    grad_ok5 = all(p.grad is not None for p in model5.parameters() if p.requires_grad)
    assert grad_ok5, "CC-iGPT: some params missing gradients"
    # 确认 coarse 与 fine 都有梯度
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model5.coarse.parameters()), \
        "CC-iGPT coarse 分支无有效梯度"
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model5.fine.parameters()), \
        "CC-iGPT fine 分支无有效梯度"
    assert model5.ctx_alpha.grad is not None, "ctx_alpha 无梯度"
    print(f"  [ccigpt.grad] coarse + fine + α 全部有梯度: OK")

    print("\nAll checks passed!")


if __name__ == '__main__':
    main()
