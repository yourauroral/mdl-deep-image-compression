#!/usr/bin/env python3
"""
Quick forward pass sanity check — 验证模型构建和 forward 是否正常。

⚠️ DO NOT RUN ON WSL — 历史上 dryrun_forward.py 在 WSL 上会触发
   死机（Triton kernel 编译 + GPU 访问通路在 WSL 下不稳定）。
   本脚本仅供 AutoDL 等真实 CUDA 环境使用；WSL 端的代码改动用 pytest
   + 静态分析验证。

包含:
  1. 默认配置 forward + weight tying / post-norm 验证
  2. CC-iGPT smoke (coarse + fine + ctx_alpha 梯度检查)
  3. Fused kernel 状态检查
  4. Numerical sanity（loss 有限、bits/dim 合理范围）

Usage:
    python scripts/dryrun_forward.py
    python scripts/dryrun_forward.py --config configs/igpt_cifar10_s_rgb.yaml
"""

import os
import sys
import argparse
import yaml
import math
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from mdlic.models.igpt import IGPT
from mdlic.models.cc_igpt import CCIGPT
from scripts.train import _build_model_from_config, _build_ccigpt_from_config
from mdlic.models.layers import get_fused_kernel_status


def _check_finite(out: dict, tag: str):
    """检查 loss 和 ce_loss 是否有限（非 NaN/Inf）。"""
    loss_val = out['loss'].item()
    ce_val = out['ce_loss'].item()
    assert math.isfinite(loss_val), f"[{tag}] loss is {loss_val} (NaN/Inf!)"
    assert math.isfinite(ce_val), f"[{tag}] ce_loss is {ce_val} (NaN/Inf!)"
    bpd = out.get('bpd')
    bpd = bpd.item() if bpd is not None else ce_val / math.log(2)
    assert 0.0 < bpd < 50.0, f"[{tag}] bits/dim={bpd:.2f} 超出合理范围 (0, 50)"
    print(f"  [{tag}] loss={loss_val:.4f}  ce_loss={ce_val:.4f}  bits/dim={bpd:.2f}  logits={out['logits'].shape}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/igpt_cifar10_s_rgb.yaml')
    parser.add_argument('--batch_size', type=int, default=1)
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

    if args.batch_size < 1:
        parser.error("--batch_size 必须 >= 1")
    image_size = int(mcfg["image_size"])
    in_channels = int(mcfg["in_channels"])
    x = torch.rand(
        args.batch_size, in_channels, image_size, image_size, device=device,
    )

    # ── 1. 默认配置 forward（按 config.model.type 分发到 iGPT / CC-iGPT） ──
    model_type = mcfg.get("type", "igpt")
    print(f"\n=== Test 1: Default Config (model.type={model_type}) ===")
    if model_type == "ccigpt":
        model = _build_ccigpt_from_config(mcfg, device)
    else:
        model = _build_model_from_config(mcfg, device)
    out = model(x)
    _check_finite(out, f"default ({model_type})")

    # Weight tying 验证
    if model_type == "ccigpt":
        assert model.fine.head.weight is model.fine.token_embed.weight, "Weight tying failed on fine!"
        assert model.coarse.head.weight is model.coarse.token_embed.weight, "Weight tying failed on coarse!"
        print("  [weight tying] OK — fine.head & coarse.head 都共享 embed")
    else:
        assert model.head.weight is model.token_embed.weight, "Weight tying failed!"
        print("  [weight tying] OK — head.weight is token_embed.weight")
        # 子像素 AR 是唯一序列布局，channel_embed 永远存在
        assert hasattr(model, 'channel_embed')
        assert model.channel_embed.weight.shape == (in_channels, mcfg["d_model"])
        out['loss'].backward()
        assert all(p.grad is not None for p in model.parameters() if p.requires_grad)
        print("  [backward] all grads computed: OK")

    # ── 2. CC-iGPT smoke (硬编码 mini 配置，与 Test 1 真实 config 路径互补) ──
    print("\n=== Test 2: CC-iGPT (Coarse-Conditioned iGPT, mini hardcoded) ===")
    # mini smoke 固定在至多 32px，避免 IN64 配置再额外构造一个 12288-token 模型。
    mini_image_size = min(image_size, 32)
    mini_pool_factor = int(mcfg.get("pool_factor", 4))
    if mini_image_size % mini_pool_factor != 0:
        mini_pool_factor = 2
    model5 = CCIGPT(
        image_size=mini_image_size, in_channels=in_channels,
        vocab_size=mcfg["vocab_size"],
        pool_factor=mini_pool_factor,
        fine_d_model=mcfg["d_model"], fine_N=2,
        fine_h=mcfg["h"], fine_d_ff=mcfg["d_ff"],
        coarse_d_model=128, coarse_N=2, coarse_h=4, coarse_d_ff=344,
        dropout=0.0,
    ).to(device)
    x_mini = torch.rand(
        args.batch_size, in_channels, mini_image_size, mini_image_size,
        device=device,
    )
    out5 = model5(x_mini)
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
