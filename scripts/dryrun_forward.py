#!/usr/bin/env python3
"""
Quick forward pass sanity check — 验证模型构建和 forward 是否正常。

包含:
  1. 默认配置 forward
  2. Weight tying / post-norm / pre-norm 验证
  3. 消融 baseline 全关测试
  4. MTP 辅助头测试
  5. Logit soft-capping 测试
  6. muP 初始化 + backward 验证
  7. Fused kernel 状态检查
  8. Numerical sanity（loss 有限、BPP 合理范围）

Usage:
    python scripts/dryrun_forward.py
    python scripts/dryrun_forward.py --config configs/igpt_cifar10_baseline.yaml
"""

import os
import sys
import argparse
import yaml
import math
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.mdlic.models.igpt import IGPT
# 复用 train.py 的模型构建函数，避免重复
from scripts.train import _build_model_from_config
from src.mdlic.models.layers import get_fused_kernel_status


def _check_finite(out: dict, tag: str):
    """检查 loss 和 ce_loss 是否有限（非 NaN/Inf）。"""
    loss_val = out['loss'].item()
    ce_val = out['ce_loss'].item()
    assert math.isfinite(loss_val), f"[{tag}] loss is {loss_val} (NaN/Inf!)"
    assert math.isfinite(ce_val), f"[{tag}] ce_loss is {ce_val} (NaN/Inf!)"
    # BPP 合理范围检查: 对 vocab=256 的离散分布，BPP 理论最大值 = log2(256)*3 = 24.0
    # 初始随机模型的 BPP 应在 8 (= log2(256)) 附近，远超 24 说明有问题
    bpp = ce_val / math.log(2) * 3  # 假设 3 通道
    assert 0.0 < bpp < 30.0, f"[{tag}] BPP={bpp:.2f} 超出合理范围 (0, 30)"
    logits_info = out['logits'].shape if out['logits'] is not None else "None (fused path)"
    print(f"  [{tag}] loss={loss_val:.4f}  ce_loss={ce_val:.4f}  BPP={bpp:.2f}  logits={logits_info}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/igpt_cifar100_baseline.yaml')
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

    # ── 1. 默认配置 forward ──
    print("\n=== Test 1: Default Config ===")
    model = _build_model_from_config(mcfg, device)
    out = model(x)
    _check_finite(out, "default")

    # 验证 weight tying: token_embed.weight 和 head.weight 应为同一对象
    assert model.head.weight is model.token_embed.weight, "Weight tying failed!"
    print("  [weight tying] OK — head.weight is token_embed.weight")

    # 验证 post-norm 模式不创建 final_norm
    if mcfg.get("use_post_norm", True):
        assert not hasattr(model, 'final_norm'), "post-norm mode should not have final_norm"
        print("  [post-norm] OK — no final_norm")

    # ── 2. 消融 baseline 全关 ──
    print("\n=== Test 2: All Ablation OFF (pre-norm + ReLU + learned PE) ===")
    model2 = IGPT(
        image_size=32, in_channels=3, vocab_size=256,
        d_model=mcfg["d_model"], N=mcfg["N"], h=mcfg["h"], d_ff=mcfg["d_ff"],
        dropout=0.1,
        use_ycbcr=False, use_rope=False, use_post_norm=False,
        use_swiglu=False, use_qk_norm=False, use_depth_scaled_init=False,
        use_zloss=False, activation_checkpointing=True,
    ).to(device)
    out2 = model2(x)
    _check_finite(out2, "ablation-off")

    # 验证 pre-norm 模式创建 final_norm
    assert hasattr(model2, 'final_norm'), "pre-norm mode should have final_norm"
    print("  [pre-norm] OK — final_norm exists")

    # ── 3. MTP 辅助头 ──
    print("\n=== Test 3: MTP (Multi-Token Prediction) ===")
    model3 = IGPT(
        image_size=32, in_channels=3, vocab_size=256,
        d_model=mcfg["d_model"], N=mcfg["N"], h=mcfg["h"], d_ff=mcfg["d_ff"],
        dropout=0.1, use_mtp=True,
    ).to(device)
    out3 = model3(x, mtp_weight=0.3)
    _check_finite(out3, "mtp")
    assert hasattr(model3, 'mtp_block'), "MTP model should have mtp_block"
    print("  [mtp] OK — mtp_block exists, loss includes MTP contribution")

    # ── 4. Logit soft-capping ──
    print("\n=== Test 4: Logit Soft-Capping ===")
    model4 = IGPT(
        image_size=32, in_channels=3, vocab_size=256,
        d_model=mcfg["d_model"], N=mcfg["N"], h=mcfg["h"], d_ff=mcfg["d_ff"],
        dropout=0.1, logit_soft_cap=30.0,
    ).to(device)
    out4 = model4(x)
    _check_finite(out4, "soft-cap")
    # Logit soft-capping 后 logits 最大值应 ≤ cap（30.0）
    max_logit = out4['logits'].abs().max().item()
    assert max_logit <= 30.0 + 0.1, f"logit soft-cap failed: max_abs_logit={max_logit:.2f} > 30.0"
    print(f"  [soft-cap] OK — max |logit| = {max_logit:.2f} ≤ 30.0")

    # ── 5. muP 初始化 + backward ──
    print("\n=== Test 5: muP Init + Backward ===")
    model._init_weights_mup(base_width=64)
    out5 = model(x)
    _check_finite(out5, "mup")

    out5['loss'].backward()
    grad_ok = all(p.grad is not None for p in model.parameters() if p.requires_grad)
    assert grad_ok, "Some parameters missing gradients after backward"
    # 检查梯度是否有限
    grad_finite = all(torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None)
    assert grad_finite, "Some gradients contain NaN/Inf"
    print("  [backward] all grads computed and finite: OK")

    # ── 参数量统计 ──
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal params: {n_params:,}")
    print("\nAll checks passed!")


if __name__ == '__main__':
    main()
