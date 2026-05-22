#!/usr/bin/env python3
"""
评测脚本 — 加载训练好的 checkpoint，计算 CIFAR test set bits/dim (bpd)，
并与传统方法（PNG/WebP）和学术 baseline 对比。

术语: 本仓库主指标为 **bpd (bits per dimension/sub-pixel)**，与 iGPT /
PixelCNN++ / Sparse Transformer 等基线原文口径一致；
真正的 bits-per-pixel = bpd × C（彩色图 C=3）。

功能:
  1. 单模型评测: --checkpoint best.pth
  2. SWA checkpoint 对比: --swa (同时评测 best.pth 和 swa.pth)
  3. 传统方法对比: --traditional (PNG/WebP lossless bpd)
  4. TTA hflip: --tta_hflip (评测时对每张图取 x 与 hflip(x) 的 bpd 均值)

输出 Markdown 格式的对比表格，可直接粘贴到论文中。

Usage:
    # 单模型评测 (主表数字)
    python scripts/evaluate.py --config configs/ccigpt_cifar10_s_rgb_ronly.yaml \
        --checkpoint experiments/ccigpt_cifar10_s_rgb_ronly/checkpoints/best.pth --tta_hflip

    # SWA vs best 对比
    python scripts/evaluate.py --config configs/ccigpt_cifar10_s_rgb_ronly.yaml \
        --checkpoint experiments/ccigpt_cifar10_s_rgb_ronly/checkpoints/best.pth --swa

参考:
  [1] Shannon, "A Mathematical Theory of Communication," 1948.
      bits/dim = -log₂ p(x) = CE / ln(2)
  [2] Salimans et al., "PixelCNN++," ICLR 2017 — CIFAR-10: 2.92 bits/dim
  [3] Parmar et al., "Image Transformer," ICML 2018 — CIFAR-10: 2.90 bits/dim
  [4] Chen et al., "PixelSNAIL," ICML 2018 — CIFAR-10: 2.85 bits/dim
"""

import os
import sys
import io
import argparse
import yaml
import math
import torch
import torch.nn.functional as F
import numpy as np
from contextlib import nullcontext

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from torch.amp import autocast
from torch.utils.data import DataLoader
from src.mdlic.utils import compute_bpd, clean_state_dict
from scripts.train import _build_model_from_config, _build_ccigpt_from_config


def _build_from_config(mcfg: dict, device):
    """根据 model.type 分发到 IGPT / CC-iGPT 构建函数。"""
    model_type = mcfg.get("type", "igpt")
    if model_type == "ccigpt":
        return _build_ccigpt_from_config(mcfg, device)
    return _build_model_from_config(mcfg, device)


# ── 学术 Baseline（直接引用论文数字）──
# 注意: PixelCNN++ / PixelSNAIL 使用 discretized logistic mixture likelihood，
# 本文使用 categorical CE（vocab=256）。两者都是 bits/dim，物理含义一致。
# Ref: [2][3][4]
ACADEMIC_BASELINES = {
    "cifar10": [
        ("PixelCNN++ [Salimans 2017]", 2.92),
        ("Image Transformer [Parmar 2018]", 2.90),
        ("PixelSNAIL [Chen 2018]", 2.85),
    ],
    "cifar100": [
        # CIFAR-100 上这些方法没有公开报告的 bits/dim
        # 仅展示本文结果
    ],
}


@torch.no_grad()
def evaluate_model(model, loader, device, amp_dtype=None, tta_hflip: bool = False):
    """
    评估模型在数据集上的 bits/dim (bpd)。

    返回:
      bpd_mean: float — 平均 bpd
      bpd_std:  float — bpd 标准差（batch-level 加权，与 train.py:validate() 同口径）
      bpd_list: list[float] — 每个 batch 的 bpd（保留供 caller 自定义聚合）
      extras:   dict — 可选的额外字段（CC-iGPT 时含 ce_coarse / ce_fine / ctx_alpha）

    TTA (Test-Time Augmentation):
      tta_hflip=True 时对每张图同时跑 x 与 hflip(x) 两次 forward，取 bpd 均值。
      hflip(x) 的 -log p 仍是 H(X) 的合法上界（hflip 在 RGB-bit-exact 域是确定函数），
      平均能降低估计方差。Ref: Sparse Transformer (Child 2019, §4.2) 采用类似 ensemble.
    """
    model.eval()
    bpd_per_batch = []
    bpd_weighted_sum = 0.0
    bpd_sq_weighted_sum = 0.0
    n_total = 0
    use_amp = amp_dtype is not None and device.type == 'cuda'

    # CC-iGPT 额外聚合 CE_c / CE_f / α。is_ccigpt 由 forward 输出 keys 推断，
    # 不依赖 isinstance（DDP wrap 后 model 是 DDP 而非 CCIGPT）。
    is_ccigpt = False
    ce_c_sum = ce_f_sum = alpha_sum = 0.0

    for batch in loader:
        if isinstance(batch, (list, tuple)):
            x = batch[0]
        else:
            x = batch
        x = x.to(device)
        B, C, _, _ = x.shape

        with autocast(device_type="cuda", dtype=amp_dtype) if use_amp else nullcontext():
            out = model(x)
            if tta_hflip:
                out_flip = model(torch.flip(x, dims=[-1]))

        if "bpd" in out and out["bpd"] is not None:
            bpd = out["bpd"]
            if tta_hflip:
                bpd = (bpd + out_flip["bpd"]) * 0.5
        else:
            ce = out["ce_loss"]
            if tta_hflip:
                ce = (ce + out_flip["ce_loss"]) * 0.5
            bpd = compute_bpd(ce)
        bpd_val = bpd.item()
        bpd_per_batch.append(bpd_val)
        bpd_weighted_sum += bpd_val * B
        bpd_sq_weighted_sum += (bpd_val ** 2) * B
        n_total += B

        if "ce_loss_coarse" in out and out["ce_loss_coarse"] is not None:
            is_ccigpt = True
            ce_c = out["ce_loss_coarse"]
            ce_f = out["ce_loss_fine"]
            if tta_hflip:
                ce_c = (ce_c + out_flip["ce_loss_coarse"]) * 0.5
                ce_f = (ce_f + out_flip["ce_loss_fine"]) * 0.5
            ce_c_sum += ce_c.item() * B
            ce_f_sum += ce_f.item() * B
            if "ctx_alpha" in out and out["ctx_alpha"] is not None:
                alpha_sum += out["ctx_alpha"].item() * B

    bpd_mean = bpd_weighted_sum / n_total
    # batch-level 加权 std，与 train.py:validate() 同公式：avoid np.std(batch_means)
    # 在末尾不足 batch 时给小 batch 过高权重。
    bpd_var = max(bpd_sq_weighted_sum / n_total - bpd_mean ** 2, 0.0)
    bpd_std = float(math.sqrt(bpd_var))

    extras = {}
    if is_ccigpt:
        extras["ce_coarse"] = ce_c_sum / n_total
        extras["ce_fine"] = ce_f_sum / n_total
        extras["ctx_alpha"] = alpha_sum / n_total
    return bpd_mean, bpd_std, bpd_per_batch, extras



def compute_traditional_bpd(dataset, method="png"):
    """
    计算传统无损压缩方法在数据集上的 bits/dim (bpd)。

    将每张图片编码为内存中的 PNG/WebP 字节流，
    bpd = 压缩后字节 × 8 / 子像素总数 (= H × W × C)。

    口径说明: PixelCNN++ / Sparse Transformer 等基线在 CIFAR-10 上报告的
    PNG≈5.87、WebP≈5.02 均为 bits/dim 单位（除以 H·W·C），与本仓库主指标
    一致。若除以 H·W 则得到的是真 bits-per-pixel（bpd × 3），数值会高 3 倍。

    参数:
      dataset: torchvision dataset（返回 (tensor, label)）
      method: "png" 或 "webp"

    返回:
      bpd_mean: float
      bpd_std:  float
    """
    from PIL import Image

    format_map = {"png": "PNG", "webp": "WEBP"}
    fmt = format_map.get(method, "PNG")
    save_kwargs = {"lossless": True} if method == "webp" else {}

    bpd_list = []
    for i in range(len(dataset)):
        img_tensor, _ = dataset[i]
        # tensor (C, H, W) [0,1] → PIL Image
        img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        img = Image.fromarray(img_np)

        # 编码到内存
        buf = io.BytesIO()
        img.save(buf, format=fmt, **save_kwargs)
        compressed_bytes = buf.tell()

        # bpd = 压缩字节 × 8 / 总子像素数 (H·W·C)
        C, H, W = img_tensor.shape
        total_dims = H * W * C
        bpd = (compressed_bytes * 8) / total_dims
        bpd_list.append(bpd)

    return float(np.mean(bpd_list)), float(np.std(bpd_list))


def print_results_table(dataset_name, model_bpd, model_std,
                         traditional_results=None,
                         model_label="iGPT (Ours)"):
    """
    打印 Markdown 格式的结果表格。

    参数:
      dataset_name: str — "cifar10" / "cifar100"
      model_bpd: float — 本文模型 bits/dim
      model_std: float — bits/dim 标准差
      traditional_results: dict[str, (float, float)] — 传统方法 {name: (bpd, std)}
    """
    print(f"\n{'='*60}")
    print(f"  评测结果 — {dataset_name.upper()}")
    print(f"{'='*60}\n")

    print("| 方法 | bits/dim ↓ | 备注 |")
    print("|------|-----------|------|")

    # 传统方法
    if traditional_results:
        for name, (bpd, std) in traditional_results.items():
            print(f"| {name} | {bpd:.2f} ± {std:.2f} | 无损压缩 |")

    # 学术 baseline
    for name, bpd in ACADEMIC_BASELINES.get(dataset_name, []):
        print(f"| {name} | {bpd:.2f} | 论文报告值 |")

    # 本文
    print(f"| **{model_label}** | **{model_bpd:.4f} ± {model_std:.4f}** | **本文** |")
    print()


def _load_dataset(config):
    """加载测试数据集"""
    from torchvision import transforms
    from torchvision.datasets import CIFAR10, CIFAR100

    dataset_name = config["data"].get("dataset", "cifar100")
    if dataset_name == "imagenet32_npy":
        from src.mdlic.data.imagenet32_npy import ImageNet32Npy
        test_dataset = ImageNet32Npy(root=config["data"]["valid"], split="val")
        return test_dataset, dataset_name
    if dataset_name not in ("cifar10", "cifar100"):
        raise ValueError(
            f"未知 dataset: '{dataset_name}'，支持 cifar10/cifar100/imagenet32_npy"
        )
    transform = transforms.ToTensor()
    DatasetClass = CIFAR10 if dataset_name == "cifar10" else CIFAR100
    test_dataset = DatasetClass(root=config["data"]["valid"], train=False,
                                download=False, transform=transform)
    return test_dataset, dataset_name


def _load_checkpoint(model, ckpt_path, device):
    """加载 checkpoint（支持完整 ckpt 和纯 state_dict）。

    透明剥 `module.` / `_orig_mod.` 前缀并过滤遗留 persistent=False buffer
    (如 RoPE inv_freq)，兼容历史 DDP / torch.compile ckpt。
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(clean_state_dict(ckpt['model_state_dict']))
        epoch = ckpt.get('epoch', '?')
        return epoch
    else:
        model.load_state_dict(clean_state_dict(ckpt))
        return '?'


def _get_amp_dtype(config):
    """从 config 获取 AMP dtype，与 train.py 保持同口径。

    支持 fp16 / bf16 / null / "none" / "fp32" 五档；后三档表示走 fp32（amp_dtype=None）。
    口径漂移会让 evaluate 与训练时精度不一致，导致 bpd 数值不可比。
    """
    # 默认 fp16 与 train.py:594 同口径；若 yaml 省略 amp_dtype，evaluate 与训练
    # 走完全相同的精度路径，避免 logits 接近 logsumexp 边界时 bpd 数值不可比。
    amp_cfg = config["train"].get("amp_dtype", "fp16")
    if amp_cfg in (None, "none", "fp32"):
        return None, str(amp_cfg)
    if amp_cfg == "bf16":
        return torch.bfloat16, "bf16"
    if amp_cfg == "fp16":
        return torch.float16, "fp16"
    raise ValueError(
        f"未知 train.amp_dtype: '{amp_cfg}'，支持 fp16/bf16/null/none/fp32"
    )


def cmd_single(args, config, device):
    """单模型评测"""
    test_dataset, dataset_name = _load_dataset(config)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=2, pin_memory=True)
    print(f"Dataset: {dataset_name} test ({len(test_dataset)} images)")

    mcfg = config["model"]
    model_type = mcfg.get("type", "igpt")
    model = _build_from_config(mcfg, device)
    epoch = _load_checkpoint(model, args.checkpoint, device)
    print(f"Loaded checkpoint: epoch {epoch} (model_type={model_type})")

    amp_dtype, amp_dtype_str = _get_amp_dtype(config)

    # 基本 bits/dim 评测
    print(f"\n评测中... (AMP: {amp_dtype_str}, TTA hflip: {args.tta_hflip})")
    bpd_mean, bpd_std, _, extras = evaluate_model(model, test_loader, device,
                                                    amp_dtype=amp_dtype,
                                                    tta_hflip=args.tta_hflip)
    print(f"{model_type.upper()} bits/dim: {bpd_mean:.4f} ± {bpd_std:.4f}")
    if extras:
        # CC-iGPT 多输出 CE_c / CE_f / α，便于诊断 fine 弱 vs coarse overhead 过大
        ce_c = extras["ce_coarse"]
        ce_f = extras["ce_fine"]
        N_c = model.coarse.seq_len
        N_f = model.fine.seq_len
        bpd_c_share = ce_c * N_c / math.log(2.0) / N_f
        bpd_f_share = ce_f * N_f / math.log(2.0) / N_f
        print(f"  CE_coarse = {ce_c:.4f}  → bpd_share = {bpd_c_share:.4f} ({100*bpd_c_share/bpd_mean:.1f}%)")
        print(f"  CE_fine   = {ce_f:.4f}  → bpd_share = {bpd_f_share:.4f} ({100*bpd_f_share/bpd_mean:.1f}%)")
        print(f"  ctx_alpha = {extras['ctx_alpha']:.4f}")

    # 传统方法（可选）
    traditional_results = None
    if args.traditional:
        print("\n计算传统方法 bits/dim...")
        traditional_results = {}

        png_bpd, png_std = compute_traditional_bpd(test_dataset, method="png")
        traditional_results["PNG (lossless)"] = (png_bpd, png_std)
        print(f"  PNG:  {png_bpd:.2f} ± {png_std:.2f}")

        try:
            webp_bpd, webp_std = compute_traditional_bpd(test_dataset, method="webp")
            traditional_results["WebP (lossless)"] = (webp_bpd, webp_std)
            print(f"  WebP: {webp_bpd:.2f} ± {webp_std:.2f}")
        except Exception as e:
            print(f"  WebP: 跳过 ({e})")

    print_results_table(dataset_name, bpd_mean, bpd_std,
                         traditional_results,
                         model_label=f"{model_type.upper()} (Ours)")


def cmd_swa(args, config, device):
    """SWA vs best checkpoint 对比评测"""
    test_dataset, dataset_name = _load_dataset(config)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=2, pin_memory=True)

    mcfg = config["model"]
    amp_dtype, amp_dtype_str = _get_amp_dtype(config)

    # best.pth
    ckpt_dir = os.path.dirname(args.checkpoint)
    best_path = args.checkpoint
    swa_path = os.path.join(ckpt_dir, "swa.pth")

    if not os.path.exists(swa_path):
        print(f"SWA checkpoint 不存在: {swa_path}")
        print("请确保训练时启用了 SWA (train.swa.enabled=true)")
        return

    results = []

    # 评测 best
    model = _build_from_config(mcfg, device)
    _load_checkpoint(model, best_path, device)
    bpd_best, std_best, _, _ = evaluate_model(model, test_loader, device,
                                                amp_dtype=amp_dtype,
                                                tta_hflip=args.tta_hflip)
    results.append(("best.pth", bpd_best, std_best))
    print(f"best.pth  bits/dim: {bpd_best:.4f} ± {std_best:.4f}")

    # 评测 swa
    model = _build_from_config(mcfg, device)
    _load_checkpoint(model, swa_path, device)
    bpd_swa, std_swa, _, _ = evaluate_model(model, test_loader, device,
                                               amp_dtype=amp_dtype,
                                               tta_hflip=args.tta_hflip)
    results.append(("swa.pth", bpd_swa, std_swa))
    print(f"swa.pth   bits/dim: {bpd_swa:.4f} ± {std_swa:.4f}")

    delta = bpd_swa - bpd_best
    print(f"\nΔ(bits/dim) (SWA - best): {delta:+.4f}")

    # 打印对比表格
    print(f"\n| Checkpoint | bits/dim ↓ | Δ |")
    print(f"|------------|-----------|---|")
    print(f"| best.pth | {bpd_best:.4f} ± {std_best:.4f} | — |")
    delta_str = f"{delta:+.4f}"
    note = "SWA 更优" if delta < 0 else "best 更优"
    print(f"| swa.pth  | {bpd_swa:.4f} ± {std_swa:.4f} | {delta_str} ({note}) |")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="iGPT 无损压缩评测",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 单模型 (主表数字 best + TTA hflip)
  python scripts/evaluate.py --config configs/ccigpt_cifar10_s_rgb_ronly.yaml \\
      --checkpoint experiments/ccigpt_cifar10_s_rgb_ronly/checkpoints/best.pth --tta_hflip

  # SWA 对比
  python scripts/evaluate.py --config configs/ccigpt_cifar10_s_rgb_ronly.yaml \\
      --checkpoint experiments/ccigpt_cifar10_s_rgb_ronly/checkpoints/best.pth --swa
        """
    )
    parser.add_argument('--config', type=str, required=True,
                        help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='模型 checkpoint 路径（best.pth）')
    parser.add_argument('--traditional', action='store_true',
                        help='同时计算 PNG/WebP 传统方法 bits/dim')
    parser.add_argument('--swa', action='store_true',
                        help='同时评测 SWA checkpoint（swa.pth vs best.pth）')
    parser.add_argument('--tta_hflip', action='store_true',
                        help='Test-Time Augmentation：对每张图同时跑 x 与 hflip(x)，bpd 取均值')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='评测 batch size')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 根据模式分发
    if args.swa and args.checkpoint:
        cmd_swa(args, config, device)
    elif args.checkpoint:
        cmd_single(args, config, device)
    else:
        parser.error("请指定 --checkpoint")


if __name__ == '__main__':
    main()
