#!/usr/bin/env python3
"""
评测脚本 — 加载训练好的 checkpoint，计算 CIFAR test set bits/dim (bpd)，
并与传统方法（PNG/WebP）和学术 baseline 对比。

术语: 本仓库主指标为 **bpd (bits per dimension/sub-pixel)**，与 iGPT /
PixelCNN++ / Sparse Transformer 等基线原文口径一致；
真正的 bits-per-pixel = bpd × C（彩色图 C=3）。

功能:
  1. 单模型评测: --checkpoint best.pth
  2. Per-channel bpd 分解: --per_channel (Y/Cb/Cr 或 R/G/B)
  3. SWA checkpoint 对比: --swa (同时评测 best.pth 和 swa.pth)
  4. 传统方法对比: --traditional (PNG/WebP lossless bpd)
  5. Per-position bpd 热力图: --heatmap (仅 iGPT)

输出 Markdown 格式的对比表格，可直接粘贴到论文中。

Usage:
    # 单模型评测
    python scripts/evaluate.py --config configs/igpt_cifar10_s.yaml \
        --checkpoint experiments/igpt_cifar10_s/checkpoints/best.pth

    # Per-channel bpd 分解
    python scripts/evaluate.py --config configs/igpt_cifar10_s.yaml \
        --checkpoint best.pth --per_channel

    # SWA vs best 对比
    python scripts/evaluate.py --config configs/igpt_cifar10_s.yaml \
        --checkpoint experiments/exp/checkpoints/best.pth --swa

    # 传统方法对比
    python scripts/evaluate.py --config configs/igpt_cifar10_s.yaml \
        --checkpoint best.pth --traditional

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
import re
import argparse
import yaml
import math
import torch
import torch.nn.functional as F
import numpy as np
from contextlib import nullcontext
from collections import OrderedDict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from torch.amp import autocast
from torch.utils.data import DataLoader
from src.mdlic.models.igpt import IGPT, rgb_to_ycbcr_int
from src.mdlic.models.cc_igpt import CCIGPT
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
def evaluate_model(model, loader, device, amp_dtype=None):
    """
    评估模型在数据集上的 bits/dim (bpd)。

    返回:
      bpd_mean: float — 平均 bpd
      bpd_std:  float — bpd 标准差（per-batch）
      bpd_list: list[float] — 每个 batch 的 bpd
      extras:   dict — 可选的额外字段（CC-iGPT 时含 ce_coarse / ce_fine / ctx_alpha）
    """
    model.eval()
    bpd_per_batch = []
    bpd_weighted_sum = 0.0
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

        if "bpd" in out and out["bpd"] is not None:
            bpd = out["bpd"]
        else:
            bpd = compute_bpd(
                out["ce_loss"],
                unit=out.get("loss_unit", "per_subpixel_nat"),
                in_channels=C,
            )
        bpd_val = bpd.item()
        bpd_per_batch.append(bpd_val)
        bpd_weighted_sum += bpd_val * B
        n_total += B

        if "ce_loss_coarse" in out and out["ce_loss_coarse"] is not None:
            is_ccigpt = True
            ce_c_sum += out["ce_loss_coarse"].item() * B
            ce_f_sum += out["ce_loss_fine"].item() * B
            if "ctx_alpha" in out and out["ctx_alpha"] is not None:
                alpha_sum += out["ctx_alpha"].item() * B

    bpd_mean = bpd_weighted_sum / n_total
    bpd_std = float(np.std(bpd_per_batch))

    extras = {}
    if is_ccigpt:
        extras["ce_coarse"] = ce_c_sum / n_total
        extras["ce_fine"] = ce_f_sum / n_total
        extras["ctx_alpha"] = alpha_sum / n_total
    return bpd_mean, bpd_std, bpd_per_batch, extras


@torch.no_grad()
def evaluate_per_channel(model, loader, device, amp_dtype=None,
                         color_transform: str = "bt601", use_subpixel_ar=False):
    """
    Per-channel bits/dim 分解 — 分别计算 Y/Cb/Cr 或 Y/Co/Cg 或 R/G/B 的 bpd。

    原理:
      模型预测整个 token 序列，将 logits 和 targets 按通道拆分后分别计算 CE loss。

      序列布局:
        - channel-first (use_subpixel_ar=False): [ch0_all, ch1_all, ch2_all]
        - pixel-first  (use_subpixel_ar=True):  [c0_p0,c1_p0,c2_p0, c0_p1,...]

      bpd_channel = CE_channel / ln(2)
      bpd_total = mean(bpd_ch0, bpd_ch1, bpd_ch2)  (三通道 token 数相等)

    参数:
      model: IGPT 模型
      loader: DataLoader
      device: torch.device
      amp_dtype: AMP 精度
      color_transform: "bt601" | "ycocg_r" | "none"，决定通道命名
      use_subpixel_ar: bool — 是否使用子像素自回归（pixel-first 布局）

    返回:
      channel_bpds: dict[str, (float, float)] — {通道名: (bpd_mean, bpd_std)}
      total_bpd: (float, float) — 总 bpd (mean, std)
    """
    model.eval()
    use_amp = amp_dtype is not None and device.type == 'cuda'
    if color_transform == "bt601":
        channel_names = ["Y", "Cb", "Cr"]
    elif color_transform == "ycocg_r":
        channel_names = ["Y", "Co", "Cg"]
    else:
        channel_names = ["R", "G", "B"]

    # 每个通道的 CE loss 列表
    channel_ce = {ch: [] for ch in channel_names}

    for batch in loader:
        if isinstance(batch, (list, tuple)):
            x = batch[0]
        else:
            x = batch
        x = x.to(device)
        B, C, H, W = x.shape
        pixels_per_channel = H * W

        with autocast(device_type="cuda", dtype=amp_dtype) if use_amp else nullcontext():
            out = model(x)

        if out["logits"] is None:
            if getattr(model, "output_head", "softmax") == "dmol":
                raise RuntimeError(
                    "evaluate_per_channel 不支持 DMoL 输出头：DMoL 的 mixture 参数"
                    "无法按 token 通道独立切片。请去掉 --per_channel，或在 softmax "
                    "ckpt 上跑此项。"
                )
            raise RuntimeError(
                "evaluate_per_channel: model returned logits=None even after disabling "
                "fused linear CE. Please check igpt.py forward path."
            )
        logits = out["logits"].float()  # (B, T, vocab_size), T = seq_len - 1

        # 复用模型自身的 tokenize 路径（保证与 forward 完全一致）
        tokens = model._tokenize(x)
        target_tokens = tokens[:, 1:]    # (B, T)

        # 按通道拆分: 根据序列布局提取每通道的 token
        for i, ch_name in enumerate(channel_names):
            if use_subpixel_ar:
                # pixel-first: target_tokens[t] 对应 tokens[t+1]，通道 = (t+1) % C
                T = target_tokens.shape[1]
                ch_mask = ((torch.arange(T, device=device) + 1) % C) == i
                if not ch_mask.any():
                    continue
                ch_logits = logits[:, ch_mask]      # (B, ch_len, vocab)
                ch_targets = target_tokens[:, ch_mask]  # (B, ch_len)
            else:
                # channel-first: 通道 i 的原始 tokens 占 [i*HW, (i+1)*HW)。
                # target_tokens = tokens[:,1:]，logits[:,t] 预测 target_tokens[:,t] = tokens[:,t+1]。
                # 因此"target 属于通道 i"等价于 tokens 下标 ∈ [i*HW, (i+1)*HW)
                # 对应 target_tokens 下标 ∈ [i*HW - 1, (i+1)*HW - 1)（通道 0 的 token[0] 无预测）。
                ch_start = max(0, i * pixels_per_channel - 1)
                ch_end = (i + 1) * pixels_per_channel - 1
                ch_end = min(ch_end, logits.shape[1])
                if ch_start >= ch_end:
                    continue
                ch_logits = logits[:, ch_start:ch_end]    # (B, ch_len, vocab)
                ch_targets = target_tokens[:, ch_start:ch_end]  # (B, ch_len)

            ch_ce = F.cross_entropy(
                ch_logits.reshape(-1, ch_logits.shape[-1]),
                ch_targets.reshape(-1),
                reduction="mean"
            )
            channel_ce[ch_name].append(ch_ce.item())

    # 汇总
    channel_bpds = {}
    for ch_name in channel_names:
        if channel_ce[ch_name]:
            ce_mean = float(np.mean(channel_ce[ch_name]))
            ce_std = float(np.std(channel_ce[ch_name]))
            # bpd = CE / ln(2)  (单通道，不乘 C)
            bpd_mean = ce_mean / math.log(2)
            bpd_std = ce_std / math.log(2)
            channel_bpds[ch_name] = (bpd_mean, bpd_std)

    # 总 bpd = 三通道 bpd 的平均（保持 bits/dim 单位与主流程一致）。
    # 三通道 token 数相等，sum/C 就是 token-level mean 的恢复。
    n_channels = len(channel_bpds)
    if n_channels > 0:
        total_mean = sum(v[0] for v in channel_bpds.values()) / n_channels
        total_std = math.sqrt(sum(v[1]**2 for v in channel_bpds.values())) / n_channels
    else:
        total_mean, total_std = 0.0, 0.0

    return channel_bpds, (total_mean, total_std)


@torch.no_grad()
def evaluate_position_bpp(model, loader, device, amp_dtype=None,
                           image_size=32, in_channels=3, color_transform: str = "bt601",
                           use_subpixel_ar=False):
    """
    Per-position BPP 热力图 — 计算每个像素位置的平均 bits-per-pixel。

    对于 CIFAR-100 32×32×3，每个位置有一个 token，
    将 per-token CE loss 重新 reshape 回 (H, W, C) 并对 batch 取平均。
    最终输出 (H, W) 的 BPP 热力图（对 C 通道求和 → bits/pixel 单位，
    与主流程的 bpd = bits/dim 差 C 倍）。

    用途:
      - 分析图像哪些空间位置难以压缩（高 BPP 区域）
      - 观察自回归方向（光栅扫描序列头部 vs 尾部）对压缩率的影响
      - 论文中可视化分析素材

    参数:
      model: IGPT 模型
      loader: DataLoader
      device: torch.device
      amp_dtype: AMP 精度
      image_size: int — 图像边长
      in_channels: int — 通道数
      color_transform: "bt601" | "ycocg_r" | "none"，决定通道命名

    返回:
      heatmap: np.ndarray (H, W) — 每个像素位置的平均 BPP (bits/pixel)
      channel_heatmaps: dict[str, np.ndarray] — 每通道 (H, W) bpd 热力图
    """
    model.eval()
    use_amp = amp_dtype is not None and device.type == 'cuda'
    C = in_channels
    H = W = image_size
    seq_len = H * W * C
    if color_transform == "bt601":
        channel_names = ["Y", "Cb", "Cr"]
    elif color_transform == "ycocg_r":
        channel_names = ["Y", "Co", "Cg"]
    else:
        channel_names = ["R", "G", "B"]

    # 累积每个 token 位置的 CE loss
    # 序列布局: [ch0_pixel0, ch0_pixel1, ..., ch1_pixel0, ..., ch2_pixelN]
    # token 数 = seq_len - 1（NTP 偏移 1）
    position_ce_sum = torch.zeros(seq_len - 1, device=device)
    n_samples = 0

    for batch in loader:
        if isinstance(batch, (list, tuple)):
            x = batch[0]
        else:
            x = batch
        x = x.to(device)
        B_cur = x.shape[0]

        with autocast(device_type="cuda", dtype=amp_dtype) if use_amp else nullcontext():
            out = model(x)

        logits = out["logits"]
        if logits is None:
            if getattr(model, "output_head", "softmax") == "dmol":
                raise RuntimeError(
                    "evaluate_position_bpp 当前实现依赖 softmax logits；DMoL 路径"
                    "下需要走 dmol_log_prob 重写热力图（per-pixel NLL）。请去掉 "
                    "--heatmap，或后续补 DMoL 分支。"
                )
            raise RuntimeError(
                "evaluate_position_bpp: model returned logits=None. "
                "Please check igpt.py forward path."
            )
        logits = logits.float()

        # 复用模型自身的 tokenize 路径
        tokens = model._tokenize(x)
        target_tokens = tokens[:, 1:]

        # Per-token CE loss: (B, T)
        per_token_ce = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            target_tokens.reshape(-1),
            reduction="none"
        ).reshape(B_cur, -1)  # (B, seq_len-1)

        # 对 batch 维度求和
        position_ce_sum += per_token_ce.sum(dim=0)
        n_samples += B_cur

    # 平均 CE per position
    position_ce_avg = (position_ce_sum / n_samples).cpu().numpy()  # (seq_len-1,)

    # 转换为 bpd: CE / ln(2)（单 token 单位即 bits/dim；后续按 C 求和才是 bits/pixel）
    position_bpd = position_ce_avg / math.log(2)

    # Reshape 回空间结构
    # 注意: 序列的第一个 token 没有 prediction（被偏移掉了），
    # 所以在 position_bpd 前面填 0，使其长度 = seq_len
    position_bpd_full = np.zeros(seq_len)
    position_bpd_full[1:] = position_bpd

    if use_subpixel_ar:
        # pixel-first: (seq_len,) → (H, W, C) → (C, H, W)
        position_bpd_hwc = position_bpd_full.reshape(H, W, C)
        position_bpd_chw = position_bpd_hwc.transpose(2, 0, 1)  # (C, H, W)
    else:
        # channel-first: (seq_len,) → (C, H, W)
        position_bpd_chw = position_bpd_full.reshape(C, H, W)

    # 总 BPP 热力图: 对 C 通道求和 → (H, W)
    # 注意: heatmap 单位是 bits/pixel（每像素 C=3 个 token bpd 之和），
    # 与主流程 evaluate_model 返回的 bits/dim (bpd) 单位差 C 倍。
    heatmap = position_bpd_chw.sum(axis=0)

    # 每通道热力图（单位仍是 bpd）
    channel_heatmaps = {}
    for i, ch_name in enumerate(channel_names):
        channel_heatmaps[ch_name] = position_bpd_chw[i]

    return heatmap, channel_heatmaps


def save_heatmap(heatmap, output_path, title="BPP Heatmap", vmin=None, vmax=None):
    """
    将 BPP 热力图保存为 PNG 图片。

    使用 matplotlib 的 'hot' colormap（高 BPP → 亮色/红色，低 BPP → 暗色）。
    若 matplotlib 不可用，保存为 .npy 文件。

    参数:
      heatmap: np.ndarray (H, W)
      output_path: str — 输出路径（.png）
      title: str — 图片标题
      vmin, vmax: float — colorbar 范围
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # 无 GUI 后端
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        im = ax.imshow(heatmap, cmap='hot', interpolation='nearest',
                        vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel('Width')
        ax.set_ylabel('Height')
        fig.colorbar(im, ax=ax, label='BPP (bits/pixel)')
        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  保存热力图: {output_path}")
    except ImportError:
        # matplotlib 不可用，保存原始数据
        npy_path = output_path.replace('.png', '.npy')
        np.save(npy_path, heatmap)
        print(f"  matplotlib 不可用，保存 numpy 数组: {npy_path}")


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
                         channel_bpds=None, model_label="iGPT (Ours)"):
    """
    打印 Markdown 格式的结果表格。

    参数:
      dataset_name: str — "cifar10" / "cifar100"
      model_bpd: float — 本文模型 bits/dim
      model_std: float — bits/dim 标准差
      traditional_results: dict[str, (float, float)] — 传统方法 {name: (bpd, std)}
      channel_bpds: dict[str, (float, float)] — per-channel {name: (bpd, std)}
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

    # Per-channel bits/dim
    if channel_bpds:
        print()
        print("| 通道 | bits/dim ↓ |")
        print("|------|-----------|")
        for ch_name, (bpd, std) in channel_bpds.items():
            print(f"| {ch_name} | {bpd:.4f} ± {std:.4f} |")
        # Total 是三通道的平均（bits/dim 单位），与 evaluate_model 返回值一致；
        # 曾经写 sum() 会多出 ~3× 的伪 Total。
        total_bpd = sum(v[0] for v in channel_bpds.values()) / len(channel_bpds)
        print(f"| **Total** | **{total_bpd:.4f}** |")
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
    """从 config 获取 AMP dtype"""
    amp_dtype_str = config["train"].get("amp_dtype", "bf16")
    return torch.bfloat16 if amp_dtype_str == "bf16" else torch.float16, amp_dtype_str


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
    print(f"\n评测中... (AMP: {amp_dtype_str})")
    bpd_mean, bpd_std, _, extras = evaluate_model(model, test_loader, device,
                                                    amp_dtype=amp_dtype)
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

    # Per-channel bits/dim（可选，仅 iGPT 支持；CC-iGPT 联合 coarse+fine 序列与单尺度通道切分不兼容）
    channel_bpds = None
    if args.per_channel:
        if model_type == "ccigpt":
            print("\n[跳过 per_channel] CC-iGPT 含 coarse 子分支，per-channel 切分仅对 fine 有意义，待实现。")
        else:
            print("\n计算 per-channel bits/dim...")
            color_transform = getattr(model, "color_transform", "bt601")
            use_subpixel_ar = mcfg.get("use_subpixel_ar", False)
            channel_bpds, (total_bpd, total_std) = evaluate_per_channel(
                model, test_loader, device, amp_dtype=amp_dtype,
                color_transform=color_transform, use_subpixel_ar=use_subpixel_ar
            )
            for ch_name, (ch_bpd, ch_std) in channel_bpds.items():
                print(f"  {ch_name}: {ch_bpd:.4f} ± {ch_std:.4f}")
            print(f"  Total: {total_bpd:.4f} ± {total_std:.4f}")

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
                         traditional_results, channel_bpds,
                         model_label=f"{model_type.upper()} (Ours)")

    # Per-position BPP 热力图（可选，仅 iGPT — CC-iGPT 含 coarse 分支待实现；
    # 注意热力图单位是 bits/pixel = bpd × C，与表格里的 bits/dim 主指标差 C 倍）
    if args.heatmap:
        if model_type == "ccigpt":
            print("\n[跳过 heatmap] CC-iGPT 含 coarse 子分支，per-position 热力图待实现。")
        else:
            print("\n生成 per-position BPP 热力图 (bits/pixel)...")
            color_transform = getattr(model, "color_transform", "bt601")
            use_subpixel_ar = mcfg.get("use_subpixel_ar", False)
            image_size = mcfg.get("image_size", 32)
            in_channels = mcfg.get("in_channels", 3)
            heatmap, channel_heatmaps = evaluate_position_bpp(
                model, test_loader, device, amp_dtype=amp_dtype,
                image_size=image_size, in_channels=in_channels,
                color_transform=color_transform, use_subpixel_ar=use_subpixel_ar
            )
            # 保存到 checkpoint 同目录
            ckpt_dir = os.path.dirname(args.checkpoint) or "."
            save_heatmap(heatmap, os.path.join(ckpt_dir, "bpp_heatmap_total.png"),
                         title=f"BPP Heatmap — {dataset_name}")
            for ch_name, ch_hm in channel_heatmaps.items():
                save_heatmap(ch_hm, os.path.join(ckpt_dir, f"bpp_heatmap_{ch_name}.png"),
                             title=f"bpd Heatmap — {ch_name}")
            # 打印统计
            print(f"  Total BPP range: [{heatmap.min():.4f}, {heatmap.max():.4f}]")
            print(f"  Mean: {heatmap.mean():.4f}, Std: {heatmap.std():.4f}")


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
                                                amp_dtype=amp_dtype)
    results.append(("best.pth", bpd_best, std_best))
    print(f"best.pth  bits/dim: {bpd_best:.4f} ± {std_best:.4f}")

    # 评测 swa
    model = _build_from_config(mcfg, device)
    _load_checkpoint(model, swa_path, device)
    bpd_swa, std_swa, _, _ = evaluate_model(model, test_loader, device,
                                               amp_dtype=amp_dtype)
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
  # 单模型
  python scripts/evaluate.py --config configs/igpt_cifar10_s.yaml \\
      --checkpoint best.pth

  # Per-channel + 传统方法
  python scripts/evaluate.py --config configs/igpt_cifar10_s.yaml \\
      --checkpoint best.pth --per_channel --traditional

  # SWA 对比
  python scripts/evaluate.py --config configs/igpt_cifar10_s.yaml \\
      --checkpoint experiments/exp/checkpoints/best.pth --swa
        """
    )
    parser.add_argument('--config', type=str, required=True,
                        help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='模型 checkpoint 路径（best.pth）')
    parser.add_argument('--traditional', action='store_true',
                        help='同时计算 PNG/WebP 传统方法 bits/dim')
    parser.add_argument('--per_channel', action='store_true',
                        help='计算 per-channel bits/dim 分解（Y/Cb/Cr）')
    parser.add_argument('--heatmap', action='store_true',
                        help='生成 per-position BPP 热力图（保存为 PNG，单位 bits/pixel）')
    parser.add_argument('--swa', action='store_true',
                        help='同时评测 SWA checkpoint（swa.pth vs best.pth）')
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
