#!/usr/bin/env python3
"""
评测脚本 — 加载训练好的 checkpoint，计算 CIFAR test set BPP，
并与传统方法（PNG/WebP）和学术 baseline 对比。

功能:
  1. 单模型评测: --checkpoint best.pth
  2. 消融批量评测: --ablation_dir experiments/ (扫描多个 checkpoint)
  3. Per-channel BPP 分解: --per_channel (Y/Cb/Cr 或 R/G/B)
  4. SWA checkpoint 对比: --swa (同时评测 best.pth 和 swa.pth)
  5. 传统方法对比: --traditional (PNG/WebP lossless BPP)

输出 Markdown 格式的对比表格，可直接粘贴到论文中。

Usage:
    # 单模型评测
    python scripts/evaluate.py --config configs/igpt_cifar10_baseline.yaml \
        --checkpoint experiments/igpt_cifar10_baseline/checkpoints/best.pth

    # 消融批量评测（扫描目录下所有实验）
    python scripts/evaluate.py --config configs/igpt_cifar10_baseline.yaml \
        --ablation_dir experiments/

    # Per-channel BPP 分解
    python scripts/evaluate.py --config configs/igpt_cifar10_baseline.yaml \
        --checkpoint best.pth --per_channel

    # SWA vs best 对比
    python scripts/evaluate.py --config configs/igpt_cifar10_baseline.yaml \
        --checkpoint experiments/exp/checkpoints/best.pth --swa

    # 传统方法对比
    python scripts/evaluate.py --config configs/igpt_cifar10_baseline.yaml \
        --checkpoint best.pth --traditional

参考:
  [1] Shannon, "A Mathematical Theory of Communication," 1948.
      BPP = -log₂ p(x) = CE / ln(2)
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
from collections import OrderedDict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from torch.amp import autocast
from torch.utils.data import DataLoader
from src.mdlic.models.igpt import IGPT, rgb_to_ycbcr_int
from src.mdlic.utils import compute_bpp
from scripts.train import _build_model_from_config


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

# 消融实验编号 → 配置覆盖映射
# Ref: future.md Section 3.2 消融实验表
ABLATION_CONFIGS = OrderedDict([
    ("E0", {}),                                           # Full Proposed Model
    ("E1", {"use_ycbcr": False}),                         # w/o YCbCr
    ("E2", {"use_rope": False}),                          # w/o RoPE
    ("E3", {"use_post_norm": False}),                     # w/o Post-Norm
    ("E4", {"use_swiglu": False}),                        # w/o SwiGLU
    ("E5", {"use_qk_norm": False}),                       # w/o QK-Norm
    ("E6", {"use_depth_scaled_init": False}),              # w/o 深度缩放 Init
    ("E7", {"use_zloss": False}),                         # w/o z-loss
])


@torch.no_grad()
def evaluate_model(model, loader, device, amp_dtype=None):
    """
    评估模型在数据集上的 BPP。

    返回:
      bpp_mean: float — 平均 BPP
      bpp_std:  float — BPP 标准差（per-batch）
      bpp_list: list[float] — 每个 batch 的 BPP
    """
    model.eval()
    bpp_list = []
    use_amp = amp_dtype is not None and device.type == 'cuda'

    for batch in loader:
        if isinstance(batch, (list, tuple)):
            x = batch[0]
        else:
            x = batch
        x = x.to(device)
        _, C, _, _ = x.shape

        with autocast(device_type="cuda", dtype=amp_dtype) if use_amp else nullcontext():
            out = model(x)

        ce_loss = out["ce_loss"]
        # BPP = CE_loss / ln(2) × C
        # Ref: Shannon 1948 [1] — 最优编码长度 = -log₂ p(x)
        bpp = compute_bpp(ce_loss, C)
        bpp_list.append(bpp.item())

    bpp_mean = float(np.mean(bpp_list))
    bpp_std = float(np.std(bpp_list))
    return bpp_mean, bpp_std, bpp_list


@torch.no_grad()
def evaluate_per_channel(model, loader, device, amp_dtype=None,
                         use_ycbcr=True, use_subpixel_ar=False):
    """
    Per-channel BPP 分解 — 分别计算 Y/Cb/Cr (或 R/G/B) 各通道的 BPP。

    注意: DMOL (loss_type="dmol") 模式下，跨通道条件化使 per-channel BPP
    分解数学上不可分（联合分布非独立乘积），此函数跳过并打印警告。
    CE 模式下正常工作。

    原理:
      模型预测整个 token 序列，将 logits 和 targets 按通道拆分后分别计算 CE loss。

      序列布局:
        - channel-first (use_subpixel_ar=False): [Y_all, Cb_all, Cr_all]
        - pixel-first  (use_subpixel_ar=True):  [Y0,Cb0,Cr0, Y1,Cb1,Cr1, ...]

      BPP_channel = CE_channel / ln(2)
      BPP_total = BPP_Y + BPP_Cb + BPP_Cr

    参数:
      model: IGPT 模型
      loader: DataLoader
      device: torch.device
      amp_dtype: AMP 精度
      use_ycbcr: bool — 模型是否使用 YCbCr（决定通道名称）
      use_subpixel_ar: bool — 是否使用子像素自回归（pixel-first 布局）

    返回:
      channel_bpps: dict[str, (float, float)] — {通道名: (bpp_mean, bpp_std)}
      total_bpp: (float, float) — 总 BPP (mean, std)
    """
    # DMOL 模式跳过 per-channel 分解
    if hasattr(model, 'loss_type') and model.loss_type == "dmol":
        print("  [WARN] DMOL 模式下跨通道条件化使 per-channel BPP 分解不可分，跳过。")
        return {}, (0.0, 0.0)
    model.eval()
    use_amp = amp_dtype is not None and device.type == 'cuda'
    channel_names = ["Y", "Cb", "Cr"] if use_ycbcr else ["R", "G", "B"]

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

        logits = out["logits"].float()  # (B, T, vocab_size), T = seq_len - 1

        # 重建 target tokens（与 model forward 一致）
        if use_ycbcr:
            tokens = rgb_to_ycbcr_int(x.clamp(0, 1))
        else:
            tokens = (x.clamp(0, 1) * 255).round().long()

        if use_subpixel_ar:
            # pixel-first: (B, C, H, W) → (B, H, W, C) → (B, H*W*C)
            tokens = tokens.permute(0, 2, 3, 1).reshape(B, -1)
        else:
            tokens = tokens.reshape(B, -1)  # channel-first: (B, seq_len)
        target_tokens = tokens[:, 1:]    # (B, T)

        # 按通道拆分: 根据序列布局提取每通道的 token
        for i, ch_name in enumerate(channel_names):
            if use_subpixel_ar:
                # pixel-first: 通道 i 的 token 在位置 i, i+C, i+2*C, ...
                # target_tokens 相对于 tokens 偏移了 1
                # 需要找到 target 中属于通道 i 的位置
                T = target_tokens.shape[1]
                # tokens 中通道 i 的位置: i, i+C, i+2C, ...
                # target_tokens[t] = tokens[t+1]，其通道 = (t+1) % C
                ch_mask = torch.arange(T, device=device) % C == ((i - 1) % C if i > 0 else (C - 1))
                # 更准确: target_tokens[t] 对应 tokens[t+1]，
                # tokens[t+1] 的通道 = (t+1) % C
                ch_mask = ((torch.arange(T, device=device) + 1) % C) == i
                if not ch_mask.any():
                    continue
                ch_logits = logits[:, ch_mask]      # (B, ch_len, vocab)
                ch_targets = target_tokens[:, ch_mask]  # (B, ch_len)
            else:
                # channel-first: 通道 i 占连续 pixels_per_channel 个 token
                start = i * pixels_per_channel
                end = (i + 1) * pixels_per_channel
                # 注意 target 偏移了 1，所以通道边界也偏移
                # target_tokens[t] 对应 logits[t] 的预测
                # 通道 i 的 target 范围: [start, end) 但受限于 T = seq_len-1
                ch_start = max(0, start - 1)   # logits index 对应 target index
                ch_end = min(end - 1, logits.shape[1])
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
    channel_bpps = {}
    for ch_name in channel_names:
        if channel_ce[ch_name]:
            ce_mean = float(np.mean(channel_ce[ch_name]))
            ce_std = float(np.std(channel_ce[ch_name]))
            # BPP = CE / ln(2)  (单通道，不乘 C)
            bpp_mean = ce_mean / math.log(2)
            bpp_std = ce_std / math.log(2)
            channel_bpps[ch_name] = (bpp_mean, bpp_std)

    # 总 BPP = 三通道 BPP 之和
    total_mean = sum(v[0] for v in channel_bpps.values())
    total_std = math.sqrt(sum(v[1]**2 for v in channel_bpps.values()))

    return channel_bpps, (total_mean, total_std)


@torch.no_grad()
def evaluate_position_bpp(model, loader, device, amp_dtype=None,
                           image_size=32, in_channels=3, use_ycbcr=True,
                           use_subpixel_ar=False):
    """
    Per-position BPP 热力图 — 计算每个像素位置的平均 BPP。

    注意: DMOL 模式下暂不支持 per-position 分解，跳过并返回空。

    对于 CIFAR-100 32×32×3，每个位置有一个 token，
    将 per-token CE loss 重新 reshape 回 (H, W, C) 并对 batch 取平均。
    最终输出 (H, W) 的 BPP 热力图（C 通道求和）。

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
      use_ycbcr: bool — 模型是否使用 YCbCr

    返回:
      heatmap: np.ndarray (H, W) — 每个像素位置的平均 BPP
      channel_heatmaps: dict[str, np.ndarray] — 每通道 (H, W) BPP 热力图
    """
    model.eval()
    # DMOL 模式下暂不支持 per-position 分解
    if hasattr(model, 'loss_type') and model.loss_type == "dmol":
        print("  [WARN] DMOL 模式下 per-position BPP 热力图暂不支持，跳过。")
        return np.zeros((image_size, image_size)), {}
    use_amp = amp_dtype is not None and device.type == 'cuda'
    C = in_channels
    seq_len = H * W * C
    channel_names = ["Y", "Cb", "Cr"] if use_ycbcr else ["R", "G", "B"]

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
            # fused path 不返回 logits，需要手动计算
            x_clamped = x.clamp(0, 1)
            if use_ycbcr:
                tokens = rgb_to_ycbcr_int(x_clamped)
            else:
                tokens = (x_clamped * 255).round().long()
            if use_subpixel_ar:
                tokens = tokens.permute(0, 2, 3, 1).reshape(B_cur, -1)
            else:
                tokens = tokens.reshape(B_cur, -1)
            target_tokens = tokens[:, 1:]
            # 手动计算 logits
            hidden = model.token_embed(tokens[:, :-1])
            if hasattr(model, 'channel_embed') and use_subpixel_ar:
                T_in = tokens.shape[1] - 1
                ch_idx = torch.arange(T_in, device=device) % C
                hidden = hidden + model.channel_embed(ch_idx).unsqueeze(0)
            if not model.use_rope:
                T = tokens.shape[1] - 1
                positions = torch.arange(T, device=device)
                hidden = hidden + model.pos_embed(positions)
            position_ids = None
            if use_subpixel_ar and model.use_rope:
                T_in = tokens.shape[1] - 1
                position_ids = torch.arange(T_in, device=device) // C
            for block in model.blocks:
                hidden = block(hidden, mask=None, position_ids=position_ids)
            if not model.use_post_norm:
                hidden = model.final_norm(hidden)
            logits = model.head(hidden).float()
        else:
            logits = logits.float()
            # 重建 target
            x_clamped = x.clamp(0, 1)
            if use_ycbcr:
                tokens = rgb_to_ycbcr_int(x_clamped)
            else:
                tokens = (x_clamped * 255).round().long()
            if use_subpixel_ar:
                tokens = tokens.permute(0, 2, 3, 1).reshape(B_cur, -1)
            else:
                tokens = tokens.reshape(B_cur, -1)
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

    # 转换为 BPP: CE / ln(2)
    position_bpp = position_ce_avg / math.log(2)

    # Reshape 回空间结构
    # 注意: 序列的第一个 token 没有 prediction（被偏移掉了），
    # 所以在 position_bpp 前面填 0，使其长度 = seq_len
    position_bpp_full = np.zeros(seq_len)
    position_bpp_full[1:] = position_bpp

    if use_subpixel_ar:
        # pixel-first: (seq_len,) → (H, W, C) → (C, H, W)
        position_bpp_hwc = position_bpp_full.reshape(H, W, C)
        position_bpp_chw = position_bpp_hwc.transpose(2, 0, 1)  # (C, H, W)
    else:
        # channel-first: (seq_len,) → (C, H, W)
        position_bpp_chw = position_bpp_full.reshape(C, H, W)

    # 总 BPP 热力图: 对 C 通道求和 → (H, W)
    heatmap = position_bpp_chw.sum(axis=0)

    # 每通道热力图
    channel_heatmaps = {}
    for i, ch_name in enumerate(channel_names):
        channel_heatmaps[ch_name] = position_bpp_chw[i]

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


def compute_traditional_bpp(dataset, method="png"):
    """
    计算传统无损压缩方法在数据集上的 BPP。

    将每张图片编码为内存中的 PNG/WebP 字节流，
    BPP = 压缩后字节 × 8 / 像素数。

    参数:
      dataset: torchvision dataset（返回 (tensor, label)）
      method: "png" 或 "webp"

    返回:
      bpp_mean: float
      bpp_std:  float
    """
    from PIL import Image

    format_map = {"png": "PNG", "webp": "WEBP"}
    fmt = format_map.get(method, "PNG")
    save_kwargs = {"lossless": True} if method == "webp" else {}

    bpp_list = []
    for i in range(len(dataset)):
        img_tensor, _ = dataset[i]
        # tensor (C, H, W) [0,1] → PIL Image
        img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        img = Image.fromarray(img_np)

        # 编码到内存
        buf = io.BytesIO()
        img.save(buf, format=fmt, **save_kwargs)
        compressed_bytes = buf.tell()

        # BPP = 压缩字节 × 8 / 总像素数
        _, H, W = img_tensor.shape
        total_pixels = H * W
        bpp = (compressed_bytes * 8) / total_pixels
        bpp_list.append(bpp)

    return float(np.mean(bpp_list)), float(np.std(bpp_list))


def print_results_table(dataset_name, model_bpp, model_std,
                         traditional_results=None,
                         channel_bpps=None):
    """
    打印 Markdown 格式的结果表格。

    参数:
      dataset_name: str — "cifar10" / "cifar100"
      model_bpp: float — 本文模型 BPP
      model_std: float — BPP 标准差
      traditional_results: dict[str, (float, float)] — 传统方法 {name: (bpp, std)}
      channel_bpps: dict[str, (float, float)] — per-channel {name: (bpp, std)}
    """
    print(f"\n{'='*60}")
    print(f"  评测结果 — {dataset_name.upper()}")
    print(f"{'='*60}\n")

    print("| 方法 | BPP ↓ | 备注 |")
    print("|------|-------|------|")

    # 传统方法
    if traditional_results:
        for name, (bpp, std) in traditional_results.items():
            print(f"| {name} | {bpp:.2f} ± {std:.2f} | 无损压缩 |")

    # 学术 baseline
    for name, bpp in ACADEMIC_BASELINES.get(dataset_name, []):
        print(f"| {name} | {bpp:.2f} | 论文报告值 |")

    # 本文
    print(f"| **iGPT (Ours)** | **{model_bpp:.4f} ± {model_std:.4f}** | **本文** |")

    # Per-channel BPP
    if channel_bpps:
        print()
        print("| 通道 | BPP ↓ |")
        print("|------|-------|")
        for ch_name, (bpp, std) in channel_bpps.items():
            print(f"| {ch_name} | {bpp:.4f} ± {std:.4f} |")
        total_bpp = sum(v[0] for v in channel_bpps.values())
        print(f"| **Total** | **{total_bpp:.4f}** |")
    print()


def print_ablation_table(results):
    """
    打印消融实验 Markdown 表格。

    参数:
      results: list[(str, str, float, float, float)] —
               [(exp_id, description, bpp, std, delta_bpp), ...]
    """
    print(f"\n{'='*80}")
    print(f"  消融实验结果")
    print(f"{'='*80}\n")

    print("| 实验 | 配置 | BPP ↓ | ΔBPP | 备注 |")
    print("|------|------|-------|------|------|")

    baseline_bpp = None
    for exp_id, desc, bpp, std, delta in results:
        if baseline_bpp is None:
            baseline_bpp = bpp
            delta_str = "—"
            note = "baseline"
        else:
            delta_str = f"+{delta:.4f}" if delta > 0 else f"{delta:.4f}"
            note = "↑ 移除有损" if delta > 0 else "↓ 移除有益" if delta < 0 else "≈ 无影响"
        print(f"| {exp_id} | {desc} | {bpp:.4f} ± {std:.4f} | {delta_str} | {note} |")
    print()


def _load_dataset(config):
    """加载测试数据集"""
    from torchvision import transforms
    from torchvision.datasets import CIFAR10, CIFAR100

    transform = transforms.ToTensor()
    dataset_name = config["data"].get("dataset", "cifar100")
    DatasetClass = CIFAR10 if dataset_name == "cifar10" else CIFAR100
    test_dataset = DatasetClass(root=config["data"]["valid"], train=False,
                                download=False, transform=transform)
    return test_dataset, dataset_name


def _load_checkpoint(model, ckpt_path, device):
    """加载 checkpoint（支持完整 ckpt 和纯 state_dict）"""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
        epoch = ckpt.get('epoch', '?')
        return epoch
    else:
        model.load_state_dict(ckpt)
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
    model = _build_model_from_config(mcfg, device)
    epoch = _load_checkpoint(model, args.checkpoint, device)
    print(f"Loaded checkpoint: epoch {epoch}")

    amp_dtype, amp_dtype_str = _get_amp_dtype(config)

    # 基本 BPP 评测
    print(f"\n评测中... (AMP: {amp_dtype_str})")
    bpp_mean, bpp_std, _ = evaluate_model(model, test_loader, device,
                                            amp_dtype=amp_dtype)
    print(f"iGPT BPP: {bpp_mean:.4f} ± {bpp_std:.4f}")

    # Per-channel BPP（可选）
    channel_bpps = None
    if args.per_channel:
        print("\n计算 per-channel BPP...")
        use_ycbcr = mcfg.get("use_ycbcr", True)
        use_subpixel_ar = mcfg.get("use_subpixel_ar", False)
        channel_bpps, (total_bpp, total_std) = evaluate_per_channel(
            model, test_loader, device, amp_dtype=amp_dtype,
            use_ycbcr=use_ycbcr, use_subpixel_ar=use_subpixel_ar
        )
        for ch_name, (ch_bpp, ch_std) in channel_bpps.items():
            print(f"  {ch_name}: {ch_bpp:.4f} ± {ch_std:.4f}")
        print(f"  Total: {total_bpp:.4f} ± {total_std:.4f}")

    # 传统方法（可选）
    traditional_results = None
    if args.traditional:
        print("\n计算传统方法 BPP...")
        traditional_results = {}

        png_bpp, png_std = compute_traditional_bpp(test_dataset, method="png")
        traditional_results["PNG (lossless)"] = (png_bpp, png_std)
        print(f"  PNG:  {png_bpp:.2f} ± {png_std:.2f}")

        try:
            webp_bpp, webp_std = compute_traditional_bpp(test_dataset, method="webp")
            traditional_results["WebP (lossless)"] = (webp_bpp, webp_std)
            print(f"  WebP: {webp_bpp:.2f} ± {webp_std:.2f}")
        except Exception as e:
            print(f"  WebP: 跳过 ({e})")

    print_results_table(dataset_name, bpp_mean, bpp_std,
                         traditional_results, channel_bpps)

    # Per-position BPP 热力图（可选）
    if args.heatmap:
        print("\n生成 per-position BPP 热力图...")
        use_ycbcr = mcfg.get("use_ycbcr", True)
        use_subpixel_ar = mcfg.get("use_subpixel_ar", False)
        image_size = mcfg.get("image_size", 32)
        in_channels = mcfg.get("in_channels", 3)
        heatmap, channel_heatmaps = evaluate_position_bpp(
            model, test_loader, device, amp_dtype=amp_dtype,
            image_size=image_size, in_channels=in_channels,
            use_ycbcr=use_ycbcr, use_subpixel_ar=use_subpixel_ar
        )
        # 保存到 checkpoint 同目录
        ckpt_dir = os.path.dirname(args.checkpoint) or "."
        save_heatmap(heatmap, os.path.join(ckpt_dir, "bpp_heatmap_total.png"),
                     title=f"BPP Heatmap — {dataset_name}")
        for ch_name, ch_hm in channel_heatmaps.items():
            save_heatmap(ch_hm, os.path.join(ckpt_dir, f"bpp_heatmap_{ch_name}.png"),
                         title=f"BPP Heatmap — {ch_name}")
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
    model = _build_model_from_config(mcfg, device)
    _load_checkpoint(model, best_path, device)
    bpp_best, std_best, _ = evaluate_model(model, test_loader, device,
                                            amp_dtype=amp_dtype)
    results.append(("best.pth", bpp_best, std_best))
    print(f"best.pth  BPP: {bpp_best:.4f} ± {std_best:.4f}")

    # 评测 swa
    model = _build_model_from_config(mcfg, device)
    _load_checkpoint(model, swa_path, device)
    bpp_swa, std_swa, _ = evaluate_model(model, test_loader, device,
                                           amp_dtype=amp_dtype)
    results.append(("swa.pth", bpp_swa, std_swa))
    print(f"swa.pth   BPP: {bpp_swa:.4f} ± {std_swa:.4f}")

    delta = bpp_swa - bpp_best
    print(f"\nΔBPP (SWA - best): {delta:+.4f}")

    # 打印对比表格
    print(f"\n| Checkpoint | BPP ↓ | ΔBPP |")
    print(f"|------------|-------|------|")
    print(f"| best.pth | {bpp_best:.4f} ± {std_best:.4f} | — |")
    delta_str = f"{delta:+.4f}"
    note = "SWA 更优" if delta < 0 else "best 更优"
    print(f"| swa.pth  | {bpp_swa:.4f} ± {std_swa:.4f} | {delta_str} ({note}) |")
    print()


def cmd_ablation(args, config, device):
    """
    消融批量评测 — 扫描实验目录下的 checkpoint，生成消融对比表。

    目录结构约定:
      {ablation_dir}/
        E0_full/checkpoints/best.pth
        E1_no_ycbcr/checkpoints/best.pth
        E2_no_rope/checkpoints/best.pth
        ...

    或者: 手动指定 checkpoint 列表（通过 --ablation_checkpoints）。
    """
    test_dataset, dataset_name = _load_dataset(config)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=2, pin_memory=True)

    mcfg = config["model"]
    amp_dtype, amp_dtype_str = _get_amp_dtype(config)

    results = []
    baseline_bpp = None

    # 方式 1: 扫描目录
    ablation_dir = args.ablation_dir
    if ablation_dir:
        # 按 E0, E1, ... 排序扫描
        exp_dirs = sorted(os.listdir(ablation_dir))
        for exp_name in exp_dirs:
            exp_path = os.path.join(ablation_dir, exp_name)
            ckpt_path = os.path.join(exp_path, "checkpoints", "best.pth")
            if not os.path.isfile(ckpt_path):
                continue

            # 从目录名推断实验编号 (E0_full → E0)
            exp_id = exp_name.split("_")[0].upper()
            ablation_overrides = ABLATION_CONFIGS.get(exp_id, {})

            # 构建模型配置（应用消融覆盖）
            mcfg_ablation = dict(mcfg)
            mcfg_ablation.update(ablation_overrides)

            desc = exp_name
            if ablation_overrides:
                removed = [f"{k}=False" for k, v in ablation_overrides.items()
                           if v is False]
                desc = ", ".join(removed) if removed else exp_name

            print(f"\n评测 {exp_id} ({desc})...")
            model = _build_model_from_config(mcfg_ablation, device)
            _load_checkpoint(model, ckpt_path, device)
            bpp, std, _ = evaluate_model(model, test_loader, device,
                                          amp_dtype=amp_dtype)

            if baseline_bpp is None:
                baseline_bpp = bpp
            delta = bpp - baseline_bpp

            results.append((exp_id, desc, bpp, std, delta))
            print(f"  BPP: {bpp:.4f} ± {std:.4f}, ΔBPP: {delta:+.4f}")

    # 方式 2: 手动指定 checkpoint 列表
    elif args.ablation_checkpoints:
        ckpts = args.ablation_checkpoints.split(",")
        for i, ckpt_path in enumerate(ckpts):
            ckpt_path = ckpt_path.strip()
            if not os.path.isfile(ckpt_path):
                print(f"跳过不存在的 checkpoint: {ckpt_path}")
                continue

            exp_id = f"E{i}"
            desc = os.path.basename(os.path.dirname(os.path.dirname(ckpt_path)))

            print(f"\n评测 {exp_id} ({desc})...")
            model = _build_model_from_config(mcfg, device)
            _load_checkpoint(model, ckpt_path, device)
            bpp, std, _ = evaluate_model(model, test_loader, device,
                                          amp_dtype=amp_dtype)

            if baseline_bpp is None:
                baseline_bpp = bpp
            delta = bpp - baseline_bpp

            results.append((exp_id, desc, bpp, std, delta))
            print(f"  BPP: {bpp:.4f} ± {std:.4f}, ΔBPP: {delta:+.4f}")

    if results:
        print_ablation_table(results)
    else:
        print("\n未找到任何 checkpoint，请检查 --ablation_dir 路径。")
        print("目录结构约定:")
        print("  {ablation_dir}/E0_full/checkpoints/best.pth")
        print("  {ablation_dir}/E1_no_ycbcr/checkpoints/best.pth")
        print("  ...")


def main():
    parser = argparse.ArgumentParser(
        description="iGPT 无损压缩评测",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 单模型
  python scripts/evaluate.py --config configs/igpt_cifar10_baseline.yaml \\
      --checkpoint best.pth

  # 消融批量
  python scripts/evaluate.py --config configs/igpt_cifar10_baseline.yaml \\
      --ablation_dir experiments/

  # Per-channel + 传统方法
  python scripts/evaluate.py --config configs/igpt_cifar10_baseline.yaml \\
      --checkpoint best.pth --per_channel --traditional

  # SWA 对比
  python scripts/evaluate.py --config configs/igpt_cifar10_baseline.yaml \\
      --checkpoint experiments/exp/checkpoints/best.pth --swa
        """
    )
    parser.add_argument('--config', type=str, required=True,
                        help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='模型 checkpoint 路径（best.pth）')
    parser.add_argument('--traditional', action='store_true',
                        help='同时计算 PNG/WebP 传统方法 BPP')
    parser.add_argument('--per_channel', action='store_true',
                        help='计算 per-channel BPP 分解（Y/Cb/Cr）')
    parser.add_argument('--heatmap', action='store_true',
                        help='生成 per-position BPP 热力图（保存为 PNG）')
    parser.add_argument('--swa', action='store_true',
                        help='同时评测 SWA checkpoint（swa.pth vs best.pth）')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='评测 batch size')
    # 消融批量评测
    parser.add_argument('--ablation_dir', type=str, default=None,
                        help='消融实验目录（扫描子目录下的 checkpoints/best.pth）')
    parser.add_argument('--ablation_checkpoints', type=str, default=None,
                        help='消融 checkpoint 列表（逗号分隔路径）')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 根据模式分发
    if args.ablation_dir or args.ablation_checkpoints:
        cmd_ablation(args, config, device)
    elif args.swa and args.checkpoint:
        cmd_swa(args, config, device)
    elif args.checkpoint:
        cmd_single(args, config, device)
    else:
        parser.error("请指定 --checkpoint 或 --ablation_dir")


if __name__ == '__main__':
    main()
