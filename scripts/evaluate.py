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
    python scripts/evaluate.py --config configs/ccigpt_cifar10_s_rgb_ronly_v2.yaml \
        --checkpoint experiments/ccigpt_cifar10_s_rgb_ronly_v2/checkpoints/best.pth --tta_hflip

    # SWA vs best 对比 (v2 配置启用了 SWA last 31 ckpts, start ep170)
    python scripts/evaluate.py --config configs/ccigpt_cifar10_s_rgb_ronly_v2.yaml \
        --checkpoint experiments/ccigpt_cifar10_s_rgb_ronly_v2/checkpoints/best.pth --swa

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
import json
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
def evaluate_model(model, loader, device, amp_dtype=None, tta_hflip: bool = False,
                   collect_per_image: bool = False):
    """
    评估模型在数据集上的 bits/dim (bpd)。

    返回:
      bpd_mean: float — 平均 bpd
      bpd_std:  float — bpd 标准差（batch-level 加权，与 train.py:validate() 同口径）
      bpd_list: list[float] — 每个 batch 的 bpd（保留供 caller 自定义聚合）
      extras:   dict — 可选的额外字段（CC-iGPT 时含 ce_coarse / ce_fine / ctx_alpha；
                collect_per_image=True 时含 per_image_bpd）

    TTA (Test-Time Augmentation):
      tta_hflip=True 时对每张图同时跑 x 与 hflip(x) 两次 forward，取 bpd 均值。
      hflip(x) 的 -log p 仍是 H(X) 的合法上界（hflip 在 RGB-bit-exact 域是确定函数），
      平均能降低估计方差。Ref: Sparse Transformer (Child 2019, §4.2) 采用类似 ensemble.
    """
    model.eval()
    bpd_per_batch = []
    bpd_weighted_sum = 0.0
    bpd_sq_weighted_sum = 0.0
    per_image_bpd = []
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
            if collect_per_image:
                bpd_img = _per_image_bpd(model, x, out)
                if tta_hflip:
                    bpd_img_flip = _per_image_bpd(
                        model, torch.flip(x, dims=[-1]), out_flip
                    )
                    bpd_img = (bpd_img + bpd_img_flip) * 0.5

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
        if collect_per_image:
            per_image_bpd.extend(bpd_img.detach().cpu().tolist())

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
    if collect_per_image:
        extras["per_image_bpd"] = per_image_bpd
    return bpd_mean, bpd_std, bpd_per_batch, extras


def _tokenize_targets(x: torch.Tensor) -> torch.Tensor:
    """与 IGPT._tokenize 同口径：RGB float [0,1] → pixel-first long token (B, T-1)。

    用于 ensemble 路径在外部独立 tokenize 一次（每个 model 拿到的 x 一样，token
    序列必然相同），避免对 K 个 model 重复 .tokenize。返回的 target 已做 NTP 切片
    `tokens[:, 1:]`，与 IGPT.forward 内部一致。
    """
    xt = x.clamp(0, 1)
    xt = (xt * 255).round().long()
    tokens = xt.permute(0, 2, 3, 1).reshape(x.size(0), -1)
    return tokens[:, 1:]


def _ensemble_log_probs(log_probs_stack: list[torch.Tensor]) -> torch.Tensor:
    """Log-probs for p_ens(y|x) = mean_k p_k(y|x), computed stably."""
    if not log_probs_stack:
        raise ValueError("log_probs_stack must contain at least one tensor")
    stacked = torch.stack(log_probs_stack, dim=0)
    return torch.logsumexp(stacked, dim=0) - math.log(stacked.size(0))


def _ce_per_image_from_logits(logits: torch.Tensor,
                              targets: torch.Tensor) -> torch.Tensor:
    """Return mean NTP cross-entropy per image, matching IGPT.forward."""
    B = targets.size(0)
    V = logits.size(-1)
    ce_tok = F.cross_entropy(
        logits.float().reshape(-1, V),
        targets.reshape(-1),
        reduction="none",
    )
    return ce_tok.view(B, -1).mean(dim=1)


def _ccigpt_coarse_forward(model, x: torch.Tensor):
    """Recompute CC-iGPT coarse branch for per-image bpd diagnostics."""
    raw_model = model.module if hasattr(model, "module") else model
    x_fp32 = x.clamp(0, 1).to(torch.float32)
    x_c_full = F.adaptive_avg_pool2d(x_fp32, raw_model.coarse_size)
    if raw_model.coarse.in_channels < raw_model.in_channels:
        x_c = x_c_full[:, :raw_model.coarse.in_channels]
    else:
        x_c = x_c_full
    return raw_model.coarse(x_c, z_loss_weight=0.0), x_c


def _per_image_bpd(model, x: torch.Tensor, out: dict) -> torch.Tensor:
    """Compute per-image bpd from logits without changing default metrics.

    For vanilla iGPT this is CE_i / ln2. For CC-iGPT this mirrors the existing
    scalar formula `(CE_c*N_c + CE_f*N_f) / ln2 / N_f`, but with CE_c/CE_f
    measured per image.
    """
    raw_model = model.module if hasattr(model, "module") else model
    target_f = _tokenize_targets(x).to(out["logits"].device)
    ce_f = _ce_per_image_from_logits(out["logits"], target_f)

    if "ce_loss_coarse" not in out or out["ce_loss_coarse"] is None:
        return ce_f / math.log(2.0)

    out_c, x_c = _ccigpt_coarse_forward(raw_model, x)
    target_c = _tokenize_targets(x_c).to(out_c["logits"].device)
    ce_c = _ce_per_image_from_logits(out_c["logits"], target_c)
    N_c, N_f = raw_model.coarse.seq_len, raw_model.fine.seq_len
    return (ce_c * N_c + ce_f * N_f) / math.log(2.0) / N_f


def _summarize_per_image_bpd(values, bootstrap_samples: int = 1000,
                             seed: int = 0) -> dict:
    """Summarize per-image bpd and optionally estimate a bootstrap CI."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("per-image bpd 列表为空")

    ddof = 1 if arr.size > 1 else 0
    summary = {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=ddof)),
        "stderr": float(arr.std(ddof=ddof) / math.sqrt(arr.size)),
    }
    if arr.size > 1 and bootstrap_samples > 0:
        rng = np.random.default_rng(seed)
        means = np.empty(int(bootstrap_samples), dtype=np.float64)
        for i in range(int(bootstrap_samples)):
            idx = rng.integers(0, arr.size, size=arr.size)
            means[i] = arr[idx].mean()
        lo, hi = np.percentile(means, [2.5, 97.5])
        summary["ci95_bootstrap"] = [float(lo), float(hi)]
        summary["bootstrap_samples"] = int(bootstrap_samples)
    return summary


def _write_per_image_json(path: str, values, summary: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "summary": summary,
        "per_image_bpd": [round(float(v), 6) for v in values],
    }
    with open(path, "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[per_image_json] wrote {len(values)} rows → {path}")


@torch.no_grad()
def evaluate_ensemble(models, loader, device, amp_dtype=None, tta_hflip: bool = False):
    """多 ckpt probability-mixture ensemble 评测（log-prob 域稳定实现）。

    每 batch：K 个 model 各 forward 拿 fine logits → log-softmax → K 档逐元素
    logsumexp - log(K) → gather target → fine NLL。这对应
    p_ens(y|x)=mean_k p_k(y|x)，不是 logit averaging，也不是各模型 NLL 均值。
    coarse 走每档 ce_coarse 数值平均（每档 coarse 架构相同、训自同一轨迹的
    不同平滑，数值平均近似 coarse-ensemble bpd）。
    bpd_total = (CE_c_avg·N_c + CE_f_ens·N_f) / ln2 / N_f，与 cc_igpt.forward 同口径。

    与 --tta_hflip 正交：TTA 路径在 evaluate_model 上是"x 与 hflip(x) 各跑一次取
    batch-mean CE 均值"；ensemble 路径同口径——hflip(x) 也跑 K 档 ensemble 取
    fine NLL，再与原序 NLL 取均值。
    """
    for m in models:
        m.eval()

    bpd_per_batch = []
    bpd_weighted_sum = 0.0
    bpd_sq_weighted_sum = 0.0
    n_total = 0
    use_amp = amp_dtype is not None and device.type == 'cuda'
    K = len(models)

    is_ccigpt = False
    ce_c_sum = ce_f_sum = alpha_sum = 0.0

    def _ensemble_fine_nll(x_in, target):
        """K 档 forward → probability mixture → gather target → mean NLL。

        副作用：把每档 ce_coarse / ctx_alpha 通过 closure 写进 ce_c_local / alpha_local
        以便外层做 batch-mean 累加。
        """
        log_probs_stack = []
        ce_c_local = []
        alpha_local = []
        amp_ctx = autocast(device_type="cuda", dtype=amp_dtype) if use_amp else nullcontext()
        for m in models:
            with amp_ctx:
                out = m(x_in)
            assert out.get("logits") is not None, (
                "ensemble 评测要求 forward 返回 logits（softmax 路径）"
            )
            log_probs_stack.append(F.log_softmax(out["logits"].float(), dim=-1))
            if "ce_loss_coarse" in out and out["ce_loss_coarse"] is not None:
                ce_c_local.append(out["ce_loss_coarse"].item())
                if "ctx_alpha" in out and out["ctx_alpha"] is not None:
                    alpha_local.append(out["ctx_alpha"].item())
        log_probs_ens = _ensemble_log_probs(log_probs_stack)   # (B, T-1, V)
        nll_per_tok = -log_probs_ens.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        return nll_per_tok.mean(), ce_c_local, alpha_local

    for batch in loader:
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x = x.to(device)
        B = x.size(0)

        target = _tokenize_targets(x)
        ce_fine, ce_c_x, alpha_x = _ensemble_fine_nll(x, target)

        if tta_hflip:
            x_flip = torch.flip(x, dims=[-1])
            target_flip = _tokenize_targets(x_flip)
            ce_fine_flip, ce_c_flip, _ = _ensemble_fine_nll(x_flip, target_flip)
            ce_fine = (ce_fine + ce_fine_flip) * 0.5
            if ce_c_x and ce_c_flip:
                assert len(ce_c_x) == len(ce_c_flip), (
                    f"TTA ensemble: coarse CE 列表长度不一致 "
                    f"(原序 K={len(ce_c_x)} vs hflip K={len(ce_c_flip)})"
                )
                ce_c_x = [(a + b) * 0.5 for a, b in zip(ce_c_x, ce_c_flip)]

        ce_fine_val = ce_fine.item()

        if ce_c_x:
            is_ccigpt = True
            ce_coarse_val = sum(ce_c_x) / len(ce_c_x)
            N_c = models[0].coarse.seq_len
            N_f = models[0].fine.seq_len
            bpd_val = (ce_coarse_val * N_c + ce_fine_val * N_f) / math.log(2.0) / N_f
            ce_c_sum += ce_coarse_val * B
            ce_f_sum += ce_fine_val * B
            if alpha_x:
                alpha_sum += (sum(alpha_x) / len(alpha_x)) * B
        else:
            bpd_val = ce_fine_val / math.log(2.0)

        bpd_per_batch.append(bpd_val)
        bpd_weighted_sum += bpd_val * B
        bpd_sq_weighted_sum += (bpd_val ** 2) * B
        n_total += B

    bpd_mean = bpd_weighted_sum / n_total
    bpd_var = max(bpd_sq_weighted_sum / n_total - bpd_mean ** 2, 0.0)
    bpd_std = float(math.sqrt(bpd_var))

    extras = {"K": K}
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

    口径说明: 文献 (Hoogeboom et al., NeurIPS 2019, Integer Discrete Flows)
    报告 CIFAR-10 PNG≈5.87、WebP (lossless)≈4.61 bits/dim（除以 H·W·C），
    与本仓库主指标一致。本脚本是用 PIL 实测；若除以 H·W 则得到的是真 bits-per-pixel（bpd × 3），数值会高 3 倍。

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
    """加载测试数据集。
    """
    from torchvision import transforms
    from torchvision.datasets import CIFAR10, CIFAR100

    dataset_name = config["data"].get("dataset", "cifar100")

    if dataset_name == "imagenet64_npy":
        from src.mdlic.data.imagenet64_npy import ImageNet64Npy
        test_dataset = ImageNet64Npy(root=config["data"]["valid"], split="val")
        return test_dataset, dataset_name
    if dataset_name not in ("cifar10", "cifar100"):
        raise ValueError(
            f"未知 dataset: '{dataset_name}'，支持 cifar10/cifar100/imagenet64_npy"
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
    collect_per_image = bool(args.per_image_stats or args.per_image_json)
    bpd_mean, bpd_std, _, extras = evaluate_model(
        model, test_loader, device,
        amp_dtype=amp_dtype,
        tta_hflip=args.tta_hflip,
        collect_per_image=collect_per_image,
    )
    print(f"{model_type.upper()} bits/dim: {bpd_mean:.4f} ± {bpd_std:.4f}")
    if collect_per_image:
        per_image = extras["per_image_bpd"]
        summary = _summarize_per_image_bpd(
            per_image,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.bootstrap_seed,
        )
        ci = summary.get("ci95_bootstrap")
        ci_str = f", 95% bootstrap CI [{ci[0]:.4f}, {ci[1]:.4f}]" if ci else ""
        print(
            "  per-image: "
            f"mean={summary['mean']:.4f}, std={summary['std']:.4f}, "
            f"stderr={summary['stderr']:.5f}{ci_str}"
        )
        if args.per_image_json:
            _write_per_image_json(args.per_image_json, per_image, summary)
    if extras:
        # CC-iGPT 多输出 CE_c / CE_f / α，便于诊断 fine 弱 vs coarse overhead 过大
        if "ce_coarse" in extras:
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

    # 评测 best
    model = _build_from_config(mcfg, device)
    _load_checkpoint(model, best_path, device)
    bpd_best, std_best, _, _ = evaluate_model(model, test_loader, device,
                                                amp_dtype=amp_dtype,
                                                tta_hflip=args.tta_hflip)
    print(f"best.pth  bits/dim: {bpd_best:.4f} ± {std_best:.4f}")

    # 评测 swa
    model = _build_from_config(mcfg, device)
    _load_checkpoint(model, swa_path, device)
    bpd_swa, std_swa, _, _ = evaluate_model(model, test_loader, device,
                                               amp_dtype=amp_dtype,
                                               tta_hflip=args.tta_hflip)
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


def cmd_ensemble(args, config, device):
    """多 ckpt probability-mixture ensemble 评测（best/swa/ema 等同源平滑组合）。"""
    test_dataset, dataset_name = _load_dataset(config)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=2, pin_memory=True)
    print(f"Dataset: {dataset_name} test ({len(test_dataset)} images)")

    ckpt_paths = [p.strip() for p in args.ensemble.split(',') if p.strip()]
    assert len(ckpt_paths) >= 2, (
        f"--ensemble 至少需要 2 个 ckpt，got {len(ckpt_paths)}"
    )

    mcfg = config["model"]
    model_type = mcfg.get("type", "igpt")
    amp_dtype, amp_dtype_str = _get_amp_dtype(config)

    print(f"Loading {len(ckpt_paths)} ckpts for ensemble...")
    models = []
    for path in ckpt_paths:
        m = _build_from_config(mcfg, device)
        epoch = _load_checkpoint(m, path, device)
        print(f"  {os.path.basename(path)} (epoch {epoch})")
        models.append(m)

    print(f"\n评测中... (AMP: {amp_dtype_str}, TTA hflip: {args.tta_hflip}, K={len(models)})")
    bpd_mean, bpd_std, _, extras = evaluate_ensemble(
        models, test_loader, device,
        amp_dtype=amp_dtype, tta_hflip=args.tta_hflip,
    )
    print(f"{model_type.upper()} ensemble bits/dim: {bpd_mean:.4f} ± {bpd_std:.4f}")
    if "ce_coarse" in extras:
        ce_c, ce_f = extras["ce_coarse"], extras["ce_fine"]
        N_c = models[0].coarse.seq_len
        N_f = models[0].fine.seq_len
        bpd_c_share = ce_c * N_c / math.log(2.0) / N_f
        bpd_f_share = ce_f * N_f / math.log(2.0) / N_f
        print(f"  CE_coarse = {ce_c:.4f}  → bpd_share = {bpd_c_share:.4f} ({100*bpd_c_share/bpd_mean:.1f}%)")
        print(f"  CE_fine   = {ce_f:.4f}  → bpd_share = {bpd_f_share:.4f} ({100*bpd_f_share/bpd_mean:.1f}%)")
        print(f"  ctx_alpha = {extras['ctx_alpha']:.4f}  (K-档平均)")

    print_results_table(dataset_name, bpd_mean, bpd_std, None,
                        model_label=f"{model_type.upper()} ensemble (K={len(models)}, Ours)")


def main():
    parser = argparse.ArgumentParser(
        description="iGPT 无损压缩评测",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 单模型 (主表数字 best + TTA hflip)
  python scripts/evaluate.py --config configs/ccigpt_cifar10_s_rgb_ronly_v2.yaml \\
      --checkpoint experiments/ccigpt_cifar10_s_rgb_ronly_v2/checkpoints/best.pth --tta_hflip

  # SWA 对比 (v2 配置启用了 SWA last 31 ckpts, start ep170)
  python scripts/evaluate.py --config configs/ccigpt_cifar10_s_rgb_ronly_v2.yaml \\
      --checkpoint experiments/ccigpt_cifar10_s_rgb_ronly_v2/checkpoints/best.pth --swa

  # 多 ckpt probability-mixture ensemble (best/swa/ema 三档同源平滑组合)
  python scripts/evaluate.py --config configs/ccigpt_cifar10_s_rgb_ronly_v2.yaml \\
      --ensemble experiments/ccigpt_cifar10_s_rgb_ronly_v2/checkpoints/best.pth,\\
experiments/ccigpt_cifar10_s_rgb_ronly_v2/checkpoints/swa.pth,\\
experiments/ccigpt_cifar10_s_rgb_ronly_v2/checkpoints/ema.pth \\
      --tta_hflip
        """
    )
    parser.add_argument('--config', type=str, required=True,
                        help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='模型 checkpoint 路径（best.pth）')
    parser.add_argument('--ensemble', type=str, default=None,
                        help='多 ckpt probability-mixture ensemble，逗号分隔 ckpt 路径列表 '
                             '（与 --checkpoint 互斥；至少 2 个 ckpt）')
    parser.add_argument('--traditional', action='store_true',
                        help='同时计算 PNG/WebP 传统方法 bits/dim')
    parser.add_argument('--swa', action='store_true',
                        help='同时评测 SWA checkpoint（swa.pth vs best.pth）')
    parser.add_argument('--tta_hflip', action='store_true',
                        help='Test-Time Augmentation：对每张图同时跑 x 与 hflip(x)，bpd 取均值')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='评测 batch size')
    parser.add_argument('--per_image_stats', action='store_true',
                        help='单 checkpoint 模式下额外计算 per-image bpd/std/stderr/CI；'
                             '默认关闭以保持评测速度和旧输出口径。')
    parser.add_argument('--per_image_json', type=str, default=None,
                        help='导出 per-image bpd JSON；会隐式启用 --per_image_stats。')
    parser.add_argument('--bootstrap_samples', type=int, default=1000,
                        help='per-image 均值 bootstrap CI 抽样次数；设 0 可关闭 CI。')
    parser.add_argument('--bootstrap_seed', type=int, default=0,
                        help='per-image bootstrap 随机种子。')
    args = parser.parse_args()

    if args.ensemble and args.checkpoint:
        parser.error("--ensemble 与 --checkpoint 互斥；ensemble 路径在 --ensemble 内逗号分隔")
    if args.ensemble and args.swa:
        parser.error("--ensemble 与 --swa 互斥；ensemble 已包含多档 ckpt 评测")
    if (args.ensemble or args.swa) and (args.per_image_stats or args.per_image_json):
        parser.error("--per_image_stats/--per_image_json 当前仅支持单 --checkpoint 模式")

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 根据模式分发
    if args.ensemble:
        cmd_ensemble(args, config, device)
    elif args.swa and args.checkpoint:
        cmd_swa(args, config, device)
    elif args.checkpoint:
        cmd_single(args, config, device)
    else:
        parser.error("请指定 --checkpoint 或 --ensemble")


if __name__ == '__main__':
    main()
