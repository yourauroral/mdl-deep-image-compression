"""
公共工具函数 — 被 train.py、evaluate.py、dryrun_forward.py 等多处复用。
"""

import math
import random
import torch
import numpy as np


def seed_everything(seed: int = 42, deterministic: bool = False):
    """
    统一设置所有随机种子，确保实验可复现。

    参考:
      [1] PyTorch Reproducibility 文档 — "Controlling sources of randomness"

    参数:
      seed: 随机种子
      deterministic: True 时启用 cudnn 确定性（性能下降，但完全可复现）
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def strip_module_prefix(sd: dict) -> dict:
    """剥离 DDP `module.` 与 torch.compile `_orig_mod.` 前缀（任意顺序、可嵌套）。

    DDP 包装给 state_dict key 加 `module.`，torch.compile 再加 `_orig_mod.`；
    两者可能嵌套出现 `module._orig_mod.xxx`。新版 train.py 已统一保存裸 state_dict
    （无前缀），但历史 ckpt 仍可能带前缀，本函数让 evaluate / linear_probe / demo
    等加载侧透明兼容。
    """
    prefixes = ('module.', '_orig_mod.')

    def _strip_one(k: str) -> str:
        changed = True
        while changed:
            changed = False
            for p in prefixes:
                if k.startswith(p):
                    k = k[len(p):]
                    changed = True
        return k

    if any(k.startswith(prefixes) for k in sd.keys()):
        return {_strip_one(k): v for k, v in sd.items()}
    return sd


# 已知不再持久化的 buffer 后缀，旧 ckpt 里可能仍保留。
# - inv_freq: RoPE 频率表，改为 persistent=False 后由模型运行时重算，
#   旧 ckpt 携带会导致 load_state_dict strict 报 unexpected key。
_LEGACY_BUFFER_SUFFIXES = ('.inv_freq',)


def filter_legacy_buffers(sd: dict) -> dict:
    """过滤掉历史 ckpt 里残留的非持久 buffer（当前只剔 `.inv_freq`）。

    让旧 ckpt 在新代码下仍可 strict=True 加载，避免每处 loader 各自写 strict=False
    掩盖真问题。若将来引入新的 persistent=False buffer，按需补 suffix。
    """
    if not any(k.endswith(_LEGACY_BUFFER_SUFFIXES) for k in sd.keys()):
        return sd
    return {k: v for k, v in sd.items()
            if not k.endswith(_LEGACY_BUFFER_SUFFIXES)}


def clean_state_dict(sd: dict) -> dict:
    """加载侧标准清洗：strip DDP/compile 前缀 + 过滤遗留 buffer。"""
    return filter_legacy_buffers(strip_module_prefix(sd))


def compute_bpd(ce_loss: torch.Tensor, unit: str = "per_subpixel_nat",
                in_channels: int = 3) -> torch.Tensor:
    """
    从损失计算 bits/dim (bpd, bits per sub-pixel)。

    bits/dim = CE_loss / ln(2)（per-subpixel）
             = NLL_per_pixel / ln(2) / C（per-pixel，需要按通道数摊到 sub-pixel）

    参数:
      ce_loss:     标量损失张量
      unit:        损失单位，影响是否需要再除以通道数：
                     "per_subpixel_nat" — 每 sub-pixel 平均 nat（softmax 路径默认）
                     "per_pixel_nat"    — 每 pixel 平均 nat（DMoL 路径），bpd 还要 ÷ C
      in_channels: 仅 unit="per_pixel_nat" 时使用

    注：术语区分——bpd（bits per dimension/sub-pixel）是生成模型无损压缩
    文献的标准度量；bpp（bits per pixel）= bpd × C（彩色图 C=3）。
    本仓库以 bpd 为主指标，与 iGPT / PixelCNN++ / Sparse Transformer 等
    基线原文口径一致。

    参考:
      [1] Shannon, "A Mathematical Theory of Communication," 1948 —
          最优编码长度 = -log₂ p(x) = -ln p(x) / ln(2)
      [2] PixelCNN++ (Salimans 2017) — CIFAR-10: 2.92 bits/dim
    """
    if unit == "per_pixel_nat":
        return ce_loss / math.log(2) / in_channels
    if unit != "per_subpixel_nat":
        raise ValueError(f"compute_bpd: unknown unit '{unit}'")
    return ce_loss / math.log(2)
