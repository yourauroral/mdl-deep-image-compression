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


def compute_bpp(ce_loss: torch.Tensor) -> torch.Tensor:
    """
    从交叉熵损失计算 bits/dim (bits per sub-pixel)。

    bits/dim = CE_loss / ln(2)

    参考:
      [1] Shannon, "A Mathematical Theory of Communication," 1948 —
          最优编码长度 = -log₂ p(x) = -ln p(x) / ln(2)
      [2] PixelCNN++ (Salimans 2017) — CIFAR-10: 2.92 bits/dim
    """
    return ce_loss / math.log(2)
