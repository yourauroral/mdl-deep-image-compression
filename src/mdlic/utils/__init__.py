"""
公共工具函数 — 被 train.py、evaluate.py、dryrun_forward.py 等多处复用。
"""

import os
import math
import random
import torch
import numpy as np


def seed_everything(seed: int = 42):
    """
    统一设置所有随机种子，确保实验可复现。

    参考:
      [1] PyTorch Reproducibility 文档 — "Controlling sources of randomness"
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def compute_bpp(ce_loss: torch.Tensor, channels: int) -> torch.Tensor:
    """
    从交叉熵损失计算 BPP (Bits Per Pixel)。

    BPP = CE_loss / ln(2) × C

    参考:
      [1] Shannon, "A Mathematical Theory of Communication," 1948 —
          最优编码长度 = -log₂ p(x) = -ln p(x) / ln(2)

    参数:
      ce_loss: scalar tensor — per-token 交叉熵（nats）
      channels: int — 通道数（通常 3）
    返回:
      scalar tensor — BPP（bits/pixel）
    """
    return (ce_loss / math.log(2)) * channels
