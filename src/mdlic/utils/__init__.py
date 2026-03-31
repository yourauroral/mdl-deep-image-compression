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

    设置范围:
      - Python random
      - NumPy
      - PyTorch CPU + CUDA（所有 GPU）
      - cuDNN deterministic（牺牲少量性能换取确定性）

    参考:
      [1] PyTorch Reproducibility 文档 — "Controlling sources of randomness"
      [2] CS336 "Language Models from Scratch," Stanford, 2024 — seed 设置最佳实践

    参数:
      seed: int — 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 确保 cuDNN 使用确定性算法（对 CIFAR 级小图影响可忽略）
    # Ref: PyTorch 文档 — torch.backends.cudnn.deterministic
    os.environ["PYTHONHASHSEED"] = str(seed)


def compute_bpp(ce_loss: torch.Tensor, channels: int) -> torch.Tensor:
    """
    从交叉熵损失计算 BPP (Bits Per Pixel)。

    BPP = CE_loss / ln(2) × C

    原理:
      CE_loss = -log p(x_t | x_{<t}) 的均值（nats/token），
      每个像素有 C 个通道（各自对应一个 token），
      除以 ln(2) 将 nats 转换为 bits。

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
