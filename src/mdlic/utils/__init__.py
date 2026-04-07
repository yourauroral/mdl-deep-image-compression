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


def build_gaussian_targets(targets: torch.Tensor, vocab_size: int,
                           sigma: float) -> torch.Tensor:
    """
    构建 Gaussian label smoothing 的 soft target 分布。

    对每个 target 值 t，生成 P(k) ∝ exp(-(k-t)²/(2σ²)), k=0..V-1，
    并通过 softmax 归一化为概率分布。保留像素值的序数关系：
    预测 127 时，128 获得的概率远高于 0。

    动机:
      标准 categorical CE 将 256 个像素值视为无序类别，
      预测 128（差 1）与预测 0（差 127）惩罚相同。
      Gaussian soft target 注入序数先验，使模型学到"接近即好"。
      这与 PixelCNN++ 使用 discretized logistic 的动机一致。

    参考:
      [1] Szegedy et al., "Rethinking the Inception Architecture," CVPR 2016,
          arXiv:1512.00567 — 提出 uniform label smoothing 正则化
      [2] Salimans et al., "PixelCNN++," ICLR 2017 —
          discretized logistic mixture 隐式保留序数关系

    参数:
      targets: (M,) long tensor，取值 [0, vocab_size-1]
      vocab_size: int，词汇表大小（256）
      sigma: float，高斯标准差（推荐 1.0）
    返回:
      (M, vocab_size) float tensor，每行和为 1 的 soft target 分布
    """
    # arange: [0, 1, ..., V-1]，shape (1, V)
    bins = torch.arange(vocab_size, device=targets.device, dtype=torch.float32).unsqueeze(0)
    # targets: (M,) → (M, 1)
    t = targets.unsqueeze(1).float()
    # log_probs = -(k - t)² / (2σ²)，shape (M, V)
    log_probs = -((bins - t) ** 2) / (2.0 * sigma * sigma)
    # softmax 归一化（数值稳定，等价于 exp + normalize）
    return torch.softmax(log_probs, dim=-1)
