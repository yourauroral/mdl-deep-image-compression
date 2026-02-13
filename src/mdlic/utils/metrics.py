"""
评估指标。

PSNR:  Peak Signal-to-Noise Ratio
       PSNR = 10 * log10(MAX² / MSE)
       MAX = 1.0（像素归一化到 [0,1]）

SSIM:  Structural Similarity Index [Wang et al., IEEE TIP 2004]
       本实现使用 scikit-image 计算（如未安装则 fallback 到简易版）。

BPP:   Bits Per Pixel
       bpp = Σ -log₂(p_i) / num_pixels
"""

import torch
import numpy as np
import math


def psnr(x: torch.Tensor, x_hat: torch.Tensor) -> float:
    """
    计算 PSNR (dB)。
    输入 x, x_hat 均为 [0, 1] 范围。
    """
    mse = torch.mean((x - x_hat) ** 2).item()
    if mse < 1e-10:
        return 100.0
    return 10.0 * math.log10(1.0 / mse)


def compute_ssim(x: torch.Tensor, x_hat: torch.Tensor) -> float:
    """
    计算 SSIM。
    需要 scikit-image；如未安装则返回 -1.0。

    参考：Wang, Bovik, Sheikh, Simoncelli,
          "Image Quality Assessment: From Error Visibility to
           Structural Similarity," IEEE TIP, 2004.
    """
    try:
        from skimage.metrics import structural_similarity as ssim_fn
    except ImportError:
        return -1.0

    # (B, C, H, W) -> 逐 batch 计算
    x_np = x.detach().cpu().numpy()
    xh_np = x_hat.detach().cpu().clamp(0, 1).numpy()

    vals = []
    for i in range(x_np.shape[0]):
        # 转成 (H, W, C)
        a = np.transpose(x_np[i], (1, 2, 0))
        b = np.transpose(xh_np[i], (1, 2, 0))
        vals.append(ssim_fn(a, b, channel_axis=2, data_range=1.0))
    return float(np.mean(vals))


def compute_bpp(likelihoods: dict, num_pixels: int) -> float:
    """
    从 likelihoods 计算总 bpp。
    bpp = Σ -log₂(p_i) / num_pixels   [2] Section 3
    """
    total = 0.0
    for key in likelihoods:
        lk = likelihoods[key]
        total += -torch.log2(lk).sum().item()
    return total / num_pixels