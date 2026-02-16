"""
Scale Hyperprior Model

架构参考：
    [2] Ballé et al., "Variational Image Compression with a Scale
        Hyperprior," ICLR 2018, Figure 2 & Section 3.

数据流：
    编码端:
        x  --(g_a)--> y  --(h_a)--> z
                                      |
                              FactorizedPrior (量化+似然)
                                      |
                                    z_hat
                                      |
                              --(h_s)--> [means, scales]
                                              |
                              GaussianConditional (量化+似然)
                                              |
                                            y_hat
    解码端:
        y_hat --(g_s)--> x_hat

参数化选择：
    [2] 原文只用 scale（σ），即 p(y|z) = N(0, σ²)。
    [3] Minnen et al. 2018 扩展为 mean+scale。
    本实现采用 [3] 的 mean+scale 版本，h_s 输出 2M 通道。
"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F

from .layers import conv, deconv, ResidualBlock
from ..entropy import FactorizedPrior, GaussianConditional

class HyperpriorModel(nn.Module):
    """
    参数:
        N: 中间通道数（h_a/h_s 使用）
        M: 主潜变量通道数（y 的通道数）
    """
    def __init__(self, N: int = 128, M: int = 192):
        super().__init__()
        self.N = N 
        self.M = M 

        # ---- 主编码器 g_a: x -> y ---- 
        # 4 次 stride-2 down sampling, 累计 /16 [2] Table 1 
        self.g_a = nn.Sequential(
            conv(3, N, k=5, s=2),  
            nn.ReLU(inplace=True),
            conv(N, N, k=5, s=2),
            nn.ReLU(inplace=True),
            ResidualBlock(N),
            conv(N, N, k=5, s=2),
            nn.ReLU(inplace=True),
            conv(N, M, k=5, s=2),
        )

        # ---- 主解码器 g_s：y_hat -> x_hat ----
        # 4 次 stride-2 上采样，累计 ×16  [2] Table 1
        self.g_s = nn.Sequential(
            deconv(M, N, k=5,s=2),
            nn.ReLU(inplace=True),
            deconv(N, N, k=5, s=2),
            nn.ReLU(inplace=True),
            ResidualBlock(N),
            deconv(N, N, k=5, s=2),
            nn.ReLU(inplace=True),
            deconv(N, 3, k=5, s=2),
            nn.Sigmoid(),
        )

        # ---- 超先验编码器 h_a: |y| -> z ----
        # 2次 stride-2 down sampling，在y基础上再 /4 [2] Table 1
        self.h_a = nn.Sequential(
            conv(M, N, k=3, s=1),
            nn.ReLU(inplace=True),
            conv(N, N, k=5, s=2),
            nn.ReLU(inplace=True),
            conv(N, N, k=5, s=2),
        )

        # ---- 超先验解码器 h_s: z_hat -> y_params ---- 
        # 2次 stride-2 up sampling, 回到y的分辨率 [2] Table 1
        # 输出 2*M 通道：前 M 通道是 mean，后 M 通道是 scale [3] 
        self.h_s = nn.Sequential(
            deconv(N, N, k=5, s=2),
            nn.ReLU(inplace=True),
            deconv(N, N, k=5, s=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(N, 2*M, kernel_size=3, stride=1, padding=1),
        )

        # ---- entropy model ---- 
        self.factorized_prior = FactorizedPrior(N)  # p(z) [1]
        self.gaussian_conditional = GaussianConditional() # p(y|z) [2][3]

    def forward(self, x: torch.Tensor) -> dict: 
        """
        完整前向传播。

        返回:
            x_hat:       重建图像
            likelihoods: {"y": ..., "z": ...} 用于计算 bpp
            y:           主潜变量（用于 MDL 分析）
        """
        # ---- encode ---- 
        y = self.g_a(x) 
        z = self.h_a(torch.abs(y))  # [2]: h_a 输入是 |y|

        # ---- z的量化 & 估计似然 ---- [1][2] 
        z_hat, z_likelihoods = self.factorized_prior(z)

        # 由 z_hat 生成 y 的条件分布参数 [2][3]
        y_params = self.h_s(z_hat)
        if y_params.shape[2:] != y.shape[2:]:
            y_params = F.interpolate(y_params, size=y.shape[2:], mode='bilinear', align_corners=False)
        
        means, scales = y_params.chunk(2, dim=1)  # 前 M 通道是 mean，后 M 通道是 scale
        scales = F.softplus(scales) + 1e-4
        # print(f"scales: min {scales.min().item():.4f}, max {scales.max().item():.4f}, mean {scales.mean().item():.4f}")
        
        # ---- y的量化 & 估计似然 ---- [2][3]
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales, means)

        # ---- decode ---- 
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {
                "y": y_likelihoods,
                "z": z_likelihoods
            },
            "y": y
        }