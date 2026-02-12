import torch
import torch.nn as nn
from .layers import conv, deconv, ResidualBlock

class HyperpriorModel(nn.Module):
    """
    Ballé 2018 scale hyperprior-style scaffold (forward only).

    输出:
      x_hat: 重建图（仅用于 shape 验证；严格实现会用量化后的 y_hat）
      y: 主潜变量
      z: 超潜变量
      y_params: 用于建模 p(y|z) 的参数（这里占位，后续 PR-3 接高斯条件分布/熵模型）
    """
    def __init__(self, N: int = 128, M: int = 192):
        super().__init__()

        self.g_a = nn.Sequential(
            conv(3, N, k=5, s=2), nn.ReLU(inplace=True),
            conv(N, N, k=5, s=2), nn.ReLU(inplace=True),
            ResidualBlock(N),
            conv(N, N, k=5, s=2), nn.ReLU(inplace=True),
            conv(N, M, k=5, s=2),
        )

        self.g_s = nn.Sequential(
            deconv(M, N, k=5, s=2), nn.ReLU(inplace=True),
            deconv(N, N, k=5, s=2), nn.ReLU(inplace=True),
            ResidualBlock(N),
            deconv(N, N, k=5, s=2), nn.ReLU(inplace=True),
            deconv(N, 3, k=5, s=2),
            nn.Sigmoid(),
        )

        self.h_a = nn.Sequential(
            conv(M, N, k=3, s=1), nn.ReLU(inplace=True),
            conv(N, N, k=5, s=2), nn.ReLU(inplace=True),
            conv(N, N, k=5, s=2),
        )

        self.h_s = nn.Sequential(
            deconv(N, N, k=5, s=2), nn.ReLU(inplace=True),
            deconv(N, N, k=5, s=2), nn.ReLU(inplace=True),
            nn.Conv2d(N, 2 * M, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor):
        y = self.g_a(x)
        z = self.h_a(torch.abs(y))
        y_params = self.h_s(z)
        x_hat = self.g_s(y)
        return {"x_hat": x_hat, "y": y, "z": z, "y_params": y_params}