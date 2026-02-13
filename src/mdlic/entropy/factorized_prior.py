"""
Factorized Prior (Entropy Bottleneck) —— 用于 z 的无条件先验。

理论依据：
    [1] Ballé et al., "End-to-end Optimized Image Compression,"
        ICLR 2017, Section 3 & Appendix C.
    [2] Ballé et al., "Variational Image Compression with a Scale
        Hyperprior," ICLR 2018, Section 3.

核心思想：
    对每个通道 i，学习一个参数化的累积分布函数 (CDF) c_i(x)。
    CDF 由一系列仿射变换 + 非线性函数复合而成：
        c(x) = sigmoid( H_K * f_{K-1}( ... f_1( H_1 * x + b_1 ) ... ) + b_K )
    其中 H_k 的元素通过 softplus 约束为正，保证 CDF 单调递增。

    量化后的似然：
        p(z_hat_i) = c_i(z_hat_i + 0.5) - c_i(z_hat_i - 0.5)
    即量化区间 [z_hat - 0.5, z_hat + 0.5) 上的概率质量。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .quantize import noise_quant


class FactorizedPrior(nn.Module):
    """
    可学习的 Factorized Prior，对 z 的每个通道独立建模。

    参数:
        channels: z 的通道数（通常等于 N）
        filters:  CDF 网络每层的宽度
                  默认 (3, 3, 3) 来自 [1] Appendix C
    """

    def __init__(self, channels: int, filters: tuple = (3, 3, 3)):
        super().__init__()
        self.channels = channels

        # 构建 CDF 参数化网络的参数
        # 维度链: 1 -> filters[0] -> filters[1] -> ... -> filters[-1] -> 1
        # 每层对 channels 个通道独立操作
        dims = (1,) + filters + (1,)
        self._matrices = nn.ParameterList()
        self._biases = nn.ParameterList()
        self._factors = nn.ParameterList()

        for i in range(len(dims) - 1):
            # 矩阵：(channels, dims[i+1], dims[i])
            # 用均匀分布初始化，经 softplus 后保证正性 -> CDF 单调 [1]
            matrix = nn.Parameter(
                torch.empty(channels, dims[i + 1], dims[i]).uniform_(-0.5, 0.5)
            )
            self._matrices.append(matrix)

            # 偏置：(channels, dims[i+1], 1)
            bias = nn.Parameter(
                torch.empty(channels, dims[i + 1], 1).uniform_(-0.5, 0.5)
            )
            self._biases.append(bias)

            # 门控因子（中间层用，控制非线性强度）[1] Appendix C
            if i < len(dims) - 2:
                factor = nn.Parameter(
                    torch.zeros(channels, dims[i + 1], 1)
                )
                self._factors.append(factor)

    def _cdf(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算参数化 CDF c(x)。

        输入 x: (batch, channels, height, width)
        输出 cdf: 同形状，值域 (0, 1)

        实现参考 [1] Appendix C:
            c(x) = sigmoid( H_K * relu_like( ... H_1 * x + b_1 ... ) + b_K )
        """
        B, C, H, W = x.shape
        # 展平空间维度: (B, C, 1, H*W)
        logits = x.reshape(B, C, 1, H * W)

        for i, (matrix, bias) in enumerate(zip(self._matrices, self._biases)):
            # softplus 保证矩阵元素为正 -> CDF 单调递增 [1]
            w = F.softplus(matrix)  # (C, out_dim, in_dim)

            # 对每个通道独立做矩阵乘法
            # w: (C, out_dim, in_dim), logits: (B, C, in_dim, HW)
            # -> (B, C, out_dim, HW)
            logits = torch.einsum("coi,bcin->bcon", w, logits)
            logits = logits + bias.unsqueeze(0)  # broadcast bias

            # 中间层：加非线性 [1] Appendix C
            if i < len(self._factors):
                factor = torch.tanh(self._factors[i])  # (C, out_dim, 1)
                logits = logits + factor.unsqueeze(0) * torch.tanh(logits)

        # 最后一层过 sigmoid 得到 CDF in (0,1)
        cdf = torch.sigmoid(logits)
        return cdf.reshape(B, C, H, W)

    def forward(self, z: torch.Tensor) -> tuple:
        """
        前向传播。

        训练时：z_hat = z + U(-0.5, 0.5)     [1] Section 3
        推理时：z_hat = round(z)

        似然计算 [1][2]:
            p(z_hat) = c(z_hat + 0.5) - c(z_hat - 0.5)

        返回:
            z_hat: 量化后的 z
            likelihoods: 每个元素的概率质量
        """
        if self.training:
            z_hat = noise_quant(z)
        else:
            z_hat = torch.round(z)

        # 似然 = CDF(z_hat + 0.5) - CDF(z_hat - 0.5)  [1]
        upper = self._cdf(z_hat + 0.5)
        lower = self._cdf(z_hat - 0.5)
        likelihoods = upper - lower

        # 数值下界保护（防止 log(0)）
        likelihoods = torch.clamp(likelihoods, min=1e-9)

        return z_hat, likelihoods