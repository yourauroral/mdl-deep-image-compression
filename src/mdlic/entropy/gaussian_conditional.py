"""
Gaussian Conditional —— 用于 y 的条件先验 p(y|z)。

理论依据：
    [2] Ballé et al., "Variational Image Compression with a Scale
        Hyperprior," ICLR 2018, Section 3.1, Eq.(9):
        p(y_hat | z) = ∏_i  [ N(0, σ_i²) * U(-0.5, 0.5) ] (y_hat_i)
                     = ∏_i  [ Φ((y_hat_i + 0.5) / σ_i) - Φ((y_hat_i - 0.5) / σ_i) ]
        其中 Φ 是标准正态分布的 CDF，σ_i 由超先验分支 h_s(z) 产生。

    [3] Minnen et al., NeurIPS 2018, Section 3:
        扩展为 mean + scale 参数化：
        p(y_hat | z) = ∏_i  [ Φ((y_hat_i - μ_i + 0.5) / σ_i)
                             - Φ((y_hat_i - μ_i - 0.5) / σ_i) ]
        其中 μ_i, σ_i 均由 h_s(z) 产生（输出通道为 2M，前半 mean，后半 scale）。

标准正态 CDF 实现：
    Φ(x) = 0.5 * erfc(-x / √2)
    使用 PyTorch 内置 torch.erfc。
"""

import torch 
import torch.nn as nn
import math 

from .quantize import noise_quant 

class GaussianConditional(nn.Module):
    """
    条件高斯熵模型。

    输入:
        y:     主潜变量 (B, M, H, W)
        means:  均值参数 (B, M, H, W)，来自 h_s(z) [3]
        scales: 标准差参数 (B, M, H, W)，来自 h_s(z) [2]

    输出:
        y_hat:       量化后的 y
        likelihoods: 每个元素的条件概率质量
    """

    SCALE_MIN = 0.11
    """
    scale 下界，防止标准差过小导致数值不稳定。
    参考 [2] 中对 σ 的 clamp 处理。
    """

    def __init__(self):
      super().__init__()
    
    @staticmethod
    def _standardized_cdf(x: torch.Tensor) -> torch.Tensor:
      """
      标准正态分布 CDF:  Φ(x) = 0.5 * erfc(-x / √2)

      使用 PyTorch 的 torch.erfc 实现，数值精度优于
      手动用 sigmoid 近似。
      """
      return 0.5 * torch.erfc(-x / math.sqrt(2)) 
    
    def forward(
      self, 
      y: torch.Tensor,
      scales: torch.Tensor,
      means: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
      """
      前向传播。

      训练：y_hat = y + U(-0.5, 0.5)     [1] Section 3
      推理：y_hat = round(y - means) + means    [3]

      似然 [2] Eq.(9) / [3] Eq.(2):
          p(y_hat_i | μ_i, σ_i) = Φ((y_hat_i - μ_i + 0.5) / σ_i)
                                  - Φ((y_hat_i - μ_i - 0.5) / σ_i)
      """

      scales = torch.clamp(scales, min=self.SCALE_MIN)

      if self.training:
        y_hat = noise_quant(y)
      else:
        # 推理时：先减去mean，round，再加回mean
        y_hat = torch.round(y - means) + means
      
      # 中心化：以mean为中心计算偏移[3]
      centered = y_hat - means

      # 似然 = Φ((centered + 0.5) / σ) - Φ((centered - 0.5) / σ)  [2] 
      upper = self._standardized_cdf((centered + 0.5) / scales)
      lower = self._standardized_cdf((centered - 0.5) / scales)
      likelihoods = upper - lower

      likelihoods = torch.clamp(likelihoods, min=1e-10) # 避免log(0)

      return y_hat, likelihoods 
      