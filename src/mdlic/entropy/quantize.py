"""
量化策略。

训练阶段:加均匀噪声代替不可导的 round 操作。
    y_hat = y + U(-0.5, 0.5)
    理论依据:[1] Ballé et al., ICLR 2017, Section 3
    "We replace the quantizer with additive uniform noise
     during training, which is a continuous relaxation."

推理阶段：直接 round。
    y_hat = round(y)

备选:Straight-Through Estimator (STE)
    前向用 round,反向直通梯度。
    参考:Bengio et al., "Estimating or Propagating Gradients
    Through Stochastic Neurons," arXiv:1308.3432, 2013
"""

import torch 

def noise_quant(x: torch.Tensor) -> torch.Tensor:
    """
    均匀噪声量化 训练用；
    参考:[1] Ballé et al., ICLR 2017, Section 3 用U(-0.5, 0.5)噪声近似量化
    使得似然计算中的积分可以用单点密度近似
    """
    noise = torch.empty_like(x).uniform_(-0.5, 0.5)
    return x + noise
  
def ste_round(x: torch.Tensor) -> torch.Tensor:
  """
  Straight-Through Estimator (STE): 前向round,反向直通梯度。
  """
  return x + (touch.round(x)-x).detach() 