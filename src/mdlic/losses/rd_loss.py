"""
率失真损失 (Rate-Distortion Loss)。

理论依据：
    [2] Ballé et al., ICLR 2018, Eq.(2):
        L = R + λ · D
        其中 R = E[-log₂ p(y_hat)] + E[-log₂ p(z_hat)]   (比特率)
             D = E[d(x, x_hat)]                           (失真)

    在实现中：
        R 通过 likelihoods 计算：
            bpp = Σ -log₂(p_i) / num_pixels
        D 使用 MSE（对应 PSNR 优化目标）。

    本实现中 loss = λ * 255² * MSE + bpp
    乘以 255² 是因为 MSE 在 [0,1] 范围计算，
    而论文中 λ 的标定基于 [0,255] 范围 [2] Section 4。
"""

class RDLoss(nn.Module):
  """
  参数:
    lmbda: 拉格朗日乘子 λ，控制率失真权衡 [2]
        λ 越大 → 越重视质量 → bpp 越高
        典型值: 0.001, 0.003, 0.01, 0.03, 0.05
  """
  def __init__(self, lmbda: float = 0.01):
    super().__init__() 
    self.lmbda = lmbda 
    self.mse = nn.MSELoss() 
  
  def forward(
    self, 
    output: dict,
    target: torch.Tensor,
  ) -> tuple[torch.Tensor, dict]:
    """
    参数:
        output: HyperpriorModel.forward() 的返回值
        target: 原始图像 x

    返回:
        loss: 标量，用于反向传播
        stats: 字典，包含 bpp / mse / psnr 等中间量
    """
    N, _, H, W = target.shape
    num_pixels = N * H * W

    # ---- Rate ---- 
    # 计算 bpp = Σ -log₂(likelihood) / num_pixels [2] 
    bpp_y = -torch.log2(output["likelihoods"]["y"]).sum() / num_pixels
    bpp_z = -torch.log2(output["likelihoods"]["z"]).sum() / num_pixels
    bpp = bpp_y + bpp_z

    # ---- Distortion ----
    mse = self.mse(output["x_hat"], target)
    # 换算到 [0,255] 范围的 PSNR 仅用于监控
    psnr = 10.0 * math.log10(1.0 / mse.item()) if mse.item() > 0 else 100.0

    # ---- RD Loss [2] Eq.(2) ----
    # λ * 255² * MSE + bpp
    loss = self.lmbda * (255.0 ** 2) * mse + bpp 

    stats = {
        "loss": loss.item(),
        "bpp": bpp.item(),
        "bpp_y": bpp_y.item(),
        "bpp_z": bpp_z.item(),
        "mse": mse.item(),
        "psnr": psnr,
    }
    
    return loss, stats