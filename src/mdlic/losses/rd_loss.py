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
    _PIXEL_MAX_SQ = 255.0 ** 2

    def __init__(self, lmbda: float = 0.01):
        super().__init__()
        self.lmbda = lmbda
        self.mse = nn.MSELoss()

    def forward(
        self,
        output: dict,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        N, _, H, W = target.shape
        num_pixels = N * H * W
        eps = 1e-10

        # ---- Rate ----
        bpp_y = -torch.log2(output["likelihoods"]["y"]).sum() / num_pixels
        bpp_z = -torch.log2(output["likelihoods"]["z"]).sum() / num_pixels
        bpp = bpp_y + bpp_z

        # ---- Distortion ----
        mse = self.mse(output["x_hat"], target)
        mse_val = mse.item()
        psnr = 10.0 * math.log10(1.0 / max(mse_val, eps))

        # ---- RD Loss ----
        loss = self.lmbda * self._PIXEL_MAX_SQ * mse + bpp

        stats = {
            "loss": loss.item(),
            "bpp": bpp.item(),
            "bpp_y": bpp_y.item(),
            "bpp_z": bpp_z.item(),
            "bpp_y_ratio": bpp_y.item() / (bpp.item() + eps),
            "mse": mse_val,
            "psnr": psnr,
        }

        return loss, stats
