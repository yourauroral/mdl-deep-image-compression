"""
熵模型单元测试。
验证：
  1. FactorizedPrior 的输出形状与似然范围
  2. GaussianConditional 的输出形状与似然范围
  3. 完整 HyperpriorModel 的 forward + bpp 可计算
"""

import torch
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mdlic.entropy import FactorizedPrior, GaussianConditional
from mdlic.models import HyperpriorModel
from mdlic.utils.metrics import compute_bpp


def test_factorized_prior():
    print("Testing FactorizedPrior...")
    fp = FactorizedPrior(channels=128)
    z = torch.randn(2, 128, 4, 4)

    # 训练模式
    fp.train()
    z_hat, lk = fp(z)
    assert z_hat.shape == z.shape
    assert lk.shape == z.shape
    assert (lk > 0).all(), "Likelihoods must be positive"
    assert (lk <= 1).all(), "Likelihoods must be <= 1"
    print(f"  [train] z_hat: {tuple(z_hat.shape)}, "
          f"lk range: [{lk.min():.6f}, {lk.max():.6f}]")

    # 推理模式
    fp.eval()
    z_hat, lk = fp(z)
    assert (z_hat == torch.round(z)).all(), "Eval should use round"
    print(f"  [eval]  z_hat: {tuple(z_hat.shape)}, "
          f"lk range: [{lk.min():.6f}, {lk.max():.6f}]")
    print("  PASSED\n")


def test_gaussian_conditional():
    print("Testing GaussianConditional...")
    gc = GaussianConditional()
    y = torch.randn(2, 192, 16, 16)
    means = torch.zeros_like(y)
    scales = torch.ones_like(y) * 0.5

    gc.train()
    y_hat, lk = gc(y, scales, means)
    assert y_hat.shape == y.shape
    assert lk.shape == y.shape
    assert (lk > 0).all()
    assert (lk <= 1).all()
    print(f"  [train] y_hat: {tuple(y_hat.shape)}, "
          f"lk range: [{lk.min():.6f}, {lk.max():.6f}]")

    gc.eval()
    y_hat, lk = gc(y, scales, means)
    print(f"  [eval]  y_hat: {tuple(y_hat.shape)}, "
          f"lk range: [{lk.min():.6f}, {lk.max():.6f}]")
    print("  PASSED\n")


def test_full_model():
    print("Testing full HyperpriorModel...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = HyperpriorModel(N=128, M=192).to(device)

    x = torch.rand(1, 3, 256, 256, device=device)

    # 训练模式
    net.train()
    out = net(x)
    assert out["x_hat"].shape == x.shape
    assert out["y"].shape == (1, 192, 16, 16)
    bpp = compute_bpp(out["likelihoods"], 1 * 256 * 256)
    print(f"  [train] x_hat: {tuple(out['x_hat'].shape)}, bpp: {bpp:.4f}")

    # 推理模式
    net.eval()
    with torch.no_grad():
        out = net(x)
    assert out["x_hat"].shape == x.shape
    bpp = compute_bpp(out["likelihoods"], 1 * 256 * 256)
    print(f"  [eval]  x_hat: {tuple(out['x_hat'].shape)}, bpp: {bpp:.4f}")
    print("  PASSED\n")


if __name__ == "__main__":
    test_factorized_prior()
    test_gaussian_conditional()
    test_full_model()
    print("All tests passed!")