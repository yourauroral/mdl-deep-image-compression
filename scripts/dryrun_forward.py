import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
from mdlic.models import HyperpriorModel  

def print_shapes(d, prefix=""):
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            print(f"{prefix}{k} {tuple(v.shape)}")
        elif isinstance(v, dict):
            print_shapes(v, prefix=f"{prefix}{k}.")
        else:
            print(f"{prefix}{k} (type: {type(v).__name__})")

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = HyperpriorModel().to(device).eval()
    x = torch.rand(1, 3, 512, 512, device=device)
    out = net(x)
    # 调试信息
    print("\n=== 似然统计 ===")
    z_like = out["likelihoods"]["z"]
    y_like = out["likelihoods"]["y"]
    print(f"z likelihood: shape {z_like.shape}, min {z_like.min().item():.3e}, max {z_like.max().item():.3e}, mean {z_like.mean().item():.3e}")
    print(f"y likelihood: shape {y_like.shape}, min {y_like.min().item():.3e}, max {y_like.max().item():.3e}, mean {y_like.mean().item():.3e}")

    # 估算 bpp
    import math
    num_pixels = x.shape[0] * x.shape[2] * x.shape[3]
    z_bpp = -torch.log2(z_like).sum() / num_pixels
    y_bpp = -torch.log2(y_like).sum() / num_pixels
    print(f"z bpp: {z_bpp.item():.6f}")
    print(f"y bpp: {y_bpp.item():.6f}")
    print(f"total bpp: {(z_bpp + y_bpp).item():.6f}")

    print("\n=== 潜在变量统计 ===")
    y = out["y"]
    print(f"y: shape {y.shape}, mean {y.mean().item():.4f}, std {y.std().item():.4f}, min {y.min().item():.4f}, max {y.max().item():.4f}")
    print(f"x_hat: mean {out['x_hat'].mean().item():.4f}, std {out['x_hat'].std().item():.4f}")

    # 额外：获取 scales 和 means（需要临时修改 GaussianConditional 的 forward 返回它们，或者通过模型内部提取）
    # 这里假设你可以访问，如果不方便则跳过，但最好能拿到
    # 如果无法直接获得，可以考虑打印 GaussianConditional 内部的 scales clamp 前后的值（在 forward 中添加打印）
    print_shapes(out)
    assert out["x_hat"].shape == x.shape
    print("OK")

if __name__ == '__main__':
    main()