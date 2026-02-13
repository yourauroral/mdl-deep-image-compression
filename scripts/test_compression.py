"""
单图压缩评估脚本。
对接实验日志 Phase 1/2 的流程：加载图片 → 编解码 → 输出 PSNR/SSIM/BPP。
"""

import torch
from PIL import Image
from torchvision import transforms

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mdlic.models import HyperpriorModel
from mdlic.utils.metrics import psnr, compute_ssim, compute_bpp

# ---- 路径配置 ----
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
IMG_DIR = os.path.join(PROJECT_ROOT, "assets", "test_images")
SAVE_DIR = os.path.join(PROJECT_ROOT, "assets", "output")

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    net = HyperpriorModel(N=128, M=192).to(device).eval()

    # 默认用 kodim01，也可以通过命令行指定
    img_name = sys.argv[1] if len(sys.argv) > 1 else "kodim01.png"
    img_path = os.path.join(IMG_DIR, img_name)

    if not os.path.exists(img_path):
        print(f"⚠️  Image not found: {img_path}")
        print("   Using random 256x256 input for shape verification.")
        x = torch.rand(1, 3, 256, 256, device=device)
        H, W = 256, 256
    else:
        img = Image.open(img_path).convert("RGB")
        # Kodak 原始分辨率 768x512，裁剪为 16 的倍数（已满足）
        img = img.resize((512, 512), Image.LANCZOS)
        x = transforms.ToTensor()(img).unsqueeze(0).to(device)
        H, W = 512, 512
        print(f"Loaded: {img_path}")

    print(f"Input shape: {tuple(x.shape)}")

    with torch.no_grad():
        out = net(x)

    assert out["x_hat"].shape == x.shape, "Shape mismatch!"

    N_batch = x.shape[0]
    num_pixels = N_batch * H * W

    psnr_val = psnr(x, out["x_hat"])
    ssim_val = compute_ssim(x, out["x_hat"])
    bpp_val = compute_bpp(out["likelihoods"], num_pixels)

    print(f"\n{'='*40}")
    print(f"  📊 PSNR:  {psnr_val:.2f} dB")
    print(f"  📊 SSIM:  {ssim_val:.4f}")
    print(f"  📊 BPP:   {bpp_val:.4f}")
    print(f"{'='*40}")

    # 保存重建图
    os.makedirs(SAVE_DIR, exist_ok=True)
    save_path = os.path.join(SAVE_DIR, f"recon_{img_name}")
    out_img = transforms.ToPILImage()(out["x_hat"].squeeze(0).cpu().clamp(0, 1))
    out_img.save(save_path)
    print(f"Saved: {save_path}")

if __name__ == "__main__":
    main()