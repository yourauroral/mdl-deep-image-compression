"""
单图压缩评估脚本。
对接实验日志 Phase 1/2 的流程：加载图片 → 编解码 → 输出 PSNR/SSIM/BPP。
"""

import torch
from PIL import Image
from torchvision import transforms
import yaml
import sys
import os

# 将项目根目录加入 Python 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# 导入模型和指标
from src.mdlic import HyperpriorModel
from src.mdlic.utils.metrics import psnr, compute_ssim, compute_bpp

# ---- 路径配置 ----
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "hyperprior_mse.yaml")
SAVE_DIR = os.path.join(PROJECT_ROOT, "assets", "output")

def load_config():
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    return config

def find_first_image(dir_path):
    """返回目录中第一张图片的路径，或 None"""
    exts = ('.png', '.jpg', '.jpeg', '.bmp')
    for f in sorted(os.listdir(dir_path)):
        if f.lower().endswith(exts):
            return os.path.join(dir_path, f)
    return None

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # 加载配置，获取 Kodak 路径
    config = load_config()
    kodak_dir = config['data']['test']['kodak']

    # 确定图像路径
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        if not os.path.isabs(img_path):
            # 如果是相对路径，则假设相对于项目根目录
            img_path = os.path.join(PROJECT_ROOT, img_path)
    else:
        # 默认使用 Kodak 第一张图
        img_path = find_first_image(kodak_dir)
        if img_path is None:
            raise FileNotFoundError(f"No images found in {kodak_dir}")

    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")

    # 加载图像
    img = Image.open(img_path).convert("RGB")
    # 保持宽高比，调整尺寸到 16 的倍数（满足模型下采样要求）
    w, h = img.size
    new_w = (w // 16) * 16
    new_h = (h // 16) * 16
    if new_w != w or new_h != h:
        img = img.resize((new_w, new_h), Image.LANCZOS)
        print(f"Resized from {w}x{h} to {new_w}x{new_h} to be divisible by 16")
    x = transforms.ToTensor()(img).unsqueeze(0).to(device)
    print(f"Loaded: {img_path}")
    print(f"Input shape: {tuple(x.shape)}")

    # 初始化模型
    net = HyperpriorModel(N=config['model']['N'], M=config['model']['M']).to(device).eval()

    with torch.no_grad():
        out = net(x)

    assert out["x_hat"].shape == x.shape, "Shape mismatch!"

    N_batch = x.shape[0]
    num_pixels = N_batch * x.shape[2] * x.shape[3]  # H*W

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
    base_name = os.path.basename(img_path)
    save_path = os.path.join(SAVE_DIR, f"recon_{base_name}")
    out_img = transforms.ToPILImage()(out["x_hat"].squeeze(0).cpu().clamp(0, 1))
    out_img.save(save_path)
    print(f"Saved: {save_path}")

if __name__ == "__main__":
    main()