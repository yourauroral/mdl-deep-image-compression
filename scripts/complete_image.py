"""图像补全 demo（AR inpainting）— Sparse Transformer 同款定性下游，跑在 AutoDL。

给定图像上半部分的真实像素，模型自回归采样补全下半部分。展示 CC-iGPT 的生成
能力（论文 §6.4 / 答辩定性面板）。AR 模型天然支持，不重训。

诚实声明（CC-iGPT 专属）
----------------------
fine 的条件 ctx 由 **整图** 下采样的 coarse thumbnail 计算（8×8 R-only）。因此
本 demo 的语义是"给定低分辨率缩略图 + 上半部分真实像素 → 补全下半部分"，coarse
是显式给定的 side-channel（与压缩时 coarse 走独立 bitstream 同源），不是偷看答案。
若要"完全不给下半任何信息"的纯补全，需让 coarse 也只看上半（本 demo 未做，注释存档）。

设计：与 verify_lossless 同源的 per-step forward（模型无 KV-cache，每步跑完整
forward，causal mask 保证 0 后缀不泄漏）。仅 CIFAR v2 实用（fine 3072 token）；
IN64 12288 token 太慢。

用法（AutoDL）:
    python scripts/complete_image.py \
        --config configs/ccigpt_cifar10_s_rgb_ronly_v2.yaml \
        --checkpoint experiments/ccigpt_cifar10_s_rgb_ronly_v2/checkpoints/best.pth \
        --num_images 4 --keep_frac 0.5 --temperature 1.0 --top_k 100 \
        --out experiments/completion_grid.png
"""
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from mdlic.completion import (
    complete_image as _complete_one,
)


def _build_model(config, checkpoint, device):
    import torch
    from mdlic.model_factory import build_model_from_config
    from mdlic.utils import clean_state_dict

    mcfg = config["model"]
    model_type = mcfg.get("type", "igpt")
    model = build_model_from_config(mcfg, device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(clean_state_dict(sd))
    model.eval()
    return model, model_type


def _save_grid(rows, out_path, scale=8):
    """rows: list of (orig, masked, completed) 每个 (C,H,W) uint8 → 拼成 PNG。

    scale: nearest 放大倍数。native 32×32 拼出来才 ~104×206 px，进 slides/PDF 被
    双线性插值放大会糊成一团；这里先按 native 拼好再整体 nearest 放大 scale 倍，
    保持像素锐利（与前端 /api/complete 的 _b64_png scale=4 同理）。
    """
    from PIL import Image
    import torch

    def to_img(t):
        return t.permute(1, 2, 0).numpy()            # (H,W,C)

    H = rows[0][0].shape[1]
    pad = 2
    ncol = 3
    grid_w = ncol * H + (ncol + 1) * pad
    grid_h = len(rows) * H + (len(rows) + 1) * pad
    canvas = torch.full((grid_h, grid_w, 3), 255, dtype=torch.uint8).numpy()
    for r, (o, m, c) in enumerate(rows):
        y = pad + r * (H + pad)
        for col, t in enumerate((o, m, c)):
            xs = pad + col * (H + pad)
            canvas[y:y + H, xs:xs + H, :] = to_img(t)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    img = Image.fromarray(canvas)
    if scale and scale > 1:
        img = img.resize((grid_w * scale, grid_h * scale), Image.Resampling.NEAREST)
    img.save(out_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--num_images", type=int, default=4)
    ap.add_argument("--keep_frac", type=float, default=0.5,
                    help="保留前 keep_frac 比例的 token（raster 上半），其余采样补全")
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="采样温度；0=贪心 argmax")
    ap.add_argument("--top_k", type=int, default=100, help="top-k 截断（0=不截断）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="experiments/completion_grid.png")
    ap.add_argument("--scale", type=int, default=8,
                    help="存图 nearest 放大倍数（native 32px 太小，进 slides 会糊；默认 8×）")
    args = ap.parse_args()

    if args.num_images < 1:
        ap.error("--num_images must be >= 1")
    if not math.isfinite(args.keep_frac) or not 0.0 < args.keep_frac <= 1.0:
        ap.error("--keep_frac must be finite and in (0, 1]")
    if not math.isfinite(args.temperature) or args.temperature < 0.0:
        ap.error("--temperature must be finite and >= 0")
    if args.top_k < 0:
        ap.error("--top_k must be >= 0")
    if args.scale < 1:
        ap.error("--scale must be >= 1")

    import torch
    import yaml
    from mdlic.data.evaluation import load_evaluation_dataset

    torch.manual_seed(args.seed)
    with open(args.config) as f:
        config = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_type = _build_model(config, args.checkpoint, device)
    print(f"== 模型加载 (type={model_type}, device={device}) ==")

    dataset, dataset_name = load_evaluation_dataset(config)
    if args.num_images > len(dataset):
        ap.error(
            f"--num_images ({args.num_images}) exceeds dataset size ({len(dataset)})"
        )
    print(f"== {dataset_name}：补全前 {args.num_images} 张, keep_frac={args.keep_frac}, "
          f"T={args.temperature}, top_k={args.top_k} ==")

    rows = []
    for idx in range(args.num_images):
        item = dataset[idx]
        x = (item[0] if isinstance(item, (tuple, list)) else item).unsqueeze(0)
        o, m, c = _complete_one(model, model_type, x, args.keep_frac,
                                args.temperature, args.top_k, device)
        rows.append((o, m, c))
        print(f"  image {idx}: done")

    _save_grid(rows, args.out, scale=args.scale)
    print(f"== 网格已保存: {args.out}（每行: 原图 | 已知上半(灰=待补) | 补全）==")


if __name__ == "__main__":
    main()
