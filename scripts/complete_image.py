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
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))


def _build_model(config, checkpoint, device):
    import torch
    from mdlic.utils import clean_state_dict
    from scripts.train import _build_ccigpt_from_config, _build_model_from_config

    mcfg = config["model"]
    model_type = mcfg.get("type", "igpt")
    model = (_build_ccigpt_from_config(mcfg, device) if model_type == "ccigpt"
             else _build_model_from_config(mcfg, device))
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(clean_state_dict(sd))
    model.eval()
    return model, model_type


def _logits_at(igpt, buf, coarse_ctx, pos):
    """buf (1, seq_len-1) long → 位置 pos 的下一 token logits (1,V) fp32。"""
    import torch
    with torch.no_grad():
        hidden, position_ids = igpt._embed_inputs(buf, coarse_ctx=coarse_ctx)
        for block in igpt.blocks:
            hidden = block(hidden, position_ids=position_ids)
        logits = igpt.head(hidden[:, pos:pos + 1, :])
    return logits.float().squeeze(1)


def _sample(logits_row, temperature, top_k):
    """logits (1,V) → 采样 token (int)。temperature>0 + 可选 top_k 截断。"""
    import torch
    logits = logits_row.squeeze(0).double()
    if temperature <= 0:
        return int(logits.argmax().item())          # 贪心
    logits = logits / temperature
    if top_k and top_k < logits.numel():
        kth = torch.topk(logits, top_k).values[-1]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, 1).item())


def _complete_one(model, model_type, x, keep_frac, temperature, top_k, device):
    """x (1,C,H,W) float[0,1] → (orig_u8, masked_u8, completed_u8) 三张 (C,H,W) uint8。"""
    import torch

    x = x.to(device).clamp(0, 1).float()
    igpt = model.fine if model_type == "ccigpt" else model
    C, H = igpt.in_channels, igpt.image_size
    T = igpt.seq_len
    seq_in = T - 1

    with torch.amp.autocast(device_type=device.type, enabled=False):
        coarse_ctx = None
        if model_type == "ccigpt":
            x_c = model._coarse_input(x)                  # bit-exact coarse 源头（单点推导）
            coarse_tokens = model.coarse._tokenize(x_c)
            coarse_ctx = model.ctx_alpha * model._compute_coarse_ctx(coarse_tokens)

        tokens = igpt._tokenize(x).clone()           # (1, T) 真实 token
        # 保留前 keep 个 token（raster pixel-first → 等价保留上半若干行）
        keep = int(round(keep_frac * T))
        keep = max(1, min(keep, T))                  # 至少保留 token0
        gen = tokens.clone()

        buf = torch.zeros((1, seq_in), dtype=torch.long, device=device)
        for m in range(keep, T):
            buf[0, :m] = gen[0, :m]                  # 前缀=已知+已采样
            logits = _logits_at(igpt, buf, coarse_ctx, m - 1)
            gen[0, m] = _sample(logits, temperature, top_k)

    def detok(t):
        return t.view(1, H, H, C).permute(0, 3, 1, 2).contiguous()[0].to(torch.uint8)

    orig = (x.clamp(0, 1) * 255).round().to(torch.uint8)[0]
    completed = detok(gen)
    # masked 可视化：已知区域真实像素，未知区域填中灰 128
    masked_tokens = tokens.clone()
    masked_tokens[0, keep:] = 128
    masked = detok(masked_tokens)
    return orig.cpu(), masked.cpu(), completed.cpu()


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

    import torch
    import yaml
    from scripts.evaluate import _load_dataset

    torch.manual_seed(args.seed)
    with open(args.config) as f:
        config = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_type = _build_model(config, args.checkpoint, device)
    print(f"== 模型加载 (type={model_type}, device={device}) ==")

    dataset, dataset_name = _load_dataset(config)
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
