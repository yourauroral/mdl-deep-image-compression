"""诊断：fast-encode (单次 forward) → gold-decode (逐 token) 跨路径一致性。

复现 Panel 10 的 /api/encode → /api/decode 真实路径，定位失步首个 token 位置。
仅 AutoDL 跑（要 GPU + v2 ckpt）。WSL 勿跑。

  python scripts/diag_fast_vs_gold.py --config configs/ccigpt_cifar10_s_rgb_ronly_v2.yaml \
      --checkpoint experiments/ccigpt_cifar10_s_rgb_ronly_v2/checkpoints/best.pth \
      --image figures/cifar10_sample_32x32.png
"""
import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--image", required=True)
    args = ap.parse_args()

    import yaml
    import torch
    import numpy as np
    from PIL import Image
    from torchvision import transforms

    from scripts.train import _build_ccigpt_from_config
    from src.mdlic.utils import clean_state_dict
    from scripts.verify_lossless import (
        _encode_sequence, _decode_sequence, _logits_from_tokens, _probs,
    )
    # 从 demo 拿 fast 路径，原样复用，保证和 /api/encode 同源
    from demo.server import _encode_image_fast, _probs_from_logits

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(args.config) as f:
        config = yaml.safe_load(f)
    mcfg = config["model"]
    model = _build_ccigpt_from_config(mcfg, device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    sd = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(clean_state_dict(sd))
    model.eval()

    img = Image.open(args.image).convert("RGB").resize((32, 32), Image.Resampling.BILINEAR)
    x = transforms.ToTensor()(img).unsqueeze(0).to(device).clamp(0, 1).float()

    print("=" * 70)
    print("STEP 1  fast-encode (单次 forward, 同 /api/encode)")
    enc = _encode_image_fast(model, "ccigpt", x, device)
    fast_recon = enc["recon_tokens"]
    print(f"  fast 自检 ok_decode = {enc['ok_decode']}  (fast→fast 自洽)")

    print("STEP 2  gold-decode 那串 fast 写的 bits (同 /api/decode)")
    import torch.nn.functional as F
    with torch.amp.autocast(device_type=device.type, enabled=False):
        # coarse: gold 解 fast 写的 c_bits
        N_c = model.coarse.seq_len
        coarse_dec = _decode_sequence(model.coarse, enc["c_bits"], N_c, None, device, "coarse", 0)
        coarse_ctx = model.ctx_alpha * model._compute_coarse_ctx(coarse_dec)
        N_f = model.fine.seq_len
        fine_dec = _decode_sequence(model.fine, enc["f_bits"], N_f, coarse_ctx, device, "fine", 0)
    gold_recon = fine_dec[0].tolist()

    same = (gold_recon == fast_recon)
    print(f"\n  fast-encode → gold-decode 还原 == fast 原 token : {same}")
    if not same:
        # 定位首个分歧 token
        first = next(i for i in range(min(len(gold_recon), len(fast_recon)))
                     if gold_recon[i] != fast_recon[i])
        print(f"  >>> 首个分歧 token 位置 i={first}: fast={fast_recon[first]} gold={gold_recon[first]}")
        print(f"      (= 该位置的概率表 fast vs gold 不一致 → 算术解码失步级联)")

    print("=" * 70)
    print("STEP 3  直接对比同一位置: 单次 forward logits vs 逐 token logits")
    # fine 真 token
    fine_tokens = model.fine._tokenize(x)
    with torch.amp.autocast(device_type=device.type, enabled=False):
        coarse_tokens = model.coarse._tokenize(F.adaptive_avg_pool2d(x, model.coarse_size)[:, :model.coarse.in_channels])
        cctx = model.ctx_alpha * model._compute_coarse_ctx(coarse_tokens)
        out_f = model.fine(x, coarse_ctx=cctx)
        full_logits = out_f["logits"].float()[0]   # (T-1, V)  单次 forward
        seq_in = model.fine.seq_len - 1
        buf = torch.zeros((1, seq_in), dtype=torch.long, device=device)
        max_abs = 0.0
        first_div = None
        for m in range(1, min(seq_in + 1, fine_tokens.shape[1])):
            buf[0, m - 1] = fine_tokens[0, m - 1]
            row_gold = _logits_from_tokens(model.fine, buf, cctx, m - 1)[0].float()  # 逐 token
            row_full = full_logits[m - 1]
            d = (row_gold - row_full).abs().max().item()
            if d > max_abs:
                max_abs = d
            if d > 0 and first_div is None:
                first_div = (m - 1, d)
    print(f"  单次forward vs 逐token logits 最大 |Δ| = {max_abs:.3e}")
    if first_div:
        print(f"  首个非零差位置 pos={first_div[0]}  |Δ|={first_div[1]:.3e}")
        print("  >>> 若 max|Δ|>0 → 两路径数值不一致 = 失步根因。修复：/api/encode 改用 gold _encode_sequence")
    else:
        print("  两路径逐位 bit-identical → desync 另有其因（非数值），需进一步查容器/位打包")


if __name__ == "__main__":
    main()
