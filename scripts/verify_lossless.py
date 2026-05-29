"""真实可解性验证 — AR 算术编解码 roundtrip（在 AutoDL GPU 上跑）。

把训练好的 CC-iGPT 当成真实无损 codec：
  encode: 图像 → coarse/fine token → 逐步条件分布 → 算术编码 → 真实 bitstream
  decode: bitstream → 逐步重建同一条件分布 → 解出 token → 还原图像
最后断言 **逐像素 bit-identical**，并报告实际码长 (bits) 与 bits/dim，
与模型 teacher-forced NLL/bpd 对照。

这是"严格无损 + bitstream 真实可解"定位的硬证据，区别于只报 bpd 的工作。

关键设计
--------
1. 模型无 KV-cache（IGPT._embed_inputs 固定 seq_len-1）。decode 每步跑一次
   **完整 forward**：缓冲区 = 已解 token 前缀 + 0 填充后缀。causal mask 保证
   位置 m-1 的 logits 只依赖 token[0..m-1]，0 后缀不泄漏。
2. encode 与 decode 用**完全相同**的逐步函数 `_step_logits`，输入张量逐位相同
   → logits 逐位相同 → cumfreq 表相同 → 可逆。不依赖 teacher-forcing 与
   per-step 之间的浮点一致性（最保险）。
3. 全程 fp32 + autocast 关闭 + eval + no_grad，保证编/解码两侧确定性。
4. token 0 模型不预测，用均匀 256-way 先验编码（8 bit），计入总码长。
5. 双尺度：先解完 coarse（独立 bitstream），从 coarse token 重建 α·coarse_ctx，
   再按序解 fine —— 与 cc_igpt.forward 的 bit-exact ctx 路径同源。

用法（AutoDL）:
    python scripts/verify_lossless.py \
        --config configs/ccigpt_cifar10_s_rgb_ronly_v2.yaml \
        --checkpoint experiments/.../checkpoints/best.pth \
        --num_images 2

    python scripts/verify_lossless.py --self_test   # 仅 coder roundtrip，无需 GPU/ckpt
"""
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.mdlic.codec.arithmetic import (
    ArithmeticEncoder, ArithmeticDecoder, build_cumfreq, pack_bits, FREQ_TOTAL,
)

# 均匀 256-way 先验（token 0 用），常量，encode/decode 共用
_UNIFORM_CUM = list(range(0, FREQ_TOTAL + 1, FREQ_TOTAL // 256))


def _self_test():
    """不加载模型，仅验证 coder 在合成分布上 roundtrip 正确 + 码长接近熵。"""
    import random
    random.seed(0)
    V = 256
    syms, tables, ideal = [], [], 0.0
    for _ in range(2000):
        logits = [random.gauss(0, 2.5) for _ in range(V)]
        m = max(logits)
        ex = [math.exp(l - m) for l in logits]
        z = sum(ex)
        p = [e / z for e in ex]
        s = random.randrange(V)
        syms.append(s); tables.append(p); ideal += -math.log2(p[s])
    enc = ArithmeticEncoder()
    for s, p in zip(syms, tables):
        enc.encode(s, build_cumfreq(p))
    bits = enc.finish()
    dec = ArithmeticDecoder(bits)
    out = [dec.decode(build_cumfreq(p)) for p in tables]
    ok = out == syms
    print(f"[self_test] roundtrip bit-exact: {ok}")
    print(f"[self_test] coded {len(bits)} bit / ideal {ideal:.1f} bit / "
          f"overhead {100*(len(bits)-ideal)/ideal:.2f}%")
    assert ok, "coder roundtrip 失败"
    return ok


# ---- 以下函数仅在真实验证时才 import torch（self_test 不需要 GPU） ----

def _build_model(config, checkpoint, device):
    import torch
    from src.mdlic.utils import clean_state_dict
    from scripts.train import _build_ccigpt_from_config, _build_model_from_config

    mcfg = config["model"]
    model_type = mcfg.get("type", "igpt")
    if model_type != "ccigpt":
        # 单尺度 iGPT 也能验证，但本脚本聚焦 CC-iGPT 双尺度可解性
        model = _build_model_from_config(mcfg, device)
    else:
        model = _build_ccigpt_from_config(mcfg, device)

    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(clean_state_dict(sd))
    model.eval()
    return model, model_type


def _logits_from_tokens(igpt, buffer_tokens, coarse_ctx, pos_index):
    """buffer_tokens: (1, seq_len-1) long（前缀真实 + 后缀 0）。
    返回位置 pos_index 处预测下一 token 的 logits（1, V）fp32。
    """
    import torch
    with torch.no_grad():
        hidden, position_ids = igpt._embed_inputs(buffer_tokens, coarse_ctx=coarse_ctx)
        for block in igpt.blocks:
            hidden = block(hidden, position_ids=position_ids)
        logits = igpt.head(hidden[:, pos_index:pos_index + 1, :])  # (1,1,V)
    return logits.float().squeeze(1)  # (1, V)


def _probs(logits_row):
    """logits (1,V) fp32 → Python float 概率表（fp64 softmax，确定性）。"""
    import torch
    return torch.softmax(logits_row.double(), dim=-1).squeeze(0).tolist()


def _encode_sequence(igpt, tokens, coarse_ctx, device, tag, log_every):
    """逐步算术编码一段 token 序列，返回 bit 列表 + 理论 NLL(bit)。"""
    import torch
    T = tokens.shape[1]
    seq_in = igpt.seq_len - 1
    enc = ArithmeticEncoder()
    ideal_bits = 0.0

    # token 0：均匀先验
    enc.encode(int(tokens[0, 0].item()), _UNIFORM_CUM)
    ideal_bits += 8.0

    buf = torch.zeros((1, seq_in), dtype=torch.long, device=device)
    for m in range(1, T):
        buf[0, m - 1] = tokens[0, m - 1]
        logits = _logits_from_tokens(igpt, buf, coarse_ctx, m - 1)
        p = _probs(logits)
        sym = int(tokens[0, m].item())
        ideal_bits += -math.log2(max(p[sym], 1e-12))
        enc.encode(sym, build_cumfreq(p))
        if log_every and m % log_every == 0:
            print(f"    [{tag} encode] {m}/{T-1}")
    return enc.finish(), ideal_bits


def _decode_sequence(igpt, bits, T, coarse_ctx, device, tag, log_every):
    """逐步算术解码出 token 序列 (1, T)。每步 logits 必须与 encode 端逐位相同。"""
    import torch
    seq_in = igpt.seq_len - 1
    dec = ArithmeticDecoder(bits)
    out = torch.zeros((1, T), dtype=torch.long, device=device)

    out[0, 0] = dec.decode(_UNIFORM_CUM)
    buf = torch.zeros((1, seq_in), dtype=torch.long, device=device)
    for m in range(1, T):
        buf[0, m - 1] = out[0, m - 1]
        logits = _logits_from_tokens(igpt, buf, coarse_ctx, m - 1)
        p = _probs(logits)
        out[0, m] = dec.decode(build_cumfreq(p))
        if log_every and m % log_every == 0:
            print(f"    [{tag} decode] {m}/{T-1}")
    return out


def _detokenize(tokens, image_size, channels):
    """(1, H*W*C) pixel-first long token → (C,H,W) uint8 tensor（_tokenize 逆）。"""
    import torch
    t = tokens.view(1, image_size, image_size, channels)
    return t.permute(0, 3, 1, 2).contiguous()[0].to(torch.uint8)


def _verify_image(model, model_type, x, device, log_every):
    """对单张图 (1,C,H,W) float[0,1] 做 encode→decode→断言 bit-identical。"""
    import torch
    import torch.nn.functional as F

    x = x.to(device).clamp(0, 1).float()
    C = model.in_channels       # CCIGPT 与 IGPT 都暴露 in_channels / image_size
    H = model.image_size

    with torch.amp.autocast(device_type=device.type, enabled=False):
        if model_type == "ccigpt":
            # ---- coarse 路径（R-only，独立 bitstream）----
            x_c_full = F.adaptive_avg_pool2d(x, model.coarse_size)
            x_c = x_c_full[:, :model.coarse.in_channels]
            coarse_tokens = model.coarse._tokenize(x_c)           # (1, N_c)
            c_bits, c_ideal = _encode_sequence(
                model.coarse, coarse_tokens, None, device, "coarse", log_every)
            coarse_dec = _decode_sequence(
                model.coarse, c_bits, coarse_tokens.shape[1], None, device,
                "coarse", log_every)
            assert torch.equal(coarse_dec, coarse_tokens), "coarse token 解码不一致！"

            # ---- 从解出的 coarse token 重建 fine 条件 ctx（decoder 视角）----
            coarse_ctx = model.ctx_alpha * model._compute_coarse_ctx(coarse_dec)

            # ---- fine 路径（条件于 coarse_ctx）----
            fine_tokens = model.fine._tokenize(x)
            f_bits, f_ideal = _encode_sequence(
                model.fine, fine_tokens, coarse_ctx, device, "fine", log_every)
            fine_dec = _decode_sequence(
                model.fine, f_bits, fine_tokens.shape[1], coarse_ctx, device,
                "fine", log_every)
            assert torch.equal(fine_dec, fine_tokens), "fine token 解码不一致！"

            recon = _detokenize(fine_dec, H, C)
            N_f = model.fine.seq_len
            total_bits = len(c_bits) + len(f_bits)
            ideal_bits = c_ideal + f_ideal
            parts = {"coarse_bits": len(c_bits), "fine_bits": len(f_bits)}
        else:
            tokens = model._tokenize(x)
            bits, ideal_bits = _encode_sequence(model, tokens, None, device,
                                                "igpt", log_every)
            dec = _decode_sequence(model, bits, tokens.shape[1], None, device,
                                   "igpt", log_every)
            assert torch.equal(dec, tokens), "token 解码不一致！"
            recon = _detokenize(dec, H, C)
            N_f = model.seq_len
            total_bits = len(bits)
            parts = {}

        # 模型 teacher-forced 参考 bpd（fp32）
        ref = model(x)
        ref_bpd = float(ref["bpd"].item())

    orig = (x.clamp(0, 1) * 255).round().to(torch.uint8)[0]
    pixel_exact = bool(torch.equal(recon, orig))

    achieved_bpd = total_bits / N_f
    ideal_bpd = ideal_bits / N_f
    return {
        "pixel_exact": pixel_exact,
        "total_bits": total_bits,
        "achieved_bpd": achieved_bpd,
        "ideal_bpd": ideal_bpd,
        "ref_bpd": ref_bpd,
        "N_f": N_f,
        **parts,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=str)
    ap.add_argument("--checkpoint", type=str)
    ap.add_argument("--num_images", type=int, default=1)
    ap.add_argument("--log_every", type=int, default=512,
                    help="每多少步打印进度（0=静默）")
    ap.add_argument("--self_test", action="store_true",
                    help="仅跑 coder 合成 roundtrip，无需 GPU/ckpt")
    args = ap.parse_args()

    if args.self_test:
        _self_test()
        return

    assert args.config and args.checkpoint, "需要 --config 和 --checkpoint（或用 --self_test）"

    import torch
    import yaml
    from scripts.evaluate import _load_dataset

    # 先跑一次 self_test，确保 coder 本身没问题，再进昂贵的模型循环
    print("== 预检：coder self_test ==")
    _self_test()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_type = _build_model(config, args.checkpoint, device)
    print(f"== 模型加载完成 (type={model_type}, device={device}) ==")

    dataset, dataset_name = _load_dataset(config)
    print(f"== 数据集 {dataset_name}，验证前 {args.num_images} 张 ==\n")

    n_pass = 0
    for idx in range(args.num_images):
        item = dataset[idx]
        x = item[0] if isinstance(item, (tuple, list)) else item
        x = x.unsqueeze(0)
        print(f"--- image {idx} ---")
        r = _verify_image(model, model_type, x, device, args.log_every)
        tag = "✅ bit-identical" if r["pixel_exact"] else "❌ MISMATCH"
        print(f"  {tag}")
        print(f"  码长: {r['total_bits']} bit  ({r['total_bits']/8:.0f} byte)"
              + (f"  [coarse {r['coarse_bits']} + fine {r['fine_bits']}]"
                 if "coarse_bits" in r else ""))
        print(f"  achieved bpd = {r['achieved_bpd']:.4f}  "
              f"(理论 NLL bpd {r['ideal_bpd']:.4f}, "
              f"模型 teacher-forced bpd {r['ref_bpd']:.4f})")
        print(f"  量化+收尾 overhead vs NLL: "
              f"{100*(r['achieved_bpd']-r['ideal_bpd'])/r['ideal_bpd']:.2f}%\n")
        n_pass += int(r["pixel_exact"])

    print(f"== 结果：{n_pass}/{args.num_images} 张 bit-identical 还原 ==")
    if n_pass != args.num_images:
        sys.exit(1)


if __name__ == "__main__":
    main()
