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
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.mdlic.codec.arithmetic import (
    ArithmeticEncoder, ArithmeticDecoder, build_cumfreq, pack_bits, unpack_bits, FREQ_TOTAL,
)

# 均匀 256-way 先验（token 0 用），常量，encode/decode 共用
_UNIFORM_CUM = list(range(0, FREQ_TOTAL + 1, FREQ_TOTAL // 256))

# .bin 容器格式（self-contained，可独立解码）：
#   magic "MDLC" (4B) | version u8 | dual u8 (1=双尺度 0=单尺度) |
#   H u8 | C u8 | coarse_nbits u32-BE | fine_nbits u32-BE |
#   coarse packed bytes | fine packed bytes
# n_bits 存真实 bit 数（pack 末尾补 0 到字节对齐，unpack 须知 n_bits 丢 padding）。
_MAGIC = b"MDLC"
_VERSION = 1
_HEADER = ">4sBBBBII"   # magic, ver, dual, H, C, coarse_nbits, fine_nbits
_HEADER_SIZE = struct.calcsize(_HEADER)


def _build_container_bytes(dual, H, C, c_bits, f_bits):
    """coarse/fine bit 列表 → 自包含 MDLC .bin 字节串（不落盘）。

    落盘（_write_container）与 demo 内存编码（/api/encode）共用同一打包逻辑，
    保证两条路径产出的容器逐字节一致。
    """
    c_bytes = pack_bits(c_bits) if c_bits else b""
    f_bytes = pack_bits(f_bits)
    header = struct.pack(_HEADER, _MAGIC, _VERSION, 1 if dual else 0,
                         H, C, len(c_bits), len(f_bits))
    return header + c_bytes + f_bytes


def _read_container_bytes(blob):
    """MDLC .bin 字节串 → (dual, H, C, c_bits, f_bits)。校验 magic/version。

    读文件（_read_container）与 demo 上传解码（/api/decode）共用，保证两条
    路径对同一字节串解出完全相同的 bit 列表。
    """
    if len(blob) < _HEADER_SIZE:
        # 截断/过短：先抛 ValueError，避免 struct.unpack 抛 struct.error
        # （struct.error 不是 ValueError 子类，会漏过调用方的 except ValueError → 500）
        raise ValueError(
            f"容器过短：{len(blob)} 字节 < header {_HEADER_SIZE} 字节")
    magic, ver, dual, H, C, c_nbits, f_nbits = struct.unpack(
        _HEADER, blob[:_HEADER_SIZE])
    if magic != _MAGIC:
        raise ValueError(f"非 MDLC 容器（magic={magic!r}）")
    if ver != _VERSION:
        raise ValueError(f"版本不匹配（{ver} != {_VERSION}）")
    off = _HEADER_SIZE
    c_nbytes = (c_nbits + 7) // 8
    f_nbytes = (f_nbits + 7) // 8
    c_bits = unpack_bits(blob[off:off + c_nbytes], c_nbits) if c_nbits else []
    off += c_nbytes
    f_bits = unpack_bits(blob[off:off + f_nbytes], f_nbits)
    return bool(dual), H, C, c_bits, f_bits


def _write_container(path, dual, H, C, c_bits, f_bits):
    """把 coarse/fine bit 列表写成自包含 .bin。返回落盘字节数。"""
    blob = _build_container_bytes(dual, H, C, c_bits, f_bits)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(blob)
    return len(blob)


def _read_container(path):
    """读回 .bin → (dual, H, C, c_bits, f_bits)。校验 magic/version。"""
    with open(path, "rb") as fh:
        blob = fh.read()
    return _read_container_bytes(blob)


def _parse_container_meta(blob):
    """MDLC .bin 字节串 → 结构元信息 dict（不解模型，纯文件级）。

    证明 .bin 自包含：仅凭文件就能读出尺寸/码长/bpd。CLI 的 --inspect 与
    demo 的 /api/inspect 共用，保证两边算出的 bpd / 自洽校验完全同口径。
    """
    total = len(blob)
    if total < _HEADER_SIZE:
        raise ValueError(f"文件太小（{total}B < header {_HEADER_SIZE}B），非 MDLC 容器")
    magic, ver, dual, H, C, c_nbits, f_nbits = struct.unpack(
        _HEADER, blob[:_HEADER_SIZE])
    if magic != _MAGIC:
        raise ValueError(f"非 MDLC 容器（magic={magic!r}）")

    c_nbytes = (c_nbits + 7) // 8
    f_nbytes = (f_nbits + 7) // 8
    payload = total - _HEADER_SIZE
    total_bits = c_nbits + f_nbits
    n_subpix = H * H * C
    bpd = total_bits / n_subpix if n_subpix else 0.0
    # payload 字节数应正好等于两段对齐字节之和（损坏检测）
    self_consistent = (payload == c_nbytes + f_nbytes)

    return {
        "total_bytes": total,
        "header_size": _HEADER_SIZE,
        "header_hex": " ".join(f"{b:02x}" for b in blob[:_HEADER_SIZE]),
        "magic": magic.decode("ascii", "replace"),
        "version": ver,
        "dual": bool(dual),
        "H": H, "C": C,
        "n_subpix": n_subpix,
        "coarse_nbits": c_nbits, "coarse_nbytes": c_nbytes,
        "fine_nbits": f_nbits, "fine_nbytes": f_nbytes,
        "payload_bytes": payload,
        "total_bits": total_bits,
        "bpd": bpd,
        "self_consistent": self_consistent,
    }


def _inspect_container(path):
    """只读不解模型：打印 MDLC 容器结构 + 头部 hex + 从文件独立算出的 bpd。

    证明 .bin 是自包含的 —— 仅凭文件（无 ckpt）就能读出尺寸/码长/bpd。
    （真正解回图像仍需模型逐 token forward，见 roundtrip 主流程。）
    """
    with open(path, "rb") as fh:
        blob = fh.read()
    m = _parse_container_meta(blob)

    print(f"== MDLC 容器: {path} ==")
    print(f"  文件大小      : {m['total_bytes']} byte")
    print(f"  header (16B)  : {m['header_hex']}")
    print(f"  magic / ver   : {m['magic']} / v{m['version']}")
    print(f"  尺度          : {'双尺度 (coarse+fine)' if m['dual'] else '单尺度 (igpt)'}")
    print(f"  图像          : {m['H']}×{m['H']}×{m['C']}  ({m['n_subpix']} 子像素)")
    if m["dual"]:
        print(f"  coarse        : {m['coarse_nbits']} bit  ({m['coarse_nbytes']} byte)")
        print(f"  fine          : {m['fine_nbits']} bit  ({m['fine_nbytes']} byte)")
    print(f"  payload       : {m['payload_bytes']} byte  "
          f"(= coarse {m['coarse_nbytes']} + fine {m['fine_nbytes']})")
    print(f"  header 开销    : {m['header_size']} byte  "
          f"({100*m['header_size']/m['total_bytes']:.1f}% of 文件)")
    print(f"  码长          : {m['total_bits']} bit")
    print(f"  achieved bpd  : {m['bpd']:.4f}  (= {m['total_bits']} bit / {m['n_subpix']} 子像素)")
    ok = m["self_consistent"]
    print(f"  自洽校验      : {'✅ payload 字节数与 header 声明一致' if ok else '❌ payload 与 header 不符（文件可能损坏）'}")
    return ok




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


def _encode_sequence_iter(igpt, tokens, coarse_ctx, device):
    """逐步算术编码生成器：每编码一个 token yield (m, T-1) 进度，
    最终 `return` 出 bit 列表 (list[int])。

    与 _encode_sequence 同一逐 token 路径（_logits_from_tokens），仅多吐进度，
    供 demo 流式端点 /api/encode 用，绕开 AutoDL 反代 idle 超时。
    关键：与 _decode_sequence_iter 用**完全相同**的 buffer 构造 + _logits_from_tokens，
    故 encode 写的 bits 与 decode 逐 token 读所需分布逐位相同 → bit-exact 可解。
    """
    import torch
    T = tokens.shape[1]
    seq_in = igpt.seq_len - 1
    enc = ArithmeticEncoder()
    enc.encode(int(tokens[0, 0].item()), _UNIFORM_CUM)   # token 0：均匀先验
    buf = torch.zeros((1, seq_in), dtype=torch.long, device=device)
    for m in range(1, T):
        buf[0, m - 1] = tokens[0, m - 1]
        logits = _logits_from_tokens(igpt, buf, coarse_ctx, m - 1)
        p = _probs(logits)
        enc.encode(int(tokens[0, m].item()), build_cumfreq(p))
        yield m, T - 1
    return enc.finish()


def _decode_sequence_iter(igpt, bits, T, coarse_ctx, device):
    """逐步算术解码生成器：每解出一个 token yield (m, T-1) 进度，
    最终 `return` 出 token 序列 (1, T)。每步 logits 必须与 encode 端逐位相同。

    CLI（_decode_sequence）与 demo 流式端点（/api/decode）共用此单一解码逻辑，
    各自决定进度怎么消费（打印 / 推送），保证两条路径不漂移。
    """
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
        yield m, T - 1
    return out


def _decode_sequence(igpt, bits, T, coarse_ctx, device, tag, log_every):
    """逐步算术解码出 token 序列 (1, T)，CLI 进度打印版。"""
    gen = _decode_sequence_iter(igpt, bits, T, coarse_ctx, device)
    out = None
    try:
        while True:
            m, total = next(gen)
            if log_every and m % log_every == 0:
                print(f"    [{tag} decode] {m}/{total}")
    except StopIteration as stop:
        out = stop.value
    return out


def _detokenize(tokens, image_size, channels):
    """(1, H*W*C) pixel-first long token → (C,H,W) uint8 tensor（_tokenize 逆）。"""
    import torch
    t = tokens.view(1, image_size, image_size, channels)
    return t.permute(0, 3, 1, 2).contiguous()[0].to(torch.uint8)


def _verify_image(model, model_type, x, device, log_every, dump_path=None):
    """对单张图 (1,C,H,W) float[0,1] 做 encode→decode→断言 bit-identical。

    dump_path 非空时，额外把 bitstream 写成自包含 .bin，再读回断言 bit 一致 ——
    证据链 图像→bits→文件→bits→图像 全程闭合（读回的 bits 即上面已证可解码的同一串，
    故无需再跑一次昂贵 forward）。
    """
    import torch

    x = x.to(device).clamp(0, 1).float()
    C = model.in_channels       # CCIGPT 与 IGPT 都暴露 in_channels / image_size
    H = model.image_size

    with torch.amp.autocast(device_type=device.type, enabled=False):
        if model_type == "ccigpt":
            # ---- coarse 路径（R-only，独立 bitstream）----
            x_c = model._coarse_input(x)                          # bit-exact coarse 源头（单点推导）
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
            _dump = (True, c_bits, f_bits)   # 落盘用：(dual, coarse_bits, fine_bits)
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
            _dump = (False, [], bits)        # 单尺度：无 coarse，fine 位即整串

        # 模型 teacher-forced 参考 bpd（fp32）
        ref = model(x)
        ref_bpd = float(ref["bpd"].item())

    orig = (x.clamp(0, 1) * 255).round().to(torch.uint8)[0]
    pixel_exact = bool(torch.equal(recon, orig))

    # ---- bitstream 落盘（可选）：写自包含 .bin，读回断言 bit 一致 ----
    # 读回的 bits 即上面已证可解码的同一串，故文件链路无需再跑 forward。
    dump_info = {}
    if dump_path is not None:
        dual, c_bits_d, f_bits_d = _dump
        file_bytes = _write_container(dump_path, dual, H, C, c_bits_d, f_bits_d)
        d_dual, d_H, d_C, d_c, d_f = _read_container(dump_path)
        file_ok = (d_dual == dual and d_H == H and d_C == C
                   and d_c == c_bits_d and d_f == f_bits_d)
        dump_info = {"dump_path": dump_path, "file_bytes": file_bytes,
                     "file_roundtrip_ok": bool(file_ok)}

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
        **dump_info,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=str)
    ap.add_argument("--checkpoint", type=str)
    ap.add_argument("--num_images", type=int, default=1)
    ap.add_argument("--log_every", type=int, default=512,
                    help="每多少步打印进度（0=静默）")
    ap.add_argument("--dump_dir", type=str, default=None,
                    help="把每张图的 bitstream 写成自包含 .bin 容器到该目录"
                         "（图像→bits→文件→bits→图像，读回断言 bit 一致）")
    ap.add_argument("--self_test", action="store_true",
                    help="仅跑 coder 合成 roundtrip，无需 GPU/ckpt")
    ap.add_argument("--inspect", type=str, default=None,
                    help="只读模式：解析一个 .bin 容器打印结构/码长/bpd，无需 GPU/ckpt/模型")
    args = ap.parse_args()

    if args.inspect:
        _inspect_container(args.inspect)
        return

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
        dump_path = (os.path.join(args.dump_dir, f"img{idx}.bin")
                     if args.dump_dir else None)
        r = _verify_image(model, model_type, x, device, args.log_every,
                          dump_path=dump_path)
        tag = "✅ bit-identical" if r["pixel_exact"] else "❌ MISMATCH"
        print(f"  {tag}")
        print(f"  码长: {r['total_bits']} bit  ({r['total_bits']/8:.0f} byte)"
              + (f"  [coarse {r['coarse_bits']} + fine {r['fine_bits']}]"
                 if "coarse_bits" in r else ""))
        print(f"  achieved bpd = {r['achieved_bpd']:.4f}  "
              f"(理论 NLL bpd {r['ideal_bpd']:.4f}, "
              f"模型 teacher-forced bpd {r['ref_bpd']:.4f})")
        print(f"  量化+收尾 overhead vs NLL: "
              f"{100*(r['achieved_bpd']-r['ideal_bpd'])/r['ideal_bpd']:.2f}%")
        if "dump_path" in r:
            fok = "✅" if r["file_roundtrip_ok"] else "❌"
            print(f"  bitstream → {r['dump_path']}  ({r['file_bytes']} byte 落盘)  "
                  f"文件读回 bit 一致 {fok}")
        print()
        n_pass += int(r["pixel_exact"])

    print(f"== 结果：{n_pass}/{args.num_images} 张 bit-identical 还原 ==")
    if n_pass != args.num_images:
        sys.exit(1)


if __name__ == "__main__":
    main()
