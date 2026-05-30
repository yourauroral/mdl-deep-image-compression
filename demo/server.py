#!/usr/bin/env python3
"""
Demo 可视化后端 — FastAPI + 静态文件。

启动:
  pip install fastapi uvicorn python-multipart
  cd demo && uvicorn server:app --reload --port 8000

端点:
  GET  /                — 前端页面
  GET  /api/metrics     — bits/dim 对比表数据
  GET  /api/probe       — Linear Probe 各层准确率
  GET  /api/kernels     — Triton Kernel 性能数据
  GET  /api/scales      — CC-iGPT coarse/fine token 分配
  GET  /api/ood         — OOD typicality AUROC 表（下游 §6.2，AutoDL 回填）
  GET  /api/transfer    — 跨数据集 bpd 泛化（下游 §6.3，AutoDL 回填）
  POST /api/predict     — 上传图片 → 返回 bpd / 双尺度 CE / 热力图
  POST /api/lossless    — 上传图片 → 真实算术编解码 roundtrip（下游 §6.5/6.6）
  POST /api/complete    — 上传图片 → AR 补全下半（下游 §6.4，实时采样，~20–40s）
"""

import json
import os
import sys
import io
import math
import base64
import threading

import numpy as np
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# 提前固定 matplotlib backend 到 Agg：必须在 import pyplot 前，且只能在主线程
# 完成一次；放在请求 handler 内会与并发请求争用全局 figure manager。
import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

# 项目根目录
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# 手写算术编解码器（纯 Python，无 torch 依赖）——/api/lossless 真实可解性 demo 用
from src.mdlic.codec.arithmetic import (
    ArithmeticEncoder, ArithmeticDecoder, build_cumfreq, FREQ_TOTAL,
)

DATA_DIR = Path(__file__).resolve().parent / "data"

# 上传图片大小上限（10 MB）和允许的 MIME 类型，防止 /api/predict 被恶意大文件
# 或非图片文件耗尽内存。
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_CONTENT_TYPES = {
    "image/jpeg", "image/png", "image/webp", "image/gif", "image/bmp",
}

app = FastAPI(title="MDL Deep Image Compression Demo")

# 显式 CORS 白名单：仅允许本地开发用 origin。部署到公网时按需扩充。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# 静态文件
app.mount("/static", StaticFiles(directory=Path(__file__).resolve().parent / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(Path(__file__).resolve().parent / "static" / "index.html")


def _load_json(name: str) -> dict:
    path = DATA_DIR / name
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"{name} not found. Run data generation scripts first.")
    with open(path) as f:
        return json.load(f)


# JSON 端点用 sync def：FastAPI 自动放进 threadpool，避免 open()/json.load()
# 同步 IO 阻塞 event loop（前端 4 个端点并发拉取时尤为重要）。
@app.get("/api/metrics")
def get_metrics():
    return _load_json("metrics.json")


@app.get("/api/probe")
def get_probe():
    return _load_json("probe.json")


@app.get("/api/kernels")
def get_kernels():
    return _load_json("kernels.json")


@app.get("/api/scales")
def get_scales():
    return _load_json("scales.json")


# OOD / transfer 是下游任务结果，由 AutoDL 跑 scripts/ood_detect.py /
# scripts/evaluate.py --dataset_override 时用 --json_out 回填到 demo/data/。
# 未回填前文件里是 generated=null 的占位，前端据此显示"待 AutoDL 跑"。
# 缺文件不报 404（与 metrics 等不同）：占位 JSON 已 checkin，正常情况恒存在；
# 万一被删，回退一个 pending 壳让前端面板优雅留白而非整页报错。
@app.get("/api/ood")
def get_ood():
    path = DATA_DIR / "ood.json"
    if not path.exists():
        return JSONResponse({"generated": None, "ood": []})
    with open(path) as f:
        return json.load(f)


@app.get("/api/transfer")
def get_transfer():
    path = DATA_DIR / "transfer.json"
    if not path.exists():
        return JSONResponse({"generated": None, "datasets": []})
    with open(path) as f:
        return json.load(f)


def _read_upload_to_tensor(file: UploadFile, size: int = 32):
    """上传图片 → 校验 → resize → (1,3,size,size) float[0,1] tensor + PIL Image。

    /api/predict 与 /api/lossless 共用，保证两条路径对上传走完全一致的
    校验 / 解码 / 缩放，避免 bpd 与可解性 demo 因预处理口径漂移而对不上。
    """
    try:
        from PIL import Image
        from torchvision import transforms
    except ImportError:
        raise HTTPException(status_code=500, detail="torch/torchvision not installed")

    if file.content_type and file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail=f"Unsupported media type: {file.content_type}")
    contents = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large (>{MAX_UPLOAD_BYTES // 1024 // 1024} MB)")
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    img = img.resize((size, size), Image.Resampling.BILINEAR)
    x = transforms.ToTensor()(img).unsqueeze(0)   # (1, 3, size, size)
    return x, img


@app.post("/api/predict")
def predict(file: UploadFile = File(...)):
    """
    上传一张图片，返回:
      - bpd: 整体 bits/dim（CC-iGPT 为 bpd_total，iGPT 由 CE 推算）
      - heatmap: base64 编码的 fine 分支 BPP 热力图 PNG
                 （单位 bits/pixel = bpd × C；CC-iGPT 仅画 fine 段，
                 coarse 8×8 的 ~2% overhead 不在该图上）

    注：sync def 形式让 FastAPI 自动放进 threadpool，避免 GPU forward 阻塞
    event loop（async def 里直接调 model(x) 会卡住其它并发请求）。
    """
    try:
        import torch
    except ImportError:
        raise HTTPException(status_code=500, detail="torch/torchvision not installed")

    x, _ = _read_upload_to_tensor(file, size=32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_type = _get_cached_model(device)
    if model is None:
        raise HTTPException(status_code=503, detail="No checkpoint available. Place a checkpoint in experiments/*/checkpoints/best.pth")

    x = x.to(device)
    model.eval()
    with torch.no_grad():
        out = model(x)

    ce_loss = out["ce_loss"].item()
    if "bpd" in out and out["bpd"] is not None:
        bpd = out["bpd"].item()
    else:
        # iGPT: CE 是 per-token nats，bits/dim = CE / ln2 （已按 H·W·C 归一化）
        bpd = ce_loss / math.log(2)

    # CC-iGPT 的 ce_loss 仅含 fine 分支 CE，与 bpd_total（含 coarse overhead）
    # 在尺度上不对应；前端"CE Loss"卡片若直接显示 ce_loss 会误导观感。
    # 这里把 ce_coarse / ce_fine / α 也一起返回，前端按 model_type 分别渲染。
    extras = {}
    if "ce_loss_coarse" in out and out["ce_loss_coarse"] is not None:
        extras["ce_coarse"] = round(out["ce_loss_coarse"].item(), 4)
        extras["ce_fine"] = round(out["ce_loss_fine"].item(), 4)
        if "ctx_alpha" in out and out["ctx_alpha"] is not None:
            extras["ctx_alpha"] = round(out["ctx_alpha"].item(), 4)

    # Per-position 热力图：对 fine 分支（CC-iGPT）或单尺度 GPT（iGPT）的
    # logits 计算 per-token CE，画 32×32 bits/pixel 热力图。
    heatmap_b64 = None
    if out.get("logits") is not None:
        heatmap_b64 = _make_heatmap_b64(model, x, out["logits"])

    return JSONResponse({
        "bpd": round(bpd, 4),
        "ce_loss": round(ce_loss, 4),
        "model_type": model_type,
        "heatmap": heatmap_b64,
        **extras,
    })


# 均匀 256-way 先验（token 0 不被模型预测，用均匀分布编码 = 8 bit），
# 与 scripts/verify_lossless.py 的 _UNIFORM_CUM 同口径。
_UNIFORM_CUM = list(range(0, FREQ_TOTAL + 1, FREQ_TOTAL // 256))


def _probs_from_logits(logits_seq):
    """(1, T-1, V) logits → list[list[float]] 概率表（fp64 softmax，确定性）。"""
    import torch
    return torch.softmax(logits_seq.double(), dim=-1)[0].tolist()


def _roundtrip_tokens(prob_table, tokens_list, first_cum):
    """对一段 token 做真实算术编/解码 roundtrip。

    prob_table: list[V] × (T-1)，位置 i 的分布预测 token[i+1]。
    tokens_list: 长度 T 的真实 token。first_cum: token0 的累积频数表（均匀先验）。

    返回 (bits:list[int], decoded:list[int])。encode 与 decode 用**同一张** cumfreq
    表（先 build 一次缓存），保证逐位可逆 —— 这是真实可解性的核心。
    """
    cum_tables = [build_cumfreq(p) for p in prob_table]   # 每位置一张，复用

    enc = ArithmeticEncoder()
    enc.encode(tokens_list[0], first_cum)
    for i, sym in enumerate(tokens_list[1:]):
        enc.encode(sym, cum_tables[i])
    bits = enc.finish()

    dec = ArithmeticDecoder(bits)
    decoded = [dec.decode(first_cum)]
    for i in range(len(tokens_list) - 1):
        decoded.append(dec.decode(cum_tables[i]))
    return bits, decoded


def _b64_png(arr, scale=4):
    """(H,W,C) uint8 numpy → base64 PNG，nearest 放大 scale 倍便于肉眼看清 32×32。

    /api/lossless 与 /api/complete 共用，保证两条路径出图口径一致。
    """
    from PIL import Image
    im = Image.fromarray(arr).resize((arr.shape[1] * scale, arr.shape[0] * scale),
                                     Image.Resampling.NEAREST)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


@app.post("/api/lossless")
def lossless(file: UploadFile = File(...)):
    """真实可解性 demo：上传图 → 算术编码出真实 bitstream → 解码 → 逐像素比对。

    与 scripts/verify_lossless.py 同一套手写 WNC 算术编解码，但为了浏览器交互
    （秒级响应、且不与 IN64 训练抢 GPU），用**一次 teacher-forced forward** 拿到
    每个位置的条件分布，而非 decode 端逐 token 重跑 T 次完整 forward。

    这不偷工：模型 causal，位置 i 的分布只依赖 token[0..i]，与真实 decoder 在已正确
    解出前缀时算出的分布**逐位相同**（归纳法）。所以 bitstream 是真实可逆的，仅省了
    decode 侧的 T 次 forward（纯加速）。严格逐步解码的版本在 verify_lossless.py。

    返回 orig/recon 的 base64 PNG、是否 bit-identical、neural 码长 (byte/bpd)、
    以及同图 PNG/WebP 无损字节数做对比。
    """
    try:
        import torch
        import torch.nn.functional as F
        from PIL import Image
    except ImportError:
        raise HTTPException(status_code=500, detail="torch/torchvision not installed")

    x, _ = _read_upload_to_tensor(file, size=32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_type = _get_cached_model(device)
    if model is None:
        raise HTTPException(status_code=503, detail="No checkpoint available. Place a checkpoint in experiments/*/checkpoints/best.pth")

    x = x.to(device).clamp(0, 1).float()
    model.eval()

    # 全程关 autocast + fp32，与 verify_lossless / cc_igpt bit-exact ctx 路径一致
    with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=False):
        if model_type == "ccigpt":
            # ---- coarse（独立 bitstream）----
            x_c = F.adaptive_avg_pool2d(x, model.coarse_size)[:, :model.coarse.in_channels]
            coarse_tokens = model.coarse._tokenize(x_c)[0].tolist()
            out_c = model.coarse(x_c)
            c_probs = _probs_from_logits(out_c["logits"])
            c_bits, c_dec = _roundtrip_tokens(c_probs, coarse_tokens, _UNIFORM_CUM)
            coarse_ok = (c_dec == coarse_tokens)

            # ---- 从 DECODED coarse token 重建 fine 条件 ctx（decoder 视角，不作弊）----
            coarse_dec_t = torch.tensor(c_dec, dtype=torch.long, device=device).view(1, -1)
            coarse_ctx = model.ctx_alpha * model._compute_coarse_ctx(coarse_dec_t)

            # ---- fine（条件于 coarse_ctx）----
            fine_tokens = model.fine._tokenize(x)[0].tolist()
            out_f = model.fine(x, coarse_ctx=coarse_ctx)
            f_probs = _probs_from_logits(out_f["logits"])
            f_bits, f_dec = _roundtrip_tokens(f_probs, fine_tokens, _UNIFORM_CUM)
            fine_ok = (f_dec == fine_tokens)

            recon_tokens = f_dec
            H = model.fine.image_size
            C = model.fine.in_channels
            N_f = model.fine.seq_len
            # 两段独立 bitstream，各自字节对齐 → 真实落盘字节数
            neural_bytes = math.ceil(len(c_bits) / 8) + math.ceil(len(f_bits) / 8)
            total_bits = len(c_bits) + len(f_bits)
            ok_decode = coarse_ok and fine_ok
            parts = {"coarse_bits": len(c_bits), "fine_bits": len(f_bits)}
        else:
            tokens = model._tokenize(x)[0].tolist()
            out = model(x)
            probs = _probs_from_logits(out["logits"])
            bits, dec = _roundtrip_tokens(probs, tokens, _UNIFORM_CUM)
            ok_decode = (dec == tokens)
            recon_tokens = dec
            H = model.image_size
            C = model.in_channels
            N_f = model.seq_len
            neural_bytes = math.ceil(len(bits) / 8)
            total_bits = len(bits)
            parts = {}

    # ---- 重建图（从 DECODED token，pixel-first 逆 tokenize）----
    recon = np.array(recon_tokens, dtype=np.uint8).reshape(H, H, C)      # (H,W,C)
    orig = (x.clamp(0, 1) * 255).round().to(torch.uint8)[0].permute(1, 2, 0).cpu().numpy()
    pixel_exact = bool(ok_decode and np.array_equal(recon, orig))

    # ---- 传统无损对照（同一张 32×32 图）----
    def _fmt_bytes(arr, fmt, **kw):
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format=fmt, **kw)
        return buf.tell()

    png_bytes = _fmt_bytes(orig, "PNG", optimize=True)
    try:
        webp_bytes = _fmt_bytes(orig, "WEBP", lossless=True)
    except Exception:
        webp_bytes = None

    achieved_bpd = total_bits / N_f
    n_pixels = H * H * C

    return JSONResponse({
        "model_type": model_type,
        "pixel_exact": pixel_exact,
        "orig_png": _b64_png(orig),
        "recon_png": _b64_png(recon),
        "neural_bytes": neural_bytes,
        "neural_bits": total_bits,
        "achieved_bpd": round(achieved_bpd, 4),
        "png_bytes": png_bytes,
        "webp_bytes": webp_bytes,
        "n_subpixels": n_pixels,
        # 传统格式 bpd = bytes×8 / 子像素数，与 neural achieved_bpd 同口径可比
        "png_bpd": round(png_bytes * 8 / n_pixels, 4),
        "webp_bpd": round(webp_bytes * 8 / n_pixels, 4) if webp_bytes else None,
        **parts,
    })


# 补全请求采样上限：keep_frac 决定要采样多少 token（越小越慢）。模型无 KV-cache，
# 每 token 一次完整 forward；CIFAR 32×32×3=3072 token，keep_frac=0.6 → 采 ~1229 步
# ≈ 20–40s（取决于 GPU）。设硬下限防止 keep_frac→0 把单请求拖到几分钟、长时间占住
# 一个 threadpool worker（且与 IN64 训练抢算力）。
_COMPLETE_MIN_KEEP_FRAC = 0.3


@app.post("/api/complete")
def complete(
    file: UploadFile = File(...),
    keep_frac: float = Form(0.6),
    temperature: float = Form(1.0),
    top_k: int = Form(100),
):
    """图像补全 demo（下游 §6.4）：上传图 → 保留前 keep_frac 的 raster token（≈上半）
    → AR 续采样补全下半 → 返回 原图 / 已知上半(灰=待补) / 补全 三张图。

    与 scripts/complete_image.py 同一套 per-step forward（无 KV-cache，causal mask
    保证 0 后缀不泄漏）。CC-iGPT 的 coarse ctx 由**整图**缩略图算 —— 故语义是"低分
    缩略图 + 上半真实像素 → 补下半"，coarse 是显式 side-channel（与压缩时独立 bitstream
    同源），非偷看答案。详见 complete_image.py docstring 与 runbook §6.4。

    sync def → FastAPI 放进 threadpool，GPU 采样不阻塞 event loop（但单请求会占住
    一个 worker ~数十秒，且与 IN64 训练共享 GPU；属预期，demo 单用户场景可接受）。
    """
    try:
        import torch
    except ImportError:
        raise HTTPException(status_code=500, detail="torch/torchvision not installed")

    # 入参夹紧：keep_frac 下限防超长采样；temperature/top_k 限合理域
    try:
        keep_frac = float(keep_frac)
        temperature = float(temperature)
        top_k = int(top_k)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="keep_frac/temperature/top_k 需为数值")
    if not (_COMPLETE_MIN_KEEP_FRAC <= keep_frac <= 0.95):
        raise HTTPException(
            status_code=400,
            detail=f"keep_frac 需 ∈ [{_COMPLETE_MIN_KEEP_FRAC}, 0.95]（越小采样越久，下限防超时）",
        )
    if not (0.0 <= temperature <= 2.0):
        raise HTTPException(status_code=400, detail="temperature 需 ∈ [0, 2]")
    top_k = max(0, min(top_k, 256))

    x, _ = _read_upload_to_tensor(file, size=32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_type = _get_cached_model(device)
    if model is None:
        raise HTTPException(status_code=503, detail="No checkpoint available. Place a checkpoint in experiments/*/checkpoints/best.pth")

    # 复用 scripts/complete_image.py 的逐步采样实现（与 runbook §6.4 完全同源），
    # 避免在 demo 侧重写一份采样逻辑导致两条路径漂移。
    from scripts.complete_image import _complete_one

    model.eval()
    with torch.no_grad():
        o, m, c = _complete_one(model, model_type, x, keep_frac, temperature, top_k, device)

    # (C,H,W) uint8 → (H,W,C) numpy → base64 PNG
    def _chw_to_png(t):
        return _b64_png(t.permute(1, 2, 0).contiguous().numpy())

    keep_pct = round(keep_frac * 100, 1)
    return JSONResponse({
        "model_type": model_type,
        "orig_png": _chw_to_png(o),
        "masked_png": _chw_to_png(m),
        "completed_png": _chw_to_png(c),
        "keep_frac": keep_frac,
        "keep_pct": keep_pct,
        "temperature": temperature,
        "top_k": top_k,
    })


_MODEL_CACHE = {"model": None, "type": None}
_MODEL_CACHE_LOCK = threading.Lock()


def _get_cached_model(device):
    """线程安全的延迟加载：double-checked locking 防止并发首请求重复 torch.load
    同一份 ckpt 造成显存峰值翻倍。

    返回 (model, model_type)：model_type ∈ {"ccigpt", "igpt"}。
    """
    if _MODEL_CACHE["model"] is not None:
        return _MODEL_CACHE["model"], _MODEL_CACHE["type"]

    with _MODEL_CACHE_LOCK:
        if _MODEL_CACHE["model"] is not None:
            return _MODEL_CACHE["model"], _MODEL_CACHE["type"]

        import yaml
        import torch

        # 按优先级尝试主路径 ckpt：v2 深窄当前主表 (2.8296) → R-only softmax v1 历史主表 (2.9035)。
        # iGPT 单尺度 baseline 已退出主线（Phase A 历史），不再作为 fallback。
        configs_dir = ROOT / "configs"
        experiments_dir = ROOT / "experiments"

        for cfg_name in [
            "ccigpt_cifar10_s_rgb_ronly_v2.yaml",
            "ccigpt_cifar10_s_rgb_ronly.yaml",
        ]:
            cfg_path = configs_dir / cfg_name
            if not cfg_path.exists():
                continue
            with open(cfg_path) as f:
                config = yaml.safe_load(f)
            exp_name = config.get("exp_name", "")
            ckpt_path = experiments_dir / exp_name / "checkpoints" / "best.pth"
            if not ckpt_path.exists():
                continue

            mcfg = config["model"]
            model_type = mcfg.get("type")
            assert model_type == "ccigpt", (
                f"demo 主线只支持 CC-iGPT，{cfg_name} 的 model.type={model_type!r} 不匹配"
            )

            from scripts.train import _build_ccigpt_from_config
            model = _build_ccigpt_from_config(mcfg, device)

            # best.pth 是裸 state_dict（torch.save(model.state_dict())），不含
            # optimizer/epoch 等 Python 对象，可安全使用 weights_only=True
            # 杜绝 pickle RCE。若未来需加载 epoch_*.pth 这类含训练状态的 ckpt，
            # 改回 False 并保证 ckpt 来源受信。
            ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=True)
            from src.mdlic.utils import clean_state_dict
            if "model_state_dict" in ckpt:
                model.load_state_dict(clean_state_dict(ckpt["model_state_dict"]))
            else:
                model.load_state_dict(clean_state_dict(ckpt))
            model.eval()

            _MODEL_CACHE["model"] = model
            _MODEL_CACHE["type"] = model_type
            return model, model_type

        return None, None


def _make_heatmap_b64(model, x, logits):
    """根据已计算的 fine logits 生成 32×32 BPP 热力图 (bits/pixel)，返回 base64 PNG。

    支持 iGPT 与 CC-iGPT 两种模型：iGPT 直接用自身 _tokenize；CC-iGPT 用
    fine 子分支的 _tokenize（coarse 分支的 8×8 不贡献热力图，其 ~2% 比特
    overhead 作为常数底色已隐含在 bpd_total 主指标中）。

    单位 bits/pixel（沿 C 通道求和），与 /api/predict 主返回字段 bpd
    (bits/dim) 差 C 倍。
    """
    import torch
    import torch.nn.functional as F

    # iGPT 自身就是 token model；CC-iGPT 把 token model 包在 .fine 下
    token_model = getattr(model, "fine", model)
    H = W = token_model.image_size
    C = token_model.in_channels

    logits = logits.float()
    tokens = token_model._tokenize(x.clamp(0, 1))
    target = tokens[:, 1:]

    per_token_ce = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        target.reshape(-1),
        reduction="none",
    )
    bpd_vals = (per_token_ce / math.log(2)).cpu().numpy()

    seq_len = C * H * W
    full = np.zeros(seq_len)
    full[1:] = bpd_vals[:seq_len - 1]
    # pixel-first → (H, W, C) → 沿 C 求和得到 (H, W) bits/pixel
    heatmap = full.reshape(H, W, C).sum(axis=-1)

    # 用 Figure + FigureCanvasAgg 绕开 pyplot 全局 figure manager；
    # threadpool 并发请求下 plt.subplots/plt.close 共享 figure 池会互相干扰。
    fig = Figure(figsize=(4, 4))
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ax.imshow(heatmap, cmap="hot", interpolation="nearest")
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    buf = io.BytesIO()
    canvas.print_png(buf)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")
