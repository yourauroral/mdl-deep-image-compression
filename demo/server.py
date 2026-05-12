#!/usr/bin/env python3
"""
Demo 可视化后端 — FastAPI + 静态文件。

启动:
  pip install fastapi uvicorn python-multipart
  cd demo && uvicorn server:app --reload --port 8000

端点:
  GET  /                — 前端页面
  GET  /api/metrics     — BPP 对比表数据
  GET  /api/probe       — Linear Probe 各层准确率
  GET  /api/kernels     — Triton Kernel 性能数据
  GET  /api/scales      — CC-iGPT coarse/fine token 分配
  POST /api/predict     — 上传图片 → 返回 BPP 热力图 + 数值
"""

import json
import os
import sys
import io
import math
import base64

import numpy as np
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# 项目根目录
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

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
async def index():
    return FileResponse(Path(__file__).resolve().parent / "static" / "index.html")


def _load_json(name: str) -> dict:
    path = DATA_DIR / name
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"{name} not found. Run data generation scripts first.")
    with open(path) as f:
        return json.load(f)


@app.get("/api/metrics")
async def get_metrics():
    return _load_json("metrics.json")


@app.get("/api/probe")
async def get_probe():
    return _load_json("probe.json")


@app.get("/api/kernels")
async def get_kernels():
    return _load_json("kernels.json")


@app.get("/api/scales")
async def get_scales():
    return _load_json("scales.json")


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    """
    上传一张图片，返回:
      - bpp: 整体 BPP（CC-iGPT 为 BPP_total，iGPT 由 CE 推算）
      - heatmap: base64 编码的 BPP 热力图 PNG（仅 iGPT）
    """
    try:
        import torch
        from PIL import Image
        from torchvision import transforms
    except ImportError:
        raise HTTPException(status_code=500, detail="torch/torchvision not installed")

    # 输入校验：MIME 类型 + 大小上限。content_type 可被客户端伪造，所以后面还会
    # 让 PIL.Image.open 二次验证；这里只是廉价的早期拒绝。
    if file.content_type and file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail=f"Unsupported media type: {file.content_type}")
    contents = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large (>{MAX_UPLOAD_BYTES // 1024 // 1024} MB)")
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    img = img.resize((32, 32), Image.BILINEAR)
    x = transforms.ToTensor()(img).unsqueeze(0)  # (1, 3, 32, 32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_type = _get_cached_model(device)
    if model is None:
        raise HTTPException(status_code=503, detail="No checkpoint available. Place a checkpoint in experiments/*/checkpoints/best.pth")

    x = x.to(device)
    model.eval()
    with torch.no_grad():
        out = model(x)

    ce_loss = out["ce_loss"].item()
    if "bpp" in out and out["bpp"] is not None:
        bpp = out["bpp"].item()
    else:
        # iGPT: CE 是 per-token nats，bits/dim = CE / ln2 （已按 H·W·C 归一化）
        bpp = ce_loss / math.log(2)

    # CC-iGPT 的 ce_loss 仅含 fine 分支 CE，与 BPP_total（含 coarse overhead）
    # 在尺度上不对应；前端"CE Loss"卡片若直接显示 ce_loss 会误导观感。
    # 这里把 ce_coarse / ce_fine / α 也一起返回，前端按 model_type 分别渲染。
    extras = {}
    if "ce_loss_coarse" in out and out["ce_loss_coarse"] is not None:
        extras["ce_coarse"] = round(out["ce_loss_coarse"].item(), 4)
        extras["ce_fine"] = round(out["ce_loss_fine"].item(), 4)
        if "ctx_alpha" in out and out["ctx_alpha"] is not None:
            extras["ctx_alpha"] = round(out["ctx_alpha"].item(), 4)

    # 仅 iGPT 支持完整 per-position 热力图（复用同一次 forward 的 logits）
    heatmap_b64 = None
    if model_type == "igpt" and out.get("logits") is not None:
        heatmap_b64 = _make_heatmap_b64(model, x, out["logits"])

    return JSONResponse({
        "bpp": round(bpp, 4),
        "ce_loss": round(ce_loss, 4),
        "model_type": model_type,
        "heatmap": heatmap_b64,
        **extras,
    })


_MODEL_CACHE = {"model": None, "type": None}


def _get_cached_model(device):
    if _MODEL_CACHE["model"] is not None:
        return _MODEL_CACHE["model"], _MODEL_CACHE["type"]

    import yaml
    import torch

    # 扫描 configs 目录寻找可用 checkpoint
    configs_dir = ROOT / "configs"
    experiments_dir = ROOT / "experiments"

    for cfg_name in ["ccigpt_cifar10_s.yaml", "igpt_cifar10_s.yaml"]:
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
        model_type = mcfg.get("type", "igpt")

        from scripts.train import _build_model_from_config, _build_ccigpt_from_config
        if model_type == "ccigpt":
            model = _build_ccigpt_from_config(mcfg, device)
        else:
            model = _build_model_from_config(mcfg, device)

        # weights_only=False 仅在 demo 加载本地受信 checkpoint 时使用（含 epoch/optimizer
        # 等非 tensor 字段，weights_only=True 会失败）。若部署到公网或允许第三方上传
        # checkpoint，必须切换到 weights_only=True 并改造为只接收 state_dict。
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt)
        model.eval()

        _MODEL_CACHE["model"] = model
        _MODEL_CACHE["type"] = model_type
        return model, model_type

    return None, None


def _make_heatmap_b64(model, x, logits):
    """根据已计算的 logits 生成 32×32 BPP 热力图，返回 base64 PNG。

    复用 predict() 中已经做过的 forward，避免重复计算。
    """
    import torch
    import torch.nn.functional as F
    try:
        from src.mdlic.models.igpt import rgb_to_ycbcr_int
    except ImportError:
        return None

    H = W = model.image_size
    C = model.in_channels
    use_ycbcr = getattr(model, "use_ycbcr", True)
    use_subpixel_ar = getattr(model, "use_subpixel_ar", False)

    logits = logits.float()
    int_tokens = (rgb_to_ycbcr_int(x.clamp(0, 1)) if use_ycbcr
                  else (x.clamp(0, 1) * 255).round().long())
    # token 排布必须与 IGPT._tokenize 一致：subpixel AR 是 pixel-first
    # [Y0,Cb0,Cr0, Y1,Cb1,Cr1, ...]，否则是 channel-first [Y_all|Cb_all|Cr_all]。
    if use_subpixel_ar:
        tokens = int_tokens.permute(0, 2, 3, 1).reshape(1, -1)
    else:
        tokens = int_tokens.reshape(1, -1)
    target = tokens[:, 1:]

    per_token_ce = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        target.reshape(-1),
        reduction="none",
    )
    bpp_vals = (per_token_ce / math.log(2)).cpu().numpy()

    seq_len = C * H * W
    full = np.zeros(seq_len)
    full[1:] = bpp_vals[:seq_len - 1]
    if use_subpixel_ar:
        # pixel-first → (H, W, C) → 沿 C 求和得到 (H, W)
        heatmap = full.reshape(H, W, C).sum(axis=-1)
    else:
        heatmap = full.reshape(C, H, W).sum(axis=0)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(heatmap, cmap="hot", interpolation="nearest")
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")
