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
  POST /api/predict     — 上传图片 → 返回 bpd / 双尺度 CE / 热力图
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
from fastapi import FastAPI, UploadFile, File, HTTPException
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

    img = img.resize((32, 32), Image.Resampling.BILINEAR)
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
