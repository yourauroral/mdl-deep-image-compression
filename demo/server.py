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
  POST /api/complete    — 上传图片 → AR 补全下半（下游 §6.2，实时采样，~20–40s）
  POST /api/encode      — 上传图片 → 逐 token gold 算术编码为模型绑定 MDLC .bin（流式 NDJSON）
  POST /api/inspect     — 上传 .bin → 即时解析容器结构/码长/bpd（无需 GPU/ckpt）
  POST /api/decode      — 上传 .bin → 流式逐 token 盲解码还原图像（NDJSON 进度）
"""

import json
import os
import sys
import io
import math
import base64
import gc
import hashlib
import hmac
import logging
import threading

import numpy as np
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
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
sys.path.insert(0, str(ROOT / "src"))

from mdlic.rate import num_model_predictions
from mdlic.request_limits import SlidingWindowRateLimiter
from mdlic.codec.container import (
    build_container_bytes,
    parse_container_meta,
    read_container_bytes,
    sha256_rgb_bytes,
    validate_decoded_rgb,
)
from mdlic.codec.sequential import decode_sequence_iter, encode_sequence_iter
from mdlic.completion import complete_image

DATA_DIR = Path(__file__).resolve().parent / "data"

# 数据集注册表：前端全局 toggle 的两个轨道。每个 dataset 给出
#   - configs: 按优先级尝试的主路径 ckpt 配置（首个 best.pth 在位者胜）
#   - suffix:  数据 JSON 后缀（"" = metrics.json，"_imagenet64" = metrics_imagenet64.json）
# CIFAR 默认；IN64 模型显存翻倍且 64×64 解码 ~12500 步/图，故按需 lazy-load（见 _get_cached_model）。
_DATASETS = {
    "cifar10": {
        "configs": ["ccigpt_cifar10_s_rgb_ronly_v2.yaml", "ccigpt_cifar10_s_rgb_ronly.yaml"],
        "suffix": "",
    },
    "imagenet64": {
        "configs": ["ccigpt_imagenet64_v1.yaml"],
        "suffix": "_imagenet64",
    },
}


def _norm_dataset(dataset: str) -> str:
    """校验并归一化 dataset 参数；非法值 → 400（不静默回退，避免前端拼错时悄悄给错数据集）。"""
    d = (dataset or "cifar10").lower()
    if d not in _DATASETS:
        raise HTTPException(status_code=400,
                            detail=f"unknown dataset {dataset!r}; expected one of {list(_DATASETS)}")
    return d


def _dataset_json(stem: str, dataset: str) -> str:
    """('metrics', 'imagenet64') → 'metrics_imagenet64.json'；cifar10 → 'metrics.json'。"""
    return f"{stem}{_DATASETS[_norm_dataset(dataset)]['suffix']}.json"

# 上传图片大小上限（10 MB）和允许的 MIME 类型，防止 /api/predict 被恶意大文件
# 或非图片文件耗尽内存。
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_DECODED_PIXELS = 25_000_000
ALLOWED_CONTENT_TYPES = {
    "image/jpeg", "image/png", "image/webp", "image/gif", "image/bmp",
}

app = FastAPI(title="MDL Deep Image Compression Demo")

# 流式端点出错时：完整堆栈记到服务端日志，客户端只收通用提示（避免在公网
# AutoDL --host 0.0.0.0 映射下把 CUDA OOM / 主机路径等内部细节泄漏给访问者）。
logger = logging.getLogger("mdlic.demo")


def _env_number(name, default, cast, minimum):
    try:
        value = cast(os.environ.get(name, default))
    except (TypeError, ValueError):
        value = cast(default)
    return max(minimum, value)


# 模型缓存是全局单槽，因此模型任务固定串行。这样切换 CIFAR/ImageNet64 时，旧模型
# 已无在途使用者，可以先释放显存再加载新模型，不会在切换窗口同时常驻两份权重。
_COMPUTE_CONCURRENCY = 1
_COMPUTE_QUEUE_LIMIT = _env_number("MDLIC_DEMO_GPU_QUEUE", 2, int, 0)
_COMPUTE_WAIT_SECONDS = _env_number("MDLIC_DEMO_GPU_WAIT_SECONDS", 30.0, float, 0.1)
_COMPUTE_SLOTS = threading.BoundedSemaphore(_COMPUTE_CONCURRENCY)
_COMPUTE_ADMISSION = threading.BoundedSemaphore(
    _COMPUTE_CONCURRENCY + _COMPUTE_QUEUE_LIMIT
)

_POST_RATE_LIMIT = _env_number("MDLIC_DEMO_RATE_LIMIT", 8, int, 0)
_POST_RATE_WINDOW = _env_number(
    "MDLIC_DEMO_RATE_WINDOW_SECONDS", 60.0, float, 1.0,
)
_REQUEST_LIMITER = SlidingWindowRateLimiter(
    limit=_POST_RATE_LIMIT,
    window_seconds=_POST_RATE_WINDOW,
)
_API_KEY = os.environ.get("MDLIC_DEMO_API_KEY")
_TRUST_PROXY = os.environ.get("MDLIC_DEMO_TRUST_PROXY", "").lower() in {
    "1", "true", "yes",
}


def _request_client_key(request: Request) -> str:
    if _TRUST_PROXY:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",", 1)[0].strip()
    return request.client.host if request.client is not None else "unknown"


@app.middleware("http")
async def protect_compute_endpoints(request: Request, call_next):
    """Authenticate and rate-limit POST APIs before they can enter the GPU queue."""
    if request.method == "POST" and request.url.path.startswith("/api/"):
        if _API_KEY:
            supplied = request.headers.get("x-api-key", "")
            if not hmac.compare_digest(supplied, _API_KEY):
                return JSONResponse(
                    status_code=401,
                    content={"detail": "API key required"},
                    headers={"X-MDLIC-API-Key-Required": "1"},
                )
        allowed, retry_after = _REQUEST_LIMITER.admit(_request_client_key(request))
        if not allowed:
            return JSONResponse(
                status_code=429,
                content={"detail": "请求过于频繁，请稍后重试"},
                headers={"Retry-After": str(retry_after)},
            )
    return await call_next(request)


class _ComputeLease:
    def __init__(self):
        self._released = False

    def release(self):
        if not self._released:
            self._released = True
            _COMPUTE_SLOTS.release()
            _COMPUTE_ADMISSION.release()


def _acquire_compute_lease():
    """取得一个有界模型执行槽；调用方必须在 finally 中 release。"""
    retry_after = str(max(1, math.ceil(_COMPUTE_WAIT_SECONDS)))
    if not _COMPUTE_ADMISSION.acquire(blocking=False):
        raise HTTPException(
            status_code=429,
            detail="模型请求队列已满，请稍后重试",
            headers={"Retry-After": retry_after},
        )
    if not _COMPUTE_SLOTS.acquire(timeout=_COMPUTE_WAIT_SECONDS):
        _COMPUTE_ADMISSION.release()
        raise HTTPException(
            status_code=429,
            detail=f"等待模型执行超过 {_COMPUTE_WAIT_SECONDS:g} 秒，请稍后重试",
            headers={"Retry-After": retry_after},
        )
    return _ComputeLease()

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
# dataset query 参数选数据集轨道（cifar10 默认 / imagenet64）。
@app.get("/api/metrics")
def get_metrics(dataset: str = "cifar10"):
    return _load_json(_dataset_json("metrics", dataset))


@app.get("/api/probe")
def get_probe(dataset: str = "cifar10"):
    return _load_json(_dataset_json("probe", dataset))


@app.get("/api/kernels")
def get_kernels():
    return _load_json("kernels.json")


@app.get("/api/scales")
def get_scales(dataset: str = "cifar10"):
    return _load_json(_dataset_json("scales", dataset))


def _model_image_size(model, model_type) -> int:
    """模型的输入边长：CC-iGPT 取 fine 分支，单尺度 iGPT 取自身。
    用于把上传图 resize 到该 dataset 模型期望的尺寸（CIFAR 32 / IN64 64），
    取代旧的写死 32。"""
    return model.fine.image_size if model_type == "ccigpt" else model.image_size


def _read_upload_to_tensor(file: UploadFile, size: int = 32, allow_resize: bool = True):
    """上传图片转模型 RGB tensor，并返回可审计的预处理元数据。"""
    try:
        from PIL import Image
        import torch
    except ImportError:
        raise HTTPException(status_code=500, detail="torch/Pillow not installed")

    if file.content_type and file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail=f"Unsupported media type: {file.content_type}")
    contents = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large (>{MAX_UPLOAD_BYTES // 1024 // 1024} MB)")
    try:
        source = Image.open(io.BytesIO(contents))
        source_size = source.size
        source_mode = source.mode
        if source_size[0] * source_size[1] > MAX_DECODED_PIXELS:
            raise HTTPException(
                status_code=413,
                detail=f"Decoded image is too large (>{MAX_DECODED_PIXELS} pixels)",
            )
        source.load()
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    expected_size = (size, size)
    resized = source_size != expected_size
    if resized and not allow_resize:
        raise HTTPException(
            status_code=422,
            detail=(f"编码要求输入恰好为 {size}x{size} 像素；收到 "
                    f"{source_size[0]}x{source_size[1]}。请先显式预处理后再编码"),
        )

    img = source.convert("RGB")
    if resized:
        img = img.resize(expected_size, Image.Resampling.BILINEAR)
    rgb = np.asarray(img, dtype=np.uint8).copy()
    x = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255).unsqueeze(0)
    preprocessing = {
        "source_width": source_size[0],
        "source_height": source_size[1],
        "source_mode": source_mode,
        "model_width": size,
        "model_height": size,
        "output_mode": "RGB",
        "converted_to_rgb": source_mode != "RGB",
        "resized": resized,
        "resize_filter": "bilinear" if resized else None,
        "pixel_exact_scope": (
            "decoded RGB pixels after preprocessing; original PNG/JPEG bytes and metadata "
            "are not preserved"
        ),
    }
    return x, img, preprocessing


@app.post("/api/predict")
def predict(file: UploadFile = File(...), dataset: str = Form("cifar10")):
    """
    上传一张图片，返回:
      - bpd: 整体 bits/dim（CC-iGPT 为 bpd_total，iGPT 由 CE 推算）
      - heatmap: base64 编码的 fine 分支 BPP 热力图 PNG
                 （单位 bits/pixel = bpd × C；CC-iGPT 仅画 fine 段，
                 coarse 分支不在该图上）

    注：sync def 形式让 FastAPI 自动放进 threadpool，避免 GPU forward 阻塞
    event loop（async def 里直接调 model(x) 会卡住其它并发请求）。
    """
    try:
        import torch
    except ImportError:
        raise HTTPException(status_code=500, detail="torch/torchvision not installed")

    lease = _acquire_compute_lease()
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, model_type = _get_cached_model(device, dataset)
        if model is None:
            raise HTTPException(status_code=503, detail="No checkpoint available. Place a checkpoint in experiments/*/checkpoints/best.pth")

        size = _model_image_size(model, model_type)
        x, _, preprocessing = _read_upload_to_tensor(file, size=size, allow_resize=True)

        x = x.to(device)
        model.eval()
        with torch.no_grad():
            out = model(x)

        ce_loss = out["ce_loss"].item()
        if "bpd" in out and out["bpd"] is not None:
            bpd = out["bpd"].item()
        else:
            bpd = ce_loss / math.log(2)

        extras = {}
        if "ce_loss_coarse" in out and out["ce_loss_coarse"] is not None:
            extras["ce_coarse"] = round(out["ce_loss_coarse"].item(), 4)
            extras["ce_fine"] = round(out["ce_loss_fine"].item(), 4)
            if "ctx_alpha" in out and out["ctx_alpha"] is not None:
                extras["ctx_alpha"] = round(out["ctx_alpha"].item(), 4)

        heatmap_b64 = None
        if out.get("logits") is not None:
            heatmap_b64 = _make_heatmap_b64(model, x, out["logits"])
    finally:
        lease.release()

    return JSONResponse({
        "bpd": round(bpd, 4),
        "ce_loss": round(ce_loss, 4),
        "model_type": model_type,
        "heatmap": heatmap_b64,
        "preprocessing": preprocessing,
        **extras,
    })


def _b64_png(arr, scale=4):
    """(H,W,C) uint8 numpy → base64 PNG，nearest 放大 scale 倍便于肉眼看清 32×32。

    /api/decode 与 /api/complete 共用，保证各路径出图口径一致。
    """
    from PIL import Image
    im = Image.fromarray(arr).resize((arr.shape[1] * scale, arr.shape[0] * scale),
                                     Image.Resampling.NEAREST)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _tokens_fingerprint(tokens):
    """token 列表 → 8-hex SHA1 指纹，用于编/解码两侧同会话交叉校验。"""
    b = bytes(int(t) & 0xFF for t in tokens)
    return hashlib.sha1(b).hexdigest()[:8]


# 上传 .bin 大小上限：CIFAR 32×32×3 在 ~3 bpd 下 ≈ 1.1KB，给 1MB 余量足够防滥用。
MAX_BIN_BYTES = 1 * 1024 * 1024
# 流式编/解码每多少步推一次进度（太密会刷爆前端，太疏过不了反代 idle 超时）。
_DECODE_PROGRESS_EVERY = 128


def _drive_coder(gen, stage, base, total):
    """驱动 _encode/_decode_sequence_iter，按**累积模型预测数** gate NDJSON 进度，
    StopIteration 时返回 (coder 结果, 新 base)。

    每条独立流的首 token 走均匀先验，不触发 forward，故不计入 total。base 是之前
    stage 已完成的预测数；累积 done = base + m。gate 条件 `cum % N == 0`
    **或** `m == per_stage_total`（每个 stage 末步必发）—— 修掉两个老坑：
      1. coarse N_c=64 < N=128 时整段零进度（per-stage `m % N` 永不命中）；
      2. fine 末段 m∈(2944, 3071] 静默 127 步，bar 卡 95.9% 直到 done。
    保证每段首尾都有进度行 flush，反代 idle keep-alive 不被这两个窗口击穿。
    """
    cum = base
    last_m = 0
    try:
        while True:
            m, per_stage_total = next(gen)
            cum = base + m
            last_m = m
            if cum % _DECODE_PROGRESS_EVERY == 0 or m == per_stage_total:
                yield json.dumps({"type": "progress", "stage": stage,
                                  "done": cum, "total": total}) + "\n"
    except StopIteration as stop:
        return stop.value, base + last_m


@app.post("/api/encode")
def encode(file: UploadFile = File(...), dataset: str = Form("cifar10")):
    """上传图 → **逐 token gold 算术编码** → 模型绑定 MDLC v2（流式 NDJSON 进度）。

    编码必须与 /api/decode 走**同一**逐 token 公共 codec 路径（`encode_sequence_iter`
    与 `decode_sequence_iter` 共用 logits 路径，prefix+0 缓冲逐位相同）：算术编码
    零容忍，logits 差一个 ULP 即可让某 token 跨累积频数边界翻符号、解码失步成噪点。同源
    保证 encode 写的 bits 与 decode 逐 token 读所需分布逐位相同 → bit-exact 可解。

    代价：(N_c-1) + (N_f-1) ≈ 3100 次 forward（与解码同量级），故同样流式吐进度
    （每 ~128 步一行 NDJSON）绕开 AutoDL 反代 idle 超时。
    """
    try:
        import torch
    except ImportError:
        raise HTTPException(status_code=500, detail="torch/torchvision not installed")

    lease = _acquire_compute_lease()
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, model_type = _get_cached_model(device, dataset)
        if model is None:
            raise HTTPException(status_code=503, detail="No checkpoint available. Place a checkpoint in experiments/*/checkpoints/best.pth")

        # 编码协议不允许静默 resize：码流只承诺输入转为 RGB 后的模型尺寸像素。
        x, _, preprocessing = _read_upload_to_tensor(
            file, size=_model_image_size(model, model_type), allow_resize=False
        )
    except Exception:
        lease.release()
        raise

    def _gen():
        try:
            try:
                model.eval()
                with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=False):
                    x_dev = x.to(device).clamp(0, 1).float()
                    source_rgb = (
                        (x_dev * 255).round().to(torch.uint8)[0]
                        .permute(1, 2, 0).contiguous().cpu().numpy().tobytes()
                    )
                    source_rgb_sha256 = sha256_rgb_bytes(source_rgb)
                    identity = model._codec_identity
                    dual = (model_type == "ccigpt")
                    if dual:
                        H = model.fine.image_size
                        C = model.fine.in_channels
                        N_c = model.coarse.seq_len
                        N_f = model.fine.seq_len
                        total_steps = num_model_predictions(N_c, N_f)
                        yield json.dumps({"type": "start", "total": total_steps,
                                          "unit": "model_predictions",
                                          "stages": ["coarse", "fine"]}) + "\n"

                        x_c = model._coarse_input(x_dev)
                        coarse_tokens = model.coarse._tokenize(x_c)
                        gen_c = encode_sequence_iter(model.coarse, coarse_tokens, None, device)
                        c_bits, base = yield from _drive_coder(gen_c, "coarse", 0, total_steps)
                        coarse_ctx = model.ctx_alpha * model._compute_coarse_ctx(coarse_tokens)
                        fine_tokens = model.fine._tokenize(x_dev)
                        gen_f = encode_sequence_iter(model.fine, fine_tokens, coarse_ctx, device)
                        f_bits, _ = yield from _drive_coder(gen_f, "fine", base, total_steps)
                        enc_tokens = coarse_tokens[0].tolist() + fine_tokens[0].tolist()
                    else:
                        H = model.image_size
                        C = model.in_channels
                        N_f = model.seq_len
                        total_steps = num_model_predictions(N_f)
                        yield json.dumps({"type": "start", "total": total_steps,
                                          "unit": "model_predictions",
                                          "stages": ["single"]}) + "\n"
                        tokens = model._tokenize(x_dev)
                        gen = encode_sequence_iter(model, tokens, None, device)
                        f_bits, _ = yield from _drive_coder(gen, "single", 0, total_steps)
                        c_bits = []
                        enc_tokens = tokens[0].tolist()

                    blob = build_container_bytes(
                        dual,
                        H,
                        C,
                        c_bits,
                        f_bits,
                        identity=identity,
                        source_rgb_sha256=source_rgb_sha256,
                    )
                    total_bits = len(c_bits) + len(f_bits)
                    container_meta = parse_container_meta(blob)
                    parts = {"coarse_bits": len(c_bits), "fine_bits": len(f_bits)} if dual else {}
                    yield json.dumps({
                        "type": "done",
                        "model_type": model_type,
                        "filename": "image.mdlc.bin",
                        "bin_b64": base64.b64encode(blob).decode("ascii"),
                        "bin_bytes": len(blob),
                        "container_version": container_meta["version"],
                        "integrity_verified": container_meta["integrity_verified"],
                        "dual": dual,
                        "H": H, "C": C,
                        "neural_bits": total_bits,
                        "achieved_bpd": round(container_meta["payload_bpd"], 4),
                        "payload_bpd": round(container_meta["payload_bpd"], 4),
                        "packed_payload_bpd": round(container_meta["packed_payload_bpd"], 4),
                        "file_bpd": round(container_meta["file_bpd"], 4),
                        "preprocessing": preprocessing,
                        "fingerprint": _tokens_fingerprint(enc_tokens),
                        **parts,
                    }) + "\n"
            except Exception:   # noqa: BLE001
                logger.exception("/api/encode 流式编码失败")
                yield json.dumps({"type": "error",
                                  "detail": "编码失败（服务端错误，详见后端日志）"}) + "\n"
        finally:
            lease.release()

    return StreamingResponse(_gen(), media_type="application/x-ndjson")


def _read_bin_upload(file: UploadFile) -> bytes:
    """上传 .bin → 校验大小 + magic → 返回原始字节。"""
    blob = file.file.read(MAX_BIN_BYTES + 1)
    if len(blob) > MAX_BIN_BYTES:
        raise HTTPException(status_code=413, detail=f".bin 过大（>{MAX_BIN_BYTES // 1024} KB）")
    if len(blob) < 4 or blob[:4] != b"MDLC":
        raise HTTPException(status_code=400, detail="不是 MDLC 容器（magic 不符）")
    return blob


@app.post("/api/inspect")
def inspect(file: UploadFile = File(...)):
    """上传 .bin → 即时解析容器结构/码长/bpd（纯文件级，无需 GPU/ckpt/模型）。

    容器结构自描述：仅凭文件就能读出尺寸/双尺度/码长/bpd/identity。与 CLI
    `verify_lossless.py --inspect` 共用 `parse_container_meta`，口径一致。
    """
    blob = _read_bin_upload(file)
    try:
        meta = parse_container_meta(blob)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    for key in ("bpd", "payload_bpd", "packed_payload_bpd", "file_bpd"):
        meta[key] = round(meta[key], 4)
    return JSONResponse(meta)


@app.post("/api/decode")
def decode(file: UploadFile = File(...), dataset: str = Form("cifar10")):
    """上传 .bin → 流式逐 token 盲解码还原图像（NDJSON 进度行）。

    真实 decoder 视角：只给文件，无原图。逐 token 重跑完整 forward（无 KV-cache,
    causal mask 保证 0 后缀不泄漏），与 scripts/verify_lossless 的 gold 路径同源
    （共用 `decode_sequence_iter`）。CIFAR 共 (64-1)+(3072-1)=3134 次预测、~60–120s,
    故用 StreamingResponse 每 ~128 步推一行进度，绕开 AutoDL 反代 idle 超时。

    解码步数由模型几何固定（model.coarse.seq_len / model.fine.seq_len），不受文件
    字段控制 → 无放大攻击面。文件 H/C 必须与当前模型一致，否则解出的分布对不上。
    """
    try:
        import torch
    except ImportError:
        raise HTTPException(status_code=500, detail="torch/torchvision not installed")

    blob = _read_bin_upload(file)
    lease = _acquire_compute_lease()
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, model_type = _get_cached_model(device, dataset)
        if model is None:
            raise HTTPException(status_code=503, detail="No checkpoint available. Place a checkpoint in experiments/*/checkpoints/best.pth")

        container_meta = parse_container_meta(blob)
        dual, H, C, c_bits, f_bits = read_container_bytes(
            blob,
            expected_identity=model._codec_identity,
        )
        W = container_meta["W"]

        m_H = model.fine.image_size if model_type == "ccigpt" else model.image_size
        m_C = model.fine.in_channels if model_type == "ccigpt" else model.in_channels
        if (H, W, C) != (m_H, m_H, m_C):
            raise HTTPException(
                status_code=400,
                detail=f".bin 图像几何 {H}×{W}×{C} 与当前模型 {m_H}×{m_H}×{m_C} 不符（需用编码时同款模型解）")
        if dual != (model_type == "ccigpt"):
            want = "CC-iGPT 双尺度" if dual else "单尺度 iGPT"
            raise HTTPException(status_code=400,
                                detail=f"{'双尺度' if dual else '单尺度'} .bin 需 {want} 模型解码，当前模型为 {model_type}")
    except ValueError as exc:
        lease.release()
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        lease.release()
        raise

    def _gen():
        try:
            try:
                model.eval()
                with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=False):
                    if model_type == "ccigpt" and dual:
                        N_c = model.coarse.seq_len
                        N_f = model.fine.seq_len
                        total_steps = num_model_predictions(N_c, N_f)
                        yield json.dumps({"type": "start", "total": total_steps,
                                          "unit": "model_predictions",
                                          "stages": ["coarse", "fine"]}) + "\n"
                        gen_c = decode_sequence_iter(model.coarse, c_bits, N_c, None, device)
                        coarse_dec, base = yield from _drive_coder(gen_c, "coarse", 0, total_steps)
                        coarse_ctx = model.ctx_alpha * model._compute_coarse_ctx(coarse_dec)
                        gen_f = decode_sequence_iter(model.fine, f_bits, N_f, coarse_ctx, device)
                        fine_dec, _ = yield from _drive_coder(gen_f, "fine", base, total_steps)
                        recon_t = fine_dec
                        total_bits = len(c_bits) + len(f_bits)
                        n_subpix = H * W * C
                        fp_tokens = coarse_dec[0].tolist() + fine_dec[0].tolist()
                    else:
                        N = model.seq_len
                        total_steps = num_model_predictions(N)
                        yield json.dumps({"type": "start", "total": total_steps,
                                          "unit": "model_predictions",
                                          "stages": ["single"]}) + "\n"
                        gen = decode_sequence_iter(model, f_bits, N, None, device)
                        dec, _ = yield from _drive_coder(gen, "single", 0, total_steps)
                        recon_t = dec
                        total_bits = len(f_bits)
                        n_subpix = H * W * C
                        fp_tokens = dec[0].tolist()

                    recon_tokens = recon_t[0].tolist()
                    recon = np.array(recon_tokens, dtype=np.uint8).reshape(H, W, C)
                    validate_decoded_rgb(recon.tobytes(), container_meta)
                    achieved_bpd = total_bits / n_subpix
                    file_bpd = len(blob) * 8 / n_subpix
                    yield json.dumps({
                        "type": "done",
                        "recon_png": _b64_png(recon),
                        "achieved_bpd": round(achieved_bpd, 4),
                        "payload_bpd": round(achieved_bpd, 4),
                        "file_bpd": round(file_bpd, 4),
                        "rgb_checksum_verified": True,
                        "neural_bits": total_bits,
                        "H": H, "C": C,
                        "fingerprint": _tokens_fingerprint(fp_tokens),
                    }) + "\n"
            except Exception:   # noqa: BLE001
                logger.exception("/api/decode 流式解码失败")
                yield json.dumps({"type": "error",
                                  "detail": "解码失败（服务端错误，详见后端日志）"}) + "\n"
        finally:
            lease.release()

    # media_type 用 ndjson；sync 生成器由 Starlette 放进 threadpool，GPU 不阻塞 event loop
    return StreamingResponse(_gen(), media_type="application/x-ndjson")


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
    dataset: str = Form("cifar10"),
):
    """图像补全 demo（下游 §6.2）：上传图 → 保留前 keep_frac 的 raster token（≈上半）
    → AR 续采样补全下半 → 返回 原图 / 已知上半(灰=待补) / 补全 三张图。

    与 scripts/complete_image.py 同一套 per-step forward（无 KV-cache，causal mask
    保证 0 后缀不泄漏）。CC-iGPT 的 coarse ctx 由**整图**缩略图算 —— 故语义是"低分
    缩略图 + 上半真实像素 → 补下半"，coarse 是显式 side-channel（与压缩时独立 bitstream
    同源），非偷看答案。详见 complete_image.py docstring。

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

    lease = _acquire_compute_lease()
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, model_type = _get_cached_model(device, dataset)
        if model is None:
            raise HTTPException(status_code=503, detail="No checkpoint available. Place a checkpoint in experiments/*/checkpoints/best.pth")

        x, _, preprocessing = _read_upload_to_tensor(
            file, size=_model_image_size(model, model_type), allow_resize=True
        )

        model.eval()
        with torch.no_grad():
            o, m, c = complete_image(
                model,
                model_type,
                x,
                keep_frac,
                temperature,
                top_k,
                device,
            )
    finally:
        lease.release()

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
        "preprocessing": preprocessing,
    })


# 全局只保留一个活跃模型。模型任务由 _COMPUTE_SLOTS 串行化，因此数据集切换时
# 可以确定旧模型没有在途使用者；释放引用并清空 CUDA allocator 后再加载新权重。
_MODEL_CACHE = {
    "dataset": None,
    "device": None,
    "model": None,
    "type": None,
}
_MODEL_CACHE_LOCK = threading.Lock()


def _evict_cached_model(torch):
    old_model = _MODEL_CACHE["model"]
    old_device = _MODEL_CACHE["device"]
    _MODEL_CACHE.update(dataset=None, device=None, model=None, type=None)
    if old_model is None:
        return
    del old_model
    gc.collect()
    if str(old_device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _get_cached_model(device, dataset: str = "cifar10"):
    """线程安全地延迟加载单个活跃模型。

    命中相同 dataset/device 时直接复用；切换时先释放旧模型，再加载目标 checkpoint，
    避免 CIFAR 与 ImageNet64 权重同时常驻 GPU。

    返回 (model, model_type)：model_type ∈ {"ccigpt", "igpt"}。
    """
    dataset = _norm_dataset(dataset)
    device_key = str(device)
    with _MODEL_CACHE_LOCK:
        if (_MODEL_CACHE["model"] is not None
                and _MODEL_CACHE["dataset"] == dataset
                and _MODEL_CACHE["device"] == device_key):
            return _MODEL_CACHE["model"], _MODEL_CACHE["type"]

        import yaml
        import torch

        # 按优先级尝试该 dataset 的主路径 ckpt（首个 best.pth 在位者胜）。
        # iGPT 单尺度 baseline 已退出主线（Phase A 历史），不作 fallback。
        configs_dir = ROOT / "configs"
        experiments_dir = ROOT / "experiments"

        candidate = None
        for cfg_name in _DATASETS[dataset]["configs"]:
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
            if model_type != "ccigpt":
                raise ValueError(
                    f"demo 主线只支持 CC-iGPT，{cfg_name} 的 "
                    f"model.type={model_type!r} 不匹配"
                )
            candidate = (mcfg, model_type, ckpt_path)
            break

        if candidate is None:
            return None, None

        mcfg, model_type, ckpt_path = candidate
        _evict_cached_model(torch)

        from mdlic.model_factory import build_ccigpt_from_config
        model = build_ccigpt_from_config(mcfg, device)

        # best.pth 是裸 state_dict（torch.save(model.state_dict())），不含
        # optimizer/epoch 等 Python 对象，可安全使用 weights_only=True
        # 杜绝 pickle RCE。若未来需加载 epoch_*.pth 这类含训练状态的 ckpt，
        # 改回 False 并保证 ckpt 来源受信。
        # 权重先落在 CPU，逐参数复制到目标设备；避免 load_state_dict 期间 GPU 上同时
        # 保留“模型参数 + 一整份 checkpoint tensor”。
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
        from mdlic.utils import clean_state_dict
        if "model_state_dict" in ckpt:
            state_dict = clean_state_dict(ckpt["model_state_dict"])
        else:
            state_dict = clean_state_dict(ckpt)
        model.load_state_dict(state_dict)
        del state_dict, ckpt
        gc.collect()
        model.eval()

        from mdlic.codec.container import (
            make_codec_identity,
            runtime_fingerprint,
            sha256_file,
        )
        model._codec_identity = make_codec_identity(
            model_type=model_type,
            model_config=mcfg,
            checkpoint_sha256=sha256_file(ckpt_path),
            runtime=runtime_fingerprint(device),
        )

        _MODEL_CACHE.update(
            dataset=dataset,
            device=device_key,
            model=model,
            type=model_type,
        )
        return model, model_type


def _make_heatmap_b64(model, x, logits):
    """根据已计算的 fine logits 生成 32×32 BPP 热力图 (bits/pixel)，返回 base64 PNG。

    支持 iGPT 与 CC-iGPT 两种模型：iGPT 直接用自身 _tokenize；CC-iGPT 用
    fine 子分支的 _tokenize（coarse 分支的 8×8 不贡献热力图；历史分解中它约占
    4.6% bpd，仍包含在 bpd_total 主指标中）。

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
