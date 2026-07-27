#!/usr/bin/env python3
"""
Training script for iGPT autoregressive image compression.

Usage:
    python scripts/train.py --config configs/igpt_cifar10_s_rgb.yaml
    python scripts/train.py --config configs/igpt_cifar10_s_rgb.yaml --resume experiments/.../epoch_10.pth
    python scripts/train.py --config configs/igpt_cifar10_s_rgb.yaml --init_from experiments/.../best.pth
    torchrun --nproc_per_node=2 scripts/train.py --config configs/igpt_cifar10_s_rgb.yaml
"""

import os
import sys
import csv
import json
import hashlib
import random
import argparse
import yaml
import math
import torch
import torch.nn as nn
import torch.optim as optim
from contextlib import nullcontext
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from torchvision import transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from mdlic.models.igpt import IGPT
from mdlic.models.cc_igpt import CCIGPT
from mdlic.models.layers import get_fused_kernel_status
from mdlic.eval_metrics import per_image_bpd
from mdlic.provenance import (
    canonical_sha256 as _canonical_provenance_hash,
    dataset_record,
    git_metadata,
    runtime_metadata,
    source_tree_record,
)
from mdlic.utils import seed_everything, compute_bpd, clean_state_dict


TRAINING_STATE_SCHEMA = "mdlic-training-state-v3"
LEGACY_TRAINING_STATE_SCHEMA = "mdlic-training-state-v2"


class DistributedEvalSampler(Sampler):
    """Deterministically shard evaluation data without padding or dropping.

    Shards may differ by one sample, so validation must call the unwrapped model:
    DDP forward can perform buffer collectives and would hang when ranks execute a
    different number of batches.
    """

    def __init__(self, dataset, num_replicas=None, rank=None):
        if num_replicas is None:
            if not dist.is_available() or not dist.is_initialized():
                raise RuntimeError("distributed process group is not initialized")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available() or not dist.is_initialized():
                raise RuntimeError("distributed process group is not initialized")
            rank = dist.get_rank()
        if not isinstance(num_replicas, int) or num_replicas <= 0:
            raise ValueError(f"num_replicas must be positive, got {num_replicas!r}")
        if not isinstance(rank, int) or not 0 <= rank < num_replicas:
            raise ValueError(f"rank must be in [0,{num_replicas}), got {rank!r}")
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.num_replicas))

    def __len__(self):
        remaining = len(self.dataset) - self.rank
        return 0 if remaining <= 0 else (remaining + self.num_replicas - 1) // self.num_replicas


def _validate_config(config: dict):
    """
    配置文件完整性校验：检查必填字段、类型和取值范围。

    在训练开始前尽早发现配置错误，避免 GPU 资源浪费。
    错误信息明确指出问题字段和期望值，方便快速定位。

    Ref: 工程最佳实践 — "Fail fast, fail loud"
         Ref: Google, "Machine Learning: The High-Interest Credit Card of
              Technical Debt," NeurIPS 2015 Workshop — 配置验证是减少 ML 技术债的关键。
    """
    # ── 必填顶层字段 ──
    for key in ['exp_name', 'model', 'data', 'train', 'eval', 'checkpoint']:
        assert key in config, f"Config 缺少必填字段: '{key}'"

    # ── model 字段 ──
    mcfg = config['model']
    model_type = mcfg.get('type', 'igpt')

    if model_type in ('igpt', 'ccigpt'):
        required_model = ['image_size', 'in_channels', 'vocab_size', 'd_model', 'N', 'h', 'd_ff', 'dropout']
        for key in required_model:
            assert key in mcfg, f"Config model 缺少必填字段: 'model.{key}'"

        assert mcfg['d_model'] % mcfg['h'] == 0, (
            f"model.d_model ({mcfg['d_model']}) 必须能被 model.h ({mcfg['h']}) 整除"
        )
        assert mcfg['d_model'] > 0, f"model.d_model 必须 > 0，got {mcfg['d_model']}"
        assert mcfg['N'] > 0, f"model.N (depth) 必须 > 0，got {mcfg['N']}"
        assert mcfg['h'] > 0, f"model.h (heads) 必须 > 0，got {mcfg['h']}"
        assert mcfg['d_ff'] > 0, f"model.d_ff 必须 > 0，got {mcfg['d_ff']}"
        assert 0.0 <= mcfg['dropout'] < 1.0, f"model.dropout 必须在 [0, 1)，got {mcfg['dropout']}"
        assert mcfg['vocab_size'] > 0, f"model.vocab_size 必须 > 0，got {mcfg['vocab_size']}"
    else:
        raise ValueError(f"未知 model.type: '{model_type}'，支持 igpt/ccigpt")

    # ── train 字段 ──
    tcfg = config['train']
    assert tcfg.get('batch_size', 0) > 0, f"train.batch_size 必须 > 0"
    assert float(tcfg.get('lr', 0)) > 0, f"train.lr 必须 > 0"
    assert tcfg.get('epochs', 0) > 0, f"train.epochs 必须 > 0"
    assert tcfg.get('clip_max_norm', 0) > 0, f"train.clip_max_norm 必须 > 0"
    assert tcfg.get('grad_accum_steps', 1) >= 1, f"train.grad_accum_steps 必须 >= 1"

    amp_dtype = tcfg.get('amp_dtype', 'fp16')
    # None / "none" / "fp32" 走 fp32 训练（数值敏感场景兜底用，主路径 forward
    # 已支持 amp_dtype is None 时跳过 autocast）
    assert amp_dtype in ('fp16', 'bf16', None, 'none', 'fp32'), (
        f"train.amp_dtype 必须是 'fp16' / 'bf16' / null / 'none' / 'fp32'，"
        f"got '{amp_dtype}'"
    )

    lr_schedule = tcfg.get('lr_schedule', 'cosine')
    assert lr_schedule in ('cosine', 'wsd'), (
        f"train.lr_schedule 必须是 cosine/wsd，got '{lr_schedule}'"
    )

    # SWA 与 epochs 交叉校验：start_epoch > epochs 时训练不会触发任何 SWA 更新，
    # 尾部 finalize 走 warning 路径但 swa.pth 不会落盘。提前 fail-fast 避免训完
    # 才发现 SWA 配错。
    swa_cfg = tcfg.get('swa', {})
    if swa_cfg.get('enabled', False):
        swa_start = swa_cfg.get('start_epoch', 0)
        assert swa_start <= tcfg['epochs'], (
            f"train.swa.start_epoch ({swa_start}) 不能大于 train.epochs "
            f"({tcfg['epochs']})，否则 SWA 永不触发，swa.pth 无法生成。"
        )

    # z-loss 权重校验（model_type 在上面 if-else 已限定 ∈ {igpt, ccigpt}）
    z_w = float(tcfg.get('z_loss_weight', 1e-4))
    assert z_w >= 0, f"train.z_loss_weight 必须 >= 0，got {z_w}"

    # CC-iGPT 额外校验
    if model_type == 'ccigpt':
        for key in ['pool_factor', 'coarse_d_model', 'coarse_N', 'coarse_h', 'coarse_d_ff']:
            assert key in mcfg, f"Config model 缺少 ccigpt 必填字段: 'model.{key}'"
        assert mcfg['image_size'] % mcfg['pool_factor'] == 0, (
            f"image_size ({mcfg['image_size']}) 必须能被 pool_factor ({mcfg['pool_factor']}) 整除"
        )
        assert mcfg['coarse_d_model'] % mcfg['coarse_h'] == 0, (
            f"coarse_d_model ({mcfg['coarse_d_model']}) 必须能被 coarse_h ({mcfg['coarse_h']}) 整除"
        )

        # coarse_in_channels：R-only 灰度先验（仅 1 或 in_channels 合法，
        # 中间值 expand 路径不可达，详见 cc_igpt.py:CCIGPT.__init__）
        coarse_ic = mcfg.get('coarse_in_channels')
        if coarse_ic is not None:
            assert coarse_ic in (1, mcfg['in_channels']), (
                f"model.coarse_in_channels ({coarse_ic}) 必须 ∈ "
                f"{{1, in_channels={mcfg['in_channels']}}}"
            )


def _shared_igpt_kwargs(mcfg: dict) -> dict:
    """提取 iGPT / CC-iGPT 共用字段。"""
    return dict(
        in_channels=mcfg["in_channels"],
        vocab_size=mcfg["vocab_size"],
        dropout=mcfg["dropout"],
        activation_checkpointing=mcfg.get("activation_checkpointing", False),
        drop_path=mcfg.get("drop_path", 0.0),
    )


def _build_model_from_config(mcfg: dict, device) -> IGPT:
    """从 config['model'] 构建 IGPT 模型（统一 train.py 和 dryrun_forward.py）。"""
    return IGPT(
        image_size=mcfg["image_size"],
        d_model=mcfg["d_model"], N=mcfg["N"], h=mcfg["h"], d_ff=mcfg["d_ff"],
        **_shared_igpt_kwargs(mcfg),
    ).to(device)


def _build_ccigpt_from_config(mcfg: dict, device) -> CCIGPT:
    """从 config['model'] 构建 CC-iGPT 模型。

    顶层字段 (image_size, d_model, N, h, d_ff, ...) 描述 fine 模型；
    额外字段 pool_factor / coarse_d_model / coarse_N / coarse_h / coarse_d_ff
    描述 coarse 子模型。
    """
    return CCIGPT(
        image_size=mcfg["image_size"],
        pool_factor=mcfg["pool_factor"],
        fine_d_model=mcfg["d_model"], fine_N=mcfg["N"],
        fine_h=mcfg["h"], fine_d_ff=mcfg["d_ff"],
        coarse_d_model=mcfg["coarse_d_model"], coarse_N=mcfg["coarse_N"],
        coarse_h=mcfg["coarse_h"], coarse_d_ff=mcfg["coarse_d_ff"],
        coarse_in_channels=mcfg.get("coarse_in_channels"),
        **_shared_igpt_kwargs(mcfg),
    ).to(device)


def _no_decay_param_names(model) -> set:
    """识别不该 weight decay 的参数名集合（embedding / norm / bias / ctx_alpha 等）。

    用 isinstance(module, …) 识别 norm/embedding 层，避免子串 'norm' 误匹配（例如
    未来若把模块命名成 `normalizer` / `ln1` 都会静默改变 wd 行为）。
    """
    from mdlic.models.layers import RMSNorm
    no_decay_modules = (nn.Embedding, RMSNorm, nn.LayerNorm)
    no_decay = set()
    for module_name, module in model.named_modules():
        if isinstance(module, no_decay_modules):
            for pname, _ in module.named_parameters(recurse=False):
                no_decay.add(f"{module_name}.{pname}" if module_name else pname)
    for name, _ in model.named_parameters():
        if name.endswith('bias') or name.endswith('ctx_alpha'):
            no_decay.add(name)
    return no_decay


def _get_param_groups(model, weight_decay=0.1):
    """构建 AdamW 参数组（decayed + no-decay 两组）。"""
    no_decay = _no_decay_param_names(model)
    return [
        {"params": [p for n, p in model.named_parameters() if n not in no_decay], "weight_decay": weight_decay},
        {"params": [p for n, p in model.named_parameters() if n in no_decay],     "weight_decay": 0.0},
    ]


# ==================== Training ====================
def _atomic_save(obj, path: str):
    """torch.save 的原子写包装：先写 .tmp 再 os.replace，避免崩溃中断产生半截文件。

    Why: 直接覆盖式 torch.save 在写入过程中被 SIGTERM/OOM 中断会留下损坏的
    checkpoint，下次 --resume 直接报错。os.replace 在同一文件系统下是原子的。
    """
    tmp_path = path + ".tmp"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def _json_ready(obj):
    """Convert common training objects into JSON-serializable values."""
    if isinstance(obj, dict):
        return {str(k): _json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_ready(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return _json_ready(obj.detach().cpu().item())
        return _json_ready(obj.detach().cpu().tolist())
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def _atomic_save_json(obj, path: str):
    """Atomically write sidecar JSON metadata next to checkpoint files."""
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(_json_ready(obj), f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp_path, path)


def _canonical_config_hash(value) -> str:
    """Stable SHA256 for JSON-compatible config fragments."""
    encoded = json.dumps(
        _json_ready(value), ensure_ascii=False, sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_resume_checkpoint(
    checkpoint: dict,
    *,
    config: dict,
    seed: int,
    world_size: int,
    provenance_fingerprint: str | None = None,
    allow_legacy_resume: bool = False,
    optimizer_count: int,
    scheduler_required: bool,
    scaler_required: bool,
    swa_enabled: bool,
    ema_enabled: bool,
) -> None:
    """Require every state needed to continue the same optimization run."""
    if not isinstance(checkpoint, dict):
        raise RuntimeError("--resume requires a training-state checkpoint mapping")
    schema = checkpoint.get("schema")
    legacy_resume = schema == LEGACY_TRAINING_STATE_SCHEMA
    if legacy_resume and not allow_legacy_resume:
        raise RuntimeError(
            "--resume checkpoint predates source/runtime/data provenance; "
            "pass --allow_legacy_resume to acknowledge non-exact provenance, "
            "or use --init_from"
        )
    if schema not in (TRAINING_STATE_SCHEMA, LEGACY_TRAINING_STATE_SCHEMA):
        raise RuntimeError(
            f"--resume requires schema {TRAINING_STATE_SCHEMA!r}; "
            "use --init_from for bare or legacy weights"
        )

    required = {
        "epoch",
        "model_state_dict",
        "optimizer_state_dicts",
        "best_bpd",
        "config_sha256",
        "model_config_sha256",
        "seed",
        "world_size",
        "rng_states_by_rank",
    }
    if not legacy_resume:
        required.update(("provenance", "provenance_sha256"))
    if scheduler_required:
        required.add("scheduler_state_dict")
    if scaler_required:
        required.add("scaler_state_dict")
    if swa_enabled:
        required.update(("swa_state", "swa_n"))
    if ema_enabled:
        required.update(("ema_state", "ema_tick"))
    missing = sorted(required - checkpoint.keys())
    if missing:
        raise RuntimeError(
            "--resume checkpoint is incomplete; missing: " + ", ".join(missing)
        )

    if not legacy_resume:
        if provenance_fingerprint is None:
            raise RuntimeError("strict resume requires the current provenance fingerprint")
        checkpoint_provenance = checkpoint["provenance"]
        if not isinstance(checkpoint_provenance, dict):
            raise RuntimeError("--resume checkpoint provenance must be a mapping")
        recorded_fingerprint = checkpoint_provenance.get("fingerprint_sha256")
        if checkpoint["provenance_sha256"] != recorded_fingerprint:
            raise RuntimeError("--resume checkpoint provenance hash is internally inconsistent")
        if recorded_fingerprint != provenance_fingerprint:
            raise RuntimeError(
                "--resume provenance mismatch: execution source, runtime, or dataset changed; "
                "use the original environment or --init_from"
            )

    if checkpoint["model_config_sha256"] != _canonical_config_hash(config["model"]):
        raise RuntimeError("--resume model config hash mismatch; use --init_from for migration")
    if checkpoint["config_sha256"] != _canonical_config_hash(config):
        raise RuntimeError("--resume full config hash mismatch; use the original config or --init_from")
    if checkpoint["seed"] != seed:
        raise RuntimeError(
            f"--resume seed mismatch: checkpoint={checkpoint['seed']} current={seed}"
        )
    if checkpoint["world_size"] != world_size:
        raise RuntimeError(
            "--resume world_size mismatch: "
            f"checkpoint={checkpoint['world_size']} current={world_size}"
        )
    if not isinstance(checkpoint["epoch"], int) or checkpoint["epoch"] < 0:
        raise RuntimeError("--resume checkpoint epoch must be a non-negative integer")

    optimizer_states = checkpoint["optimizer_state_dicts"]
    if not isinstance(optimizer_states, list) or len(optimizer_states) != optimizer_count:
        raise RuntimeError(
            "--resume optimizer count mismatch: "
            f"checkpoint={len(optimizer_states) if isinstance(optimizer_states, list) else 'invalid'} "
            f"current={optimizer_count}"
        )
    rng_states = checkpoint["rng_states_by_rank"]
    if not isinstance(rng_states, list) or len(rng_states) != world_size:
        raise RuntimeError(
            "--resume requires exactly one RNG state per rank: "
            f"checkpoint={len(rng_states) if isinstance(rng_states, list) else 'invalid'} "
            f"current={world_size}"
        )


def _load_init_from_state_dict(model, checkpoint: dict) -> dict:
    """Load shape-compatible weights only and report migration coverage."""
    if not isinstance(checkpoint, dict):
        raise RuntimeError("--init_from checkpoint must be a state_dict or mapping")
    source = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(source, dict):
        raise RuntimeError("--init_from checkpoint has no valid model state_dict")
    source = clean_state_dict(source)
    target = model.state_dict()

    compatible = {}
    unexpected = []
    shape_mismatch = []
    for name, value in source.items():
        if name not in target:
            unexpected.append(name)
        elif not torch.is_tensor(value) or value.shape != target[name].shape:
            shape_mismatch.append(name)
        else:
            compatible[name] = value

    parameter_numel = {name: value.numel() for name, value in model.named_parameters()}
    matched_parameter_numel = sum(
        parameter_numel[name] for name in compatible if name in parameter_numel
    )
    total_parameter_numel = sum(parameter_numel.values())
    if matched_parameter_numel == 0:
        raise RuntimeError("--init_from matched no trainable model parameters")

    incompatible = model.load_state_dict(compatible, strict=False)
    return {
        "matched_keys": len(compatible),
        "missing_keys": len(incompatible.missing_keys),
        "unexpected_keys": len(unexpected),
        "shape_mismatch_keys": len(shape_mismatch),
        "matched_parameter_numel": matched_parameter_numel,
        "total_parameter_numel": total_parameter_numel,
        "parameter_coverage": matched_parameter_numel / total_parameter_numel,
    }


def _capture_rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict) -> None:
    """Restore a state produced by _capture_rng_state, including CUDA ranks."""
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([rng.cpu() for rng in state["torch_cuda"]])


def _gather_rng_states(distributed: bool, rank: int):
    """Collect one RNG snapshot per rank; return the list on rank 0 only."""
    local_state = _capture_rng_state()
    if not distributed:
        return [local_state]
    gathered = [None] * dist.get_world_size() if rank == 0 else None
    dist.gather_object(local_state, gathered, dst=0)
    return gathered


_CSV_HEADER = [
    "epoch", "train_loss", "train_bpd", "val_loss",
    "val_bpd", "val_bpd_std", "lr",
]


def _open_training_csv(path: str, resume: bool, start_epoch: int):
    """Open a curve CSV without truncating or duplicating resumed epochs."""
    if resume and os.path.exists(path):
        with open(path, "r", newline="") as existing:
            rows = list(csv.reader(existing))
        if not rows or rows[0] != _CSV_HEADER:
            raise ValueError(f"CSV header 不兼容: {path}")
        data_rows = [row for row in rows[1:] if row]
        if data_rows:
            try:
                last_epoch = int(data_rows[-1][0])
            except (ValueError, IndexError) as exc:
                raise ValueError(f"CSV 最后一行 epoch 非法: {path}") from exc
            if last_epoch >= start_epoch:
                raise ValueError(
                    f"CSV 已记录到 epoch {last_epoch}，但 resume 将从 {start_epoch} 开始；"
                    "拒绝写入重复/倒序曲线")
        fh = open(path, "a", newline="")
        return fh, csv.writer(fh)

    fh = open(path, "w", newline="")
    writer = csv.writer(fh)
    writer.writerow(_CSV_HEADER)
    fh.flush()
    return fh, writer


def _build_training_provenance(
    config: dict,
    train_dataset,
    valid_dataset,
    dataset_name: str,
    device,
) -> dict:
    """Capture immutable execution inputs once at training startup."""
    source = source_tree_record()
    git = git_metadata(source_tree=source)
    runtime = runtime_metadata(device)
    augment = config["data"].get("augment", {}) or {}
    common_preprocessing = {
        "schema": "mdlic-training-data-v1",
        "storage": (
            "uint8 HWC npy -> float32 CHW / 255"
            if dataset_name == "imagenet64_npy"
            else "torchvision CIFAR uint8 HWC -> float32 CHW / 255"
        ),
    }
    datasets = {
        "train": dataset_record(
            train_dataset,
            name=dataset_name,
            split="train",
            configured_path=config["data"].get("train"),
            preprocessing={
                **common_preprocessing,
                "augmentation": augment,
            },
        ),
        "validation": dataset_record(
            valid_dataset,
            name=dataset_name,
            split="test" if dataset_name in ("cifar10", "cifar100") else "val",
            configured_path=config["data"].get("valid"),
            preprocessing={
                **common_preprocessing,
                "augmentation": None,
            },
        ),
    }
    identity = {
        "schema": "mdlic-training-provenance-v1",
        "execution_source_sha256": source["fingerprint_sha256"],
        "runtime": runtime,
        "dataset_fingerprints": {
            split: record["fingerprint_sha256"]
            for split, record in datasets.items()
        },
    }
    return {
        **identity,
        "git": git,
        "datasets": datasets,
        "fingerprint_sha256": _canonical_provenance_hash(identity),
    }


def _broadcast_rank0_object(value, *, distributed: bool, rank: int, device):
    if not distributed:
        return value
    values = [value if rank == 0 else None]
    dist.broadcast_object_list(values, src=0, device=device)
    return values[0]


def _checkpoint_meta(config: dict, args, epoch: int, kind: str, seed: int,
                     metrics: dict, extra: dict = None,
                     provenance: dict | None = None) -> dict:
    """Small, human-readable metadata sidecar for checkpoint provenance."""
    mcfg = config.get("model", {})
    tcfg = config.get("train", {})
    return {
        "metadata_schema": "mdlic-checkpoint-meta-v3",
        "checkpoint_semantics": "weights-only",
        "resume_capable": False,
        "kind": kind,
        "epoch": epoch,
        "exp_name": config.get("exp_name"),
        "config_path": getattr(args, "config", None),
        "resume_path": getattr(args, "resume", None),
        "init_from_path": getattr(args, "init_from", None),
        "initialization_mode": (
            "resume" if getattr(args, "resume", None)
            else "init_from" if getattr(args, "init_from", None)
            else "scratch"
        ),
        "seed": seed,
        "config_sha256": _canonical_config_hash(config),
        "model_config_sha256": _canonical_config_hash(mcfg),
        "model": {
            "type": mcfg.get("type", "igpt"),
            "image_size": mcfg.get("image_size"),
            "in_channels": mcfg.get("in_channels"),
            "vocab_size": mcfg.get("vocab_size"),
            "d_model": mcfg.get("d_model"),
            "N": mcfg.get("N"),
            "h": mcfg.get("h"),
            "coarse_in_channels": mcfg.get("coarse_in_channels"),
            "coarse_d_model": mcfg.get("coarse_d_model"),
            "coarse_N": mcfg.get("coarse_N"),
            "pool_factor": mcfg.get("pool_factor"),
        },
        "train": {
            "amp_dtype": tcfg.get("amp_dtype", "fp16"),
            "batch_size": tcfg.get("batch_size"),
            "grad_accum_steps": tcfg.get("grad_accum_steps", 1),
            "lr": tcfg.get("lr"),
            "epochs": tcfg.get("epochs"),
            "lr_schedule": tcfg.get("lr_schedule", "cosine"),
        },
        "metrics": metrics,
        "extra": extra or {},
        "provenance": provenance,
    }


def _grad_accum_window_size(step_index: int, steps: int, grad_accum_steps: int) -> int:
    """Actual micro-batch count in the accumulation window containing step_index."""
    assert 0 <= step_index < steps, f"step_index={step_index} out of range for steps={steps}"
    assert grad_accum_steps >= 1, f"grad_accum_steps must be >= 1, got {grad_accum_steps}"
    remainder = steps % grad_accum_steps
    if remainder and step_index >= steps - remainder:
        return remainder
    return grad_accum_steps


def _global_weighted_means(metric_sums: dict[str, torch.Tensor], sample_count: int,
                           device, distributed: bool) -> dict[str, float]:
    """Convert sample-weighted sums into means, aggregating all ranks if needed."""
    names = tuple(metric_sums)
    packed = torch.stack([
        *[
            torch.as_tensor(metric_sums[name], device=device, dtype=torch.float64)
            for name in names
        ],
        torch.tensor(float(sample_count), device=device, dtype=torch.float64),
    ])
    if distributed:
        if not (dist.is_available() and dist.is_initialized()):
            raise RuntimeError("distributed metric reduction requires an initialized process group")
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)

    global_count = packed[-1].item()
    if global_count <= 0:
        raise RuntimeError("training loader produced no samples across all ranks")
    return {
        name: packed[index].item() / global_count
        for index, name in enumerate(names)
    }


def train_one_epoch(model, loader, optimizers, scaler, device,
                    epoch, log_freq, writer, clip_max_norm,
                    amp_dtype=torch.float16, grad_accum_steps=1,
                    z_loss_weight=1e-4,
                    distributed=False, rank=0,
                    on_optimizer_step=None):
    """optimizers: list[Optimizer] — 单 AdamW 即可；保留 list 接口兼容历史 ckpt
    格式 (optimizer_state_dicts list)。每个 step 都遍历列表分别 step/zero_grad。
    """
    model.train()
    total_loss = torch.zeros((), device=device, dtype=torch.float64)
    total_bpd = torch.zeros((), device=device, dtype=torch.float64)
    total_samples = 0
    steps = len(loader)

    for opt in optimizers:
        opt.zero_grad(set_to_none=True)

    for i, batch in enumerate(loader):
        if isinstance(batch, (list, tuple)):
            x = batch[0]
        else:
            x = batch

        x = x.to(device)
        batch_size = x.size(0)

        # DDP no_sync: 梯度累积中间步跳过 AllReduce，只在同步步通信。
        # Ref: PyTorch DDP 文档 — `DistributedDataParallel.no_sync()` 上下文
        #      在退出前不会触发梯度 AllReduce，省去 N-1 次跨卡通信。
        #
        # is_last_step: 当 len(loader) 不能被 grad_accum_steps 整除时，
        # epoch 末尾会残留若干个仅完成 backward、未 step 的 micro-batch；
        # 若不强制 flush，下一轮 zero_grad 会把它们清掉，造成梯度信息丢失
        # （等价于每 epoch 训练样本少了 (len(loader) % grad_accum_steps) 个）。
        # 因此最后一步一律视为同步步：执行 step + zero_grad，并打开 AllReduce。
        is_last_step = (i + 1) == steps
        is_accumulating = ((i + 1) % grad_accum_steps != 0) and (not is_last_step)
        current_accum_steps = _grad_accum_window_size(i, steps, grad_accum_steps)
        sync_context = model.no_sync() if (distributed and is_accumulating) else nullcontext()

        with sync_context:
            # AMP autocast：有 scaler 时启用 fp16/bf16 混合精度
            amp_ctx = autocast(device_type="cuda", dtype=amp_dtype) if amp_dtype is not None else nullcontext()
            with amp_ctx:
                out = model(x, z_loss_weight=z_loss_weight)
                loss = out["loss"]
                ce_loss = out["ce_loss"]
                # bits/dim：优先使用模型返回的 bpd（CC-iGPT 联合 coarse+fine
                # 的 bpd_total 已按 H·W·C 子像素数归一化），否则按 iGPT 约定从
                # CE 推算。bpd = bits per dimension/sub-pixel，与 iGPT /
                # PixelCNN++ 等基线原文口径一致；真正的 bits-per-pixel = bpd × C。
                # Ref: Shannon, "A Mathematical Theory of Communication," 1948
                if "bpd" in out and out["bpd"] is not None:
                    bpd = out["bpd"]
                else:
                    bpd = compute_bpd(ce_loss)
            # 残余窗口内的每个 micro-batch 都除以该窗口实际长度，保持梯度均值口径。
            loss_scaled = loss / current_accum_steps
            if scaler is not None:
                scaler.scale(loss_scaled).backward()
            else:
                loss_scaled.backward()

        # 同步步：完整的累积窗口完成 OR epoch 末尾残余 micro-batch（见上方 is_last_step 注释）
        if (i + 1) % grad_accum_steps == 0 or (i + 1) == steps:
            # Unscale → clip → step → update
            if scaler is not None:
                for opt in optimizers:
                    scaler.unscale_(opt)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
            if scaler is not None:
                for opt in optimizers:
                    scaler.step(opt)
                scaler.update()
            else:
                for opt in optimizers:
                    opt.step()

            if writer and (i + 1) % log_freq == 0:
                step = epoch * steps + i
                writer.add_scalar('train/grad_norm', grad_norm.item(), step)
                if scaler is not None:
                    writer.add_scalar('train/loss_scale', scaler.get_scale(), step)

            for opt in optimizers:
                opt.zero_grad(set_to_none=True)

            if on_optimizer_step is not None:
                on_optimizer_step()

        # 用 tensor 累加，避免每步 .item() 触发 GPU→CPU 同步
        # Ref: CS336 — 只在 log 时才 .item()
        total_loss += loss.detach().double() * batch_size
        total_bpd += bpd.detach().double() * batch_size
        total_samples += batch_size

        # NaN/Inf 检测：loss 异常时提前警告，避免浪费 GPU 时间
        # Ref: PyTorch Lightning — NaN detection callback 的设计思路
        if (i + 1) % log_freq == 0:
            log_sums = {
                "loss": loss.detach().double() * batch_size,
                "bpd": bpd.detach().double() * batch_size,
            }
            if "ce_loss_coarse" in out and "ce_loss_fine" in out:
                log_sums.update({
                    "ce_coarse": out["ce_loss_coarse"].detach().double() * batch_size,
                    "ce_fine": out["ce_loss_fine"].detach().double() * batch_size,
                    "ctx_alpha": out["ctx_alpha"].detach().double() * batch_size,
                })
            log_means = _global_weighted_means(
                log_sums, batch_size, device=device, distributed=distributed,
            )
            loss_val = log_means["loss"]
            bpd_val = log_means["bpd"]
            if not math.isfinite(loss_val) and rank == 0:
                print(f"WARNING: loss is {loss_val} at epoch {epoch} step {i+1}/{steps}. "
                      f"LR={optimizers[0].param_groups[0]['lr']:.2e}. Training may diverge.")
            if rank == 0:
                # CC-iGPT 路径：loss = ce_c + z·z_c + ce_f + z·z_f，数值大但视觉上让 CE_c
                # 错觉主导，实际 bpd_total 中 coarse 仅 ~5%（N_c=64 vs N_f=3072）。
                # 这里改为 CC-iGPT 只打印 bpd + CE 分解 + α，省略易误读的 Loss 汇总；
                # iGPT 单尺度无 coarse overhead，保留 Loss + bpd 双轴。
                if "ce_loss_coarse" in out and "ce_loss_fine" in out:
                    print(f"Epoch {epoch} Step {i+1}/{steps} | bits/dim: {bpd_val:.4f}"
                          f" | CE_c: {log_means['ce_coarse']:.4f}"
                          f" | CE_f: {log_means['ce_fine']:.4f}"
                          f" | α: {log_means['ctx_alpha']:.3f}")
                else:
                    print(f"Epoch {epoch} Step {i+1}/{steps} | Loss: {loss_val:.4f} | bits/dim: {bpd_val:.4f}")
            if writer:
                step = epoch * steps + i
                writer.add_scalar('train/loss', loss_val, step)
                writer.add_scalar('train/bpd',  bpd_val,  step)
                if "ce_loss_coarse" in out and "ce_loss_fine" in out:
                    writer.add_scalar('train/ce_coarse', log_means['ce_coarse'], step)
                    writer.add_scalar('train/ce_fine',   log_means['ce_fine'],   step)
                    writer.add_scalar('train/ctx_alpha', log_means['ctx_alpha'], step)

    return_values = _global_weighted_means(
        {"loss": total_loss, "bpd": total_bpd},
        total_samples,
        device=device,
        distributed=distributed,
    )
    return return_values["loss"], return_values["bpd"]


# ==================== Validation ====================
@torch.no_grad()
def validate(model, loader, device, amp_dtype=None):
    """验证集评估，返回 (avg_bpd, std_bpd_per_image, avg_loss)。

    DDP-aware：每个 rank 通过无 padding sampler 处理唯一分片，最后用
    all_reduce(SUM) 聚合 weighted sums + n_total。调用方必须传 unwrapped model，
    因为不等长分片不能安全经过可能执行 buffer collective 的 DDP forward。

    amp_dtype: 传入 AMP dtype（如 torch.bfloat16）以在验证时也使用混合精度，
               减少显存占用和加速。None 则使用 fp32。

    std_bpd 是逐图样本标准差，与 evaluate.py 使用同一共享实现；改变 batch size
    或 DDP 分片不会改变统计含义。
    """
    model.eval()
    total_bpd_weighted = 0.0
    total_loss_weighted = 0.0
    sum_sq_bpd_image = 0.0
    n_total = 0
    use_amp = amp_dtype is not None and device.type == 'cuda'
    for batch in loader:
        if isinstance(batch, (list, tuple)):
            x = batch[0]
        else:
            x = batch
        x = x.to(device)
        B = x.size(0)
        with autocast(device_type="cuda", dtype=amp_dtype) if use_amp else nullcontext():
            out = model(x)
        loss = out["loss"]
        bpd_image = per_image_bpd(model, x, out)
        loss_val = loss.item()
        total_bpd_weighted += bpd_image.double().sum().item()
        total_loss_weighted += loss_val * B
        sum_sq_bpd_image += bpd_image.double().square().sum().item()
        n_total += B

    # DDP 聚合：跨 rank 求和后再求平均，结果在所有 rank 上一致
    if dist.is_available() and dist.is_initialized():
        agg = torch.tensor([total_bpd_weighted, total_loss_weighted, sum_sq_bpd_image, float(n_total)],
                           device=device, dtype=torch.float64)
        dist.all_reduce(agg, op=dist.ReduceOp.SUM)
        total_bpd_weighted, total_loss_weighted, sum_sq_bpd_image, n_total_f = agg.tolist()
        n_total = int(n_total_f)

    if n_total <= 0:
        raise RuntimeError("validation loader produced no samples across all ranks")
    avg_bpd = total_bpd_weighted / n_total
    avg_loss = total_loss_weighted / n_total
    centered_sum = max(sum_sq_bpd_image - n_total * avg_bpd ** 2, 0.0)
    std_bpd_image = math.sqrt(centered_sum / (n_total - 1)) if n_total > 1 else 0.0
    return avg_bpd, std_bpd_image, avg_loss



# ==================== Main ====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config yaml')
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument(
        '--resume', type=str, default=None,
        help='Strictly restore a complete MDLIC training-state checkpoint',
    )
    initialization.add_argument(
        '--init_from', type=str, default=None,
        help='Initialize shape-compatible model weights only; optimizer/epoch/RNG start fresh',
    )
    parser.add_argument(
        '--allow_legacy_resume', action='store_true',
        help='Allow v2 resume checkpoints that lack source/runtime/data provenance',
    )
    # --seed: 命令行覆盖 config 中的 seed，方便多次独立运行取均值
    # 用法: python train.py --config ... --seed 0 / --seed 1 / --seed 2
    parser.add_argument('--seed', type=int, default=None, help='Random seed (overrides config)')
    # --export_csv: 导出训练/验证曲线为 CSV，方便 matplotlib 画论文图
    parser.add_argument('--export_csv', action='store_true',
                        help='导出 loss/bpd/LR 曲线为 CSV 文件')
    args = parser.parse_args()
    if args.allow_legacy_resume and not args.resume:
        parser.error("--allow_legacy_resume requires --resume")

    if not os.path.isfile(args.config):
        raise FileNotFoundError(f"配置文件不存在: {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict) or "model" not in config or "train" not in config:
        raise ValueError(f"配置文件格式非法，需包含 model / train 字段: {args.config}")

    # 配置完整性校验：尽早发现错误，避免 GPU 资源浪费
    _validate_config(config)

    # Seed: CLI --seed 优先于 config，方便消融实验多次独立运行
    seed = args.seed if args.seed is not None else config["train"].get("seed", 42)
    seed_everything(seed)

    exp_name = config['exp_name']
    checkpoint_dir = os.path.join('experiments', exp_name, 'checkpoints')
    log_dir = os.path.join('experiments', exp_name, 'logs')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # CSV 导出器延迟到 rank 检测之后初始化（仅 rank 0 写盘，避免 DDP 下多 rank
    # 用 'w' 模式同时打开同一文件造成 truncate race / 半截文件）。
    csv_file = None
    csv_writer = None

    # Distributed training: 自动检测 torchrun 环境
    # torchrun 会设置 WORLD_SIZE、RANK、LOCAL_RANK 等环境变量，
    # 无需在 config 中手动指定 distributed.enabled。
    # Ref: PyTorch Distributed — torchrun elastic launch 文档
    distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1
    if distributed:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        # 训练和验证指标都在各 rank 聚合；放宽超时以容纳长序列评测及 checkpoint I/O。
        from datetime import timedelta
        dist.init_process_group(backend='nccl', timeout=timedelta(hours=2))
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        local_rank = 0
        rank = 0
        world_size = 1

    device = torch.device(f'cuda:{local_rank}' if distributed else 'cuda' if torch.cuda.is_available() else 'cpu')
    writer = SummaryWriter(log_dir=log_dir) if rank == 0 else None

    csv_path = (os.path.join(log_dir, 'training_curves.csv')
                if args.export_csv and rank == 0 else None)

    # Dataset: 根据 config 选择 CIFAR-10 / CIFAR-100 / ImageNet64 npy
    from torchvision.datasets import CIFAR10, CIFAR100
    aug_cfg = config["data"].get("augment", {}) or {}
    train_tf_list = []
    crop_cfg = aug_cfg.get("random_crop") or {}
    if crop_cfg:
        train_tf_list.append(transforms.RandomCrop(
            size=crop_cfg.get("size", 32),
            padding=crop_cfg.get("padding", 4),
            padding_mode=crop_cfg.get("padding_mode", "reflect"),
        ))
    if aug_cfg.get("hflip", False):
        train_tf_list.append(transforms.RandomHorizontalFlip(p=0.5))
    train_tf_list.append(transforms.ToTensor())
    train_transform = transforms.Compose(train_tf_list)
    valid_transform = transforms.ToTensor()
    if rank == 0:
        active = []
        if crop_cfg:
            active.append(f"RandomCrop(size={crop_cfg.get('size',32)},pad={crop_cfg.get('padding',4)},{crop_cfg.get('padding_mode','reflect')})")
        if aug_cfg.get("hflip", False):
            active.append("RandomHorizontalFlip(p=0.5)")
        if active:
            print("Augment (train only): " + " + ".join(active))
    dataset_name = config["data"].get("dataset", "cifar100")

    if dataset_name in ("cifar10", "cifar100"):
        DatasetClass = CIFAR10 if dataset_name == "cifar10" else CIFAR100
        train_dataset = DatasetClass(root=config["data"]["train"], train=True,  download=False, transform=train_transform)
        valid_dataset = DatasetClass(root=config["data"]["valid"], train=False, download=False, transform=valid_transform)
    elif dataset_name == "imagenet64_npy":
        from mdlic.data.imagenet64_npy import ImageNet64Npy
        if rank == 0 and crop_cfg:
            print("WARNING: data.augment.random_crop 在 imagenet64_npy 路径上当前未实现，已忽略")
        hflip_flag = bool(aug_cfg.get("hflip", False))
        train_dataset = ImageNet64Npy(root=config["data"]["train"], split="train", hflip=hflip_flag)
        valid_dataset = ImageNet64Npy(root=config["data"]["valid"], split="val", hflip=False)
    else:
        raise ValueError(f"未知 dataset: '{dataset_name}'，支持 cifar10/cifar100/imagenet64_npy")
    if rank == 0:
        print(f"Dataset: {dataset_name} | Train: {len(train_dataset)} | Valid: {len(valid_dataset)}")

    training_provenance = (
        _build_training_provenance(
            config, train_dataset, valid_dataset, dataset_name, device,
        )
        if rank == 0 else None
    )
    training_provenance = _broadcast_rank0_object(
        training_provenance,
        distributed=distributed,
        rank=rank,
        device=device,
    )
    if rank == 0:
        print(
            "Provenance: "
            f"source={training_provenance['execution_source_sha256'][:12]} "
            f"run={training_provenance['fingerprint_sha256'][:12]}"
        )

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if distributed else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['train']['batch_size'],
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=config['data']['num_workers'],
        pin_memory=True,
        drop_last=True,
    )
    # 验证集: batch_size 和 num_workers 从 config 读取，支持不同 GPU 显存调整
    valid_batch_size = config['data'].get('valid_batch_size', 64)
    valid_num_workers = config['data'].get('valid_num_workers', 2)
    # No padding and no drop: every validation sample appears on exactly one rank.
    # Rank shard lengths may differ by one, so validate() uses raw_model below.
    valid_sampler = DistributedEvalSampler(valid_dataset) if distributed else None
    valid_loader = DataLoader(valid_dataset, batch_size=valid_batch_size, shuffle=False,
                              sampler=valid_sampler,
                              num_workers=valid_num_workers, pin_memory=True)

    # Model — 根据 type 分发构建
    mcfg = config["model"]
    model_type = mcfg.get("type", "igpt")

    if model_type == "ccigpt":
        model = _build_ccigpt_from_config(mcfg, device)
        if rank == 0:
            n_params = sum(p.numel() for p in model.parameters())
            n_coarse = sum(p.numel() for p in model.coarse.parameters())
            n_fine   = sum(p.numel() for p in model.fine.parameters())
            print(f"CC-iGPT model: {n_params:,} params "
                  f"(coarse {n_coarse:,} + fine {n_fine:,}), "
                  f"pool_factor={mcfg['pool_factor']}, "
                  f"coarse_seq={model.coarse.seq_len}, fine_seq={model.fine.seq_len}")
    else:
        model = _build_model_from_config(mcfg, device)

    # Fused kernel 状态日志（iGPT/CC-iGPT 使用 Triton kernels）
    if rank == 0 and model_type in ('igpt', 'ccigpt'):
        kernel_status = get_fused_kernel_status()
        active = sum(kernel_status.values())
        print(f"Fused Triton kernels: {active}/{len(kernel_status)} active")
        for name, avail in kernel_status.items():
            print(f"  {name}: {'ON' if avail else 'OFF (fallback to PyTorch)'}")

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank
        )

    # Optimizer: 单 AdamW
    base_lr = float(config["train"]["lr"])
    optimizer_main = optim.AdamW(
        _get_param_groups(model, weight_decay=0.1),
        lr=base_lr,
        betas=(0.9, 0.95),
        eps=float(config["train"].get("eps", 1e-8)),
    )
    optimizers = [optimizer_main]

    # Learning rate scheduler
    #
    # Cosine decay with linear warmup（默认）:
    #   参考:
    #     [1] OLMo 2 Tech Report, arXiv:2501.00656, 2025, Section 3.3.
    #     [2] Loshchilov & Hutter, "SGDR," arXiv:1608.03983, 2016.
    #     [3] CS336 "Language Models from Scratch," Stanford, Spring 2024.
    #
    # WSD (Warmup-Stable-Decay):
    #   参考:
    #     [4] Hu et al., "MiniCPM," arXiv:2404.06395, 2024.
    #         三阶段 schedule: warmup → stable (lr=1) → power-law decay.
    #     [5] Hagele et al., "Scaling Data-Constrained LMs," arXiv:2405.18392, 2024.
    lr_schedule = config["train"].get("lr_schedule", "cosine")
    warmup_epochs = config["train"].get("warmup_epochs", 5)
    total_epochs  = config["train"]["epochs"]

    if lr_schedule == "cosine":
        # min_lr_ratio: cosine 末段 LR 下限相对峰值的比例（默认 0.0 即衰到 0）。
        # 设为 >0 可避免末段 LR 过低导致 SWA 平均的是几乎相同的快照（"假平均"）。
        # Ref: Hagele et al., arXiv:2405.18392 §4.2 — SWA 需要权重仍在变化才有意义。
        min_lr_ratio = float(config["train"].get("min_lr_ratio", 0.0))
        assert 0.0 <= min_lr_ratio < 1.0, (
            f"train.min_lr_ratio 必须在 [0, 1)，got {min_lr_ratio}"
        )
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return float(epoch + 1) / float(max(1, warmup_epochs))
            progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
        scheduler = optim.lr_scheduler.LambdaLR(optimizers[0], lr_lambda)
    elif lr_schedule == "wsd":
        # WSD: Warmup-Stable-Decay
        # warmup: 线性 0→1（warmup_epochs 个 epoch）
        # stable: 维持 1.0（到 stable_epochs）
        # decay:  (1 - progress)^beta（stable_epochs 到 total_epochs）
        # Ref: MiniCPM arXiv:2404.06395 [4]
        stable_epochs = config["train"].get("stable_epochs", 70)
        decay_beta = config["train"].get("decay_beta", 1.0)
        def lr_lambda_wsd(epoch):
            if epoch < warmup_epochs:
                # Warmup: 线性从 0 到 1
                return float(epoch + 1) / float(max(1, warmup_epochs))
            elif epoch < stable_epochs:
                # Stable: 维持峰值 LR
                return 1.0
            else:
                # Decay: power-law 衰减 (1 - progress)^beta
                decay_length = max(1, total_epochs - stable_epochs)
                progress = float(epoch - stable_epochs) / float(decay_length)
                return max(0.0, (1.0 - progress) ** decay_beta)
        scheduler = optim.lr_scheduler.LambdaLR(optimizers[0], lr_lambda_wsd)
    else:
        # 经 _validate_config 限定 ∈ {cosine, wsd}；此分支理论不可达，留作防御性兜底
        scheduler = None

    # Mixed precision: None/"none"/"fp32" → 禁用 AMP 走 fp32（数值敏感场景兜底）
    amp_cfg = config["train"].get("amp_dtype", "fp16")
    if amp_cfg in (None, "none", "fp32"):
        amp_dtype = None
    elif amp_cfg == "bf16":
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = torch.float16
    scaler = GradScaler("cuda") if amp_dtype == torch.float16 else None
    grad_accum_steps = config["train"].get("grad_accum_steps", 1)

    # SWA (Stochastic Weight Averaging) 初始化
    # 手写实现，不用 torch.optim.swa_utils
    # Ref: Izmailov et al., "Averaging Weights Leads to Wider Optima and
    #       Better Generalization," UAI 2018, arXiv:1803.05407.
    # Ref: Hagele et al., arXiv:2405.18392 — WSD + SWA 组合
    swa_cfg = config["train"].get("swa", {})
    swa_enabled = swa_cfg.get("enabled", False)
    swa_start_epoch = swa_cfg.get("start_epoch", 80)
    swa_update_interval = swa_cfg.get("update_interval", 1)
    swa_state = None
    swa_n = 0  # SWA 累计更新次数

    if swa_enabled:
        # 延迟初始化：在第一次 SWA 更新时 clone 权重
        if rank == 0:
            print(f"SWA enabled: start_epoch={swa_start_epoch}, interval={swa_update_interval}")

    # EMA (Exponential Moving Average) 初始化
    # rank 0 维护 fp32 影子权重；每个 optimizer.step() 后更新；finalize 时
    # broadcast 到全 rank → validate → 保存 ema.pth。
    # Ref: Polyak & Juditsky 1992；iGPT (Chen 2020) / EDM / Stable Diffusion 标配。
    ema_cfg = config["train"].get("ema", {})
    ema_enabled = ema_cfg.get("enabled", False)
    ema_decay = float(ema_cfg.get("decay", 0.999))
    ema_state = None
    _ema_tick = 0
    # NaN 检查降频：每 N 个 ema 更新一次。每步 .item() 会触发 GPU→CPU 同步
    # （在 32 层 ~600 参数上）；NaN 极其罕见，间隔 100 步检测一次足够保险。
    _ema_nan_check_interval = 100
    if ema_enabled and rank == 0:
        print(f"EMA enabled: decay={ema_decay} (NaN check every {_ema_nan_check_interval} ticks)")

    best_bpd = float('inf')
    start_epoch = 1
    legacy_resume_used = False
    legacy_lineage_unverified = False

    # ── DDP / torch.compile state_dict 统一处理 ──
    # DistributedDataParallel 会在所有参数名前加 `module.` 前缀
    # (torch/nn/parallel/distributed.py: DDP.__init__ 将原 module 挂到 self.module)，
    # torch.compile 会再包一层 OptimizedModule，state_dict key 多 `_orig_mod.` 前缀。
    # 两者可能同时出现（DDP(compile(model)) 时 key 形如 `module._orig_mod.xxx`）。
    # 若直接保存 wrap 后的 `model.state_dict()`，单卡裸模型加载时全部 key mismatch。
    # 这里统一用 raw_model（解 DDP + 解 compile 后的原始 nn.Module）来保存；加载
    # 时再用 clean_state_dict 兼容历史遗留的前缀 / inv_freq buffer。
    #
    # Ref: PyTorch DDP Tutorial
    #      https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html
    #      ("Save and Load Checkpoints" 小节推荐 `model.module.state_dict()`)
    raw_model = model.module if distributed else model
    # torch.compile 包装后 raw_model 是 OptimizedModule，真正的 nn.Module 在
    # _orig_mod 上；保存其 state_dict 才能被未 compile 的脚本直接 load。
    raw_model = getattr(raw_model, '_orig_mod', raw_model)

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        legacy_resume_used = ckpt.get("schema") == LEGACY_TRAINING_STATE_SCHEMA
        legacy_lineage_unverified = legacy_resume_used or bool(
            ckpt.get("provenance", {}).get("legacy_resume_source_unverified", False)
        )
        _validate_resume_checkpoint(
            ckpt,
            config=config,
            seed=seed,
            world_size=world_size,
            provenance_fingerprint=training_provenance["fingerprint_sha256"],
            allow_legacy_resume=args.allow_legacy_resume,
            optimizer_count=len(optimizers),
            scheduler_required=scheduler is not None,
            scaler_required=scaler is not None,
            swa_enabled=swa_enabled,
            ema_enabled=ema_enabled,
        )
        raw_model.load_state_dict(clean_state_dict(ckpt['model_state_dict']), strict=True)
        for opt, opt_state in zip(optimizers, ckpt['optimizer_state_dicts']):
            opt.load_state_dict(opt_state)
        if scheduler is not None:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        if scaler is not None:
            scaler.load_state_dict(ckpt['scaler_state_dict'])
        if swa_enabled:
            swa_state = ckpt['swa_state'] if rank == 0 else None
            swa_n = ckpt['swa_n'] if rank == 0 else 0
        if ema_enabled:
            ema_state = ckpt['ema_state'] if rank == 0 else None
            _ema_tick = ckpt['ema_tick']
        start_epoch = ckpt['epoch'] + 1
        best_bpd = ckpt['best_bpd']
        _restore_rng_state(ckpt['rng_states_by_rank'][rank])
        if legacy_lineage_unverified:
            training_provenance["legacy_resume_source_unverified"] = True
        if rank == 0:
            current_lr = optimizers[0].param_groups[0]['lr']
            resume_label = "Legacy-resumed" if legacy_resume_used else "Strictly resumed"
            print(
                f"{resume_label} '{args.resume}' at epoch {ckpt['epoch']} "
                f"(next={start_epoch}, best_bpd={best_bpd:.4f}, lr={current_lr:.2e})"
            )
        del ckpt
    elif args.init_from:
        ckpt = torch.load(args.init_from, map_location=device, weights_only=False)
        init_stats = _load_init_from_state_dict(raw_model, ckpt)
        if rank == 0:
            print(
                f"Initialized weights from '{args.init_from}': "
                f"{init_stats['matched_keys']} keys, "
                f"{init_stats['parameter_coverage']:.2%} trainable parameters; "
                f"missing={init_stats['missing_keys']}, "
                f"unexpected={init_stats['unexpected_keys']}, "
                f"shape_mismatch={init_stats['shape_mismatch_keys']}. "
                "Optimizer/scheduler/RNG start fresh at epoch 1."
            )
        del ckpt

    if csv_path is not None:
        csv_file, csv_writer = _open_training_csv(
            csv_path, resume=bool(args.resume), start_epoch=start_epoch,
        )

    def _ema_step():
        """rank 0 上每个 optimizer.step() 后调用：用 fp32 累加更新影子权重。

        ema_state 延迟初始化：第一次调用时按 raw_model 当前权重 clone 出 fp32 副本。
        非 rank 0 走空操作（ema_state 始终 None，由 finalize broadcast 同步）。

        NaN 守卫：与 SWA 同款 batched 检查 — EMA 是累积平均，单步 NaN
        会通过 (1-decay) 项渗入并永久污染；首次克隆时 NaN 也会让所有
        后续 add_ 输出 NaN。每 _ema_nan_check_interval 个 tick 检查一次，
        避免每步 .item() GPU→CPU 同步。
        """
        nonlocal ema_state, _ema_tick
        if not ema_enabled or rank != 0:
            return
        _ema_tick += 1
        do_nan_check = (_ema_tick % _ema_nan_check_interval == 0) or (ema_state is None)
        if do_nan_check:
            has_nan = torch.stack([torch.isnan(p.data).any()
                                   for p in raw_model.parameters()]).any().item()
            if has_nan:
                return
        if ema_state is None:
            ema_state = {name: p.data.detach().float().clone()
                         for name, p in raw_model.named_parameters()}
            return
        for name, p in raw_model.named_parameters():
            ema_state[name].mul_(ema_decay).add_(p.data.float(), alpha=1.0 - ema_decay)

    for epoch in range(start_epoch, config['train']['epochs'] + 1):
        if distributed:
            train_sampler.set_epoch(epoch)

        # ── 训练一个 epoch (iGPT/CC-iGPT 共用 NTP 训练循环) ──
        avg_loss, avg_bpd = train_one_epoch(
            model, train_loader, optimizers, scaler,
            device=device,
            epoch=epoch,
            log_freq=config['train']['log_freq'],
            writer=writer,
            clip_max_norm=config['train']['clip_max_norm'],
            amp_dtype=amp_dtype,
            grad_accum_steps=grad_accum_steps,
            z_loss_weight=float(config["train"].get("z_loss_weight", 1e-4)),
            distributed=distributed,
            rank=rank,
            on_optimizer_step=_ema_step if ema_enabled else None,
        )

        if rank == 0:
            current_lr = optimizers[0].param_groups[0]['lr']
            # CC-iGPT 路径下 avg_loss 是 ce_c+z+ce_f+z 之和，非 bpd 同口径，省略避免误读；
            # iGPT 单尺度无 coarse overhead，loss ≈ ce_loss，保留供监控参考。
            if model_type == "ccigpt":
                print(f"Epoch {epoch} | bits/dim: {avg_bpd:.4f} | LR: {current_lr:.2e}")
            else:
                print(f"Epoch {epoch} | Loss: {avg_loss:.4f} | bits/dim: {avg_bpd:.4f} | LR: {current_lr:.2e}")
            if writer:
                writer.add_scalar('train/lr', current_lr, epoch)

        # ── 验证 (DDP: 所有 rank 跑分片，all_reduce 聚合；rank 0 写日志/保存) ──
        if epoch % config['eval']['interval'] == 0:
            bpd_avg, std_bpd, loss_avg = validate(
                raw_model, valid_loader, device, amp_dtype=amp_dtype,
            )
            if rank == 0:
                print(f"Validation: Loss {loss_avg:.4f} | bits/dim {bpd_avg:.4f} ± {std_bpd:.4f}")
                if writer:
                    writer.add_scalar('val/loss', loss_avg, epoch)
                    writer.add_scalar('val/bpd', bpd_avg, epoch)
                    writer.add_scalar('val/bpd_std', std_bpd, epoch)
                if csv_writer:
                    current_lr = optimizers[0].param_groups[0]['lr']
                    csv_writer.writerow([epoch, f'{avg_loss:.6f}', f'{avg_bpd:.6f}',
                                         f'{loss_avg:.6f}', f'{bpd_avg:.6f}',
                                         f'{std_bpd:.6f}', f'{current_lr:.2e}'])
                    csv_file.flush()
                if bpd_avg < best_bpd:
                    best_bpd = bpd_avg
                    _atomic_save(raw_model.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))
                    current_lr = optimizers[0].param_groups[0]['lr']
                    _atomic_save_json(
                        _checkpoint_meta(
                            config, args, epoch, "best", seed,
                            metrics={
                                "train_loss": avg_loss,
                                "train_bpd": avg_bpd,
                                "val_loss": loss_avg,
                                "val_bpd": bpd_avg,
                                "val_bpd_std": std_bpd,
                                "best_bpd": best_bpd,
                                "lr": current_lr,
                            },
                            extra={"checkpoint": "best.pth"},
                            provenance=training_provenance,
                        ),
                        os.path.join(checkpoint_dir, 'best.meta.json'),
                    )

        if scheduler is not None:
            scheduler.step()

        # SWA 更新：running average of model weights，**用 fp32 累加**避免
        # bf16 mantissa (~7e-3) 在 21+ checkpoint 上累计的舍入误差。
        # swa_state[name] = swa_state[name] + (param - swa_state[name]) / (n+1)
        #                  = swa_state[name].lerp_(param, 1/(n+1))
        # Ref: Izmailov et al., arXiv:1803.05407
        if swa_enabled and rank == 0 and epoch >= swa_start_epoch and (epoch - swa_start_epoch) % swa_update_interval == 0:
            # 一次性 batched NaN 检查，避免逐参数 .item() 多次同步
            has_nan = torch.stack([torch.isnan(p.data).any() for p in raw_model.parameters()]).any().item()
            if has_nan:
                # 跳过本 tick，保留 swa_state（无论 None 或已累积），下一 tick 自动回到正确分支
                print(f"  WARNING: skipping SWA update at epoch {epoch} — model contains NaN weights")
            elif swa_state is None:
                # 首次以 fp32 拷贝（含"NaN 跳过若干 tick 后才首次成功累积"的情况）
                swa_state = {name: param.data.detach().float().clone()
                             for name, param in raw_model.named_parameters()}
                swa_n = 1
            else:
                swa_n += 1
                for name, param in raw_model.named_parameters():
                    # 在 fp32 域更新（param 自动 upcast）
                    swa_state[name].lerp_(param.data.float(), 1.0 / swa_n)
            if not has_nan:
                print(f"  SWA update #{swa_n} at epoch {epoch}")

        should_save_epoch = (
            epoch % config['checkpoint']['save_interval'] == 0
            and epoch >= config['checkpoint'].get('save_start_epoch', 1)
        )
        rng_states_by_rank = (
            _gather_rng_states(distributed, rank) if should_save_epoch else None
        )
        if rank == 0 and should_save_epoch:
            # 完整保存训练状态，确保 --resume 后所有组件正确恢复
            ckpt_data = {
                'schema': TRAINING_STATE_SCHEMA,
                'resume_semantics': 'strict-exact-state-v3',
                'checkpoint_semantics': 'complete-training-state',
                'epoch': epoch,
                'model_state_dict': raw_model.state_dict(),
                # 多 optimizer 格式：list of state_dicts（当前为单元素 list）
                'optimizer_state_dicts': [opt.state_dict() for opt in optimizers],
                'loss': avg_loss,
                'best_bpd': best_bpd,
                'config_sha256': _canonical_config_hash(config),
                'model_config_sha256': _canonical_config_hash(config['model']),
                'seed': seed,
                'world_size': world_size,
                'rng_states_by_rank': rng_states_by_rank,
                'provenance': training_provenance,
                'provenance_sha256': training_provenance['fingerprint_sha256'],
                'initialization': {
                    'mode': (
                        'resume' if args.resume
                        else 'init_from' if args.init_from
                        else 'scratch'
                    ),
                    'resume_path': args.resume,
                    'init_from_path': args.init_from,
                    'legacy_provenance_unverified': legacy_lineage_unverified,
                },
            }
            if scheduler is not None:
                ckpt_data['scheduler_state_dict'] = scheduler.state_dict()
            if scaler is not None:
                ckpt_data['scaler_state_dict'] = scaler.state_dict()
            if swa_enabled:
                ckpt_data['swa_state'] = swa_state
                ckpt_data['swa_n'] = swa_n
            if ema_enabled:
                ckpt_data['ema_state'] = ema_state
                ckpt_data['ema_tick'] = _ema_tick
            _atomic_save(ckpt_data, os.path.join(checkpoint_dir, f'epoch_{epoch}.pth'))

    # EMA finalize：rank 0 替换权重 + broadcast → 全 rank 重新验证 → rank 0 保存
    # 顺序上必须在 SWA finalize 之前，否则会被 SWA 替换后的权重覆盖。
    # 同样的 DDP 死锁规避：rank 0 广播 0/1 标量决定全 rank 是否同时 finalize。
    if ema_enabled:
        ema_done = torch.tensor(
            [1 if ema_state is not None else 0],
            device=device, dtype=torch.int32,
        )
        if dist.is_available() and dist.is_initialized():
            dist.broadcast(ema_done, src=0)

        if ema_done.item() == 1:
            # 备份训练末权重，EMA 验证完后还原（避免覆盖到 SWA finalize 输入）
            backup_state = {name: p.data.detach().clone()
                            for name, p in raw_model.named_parameters()}
            if rank == 0:
                for name, p in raw_model.named_parameters():
                    p.data.copy_(ema_state[name].to(p.dtype))
                print(f"EMA: replaced model weights (decay={ema_decay})")
            if dist.is_available() and dist.is_initialized():
                for p in raw_model.parameters():
                    dist.broadcast(p.data, src=0)
            bpd_avg, std_bpd, loss_avg = validate(
                raw_model, valid_loader, device, amp_dtype=amp_dtype,
            )
            if rank == 0:
                print(f"EMA Validation: Loss {loss_avg:.4f} | bits/dim {bpd_avg:.4f} ± {std_bpd:.4f}")
                _atomic_save(raw_model.state_dict(), os.path.join(checkpoint_dir, 'ema.pth'))
                _atomic_save_json(
                    _checkpoint_meta(
                        config, args, total_epochs, "ema", seed,
                        metrics={
                            "val_loss": loss_avg,
                            "val_bpd": bpd_avg,
                            "val_bpd_std": std_bpd,
                        },
                        extra={"checkpoint": "ema.pth", "ema_decay": ema_decay},
                        provenance=training_provenance,
                    ),
                    os.path.join(checkpoint_dir, 'ema.meta.json'),
                )
                if writer:
                    writer.add_scalar('val/ema_bpd', bpd_avg, total_epochs)
            # 还原训练末权重，给 SWA finalize 干净的输入
            for name, p in raw_model.named_parameters():
                p.data.copy_(backup_state[name])

    # SWA 后处理：rank 0 替换权重 + broadcast → 全 rank 重新验证 → rank 0 保存
    #
    # DDP 死锁规避：SWA 状态只在 rank 0 上累积（swa_state 在其余 rank 永远是 None），
    # 因此 finalize 必须由 rank 0 来"发起"、其余 rank 通过 broadcast 同步参与。
    # 直接用 `swa_state is not None` 守护 → 仅 rank 0 进入 broadcast/validate，
    # 其余 rank 退出循环，集合通信永久挂起。这里改用 rank 0 广播一个 0/1 标量
    # 决定全 rank 是否同时进入 finalize 路径。
    if swa_enabled:
        swa_done = torch.tensor(
            [1 if swa_state is not None else 0],
            device=device, dtype=torch.int32,
        )
        if dist.is_available() and dist.is_initialized():
            dist.broadcast(swa_done, src=0)

        if swa_done.item() == 1:
            if rank == 0:
                for name, param in raw_model.named_parameters():
                    param.data.copy_(swa_state[name])
                print(f"SWA: replaced model weights (averaged over {swa_n} checkpoints)")
            if dist.is_available() and dist.is_initialized():
                for param in raw_model.parameters():
                    dist.broadcast(param.data, src=0)
            bpd_avg, std_bpd, loss_avg = validate(
                raw_model, valid_loader, device, amp_dtype=amp_dtype,
            )
            if rank == 0:
                print(f"SWA Validation: Loss {loss_avg:.4f} | bits/dim {bpd_avg:.4f} ± {std_bpd:.4f}")
                _atomic_save(raw_model.state_dict(), os.path.join(checkpoint_dir, 'swa.pth'))
                _atomic_save_json(
                    _checkpoint_meta(
                        config, args, total_epochs, "swa", seed,
                        metrics={
                            "val_loss": loss_avg,
                            "val_bpd": bpd_avg,
                            "val_bpd_std": std_bpd,
                        },
                        extra={"checkpoint": "swa.pth", "swa_n": swa_n},
                        provenance=training_provenance,
                    ),
                    os.path.join(checkpoint_dir, 'swa.meta.json'),
                )
                if writer:
                    writer.add_scalar('val/swa_bpd', bpd_avg, total_epochs)
        elif rank == 0:
            print("SWA: 训练期间未发生任何 SWA 更新（swa_start_epoch 可能大于实际 epoch 数），跳过 finalize")

    if rank == 0:
        if csv_file:
            csv_file.close()
            print(f"Training curves saved to {os.path.join(log_dir, 'training_curves.csv')}")
        if writer:
            writer.close()
        print("Training finished.")

    if distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
