#!/usr/bin/env python3
"""
评测脚本 — 加载训练好的 checkpoint，计算 CIFAR test set bits/dim (bpd)，
并与传统方法（PNG/WebP）和学术 baseline 对比。

术语: 本仓库主指标为 **bpd (bits per dimension/sub-pixel)**，与 iGPT /
PixelCNN++ / Sparse Transformer 等基线原文口径一致；
真正的 bits-per-pixel = bpd × C（彩色图 C=3）。

功能:
  1. 单模型评测: --checkpoint best.pth
  2. SWA checkpoint 对比: --swa (同时评测 best.pth 和 swa.pth)
  3. 传统方法对比: --traditional (PNG/WebP lossless bpd)
  4. TTA hflip: --tta_hflip（诊断分数；当前 codec 未实现该协议）

输出 Markdown 格式的对比表格，可直接粘贴到论文中。

Usage:
    # 单模型评测（主表协议：单 checkpoint、无 TTA）
    python scripts/evaluate.py --config configs/ccigpt_cifar10_s_rgb_ronly_v2.yaml \
        --checkpoint experiments/ccigpt_cifar10_s_rgb_ronly_v2/checkpoints/best.pth \
        --per_image_stats

    # SWA vs best 对比 (v2 配置启用了 SWA last 31 ckpts, start ep170)
    python scripts/evaluate.py --config configs/ccigpt_cifar10_s_rgb_ronly_v2.yaml \
        --checkpoint experiments/ccigpt_cifar10_s_rgb_ronly_v2/checkpoints/best.pth --swa

参考:
  [1] Shannon, "A Mathematical Theory of Communication," 1948.
      bits/dim = -log₂ p(x) = CE / ln(2)
  [2] Salimans et al., "PixelCNN++," ICLR 2017 — CIFAR-10: 2.92 bits/dim
  [3] Parmar et al., "Image Transformer," ICML 2018 — CIFAR-10: 2.90 bits/dim
  [4] Chen et al., "PixelSNAIL," ICML 2018 — CIFAR-10: 2.85 bits/dim
"""

import os
import sys
import json
import argparse
import yaml
import math
import torch
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
from contextlib import nullcontext

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from torch.amp import autocast
from torch.utils.data import DataLoader, Subset
from mdlic.eval_metrics import (
    ce_per_image as _ce_per_image_from_logits,
    per_image_bpd as _per_image_bpd,
    per_image_rate_components as _per_image_rate_components,
    tokenize_targets as _tokenize_targets,
)
from mdlic.codec.probability import (
    codec_aligned_score_metadata,
    diagnostic_score_metadata,
)
from mdlic.provenance import (
    dataset_record,
    git_metadata as _git_metadata,
    runtime_metadata as _runtime_metadata,
    sha256_file as _sha256_file,
)
from mdlic.rate import dual_stream_bpd, ideal_stream_bits, single_stream_bpd
from mdlic.traditional_codecs import codec_metadata, encode_rgb_array, get_codec_spec
from mdlic.utils import clean_state_dict
from scripts.train import _build_model_from_config, _build_ccigpt_from_config


# ── 多卡评测（DDP）──
# 评测是纯 forward（无梯度、无参数同步），不需要 DDP wrap model。做法：
#   1. torchrun 拉起 N 个进程，每进程 _init_distributed() 初始化 gloo 进程组 + 绑定一张卡
#   2. _shard_dataset() 用 stride 切片把 val 集分成 N 份不相交子集（并集 = 全集，无 padding）
#   3. 每 rank 独立 build+load ckpt 到自己的卡、只跑自己那份
#   4. evaluate_* 末尾 all-reduce(SUM) 加权累加量 → 全局 bpd_mean
# 精度说明：mean/std 都从逐图 bpd 的一阶、二阶矩聚合，batch size 与 DDP 分片
# 不改变统计定义；不同浮点求和顺序只可能造成末位舍入差异。--per_image_stats
# 进一步 gather 逐图记录，并可计算固定 seed 的 bootstrap CI。
# 单卡运行（不经 torchrun）时所有 helper 退化为 no-op，行为与改造前完全相同。

def _init_distributed():
    """从 torchrun 环境变量初始化分布式。返回 (rank, world_size, local_rank, is_dist)。

    未经 torchrun 启动（无 RANK/WORLD_SIZE 环境变量）时返回单卡占位，不初始化进程组。

    后端固定用 **gloo**（CPU/TCP socket）：评测 forward 是各 rank 独立的（无 GPU-GPU
    通信），只在末尾归约 6 个标量 + gather per-image 列表。AutoDL 等容器里 NCCL 的
    GPU P2P/SHM 常不通，会让哪怕 6-float 的 all_reduce 卡死超时（实测 ALLREDUCE
    NumelIn=6 hang 600s）；gloo 走 socket、在单机多卡容器里稳定，且标量通信无性能损失。
    模型仍各自跑在 cuda:local_rank 上（PG 后端只管集合通信、不决定张量所在设备）。
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        dist.init_process_group(backend="gloo")
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank, True
    return 0, 1, 0, False


def _is_dist():
    return dist.is_available() and dist.is_initialized()


def _shard_dataset(dataset):
    """分布式时按 stride 把 dataset 切成 world_size 份不相交子集，返回本 rank 那份。

    indices = range(rank, N, world_size)：各 rank 子集不相交、并集 = 全集、无重复 padding，
    因此 all-reduce(SUM) 后 n_total == 原始 N，bpd_mean 与单卡一致（到打印精度）。
    bpd_std 由逐图一阶/二阶矩聚合；不同分片只可能因浮点求和顺序产生末位差异。
    单卡时原样返回。
    """
    if not _is_dist():
        return dataset
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    indices = list(range(rank, len(dataset), world_size))
    return Subset(dataset, indices)


def _ordered_sample_ids(loader) -> list[int]:
    """Return dataset indices in the deterministic order consumed by loader."""
    dataset = loader.dataset
    if isinstance(dataset, Subset):
        return [int(i) for i in dataset.indices]
    return list(range(len(dataset)))


def _progress(loader, desc):
    """tqdm 进度条，只在 rank0（或单卡）显示；其余 rank 原样返回 loader 不打印。

    分布式时各 rank 跑等量 batch（stride 切分 + 同 batch_size），rank0 的进度条
    即代表整体进度。total 用本 rank 的 batch 数（len(loader)）。
    """
    if _is_dist() and dist.get_rank() != 0:
        return loader
    try:
        from tqdm import tqdm
    except ImportError:
        return loader
    return tqdm(loader, desc=desc, total=len(loader), dynamic_ncols=True)


def _dist_reduce_sum(values: dict) -> dict:
    """对一组标量做跨 rank all-reduce(SUM)。单卡时原样返回。

    用于把各 rank 的加权累加量（bpd_weighted_sum / n_total / ce_*_sum 等）汇总成全局量，
    再在 caller 里除以全局 n_total 得到与单卡一致的均值/方差。

    后端是 gloo，all_reduce 张量必须在 **CPU**（gloo 不走 GPU）；标量量小，CPU 归约无开销。
    """
    if not _is_dist():
        return values
    keys = list(values.keys())
    t = torch.tensor([values[k] for k in keys], dtype=torch.float64)  # CPU tensor for gloo
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return {k: t[i].item() for i, k in enumerate(keys)}


def _mean_and_sample_std(total: float, square_total: float, count: int):
    if count <= 0:
        raise ValueError("cannot summarize an empty evaluation set")
    mean = total / count
    if count == 1:
        return mean, 0.0
    centered_sum = max(square_total - count * mean * mean, 0.0)
    return mean, math.sqrt(centered_sum / (count - 1))


def _build_from_config(mcfg: dict, device):
    """根据 model.type 分发到 IGPT / CC-iGPT 构建函数。"""
    model_type = mcfg.get("type", "igpt")
    if model_type == "ccigpt":
        return _build_ccigpt_from_config(mcfg, device)
    return _build_model_from_config(mcfg, device)


# ── 学术 Baseline（直接引用论文数字）──
# 注意: PixelCNN++ / PixelSNAIL 使用 discretized logistic mixture likelihood，
# 本文使用 categorical CE（vocab=256）。两者都是 bits/dim，物理含义一致。
# Ref: [2][3][4]
ACADEMIC_BASELINES = {
    "cifar10": [
        ("PixelCNN++ [Salimans 2017]", 2.92),
        ("Image Transformer [Parmar 2018]", 2.90),
        ("PixelSNAIL [Chen 2018]", 2.85),
    ],
    "cifar100": [
        # CIFAR-100 上这些方法没有公开报告的 bits/dim
        # 仅展示本文结果
    ],
}


@torch.no_grad()
def evaluate_model(model, loader, device, amp_dtype=None, tta_hflip: bool = False,
                   collect_per_image: bool = False,
                   codec_numerics: bool = False):
    """
    评估模型在数据集上的 bits/dim (bpd)。

    返回:
      bpd_mean: float — 平均 bpd
      bpd_std:  float — 真正的 per-image bpd 标准差
      bpd_list: list[float] — 每个 batch 的 bpd（保留供 caller 自定义聚合）
      extras:   dict — 可选的额外字段（CC-iGPT 时含 ce_coarse / ce_fine / ctx_alpha；
                collect_per_image=True 时含 per_image_bpd）

    TTA (Test-Time Augmentation):
      tta_hflip=True 时对每张图同时跑 x 与 hflip(x) 两次 forward，取 bpd 均值。
      hflip(x) 的 -log p 仍是 H(X) 的合法上界（hflip 在 RGB-bit-exact 域是确定函数），
      平均能降低估计方差。Ref: Sparse Transformer (Child 2019, §4.2) 采用类似 ensemble.
    """
    if codec_numerics and amp_dtype is not None:
        raise ValueError("codec_numerics requires autocast to be disabled")
    model.eval()
    bpd_per_batch = []
    bpd_weighted_sum = 0.0
    bpd_sq_weighted_sum = 0.0
    per_image_bpd = []
    per_image_records = []
    sample_ids = _ordered_sample_ids(loader)
    sample_cursor = 0
    n_total = 0
    use_amp = amp_dtype is not None and device.type == 'cuda'

    # CC-iGPT 额外聚合 CE_c / CE_f / α。is_ccigpt 由 forward 输出 keys 推断，
    # 不依赖 isinstance（DDP wrap 后 model 是 DDP 而非 CCIGPT）。
    is_ccigpt = False
    ce_c_sum = ce_f_sum = alpha_sum = 0.0

    for batch in _progress(loader, "eval"):
        if isinstance(batch, (list, tuple)):
            x = batch[0]
        else:
            x = batch
        x = x.to(device)
        B = x.shape[0]

        with autocast(device_type="cuda", dtype=amp_dtype) if use_amp else nullcontext():
            out = model(x)
            if tta_hflip:
                out_flip = model(torch.flip(x, dims=[-1]))
            components = _per_image_rate_components(
                model, x, out, codec_numerics=codec_numerics,
            )
            bpd_img = components["bpd"]
            if tta_hflip:
                components_flip = _per_image_rate_components(
                    model, torch.flip(x, dims=[-1]), out_flip,
                    codec_numerics=codec_numerics,
                )
                bpd_img = (bpd_img + components_flip["bpd"]) * 0.5

        bpd_val = bpd_img.mean().item()
        bpd_per_batch.append(bpd_val)
        bpd_weighted_sum += bpd_img.double().sum().item()
        bpd_sq_weighted_sum += bpd_img.double().square().sum().item()
        n_total += B
        if collect_per_image:
            values = bpd_img.detach().cpu().tolist()
            ids = sample_ids[sample_cursor:sample_cursor + B]
            per_image_bpd.extend(values)
            per_image_records.extend(
                {"sample_id": int(sample_id), "ideal_model_bpd": float(value)}
                for sample_id, value in zip(ids, values)
            )
        sample_cursor += B

        if "ce_coarse" in components:
            is_ccigpt = True
            ce_c_img = components["ce_coarse"]
            ce_f_img = components["ce_fine"]
            if tta_hflip:
                ce_c_img = (ce_c_img + components_flip["ce_coarse"]) * 0.5
                ce_f_img = (ce_f_img + components_flip["ce_fine"]) * 0.5
            ce_c_sum += ce_c_img.double().sum().item()
            ce_f_sum += ce_f_img.double().sum().item()
            if "ctx_alpha" in out and out["ctx_alpha"] is not None:
                alpha_sum += out["ctx_alpha"].item() * B

    # 逐图一阶/二阶矩：batch size 和 DDP 分片只改变浮点求和顺序，不改变统计口径。
    agg = _dist_reduce_sum({
        "wsum": bpd_weighted_sum,
        "sqsum": bpd_sq_weighted_sum,
        "n": n_total,
        "ce_c": ce_c_sum,
        "ce_f": ce_f_sum,
        "alpha": alpha_sum,
    })
    n_global = agg["n"]
    bpd_mean, bpd_std = _mean_and_sample_std(
        agg["wsum"], agg["sqsum"], int(n_global),
    )

    extras = {
        "score_numerics": (
            codec_aligned_score_metadata()
            if codec_numerics
            else diagnostic_score_metadata(amp_dtype)
        ),
    }
    if is_ccigpt:
        extras["ce_coarse"] = agg["ce_c"] / n_global
        extras["ce_fine"] = agg["ce_f"] / n_global
        extras["ctx_alpha"] = agg["alpha"] / n_global
    if collect_per_image:
        # 分布式：跨 rank gather 各自子集的 per-image 列表拼成全集（顺序不保证，但
        # per-image 统计是顺序无关的 mean/std/CI，无影响）。单卡时直接用本地列表。
        if _is_dist():
            gathered = [None] * dist.get_world_size()
            dist.all_gather_object(gathered, per_image_records)
            per_image_records = [v for part in gathered for v in part]
            per_image_records.sort(key=lambda row: row["sample_id"])
            per_image_bpd = [row["ideal_model_bpd"] for row in per_image_records]
        extras["per_image_bpd"] = per_image_bpd
        extras["per_image_records"] = per_image_records
    return bpd_mean, bpd_std, bpd_per_batch, extras


def _ensemble_log_probs(log_probs_stack: list[torch.Tensor]) -> torch.Tensor:
    """Log-probs for p_ens(y|x) = mean_k p_k(y|x), computed stably."""
    if not log_probs_stack:
        raise ValueError("log_probs_stack must contain at least one tensor")
    stacked = torch.stack(log_probs_stack, dim=0)
    return torch.logsumexp(stacked, dim=0) - math.log(stacked.size(0))


def _summarize_per_image_bpd(values, bootstrap_samples: int = 1000,
                             seed: int = 0) -> dict:
    """Summarize per-image bpd and optionally estimate a bootstrap CI."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("per-image bpd 列表为空")

    ddof = 1 if arr.size > 1 else 0
    summary = {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=ddof)),
        "stderr": float(arr.std(ddof=ddof) / math.sqrt(arr.size)),
    }
    if arr.size > 1 and bootstrap_samples > 0:
        rng = np.random.default_rng(seed)
        means = np.empty(int(bootstrap_samples), dtype=np.float64)
        for i in range(int(bootstrap_samples)):
            idx = rng.integers(0, arr.size, size=arr.size)
            means[i] = arr[idx].mean()
        lo, hi = np.percentile(means, [2.5, 97.5])
        summary["ci95_bootstrap"] = [float(lo), float(hi)]
        summary["bootstrap_samples"] = int(bootstrap_samples)
    return summary


def _write_per_image_json(path: str, values, summary: dict, records=None) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "schema_version": 2,
        "summary": summary,
        "per_image_bpd": [round(float(v), 6) for v in values],
    }
    if records is not None:
        payload["records"] = records
    with open(path, "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[per_image_json] wrote {len(values)} rows → {path}")


def _write_result_manifest(
    path: str,
    *,
    args,
    config: dict,
    dataset_name: str,
    dataset_size: int,
    dataset_metadata: dict,
    models,
    checkpoint_paths,
    bpd_mean: float,
    bpd_std: float,
    summary: dict,
    extras: dict,
    device,
    traditional_codecs: dict | None = None,
) -> None:
    """Write a reproducible evaluation record for a single or ensemble run."""
    if dataset_metadata.get("name") != dataset_name:
        raise ValueError("dataset metadata name does not match the evaluated dataset")
    if dataset_metadata.get("size") != dataset_size:
        raise ValueError("dataset metadata size does not match the evaluated dataset")
    model_list = list(models)
    checkpoint_paths = list(checkpoint_paths)
    members = len(model_list)
    params_per_member = sum(p.numel() for p in model_list[0].parameters())
    protocol = "single_model_nll" if members == 1 else "ensemble_probability_mixture"
    if args.tta_hflip:
        protocol += "_with_hflip_diagnostic"

    config_abs = os.path.abspath(args.config)
    checkpoints = [
        {
            "path": os.path.abspath(checkpoint_path),
            "sha256": _sha256_file(checkpoint_path),
        }
        for checkpoint_path in checkpoint_paths
    ]
    metric_components = {
        key: extras[key]
        for key in ("ce_coarse", "ce_fine", "ctx_alpha")
        if key in extras
    }
    score_numerics = extras.get("score_numerics", {"name": "unspecified"})
    probability_numerics_aligned = (
        score_numerics.get("name")
        == codec_aligned_score_metadata()["name"]
    )
    codec_supported, codec_reason = _current_codec_compatibility(
        members=members,
        tta_hflip=args.tta_hflip,
        probability_numerics_aligned=probability_numerics_aligned,
    )
    payload = {
        "schema_version": 4,
        "protocol": protocol,
        "decodable_by_current_codec": codec_supported,
        "codec_compatibility": {
            "supported": codec_supported,
            "reason": codec_reason,
            "probability_numerics_aligned": probability_numerics_aligned,
            "actual_arithmetic_coding_run": bool(
                score_numerics.get("actual_arithmetic_coding", False)
            ),
        },
        "command": [sys.executable, *sys.argv],
        "git": _git_metadata(),
        "config": {
            "path": config_abs,
            "sha256": _sha256_file(config_abs),
            "model_type": config["model"].get("type", "igpt"),
        },
        "checkpoints": checkpoints,
        "model": {
            "parameters_per_member": params_per_member,
            "members": members,
            "stored_parameters": params_per_member * members,
            "forward_passes_per_image": members * (2 if args.tta_hflip else 1),
        },
        "dataset": dataset_metadata,
        "rate_accounting": {
            "metric": "ideal_model_bpd",
            "ce_predictions_per_stream": "num_tokens_minus_one",
            "first_token_prior": "uniform_256",
            "first_token_bits_per_stream": 8,
            "includes_container_header": False,
        },
        "score_numerics": score_numerics,
        "result": {
            "mean": bpd_mean,
            "std_per_image": bpd_std,
            "per_image_summary": summary,
            **metric_components,
        },
        "runtime": _runtime_metadata(device),
        "per_image": extras.get("per_image_records", []),
    }
    if traditional_codecs:
        payload["traditional_codecs"] = traditional_codecs

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}"
    with open(tmp_path, "w") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)
    print(f"[result_json] wrote manifest → {path}")


def _current_codec_compatibility(
    *,
    members: int,
    tta_hflip: bool,
    probability_numerics_aligned: bool = True,
) -> tuple[bool, str]:
    """Describe whether the current single-checkpoint codec can realize a score."""
    if members < 1:
        raise ValueError("members must be >= 1")
    if tta_hflip:
        return False, "hflip TTA has no corresponding current bitstream protocol"
    if members != 1:
        return False, "the current codec accepts exactly one checkpoint"
    if not probability_numerics_aligned:
        return False, "score does not use the codec's fp32-logits/fp64-softmax numerics"
    return True, "single checkpoint without TTA uses codec-aligned probability numerics"


@torch.no_grad()
def evaluate_ensemble(models, loader, device, amp_dtype=None,
                      tta_hflip: bool = False,
                      collect_per_image: bool = False):
    """Evaluate a per-token probability-mixture ensemble.

    Fine and coarse streams both use ``mean_k p_k(token | prefix)``.  This is
    a normalized autoregressive distribution when TTA is disabled.  Horizontal
    flip averaging remains a diagnostic score because the current codec does
    not define a corresponding decodable protocol.
    """
    for m in models:
        m.eval()

    bpd_per_batch = []
    bpd_weighted_sum = 0.0
    bpd_sq_weighted_sum = 0.0
    per_image_bpd = []
    per_image_records = []
    sample_ids = _ordered_sample_ids(loader)
    sample_cursor = 0
    n_total = 0
    use_amp = amp_dtype is not None and device.type == 'cuda'
    K = len(models)

    is_ccigpt = False
    ce_c_sum = ce_f_sum = alpha_sum = 0.0

    def _ensemble_forward(x_in):
        """Return per-image coarse/fine NLL under the probability mixture."""
        fine_log_probs = []
        coarse_log_probs = []
        alpha_local = []
        for m in models:
            amp_ctx = autocast(device_type="cuda", dtype=amp_dtype) if use_amp else nullcontext()
            with amp_ctx:
                out = m(x_in)
            assert out.get("logits") is not None, (
                "ensemble 评测要求 forward 返回 logits（softmax 路径）"
            )
            fine_log_probs.append(F.log_softmax(out["logits"].float(), dim=-1))
            if "ce_loss_coarse" in out and out["ce_loss_coarse"] is not None:
                assert out.get("logits_coarse") is not None, (
                    "CC-iGPT ensemble 需要 logits_coarse 才能对 coarse 概率做 mixture"
                )
                coarse_log_probs.append(
                    F.log_softmax(out["logits_coarse"].float(), dim=-1)
                )
                if "ctx_alpha" in out and out["ctx_alpha"] is not None:
                    alpha_local.append(out["ctx_alpha"].item())

        target_f = _tokenize_targets(x_in)
        log_probs_f = _ensemble_log_probs(fine_log_probs)
        ce_f_img = -log_probs_f.gather(
            -1, target_f.unsqueeze(-1)
        ).squeeze(-1).mean(dim=1)

        ce_c_img = None
        if coarse_log_probs:
            x_c = models[0]._coarse_input(x_in.clamp(0, 1).to(torch.float32))
            target_c = _tokenize_targets(x_c)
            log_probs_c = _ensemble_log_probs(coarse_log_probs)
            ce_c_img = -log_probs_c.gather(
                -1, target_c.unsqueeze(-1)
            ).squeeze(-1).mean(dim=1)
        return ce_c_img, ce_f_img, alpha_local

    for batch in _progress(loader, "ensemble"):
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x = x.to(device)
        B = x.size(0)

        ce_c_img, ce_f_img, alpha_x = _ensemble_forward(x)

        if tta_hflip:
            x_flip = torch.flip(x, dims=[-1])
            ce_c_flip, ce_f_flip, _ = _ensemble_forward(x_flip)
            ce_f_img = (ce_f_img + ce_f_flip) * 0.5
            if ce_c_img is not None:
                assert ce_c_flip is not None
                ce_c_img = (ce_c_img + ce_c_flip) * 0.5

        ce_fine_val = ce_f_img.mean().item()

        if ce_c_img is not None:
            is_ccigpt = True
            ce_coarse_val = ce_c_img.mean().item()
            N_c = models[0].coarse.seq_len
            N_f = models[0].fine.seq_len
            bpd_img = dual_stream_bpd(ce_c_img, N_c, ce_f_img, N_f)
            ce_c_sum += ce_c_img.double().sum().item()
            ce_f_sum += ce_f_img.double().sum().item()
            if alpha_x:
                alpha_sum += (sum(alpha_x) / len(alpha_x)) * B
        else:
            bpd_img = single_stream_bpd(ce_f_img, models[0].seq_len)

        bpd_val = bpd_img.mean().item()
        bpd_per_batch.append(bpd_val)
        bpd_weighted_sum += bpd_img.double().sum().item()
        bpd_sq_weighted_sum += bpd_img.double().square().sum().item()
        n_total += B
        if collect_per_image:
            values = bpd_img.detach().cpu().tolist()
            ids = sample_ids[sample_cursor:sample_cursor + B]
            per_image_bpd.extend(values)
            per_image_records.extend(
                {"sample_id": int(sample_id), "ideal_model_bpd": float(value)}
                for sample_id, value in zip(ids, values)
            )
        sample_cursor += B

    # 逐图一阶/二阶矩在 batch size 与 DDP 分片变化时保持同一统计口径。
    agg = _dist_reduce_sum({
        "wsum": bpd_weighted_sum,
        "sqsum": bpd_sq_weighted_sum,
        "n": n_total,
        "ce_c": ce_c_sum,
        "ce_f": ce_f_sum,
        "alpha": alpha_sum,
    })
    n_global = agg["n"]
    bpd_mean, bpd_std = _mean_and_sample_std(
        agg["wsum"], agg["sqsum"], int(n_global),
    )

    extras = {
        "K": K,
        "score_numerics": diagnostic_score_metadata(amp_dtype),
    }
    if is_ccigpt:
        extras["ce_coarse"] = agg["ce_c"] / n_global
        extras["ce_fine"] = agg["ce_f"] / n_global
        extras["ctx_alpha"] = agg["alpha"] / n_global
    if collect_per_image:
        if _is_dist():
            gathered = [None] * dist.get_world_size()
            dist.all_gather_object(gathered, per_image_records)
            per_image_records = [v for part in gathered for v in part]
            per_image_records.sort(key=lambda row: row["sample_id"])
            per_image_bpd = [row["ideal_model_bpd"] for row in per_image_records]
        extras["per_image_bpd"] = per_image_bpd
        extras["per_image_records"] = per_image_records
    return bpd_mean, bpd_std, bpd_per_batch, extras




def compute_traditional_bpd(dataset, method="png", *, return_details=False):
    """
    计算传统无损压缩方法在数据集上的 bits/dim (bpd)。

    将每张图片编码为内存中的 PNG/WebP 字节流，
    bpd = 压缩后字节 × 8 / 子像素总数 (= H × W × C)。

    PNG 显式使用 ``optimize=True``，WebP 显式使用 ``lossless=True``。
    Pillow 和底层 zlib/libwebp 版本会进入 details/manifest。若除以 H·W，
    得到的是真 bits-per-pixel（bpd × 3），不是本项目使用的 bits/dim。

    参数:
      dataset: torchvision dataset（返回 (tensor, label)）
      method: "png" 或 "webp"

    返回:
      bpd_mean: float
      bpd_std:  float
    """
    spec = get_codec_spec(method)
    if len(dataset) == 0:
        raise ValueError("cannot evaluate a traditional codec on an empty dataset")
    bpd_list = []
    compressed_bytes_total = 0
    for i in _progress(range(len(dataset)), f"traditional/{spec.method}"):
        img_tensor, _ = dataset[i]
        # tensor (C, H, W) [0,1] → PIL Image
        img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).round() \
            .clip(0, 255).astype(np.uint8)
        compressed_bytes = len(encode_rgb_array(img_np, spec.method))
        compressed_bytes_total += compressed_bytes

        # bpd = 压缩字节 × 8 / 总子像素数 (H·W·C)
        C, H, W = img_tensor.shape
        total_dims = H * W * C
        bpd = (compressed_bytes * 8) / total_dims
        bpd_list.append(bpd)

    values = np.asarray(bpd_list, dtype=np.float64)
    details = {
        **codec_metadata(spec.method),
        "dataset_size": len(dataset),
        "mean_bpd": float(values.mean()),
        "std_per_image": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "compressed_bytes_total": compressed_bytes_total,
        "denominator": "H*W*C per image",
    }
    if return_details:
        return details
    return details["mean_bpd"], details["std_per_image"]


def print_results_table(dataset_name, model_bpd, model_std,
                         traditional_results=None,
                         model_label="iGPT (Ours)"):
    """
    打印 Markdown 格式的结果表格。

    参数:
      dataset_name: str — "cifar10" / "cifar100"
      model_bpd: float — 本文模型 bits/dim
      model_std: float — bits/dim 标准差
      traditional_results: dict[str, (float, float)] — 传统方法 {name: (bpd, std)}
    """
    print(f"\n{'='*60}")
    print(f"  评测结果 — {dataset_name.upper()}")
    print(f"{'='*60}\n")

    print("| 方法 | bits/dim ↓ | 备注 |")
    print("|------|-----------|------|")

    # 传统方法
    if traditional_results:
        for name, (bpd, std) in traditional_results.items():
            print(f"| {name} | {bpd:.2f} ± {std:.2f} | 无损压缩 |")

    # 学术 baseline
    for name, bpd in ACADEMIC_BASELINES.get(dataset_name, []):
        print(f"| {name} | {bpd:.2f} | 论文报告值 |")

    # 本文
    print(f"| **{model_label}** | **{model_bpd:.4f} ± {model_std:.4f}** | **本文** |")
    print()


def _load_dataset(config):
    """加载测试数据集。
    """
    from torchvision import transforms
    from torchvision.datasets import CIFAR10, CIFAR100

    dataset_name = config["data"].get("dataset", "cifar100")

    if dataset_name == "imagenet64_npy":
        from mdlic.data.imagenet64_npy import ImageNet64Npy
        test_dataset = ImageNet64Npy(root=config["data"]["valid"], split="val")
        return test_dataset, dataset_name
    if dataset_name not in ("cifar10", "cifar100"):
        raise ValueError(
            f"未知 dataset: '{dataset_name}'，支持 cifar10/cifar100/imagenet64_npy"
        )
    transform = transforms.ToTensor()
    DatasetClass = CIFAR10 if dataset_name == "cifar10" else CIFAR100
    test_dataset = DatasetClass(root=config["data"]["valid"], train=False,
                                download=False, transform=transform)
    return test_dataset, dataset_name


def _evaluation_dataset_record(dataset, dataset_name: str, config: dict) -> dict:
    split = "test" if dataset_name in ("cifar10", "cifar100") else "val"
    if dataset_name == "imagenet64_npy":
        preprocessing = {
            "schema": "mdlic-rgb-preprocess-v1",
            "steps": [
                "read uint8 HWC sample from val.npy",
                "transpose HWC to CHW",
                "convert to float32 and divide by 255",
            ],
            "augmentation": None,
        }
    else:
        preprocessing = {
            "schema": "mdlic-rgb-preprocess-v1",
            "steps": [
                "torchvision.transforms.ToTensor: uint8 HWC to float32 CHW in [0,1]",
            ],
            "augmentation": None,
        }
    return dataset_record(
        dataset,
        name=dataset_name,
        split=split,
        configured_path=config["data"].get("valid"),
        preprocessing=preprocessing,
    )


def _load_checkpoint(model, ckpt_path, device):
    """加载 checkpoint（支持完整 ckpt 和纯 state_dict）。

    透明剥 `module.` / `_orig_mod.` 前缀并过滤遗留 persistent=False buffer
    (如 RoPE inv_freq)，兼容历史 DDP / torch.compile ckpt。
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(clean_state_dict(ckpt['model_state_dict']))
        epoch = ckpt.get('epoch', '?')
        return epoch
    else:
        model.load_state_dict(clean_state_dict(ckpt))
        return '?'


def _get_amp_dtype(config):
    """从 config 获取 AMP dtype，与 train.py 保持同口径。

    支持 fp16 / bf16 / null / "none" / "fp32" 五档；后三档表示走 fp32（amp_dtype=None）。
    口径漂移会让 evaluate 与训练时精度不一致，导致 bpd 数值不可比。
    """
    # 默认 fp16 与 train.py:594 同口径；若 yaml 省略 amp_dtype，evaluate 与训练
    # 走完全相同的精度路径，避免 logits 接近 logsumexp 边界时 bpd 数值不可比。
    amp_cfg = config["train"].get("amp_dtype", "fp16")
    if amp_cfg in (None, "none", "fp32"):
        return None, str(amp_cfg)
    if amp_cfg == "bf16":
        return torch.bfloat16, "bf16"
    if amp_cfg == "fp16":
        return torch.float16, "fp16"
    raise ValueError(
        f"未知 train.amp_dtype: '{amp_cfg}'，支持 fp16/bf16/null/none/fp32"
    )


def cmd_single(args, config, device):
    """单模型评测"""
    test_dataset, dataset_name = _load_dataset(config)
    shard = _shard_dataset(test_dataset)
    test_loader = DataLoader(shard, batch_size=args.batch_size,
                             shuffle=False, num_workers=2, pin_memory=True)
    print(f"Dataset: {dataset_name} test ({len(test_dataset)} images)")

    mcfg = config["model"]
    model_type = mcfg.get("type", "igpt")
    model = _build_from_config(mcfg, device)
    epoch = _load_checkpoint(model, args.checkpoint, device)
    print(f"Loaded checkpoint: epoch {epoch} (model_type={model_type})")

    # 正式单模型、无 TTA 评测与 arithmetic codec 共用 fp32 logits / fp64
    # softmax 口径。TTA 仍是沿用训练 AMP 的诊断模式。
    codec_numerics = not args.tta_hflip
    if codec_numerics:
        amp_dtype, amp_dtype_str = None, "fp32 logits + fp64 softmax (codec-aligned)"
    else:
        amp_dtype, amp_dtype_str = _get_amp_dtype(config)

    # 基本 bits/dim 评测
    print(f"\n评测中... (AMP: {amp_dtype_str}, TTA hflip: {args.tta_hflip})")
    collect_per_image = bool(
        args.per_image_stats or args.per_image_json or args.result_json
    )
    bpd_mean, bpd_std, _, extras = evaluate_model(
        model, test_loader, device,
        amp_dtype=amp_dtype,
        tta_hflip=args.tta_hflip,
        collect_per_image=collect_per_image,
        codec_numerics=codec_numerics,
    )
    print(f"{model_type.upper()} bits/dim: {bpd_mean:.4f} ± {bpd_std:.4f}")
    if collect_per_image:
        per_image = extras["per_image_bpd"]
        summary = _summarize_per_image_bpd(
            per_image,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.bootstrap_seed,
        )
        ci = summary.get("ci95_bootstrap")
        ci_str = f", 95% bootstrap CI [{ci[0]:.4f}, {ci[1]:.4f}]" if ci else ""
        print(
            "  per-image: "
            f"mean={summary['mean']:.4f}, std={summary['std']:.4f}, "
            f"stderr={summary['stderr']:.5f}{ci_str}"
        )
        if args.per_image_json and (not _is_dist() or dist.get_rank() == 0):
            # 只 rank0 落盘：DDP 下各 rank 经 all_gather_object 持有相同全集列表，
            # 若不 gate，N 个进程并发 open('w') 同一路径会交错/损坏 JSON。
            _write_per_image_json(
                args.per_image_json, per_image, summary,
                records=extras.get("per_image_records"),
            )
    if extras:
        # CC-iGPT 多输出 CE_c / CE_f / α，便于诊断 fine 弱 vs coarse overhead 过大
        if "ce_coarse" in extras:
            ce_c = extras["ce_coarse"]
            ce_f = extras["ce_fine"]
            N_c = model.coarse.seq_len
            N_f = model.fine.seq_len
            bpd_c_share = ideal_stream_bits(ce_c, N_c) / N_f
            bpd_f_share = ideal_stream_bits(ce_f, N_f) / N_f
            print(f"  CE_coarse = {ce_c:.4f}  → bpd_share = {bpd_c_share:.4f} ({100*bpd_c_share/bpd_mean:.1f}%)")
            print(f"  CE_fine   = {ce_f:.4f}  → bpd_share = {bpd_f_share:.4f} ({100*bpd_f_share/bpd_mean:.1f}%)")
            print(f"  ctx_alpha = {extras['ctx_alpha']:.4f}")

    # 传统方法（可选）
    traditional_results = None
    traditional_manifest = None
    is_primary_rank = not _is_dist() or dist.get_rank() == 0
    if args.traditional and is_primary_rank:
        print("\n计算传统方法 bits/dim...")
        traditional_results = {}
        traditional_manifest = {}

        png = compute_traditional_bpd(
            test_dataset, method="png", return_details=True,
        )
        traditional_results[png["display_name"]] = (
            png["mean_bpd"], png["std_per_image"],
        )
        traditional_manifest["png"] = png
        print(f"  PNG:  {png['mean_bpd']:.4f} ± {png['std_per_image']:.4f}")

        try:
            webp = compute_traditional_bpd(
                test_dataset, method="webp", return_details=True,
            )
            traditional_results[webp["display_name"]] = (
                webp["mean_bpd"], webp["std_per_image"],
            )
            traditional_manifest["webp"] = webp
            print(f"  WebP: {webp['mean_bpd']:.4f} ± {webp['std_per_image']:.4f}")
        except Exception as e:
            print(f"  WebP: 跳过 ({e})")

    if is_primary_rank:
        print_results_table(dataset_name, bpd_mean, bpd_std,
                            traditional_results,
                            model_label=f"{model_type.upper()} (Ours)")
    if args.result_json and is_primary_rank:
        _write_result_manifest(
            args.result_json,
            args=args,
            config=config,
            dataset_name=dataset_name,
            dataset_size=len(test_dataset),
            dataset_metadata=_evaluation_dataset_record(
                test_dataset, dataset_name, config,
            ),
            models=[model],
            checkpoint_paths=[args.checkpoint],
            bpd_mean=bpd_mean,
            bpd_std=bpd_std,
            summary=summary,
            extras=extras,
            device=device,
            traditional_codecs=traditional_manifest,
        )


def cmd_swa(args, config, device):
    """SWA vs best checkpoint 对比评测"""
    test_dataset, _ = _load_dataset(config)
    shard = _shard_dataset(test_dataset)
    test_loader = DataLoader(shard, batch_size=args.batch_size,
                             shuffle=False, num_workers=2, pin_memory=True)

    mcfg = config["model"]
    amp_dtype, _ = _get_amp_dtype(config)

    # best.pth
    ckpt_dir = os.path.dirname(args.checkpoint)
    best_path = args.checkpoint
    swa_path = os.path.join(ckpt_dir, "swa.pth")

    if not os.path.exists(swa_path):
        print(f"SWA checkpoint 不存在: {swa_path}")
        print("请确保训练时启用了 SWA (train.swa.enabled=true)")
        return

    # 评测 best
    model = _build_from_config(mcfg, device)
    _load_checkpoint(model, best_path, device)
    bpd_best, std_best, _, _ = evaluate_model(model, test_loader, device,
                                                amp_dtype=amp_dtype,
                                                tta_hflip=args.tta_hflip)
    print(f"best.pth  bits/dim: {bpd_best:.4f} ± {std_best:.4f}")

    # 评测 swa
    model = _build_from_config(mcfg, device)
    _load_checkpoint(model, swa_path, device)
    bpd_swa, std_swa, _, _ = evaluate_model(model, test_loader, device,
                                               amp_dtype=amp_dtype,
                                               tta_hflip=args.tta_hflip)
    print(f"swa.pth   bits/dim: {bpd_swa:.4f} ± {std_swa:.4f}")

    delta = bpd_swa - bpd_best
    print(f"\nΔ(bits/dim) (SWA - best): {delta:+.4f}")

    # 打印对比表格
    print(f"\n| Checkpoint | bits/dim ↓ | Δ |")
    print(f"|------------|-----------|---|")
    print(f"| best.pth | {bpd_best:.4f} ± {std_best:.4f} | — |")
    delta_str = f"{delta:+.4f}"
    note = "SWA 更优" if delta < 0 else "best 更优"
    print(f"| swa.pth  | {bpd_swa:.4f} ± {std_swa:.4f} | {delta_str} ({note}) |")
    print()


def cmd_ensemble(args, config, device):
    """多 ckpt probability-mixture ensemble 评测（best/swa/ema 等同源平滑组合）。"""
    test_dataset, dataset_name = _load_dataset(config)
    shard = _shard_dataset(test_dataset)
    test_loader = DataLoader(shard, batch_size=args.batch_size,
                             shuffle=False, num_workers=2, pin_memory=True)
    print(f"Dataset: {dataset_name} test ({len(test_dataset)} images)")

    ckpt_paths = [p.strip() for p in args.ensemble.split(',') if p.strip()]
    assert len(ckpt_paths) >= 2, (
        f"--ensemble 至少需要 2 个 ckpt，got {len(ckpt_paths)}"
    )

    mcfg = config["model"]
    model_type = mcfg.get("type", "igpt")
    amp_dtype, amp_dtype_str = _get_amp_dtype(config)

    print(f"Loading {len(ckpt_paths)} ckpts for ensemble...")
    models = []
    for path in ckpt_paths:
        m = _build_from_config(mcfg, device)
        epoch = _load_checkpoint(m, path, device)
        print(f"  {os.path.basename(path)} (epoch {epoch})")
        models.append(m)

    print(f"\n评测中... (AMP: {amp_dtype_str}, TTA hflip: {args.tta_hflip}, K={len(models)})")
    collect_per_image = bool(
        args.per_image_stats or args.per_image_json or args.result_json
    )
    bpd_mean, bpd_std, _, extras = evaluate_ensemble(
        models, test_loader, device,
        amp_dtype=amp_dtype, tta_hflip=args.tta_hflip,
        collect_per_image=collect_per_image,
    )
    print(f"{model_type.upper()} ensemble bits/dim: {bpd_mean:.4f} ± {bpd_std:.4f}")
    if args.tta_hflip:
        print("  注意：hflip 是诊断 NLL；当前 MDLC codec 未实现对应的可解码协议。")
    if collect_per_image:
        per_image = extras["per_image_bpd"]
        summary = _summarize_per_image_bpd(
            per_image,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.bootstrap_seed,
        )
        ci = summary.get("ci95_bootstrap")
        ci_str = f", 95% bootstrap CI [{ci[0]:.4f}, {ci[1]:.4f}]" if ci else ""
        print(
            "  per-image: "
            f"mean={summary['mean']:.4f}, std={summary['std']:.4f}, "
            f"stderr={summary['stderr']:.5f}{ci_str}"
        )
        if args.per_image_json and (not _is_dist() or dist.get_rank() == 0):
            _write_per_image_json(
                args.per_image_json, per_image, summary,
                records=extras.get("per_image_records"),
            )
    if "ce_coarse" in extras:
        ce_c, ce_f = extras["ce_coarse"], extras["ce_fine"]
        N_c = models[0].coarse.seq_len
        N_f = models[0].fine.seq_len
        bpd_c_share = ideal_stream_bits(ce_c, N_c) / N_f
        bpd_f_share = ideal_stream_bits(ce_f, N_f) / N_f
        print(f"  CE_coarse = {ce_c:.4f}  → bpd_share = {bpd_c_share:.4f} ({100*bpd_c_share/bpd_mean:.1f}%)")
        print(f"  CE_fine   = {ce_f:.4f}  → bpd_share = {bpd_f_share:.4f} ({100*bpd_f_share/bpd_mean:.1f}%)")
        print(f"  ctx_alpha = {extras['ctx_alpha']:.4f}  (K-档平均)")

    print_results_table(dataset_name, bpd_mean, bpd_std, None,
                        model_label=f"{model_type.upper()} ensemble (K={len(models)}, Ours)")
    if args.result_json and (not _is_dist() or dist.get_rank() == 0):
        _write_result_manifest(
            args.result_json,
            args=args,
            config=config,
            dataset_name=dataset_name,
            dataset_size=len(test_dataset),
            dataset_metadata=_evaluation_dataset_record(
                test_dataset, dataset_name, config,
            ),
            models=models,
            checkpoint_paths=ckpt_paths,
            bpd_mean=bpd_mean,
            bpd_std=bpd_std,
            summary=summary,
            extras=extras,
            device=device,
        )


def main():
    parser = argparse.ArgumentParser(
        description="iGPT 无损压缩评测",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 单模型主表（无 TTA）
  python scripts/evaluate.py --config configs/ccigpt_cifar10_s_rgb_ronly_v2.yaml \\
      --checkpoint experiments/ccigpt_cifar10_s_rgb_ronly_v2/checkpoints/best.pth \
      --per_image_stats

  # SWA 对比 (v2 配置启用了 SWA last 31 ckpts, start ep170)
  python scripts/evaluate.py --config configs/ccigpt_cifar10_s_rgb_ronly_v2.yaml \\
      --checkpoint experiments/ccigpt_cifar10_s_rgb_ronly_v2/checkpoints/best.pth --swa

  # 多 ckpt probability-mixture ensemble (best/swa/ema 三档同源平滑组合)
  python scripts/evaluate.py --config configs/ccigpt_cifar10_s_rgb_ronly_v2.yaml \\
      --ensemble experiments/ccigpt_cifar10_s_rgb_ronly_v2/checkpoints/best.pth,\\
experiments/ccigpt_cifar10_s_rgb_ronly_v2/checkpoints/swa.pth,\\
experiments/ccigpt_cifar10_s_rgb_ronly_v2/checkpoints/ema.pth \\
      --tta_hflip
        """
    )
    parser.add_argument('--config', type=str, required=True,
                        help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='模型 checkpoint 路径（best.pth）')
    parser.add_argument('--ensemble', type=str, default=None,
                        help='多 ckpt probability-mixture ensemble，逗号分隔 ckpt 路径列表 '
                             '（与 --checkpoint 互斥；至少 2 个 ckpt）')
    parser.add_argument('--traditional', action='store_true',
                        help='同时计算 PNG/WebP 传统方法 bits/dim')
    parser.add_argument('--swa', action='store_true',
                        help='同时评测 SWA checkpoint（swa.pth vs best.pth）')
    parser.add_argument('--tta_hflip', action='store_true',
                        help='诊断性 Test-Time Augmentation；当前 codec 未实现对应协议')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='评测 batch size')
    parser.add_argument('--per_image_stats', action='store_true',
                        help='额外打印 per-image bpd/std/stderr/CI')
    parser.add_argument('--per_image_json', type=str, default=None,
                        help='导出 per-image bpd JSON；会隐式启用 --per_image_stats。')
    parser.add_argument('--result_json', type=str, default=None,
                        help='导出可复现评测 manifest（隐式收集逐图结果）')
    parser.add_argument('--bootstrap_samples', type=int, default=1000,
                        help='per-image 均值 bootstrap CI 抽样次数；设 0 可关闭 CI。')
    parser.add_argument('--bootstrap_seed', type=int, default=0,
                        help='per-image bootstrap 随机种子。')
    args = parser.parse_args()

    if args.ensemble and args.checkpoint:
        parser.error("--ensemble 与 --checkpoint 互斥；ensemble 路径在 --ensemble 内逗号分隔")
    if args.ensemble and args.swa:
        parser.error("--ensemble 与 --swa 互斥；ensemble 已包含多档 ckpt 评测")
    if args.swa and (args.per_image_stats or args.per_image_json or args.result_json):
        parser.error("--swa 对比模式暂不导出逐图结果；请分别评测具体 checkpoint")

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # DDP：torchrun 拉起多进程时初始化 NCCL + 绑卡；单卡直接走 cuda:0 / cpu。
    rank, world_size, local_rank, is_dist = _init_distributed()
    if is_dist:
        device = torch.device(f'cuda:{local_rank}')
        # 非 rank0 静默：评测的 collective（all-reduce/all-gather）所有 rank 都参与，
        # 但只 rank0 打印，避免 N 份重复输出。
        if rank != 0:
            sys.stdout = open(os.devnull, 'w')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}" + (f"  [DDP rank {rank}/{world_size}]" if is_dist else ""))

    try:
        # 根据模式分发
        if args.ensemble:
            cmd_ensemble(args, config, device)
        elif args.swa and args.checkpoint:
            cmd_swa(args, config, device)
        elif args.checkpoint:
            cmd_single(args, config, device)
        else:
            parser.error("请指定 --checkpoint 或 --ensemble")
    finally:
        if is_dist:
            dist.barrier()
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
