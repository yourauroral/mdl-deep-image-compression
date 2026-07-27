#!/usr/bin/env python3
"""
Linear Probe 评估：加载预训练 iGPT checkpoint，提取各层特征，
训练线性分类器，报告每层分类准确率。

用法:
  # CC-iGPT v2 主表配置 (32 层 fine + α·coarse_ctx 注入)
  python scripts/linear_probe.py \
    --config configs/ccigpt_cifar10_s_rgb_ronly_v2.yaml \
    --checkpoint experiments/ccigpt_cifar10_s_rgb_ronly_v2/checkpoints/best.pth \
    --layers all --epochs 100 --lr 0.1 --batch_size 256

  # iGPT baseline (Phase A 单尺度对照)
  python scripts/linear_probe.py \
    --config configs/igpt_cifar10_s_rgb.yaml \
    --checkpoint experiments/igpt_cifar10_s_rgb/checkpoints/best.pth \
    --layers all --epochs 100 --lr 0.1 --batch_size 256

原理:
  iGPT (Chen et al., 2020) 发现自回归预训练的 Transformer 中间层
  能学到高质量图像表征。通过冻结预训练模型，在每层 hidden state
  上训练一个线性分类器 (linear probe)，可以衡量该层表征的判别能力。
  特征聚合方式: 对 token 序列做全局平均池化 → (B, d_model)。

Ref:
  [1] Chen et al., "Generative Pretraining from Pixels," ICML 2020
  [2] Alain & Bengio, "Understanding intermediate layers using linear
      classifier probes," ICLR 2017 Workshop
"""

import argparse
import csv
import json
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100

# 添加项目根目录到 sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from scripts.train import _build_model_from_config, _build_ccigpt_from_config
from mdlic.provenance import (
    canonical_sha256,
    dataset_record,
    file_record,
    git_metadata,
    runtime_metadata,
)
from mdlic.utils import clean_state_dict


# ──────────────────────────────────────────────────────────────
# 1. 工具函数
# ──────────────────────────────────────────────────────────────

def load_config(path):
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def build_model(mcfg, device):
    """构建 IGPT 或 CCIGPT。

    复用 train.py 的工厂函数，确保配置键路径与训练完全一致：
    CC-iGPT yaml 用平铺键 (d_model 描述 fine、coarse_d_model 描述 coarse)。
    """
    model_type = mcfg.get("type", "igpt")
    if model_type == "ccigpt":
        return _build_ccigpt_from_config(mcfg, device)
    return _build_model_from_config(mcfg, device)


def load_checkpoint(model, ckpt_path, device):
    """加载 checkpoint（支持完整 ckpt 和纯 state_dict）。

    透明剥 `module.` / `_orig_mod.` 前缀并过滤遗留 persistent=False buffer
    (如 RoPE inv_freq)，兼容历史 DDP / torch.compile ckpt。
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(clean_state_dict(ckpt['model_state_dict']))
        epoch = ckpt.get('epoch', '?')
        print(f"[Linear Probe] 加载 checkpoint (epoch {epoch}): {ckpt_path}")
    else:
        model.load_state_dict(clean_state_dict(ckpt))
        print(f"[Linear Probe] 加载 state_dict: {ckpt_path}")


def parse_seeds(value):
    """解析逗号分隔的随机种子，并保留输入顺序。"""
    seeds = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        seed = int(item)
        if seed < 0:
            raise ValueError("random seeds must be non-negative")
        if seed not in seeds:
            seeds.append(seed)
    if not seeds:
        raise ValueError("at least one random seed is required")
    return seeds


def _probe_provenance(
    *,
    args,
    device,
    dataset_name: str,
    data_root: str,
    train_dataset,
    test_dataset,
    preprocessing: dict,
    selection_train_idx: torch.Tensor,
    validation_idx: torch.Tensor,
) -> dict:
    """Build the immutable inputs needed to reproduce a probe result."""
    train_record = dataset_record(
        train_dataset,
        name=dataset_name,
        split="train",
        configured_path=data_root,
        preprocessing=preprocessing,
    )
    test_record = dataset_record(
        test_dataset,
        name=dataset_name,
        split="test",
        configured_path=data_root,
        preprocessing=preprocessing,
    )
    return {
        "artifacts": {
            "config": file_record(args.config),
            "checkpoint": file_record(args.checkpoint),
        },
        "git": git_metadata(),
        "runtime": runtime_metadata(device),
        "datasets": {
            "train": train_record,
            "test": test_record,
        },
        "selection_split": {
            "train_count": int(selection_train_idx.numel()),
            "validation_count": int(validation_idx.numel()),
            "train_indices_sha256": canonical_sha256(selection_train_idx.cpu().tolist()),
            "validation_indices_sha256": canonical_sha256(validation_idx.cpu().tolist()),
        },
    }


def stratified_train_val_split(labels, val_fraction=0.1, seed=0):
    """返回确定性的分层 train/validation 索引。

    样本数至少为 2 的类别会同时出现在两个子集中；只有一个样本的类别保留在
    train，避免为了验证而删除该类别唯一的训练样本。
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")

    labels = torch.as_tensor(labels, dtype=torch.long, device="cpu").reshape(-1)
    if labels.numel() < 2:
        raise ValueError("at least two samples are required for a train/validation split")

    generator = torch.Generator().manual_seed(seed)
    train_parts = []
    val_parts = []
    for class_id in torch.unique(labels, sorted=True):
        class_indices = torch.nonzero(labels == class_id, as_tuple=False).flatten()
        class_indices = class_indices[
            torch.randperm(class_indices.numel(), generator=generator)
        ]
        if class_indices.numel() < 2:
            train_parts.append(class_indices)
            continue

        val_count = int(round(class_indices.numel() * val_fraction))
        val_count = min(max(val_count, 1), class_indices.numel() - 1)
        val_parts.append(class_indices[:val_count])
        train_parts.append(class_indices[val_count:])

    if not val_parts:
        raise ValueError("validation split is empty; each class has fewer than two samples")

    train_indices = torch.cat(train_parts)
    val_indices = torch.cat(val_parts)
    train_indices = train_indices[
        torch.randperm(train_indices.numel(), generator=generator)
    ]
    val_indices = val_indices[
        torch.randperm(val_indices.numel(), generator=generator)
    ]
    return train_indices, val_indices


def summarize_accuracies(values):
    """汇总分类器随机种子波动；这些量是描述性统计，不是样本置信区间。"""
    scores = torch.as_tensor(values, dtype=torch.float64)
    if scores.numel() == 0:
        raise ValueError("cannot summarize an empty accuracy list")
    mean = scores.mean().item()
    std = scores.std(unbiased=True).item() if scores.numel() > 1 else 0.0
    return {
        "classifier_seed_mean": mean,
        "classifier_seed_std": std,
        "classifier_seed_min": scores.min().item(),
        "classifier_seed_max": scores.max().item(),
        "num_classifier_seeds": scores.numel(),
    }


def bootstrap_sample_accuracy(correctness_by_seed, bootstrap_samples=2000, seed=0):
    """对评估样本做 percentile bootstrap，估计固定分类器集合的准确率区间。

    输入形状为 (num_classifier_seeds, num_eval_samples)，元素表示逐样本是否分类
    正确。先对固定分类器 seeds 求逐样本平均，再重采样样本；因此该区间只反映评估
    样本变动，不包含 classifier/pretraining seed 的训练不确定性。
    """
    if bootstrap_samples < 0:
        raise ValueError("bootstrap_samples must be non-negative")
    if seed < 0:
        raise ValueError("bootstrap seed must be non-negative")

    correctness = torch.as_tensor(correctness_by_seed, dtype=torch.float64, device="cpu")
    if correctness.ndim == 1:
        correctness = correctness.unsqueeze(0)
    if correctness.ndim != 2 or correctness.shape[0] == 0 or correctness.shape[1] == 0:
        raise ValueError("correctness_by_seed must have shape (num_seeds, num_samples)")
    if not torch.isfinite(correctness).all() or ((correctness < 0) | (correctness > 1)).any():
        raise ValueError("correctness values must be finite and in [0, 1]")

    per_sample = correctness.mean(dim=0)
    point = per_sample.mean().item() * 100.0
    result = {
        "accuracy": point,
        "num_eval_samples": per_sample.numel(),
        "bootstrap_samples": int(bootstrap_samples),
        "bootstrap_seed": int(seed),
        "ci95": None,
        "estimand": "accuracy averaged over the fixed classifier seeds",
    }
    if bootstrap_samples == 0:
        return result

    generator = torch.Generator(device="cpu").manual_seed(seed)
    means = torch.empty(bootstrap_samples, dtype=torch.float64)
    # 控制临时索引矩阵在约 100 万元素以内，避免 10k test × 多次 bootstrap 峰值过高。
    chunk_size = max(1, min(bootstrap_samples, 1_000_000 // per_sample.numel()))
    for start in range(0, bootstrap_samples, chunk_size):
        stop = min(start + chunk_size, bootstrap_samples)
        indices = torch.randint(
            per_sample.numel(),
            (stop - start, per_sample.numel()),
            generator=generator,
        )
        means[start:stop] = per_sample[indices].mean(dim=1) * 100.0
    lo, hi = torch.quantile(means, torch.tensor([0.025, 0.975], dtype=torch.float64))
    result["ci95"] = [lo.item(), hi.item()]
    return result


def select_best_layer(validation_results):
    """仅依据验证集均值选层；并列时选择层号较小者。"""
    if not validation_results:
        raise ValueError("validation_results must not be empty")
    return max(
        validation_results,
        key=lambda item: (item["classifier_seed_mean"], -item["layer"]),
    )["layer"]


# ──────────────────────────────────────────────────────────────
# 2. 特征提取
# ──────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_features(model, dataloader, layer_indices, device, amp_dtype=None,
                     use_coarse_ctx: bool = True, desc: str = "extract"):
    """
    提取指定层的 hidden states，全局平均池化后返回。

    参数:
      model:         预训练 IGPT（已冻结）
      dataloader:    图像数据 DataLoader
      layer_indices: 要提取的层索引列表，如 [0, 3, 7, 11]
      device:        torch.device
      amp_dtype:     混合精度 dtype（None / torch.bfloat16 / torch.float16）
      use_coarse_ctx: 仅对 CC-iGPT 生效。True (默认) 测"条件后表征"，
                      False 测"裸 fine 表征"以做消融对比。

    返回:
      features: dict[int, Tensor]  — {layer_idx: (N, d_model) float32}
      labels:   Tensor (N,) long   — 分类标签
    """
    model.eval()
    max_layer = max(layer_indices)

    all_features = {idx: [] for idx in layer_indices}
    all_labels = []

    use_amp = amp_dtype is not None and device.type == 'cuda'
    from contextlib import nullcontext
    from torch.amp import autocast

    # CC-iGPT 的 encode 接受 use_coarse_ctx；普通 iGPT 没有这个参数。
    is_ccigpt = hasattr(model, "fine") and hasattr(model, "coarse")

    from tqdm import tqdm
    for batch in tqdm(dataloader, desc=desc, total=len(dataloader), dynamic_ncols=True):
        images, targets = batch
        images = images.to(device)

        with autocast("cuda", dtype=amp_dtype) if use_amp else nullcontext():
            if is_ccigpt:
                layer_outs = model.encode(images, max_layer=max_layer, pool=True,
                                          use_coarse_ctx=use_coarse_ctx)
            else:
                layer_outs = model.encode(images, max_layer=max_layer, pool=True)

        for idx in layer_indices:
            all_features[idx].append(layer_outs[idx])
        all_labels.append(targets)

    features = {idx: torch.cat(all_features[idx], dim=0) for idx in layer_indices}
    labels = torch.cat(all_labels, dim=0)

    return features, labels


# ──────────────────────────────────────────────────────────────
# 3. 线性分类器训练
# ──────────────────────────────────────────────────────────────

def train_linear_probe(train_features, train_labels, eval_features, eval_labels,
                       d_model, num_classes, epochs=100, lr=0.1, batch_size=256,
                       device='cpu', seed=0, return_correctness=False):
    """
    训练线性分类器并返回指定评估集上的最终准确率。

    遵循 iGPT 原论文设置:
      - SGD, momentum=0.9, no weight decay
      - Cosine annealing LR schedule
      - 100 epochs

    参数:
      train_features: (N_train, d_model) float32
      train_labels:   (N_train,) long
      eval_features:  (N_eval, d_model) float32
      eval_labels:    (N_eval,) long

    返回:
      final_acc: float — 训练结束时的评估准确率 (%)。return_correctness=True 时
      另返回逐评估样本的 bool correctness，用于样本 bootstrap。
    """
    # 特征标准化 (zero mean, unit variance)
    mean = train_features.mean(dim=0)
    std = train_features.std(dim=0).clamp(min=1e-6)
    train_features = (train_features - mean) / std
    eval_features = (eval_features - mean) / std

    train_dataset = TensorDataset(train_features, train_labels)
    loader_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, drop_last=False,
                              generator=loader_generator)

    device = torch.device(device)
    cuda_devices = ([device.index if device.index is not None else torch.cuda.current_device()]
                    if device.type == "cuda" else [])
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed)
        classifier = nn.Linear(d_model, num_classes).to(device)
        nn.init.zeros_(classifier.bias)
        nn.init.normal_(classifier.weight, std=0.01)

    optimizer = torch.optim.SGD(classifier.parameters(), lr=lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    eval_x = eval_features.to(device)
    eval_y = eval_labels.to(device)

    for epoch in range(epochs):
        classifier.train()
        for feat_batch, label_batch in train_loader:
            feat_batch = feat_batch.to(device)
            label_batch = label_batch.to(device)

            logits = classifier(feat_batch)
            loss = F.cross_entropy(logits, label_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()

    # 只在训练结束后评估，避免按 epoch 选择最大值造成评估集泄漏。
    classifier.eval()
    with torch.no_grad():
        logits = classifier(eval_x)
        preds = logits.argmax(dim=1)
        correctness = preds.eq(eval_y).cpu()
        final_acc = correctness.float().mean().item() * 100

    if return_correctness:
        return final_acc, correctness
    return final_acc


# ──────────────────────────────────────────────────────────────
# 4. 主函数
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Linear Probe: 评估 iGPT 各层表征的分类能力"
    )
    parser.add_argument("--config", type=str, required=True,
                        help="YAML 配置文件路径")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="预训练 iGPT checkpoint 路径")
    parser.add_argument("--layers", type=str, default="all",
                        help="提取哪些层的特征, 如 'all' 或 '0,3,7,11'")
    parser.add_argument("--epochs", type=int, default=100,
                        help="线性分类器训练轮数 (default: 100)")
    parser.add_argument("--lr", type=float, default=0.1,
                        help="SGD 学习率 (default: 0.1)")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="特征提取和分类器训练 batch size。默认 None=按模型 fine seq 自适应"
                             "（CIFAR seq=3072 → 256；IN64 seq=12288 → 64），避免长序列下 "
                             "fused projection 的 B·T·3d 偏移溢出 int32 → CUDA illegal memory access。"
                             "显式指定可覆盖。")
    parser.add_argument("--export_csv", type=str, default=None,
                        help="导出结果到 CSV 文件")
    parser.add_argument("--export_json", type=str, default=None,
                        help="导出带协议元数据和逐 seed 结果的 JSON 文件")
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4",
                        help="线性分类器随机种子，逗号分隔 (default: 0,1,2,3,4)")
    parser.add_argument("--val_fraction", type=float, default=0.1,
                        help="从训练集分层切出的选层验证集比例 (default: 0.1)")
    parser.add_argument("--split_seed", type=int, default=0,
                        help="分层 train/validation 切分随机种子 (default: 0)")
    parser.add_argument("--bootstrap_samples", type=int, default=2000,
                        help="最终 test 按样本 bootstrap 次数；0 表示关闭 (default: 2000)")
    parser.add_argument("--bootstrap_seed", type=int, default=0,
                        help="最终 test 样本 bootstrap 随机种子 (default: 0)")
    parser.add_argument("--no_coarse_ctx", action="store_true",
                        help="仅 CC-iGPT 生效：跳过 α·coarse_ctx 注入，"
                             "测裸 fine 表征（消融对照组）")
    parser.add_argument("--probe_dataset", type=str, default=None,
                        choices=["cifar10", "cifar100"],
                        help="覆盖探针数据集（用于跨数据集 transfer probe，如 IN64 预训权重"
                             "探在 CIFAR-10 上）。默认沿用 config['data']['dataset']")
    parser.add_argument("--probe_data_root", type=str, default=None,
                        help="探针数据集根目录（默认沿用 config['data']['train']；"
                             "IN64 config 的 data 路径不含 CIFAR 时需显式指定，如 datasets/）")
    args = parser.parse_args()

    try:
        seeds = parse_seeds(args.seeds)
    except ValueError as exc:
        parser.error(str(exc))
    if not 0.0 < args.val_fraction < 1.0:
        parser.error("--val_fraction must be between 0 and 1")
    if args.split_seed < 0:
        parser.error("--split_seed must be non-negative")
    if args.bootstrap_samples < 0:
        parser.error("--bootstrap_samples must be non-negative")
    if args.bootstrap_seed < 0:
        parser.error("--bootstrap_seed must be non-negative")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config(args.config)
    mcfg = config["model"]

    # --- 构建并加载模型 ---
    model = build_model(mcfg, device)
    load_checkpoint(model, args.checkpoint, device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    model_type = mcfg.get("type", "igpt")
    if model_type == "ccigpt":
        # CC-iGPT 的 probe 在 fine 子模型的各层 hidden 上做（encode 内部已注入 coarse_ctx）
        N = model.fine.N_layers
        d_model = model.fine.d_model
        fine_seq = model.fine.seq_len
    else:
        N = mcfg["N"]
        d_model = mcfg["d_model"]
        fine_seq = model.seq_len

    # --- batch_size 自适应 ---
    # 默认 None → 按 fine 序列长度反推一个 token 预算（256×3072=CIFAR 老默认），
    # 使 B·seq 维持常量。CIFAR seq=3072 → 256；IN64 seq=12288 → 64。
    # 动机：长序列 + 大 batch 下，fused QKV projection 的 flat 偏移 B·seq·(3·d_model)
    # 会溢出 int32（IN64 batch256：256·12287·1344 ≈ 4.2e9 > 2.1e9）→ CUDA illegal
    # memory access（非 OOM）。token 预算法把 IN64 自动压到 batch 64（实测可跑）。
    if args.batch_size is None:
        TOKEN_BUDGET = 256 * 3072
        args.batch_size = max(8, min(256, TOKEN_BUDGET // fine_seq))
        print(f"[probe] batch_size 自适应 = {args.batch_size}（fine seq={fine_seq}，"
              f"token 预算 {TOKEN_BUDGET}）")

    # --- 解析层索引 ---
    if args.layers == "all":
        layer_indices = list(range(N))
    else:
        layer_indices = [int(x) for x in args.layers.split(",")]
        for idx in layer_indices:
            if idx < 0 or idx >= N:
                print(f"[Error] 层索引 {idx} 超出范围 [0, {N-1}]")
                sys.exit(1)

    # --- 加载数据集 ---
    # 支持跨数据集 transfer probe：--probe_dataset 覆盖预训练 config 里的数据集，
    # 让 IN64 预训练权重也能探在 CIFAR-10 上。注意：IN64 模型 image_size=64，CIFAR
    # 图会被 resize 32→64（见下方 model_img_size 分支），与 32-native CIFAR 模型的
    # 66.93/79.33 不是同协议数字（输入分辨率不同），只能作 transfer 趋势看，勿直接横比。
    dataset_name = args.probe_dataset or config["data"].get("dataset", "cifar100")
    if dataset_name not in ("cifar10", "cifar100"):
        print(f"[Error] linear probe 仅支持 cifar10/cifar100 探针集，收到 '{dataset_name}'。"
              f"\n  IN64 1000-way probe 需 label-保留 prepare（见 future.md §6.1）。")
        sys.exit(1)
    DatasetClass = CIFAR10 if dataset_name == "cifar10" else CIFAR100
    num_classes = 10 if dataset_name == "cifar10" else 100
    data_root = args.probe_data_root or config["data"]["train"]

    # 模型 tokenize / position buffer 锁定在 model.image_size（IN64=64, CIFAR 配置=32）。
    # 探针图必须 resize 到该分辨率，否则 _embed_inputs 的定长断言命中。
    model_img_size = model.image_size           # IGPT / CCIGPT 都暴露 image_size
    tf_list = []
    transform_steps = []
    if model_img_size != 32:
        # CIFAR 原生 32×32；模型若期望别的分辨率（如 IN64 的 64）则双线性放缩
        tf_list.append(transforms.Resize(model_img_size,
                                         interpolation=transforms.InterpolationMode.BILINEAR))
        transform_steps.append({
            "op": "Resize",
            "size": [model_img_size, model_img_size],
            "interpolation": "bilinear",
            "antialias": True,
        })
        print(f"[probe] resize CIFAR 32→{model_img_size}（匹配预训练 image_size）")
    tf_list.append(transforms.ToTensor())
    transform_steps.append({
        "op": "ToTensor",
        "conversion": "uint8 HWC to float32 CHW in [0,1]",
    })
    transform = transforms.Compose(tf_list)
    preprocessing = {
        "schema": "mdlic-linear-probe-input-v1",
        "steps": transform_steps,
        "augmentation": None,
    }

    train_dataset = DatasetClass(root=data_root, train=True,
                                 download=True, transform=transform)
    test_dataset = DatasetClass(root=data_root, train=False,
                                download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=4, pin_memory=True)

    # --- AMP dtype ---
    amp_str = config.get("train", {}).get("amp_dtype", None)
    amp_dtype = None
    if amp_str == "bf16":
        amp_dtype = torch.bfloat16
    elif amp_str == "fp16":
        amp_dtype = torch.float16

    # --- 特征提取 ---
    print(f"\n{'='*60}")
    print(f"Linear Probe — {dataset_name.upper()}, {num_classes} classes")
    print(f"Model: d_model={d_model}, N={N}, layers={layer_indices}")
    print(f"{'='*60}")

    use_coarse_ctx = not args.no_coarse_ctx
    if model_type == "ccigpt":
        ctx_label = "with α·coarse_ctx" if use_coarse_ctx else "WITHOUT coarse_ctx (ablation)"
        print(f"CC-iGPT probe mode: {ctx_label}")

    print("\n[1/4] 提取训练集特征 ...")
    train_features, train_labels = extract_features(
        model, train_loader, layer_indices, device, amp_dtype,
        use_coarse_ctx=use_coarse_ctx, desc="train feats")
    print(f"      训练集: {train_labels.shape[0]} 样本, "
          f"每层特征 shape: ({train_labels.shape[0]}, {d_model})")

    selection_train_idx, val_idx = stratified_train_val_split(
        train_labels, val_fraction=args.val_fraction, seed=args.split_seed
    )
    print(f"      选层切分: train={selection_train_idx.numel()}, "
          f"validation={val_idx.numel()}, split_seed={args.split_seed}")

    # --- 只在 validation 上训练并选择层 ---
    print(f"[2/4] 在 validation 上比较各层 "
          f"(epochs={args.epochs}, lr={args.lr}, seeds={seeds}) ...\n")

    validation_results = []
    for idx in layer_indices:
        seed_accuracies = []
        for seed in seeds:
            accuracy = train_linear_probe(
                train_features[idx][selection_train_idx],
                train_labels[selection_train_idx],
                train_features[idx][val_idx],
                train_labels[val_idx],
                d_model=d_model,
                num_classes=num_classes,
                epochs=args.epochs,
                lr=args.lr,
                batch_size=args.batch_size,
                device=device,
                seed=seed,
            )
            seed_accuracies.append(accuracy)
        summary = summarize_accuracies(seed_accuracies)
        result = {
            "layer": idx,
            "classifier_seed_accuracies": dict(zip(seeds, seed_accuracies)),
            **summary,
        }
        validation_results.append(result)
        print(f"  Layer {idx:2d}: classifier-seed mean "
              f"{summary['classifier_seed_mean']:.2f}%, "
              f"std {summary['classifier_seed_std']:.2f}%")

    best_layer = select_best_layer(validation_results)
    best_validation = next(item for item in validation_results
                           if item["layer"] == best_layer)

    # 测试特征只提取已固定的层，杜绝利用 test 曲线重新选层。
    print(f"\n[3/4] validation 选定 Layer {best_layer}; 现在提取该层测试特征 ...")
    test_features, test_labels = extract_features(
        model, test_loader, [best_layer], device, amp_dtype,
        use_coarse_ctx=use_coarse_ctx, desc="selected test feats")
    print(f"      测试集: {test_labels.shape[0]} 样本")

    print("[4/4] 在完整训练集重训已选层并做最终测试 ...")
    test_seed_accuracies = []
    test_correctness = []
    for seed in seeds:
        accuracy, correctness = train_linear_probe(
            train_features[best_layer], train_labels,
            test_features[best_layer], test_labels,
            d_model=d_model,
            num_classes=num_classes,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            device=device,
            seed=seed,
            return_correctness=True,
        )
        test_seed_accuracies.append(accuracy)
        test_correctness.append(correctness)
        print(f"  seed {seed}: {accuracy:.2f}%")
    test_summary = summarize_accuracies(test_seed_accuracies)
    test_bootstrap = bootstrap_sample_accuracy(
        torch.stack(test_correctness),
        bootstrap_samples=args.bootstrap_samples,
        seed=args.bootstrap_seed,
    )

    # --- 结果汇总 ---
    print(f"\n{'='*60}")
    print(f"{'Layer':>8s}  {'Seed mean':>10s}  {'Seed std':>9s}")
    print(f"{'-'*8:>8s}  {'-'*10:>10s}  {'-'*9:>9s}")
    for result in validation_results:
        print(f"{'L'+str(result['layer']):>8s}  "
              f"{result['classifier_seed_mean']:>9.2f}%  "
              f"{result['classifier_seed_std']:>8.2f}%")

    print(f"\n按 validation 选择: Layer {best_layer}, "
          f"classifier-seed mean {best_validation['classifier_seed_mean']:.2f}%, "
          f"std {best_validation['classifier_seed_std']:.2f}%")
    print(f"最终 test（仅 Layer {best_layer}）: classifier-seed mean "
          f"{test_summary['classifier_seed_mean']:.2f}%, "
          f"std {test_summary['classifier_seed_std']:.2f}%")
    if test_bootstrap["ci95"] is not None:
        lo, hi = test_bootstrap["ci95"]
        print(f"最终 test 样本 bootstrap 95% CI: [{lo:.2f}%, {hi:.2f}%] "
              f"({args.bootstrap_samples} resamples, seed={args.bootstrap_seed})")
    print(f"{'='*60}")

    ctx_mode = (("with" if use_coarse_ctx else "without (ablation)")
                if model_type == "ccigpt" else "n/a")
    provenance = _probe_provenance(
        args=args,
        device=device,
        dataset_name=dataset_name,
        data_root=data_root,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        preprocessing=preprocessing,
        selection_train_idx=selection_train_idx,
        validation_idx=val_idx,
    )
    export_data = {
        "protocol": {
            "schema": "mdlic-linear-probe-v3",
            "selection_split": "stratified validation subset of training data",
            "test_policy": "selected layer only; retrained on full training set",
            "val_fraction": args.val_fraction,
            "split_seed": args.split_seed,
            "classifier_seeds": seeds,
            "classifier_seed_variation": (
                "sample standard deviation across linear-head seeds; descriptive, not a CI"
            ),
            "sample_bootstrap_ci": (
                "95% percentile bootstrap over final-test samples after averaging "
                "per-sample correctness across the fixed classifier seeds"
            ),
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_seed": args.bootstrap_seed,
        },
        "run": {
            "config": args.config,
            "checkpoint": args.checkpoint,
            "model_type": model_type,
            "probe_dataset": dataset_name,
            "num_classes": num_classes,
            "N_layers": N,
            "d_model": d_model,
            "model_image_size": model_img_size,
            "coarse_ctx": ctx_mode,
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
        },
        **provenance,
        "validation_curve": validation_results,
        "selected_layer": best_layer,
        "final_test": {
            "layer": best_layer,
            "classifier_seed_accuracies": dict(zip(seeds, test_seed_accuracies)),
            **test_summary,
            "sample_bootstrap": test_bootstrap,
        },
    }

    # --- 导出 CSV ---
    if args.export_csv:
        os.makedirs(os.path.dirname(args.export_csv) or ".", exist_ok=True)
        bootstrap_ci = test_bootstrap["ci95"] or ["", ""]
        meta = [
            ("schema", "mdlic-linear-probe-v3"),
            ("config", args.config),
            ("config_sha256", provenance["artifacts"]["config"]["sha256"]),
            ("checkpoint", args.checkpoint),
            ("checkpoint_sha256", provenance["artifacts"]["checkpoint"]["sha256"]),
            ("git_commit", provenance["git"]["commit"]),
            ("git_dirty", provenance["git"]["dirty"]),
            ("model_type", model_type),
            ("probe_dataset", dataset_name),
            ("train_dataset_fingerprint",
             provenance["datasets"]["train"]["fingerprint_sha256"]),
            ("test_dataset_fingerprint",
             provenance["datasets"]["test"]["fingerprint_sha256"]),
            ("preprocessing_json",
             json.dumps(preprocessing, sort_keys=True, separators=(",", ":"))),
            ("num_classes", num_classes),
            ("N_layers", N),
            ("d_model", d_model),
            ("model_image_size", model_img_size),
            ("coarse_ctx", ctx_mode),
            ("epochs", args.epochs),
            ("lr", args.lr),
            ("batch_size", args.batch_size),
            ("selection_split", "stratified validation subset of training data"),
            ("val_fraction", args.val_fraction),
            ("split_seed", args.split_seed),
            ("selection_train_indices_sha256",
             provenance["selection_split"]["train_indices_sha256"]),
            ("selection_validation_indices_sha256",
             provenance["selection_split"]["validation_indices_sha256"]),
            ("classifier_seeds", args.seeds),
            ("selected_layer", best_layer),
            ("classifier_seed_variation", "descriptive sample std; not a CI"),
            ("final_test_classifier_seed_mean",
             f"{test_summary['classifier_seed_mean']:.4f}"),
            ("final_test_classifier_seed_std",
             f"{test_summary['classifier_seed_std']:.4f}"),
            ("sample_bootstrap_estimand", test_bootstrap["estimand"]),
            ("sample_bootstrap_samples", args.bootstrap_samples),
            ("sample_bootstrap_seed", args.bootstrap_seed),
            ("final_test_sample_bootstrap_ci95_low", bootstrap_ci[0]),
            ("final_test_sample_bootstrap_ci95_high", bootstrap_ci[1]),
        ]
        with open(args.export_csv, "w", newline="") as f:
            for k, v in meta:
                f.write(f"# {k},{v}\n")
            writer = csv.writer(f)
            writer.writerow([
                "stage", "layer", "classifier_seed", "accuracy",
                "classifier_seed_mean", "classifier_seed_std",
                "classifier_seed_min", "classifier_seed_max",
                "sample_bootstrap_ci95_low", "sample_bootstrap_ci95_high",
            ])
            for result in validation_results:
                for seed, accuracy in result["classifier_seed_accuracies"].items():
                    writer.writerow(["validation", result["layer"], seed,
                                     f"{accuracy:.4f}", "", "", "", "", "", ""])
                writer.writerow(["validation_summary", result["layer"], "", "",
                                 f"{result['classifier_seed_mean']:.4f}",
                                 f"{result['classifier_seed_std']:.4f}",
                                 f"{result['classifier_seed_min']:.4f}",
                                 f"{result['classifier_seed_max']:.4f}", "", ""])
            for seed, accuracy in export_data["final_test"][
                    "classifier_seed_accuracies"].items():
                writer.writerow(["final_test", best_layer, seed,
                                 f"{accuracy:.4f}", "", "", "", "", "", ""])
            writer.writerow(["final_test_summary", best_layer, "", "",
                             f"{test_summary['classifier_seed_mean']:.4f}",
                             f"{test_summary['classifier_seed_std']:.4f}",
                             f"{test_summary['classifier_seed_min']:.4f}",
                             f"{test_summary['classifier_seed_max']:.4f}",
                             bootstrap_ci[0], bootstrap_ci[1]])
        print(f"\n结果已导出: {args.export_csv}")
        acc_arr = ", ".join(
            f"{result['classifier_seed_mean']:.2f}" for result in validation_results
        )
        print(f"[backfill] layers       = {layer_indices}")
        print(f"[backfill] validation_classifier_seed_mean = [{acc_arr}]")
        print(f"[backfill] selected_layer = {best_layer}")
        print(f"[backfill] final_test_classifier_seed_mean = "
              f"{test_summary['classifier_seed_mean']:.2f}")
        print(f"[backfill] final_test_sample_bootstrap_ci95 = {test_bootstrap['ci95']}")

    if args.export_json:
        os.makedirs(os.path.dirname(args.export_json) or ".", exist_ok=True)
        with open(args.export_json, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print(f"JSON 结果已导出: {args.export_json}")


if __name__ == "__main__":
    main()
