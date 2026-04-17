#!/usr/bin/env python3
"""
Linear Probe 评估：加载预训练 iGPT checkpoint，提取各层特征，
训练线性分类器，报告每层分类准确率。

用法:
  python scripts/linear_probe.py \
    --config configs/igpt_cifar10_baseline.yaml \
    --checkpoint experiments/igpt_cifar10_baseline/checkpoints/best.pth \
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
import math
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

from src.mdlic.models.igpt import IGPT


# ──────────────────────────────────────────────────────────────
# 1. 工具函数
# ──────────────────────────────────────────────────────────────

def load_config(path):
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def build_model(mcfg, device):
    """从 config['model'] 构建 IGPT（复用 train.py 逻辑）"""
    return IGPT(
        image_size=mcfg["image_size"],
        in_channels=mcfg["in_channels"],
        vocab_size=mcfg["vocab_size"],
        d_model=mcfg["d_model"],
        N=mcfg["N"],
        h=mcfg["h"],
        d_ff=mcfg["d_ff"],
        dropout=mcfg["dropout"],
        use_ycbcr=mcfg.get("use_ycbcr", True),
        use_rope=mcfg.get("use_rope", True),
        use_post_norm=mcfg.get("use_post_norm", True),
        use_swiglu=mcfg.get("use_swiglu", True),
        use_qk_norm=mcfg.get("use_qk_norm", True),
        use_depth_scaled_init=mcfg.get("use_depth_scaled_init", True),
        use_zloss=mcfg.get("use_zloss", True),
        activation_checkpointing=False,  # 推理不需要
        use_subpixel_ar=mcfg.get("use_subpixel_ar", False),
    ).to(device)


def load_checkpoint(model, ckpt_path, device):
    """加载 checkpoint（支持完整 ckpt 和纯 state_dict）"""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
        epoch = ckpt.get('epoch', '?')
        print(f"[Linear Probe] 加载 checkpoint (epoch {epoch}): {ckpt_path}")
    else:
        model.load_state_dict(ckpt)
        print(f"[Linear Probe] 加载 state_dict: {ckpt_path}")


# ──────────────────────────────────────────────────────────────
# 2. 特征提取
# ──────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_features(model, dataloader, layer_indices, device, amp_dtype=None):
    """
    提取指定层的 hidden states，全局平均池化后返回。

    参数:
      model:         预训练 IGPT（已冻结）
      dataloader:    图像数据 DataLoader
      layer_indices: 要提取的层索引列表，如 [0, 3, 7, 11]
      device:        torch.device
      amp_dtype:     混合精度 dtype（None / torch.bfloat16 / torch.float16）

    返回:
      features: dict[int, Tensor]  — {layer_idx: (N, d_model) float32}
      labels:   Tensor (N,) long   — 分类标签
    """
    model.eval()
    max_layer = max(layer_indices)
    wanted = set(layer_indices)

    all_features = {idx: [] for idx in layer_indices}
    all_labels = []

    use_amp = amp_dtype is not None
    from contextlib import nullcontext
    from torch.amp import autocast

    for batch in dataloader:
        images, targets = batch
        images = images.to(device)

        with autocast("cuda", dtype=amp_dtype) if use_amp else nullcontext():
            # 只跑到 max_layer 且不经过 output head / loss，避免冗余计算
            layer_outs = model.encode(images, max_layer=max_layer)

        for idx in layer_indices:
            # GPTBlock 输出 (B, T, d_model) → 全局平均池化 → (B, d_model)
            all_features[idx].append(layer_outs[idx].float().mean(dim=1).cpu())
        all_labels.append(targets)

    features = {idx: torch.cat(all_features[idx], dim=0) for idx in layer_indices}
    labels = torch.cat(all_labels, dim=0)

    return features, labels


# ──────────────────────────────────────────────────────────────
# 3. 线性分类器训练
# ──────────────────────────────────────────────────────────────

def train_linear_probe(train_features, train_labels, test_features, test_labels,
                       d_model, num_classes, epochs=100, lr=0.1, batch_size=256,
                       device='cpu'):
    """
    训练线性分类器并返回最佳测试准确率。

    遵循 iGPT 原论文设置:
      - SGD, momentum=0.9, no weight decay
      - Cosine annealing LR schedule
      - 100 epochs

    参数:
      train_features: (N_train, d_model) float32
      train_labels:   (N_train,) long
      test_features:  (N_test, d_model) float32
      test_labels:    (N_test,) long

    返回:
      best_acc: float — 最佳测试准确率 (%)
    """
    # 特征标准化 (zero mean, unit variance)
    mean = train_features.mean(dim=0)
    std = train_features.std(dim=0).clamp(min=1e-6)
    train_features = (train_features - mean) / std
    test_features = (test_features - mean) / std

    train_dataset = TensorDataset(train_features, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, drop_last=False)

    classifier = nn.Linear(d_model, num_classes).to(device)
    nn.init.zeros_(classifier.bias)
    nn.init.normal_(classifier.weight, std=0.01)

    optimizer = torch.optim.SGD(classifier.parameters(), lr=lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    test_x = test_features.to(device)
    test_y = test_labels.to(device)

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

        # 测试准确率
        classifier.eval()
        with torch.no_grad():
            logits = classifier(test_x)
            preds = logits.argmax(dim=1)
            acc = (preds == test_y).float().mean().item() * 100

        if acc > best_acc:
            best_acc = acc

    return best_acc


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
    parser.add_argument("--batch_size", type=int, default=256,
                        help="特征提取和分类器训练 batch size (default: 256)")
    parser.add_argument("--export_csv", type=str, default=None,
                        help="导出结果到 CSV 文件")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config(args.config)
    mcfg = config["model"]

    # --- 构建并加载模型 ---
    model = build_model(mcfg, device)
    load_checkpoint(model, args.checkpoint, device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    N = mcfg["N"]
    d_model = mcfg["d_model"]

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
    dataset_name = config["data"].get("dataset", "cifar100")
    DatasetClass = CIFAR10 if dataset_name == "cifar10" else CIFAR100
    num_classes = 10 if dataset_name == "cifar10" else 100
    data_root = config["data"]["train"]

    transform = transforms.ToTensor()
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

    print("\n[1/3] 提取训练集特征 ...")
    train_features, train_labels = extract_features(
        model, train_loader, layer_indices, device, amp_dtype)
    print(f"      训练集: {train_labels.shape[0]} 样本, "
          f"每层特征 shape: ({train_labels.shape[0]}, {d_model})")

    print("[2/3] 提取测试集特征 ...")
    test_features, test_labels = extract_features(
        model, test_loader, layer_indices, device, amp_dtype)
    print(f"      测试集: {test_labels.shape[0]} 样本")

    # --- 训练线性分类器 ---
    print(f"[3/3] 训练线性分类器 (epochs={args.epochs}, lr={args.lr}) ...\n")

    results = []
    for idx in layer_indices:
        acc = train_linear_probe(
            train_features[idx], train_labels,
            test_features[idx], test_labels,
            d_model=d_model,
            num_classes=num_classes,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            device=device,
        )
        results.append((idx, acc))
        print(f"  Layer {idx:2d}: {acc:.2f}%")

    # --- 结果汇总 ---
    print(f"\n{'='*60}")
    print(f"{'Layer':>8s}  {'Accuracy':>10s}")
    print(f"{'-'*8:>8s}  {'-'*10:>10s}")
    for idx, acc in results:
        print(f"{'L'+str(idx):>8s}  {acc:>9.2f}%")

    best_layer, best_acc = max(results, key=lambda x: x[1])
    print(f"\n最佳: Layer {best_layer}, Accuracy {best_acc:.2f}%")
    print(f"{'='*60}")

    # --- 导出 CSV ---
    if args.export_csv:
        os.makedirs(os.path.dirname(args.export_csv) or ".", exist_ok=True)
        with open(args.export_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["layer", "accuracy"])
            for idx, acc in results:
                writer.writerow([idx, f"{acc:.2f}"])
        print(f"\n结果已导出: {args.export_csv}")


if __name__ == "__main__":
    main()
