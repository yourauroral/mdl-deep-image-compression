#!/usr/bin/env python3
"""
Training script for iGPT autoregressive image compression.

Usage:
    python scripts/train.py --config configs/igpt_cifar100_baseline.yaml
    python scripts/train.py --config configs/igpt_cifar100_baseline.yaml --resume experiments/.../epoch_10.pth
    torchrun --nproc_per_node=4 scripts/train.py --config configs/igpt_cifar100_baseline.yaml
"""

import os
import sys
import csv
import argparse
import yaml
import math
import torch
import torch.optim as optim
from contextlib import nullcontext
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
# torch.cuda.amp.autocast 和 GradScaler 在 PyTorch 2.4+ 已废弃，
# 迁移至 torch.amp 统一接口。
# Ref: PyTorch 2.4 Release Notes — "torch.cuda.amp.autocast is deprecated"
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from torchvision import transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.mdlic.models.igpt import IGPT
from src.mdlic.models.layers import get_fused_kernel_status
from src.mdlic.utils import seed_everything, compute_bpp


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

    # ── train 字段 ──
    tcfg = config['train']
    assert tcfg.get('batch_size', 0) > 0, f"train.batch_size 必须 > 0"
    assert float(tcfg.get('lr', 0)) > 0, f"train.lr 必须 > 0"
    assert tcfg.get('epochs', 0) > 0, f"train.epochs 必须 > 0"
    assert tcfg.get('clip_max_norm', 0) > 0, f"train.clip_max_norm 必须 > 0"
    assert tcfg.get('grad_accum_steps', 1) >= 1, f"train.grad_accum_steps 必须 >= 1"

    amp_dtype = tcfg.get('amp_dtype', 'fp16')
    assert amp_dtype in ('fp16', 'bf16'), f"train.amp_dtype 必须是 'fp16' 或 'bf16'，got '{amp_dtype}'"

    lr_schedule = tcfg.get('lr_schedule', 'cosine')
    assert lr_schedule in ('cosine', 'wsd', 'multistep'), (
        f"train.lr_schedule 必须是 cosine/wsd/multistep，got '{lr_schedule}'"
    )

    # z_loss_weight 非负
    z_w = float(tcfg.get('z_loss_weight', 1e-4))
    assert z_w >= 0, f"train.z_loss_weight 必须 >= 0，got {z_w}"

    # loss_type 校验
    loss_type = mcfg.get('loss_type', 'ce')
    assert loss_type in ('ce', 'dmol'), (
        f"model.loss_type 必须是 'ce' 或 'dmol'，got '{loss_type}'"
    )
    if loss_type == 'dmol':
        K = int(mcfg.get('num_mixtures', 10))
        assert K > 0, f"model.num_mixtures 必须 > 0，got {K}"
        # Gaussian label smoothing 与 DMOL 不兼容（DMOL 自身已建模序数结构）
        sigma = float(tcfg.get('label_smoothing_sigma', 0.0))
        assert sigma == 0.0, (
            f"DMOL 与 label_smoothing_sigma 不兼容，got sigma={sigma}"
        )
        # Logit soft-capping 仅适用于 CE logits，DMOL 参数语义不同
        cap = float(mcfg.get('logit_soft_cap', 0.0))
        assert cap == 0.0, (
            f"DMOL 不使用 logit_soft_cap（参数语义不同），got cap={cap}"
        )


def _build_model_from_config(mcfg: dict, device) -> IGPT:
    """从 config['model'] 构建 IGPT 模型（统一 train.py 和 dryrun_forward.py）。"""
    return IGPT(
        image_size=mcfg["image_size"],
        in_channels=mcfg["in_channels"],
        vocab_size=mcfg["vocab_size"],
        d_model=mcfg["d_model"],
        N=mcfg["N"],
        h=mcfg["h"],
        d_ff=mcfg["d_ff"],
        dropout=mcfg["dropout"],
        use_mtp=mcfg.get("use_mtp", False),
        use_ycbcr=mcfg.get("use_ycbcr", True),
        use_rope=mcfg.get("use_rope", True),
        use_post_norm=mcfg.get("use_post_norm", True),
        use_swiglu=mcfg.get("use_swiglu", True),
        use_qk_norm=mcfg.get("use_qk_norm", True),
        use_depth_scaled_init=mcfg.get("use_depth_scaled_init", True),
        use_zloss=mcfg.get("use_zloss", True),
        activation_checkpointing=mcfg.get("activation_checkpointing", False),
        logit_soft_cap=float(mcfg.get("logit_soft_cap", 0.0)),
        use_subpixel_ar=mcfg.get("use_subpixel_ar", False),
        sliding_window_size=int(mcfg.get("sliding_window_size", -1)),
        full_attn_every_n=int(mcfg.get("full_attn_every_n", 0)),
        loss_type=mcfg.get("loss_type", "ce"),
        num_mixtures=int(mcfg.get("num_mixtures", 10)),
    ).to(device)


def _get_param_groups(model, weight_decay=0.1, mup_enabled=False,
                      mup_base_width=64, d_model=128, base_lr=1e-4):
    """
    构建 AdamW 参数组：embedding/norm/bias 不做 weight decay。
    muP 启用时，hidden 层 LR 按 base_width/d_model 缩放。
    Ref: Yang et al., arXiv:2203.03466, Table 8.
    """
    no_decay = set()
    mup_hidden = set()
    for name, _ in model.named_parameters():
        if any(nd in name for nd in ['token_embed', 'pos_embed', 'norm', 'bias']):
            no_decay.add(name)
        if mup_enabled and not name.endswith('head.weight') \
           and 'token_embed' not in name and 'pos_embed' not in name:
            mup_hidden.add(name)

    if mup_enabled:
        mup_lr_scale = mup_base_width / d_model
        groups = [
            {"params": [p for n, p in model.named_parameters()
                        if n not in no_decay and n in mup_hidden],
             "weight_decay": weight_decay, "lr": base_lr * mup_lr_scale},
            {"params": [p for n, p in model.named_parameters()
                        if n in no_decay and n in mup_hidden],
             "weight_decay": 0.0, "lr": base_lr * mup_lr_scale},
            {"params": [p for n, p in model.named_parameters()
                        if n not in no_decay and n not in mup_hidden],
             "weight_decay": weight_decay, "lr": base_lr},
            {"params": [p for n, p in model.named_parameters()
                        if n in no_decay and n not in mup_hidden],
             "weight_decay": 0.0, "lr": base_lr},
        ]
        return [g for g in groups if g["params"]]
    else:
        return [
            {"params": [p for n, p in model.named_parameters() if n not in no_decay], "weight_decay": weight_decay},
            {"params": [p for n, p in model.named_parameters() if n in no_decay],     "weight_decay": 0.0},
        ]


# ==================== Training ====================
def train_one_epoch(model, loader, optimizer, scaler, device,
                    epoch, log_freq, writer, clip_max_norm,
                    amp_dtype=torch.float16, grad_accum_steps=1,
                    z_loss_weight=1e-4, mtp_weight=0.1,
                    distributed=False, optimizers=None,
                    label_smoothing_sigma=0.0):
    """
    optimizers: Muon 模式下传入 [muon_opt, adamw_opt] 列表，
                非 Muon 模式下为 None（仅使用 optimizer 参数）。
    """
    model.train()
    total_loss = 0
    total_bpp = 0
    steps = len(loader)

    # 统一的 zero_grad / step 帮助函数
    all_opts = optimizers if optimizers else [optimizer]
    for opt in all_opts:
        opt.zero_grad(set_to_none=True)

    for i, batch in enumerate(loader):
        if isinstance(batch, (list, tuple)):
            x = batch[0]
        else:
            x = batch

        x = x.to(device)
        _, C, _, _ = x.shape

        # DDP no_sync: 梯度累积中间步跳过 AllReduce，只在同步步通信
        # Ref: PyTorch DDP 文档 — model.no_sync() 跳过梯度 AllReduce
        is_accumulating = (i + 1) % grad_accum_steps != 0
        sync_context = model.no_sync() if (distributed and is_accumulating) else nullcontext()

        with sync_context:
            # AMP autocast：有 scaler 时启用 fp16/bf16 混合精度
            amp_ctx = autocast(device_type="cuda", dtype=amp_dtype) if scaler is not None else nullcontext()
            with amp_ctx:
                out = model(x, z_loss_weight=z_loss_weight, mtp_weight=mtp_weight,
                            label_smoothing_sigma=label_smoothing_sigma)
                loss = out["loss"]
                ce_loss = out["ce_loss"]
                # BPP 只用 ce_loss（纯交叉熵），不含 z_loss 正则项。
                # Ref: Shannon, "A Mathematical Theory of Communication," 1948
                bpp = compute_bpp(ce_loss, C)
            loss_scaled = loss / grad_accum_steps
            if scaler is not None:
                scaler.scale(loss_scaled).backward()
            else:
                loss_scaled.backward()

        if (i + 1) % grad_accum_steps == 0:
            # Unscale → clip → step → update（统一 scaler/非 scaler 路径）
            if scaler is not None:
                for opt in all_opts:
                    scaler.unscale_(opt)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
            if scaler is not None:
                for opt in all_opts:
                    scaler.step(opt)
                scaler.update()
            else:
                for opt in all_opts:
                    opt.step()

            if writer and (i + 1) % log_freq == 0:
                step = epoch * steps + i
                writer.add_scalar('train/grad_norm', grad_norm.item(), step)
                if scaler is not None:
                    writer.add_scalar('train/loss_scale', scaler.get_scale(), step)

            for opt in all_opts:
                opt.zero_grad(set_to_none=True)

        # 用 tensor 累加，避免每步 .item() 触发 GPU→CPU 同步
        # Ref: CS336 — 只在 log 时才 .item()
        total_loss += loss.detach()
        total_bpp  += bpp.detach()

        # NaN/Inf 检测：loss 异常时提前警告，避免浪费 GPU 时间
        # Ref: PyTorch Lightning — NaN detection callback 的设计思路
        if (i + 1) % log_freq == 0:
            loss_val = loss.item()
            bpp_val  = bpp.item()
            if not math.isfinite(loss_val):
                print(f"WARNING: loss is {loss_val} at epoch {epoch} step {i+1}/{steps}. "
                      f"LR={optimizer.param_groups[0]['lr']:.2e}. Training may diverge.")
            print(f"Epoch {epoch} Step {i+1}/{steps} | Loss: {loss_val:.4f} | BPP: {bpp_val:.4f}")
            if writer:
                step = epoch * steps + i
                writer.add_scalar('train/loss', loss_val, step)
                writer.add_scalar('train/bpp',  bpp_val,  step)

    return total_loss.item() / steps, total_bpp.item() / steps


# ==================== Validation ====================
@torch.no_grad()
def validate(model, loader, device, amp_dtype=None):
    """验证集评估，返回 (avg_bpp, std_bpp, avg_loss)。

    amp_dtype: 传入 AMP dtype（如 torch.bfloat16）以在验证时也使用混合精度，
               减少显存占用和加速。None 则使用 fp32。
    """
    model.eval()
    bpp_list = []
    total_loss = 0
    use_amp = amp_dtype is not None and device.type == 'cuda'
    for batch in loader:
        if isinstance(batch, (list, tuple)):
            x = batch[0]
        else:
            x = batch
        x = x.to(device)
        _, C, _, _ = x.shape
        # 验证时也使用 AMP，与训练保持一致的数值精度，同时节省显存
        with autocast(device_type="cuda", dtype=amp_dtype) if use_amp else nullcontext():
            out = model(x)
        loss = out["loss"]
        ce_loss = out["ce_loss"]
        # 验证 BPP：同训练一样，只用 ce_loss 计算，排除 z_loss 正则项
        bpp = compute_bpp(ce_loss, C)
        bpp_list.append(bpp.item())
        total_loss += loss.item()
    avg_bpp = float(np.mean(bpp_list))
    std_bpp = float(np.std(bpp_list))
    avg_loss = total_loss / len(bpp_list)
    return avg_bpp, std_bpp, avg_loss


# ==================== Main ====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config yaml')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    # --seed: 命令行覆盖 config 中的 seed，方便多次独立运行取均值
    # 用法: python train.py --config ... --seed 0 / --seed 1 / --seed 2
    parser.add_argument('--seed', type=int, default=None, help='Random seed (overrides config)')
    # --export_csv: 导出训练/验证曲线为 CSV，方便 matplotlib 画论文图
    parser.add_argument('--export_csv', action='store_true',
                        help='导出 loss/BPP/LR 曲线为 CSV 文件')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

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

    # CSV 导出器：将训练曲线写入 CSV，方便 matplotlib 画论文图
    csv_file = None
    csv_writer = None
    if args.export_csv:
        csv_path = os.path.join(log_dir, 'training_curves.csv')
        csv_file = open(csv_path, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['epoch', 'train_loss', 'train_bpp', 'val_loss',
                             'val_bpp', 'val_bpp_std', 'lr'])

    # Distributed training: 自动检测 torchrun 环境
    # torchrun 会设置 WORLD_SIZE、RANK、LOCAL_RANK 等环境变量，
    # 无需在 config 中手动指定 distributed.enabled。
    # Ref: PyTorch Distributed — torchrun elastic launch 文档
    distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1
    if distributed:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')
        rank = dist.get_rank()
    else:
        local_rank = 0
        rank = 0

    device = torch.device(f'cuda:{local_rank}' if distributed else 'cuda' if torch.cuda.is_available() else 'cpu')
    writer = SummaryWriter(log_dir=log_dir) if rank == 0 else None

    # Dataset: 根据 config 选择 CIFAR-10 或 CIFAR-100
    from torchvision.datasets import CIFAR10, CIFAR100
    transform = transforms.ToTensor()
    dataset_name = config["data"].get("dataset", "cifar100")
    DatasetClass = CIFAR10 if dataset_name == "cifar10" else CIFAR100
    train_dataset = DatasetClass(root=config["data"]["train"], train=True,  download=False, transform=transform)
    valid_dataset = DatasetClass(root=config["data"]["valid"], train=False, download=False, transform=transform)
    if rank == 0:
        print(f"Dataset: {dataset_name} | Train: {len(train_dataset)} | Valid: {len(valid_dataset)}")

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
    valid_loader = DataLoader(valid_dataset, batch_size=valid_batch_size, shuffle=False,
                              num_workers=valid_num_workers, pin_memory=True)

    # Model — 传递所有消融开关
    mcfg = config["model"]
    model = _build_model_from_config(mcfg, device)

    # Fused kernel 状态日志：帮助用户确认 Triton kernel 是否生效
    # 若全部为 False，训练仍可正常进行（回退到 PyTorch 实现），但性能会降低。
    if rank == 0:
        kernel_status = get_fused_kernel_status()
        active = sum(kernel_status.values())
        print(f"Fused Triton kernels: {active}/{len(kernel_status)} active")
        for name, avail in kernel_status.items():
            print(f"  {name}: {'ON' if avail else 'OFF (fallback to PyTorch)'}")

    # muP 初始化（可选）
    # Ref: Yang et al., "Tensor Programs V," arXiv:2203.03466, 2022.
    mup_cfg = config["train"].get("mup", {})
    mup_enabled = mup_cfg.get("enabled", False)
    mup_base_width = mup_cfg.get("base_width", 64)
    if mup_enabled:
        model._init_weights_mup(mup_base_width)
        if rank == 0:
            print(f"muP init enabled: base_width={mup_base_width}, d_model={mcfg['d_model']}")

    # torch.compile（Phase 2.3）— 在 DDP 包装前编译
    # Ref: PyTorch 2.0+ torch.compile 文档
    if config["train"].get("compile", False):
        model = torch.compile(model, fullgraph=True)
        if rank == 0:
            print("torch.compile enabled (fullgraph=True)")

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank
        )

    # Optimizer with parameter groups (no weight decay for embeddings/norm/bias)
    # muP 启用时: hidden 层 LR 按 base_width/d_model 缩放 [1] Table 8
    base_lr = float(config["train"]["lr"])
    d_model = mcfg["d_model"]

    # Muon 优化器（可选）：对 2D 权重矩阵使用 Newton-Schulz 正交化 SGD+Momentum
    # 非 2D 参数（embedding, norm, bias）仍用 AdamW
    # Ref: Keller Jordan, 2024; arXiv:2502.16982
    muon_cfg = config["train"].get("muon", {})
    muon_enabled = muon_cfg.get("enabled", False)

    if muon_enabled:
        from src.mdlic.optim.muon import build_muon_adamw
        muon_opt, adamw_opt = build_muon_adamw(
            model,
            muon_lr=float(muon_cfg.get("lr", 0.02)),
            muon_momentum=float(muon_cfg.get("momentum", 0.95)),
            muon_ns_steps=int(muon_cfg.get("ns_steps", 5)),
            adamw_lr=base_lr,
            adamw_betas=(0.9, 0.95),
            adamw_eps=float(config["train"].get("eps", 1e-8)),
            adamw_wd=0.1,
            no_decay_keywords=['token_embed', 'pos_embed', 'norm', 'bias'],
        )
        # 包装成统一接口：optimizer 是一个 list，后续代码统一处理
        optimizers = [opt for opt in [muon_opt, adamw_opt] if opt is not None]
        # 用 adamw_opt 作为主 optimizer（scheduler 挂在它上面）
        optimizer = adamw_opt if adamw_opt is not None else muon_opt
        if rank == 0:
            n_muon = sum(p.numel() for g in muon_opt.param_groups for p in g['params']) if muon_opt else 0
            n_adamw = sum(p.numel() for g in adamw_opt.param_groups for p in g['params']) if adamw_opt else 0
            print(f"Muon enabled: {n_muon:,} params (2D) → Muon (lr={muon_cfg.get('lr', 0.02)}), "
                  f"{n_adamw:,} params → AdamW (lr={base_lr})")
    else:
        optimizers = None  # 标记未使用 Muon

        optimizer = optim.AdamW(
            _get_param_groups(model, weight_decay=0.1, mup_enabled=mup_enabled,
                              mup_base_width=mup_base_width, d_model=d_model,
                              base_lr=base_lr),
            lr=base_lr,
            betas=(0.9, 0.95),
            eps=float(config["train"].get("eps", 1e-8)),
        )

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
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return float(epoch + 1) / float(max(1, warmup_epochs))
            progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
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
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda_wsd)
    elif lr_schedule == "multistep":
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=config["train"]["lr_milestones"],
            gamma=config["train"].get("lr_gamma", 0.1),
        )
    else:
        scheduler = None

    # Mixed precision
    amp_dtype = torch.bfloat16 if config["train"].get("amp_dtype", "fp16") == "bf16" else torch.float16
    scaler = GradScaler() if amp_dtype == torch.float16 else None
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

    best_bpp = float('inf')
    start_epoch = 1

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        # Checkpoint 兼容性检查：验证 state_dict keys 匹配
        # strict=True 时 key 不匹配会直接报错；改为先检查再加载，
        # 提供更清晰的诊断信息，避免 resume 后 silent failure。
        # Ref: PyTorch 文档 — load_state_dict(strict=True) 的 missing/unexpected keys
        model_keys = set(model.state_dict().keys())
        ckpt_keys = set(ckpt['model_state_dict'].keys())
        missing = model_keys - ckpt_keys
        unexpected = ckpt_keys - model_keys
        if missing or unexpected:
            if rank == 0:
                if missing:
                    print(f"WARNING: checkpoint 缺少 {len(missing)} 个参数: {list(missing)[:5]}...")
                if unexpected:
                    print(f"WARNING: checkpoint 多余 {len(unexpected)} 个参数: {list(unexpected)[:5]}...")
                print("尝试 non-strict 加载（missing 参数保持随机初始化）...")
            model.load_state_dict(ckpt['model_state_dict'], strict=False)
        else:
            model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        # Muon 模式下恢复所有 optimizer 状态
        if optimizers and 'optimizers_state_dict' in ckpt:
            for opt, sd in zip(optimizers, ckpt['optimizers_state_dict']):
                opt.load_state_dict(sd)
        start_epoch = ckpt['epoch'] + 1
        # 恢复 best_bpp，避免 resume 后重复保存非最优 checkpoint
        best_bpp = ckpt.get('best_bpp', float('inf'))
        # 恢复 scheduler，避免 LR schedule 从头开始
        if scheduler is not None and 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        # 恢复 GradScaler（fp16 模式下的 loss scale 状态）
        if scaler is not None and 'scaler_state_dict' in ckpt:
            scaler.load_state_dict(ckpt['scaler_state_dict'])
        # 恢复 SWA 状态
        if swa_enabled and 'swa_state' in ckpt:
            swa_state = ckpt['swa_state']
            swa_n = ckpt.get('swa_n', 0)
        if rank == 0:
            print(f"Resumed from checkpoint '{args.resume}' (epoch {ckpt['epoch']}, best_bpp={best_bpp:.4f})")

    for epoch in range(start_epoch, config['train']['epochs'] + 1):
        if distributed:
            train_sampler.set_epoch(epoch)

        avg_loss, avg_bpp = train_one_epoch(
            model, train_loader, optimizer, scaler,
            device=device,
            epoch=epoch,
            log_freq=config['train']['log_freq'],
            writer=writer,
            clip_max_norm=config['train']['clip_max_norm'],
            amp_dtype=amp_dtype,
            grad_accum_steps=grad_accum_steps,
            z_loss_weight=float(config["train"].get("z_loss_weight", 1e-4)),
            mtp_weight=float(config["train"].get("mtp_weight", 0.1)),
            distributed=distributed,
            optimizers=optimizers,
            label_smoothing_sigma=float(config["train"].get("label_smoothing_sigma", 0.0)),
        )

        if rank == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch} | Loss: {avg_loss:.4f} | BPP: {avg_bpp:.4f} | LR: {current_lr:.2e}")
            if writer:
                writer.add_scalar('train/lr', current_lr, epoch)

        if epoch % config['eval']['interval'] == 0:
            bpp_avg, std_bpp, loss_avg = validate(model, valid_loader, device, amp_dtype=amp_dtype)
            if rank == 0:
                print(f"Validation: Loss {loss_avg:.4f} | BPP {bpp_avg:.4f} ± {std_bpp:.4f}")
                if writer:
                    writer.add_scalar('val/loss', loss_avg, epoch)
                    writer.add_scalar('val/bpp', bpp_avg, epoch)
                    writer.add_scalar('val/bpp_std', std_bpp, epoch)
                # CSV 导出
                if csv_writer:
                    current_lr = optimizer.param_groups[0]['lr']
                    csv_writer.writerow([epoch, f'{avg_loss:.6f}', f'{avg_bpp:.6f}',
                                         f'{loss_avg:.6f}', f'{bpp_avg:.6f}',
                                         f'{std_bpp:.6f}', f'{current_lr:.2e}'])
                    csv_file.flush()
            if bpp_avg < best_bpp:
                best_bpp = bpp_avg
                if rank == 0:
                    torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))

        if scheduler is not None:
            scheduler.step()

        # SWA 更新：running average of model weights
        # swa_state[name] = swa_state[name] + (param - swa_state[name]) / (n+1)
        #                  = swa_state[name].lerp_(param, 1/(n+1))
        # Ref: Izmailov et al., arXiv:1803.05407
        if swa_enabled and epoch >= swa_start_epoch and epoch % swa_update_interval == 0:
            raw_model = model.module if distributed else model
            # NaN 防护：检查当前参数是否包含 NaN，避免污染 SWA 平均权重
            # Ref: Izmailov et al., arXiv:1803.05407 — SWA 依赖各 checkpoint 权重的质量
            has_nan = any(torch.isnan(p.data).any().item() for p in raw_model.parameters())
            if has_nan:
                if rank == 0:
                    print(f"  WARNING: skipping SWA update at epoch {epoch} — model contains NaN weights")
            elif swa_state is None:
                # 第一次 SWA 更新：初始化为当前权重的 clone
                swa_state = {name: param.data.clone()
                             for name, param in raw_model.named_parameters()}
                swa_n = 1
            else:
                swa_n += 1
                for name, param in raw_model.named_parameters():
                    # lerp_: swa = swa + (param - swa) / n = running mean
                    swa_state[name].lerp_(param.data, 1.0 / swa_n)
            if rank == 0 and not has_nan:
                print(f"  SWA update #{swa_n} at epoch {epoch}")

        if rank == 0 and epoch % config['checkpoint']['save_interval'] == 0:
            # 完整保存训练状态，确保 --resume 后所有组件正确恢复
            ckpt_data = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'best_bpp': best_bpp,
            }
            # Muon 模式下额外保存所有 optimizer 状态
            if optimizers:
                ckpt_data['optimizers_state_dict'] = [opt.state_dict() for opt in optimizers]
            if scheduler is not None:
                ckpt_data['scheduler_state_dict'] = scheduler.state_dict()
            if scaler is not None:
                ckpt_data['scaler_state_dict'] = scaler.state_dict()
            if swa_state is not None:
                ckpt_data['swa_state'] = swa_state
                ckpt_data['swa_n'] = swa_n
            torch.save(ckpt_data, os.path.join(checkpoint_dir, f'epoch_{epoch}.pth'))

    # SWA 后处理：替换权重 → 重新验证 → 保存
    if swa_enabled and swa_state is not None:
        raw_model = model.module if distributed else model
        for name, param in raw_model.named_parameters():
            param.data.copy_(swa_state[name])
        if rank == 0:
            print(f"SWA: replaced model weights (averaged over {swa_n} checkpoints)")
        # 用 SWA 权重重新验证
        bpp_avg, std_bpp, loss_avg = validate(model, valid_loader, device, amp_dtype=amp_dtype)
        if rank == 0:
            print(f"SWA Validation: Loss {loss_avg:.4f} | BPP {bpp_avg:.4f} ± {std_bpp:.4f}")
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'swa.pth'))
            if writer:
                writer.add_scalar('val/swa_bpp', bpp_avg, total_epochs)

    if rank == 0:
        if csv_file:
            csv_file.close()
            print(f"Training curves saved to {os.path.join(log_dir, 'training_curves.csv')}")
        if writer:
            writer.close()
        print("Training finished.")


if __name__ == '__main__':
    main()
