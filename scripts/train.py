#!/usr/bin/env python3
"""
Training script for iGPT autoregressive image compression.

Usage:
    python scripts/train.py --config configs/igpt_cifar10_s_rgb.yaml
    python scripts/train.py --config configs/igpt_cifar10_s_rgb.yaml --resume experiments/.../epoch_10.pth
    torchrun --nproc_per_node=2 scripts/train.py --config configs/igpt_cifar10_s_rgb.yaml
"""

import os
import sys
import csv
import argparse
import yaml
import math
import torch
import torch.nn as nn
import torch.optim as optim
from contextlib import nullcontext
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from torchvision import transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.mdlic.models.igpt import IGPT
from src.mdlic.models.cc_igpt import CCIGPT
from src.mdlic.models.layers import get_fused_kernel_status
from src.mdlic.utils import seed_everything, compute_bpd, clean_state_dict


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
    assert lr_schedule in ('cosine', 'wsd', 'multistep', 'constant'), (
        f"train.lr_schedule 必须是 cosine/wsd/multistep/constant，got '{lr_schedule}'"
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
    from src.mdlic.models.layers import RMSNorm
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
    total_loss = 0
    total_bpd = 0
    steps = len(loader)

    for opt in optimizers:
        opt.zero_grad(set_to_none=True)

    for i, batch in enumerate(loader):
        if isinstance(batch, (list, tuple)):
            x = batch[0]
        else:
            x = batch

        x = x.to(device)

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
            loss_scaled = loss / grad_accum_steps
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
        total_loss += loss.detach()
        total_bpd  += bpd.detach()

        # NaN/Inf 检测：loss 异常时提前警告，避免浪费 GPU 时间
        # Ref: PyTorch Lightning — NaN detection callback 的设计思路
        if (i + 1) % log_freq == 0:
            loss_val = loss.item()
            bpd_val  = bpd.item()
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
                          f" | CE_c: {out['ce_loss_coarse'].item():.4f}"
                          f" | CE_f: {out['ce_loss_fine'].item():.4f}"
                          f" | α: {out['ctx_alpha'].item():.3f}")
                else:
                    print(f"Epoch {epoch} Step {i+1}/{steps} | Loss: {loss_val:.4f} | bits/dim: {bpd_val:.4f}")
            if writer:
                step = epoch * steps + i
                writer.add_scalar('train/loss', loss_val, step)
                writer.add_scalar('train/bpd',  bpd_val,  step)
                if "ce_loss_coarse" in out and "ce_loss_fine" in out:
                    writer.add_scalar('train/ce_coarse', out['ce_loss_coarse'].item(), step)
                    writer.add_scalar('train/ce_fine',   out['ce_loss_fine'].item(),   step)
                    writer.add_scalar('train/ctx_alpha', out['ctx_alpha'].item(),      step)

    return total_loss.item() / steps, total_bpd.item() / steps


# ==================== Validation ====================
@torch.no_grad()
def validate(model, loader, device, amp_dtype=None):
    """验证集评估，返回 (avg_bpd, std_bpd_batch, avg_loss)。

    DDP-aware：每个 rank 通过 DistributedSampler 处理一个分片，最后用
    all_reduce(SUM) 聚合 weighted sums + n_total，避免 rank 0 单跑导致 barrier 超时。

    amp_dtype: 传入 AMP dtype（如 torch.bfloat16）以在验证时也使用混合精度，
               减少显存占用和加速。None 则使用 fp32。

    std_bpd 语义（重要，论文报告时注意）:
        当前的 `std_bpd` 是 **batch 级加权波动**，不是图像级标准差。
        公式 var = E[batch_mean_bpd² · B] / n_total − avg_bpd² 里 batch_mean_bpd
        已经是 batch 内平均，按 B 加权后得到的是 batch 间平均 bpd 的加权方差，
        数值通常比 per-image std 小（batch 内部方差被平均抹掉）。
        论文若要报告 per-image std，需让 forward 返回 reduction='none' 的
        per-token/per-image CE 后重算。此处保留 batch-level 便于训练监控，
        evaluate.py cmd_single 也使用同样语义。
    """
    model.eval()
    total_bpd_weighted = 0.0
    total_loss_weighted = 0.0
    sum_sq_bpd_batch = 0.0   # Σ (batch_mean_bpd)² · B ，batch-level std
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
        ce_loss = out["ce_loss"]
        if "bpd" in out and out["bpd"] is not None:
            bpd = out["bpd"]
        else:
            bpd = compute_bpd(ce_loss)
        bpd_val = bpd.item()
        loss_val = loss.item()
        total_bpd_weighted += bpd_val * B
        total_loss_weighted += loss_val * B
        sum_sq_bpd_batch += (bpd_val ** 2) * B
        n_total += B

    # DDP 聚合：跨 rank 求和后再求平均，结果在所有 rank 上一致
    if dist.is_available() and dist.is_initialized():
        agg = torch.tensor([total_bpd_weighted, total_loss_weighted, sum_sq_bpd_batch, float(n_total)],
                           device=device, dtype=torch.float64)
        dist.all_reduce(agg, op=dist.ReduceOp.SUM)
        total_bpd_weighted, total_loss_weighted, sum_sq_bpd_batch, n_total_f = agg.tolist()
        n_total = int(n_total_f)

    avg_bpd = total_bpd_weighted / n_total
    avg_loss = total_loss_weighted / n_total
    var_bpd_batch = max(sum_sq_bpd_batch / n_total - avg_bpd ** 2, 0.0)
    std_bpd_batch = float(math.sqrt(var_bpd_batch))
    return avg_bpd, std_bpd_batch, avg_loss



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
                        help='导出 loss/bpd/LR 曲线为 CSV 文件')
    args = parser.parse_args()

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
        # rank 0 单独跑 validation 期间，rank 1 在 barrier 等待，需放宽 NCCL 超时（默认 10min）
        from datetime import timedelta
        dist.init_process_group(backend='nccl', timeout=timedelta(hours=2))
        rank = dist.get_rank()
    else:
        local_rank = 0
        rank = 0

    device = torch.device(f'cuda:{local_rank}' if distributed else 'cuda' if torch.cuda.is_available() else 'cpu')
    writer = SummaryWriter(log_dir=log_dir) if rank == 0 else None

    if args.export_csv and rank == 0:
        csv_path = os.path.join(log_dir, 'training_curves.csv')
        csv_file = open(csv_path, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['epoch', 'train_loss', 'train_bpd', 'val_loss',
                             'val_bpd', 'val_bpd_std', 'lr'])

    # Dataset: 根据 config 选择 CIFAR-10 / CIFAR-100 / ImageNet32 npy
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
    elif dataset_name == "imagenet32_npy":
        from src.mdlic.data.imagenet32_npy import ImageNet32Npy
        if rank == 0 and (aug_cfg.get("hflip", False) or crop_cfg):
            print("WARNING: data.augment (hflip/random_crop) 在 imagenet32_npy 路径上当前未实现，已忽略")
        train_dataset = ImageNet32Npy(root=config["data"]["train"], split="train")
        valid_dataset = ImageNet32Npy(root=config["data"]["valid"], split="val")
    else:
        raise ValueError(f"未知 dataset: '{dataset_name}'，支持 cifar10/cifar100/imagenet32_npy")
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
    # drop_last=True: 避免 DistributedSampler 默认 padding（重复 dataset 头部样本
    # 让每个 rank 拿到整除分片）造成验证集统计偏差。代价是最多丢 world_size-1
    # 个样本（CIFAR-10/100 val=10000、ImageNet32 val=50000 在 world=2 下整除，
    # 不丢；world=3 时丢 1 个，不影响 bpd 数值精度）。
    valid_sampler = (DistributedSampler(valid_dataset, shuffle=False, drop_last=True)
                     if distributed else None)
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
    elif lr_schedule == "multistep":
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizers[0],
            milestones=config["train"]["lr_milestones"],
            gamma=config["train"].get("lr_gamma", 0.1),
        )
    else:
        scheduler = None

    # Mixed precision: None/"none"/"fp32" → 禁用 AMP 走 fp32（数值敏感场景兜底）
    amp_cfg = config["train"].get("amp_dtype", "fp16")
    if amp_cfg in (None, "none", "fp32"):
        amp_dtype = None
    elif amp_cfg == "bf16":
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = torch.float16
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

    # EMA (Exponential Moving Average) 初始化
    # rank 0 维护 fp32 影子权重；每个 optimizer.step() 后更新；finalize 时
    # broadcast 到全 rank → validate → 保存 ema.pth。
    # Ref: Polyak & Juditsky 1992；iGPT (Chen 2020) / EDM / Stable Diffusion 标配。
    ema_cfg = config["train"].get("ema", {})
    ema_enabled = ema_cfg.get("enabled", False)
    ema_decay = float(ema_cfg.get("decay", 0.999))
    ema_state = None
    if ema_enabled and rank == 0:
        print(f"EMA enabled: decay={ema_decay}")

    best_bpd = float('inf')
    start_epoch = 1

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
        if 'model_state_dict' not in ckpt:
            # 允许 resume 参数指向裸 state_dict
            raw_model.load_state_dict(clean_state_dict(ckpt))
            start_epoch = 1
            if rank == 0:
                print(f"Loaded bare state_dict from '{args.resume}' (resume 将从 epoch 1 开始)")
        else:
            sd = clean_state_dict(ckpt['model_state_dict'])
            model_keys = set(raw_model.state_dict().keys())
            ckpt_keys = set(sd.keys())
            missing = model_keys - ckpt_keys
            unexpected = ckpt_keys - model_keys
            if missing or unexpected:
                critical_keys = {k for k in model_keys
                                 if 'token_embed' in k or 'blocks.' in k or 'head.' in k}
                critical_missing = missing & critical_keys
                if critical_missing:
                    raise RuntimeError(
                        f"Checkpoint 缺少 {len(critical_missing)} 个关键参数: "
                        f"{list(critical_missing)[:10]}。"
                        "请确认 --resume 指向与当前 config 兼容的 checkpoint。"
                    )
                if rank == 0:
                    if missing:
                        print(f"WARNING: checkpoint 缺少 {len(missing)} 个非关键参数: {list(missing)[:5]}...")
                    if unexpected:
                        print(f"WARNING: checkpoint 多余 {len(unexpected)} 个参数: {list(unexpected)[:5]}...")
                    print("尝试 non-strict 加载（missing 参数保持随机初始化）...")
                raw_model.load_state_dict(sd, strict=False)
            else:
                raw_model.load_state_dict(sd)
            if 'optimizer_state_dicts' in ckpt:
                # 多 optimizer 格式（list）— 当前为单元素，保留 list 接口兼容
                opt_states = ckpt['optimizer_state_dicts']
                assert len(opt_states) == len(optimizers), (
                    f"ckpt 含 {len(opt_states)} 个 optimizer state，当前模型用 {len(optimizers)} 个"
                )
                for opt, sd_opt in zip(optimizers, opt_states):
                    opt.load_state_dict(sd_opt)
            elif 'optimizer_state_dict' in ckpt:
                # 历史单 optimizer 格式（标量 dict 而非 list）
                optimizers[0].load_state_dict(ckpt['optimizer_state_dict'])
            start_epoch = ckpt.get('epoch', 0) + 1
            best_bpd = ckpt.get('best_bpd', float('inf'))
            if scheduler is not None and 'scheduler_state_dict' in ckpt:
                scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            if scaler is not None and 'scaler_state_dict' in ckpt:
                scaler.load_state_dict(ckpt['scaler_state_dict'])
            if swa_enabled and 'swa_state' in ckpt:
                swa_state = ckpt['swa_state']
                swa_n = ckpt.get('swa_n', 0)
            if ema_enabled and 'ema_state' in ckpt:
                ema_state = ckpt['ema_state']
            # resume 后用 config 中的 lr 覆盖 checkpoint 里的旧值，
            # 使得修改 yaml lr 后 resume 能立即生效。
            for pg in optimizers[0].param_groups:
                pg['initial_lr'] = base_lr
                pg['lr'] = base_lr
            # Scheduler base_lrs 同步（cosine/wsd 的 lr_lambda 乘以 base_lr）。
            # last_epoch=start_epoch-2 后 step() 一次推进到 start_epoch-1，并触发
            # lr_lambda 把 pg['lr'] 同步成正确的衰减值。否则 LambdaLR.__init__ 后
            # 主循环 epoch 末才 step()，resume 后第一个 epoch 会跑全峰值 lr。
            if scheduler is not None:
                scheduler.base_lrs = [pg['initial_lr'] for pg in optimizers[0].param_groups]
                scheduler.last_epoch = start_epoch - 2
                scheduler.step()

            if rank == 0:
                print(f"Resumed from checkpoint '{args.resume}' (epoch {ckpt.get('epoch', '?')}, best_bpd={best_bpd:.4f})")
                print(f"  LR overridden to config value: {base_lr:.2e} (schedule continues from epoch {start_epoch})")

    def _ema_step():
        """rank 0 上每个 optimizer.step() 后调用：用 fp32 累加更新影子权重。

        ema_state 延迟初始化：第一次调用时按 raw_model 当前权重 clone 出 fp32 副本。
        非 rank 0 走空操作（ema_state 始终 None，由 finalize broadcast 同步）。

        NaN 守卫：与 SWA 同款 batched 检查 — EMA 是累积平均，单步 NaN
        会通过 (1-decay) 项渗入并永久污染；首次克隆时 NaN 也会让所有
        后续 add_ 输出 NaN。检测到则跳过本 tick，下一步自动重试。
        """
        nonlocal ema_state
        if not ema_enabled or rank != 0:
            return
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
            bpd_avg, std_bpd, loss_avg = validate(model, valid_loader, device, amp_dtype=amp_dtype)
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

        if rank == 0 and epoch % config['checkpoint']['save_interval'] == 0:
            # 完整保存训练状态，确保 --resume 后所有组件正确恢复
            ckpt_data = {
                'epoch': epoch,
                'model_state_dict': raw_model.state_dict(),
                # 多 optimizer 格式：list of state_dicts（当前为单元素 list）
                'optimizer_state_dicts': [opt.state_dict() for opt in optimizers],
                'loss': avg_loss,
                'best_bpd': best_bpd,
            }
            if scheduler is not None:
                ckpt_data['scheduler_state_dict'] = scheduler.state_dict()
            if scaler is not None:
                ckpt_data['scaler_state_dict'] = scaler.state_dict()
            if swa_state is not None:
                ckpt_data['swa_state'] = swa_state
                ckpt_data['swa_n'] = swa_n
            if ema_state is not None:
                ckpt_data['ema_state'] = ema_state
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
            bpd_avg, std_bpd, loss_avg = validate(model, valid_loader, device, amp_dtype=amp_dtype)
            if rank == 0:
                print(f"EMA Validation: Loss {loss_avg:.4f} | bits/dim {bpd_avg:.4f} ± {std_bpd:.4f}")
                _atomic_save(raw_model.state_dict(), os.path.join(checkpoint_dir, 'ema.pth'))
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
            bpd_avg, std_bpd, loss_avg = validate(model, valid_loader, device, amp_dtype=amp_dtype)
            if rank == 0:
                print(f"SWA Validation: Loss {loss_avg:.4f} | bits/dim {bpd_avg:.4f} ± {std_bpd:.4f}")
                _atomic_save(raw_model.state_dict(), os.path.join(checkpoint_dir, 'swa.pth'))
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
