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
import argparse
import yaml
import math
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from torchvision import transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.mdlic.models.igpt import IGPT


# ==================== Training ====================
def train_one_epoch(model, loader, optimizer, scaler, device,
                    epoch, log_freq, writer, clip_max_norm,
                    amp_dtype=torch.float16, grad_accum_steps=1,
                    z_loss_weight=1e-4, mtp_weight=0.1):
    model.train()
    total_loss = 0
    total_bpp = 0
    steps = len(loader)

    optimizer.zero_grad(set_to_none=True)

    for i, batch in enumerate(loader):
        if isinstance(batch, (list, tuple)):
            x = batch[0]
        else:
            x = batch

        x = x.to(device)
        _, C, _, _ = x.shape

        if scaler is not None:
            with autocast(dtype=amp_dtype):
                out = model(x, z_loss_weight=z_loss_weight, mtp_weight=mtp_weight)
                loss = out["loss"]
                bpp = (loss / math.log(2)) * C
            loss_scaled = loss / grad_accum_steps
            scaler.scale(loss_scaled).backward()
        else:
            out = model(x, z_loss_weight=z_loss_weight, mtp_weight=mtp_weight)
            loss = out["loss"]
            bpp = (loss / math.log(2)) * C
            loss_scaled = loss / grad_accum_steps
            loss_scaled.backward()

        if (i + 1) % grad_accum_steps == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
                scaler.step(optimizer)
                scaler.update()
                if (i + 1) % log_freq == 0:
                    print(f"  [AMP] loss_scale={scaler.get_scale():.1f}")
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
                optimizer.step()
                if (i + 1) % log_freq == 0:
                    total_norm = sum(
                        p.grad.norm().item() ** 2
                        for p in model.parameters()
                        if p.grad is not None
                    ) ** 0.5
                    print(f"  [GRAD] grad_norm={total_norm:.4f}")

            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item()
        total_bpp += bpp.item()

        if (i + 1) % log_freq == 0:
            print(f"Epoch {epoch} Step {i+1}/{steps} | Loss: {loss.item():.4f} | BPP: {bpp.item():.4f}")
            if writer:
                step = epoch * steps + i
                writer.add_scalar('train/loss', loss.item(), step)
                writer.add_scalar('train/bpp', bpp.item(), step)

    return total_loss / steps, total_bpp / steps


# ==================== Validation ====================
@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    bpp_list = []
    total_loss = 0
    for batch in loader:
        if isinstance(batch, (list, tuple)):
            x = batch[0]
        else:
            x = batch
        x = x.to(device)
        _, C, _, _ = x.shape
        out = model(x)
        loss = out["loss"]
        bpp = (loss / math.log(2)) * C
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
    parser.add_argument('--local_rank', type=int, default=-1, help='For distributed training')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    seed = config["train"].get("seed", 42)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    exp_name = config['exp_name']
    checkpoint_dir = os.path.join('experiments', exp_name, 'checkpoints')
    log_dir = os.path.join('experiments', exp_name, 'logs')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Distributed training
    distributed = config.get('distributed', {}).get('enabled', False)
    if distributed:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')
        rank = dist.get_rank()
    else:
        rank = 0

    device = torch.device(f'cuda:{args.local_rank}' if distributed else 'cuda' if torch.cuda.is_available() else 'cpu')
    writer = SummaryWriter(log_dir=log_dir) if rank == 0 else None

    # Dataset (CIFAR-100)
    from torchvision.datasets import CIFAR100
    transform = transforms.ToTensor()
    train_dataset = CIFAR100(root=config["data"]["train"], train=True,  download=False, transform=transform)
    valid_dataset = CIFAR100(root=config["data"]["valid"], train=False, download=False, transform=transform)

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
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    # Model
    model = IGPT(
        image_size=config["model"]["image_size"],
        in_channels=config["model"]["in_channels"],
        vocab_size=config["model"]["vocab_size"],
        d_model=config["model"]["d_model"],
        N=config["model"]["N"],
        h=config["model"]["h"],
        d_ff=config["model"]["d_ff"],
        dropout=config["model"]["dropout"],
        use_mtp=config["model"].get("use_mtp", False),
    ).to(device)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank
        )

    # Optimizer with parameter groups (no weight decay for embeddings/norm/bias)
    def get_param_groups(model, weight_decay=0.1):
        no_decay = set()
        for name, _ in model.named_parameters():
            if any(nd in name for nd in ['token_embed', 'norm', 'bias']):
                no_decay.add(name)
        return [
            {"params": [p for n, p in model.named_parameters() if n not in no_decay], "weight_decay": weight_decay},
            {"params": [p for n, p in model.named_parameters() if n in no_decay],     "weight_decay": 0.0},
        ]

    optimizer = optim.AdamW(
        get_param_groups(model),
        lr=float(config["train"]["lr"]),
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
    lr_schedule = config["train"].get("lr_schedule", "cosine")
    if lr_schedule == "cosine":
        warmup_epochs = config["train"].get("warmup_epochs", 5)
        total_epochs  = config["train"]["epochs"]
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return float(epoch + 1) / float(max(1, warmup_epochs))
            progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
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

    best_bpp = float('inf')
    start_epoch = 1

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        if rank == 0:
            print(f"Resumed from checkpoint '{args.resume}' (epoch {ckpt['epoch']})")

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
            z_loss_weight=config["train"].get("z_loss_weight", 1e-4),
            mtp_weight=config["train"].get("mtp_weight", 0.1),
        )

        if rank == 0:
            print(f"Epoch {epoch} | Loss: {avg_loss:.4f} | BPP: {avg_bpp:.4f}")

        if epoch % config['eval']['interval'] == 0:
            bpp_avg, std_bpp, loss_avg = validate(model, valid_loader, device)
            if rank == 0:
                print(f"Validation: Loss {loss_avg:.4f} | BPP {bpp_avg:.4f} ± {std_bpp:.4f}")
                if writer:
                    writer.add_scalar('val/loss', loss_avg, epoch)
                    writer.add_scalar('val/bpp', bpp_avg, epoch)
                    writer.add_scalar('val/bpp_std', std_bpp, epoch)
            if bpp_avg < best_bpp:
                best_bpp = bpp_avg
                if rank == 0:
                    torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))

        if scheduler is not None:
            scheduler.step()

        if rank == 0 and epoch % config['checkpoint']['save_interval'] == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, os.path.join(checkpoint_dir, f'epoch_{epoch}.pth'))

    if rank == 0:
        if writer:
            writer.close()
        print("Training finished.")


if __name__ == '__main__':
    main()
