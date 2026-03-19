#!/usr/bin/env python3
"""
Training script for Scale Hyperprior image compression model.

Usage:
    python scripts/train.py --config configs/igpt_small.yaml
"""

import os
import sys
import argparse
import yaml
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image
from torchvision import transforms

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import model and metrics
from src.mdlic import HyperpriorModel
from src.mdlic.utils.metrics import psnr, compute_ssim, compute_bpp


# ==================== Dataset ====================
class ImageFolderDataset(Dataset):
    """Read images from folder, optionally random crop."""
    def __init__(self, root, patch_size=None, transform=None):
        self.filenames = [os.path.join(root, f) for f in os.listdir(root)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        self.patch_size = patch_size
        self.transform = transform or transforms.ToTensor()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img = Image.open(self.filenames[idx]).convert('RGB')
        if self.patch_size is not None:
            w, h = img.size
            # Ensure image is at least patch_size
            if w < self.patch_size or h < self.patch_size:
                scale = self.patch_size / min(w, h)
                new_w, new_h = int(w * scale), int(h * scale)
                img = img.resize((new_w, new_h), Image.BICUBIC)
                w, h = new_w, new_h
            i = np.random.randint(0, w - self.patch_size + 1)
            j = np.random.randint(0, h - self.patch_size + 1)
            img = img.crop((i, j, i + self.patch_size, j + self.patch_size))
        x = self.transform(img)
        return x

# ==================== Loss ====================
def rate_distortion_loss(x, x_hat, likelihoods, num_pixels, lmbda, distortion_type='mse'):
    total_bits = 0
    for key in likelihoods:
        lk = likelihoods[key]
        lk = torch.clamp(lk, min=1e-10)
        total_bits += -torch.log2(lk).sum()
    bpp = total_bits / num_pixels

    if distortion_type == 'mse':
        distortion = F.mse_loss(x, x_hat, reduction='mean')
    elif distortion_type == 'ms-ssim':
        try:
            from pytorch_msssim import ms_ssim
            distortion = 1 - ms_ssim(x, x_hat, data_range=1.0)
        except ImportError:
            raise ImportError("pytorch_msssim not installed. Use 'mse' or install it.")
    else:
        raise ValueError(f"Unknown distortion type: {distortion_type}")

    loss = bpp + lmbda * distortion
    return loss, bpp, distortion

# ==================== Training ====================
def train_one_epoch(model, loader, optimizer, scaler, lmbda, device,
                    epoch, log_freq, writer, clip_max_norm, model_type,
                    amp_dtype=torch.float16, grad_accum_steps=1,
                    z_loss_weight=1e-4, mtp_weight=0.1):
    model.train()
    total_loss = 0
    total_bpp = 0
    total_mse = 0
    steps = len(loader)

    optimizer.zero_grad(set_to_none=True) 

    for i, batch in enumerate(loader):
        if isinstance(batch, (list, tuple)):
            x = batch[0]
        else:
            x = batch 
        
        x = x.to(device)
        B, C, H, W = x.shape
        num_pixels = B * H * W 

        if scaler is not None:
            with autocast(dtype=amp_dtype):
                out = model(x) if model_type != "igpt" else model(
                    x, z_loss_weight=z_loss_weight, mtp_weight=mtp_weight
                )
                if model_type == "igpt":
                    loss = out["loss"]
                    bpp = (loss / math.log(2)) * C
                    mse = torch.tensor(0.0, device=device)
                else:
                    loss, bpp, mse = rate_distortion_loss(
                        x, out['x_hat'], out['likelihoods'],
                        num_pixels, lmbda, 'mse'
                    )
        else:
            out = model(x) if model_type != "igpt" else model(
                x, z_loss_weight=z_loss_weight, mtp_weight=mtp_weight
            )
            if model_type == "igpt":
                loss = out["loss"]
                bpp = (loss / math.log(2)) * C
                mse = torch.tensor(0.0, device=device)
            else:
                loss, bpp, mse = rate_distortion_loss(
                    x, out['x_hat'], out['likelihoods'],
                    num_pixels, lmbda, 'mse'
                )
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
        total_mse += mse.item()

        if (i + 1) % log_freq == 0:
            print(f"Epoch {epoch} Step {i+1}/{steps} | Loss: {loss.item():.4f} | "
                  f"Bpp: {bpp.item():.4f} | MSE: {mse.item():.6f}")
            if writer:
                step = epoch * steps + i
                writer.add_scalar('train/loss', loss.item(), step)
                writer.add_scalar('train/bpp', bpp.item(), step)
                writer.add_scalar('train/mse', mse.item(), step)

    avg_loss = total_loss / steps
    avg_bpp = total_bpp / steps
    avg_mse = total_mse / steps
    return avg_loss, avg_bpp, avg_mse


# ==================== Validation ====================
@torch.no_grad()
def validate(model, loader, device, model_type="hyperprior"):
    model.eval()
    if model_type == "igpt":
        bpp_list = []
        total_loss = 0
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch
            x = x.to(device)
            out = model(x)
            loss = out["loss"]
            _, C, _, _ = x.shape
            bpp = (loss / math.log(2)) * C
            bpp_list.append(bpp.item())
            total_loss += loss.item()
        avg_bpp = float(np.mean(bpp_list))
        std_bpp = float(np.std(bpp_list))
        avg_loss = total_loss / len(bpp_list)
        return 0.0, 0.0, avg_bpp, avg_loss, std_bpp
    else:
        total_psnr = 0
        total_ssim = 0
        total_bpp = 0
        count = 0
        for x in loader:
            if isinstance(x, (list, tuple)):
                x = x[0]
            x = x.to(device)
            _, _, h, w = x.shape

            # Pad to multiples of 16 (required by model)
            pad_h = (16 - h % 16) % 16
            pad_w = (16 - w % 16) % 16
            if pad_h > 0 or pad_w > 0:
                x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
            else:
                x_pad = x

            out = model(x_pad)

            # Crop back to original size
            if pad_h > 0 or pad_w > 0:
                out['x_hat'] = out['x_hat'][:, :, :h, :w]

            num_pixels = h * w * x.shape[0]  # original pixel count
            bpp = compute_bpp(out['likelihoods'], num_pixels)
            psnr_val = psnr(x, out['x_hat'])
            ssim_val = compute_ssim(x, out['x_hat'])
            total_psnr += psnr_val
            total_ssim += ssim_val
            total_bpp += bpp
            count += 1
        return total_psnr / count, total_ssim / count, total_bpp / count, 0.0, 0.0

# ==================== Main ====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config yaml')
    parser.add_argument('--local_rank', type=int, default=-1, help='For distributed training')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    seed = config["train"].get("seed", 42)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    exp_name = config['exp_name']
    out_dir = os.path.join('experiments', exp_name)
    checkpoint_dir = os.path.join(out_dir, 'checkpoints')
    log_dir = os.path.join(out_dir, 'logs')
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

    if torch.cuda.is_available():
        if distributed:
            device = torch.device(f'cuda:{args.local_rank}')
        else:
            device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    if rank == 0:
        writer = SummaryWriter(log_dir=log_dir)
    else:
        writer = None

    # Datasets
    from torchvision.datasets import CIFAR100
    from torchvision import transforms
    if config["model"]["type"] == "igpt": 
        transform = transforms.ToTensor()
        train_dataset = CIFAR100(
            root=config["data"]["train"],
            train=True,
            download=False,
            transform=transform
        )
        valid_dataset = CIFAR100(
            root=config["data"]["valid"],
            train=False,
            download=False,
            transform=transform
        )
        test_loaders = {} 
    else:
        # Hyperprior / compression models 
        train_dataset = ImageFolderDataset(
            root=config['data']['train'],
            patch_size=config['data']['patch_size']
        )
        valid_dataset = ImageFolderDataset(
            root=config['data']['valid'],
            patch_size=None 
        )
        test_loaders = {} 

    if distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        shuffle = False
    else:
        train_sampler = None
        shuffle = True
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['train']['batch_size'],
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=config['data']['num_workers'],
        pin_memory=True,
        drop_last=True
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )

    # Test datasets (optional)
    test_loaders = {}
    if 'test' in config['data']:
        for name, path in config['data']['test'].items():
            test_dataset = ImageFolderDataset(root=path, patch_size=None)
            test_loaders[name] = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)

    # Model
    from src.mdlic.models.igpt import IGPT 
    if config["model"]["type"] == "hyperprior":
        model = HyperpriorModel(
            N=config['model']['N'], 
            M=config['model']['M']
        )
    elif config["model"]["type"] == "igpt":
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
        )
    model = model.to(device)
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank
        )

    # Optimizer    def get_param_groups(model, weight_decay=0.1):
        """Embeddings 和 bias/norm 不加 weight decay"""
        no_decay = set()
        for name, _ in model.named_parameters():
            if any(nd in name for nd in ['token_embed', 'norm', 'bias']):
                no_decay.add(name)

        params_decay = [p for n, p in model.named_parameters() if n not in no_decay]
        params_no_decay = [p for n, p in model.named_parameters() if n in no_decay]

        return [
            {"params": params_decay,    "weight_decay": weight_decay},
            {"params": params_no_decay, "weight_decay": 0.0},
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
    #         采用 warmup + cosine cooldown 调度，强调 cooldown 阶段对最终
    #         模型质量的决定性影响。
    #     [2] Loshchilov & Hutter, "SGDR: Stochastic Gradient Descent with
    #         Warm Restarts," arXiv:1608.03983, 2016. Cosine annealing 原始论文。
    #     [3] CS336 "Language Models from Scratch," Stanford, Spring 2024.
    #         推荐 cosine decay + warmup 作为 transformer 训练的默认调度。
    #
    # 相比 MultiStepLR 的优势：
    #   - 无需手动调 milestone，LR 平滑下降，训练末期细粒度更新；
    #   - warmup 阶段防止大 LR 在训练初期破坏随机初始化的参数。
    #
    # 保留 MultiStepLR 作为 fallback，通过 config lr_schedule 字段切换。
    lr_schedule = config["train"].get("lr_schedule", "cosine")
    if lr_schedule == "cosine":
        warmup_epochs = config["train"].get("warmup_epochs", 5)
        total_epochs  = config["train"]["epochs"]
        def lr_lambda(epoch):
            # 线性 warmup: epoch 0 → warmup_epochs
            if epoch < warmup_epochs:
                return float(epoch + 1) / float(max(1, warmup_epochs))
            # Cosine decay: warmup_epochs → total_epochs  [2]
            progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    elif lr_schedule == "multistep":
        # 保留旧行为，通过 lr_milestones / lr_gamma 配置
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=config["train"]["lr_milestones"],
            gamma=config["train"].get("lr_gamma", 0.1)
        )
    else:
        scheduler = None

    # Mixed precision
    amp_dtype = torch.float16 if config["train"].get("amp_dtype", "fp16") == "fp16" else torch.bfloat16
    scaler = GradScaler() if amp_dtype == torch.float16 else None  
    grad_accum_steps = config["train"].get("grad_accum_steps", 1)

    best_psnr = 0.0
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

        loss, bpp, mse = train_one_epoch(
            model, train_loader, optimizer, scaler,
            lmbda=config["train"]["lmbda"],
            device=device,
            epoch=epoch,
            log_freq=config['train']['log_freq'],
            writer=writer,
            clip_max_norm=config['train']['clip_max_norm'],
            model_type=config["model"]["type"],
            amp_dtype=amp_dtype,
            grad_accum_steps=grad_accum_steps,
            z_loss_weight=config["train"].get("z_loss_weight", 1e-4),
            mtp_weight=config["train"].get("mtp_weight", 0.1),
        )

        if rank == 0:
            print(f"Epoch {epoch} finished. Avg Loss: {loss:.4f}, Bpp: {bpp:.4f}, MSE: {mse:.6f}")

        # Validation
        if epoch % config['eval']['interval'] == 0:
            model_type = config["model"]["type"]
            if model_type == "igpt":
                # igpt return (psnr = 0.0, ssim = 0.0 , bpp, loss)
                _, _, bpp_avg, loss_avg, std_bpp = validate(model, valid_loader, device, model_type)
                psnr_avg, ssim_avg = 0.0, 0.0
                if rank == 0:
                    print(f"Validation: Loss {loss_avg:.4f}, BPP {bpp_avg:.4f} ± {std_bpp:.4f}")
                    if writer:
                        writer.add_scalar('val/loss', loss_avg, epoch)
                        writer.add_scalar('val/bpp', bpp_avg, epoch)
                        writer.add_scalar('val/bpp_std', std_bpp, epoch)
            else: 
                psnr_avg, ssim_avg, bpp_avg, _, _ = validate(model, valid_loader, device, model_type)
                if rank == 0:
                    print(f"Validation: PSNR {psnr_avg:.2f} dB, SSIM {ssim_avg:.4f}, Bpp {bpp_avg:.4f}")
                    if writer:
                        writer.add_scalar('val/psnr', psnr_avg, epoch)
                        writer.add_scalar('val/ssim', ssim_avg, epoch)
                        writer.add_scalar('val/bpp', bpp_avg, epoch)

            # Save best model
            if model_type == "igpt":
                if bpp_avg < best_bpp:
                    best_bpp = bpp_avg
                    torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best_igpt.pth'))
            else:
                if psnr_avg > best_psnr:
                    best_psnr = psnr_avg
                    torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))     

            # Test Datasets
            if model_type != "igpt":
                for name, loader in test_loaders.items():
                    psnr_test, ssim_test, bpp_test, _, _ = validate(model, loader, device, model_type)
                    if rank == 0:
                        print(f"Test {name}: PSNR {psnr_test:.2f} dB, SSIM {ssim_test:.4f}, Bpp {bpp_test:.4f}")
                        if writer:
                            writer.add_scalar(f'test/{name}_psnr', psnr_test, epoch)
                            writer.add_scalar(f'test/{name}_ssim', ssim_test, epoch)
                            writer.add_scalar(f'test/{name}_bpp', bpp_test, epoch)

        # Step scheduler
        if scheduler is not None:
            scheduler.step()

        # Save checkpoint
        if rank == 0 and epoch % config['checkpoint']['save_interval'] == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, os.path.join(checkpoint_dir, f'epoch_{epoch}.pth'))

    if rank == 0:
        writer.close()
        print("Training finished.")


if __name__ == '__main__':
    main()