#!/usr/bin/env python3
"""
Training script for Scale Hyperprior image compression model.

Usage:
    python scripts/train.py --config configs/hyperprior_mse.yaml
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
                    epoch, log_freq, writer, clip_max_norm):
    model.train()
    total_loss = 0
    total_bpp = 0
    total_mse = 0
    steps = len(loader)

    for i, x in enumerate(loader):
        x = x.to(device)
        num_pixels = x.shape[2] * x.shape[3] * x.shape[0]  # batch * H * W

        optimizer.zero_grad()

        if scaler is not None:
            with autocast():
                out = model(x)
                loss, bpp, mse = rate_distortion_loss(
                    x, out['x_hat'], out['likelihoods'],
                    num_pixels, lmbda, 'mse'
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(x)
            loss, bpp, mse = rate_distortion_loss(
                x, out['x_hat'], out['likelihoods'],
                num_pixels, lmbda, 'mse'
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
            optimizer.step()

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
def validate(model, loader, device):
    model.eval()
    total_psnr = 0
    total_ssim = 0
    total_bpp = 0
    count = 0
    for x in loader:
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
    return total_psnr / count, total_ssim / count, total_bpp / count

# ==================== Main ====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config yaml')
    parser.add_argument('--local_rank', type=int, default=-1, help='For distributed training')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

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
    train_dataset = ImageFolderDataset(
        root=config['data']['train'],
        patch_size=config['data']['patch_size']
    )
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

    valid_dataset = ImageFolderDataset(
        root=config['data']['valid'],
        patch_size=None
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
    model = HyperpriorModel(N=config['model']['N'], M=config['model']['M'])
    model = model.to(device)
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank
        )

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['train']['lr'])

    # Learning rate scheduler
    if 'lr_milestones' in config['train']:
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=config['train']['lr_milestones'],
            gamma=config['train'].get('lr_gamma', 0.1)
        )
    else:
        scheduler = None

    # Mixed precision
    scaler = GradScaler() if torch.cuda.is_available() else None

    best_psnr = 0

    for epoch in range(1, config['train']['epochs'] + 1):
        if distributed:
            train_sampler.set_epoch(epoch)

        loss, bpp, mse = train_one_epoch(
            model, train_loader, optimizer, scaler,
            lmbda=config['train']['lmbda'],
            device=device,
            epoch=epoch,
            log_freq=config['train']['log_freq'],
            writer=writer,
            clip_max_norm=config['train']['clip_max_norm']
        )

        if rank == 0:
            print(f"Epoch {epoch} finished. Avg Loss: {loss:.4f}, Bpp: {bpp:.4f}, MSE: {mse:.6f}")

        # Validation
        if epoch % config['eval']['interval'] == 0:
            psnr_avg, ssim_avg, bpp_avg = validate(model, valid_loader, device)
            if rank == 0:
                print(f"Validation: PSNR {psnr_avg:.2f} dB, SSIM {ssim_avg:.4f}, Bpp {bpp_avg:.4f}")
                if writer:
                    writer.add_scalar('val/psnr', psnr_avg, epoch)
                    writer.add_scalar('val/ssim', ssim_avg, epoch)
                    writer.add_scalar('val/bpp', bpp_avg, epoch)

                # Save best model
                if psnr_avg > best_psnr:
                    best_psnr = psnr_avg
                    torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))

            # Test datasets
            for name, loader in test_loaders.items():
                psnr_test, ssim_test, bpp_test = validate(model, loader, device)
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