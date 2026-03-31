import sys
import os
import yaml
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100

# 将项目根目录添加到python路径下（脚本在tests/下，根目录在上一级）
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src.mdlic.models.igpt import rgb_to_ycbcr_int

if __name__ == "__main__":
  config_path = os.path.join(project_root, "configs/igpt_cifar100_baseline.yaml")
  print(f"Using config file: {config_path}")

  with open(config_path, "r") as f:
    config = yaml.safe_load(f)

  dataset_name = config["data"].get("dataset", "cifar100")
  DatasetClass = CIFAR10 if dataset_name == "cifar10" else CIFAR100
  transform = transforms.ToTensor()

  train_dataset = DatasetClass(
    root=config["data"]["train"], train=True, download=False, transform=transform
  )
  print(f"Dataset: {dataset_name} | Number of training samples: {len(train_dataset)}")

  train_loader = DataLoader(
    train_dataset,
    batch_size=config["train"]["batch_size"],
    shuffle=True,
    num_workers=config["data"]["num_workers"],
  )

  for i, (x, _) in enumerate(train_loader):
    print(f"Batch {i}: image tensor shape: {x.shape}")
    print(f"  RGB  min: {x.min():.4f}, max: {x.max():.4f}")
    ycbcr = rgb_to_ycbcr_int(x)
    print(f"  YCbCr shape: {ycbcr.shape}, dtype: {ycbcr.dtype}")
    print(f"  YCbCr min: {ycbcr.min().item()}, max: {ycbcr.max().item()}")
    break
