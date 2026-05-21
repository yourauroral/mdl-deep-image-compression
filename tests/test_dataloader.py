import os
import sys

import pytest
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)


def test_dataloader_smoke():
  config_path = os.path.join(project_root, "configs/igpt_cifar10_s_rgb.yaml")
  with open(config_path, "r") as f:
    config = yaml.safe_load(f)

  dataset_name = config["data"].get("dataset", "cifar100")
  DatasetClass = CIFAR10 if dataset_name == "cifar10" else CIFAR100
  data_root = config["data"]["train"]
  if not os.path.isdir(os.path.join(data_root, "cifar-10-batches-py")) and \
     not os.path.isdir(os.path.join(data_root, "cifar-100-python")):
    pytest.skip(f"Dataset {dataset_name} not present at {data_root}")

  train_dataset = DatasetClass(
    root=data_root, train=True, download=False, transform=transforms.ToTensor()
  )
  train_loader = DataLoader(
    train_dataset,
    batch_size=config["train"]["batch_size"],
    shuffle=True,
    num_workers=0,
  )
  x, _ = next(iter(train_loader))
  assert x.dim() == 4 and x.size(1) == 3
  assert 0.0 <= x.min().item() and x.max().item() <= 1.0
