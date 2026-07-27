"""ImageNet64 npy-backed Dataset (预处理后的 uint8 HWC 数组)。"""
import os

import numpy as np
import torch
from torch.utils.data import Dataset


class ImageNet64Npy(Dataset):
    def __init__(self, root: str, split: str, hflip: bool = False):
        path = os.path.join(root, f"{split}.npy")
        self.data = np.load(path, mmap_mode="r")
        manifest_path = os.path.join(root, "dataset_manifest.json")
        self.manifest_path = manifest_path if os.path.isfile(manifest_path) else None
        assert self.data.ndim == 4 and self.data.shape[1:] == (64, 64, 3), \
            f"unexpected shape {self.data.shape}, expected (N, 64, 64, 3)"
        assert self.data.dtype == np.uint8
        self.hflip = hflip

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: int):
        img = np.array(self.data[idx], dtype=np.uint8)
        # torch.rand 复用 DataLoader worker_init_fn 设的种子，多 worker 安全
        if self.hflip and torch.rand(1).item() < 0.5:
            img = img[:, ::-1, :].copy()
        tensor = torch.from_numpy(img).permute(2, 0, 1).contiguous().float().div_(255.0)
        return tensor, 0
