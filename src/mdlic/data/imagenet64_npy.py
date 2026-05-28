"""ImageNet64 npy-backed Dataset (预处理后的 uint8 HWC 数组)。"""
import os

import numpy as np
import torch
from torch.utils.data import Dataset


class ImageNet64Npy(Dataset):
    def __init__(self, root: str, split: str):
        path = os.path.join(root, f"{split}.npy")
        self.data = np.load(path, mmap_mode="r")
        assert self.data.ndim == 4 and self.data.shape[1:] == (64, 64, 3), \
            f"unexpected shape {self.data.shape}, expected (N, 64, 64, 3)"
        assert self.data.dtype == np.uint8

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: int):
        img = np.array(self.data[idx], dtype=np.uint8)
        tensor = torch.from_numpy(img).permute(2, 0, 1).contiguous().float().div_(255.0)
        return tensor, 0
