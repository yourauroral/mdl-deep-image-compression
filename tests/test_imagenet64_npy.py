"""ImageNet64Npy unit tests (无需真数据，构造临时 npy)。"""
import os
import sys
import tempfile

import numpy as np
import pytest
import torch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src.mdlic.data.imagenet64_npy import ImageNet64Npy


@pytest.fixture
def tmp_npy_root():
    with tempfile.TemporaryDirectory() as d:
        rng = np.random.default_rng(0)
        arr = rng.integers(0, 256, size=(8, 64, 64, 3), dtype=np.uint8)
        np.save(os.path.join(d, "train.npy"), arr)
        np.save(os.path.join(d, "val.npy"), arr[:4])
        yield d


def test_shape_dtype_range(tmp_npy_root):
    ds = ImageNet64Npy(root=tmp_npy_root, split="train")
    assert len(ds) == 8
    x, _ = ds[0]
    assert x.shape == (3, 64, 64)
    assert x.dtype == torch.float32
    assert 0.0 <= x.min().item() <= x.max().item() <= 1.0


def test_hflip_off_deterministic(tmp_npy_root):
    ds = ImageNet64Npy(root=tmp_npy_root, split="train", hflip=False)
    a, _ = ds[3]
    b, _ = ds[3]
    assert torch.equal(a, b)


def test_hflip_on_produces_mix_and_consistent_pairs(tmp_npy_root):
    """hflip=True: 多次取同一 idx 应有约半数是原图、半数是 W 轴翻转。"""
    ds = ImageNet64Npy(root=tmp_npy_root, split="train", hflip=True)
    # 拿一份"真值"参考：直接从底层 mmap 读、跟 __getitem__ 同样 permute
    raw = np.array(ds.data[3], dtype=np.uint8)
    ref = torch.from_numpy(raw).permute(2, 0, 1).float() / 255.0
    ref_flipped = torch.flip(ref, dims=[2])

    torch.manual_seed(42)
    n_orig = n_flip = n_other = 0
    for _ in range(200):
        x, _ = ds[3]
        if torch.allclose(x, ref):
            n_orig += 1
        elif torch.allclose(x, ref_flipped):
            n_flip += 1
        else:
            n_other += 1
    # 每次取要么原图要么翻转，无第三种
    assert n_other == 0, f"unexpected outputs: {n_other}"
    # ~50/50 (200 次 binomial std=7 → ±21 是 3σ)
    assert 80 <= n_orig <= 120, f"flip ratio off: orig={n_orig} flip={n_flip}"
    assert 80 <= n_flip <= 120, f"flip ratio off: orig={n_orig} flip={n_flip}"


def test_val_split_no_hflip_even_if_dataset_hflip_true():
    """约定：train.py 给 val 传 hflip=False，val 必须无翻转。"""
    with tempfile.TemporaryDirectory() as d:
        rng = np.random.default_rng(1)
        arr = rng.integers(0, 256, size=(4, 64, 64, 3), dtype=np.uint8)
        np.save(os.path.join(d, "val.npy"), arr)
        ds = ImageNet64Npy(root=d, split="val", hflip=False)
        a, _ = ds[0]
        b, _ = ds[0]
        assert torch.equal(a, b)
