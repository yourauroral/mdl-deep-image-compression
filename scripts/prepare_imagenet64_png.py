"""
PNG flat dir → 64×64 npy 一次性预处理。

输入约定 (PNG 已为 64×64 RGB uint8, 文件名 NNNNNNN.png 序号无要求):
    {train_dir}/*.png
    {val_dir}/*.png

输出:
    {out_dir}/train.npy   uint8, shape (N_train, 64, 64, 3)
    {out_dir}/val.npy     uint8, shape (N_val,   64, 64, 3)
    {out_dir}/dataset_manifest.json  实际 shape/hash/工具版本

并行 decode (multiprocessing.Pool, 默认 os.cpu_count())，写入 np.memmap
避免 15GB train 数组全驻内存。

实际样本数由 glob 动态发现 (HF imagenet-1k 解码后 train ≈ 1281149,
val ≈ 49999，比标称少几张，硬编码会 mismatch)。

用法:
    python scripts/prepare_imagenet64_png.py \
        --train_dir experiments/train_64x64 \
        --val_dir   experiments/valid_64x64 \
        --out_dir   datasets/imagenet64_png \
        --workers 16
"""
import argparse
import glob
import hashlib
import os
import sys
from multiprocessing import Pool

import numpy as np
from PIL import Image

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        del kwargs
        return iterable

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mdlic.data.manifest import fsync_file, write_imagenet64_manifest


IMG_SIZE = 64


def _decode_one(path: str) -> np.ndarray:
    with Image.open(path) as im:
        if im.mode != "RGB":
            im = im.convert("RGB")
        if im.size != (IMG_SIZE, IMG_SIZE):
            im = im.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BOX)
        return np.asarray(im, dtype=np.uint8).copy()


def process_split(src_dir: str, out_path: str, split_name: str, workers: int) -> dict:
    files = sorted(glob.glob(os.path.join(src_dir, "*.png")))
    if not files:
        raise FileNotFoundError(f"no *.png in {src_dir}")
    n = len(files)
    print(f"[{split_name}] {n} PNGs from {src_dir} -> {out_path}")

    out = np.lib.format.open_memmap(
        out_path, mode="w+", dtype=np.uint8,
        shape=(n, IMG_SIZE, IMG_SIZE, 3),
    )

    chunksize = max(1, n // (workers * 32))
    with Pool(processes=workers) as pool:
        for i, arr in enumerate(tqdm(
            pool.imap(_decode_one, files, chunksize=chunksize),
            total=n, desc=split_name, unit="img",
        )):
            out[i] = arr

    out.flush()
    del out
    fsync_file(out_path)
    size_gb = os.path.getsize(out_path) / 1e9
    print(f"[{split_name}] saved ({n}, {IMG_SIZE}, {IMG_SIZE}, 3) uint8 "
          f"-> {out_path} ({size_gb:.2f} GB)")
    order_digest = hashlib.sha256()
    for path in files:
        order_digest.update(os.path.basename(path).encode("utf-8"))
        order_digest.update(b"\0")
        order_digest.update(str(os.path.getsize(path)).encode("ascii"))
        order_digest.update(b"\n")
    return {
        "glob": "*.png sorted lexicographically",
        "files": n,
        "ordered_names_and_sizes_sha256": order_digest.hexdigest(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", required=True, help="目录含 *.png (训练集)")
    ap.add_argument("--val_dir", required=True, help="目录含 *.png (验证集)")
    ap.add_argument("--out_dir", required=True, help="输出 train.npy / val.npy 的目录")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 8)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    val_source = process_split(
        args.val_dir, os.path.join(args.out_dir, "val.npy"),
        "val", args.workers,
    )
    train_source = process_split(
        args.train_dir, os.path.join(args.out_dir, "train.npy"),
        "train", args.workers,
    )
    manifest = write_imagenet64_manifest(
        args.out_dir,
        producer="scripts/prepare_imagenet64_png.py",
        preprocessing={
            "decode": "Pillow Image.open",
            "color": "convert to RGB when needed",
            "resize": "Pillow Resampling.BOX to 64x64 when needed",
            "output": "uint8 HWC npy",
        },
        sources={"train": train_source, "val": val_source},
    )
    print(f"[manifest] sha256:{manifest['fingerprint_sha256']}")


if __name__ == "__main__":
    main()
