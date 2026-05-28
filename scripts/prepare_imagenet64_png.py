"""
PNG flat dir → 64×64 npy 一次性预处理。

输入约定 (PNG 已为 64×64 RGB uint8, 文件名 NNNNNNN.png 序号无要求):
    {train_dir}/*.png
    {val_dir}/*.png

输出:
    {out_dir}/train.npy   uint8, shape (N_train, 64, 64, 3)
    {out_dir}/val.npy     uint8, shape (N_val,   64, 64, 3)

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
import os
from multiprocessing import Pool

import numpy as np
from PIL import Image
from tqdm import tqdm


IMG_SIZE = 64


def _decode_one(path: str) -> np.ndarray:
    im = Image.open(path)
    if im.mode != "RGB":
        im = im.convert("RGB")
    if im.size != (IMG_SIZE, IMG_SIZE):
        im = im.resize((IMG_SIZE, IMG_SIZE), Image.BOX)
    return np.asarray(im, dtype=np.uint8)


def process_split(src_dir: str, out_path: str, split_name: str, workers: int) -> None:
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
    size_gb = os.path.getsize(out_path) / 1e9
    print(f"[{split_name}] saved ({n}, {IMG_SIZE}, {IMG_SIZE}, 3) uint8 "
          f"-> {out_path} ({size_gb:.2f} GB)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", required=True, help="目录含 *.png (训练集)")
    ap.add_argument("--val_dir", required=True, help="目录含 *.png (验证集)")
    ap.add_argument("--out_dir", required=True, help="输出 train.npy / val.npy 的目录")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 8)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    process_split(args.val_dir, os.path.join(args.out_dir, "val.npy"),
                  "val", args.workers)
    process_split(args.train_dir, os.path.join(args.out_dir, "train.npy"),
                  "train", args.workers)


if __name__ == "__main__":
    main()
