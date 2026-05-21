"""
HuggingFace imagenet-1k parquet → 64×64 npy 一次性预处理。

输入约定:
    {raw_dir}/train-0000{0..6}-of-00007.parquet
    {raw_dir}/val-00000-of-00001.parquet

输出:
    {out_dir}/train.npy   uint8, shape (N_train, 64, 64, 3)
    {out_dir}/val.npy     uint8, shape (N_val,   64, 64, 3)

Resize: PIL.Image.BOX (box filter)，对齐 Chrabaszcz 2017 ImageNet64 协议。

用法:
    python scripts/prepare_imagenet64.py \
        --raw_dir datasets/imagenet64_hf/raw \
        --out_dir datasets/imagenet64_hf
"""
import argparse
import glob
import io
import os

import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm


IMG_SIZE = 64


def decode_and_resize(img_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(img_bytes))
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BOX)
    return np.asarray(img, dtype=np.uint8)


def process_split(parquet_paths: list[str], out_path: str, split_name: str) -> None:
    total_rows = 0
    for p in parquet_paths:
        total_rows += pq.ParquetFile(p).metadata.num_rows
    print(f"[{split_name}] {len(parquet_paths)} shards, {total_rows} rows -> {out_path}")

    out = np.empty((total_rows, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    cursor = 0
    pbar = tqdm(total=total_rows, desc=split_name, unit="img")
    for path in parquet_paths:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=1024, columns=["image"]):
            images = batch.column("image").to_pylist()
            for rec in images:
                out[cursor] = decode_and_resize(rec["bytes"])
                cursor += 1
            pbar.update(len(images))
    pbar.close()
    assert cursor == total_rows, f"row count mismatch: {cursor} vs {total_rows}"
    np.save(out_path, out)
    print(f"[{split_name}] saved {out.shape} {out.dtype} -> {out_path} ({os.path.getsize(out_path) / 1e9:.2f} GB)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True, help="目录含 train-*.parquet 与 val-*.parquet")
    ap.add_argument("--out_dir", required=True, help="输出 train.npy / val.npy 的目录")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    train_files = sorted(glob.glob(os.path.join(args.raw_dir, "train-*.parquet")))
    val_files = sorted(glob.glob(os.path.join(args.raw_dir, "val-*.parquet")))
    if not train_files:
        raise FileNotFoundError(f"no train-*.parquet in {args.raw_dir}")
    if not val_files:
        raise FileNotFoundError(f"no val-*.parquet in {args.raw_dir}")

    process_split(train_files, os.path.join(args.out_dir, "train.npy"), "train")
    process_split(val_files, os.path.join(args.out_dir, "val.npy"), "val")


if __name__ == "__main__":
    main()
