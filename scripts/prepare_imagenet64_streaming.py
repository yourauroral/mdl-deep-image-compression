"""
ImageNet-1k parquet (FileZilla 手动上传) → 64×64 npy 流式预处理（磁盘紧张版）。

工作流（70GB 磁盘约束）：
  1. FileZilla 上传一批 parquet 到 {raw_dir}/（一次几个）
  2. 跑本脚本：扫描 raw_dir，处理所有未处理 parquet，写进 out_dir/{train,val}.npy
     游标记进 {out_dir}/.prepare_state.json，已处理 parquet 自动删除（节省磁盘）
  3. FileZilla 上传下一批，再跑本脚本，断点续传
  4. 总行数 cursor 凑够 N_TRAIN / N_VAL 自动结束

输出:
    {out_dir}/train.npy   uint8, shape (N_TRAIN, 64, 64, 3) ~15.0 GB
    {out_dir}/val.npy     uint8, shape (N_VAL,   64, 64, 3) ~0.6 GB

约定: parquet 命名匹配 train-*.parquet / val-*.parquet (HF imagenet-1k 标准命名)
shard 数量由 glob 动态发现，不硬编码。

依赖: pyarrow pillow tqdm numpy

用法:
    python scripts/prepare_imagenet64_streaming.py \
        --raw_dir datasets/imagenet64_hf/raw \
        --out_dir datasets/imagenet64_hf
"""
import argparse
import glob
import io
import json
import os

import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm


IMG_SIZE = 64
N_TRAIN = 1281167
N_VAL = 50000
STATE_FILENAME = ".prepare_state.json"


def decode_and_resize(img_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(img_bytes))
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BOX)
    return np.asarray(img, dtype=np.uint8)


def load_state(state_path: str) -> dict:
    if os.path.isfile(state_path):
        with open(state_path) as f:
            return json.load(f)
    return {"train": {"cursor": 0, "done": []}, "val": {"cursor": 0, "done": []}}


def save_state(state_path: str, state: dict) -> None:
    tmp = state_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, state_path)


def open_or_create_memmap(path: str, n_total: int) -> np.memmap:
    if os.path.isfile(path):
        return np.lib.format.open_memmap(path, mode="r+")
    return np.lib.format.open_memmap(
        path, mode="w+", dtype=np.uint8,
        shape=(n_total, IMG_SIZE, IMG_SIZE, 3),
    )


def process_shard(parquet_path: str, out: np.memmap, cursor: int, pbar: tqdm) -> int:
    pf = pq.ParquetFile(parquet_path)
    n_rows = pf.metadata.num_rows
    if cursor + n_rows > out.shape[0]:
        raise RuntimeError(
            f"shard {os.path.basename(parquet_path)} would overflow output: "
            f"cursor={cursor}, n_rows={n_rows}, capacity={out.shape[0]}"
        )
    for batch in pf.iter_batches(batch_size=1024, columns=["image"]):
        images = batch.column("image").to_pylist()
        for rec in images:
            out[cursor] = decode_and_resize(rec["bytes"])
            cursor += 1
        pbar.update(len(images))
    return cursor


def process_split(raw_dir: str, out_path: str, split_name: str, n_total: int,
                  glob_pattern: str, state: dict, state_path: str) -> bool:
    """处理 split 中所有当前可见且未处理的 parquet。返回 cursor 是否凑齐 n_total。"""
    sub = state[split_name]
    done_set = set(sub["done"])
    cursor = sub["cursor"]

    if cursor >= n_total:
        print(f"[{split_name}] 已完成 ({cursor}/{n_total})")
        return True

    pending = sorted(
        p for p in glob.glob(os.path.join(raw_dir, glob_pattern))
        if os.path.basename(p) not in done_set
    )
    if not pending:
        print(f"[{split_name}] 当前无新 parquet，cursor={cursor}/{n_total}")
        return False

    print(f"[{split_name}] cursor={cursor}/{n_total}, 本轮新增 {len(pending)} 个 shard")

    out = open_or_create_memmap(out_path, n_total)
    pbar = tqdm(total=n_total, initial=cursor, desc=split_name, unit="img")
    try:
        for path in pending:
            name = os.path.basename(path)
            cursor = process_shard(path, out, cursor, pbar)
            sub["cursor"] = cursor
            sub["done"].append(name)
            save_state(state_path, state)
            try:
                os.remove(path)
            except OSError:
                pass
            if cursor >= n_total:
                break
    finally:
        pbar.close()
        out.flush()
        del out

    if cursor >= n_total:
        if cursor != n_total:
            raise RuntimeError(
                f"[{split_name}] cursor {cursor} 超过期望 {n_total}"
            )
        print(f"[{split_name}] DONE: {n_total} 行写入 {out_path}")
        return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.raw_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    state_path = os.path.join(args.out_dir, STATE_FILENAME)
    state = load_state(state_path)

    val_done = process_split(
        args.raw_dir, os.path.join(args.out_dir, "val.npy"),
        "val", N_VAL, "val-*.parquet", state, state_path,
    )
    train_done = process_split(
        args.raw_dir, os.path.join(args.out_dir, "train.npy"),
        "train", N_TRAIN, "train-*.parquet", state, state_path,
    )

    if val_done and train_done:
        print("\nAll done. 可删除 raw_dir 与 .prepare_state.json 释放磁盘。")
    else:
        print("\n本轮处理结束。FileZilla 继续上传剩余 parquet 后重跑本脚本即可断点续传。")


if __name__ == "__main__":
    main()
