"""
ImageNet-1k parquet (FileZilla 手动上传) → 64×64 npy 流式预处理（磁盘紧张版）。

工作流（70GB 磁盘约束）：
  1. FileZilla 上传一批 parquet 到 {raw_dir}/（一次几个）
  2. 跑本脚本：扫描 raw_dir，处理所有未处理 parquet，写进 out_dir/{train,val}.npy
     游标记进 {out_dir}/.prepare_state.json；传 --delete_processed 后删除已提交 shard
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
        --out_dir datasets/imagenet64_hf \
        --delete_processed
"""
import argparse
import glob
import hashlib
import io
import json
import os
import sys

import numpy as np
from PIL import Image

try:
    from tqdm import tqdm
except ImportError:
    class tqdm:  # type: ignore[no-redef]
        def __init__(self, iterable=None, **kwargs):
            self.iterable = iterable

        def __iter__(self):
            return iter(self.iterable)

        def update(self, count):
            del count

        def close(self):
            pass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mdlic.data.manifest import (
    atomic_write_json,
    fsync_directory,
    fsync_file,
    write_imagenet64_manifest,
)
from mdlic.provenance import file_record


IMG_SIZE = 64
N_TRAIN = 1281167
N_VAL = 50000
STATE_FILENAME = ".prepare_state.json"
STATE_SCHEMA = "mdlic-imagenet64-prepare-state-v2"


def decode_and_resize(img_bytes: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(img_bytes)) as img:
        if img.mode != "RGB":
            img = img.convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BOX)
        return np.asarray(img, dtype=np.uint8).copy()


def _empty_state() -> dict:
    return {
        "schema": STATE_SCHEMA,
        "train": {"cursor": 0, "shards": []},
        "val": {"cursor": 0, "shards": []},
    }


def _validate_state(state: dict) -> None:
    if not isinstance(state, dict) or state.get("schema") != STATE_SCHEMA:
        raise ValueError(f"prepare state schema must be {STATE_SCHEMA!r}")
    for split in ("train", "val"):
        sub = state.get(split)
        if not isinstance(sub, dict):
            raise ValueError(f"prepare state missing split {split!r}")
        cursor = sub.get("cursor")
        shards = sub.get("shards")
        if not isinstance(cursor, int) or isinstance(cursor, bool) or cursor < 0:
            raise ValueError(f"prepare state {split}.cursor must be a non-negative integer")
        if not isinstance(shards, list):
            raise ValueError(f"prepare state {split}.shards must be a list")
        names = [record.get("name") for record in shards if isinstance(record, dict)]
        if len(names) != len(shards) or any(not name for name in names):
            raise ValueError(f"prepare state {split}.shards contains an invalid record")
        if len(names) != len(set(names)):
            raise ValueError(f"prepare state {split}.shards contains duplicate names")


def load_state(state_path: str) -> dict:
    if os.path.isfile(state_path):
        with open(state_path) as f:
            state = json.load(f)
        if "schema" not in state:
            # Preserve old cursors and names. Earlier slices cannot be audited
            # retroactively, so the final manifest marks them as legacy.
            state = {
                "schema": STATE_SCHEMA,
                **{
                    split: {
                        "cursor": state.get(split, {}).get("cursor", 0),
                        "shards": [
                            {"name": name, "legacy_unverified": True}
                            for name in state.get(split, {}).get("done", [])
                        ],
                    }
                    for split in ("train", "val")
                },
            }
        _validate_state(state)
        return state
    return _empty_state()


def save_state(state_path: str, state: dict) -> None:
    _validate_state(state)
    atomic_write_json(state_path, state)


def open_or_create_memmap(path: str, n_total: int, *, cursor: int = 0) -> np.memmap:
    if os.path.isfile(path):
        out = np.lib.format.open_memmap(path, mode="r+")
        expected_shape = (n_total, IMG_SIZE, IMG_SIZE, 3)
        if tuple(out.shape) != expected_shape or out.dtype != np.uint8:
            actual = (tuple(out.shape), str(out.dtype))
            del out
            raise RuntimeError(
                f"existing output {path} has shape/dtype {actual}, "
                f"expected {expected_shape}/uint8"
            )
        return out
    if cursor:
        raise RuntimeError(
            f"prepare state cursor is {cursor}, but output file is missing: {path}"
        )
    out = np.lib.format.open_memmap(
        path, mode="w+", dtype=np.uint8,
        shape=(n_total, IMG_SIZE, IMG_SIZE, 3),
    )
    out.flush()
    fsync_file(path)
    return out


def _slice_sha256(out: np.memmap, start: int, end: int,
                  *, rows_per_chunk: int = 256) -> str:
    digest = hashlib.sha256()
    for offset in range(start, end, rows_per_chunk):
        chunk = np.ascontiguousarray(out[offset:min(offset + rows_per_chunk, end)])
        digest.update(chunk.tobytes())
    return digest.hexdigest()


def _flush_and_fsync_memmap(out: np.memmap, out_path: str) -> None:
    out.flush()
    fsync_file(out_path)


def process_shard(parquet_path: str, out: np.memmap, cursor: int,
                  pbar: tqdm) -> tuple[int, str, int]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "streaming ImageNet64 preparation requires pyarrow; "
            "install the imagenet-prep extra"
        ) from exc

    pf = pq.ParquetFile(parquet_path)
    n_rows = pf.metadata.num_rows
    if cursor + n_rows > out.shape[0]:
        raise RuntimeError(
            f"shard {os.path.basename(parquet_path)} would overflow output: "
            f"cursor={cursor}, n_rows={n_rows}, capacity={out.shape[0]}"
        )
    output_digest = hashlib.sha256()
    for batch in pf.iter_batches(batch_size=1024, columns=["image"]):
        images = batch.column("image").to_pylist()
        for rec in images:
            decoded = decode_and_resize(rec["bytes"])
            out[cursor] = decoded
            output_digest.update(decoded.tobytes())
            cursor += 1
        pbar.update(len(images))
    return cursor, output_digest.hexdigest(), n_rows


def process_split(
    raw_dir: str,
    out_path: str,
    split_name: str,
    n_total: int,
    glob_pattern: str,
    state: dict,
    state_path: str,
    *,
    delete_processed: bool = False,
) -> bool:
    """Transactionally append visible parquet shards to one split."""
    _validate_state(state)
    sub = state[split_name]
    cursor = sub["cursor"]
    if cursor > n_total:
        raise RuntimeError(
            f"[{split_name}] state cursor {cursor} exceeds expected total {n_total}"
        )

    # Validate the output even when state says the split is complete. This
    # prevents a missing/replaced npy from being accepted on resume.
    if cursor:
        existing = open_or_create_memmap(out_path, n_total, cursor=cursor)
        del existing
    if cursor == n_total:
        print(f"[{split_name}] 已完成 ({cursor}/{n_total})")
        return True

    done_set = {record["name"] for record in sub["shards"]}
    pending = sorted(
        path for path in glob.glob(os.path.join(raw_dir, glob_pattern))
        if os.path.basename(path) not in done_set
    )
    if not pending:
        print(f"[{split_name}] 当前无新 parquet，cursor={cursor}/{n_total}")
        return False

    print(f"[{split_name}] cursor={cursor}/{n_total}, 本轮新增 {len(pending)} 个 shard")

    out = open_or_create_memmap(out_path, n_total, cursor=cursor)
    pbar = tqdm(total=n_total, initial=cursor, desc=split_name, unit="img")
    try:
        for path in pending:
            name = os.path.basename(path)
            start = cursor
            input_record = file_record(path)
            end, written_sha256, sample_count = process_shard(
                path, out, start, pbar,
            )
            if end != start + sample_count or end > n_total:
                raise RuntimeError(
                    f"[{split_name}] invalid shard extent for {name}: "
                    f"start={start}, end={end}, samples={sample_count}, total={n_total}"
                )

            _flush_and_fsync_memmap(out, out_path)
            readback_sha256 = _slice_sha256(out, start, end)
            if readback_sha256 != written_sha256:
                raise RuntimeError(
                    f"[{split_name}] durable readback hash mismatch for {name}"
                )

            shard_record = {
                "name": name,
                "input_size_bytes": input_record["size_bytes"],
                "input_sha256": input_record["sha256"],
                "start": start,
                "end": end,
                "samples": sample_count,
                "output_rgb_sha256": readback_sha256,
            }
            sub["cursor"] = end
            sub["shards"].append(shard_record)
            try:
                save_state(state_path, state)
            except Exception:
                sub["shards"].pop()
                sub["cursor"] = start
                raise
            cursor = end

            if delete_processed:
                try:
                    os.remove(path)
                    fsync_directory(path)
                except OSError as exc:
                    print(f"WARNING: processed shard could not be deleted: {path}: {exc}")
            if cursor >= n_total:
                break
    finally:
        pbar.close()
        out.flush()
        del out

    if cursor > n_total:
        raise RuntimeError(
            f"[{split_name}] cursor {cursor} exceeds expected total {n_total}"
        )
    if cursor == n_total:
        print(f"[{split_name}] DONE: {n_total} rows written to {out_path}")
        return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--train_count", type=int, default=N_TRAIN)
    ap.add_argument("--val_count", type=int, default=N_VAL)
    ap.add_argument(
        "--delete_processed", action="store_true",
        help="delete each parquet only after output fsync, readback verification, and state commit",
    )
    args = ap.parse_args()
    if args.train_count < 1 or args.val_count < 1:
        ap.error("--train_count and --val_count must be positive")

    os.makedirs(args.raw_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    state_path = os.path.join(args.out_dir, STATE_FILENAME)
    state = load_state(state_path)

    val_done = process_split(
        args.raw_dir, os.path.join(args.out_dir, "val.npy"),
        "val", args.val_count, "val-*.parquet", state, state_path,
        delete_processed=args.delete_processed,
    )
    train_done = process_split(
        args.raw_dir, os.path.join(args.out_dir, "train.npy"),
        "train", args.train_count, "train-*.parquet", state, state_path,
        delete_processed=args.delete_processed,
    )

    if val_done and train_done:
        manifest = write_imagenet64_manifest(
            args.out_dir,
            producer="scripts/prepare_imagenet64_streaming.py",
            preprocessing={
                "decode": "Pillow Image.open from parquet image.bytes",
                "color": "convert to RGB when needed",
                "resize": "Pillow Resampling.BOX to 64x64",
                "output": "uint8 HWC npy",
            },
            sources={
                split: state[split]["shards"]
                for split in ("train", "val")
            },
        )
        print(f"[manifest] sha256:{manifest['fingerprint_sha256']}")
        print("\nAll done. State may be retained as the per-shard audit log.")
    else:
        print("\n本轮处理结束。FileZilla 继续上传剩余 parquet 后重跑本脚本即可断点续传。")


if __name__ == "__main__":
    main()
