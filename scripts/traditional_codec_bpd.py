#!/usr/bin/env python3
"""Compute PNG/WebP lossless bits-per-dimension for an ImageNet64 .npy file."""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mdlic.data.manifest import atomic_write_json, npy_split_record
from mdlic.traditional_codecs import codec_metadata, encode_rgb_array


def compute_bpd(path: Path, limit: int | None) -> dict:
    data = np.load(path, mmap_mode="r")
    if data.ndim != 4 or data.shape[1:] != (64, 64, 3) or data.dtype != np.uint8:
        raise ValueError(
            f"expected uint8 array with shape (N, 64, 64, 3), got "
            f"{data.shape} {data.dtype}"
        )

    n = data.shape[0] if limit is None else min(limit, data.shape[0])
    if n <= 0:
        raise ValueError("dataset and --limit must select at least one image")
    dims = data.shape[1] * data.shape[2] * data.shape[3]
    bpd_values = {method: [] for method in ("png", "webp")}
    compressed_bytes = {method: 0 for method in bpd_values}

    for i in range(n):
        image = np.asarray(data[i], dtype=np.uint8)
        for method in bpd_values:
            encoded_size = len(encode_rgb_array(image, method))
            compressed_bytes[method] += encoded_size
            bpd_values[method].append(encoded_size * 8 / dims)

    results = {}
    for method, values in bpd_values.items():
        array = np.asarray(values, dtype=np.float64)
        results[method] = {
            **codec_metadata(method),
            "sample_count": n,
            "mean_bpd": float(array.mean()),
            "std_per_image": float(array.std(ddof=1)) if n > 1 else 0.0,
            "compressed_bytes_total": compressed_bytes[method],
        }

    split = npy_split_record(path)
    return {
        "schema_version": 1,
        "protocol": "traditional_lossless_codec_bpd",
        "dataset": {
            "name": "ImageNet64",
            "split": "val" if path.name == "val.npy" else "unknown",
            "selection": "prefix",
            "sample_count": n,
            "total_samples": int(data.shape[0]),
            "path": str(path.resolve()),
            "file": {
                key: split[key]
                for key in ("name", "size_bytes", "sha256", "samples", "shape", "dtype")
            },
        },
        "rate_accounting": {
            "metric": "complete_encoded_file_bpd",
            "denominator": "H*W*C per image",
            "includes_file_headers": True,
            "selection": "first N samples in dataset order",
        },
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute PNG/WebP lossless bpd for ImageNet64 npy data."
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Path to val.npy/train.npy with shape (N, 64, 64, 3).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=2000,
        help="Maximum number of images to sample; use 0 for the full file.",
    )
    parser.add_argument(
        "--result_json",
        type=Path,
        default=None,
        help="Write a reproducible JSON manifest for this measurement.",
    )
    args = parser.parse_args()

    limit = None if args.limit == 0 else args.limit
    manifest = compute_bpd(args.path, limit)
    for method in ("png", "webp"):
        result = manifest["results"][method]
        print(f"{result['display_name']} bpd = {result['mean_bpd']:.3f}")
        print(
            f"  std/image = {result['std_per_image']:.4f}; "
            f"compressed bytes = {result['compressed_bytes_total']}"
        )
        print(
            f"  Pillow {result['pillow_version']}, "
            f"{result['backend']} {result['backend_version']}, "
            f"settings={result['save_kwargs']}"
        )
    if args.result_json:
        args.result_json.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(args.result_json, manifest)
        print(f"[result_json] wrote manifest -> {args.result_json}")


if __name__ == "__main__":
    main()
