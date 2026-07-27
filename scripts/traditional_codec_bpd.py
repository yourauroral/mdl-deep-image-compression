#!/usr/bin/env python3
"""Compute PNG/WebP lossless bits-per-dimension for an ImageNet64 .npy file."""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mdlic.traditional_codecs import codec_metadata, encode_rgb_array


def compute_bpd(path: Path, limit: int | None) -> tuple[float, float]:
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
    png_bits = 0
    webp_bits = 0

    for i in range(n):
        image = np.asarray(data[i], dtype=np.uint8)
        png_bits += len(encode_rgb_array(image, "png")) * 8
        webp_bits += len(encode_rgb_array(image, "webp")) * 8

    return png_bits / n / dims, webp_bits / n / dims


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
    args = parser.parse_args()

    limit = None if args.limit == 0 else args.limit
    png_bpd, webp_bpd = compute_bpd(args.path, limit)
    print(f"PNG  bpd = {png_bpd:.3f}")
    print(f"WebP bpd = {webp_bpd:.3f}")
    for method in ("png", "webp"):
        meta = codec_metadata(method)
        print(
            f"{meta['display_name']}: Pillow {meta['pillow_version']}, "
            f"{meta['backend']} {meta['backend_version']}, "
            f"settings={meta['save_kwargs']}"
        )


if __name__ == "__main__":
    main()
