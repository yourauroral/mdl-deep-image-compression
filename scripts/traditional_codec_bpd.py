#!/usr/bin/env python3
"""Compute PNG/WebP lossless bits-per-dimension for an ImageNet64 .npy file."""

import argparse
import io
from pathlib import Path

import numpy as np
from PIL import Image


def compute_bpd(path: Path, limit: int | None) -> tuple[float, float]:
    data = np.load(path, mmap_mode="r")
    if data.ndim != 4 or data.shape[1:] != (64, 64, 3) or data.dtype != np.uint8:
        raise ValueError(
            f"expected uint8 array with shape (N, 64, 64, 3), got "
            f"{data.shape} {data.dtype}"
        )

    n = data.shape[0] if limit is None else min(limit, data.shape[0])
    dims = data.shape[1] * data.shape[2] * data.shape[3]
    png_bits = 0
    webp_bits = 0

    for i in range(n):
        image = Image.fromarray(np.asarray(data[i], dtype=np.uint8))

        png_buf = io.BytesIO()
        image.save(png_buf, "PNG")
        png_bits += png_buf.tell() * 8

        webp_buf = io.BytesIO()
        image.save(webp_buf, "WEBP", lossless=True)
        webp_bits += webp_buf.tell() * 8

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


if __name__ == "__main__":
    main()
