"""Durable metadata helpers for prepared ImageNet64 ``.npy`` datasets."""

from __future__ import annotations

import json
import os
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Mapping

import numpy as np

from mdlic.provenance import canonical_sha256, file_record


DATASET_MANIFEST_FILENAME = "dataset_manifest.json"
DATASET_MANIFEST_SCHEMA = "mdlic-imagenet64-dataset-v1"


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def fsync_file(path: str | os.PathLike[str]) -> None:
    """Make prior writes to an existing file durable."""
    descriptor = os.open(os.fspath(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def fsync_directory(path: str | os.PathLike[str]) -> None:
    """Persist directory entry updates such as replace and unlink."""
    directory = os.path.dirname(os.path.abspath(os.fspath(path))) or "."
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_json(path: str | os.PathLike[str], value: Any) -> None:
    """Atomically replace a JSON file and fsync both data and directory entry."""
    destination = os.path.abspath(os.fspath(path))
    temporary = f"{destination}.tmp.{os.getpid()}"
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=True, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        fsync_directory(destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def npy_split_record(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Validate and identify one uint8 HWC ImageNet64 array."""
    absolute = os.path.abspath(os.fspath(path))
    array = np.load(absolute, mmap_mode="r")
    try:
        if array.ndim != 4 or tuple(array.shape[1:]) != (64, 64, 3):
            raise ValueError(
                f"unexpected ImageNet64 shape {array.shape} in {absolute}"
            )
        if array.dtype != np.uint8:
            raise ValueError(
                f"unexpected ImageNet64 dtype {array.dtype} in {absolute}"
            )
        shape = list(array.shape)
        dtype = str(array.dtype)
    finally:
        del array
    record = file_record(absolute)
    return {
        **record,
        "name": os.path.basename(absolute),
        "samples": shape[0],
        "shape": shape,
        "dtype": dtype,
    }


def write_imagenet64_manifest(
    out_dir: str | os.PathLike[str],
    *,
    producer: str,
    preprocessing: Mapping[str, Any],
    sources: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write one factual manifest for the existing train/val dataset files."""
    output_directory = os.path.abspath(os.fspath(out_dir))
    splits = {
        split: npy_split_record(os.path.join(output_directory, f"{split}.npy"))
        for split in ("train", "val")
    }
    identity = {
        "schema": DATASET_MANIFEST_SCHEMA,
        "producer": producer,
        "preprocessing": dict(preprocessing),
        "tool_versions": {
            "numpy": np.__version__,
            "pillow": _package_version("Pillow"),
            "pyarrow": _package_version("pyarrow"),
        },
        "splits": {
            split: {
                key: record[key]
                for key in ("name", "size_bytes", "sha256", "samples", "shape", "dtype")
            }
            for split, record in splits.items()
        },
        "sources": dict(sources or {}),
    }
    payload = {
        **identity,
        "output_files": {
            split: {**record, "path": record["name"]}
            for split, record in splits.items()
        },
        "fingerprint_sha256": canonical_sha256(identity),
    }
    atomic_write_json(
        os.path.join(output_directory, DATASET_MANIFEST_FILENAME),
        payload,
    )
    return payload
