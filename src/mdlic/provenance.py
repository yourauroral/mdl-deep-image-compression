"""Reproducibility metadata shared by training and evaluation tools."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Iterable


SOURCE_TREE_SCHEMA = "mdlic-execution-source-v1"
DEFAULT_SOURCE_PATHS = (
    "src",
    "scripts",
    "pyproject.toml",
)
SOURCE_SUFFIXES = {".py", ".pyi", ".toml", ".lock", ".txt"}


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_record(path: str | os.PathLike[str]) -> dict:
    absolute = os.path.abspath(os.fspath(path))
    return {
        "path": absolute,
        "size_bytes": os.path.getsize(absolute),
        "sha256": sha256_file(absolute),
    }


def _source_candidates(repo_root: Path, include_paths: Iterable[str]) -> list[Path]:
    include_paths = tuple(os.fspath(path) for path in include_paths)
    process = subprocess.run(
        [
            "git", "ls-files", "-z", "--cached", "--others",
            "--exclude-standard", "--", *include_paths,
        ],
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if process.returncode == 0:
        relative_paths = [
            Path(os.fsdecode(value))
            for value in process.stdout.split(b"\0")
            if value
        ]
    else:
        relative_paths = []
        for configured in include_paths:
            path = repo_root / configured
            if path.is_dir():
                relative_paths.extend(
                    candidate.relative_to(repo_root)
                    for candidate in path.rglob("*")
                    if candidate.is_file()
                )
            elif path.is_file():
                relative_paths.append(path.relative_to(repo_root))

    return sorted({
        path
        for path in relative_paths
        if (repo_root / path).is_file()
        and (path.suffix in SOURCE_SUFFIXES or path.name == "pyproject.toml")
    }, key=lambda path: path.as_posix())


def source_tree_record(
    repo_root: str | os.PathLike[str] | None = None,
    *,
    include_paths: Iterable[str] = DEFAULT_SOURCE_PATHS,
) -> dict:
    """Fingerprint the execution-relevant contents of the current worktree."""
    root = (
        Path(repo_root).resolve()
        if repo_root is not None
        else Path(__file__).resolve().parents[2]
    )
    files = []
    for relative in _source_candidates(root, include_paths):
        absolute = root / relative
        files.append({
            "path": relative.as_posix(),
            "size_bytes": absolute.stat().st_size,
            "sha256": sha256_file(absolute),
        })
    identity = {
        "schema": SOURCE_TREE_SCHEMA,
        "files": files,
    }
    return {
        **identity,
        "file_count": len(files),
        "fingerprint_sha256": canonical_sha256(identity),
    }


def git_metadata(
    repo_root: str | os.PathLike[str] | None = None,
    *,
    source_tree: dict | None = None,
) -> dict:
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[2]

    def run(*args):
        process = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return process.stdout.strip() if process.returncode == 0 else None

    status = run("status", "--porcelain")
    source = source_tree if source_tree is not None else source_tree_record(repo_root)
    return {
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(status) if status is not None else None,
        "execution_source_sha256": source["fingerprint_sha256"],
        "execution_source_file_count": source["file_count"],
        "execution_source": source,
    }


def _package_version(distribution: str) -> str | None:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def _nvidia_driver_version() -> str | None:
    try:
        process = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=5,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None
    if process.returncode != 0:
        return None
    versions = sorted({line.strip() for line in process.stdout.splitlines() if line.strip()})
    return ",".join(versions) or None


def runtime_metadata(device=None) -> dict:
    import torch

    if device is not None:
        device = torch.device(device)
    device_name = None
    device_capability = None
    if device is not None and device.type == "cuda" and torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(device)
        device_capability = list(torch.cuda.get_device_capability(device))

    cuda_backend = torch.backends.cuda
    backend_flag_names = (
        "flash_sdp_enabled",
        "mem_efficient_sdp_enabled",
        "math_sdp_enabled",
    )
    backend_flags = {}
    for name in backend_flag_names:
        getter = getattr(cuda_backend, name, None)
        backend_flags[name] = bool(getter()) if callable(getter) else None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            "numpy": _package_version("numpy"),
            "pillow": _package_version("Pillow"),
            "torch": torch.__version__,
            "torchvision": _package_version("torchvision"),
            "triton": _package_version("triton"),
        },
        "cuda_runtime": torch.version.cuda,
        "cuda_driver": _nvidia_driver_version() if torch.cuda.is_available() else None,
        "cudnn": torch.backends.cudnn.version(),
        "device": str(device) if device is not None else None,
        "device_type": device.type if device is not None else None,
        "device_name": device_name,
        "device_capability": device_capability,
        "numerics": {
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "deterministic_warn_only": bool(
                getattr(torch, "is_deterministic_algorithms_warn_only_enabled", lambda: False)()
            ),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "allow_tf32_matmul": bool(
                getattr(torch.backends.cuda.matmul, "allow_tf32", False)
            ),
            "allow_tf32_cudnn": bool(
                getattr(torch.backends.cudnn, "allow_tf32", False)
            ),
            **backend_flags,
        },
        "environment": {
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "NVIDIA_TF32_OVERRIDE": os.environ.get("NVIDIA_TF32_OVERRIDE"),
        },
    }


def _dataset_source_paths(dataset, split: str) -> list[str]:
    data = getattr(dataset, "data", None)
    filename = getattr(data, "filename", None)
    if filename:
        return [os.fspath(filename)]

    root = getattr(dataset, "root", None)
    base_folder = getattr(dataset, "base_folder", None)
    file_list = getattr(
        dataset,
        "train_list" if split == "train" else "test_list",
        None,
    )
    if root is not None and base_folder is not None and file_list:
        return [
            os.path.join(os.fspath(root), os.fspath(base_folder), entry[0])
            for entry in file_list
        ]
    return []


def dataset_record(
    dataset,
    *,
    name: str,
    split: str,
    configured_path: str | None,
    preprocessing: dict,
    source_paths: Iterable[str | os.PathLike[str]] | None = None,
) -> dict:
    """Describe raw files and preprocessing well enough to identify a dataset."""
    if source_paths is None:
        source_paths = _dataset_source_paths(dataset, split)
    source_paths = list(source_paths)
    if not source_paths:
        raise ValueError(f"cannot identify source files for dataset {name!r} split {split!r}")

    data = getattr(dataset, "data", None)
    shape = list(getattr(data, "shape", ())) or None
    dtype = str(getattr(data, "dtype", "unknown"))
    sources = [file_record(path) for path in source_paths]
    identity = {
        "name": name,
        "split": split,
        "size": len(dataset),
        "storage_shape": shape,
        "storage_dtype": dtype,
        "preprocessing": preprocessing,
        "sources": [
            {
                "name": os.path.basename(source["path"]),
                "size_bytes": source["size_bytes"],
                "sha256": source["sha256"],
            }
            for source in sources
        ],
    }
    result = {
        **identity,
        "configured_path": (
            os.path.abspath(os.path.expanduser(configured_path))
            if configured_path is not None
            else None
        ),
        "source_files": sources,
        "fingerprint_sha256": canonical_sha256(identity),
    }
    manifest_path = getattr(dataset, "manifest_path", None)
    if manifest_path:
        result["preparation_manifest"] = file_record(manifest_path)
    return result
