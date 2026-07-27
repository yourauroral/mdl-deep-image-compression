#!/usr/bin/env python3
"""Run reproducible formal AR evaluations on the AutoDL CUDA host."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNS = {
    "cifar10": {
        "config": ROOT / "configs/ccigpt_cifar10_s_rgb_ronly_v2.yaml",
        "checkpoint": ROOT / "experiments/ccigpt_cifar10_s_rgb_ronly_v2/checkpoints/best.pth",
    },
    "imagenet64": {
        "config": ROOT / "configs/ccigpt_imagenet64_v1.yaml",
        "checkpoint": ROOT / "experiments/ccigpt_imagenet64_v1/checkpoints/best.pth",
    },
}


def _evaluation_command(
    *,
    config: Path,
    checkpoint: Path,
    result_json: Path,
    batch_size: int,
    nproc_per_node: int,
    bootstrap_samples: int,
    codec_verification_json: Path | None,
) -> list[str]:
    if nproc_per_node == 1:
        command = [sys.executable]
    else:
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={nproc_per_node}",
        ]
    command.extend([
        str(ROOT / "scripts/evaluate.py"),
        "--formal",
        "--config", str(config),
        "--checkpoint", str(checkpoint),
        "--batch_size", str(batch_size),
        "--bootstrap_samples", str(bootstrap_samples),
        "--result_json", str(result_json),
    ])
    if codec_verification_json is not None:
        command.extend([
            "--codec_verification_json", str(codec_verification_json),
        ])
    return command


def _verification_command(
    *,
    config: Path,
    checkpoint: Path,
    result_json: Path,
    dump_dir: Path,
    num_images: int,
) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "scripts/verify_lossless.py"),
        "--config", str(config),
        "--checkpoint", str(checkpoint),
        "--num_images", str(num_images),
        "--dump_dir", str(dump_dir),
        "--result_json", str(result_json),
    ]


def _run(command: list[str], *, dry_run: bool) -> None:
    print(f"$ {shlex.join(command)}", flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "datasets", nargs="*", choices=sorted(RUNS),
        help="要评测的数据集；默认两者都跑",
    )
    parser.add_argument("--output_dir", type=Path, default=ROOT / "results/formal")
    parser.add_argument("--nproc_per_node", type=int, default=1)
    parser.add_argument("--bootstrap_samples", type=int, default=1000)
    parser.add_argument("--cifar_checkpoint", type=Path)
    parser.add_argument("--imagenet64_checkpoint", type=Path)
    parser.add_argument("--cifar_batch_size", type=int, default=48)
    parser.add_argument("--imagenet64_batch_size", type=int, default=8)
    parser.add_argument("--verify_cifar_images", type=int, default=0)
    parser.add_argument("--verify_imagenet64_images", type=int, default=0)
    parser.add_argument(
        "--cifar_verification_json", type=Path,
        help="显式复用已有 CIFAR-10 sequential roundtrip manifest",
    )
    parser.add_argument(
        "--imagenet64_verification_json", type=Path,
        help="显式复用已有 ImageNet64 sequential roundtrip manifest",
    )
    parser.add_argument("--dry_run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.nproc_per_node < 1:
        parser.error("--nproc_per_node must be >= 1")
    if args.bootstrap_samples < 0:
        parser.error("--bootstrap_samples must be >= 0")
    if min(
        args.cifar_batch_size,
        args.imagenet64_batch_size,
        args.verify_cifar_images,
        args.verify_imagenet64_images,
    ) < 0:
        parser.error("batch sizes and verification counts must be non-negative")
    if args.cifar_batch_size < 1 or args.imagenet64_batch_size < 1:
        parser.error("batch sizes must be >= 1")

    overrides = {
        "cifar10": args.cifar_checkpoint,
        "imagenet64": args.imagenet64_checkpoint,
    }
    batch_sizes = {
        "cifar10": args.cifar_batch_size,
        "imagenet64": args.imagenet64_batch_size,
    }
    verification_counts = {
        "cifar10": args.verify_cifar_images,
        "imagenet64": args.verify_imagenet64_images,
    }
    verification_overrides = {
        "cifar10": args.cifar_verification_json,
        "imagenet64": args.imagenet64_verification_json,
    }
    for dataset_name, count in verification_counts.items():
        if count > 0 and verification_overrides[dataset_name] is not None:
            parser.error(
                f"{dataset_name}: choose either a verification image count "
                "or an existing verification manifest, not both"
            )

    datasets = args.datasets or sorted(RUNS)
    for dataset_name in datasets:
        spec = RUNS[dataset_name]
        config = spec["config"]
        checkpoint = (overrides[dataset_name] or spec["checkpoint"]).resolve()
        if not config.is_file():
            parser.error(f"missing config: {config}")
        if not args.dry_run and not checkpoint.is_file():
            parser.error(f"missing checkpoint: {checkpoint}")

        output_dir = args.output_dir.resolve() / dataset_name
        result_json = output_dir / "teacher_forced.json"
        verification_json = output_dir / "sequential_roundtrip.json"
        verification_count = verification_counts[dataset_name]
        attachment = verification_overrides[dataset_name]
        if verification_count > 0:
            _run(
                _verification_command(
                    config=config,
                    checkpoint=checkpoint,
                    result_json=verification_json,
                    dump_dir=output_dir / "bitstreams",
                    num_images=verification_count,
                ),
                dry_run=args.dry_run,
            )
            attachment = verification_json
        elif attachment is not None:
            attachment = attachment.resolve()
            if not args.dry_run and not attachment.is_file():
                parser.error(f"missing codec verification manifest: {attachment}")
        _run(
            _evaluation_command(
                config=config,
                checkpoint=checkpoint,
                result_json=result_json,
                batch_size=batch_sizes[dataset_name],
                nproc_per_node=args.nproc_per_node,
                bootstrap_samples=args.bootstrap_samples,
                codec_verification_json=attachment,
            ),
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
