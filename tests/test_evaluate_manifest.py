import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mdlic.codec.probability import codec_teacher_forced_score_metadata
from mdlic.provenance import dataset_record, source_tree_record
from scripts.evaluate import (
    _current_codec_protocol_support,
    _write_result_manifest,
)


def test_current_codec_protocol_support_requires_single_model_without_tta():
    assert _current_codec_protocol_support(members=1, tta_hflip=False)[0] is True
    assert _current_codec_protocol_support(members=2, tta_hflip=False)[0] is False
    assert _current_codec_protocol_support(members=1, tta_hflip=True)[0] is False
    assert _current_codec_protocol_support(
        members=1,
        tta_hflip=False,
        probability_numerics_match=False,
    )[0] is False


def test_source_tree_fingerprint_tracks_actual_source_contents(tmp_path):
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    module = source_dir / "module.py"
    module.write_text("VALUE = 1\n")

    first = source_tree_record(tmp_path, include_paths=("src",))
    module.write_text("VALUE = 2\n")
    second = source_tree_record(tmp_path, include_paths=("src",))

    assert first["file_count"] == 1
    assert first["files"][0]["path"] == "src/module.py"
    assert first["fingerprint_sha256"] != second["fingerprint_sha256"]


def test_ensemble_manifest_separates_protocol_support_from_roundtrip(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("model:\n  type: igpt\ndata:\n  valid: datasets/\n")
    checkpoints = []
    for index in range(2):
        path = tmp_path / f"model-{index}.pth"
        path.write_bytes(f"checkpoint-{index}".encode())
        checkpoints.append(str(path))

    result_path = tmp_path / "result.json"
    model_a = torch.nn.Linear(2, 2)
    model_b = torch.nn.Linear(2, 2)
    args = SimpleNamespace(config=str(config_path), tta_hflip=False)
    source_path = tmp_path / "test_batch"
    source_path.write_bytes(b"raw-cifar-test-data")

    class TinyDataset:
        data = np.zeros((2, 32, 32, 3), dtype=np.uint8)

        def __len__(self):
            return 2

    dataset_metadata = dataset_record(
        TinyDataset(),
        name="cifar10",
        split="test",
        configured_path=str(tmp_path),
        preprocessing={
            "schema": "mdlic-rgb-preprocess-v1",
            "steps": ["ToTensor"],
            "augmentation": None,
        },
        source_paths=[source_path],
    )
    _write_result_manifest(
        str(result_path),
        args=args,
        config={"model": {"type": "igpt"}, "data": {"valid": "datasets/"}},
        dataset_name="cifar10",
        dataset_size=2,
        dataset_metadata=dataset_metadata,
        models=[model_a, model_b],
        checkpoint_paths=checkpoints,
        bpd_mean=3.0,
        bpd_std=0.1,
        summary={"n": 2, "mean": 3.0},
        extras={},
        device=torch.device("cpu"),
    )

    payload = json.loads(result_path.read_text())
    assert payload["schema_version"] == 5
    assert "decodable_by_current_codec" not in payload
    assert payload["codec_protocol_support"]["supported"] is False
    assert "exactly one checkpoint" in payload["codec_protocol_support"]["reason"]
    assert payload["sequential_roundtrip"]["status"] == "not_run"
    assert payload["sequential_roundtrip"]["actual_arithmetic_coding_run"] is False
    assert payload["rate_accounting"]["metric"] == "teacher_forced_ideal_model_bpd"
    assert payload["evaluation_execution"] == {
        "distributed": False,
        "world_size": 1,
        "collective_backend": None,
        "dataset_sharding": "full-dataset",
        "batch_size_per_rank": None,
        "dataloader_workers_per_rank": 2,
    }
    assert payload["model"]["stored_parameters"] == sum(
        p.numel() for p in model_a.parameters()
    ) * 2
    assert payload["dataset"]["storage_shape"] == [2, 32, 32, 3]
    assert payload["dataset"]["storage_dtype"] == "uint8"
    assert payload["dataset"]["source_files"][0]["sha256"]
    assert payload["dataset"]["fingerprint_sha256"]
    assert payload["score_numerics"]["name"] == "unspecified"


def test_formal_teacher_forced_manifest_does_not_claim_roundtrip(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("model:\n  type: igpt\ndata:\n  valid: datasets/\n")
    checkpoint = tmp_path / "best.pth"
    checkpoint.write_bytes(b"checkpoint")
    source_path = tmp_path / "test_batch"
    source_path.write_bytes(b"dataset")

    class TinyDataset:
        data = np.zeros((1, 2, 2, 1), dtype=np.uint8)

        def __len__(self):
            return 1

    metadata = dataset_record(
        TinyDataset(),
        name="cifar10",
        split="test",
        configured_path=str(tmp_path),
        preprocessing={"schema": "test", "steps": [], "augmentation": None},
        source_paths=[source_path],
    )
    result_path = tmp_path / "result.json"
    args = SimpleNamespace(
        config=str(config_path),
        tta_hflip=False,
        formal=True,
        codec_verification_json=None,
    )
    _write_result_manifest(
        str(result_path),
        args=args,
        config={"model": {"type": "igpt"}, "data": {"valid": "datasets/"}},
        dataset_name="cifar10",
        dataset_size=1,
        dataset_metadata=metadata,
        models=[torch.nn.Linear(2, 2)],
        checkpoint_paths=[checkpoint],
        bpd_mean=3.0,
        bpd_std=0.0,
        summary={"n": 1, "mean": 3.0, "std": 0.0, "stderr": 0.0},
        extras={
            "score_numerics": codec_teacher_forced_score_metadata(),
            "per_image_records": [
                {"sample_id": 0, "ideal_model_bpd": 3.0},
            ],
        },
        device=torch.device("cpu"),
    )

    payload = json.loads(result_path.read_text())
    assert payload["reporting_tier"] == "formal_single_model_teacher_forced"
    assert payload["codec_protocol_support"]["supported"] is True
    assert payload["sequential_roundtrip"]["status"] == "not_run"
    assert payload["rate_accounting"]["actual_arithmetic_coding_run"] is False


def test_formal_manifest_rejects_incomplete_per_image_coverage(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("model:\n  type: igpt\ndata:\n  valid: datasets/\n")
    checkpoint = tmp_path / "best.pth"
    checkpoint.write_bytes(b"checkpoint")
    source_path = tmp_path / "test_batch"
    source_path.write_bytes(b"dataset")

    class TinyDataset:
        data = np.zeros((2, 2, 2, 1), dtype=np.uint8)

        def __len__(self):
            return 2

    metadata = dataset_record(
        TinyDataset(),
        name="cifar10",
        split="test",
        configured_path=str(tmp_path),
        preprocessing={"schema": "test", "steps": [], "augmentation": None},
        source_paths=[source_path],
    )
    args = SimpleNamespace(
        config=str(config_path),
        tta_hflip=False,
        formal=True,
        codec_verification_json=None,
    )

    with pytest.raises(ValueError, match="one per-image record"):
        _write_result_manifest(
            str(tmp_path / "result.json"),
            args=args,
            config={"model": {"type": "igpt"}, "data": {"valid": "datasets/"}},
            dataset_name="cifar10",
            dataset_size=2,
            dataset_metadata=metadata,
            models=[torch.nn.Linear(2, 2)],
            checkpoint_paths=[checkpoint],
            bpd_mean=3.0,
            bpd_std=0.0,
            summary={"n": 1, "mean": 3.0},
            extras={
                "score_numerics": codec_teacher_forced_score_metadata(),
                "per_image_records": [
                    {"sample_id": 0, "ideal_model_bpd": 3.0},
                ],
            },
            device=torch.device("cpu"),
        )
