import json
from types import SimpleNamespace

import numpy as np
import torch

from mdlic.provenance import dataset_record, source_tree_record
from scripts.evaluate import (
    _current_codec_compatibility,
    _write_result_manifest,
)


def test_current_codec_compatibility_requires_single_model_without_tta():
    assert _current_codec_compatibility(members=1, tta_hflip=False)[0] is True
    assert _current_codec_compatibility(members=2, tta_hflip=False)[0] is False
    assert _current_codec_compatibility(members=1, tta_hflip=True)[0] is False
    assert _current_codec_compatibility(
        members=1,
        tta_hflip=False,
        probability_numerics_aligned=False,
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


def test_ensemble_manifest_is_not_marked_decodable(tmp_path):
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
    assert payload["schema_version"] == 4
    assert payload["decodable_by_current_codec"] is False
    assert payload["codec_compatibility"]["supported"] is False
    assert "exactly one checkpoint" in payload["codec_compatibility"]["reason"]
    assert payload["model"]["stored_parameters"] == sum(
        p.numel() for p in model_a.parameters()
    ) * 2
    assert payload["dataset"]["storage_shape"] == [2, 32, 32, 3]
    assert payload["dataset"]["storage_dtype"] == "uint8"
    assert payload["dataset"]["source_files"][0]["sha256"]
    assert payload["dataset"]["fingerprint_sha256"]
    assert payload["score_numerics"]["name"] == "unspecified"
