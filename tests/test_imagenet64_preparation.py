import hashlib
import json

import numpy as np
import pytest

from mdlic.data.manifest import write_imagenet64_manifest
from mdlic.data.imagenet64_npy import ImageNet64Npy
from mdlic.provenance import dataset_record
from scripts import prepare_imagenet64_streaming as streaming


def _write_arrays(root, *, train_count=3, val_count=2):
    np.save(root / "train.npy", np.zeros((train_count, 64, 64, 3), dtype=np.uint8))
    np.save(root / "val.npy", np.ones((val_count, 64, 64, 3), dtype=np.uint8))


def test_dataset_manifest_identifies_actual_npy_files(tmp_path):
    _write_arrays(tmp_path)

    manifest = write_imagenet64_manifest(
        tmp_path,
        producer="test",
        preprocessing={"resize": "BOX"},
        sources={"train": {"files": 3}, "val": {"files": 2}},
    )

    persisted = json.loads((tmp_path / "dataset_manifest.json").read_text())
    assert persisted == manifest
    assert manifest["splits"]["train"]["samples"] == 3
    assert manifest["splits"]["val"]["shape"] == [2, 64, 64, 3]
    assert len(manifest["splits"]["train"]["sha256"]) == 64
    assert len(manifest["fingerprint_sha256"]) == 64

    dataset = ImageNet64Npy(str(tmp_path), "val")
    record = dataset_record(
        dataset,
        name="imagenet64_npy",
        split="val",
        configured_path=str(tmp_path),
        preprocessing={"augmentation": None},
    )
    assert record["preparation_manifest"]["sha256"]


def test_legacy_streaming_state_is_migrated_without_losing_cursor(tmp_path):
    state_path = tmp_path / streaming.STATE_FILENAME
    state_path.write_text(json.dumps({
        "train": {"cursor": 12, "done": ["train-000.parquet"]},
        "val": {"cursor": 3, "done": []},
    }))

    state = streaming.load_state(str(state_path))

    assert state["schema"] == streaming.STATE_SCHEMA
    assert state["train"]["cursor"] == 12
    assert state["train"]["shards"] == [{
        "name": "train-000.parquet",
        "legacy_unverified": True,
    }]


def test_streaming_shard_is_committed_after_durable_readback(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    shard = raw_dir / "train-000.parquet"
    shard.write_bytes(b"fake parquet")
    out_path = tmp_path / "train.npy"
    state_path = tmp_path / streaming.STATE_FILENAME
    state = streaming._empty_state()

    values = np.stack([
        np.full((64, 64, 3), 11, dtype=np.uint8),
        np.full((64, 64, 3), 22, dtype=np.uint8),
    ])

    def fake_process(path, out, cursor, pbar):
        assert path == str(shard)
        out[cursor:cursor + len(values)] = values
        pbar.update(len(values))
        return (
            cursor + len(values),
            hashlib.sha256(values.tobytes()).hexdigest(),
            len(values),
        )

    monkeypatch.setattr(streaming, "process_shard", fake_process)
    complete = streaming.process_split(
        str(raw_dir), str(out_path), "train", 2, "train-*.parquet",
        state, str(state_path), delete_processed=True,
    )

    assert complete is True
    assert not shard.exists()
    assert state["train"]["cursor"] == 2
    assert state["train"]["shards"][0]["start"] == 0
    assert state["train"]["shards"][0]["end"] == 2
    assert json.loads(state_path.read_text()) == state
    assert np.array_equal(np.load(out_path), values)


def test_flush_failure_does_not_commit_state_or_delete_source(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    shard = raw_dir / "val-000.parquet"
    shard.write_bytes(b"fake parquet")
    state = streaming._empty_state()
    state_path = tmp_path / streaming.STATE_FILENAME

    def fake_process(path, out, cursor, pbar):
        value = np.full((64, 64, 3), 7, dtype=np.uint8)
        out[cursor] = value
        return cursor + 1, hashlib.sha256(value.tobytes()).hexdigest(), 1

    monkeypatch.setattr(streaming, "process_shard", fake_process)
    monkeypatch.setattr(
        streaming,
        "_flush_and_fsync_memmap",
        lambda out, path: (_ for _ in ()).throw(RuntimeError("flush failed")),
    )

    with pytest.raises(RuntimeError, match="flush failed"):
        streaming.process_split(
            str(raw_dir), str(tmp_path / "val.npy"), "val", 1,
            "val-*.parquet", state, str(state_path), delete_processed=True,
        )

    assert state["val"] == {"cursor": 0, "shards": []}
    assert shard.exists()
    assert not state_path.exists()


def test_state_commit_failure_rolls_back_memory_and_keeps_source(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    shard = raw_dir / "val-000.parquet"
    shard.write_bytes(b"fake parquet")
    state = streaming._empty_state()

    def fake_process(path, out, cursor, pbar):
        value = np.full((64, 64, 3), 9, dtype=np.uint8)
        out[cursor] = value
        return cursor + 1, hashlib.sha256(value.tobytes()).hexdigest(), 1

    monkeypatch.setattr(streaming, "process_shard", fake_process)
    monkeypatch.setattr(
        streaming,
        "save_state",
        lambda path, value: (_ for _ in ()).throw(RuntimeError("state failed")),
    )

    with pytest.raises(RuntimeError, match="state failed"):
        streaming.process_split(
            str(raw_dir), str(tmp_path / "val.npy"), "val", 1,
            "val-*.parquet", state, str(tmp_path / streaming.STATE_FILENAME),
            delete_processed=True,
        )

    assert state["val"] == {"cursor": 0, "shards": []}
    assert shard.exists()


def test_nonzero_cursor_rejects_missing_output(tmp_path):
    state = streaming._empty_state()
    state["train"] = {
        "cursor": 1,
        "shards": [{"name": "train-000.parquet", "legacy_unverified": True}],
    }

    with pytest.raises(RuntimeError, match="output file is missing"):
        streaming.process_split(
            str(tmp_path), str(tmp_path / "train.npy"), "train", 2,
            "train-*.parquet", state, str(tmp_path / streaming.STATE_FILENAME),
        )
