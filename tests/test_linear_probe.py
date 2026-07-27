from types import SimpleNamespace

import numpy as np
import torch

from scripts.linear_probe import (
    _probe_provenance,
    bootstrap_sample_accuracy,
    parse_seeds,
    select_best_layer,
    stratified_train_val_split,
    summarize_accuracies,
    train_linear_probe,
)


def test_parse_seeds_deduplicates_and_preserves_order():
    assert parse_seeds("3, 1,3, 2") == [3, 1, 2]


def test_stratified_split_is_deterministic_and_keeps_classes():
    labels = torch.tensor([0] * 10 + [1] * 20 + [2] * 30)

    train_a, val_a = stratified_train_val_split(labels, val_fraction=0.2, seed=7)
    train_b, val_b = stratified_train_val_split(labels, val_fraction=0.2, seed=7)

    assert torch.equal(train_a, train_b)
    assert torch.equal(val_a, val_b)
    assert set(train_a.tolist()).isdisjoint(val_a.tolist())
    assert sorted(torch.cat((train_a, val_a)).tolist()) == list(range(len(labels)))
    assert torch.bincount(labels[val_a], minlength=3).tolist() == [2, 4, 6]
    assert torch.bincount(labels[train_a], minlength=3).tolist() == [8, 16, 24]


def test_linear_probe_seed_is_reproducible_on_cpu():
    generator = torch.Generator().manual_seed(123)
    features = torch.randn(48, 6, generator=generator)
    labels = (features[:, 0] + features[:, 1] > 0).long()

    first = train_linear_probe(
        features[:36], labels[:36], features[36:], labels[36:],
        d_model=6, num_classes=2, epochs=4, lr=0.05,
        batch_size=9, device="cpu", seed=11,
    )
    second = train_linear_probe(
        features[:36], labels[:36], features[36:], labels[36:],
        d_model=6, num_classes=2, epochs=4, lr=0.05,
        batch_size=9, device="cpu", seed=11,
    )

    assert first == second


def test_layer_selection_uses_validation_mean_with_stable_tie_break():
    validation_results = [
        {"layer": 2, "classifier_seed_mean": 70.0},
        {"layer": 5, "classifier_seed_mean": 72.0},
        {"layer": 3, "classifier_seed_mean": 72.0},
    ]

    assert select_best_layer(validation_results) == 3


def test_accuracy_summary_reports_classifier_seed_variation_without_ci():
    summary = summarize_accuracies([70.0, 72.0, 74.0])

    assert summary["classifier_seed_mean"] == 72.0
    assert summary["classifier_seed_std"] == 2.0
    assert summary["classifier_seed_min"] == 70.0
    assert summary["classifier_seed_max"] == 74.0
    assert summary["num_classifier_seeds"] == 3
    assert "ci95" not in summary


def test_sample_bootstrap_is_reproducible_and_separate_from_seed_variation():
    correctness = torch.tensor([
        [1, 0, 1, 0, 1, 0],
        [1, 1, 0, 0, 1, 0],
    ], dtype=torch.bool)

    first = bootstrap_sample_accuracy(correctness, bootstrap_samples=500, seed=9)
    second = bootstrap_sample_accuracy(correctness, bootstrap_samples=500, seed=9)

    assert first == second
    assert first["accuracy"] == 50.0
    assert first["num_eval_samples"] == 6
    assert first["bootstrap_samples"] == 500
    assert first["ci95"][0] <= first["accuracy"] <= first["ci95"][1]
    assert "fixed classifier seeds" in first["estimand"]


def test_linear_probe_can_return_per_sample_correctness():
    generator = torch.Generator().manual_seed(77)
    features = torch.randn(30, 4, generator=generator)
    labels = (features[:, 0] > 0).long()

    accuracy, correctness = train_linear_probe(
        features[:20], labels[:20], features[20:], labels[20:],
        d_model=4, num_classes=2, epochs=2, lr=0.05,
        batch_size=5, device="cpu", seed=3, return_correctness=True,
    )

    assert correctness.dtype == torch.bool
    assert correctness.shape == (10,)
    assert accuracy == correctness.float().mean().item() * 100


def test_probe_provenance_records_artifacts_datasets_and_split(tmp_path):
    raw_dir = tmp_path / "cifar-10-batches-py"
    raw_dir.mkdir()
    (raw_dir / "data_batch_1").write_bytes(b"train")
    (raw_dir / "test_batch").write_bytes(b"test")
    config_path = tmp_path / "config.yaml"
    checkpoint_path = tmp_path / "model.pth"
    config_path.write_text("model: {}\n")
    checkpoint_path.write_bytes(b"weights")

    class TinyCifar:
        root = str(tmp_path)
        base_folder = "cifar-10-batches-py"
        train_list = [("data_batch_1", None)]
        test_list = [("test_batch", None)]

        def __init__(self, size):
            self.data = np.zeros((size, 32, 32, 3), dtype=np.uint8)

        def __len__(self):
            return len(self.data)

    payload = _probe_provenance(
        args=SimpleNamespace(
            config=str(config_path),
            checkpoint=str(checkpoint_path),
        ),
        device=torch.device("cpu"),
        dataset_name="cifar10",
        data_root=str(tmp_path),
        train_dataset=TinyCifar(4),
        test_dataset=TinyCifar(2),
        preprocessing={
            "schema": "mdlic-linear-probe-input-v1",
            "steps": [{"op": "ToTensor"}],
            "augmentation": None,
        },
        selection_train_idx=torch.tensor([0, 2, 3]),
        validation_idx=torch.tensor([1]),
    )

    assert payload["artifacts"]["config"]["sha256"]
    assert payload["artifacts"]["checkpoint"]["sha256"]
    assert payload["datasets"]["train"]["storage_shape"] == [4, 32, 32, 3]
    assert payload["datasets"]["test"]["source_files"][0]["sha256"]
    assert payload["selection_split"]["train_count"] == 3
    assert payload["selection_split"]["validation_indices_sha256"]
    assert payload["runtime"]["packages"]["torch"]
