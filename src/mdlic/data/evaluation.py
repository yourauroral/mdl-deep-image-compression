"""Evaluation dataset loading and provenance shared by CLI entry points."""

from __future__ import annotations

from collections.abc import Mapping

from ..provenance import dataset_record


def load_evaluation_dataset(config: Mapping[str, object]):
    """Load the configured test/validation dataset without downloading data."""
    from torchvision import transforms
    from torchvision.datasets import CIFAR10, CIFAR100

    data_config = config["data"]
    dataset_name = data_config.get("dataset", "cifar100")
    if dataset_name == "imagenet64_npy":
        from .imagenet64_npy import ImageNet64Npy

        dataset = ImageNet64Npy(root=data_config["valid"], split="val")
        return dataset, dataset_name
    if dataset_name not in ("cifar10", "cifar100"):
        raise ValueError(
            f"unsupported dataset {dataset_name!r}; expected "
            "cifar10, cifar100, or imagenet64_npy"
        )
    dataset_class = CIFAR10 if dataset_name == "cifar10" else CIFAR100
    dataset = dataset_class(
        root=data_config["valid"],
        train=False,
        download=False,
        transform=transforms.ToTensor(),
    )
    return dataset, dataset_name


def evaluation_dataset_record(dataset, dataset_name: str, config: Mapping) -> dict:
    """Record source files and the exact deterministic evaluation transform."""
    split = "test" if dataset_name in ("cifar10", "cifar100") else "val"
    if dataset_name == "imagenet64_npy":
        preprocessing = {
            "schema": "mdlic-rgb-preprocess-v1",
            "steps": [
                "read uint8 HWC sample from val.npy",
                "transpose HWC to CHW",
                "convert to float32 and divide by 255",
            ],
            "augmentation": None,
        }
    else:
        preprocessing = {
            "schema": "mdlic-rgb-preprocess-v1",
            "steps": [
                "torchvision.transforms.ToTensor: uint8 HWC to float32 CHW in [0,1]",
            ],
            "augmentation": None,
        }
    return dataset_record(
        dataset,
        name=dataset_name,
        split=split,
        configured_path=config["data"].get("valid"),
        preprocessing=preprocessing,
    )
