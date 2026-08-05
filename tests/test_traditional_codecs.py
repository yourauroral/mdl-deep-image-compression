import io

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import TensorDataset

from mdlic.traditional_codecs import (
    codec_metadata,
    encode_rgb_array,
    get_codec_spec,
)
from scripts.evaluate import compute_traditional_bpd
from scripts.traditional_codec_bpd import compute_bpd as compute_imagenet_traditional_bpd


def test_codec_specs_pin_reported_cifar_options():
    assert get_codec_spec("png").kwargs() == {"optimize": True}
    assert get_codec_spec("webp").kwargs() == {"lossless": True}
    with pytest.raises(ValueError, match="unsupported traditional codec"):
        get_codec_spec("jpeg")


@pytest.mark.parametrize("method", ["png", "webp"])
def test_traditional_codec_roundtrip_is_rgb_exact(method):
    image = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
    payload = encode_rgb_array(image, method)
    decoded = np.asarray(Image.open(io.BytesIO(payload)).convert("RGB"))
    assert np.array_equal(decoded, image)


def test_compute_traditional_bpd_reports_protocol_metadata():
    images = torch.stack([
        torch.zeros(3, 4, 4),
        torch.ones(3, 4, 4),
    ])
    dataset = TensorDataset(images, torch.zeros(2, dtype=torch.long))
    details = compute_traditional_bpd(
        dataset,
        method="png",
        return_details=True,
    )

    assert details["dataset_size"] == 2
    assert details["save_kwargs"] == {"optimize": True}
    assert details["denominator"] == "H*W*C per image"
    assert details["mean_bpd"] > 0
    assert details["std_per_image"] >= 0
    assert details["compressed_bytes_total"] > 0
    assert details["pillow_version"] == codec_metadata("png")["pillow_version"]


def test_imagenet_traditional_manifest_records_dataset_and_rates(tmp_path):
    path = tmp_path / "val.npy"
    data = np.zeros((2, 64, 64, 3), dtype=np.uint8)
    data[1, ::2, ::2, 0] = 255
    np.save(path, data)

    manifest = compute_imagenet_traditional_bpd(path, limit=None)

    assert manifest["schema_version"] == 1
    assert manifest["protocol"] == "traditional_lossless_codec_bpd"
    assert manifest["dataset"]["sample_count"] == 2
    assert manifest["dataset"]["total_samples"] == 2
    assert manifest["dataset"]["file"]["sha256"]
    assert manifest["rate_accounting"]["includes_file_headers"] is True
    for method in ("png", "webp"):
        result = manifest["results"][method]
        assert result["sample_count"] == 2
        assert result["mean_bpd"] > 0
        assert result["std_per_image"] >= 0
        assert result["compressed_bytes_total"] > 0


def test_encode_rejects_non_rgb_uint8_input():
    with pytest.raises(ValueError, match="HWC uint8 RGB"):
        encode_rgb_array(np.zeros((4, 4), dtype=np.uint8), "png")
    with pytest.raises(ValueError, match="HWC uint8 RGB"):
        encode_rgb_array(np.zeros((4, 4, 3), dtype=np.float32), "png")
