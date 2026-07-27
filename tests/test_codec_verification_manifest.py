import pytest

from mdlic.codec.container import make_codec_identity
from mdlic.codec.verification import (
    ROUNDTRIP_MANIFEST_SCHEMA,
    build_roundtrip_manifest,
    load_verified_roundtrip,
    write_roundtrip_manifest,
)


def _identity():
    return make_codec_identity(
        model_type="igpt",
        model_config={"type": "igpt", "image_size": 2},
        checkpoint_sha256="11" * 32,
        runtime={"schema": "test-runtime"},
        implementation={
            "schema": "mdlic-codec-implementation-v1",
            "execution_source_sha256": "22" * 32,
            "execution_source_file_count": 1,
        },
    )


def _row():
    return {
        "sample_index": 0,
        "pixel_exact": True,
        "integrity_verified": True,
        "decoded_rgb_checksum_verified": True,
        "container_version": 2,
        "source_rgb_sha256": "33" * 32,
        "ideal_bpd": 2.9,
        "ref_bpd": 2.900001,
        "cdf_ideal_bpd": 2.91,
        "payload_bpd": 2.911,
        "packed_payload_bpd": 2.912,
        "file_bpd": 3.1,
        "cdf_quantization_delta_bpd": 0.01,
        "coder_overhead_bpd": 0.001,
    }


DATASET_FINGERPRINT = "66" * 32


def test_verified_roundtrip_manifest_can_be_attached(tmp_path):
    identity = _identity()
    payload = build_roundtrip_manifest(
        command=["python3", "verify_lossless.py"],
        git={"commit": "test", "dirty": False},
        config={"path": "config.yaml", "sha256": "44" * 32},
        checkpoint={"path": "best.pth", "sha256": "11" * 32},
        model_type="igpt",
        codec_identity=identity,
        dataset={
            "name": "synthetic",
            "fingerprint_sha256": DATASET_FINGERPRINT,
        },
        runtime={"schema": "test-runtime"},
        per_image=[_row()],
    )
    path = tmp_path / "roundtrip.json"
    write_roundtrip_manifest(str(path), payload)

    attachment = load_verified_roundtrip(
        str(path),
        expected_config_sha256="44" * 32,
        expected_checkpoint_sha256="11" * 32,
        expected_model_type="igpt",
        expected_codec_identity=identity,
        expected_dataset_fingerprint_sha256=DATASET_FINGERPRINT,
    )

    assert payload["schema"] == ROUNDTRIP_MANIFEST_SCHEMA
    assert attachment["status"] == "verified_on_subset"
    assert attachment["actual_arithmetic_coding_run"] is True
    assert attachment["image_count"] == 1
    assert len(attachment["manifest"]["sha256"]) == 64


def test_roundtrip_attachment_rejects_a_different_checkpoint(tmp_path):
    identity = _identity()
    payload = build_roundtrip_manifest(
        command=[],
        git={},
        config={"sha256": "44" * 32},
        checkpoint={"sha256": "11" * 32},
        model_type="igpt",
        codec_identity=identity,
        dataset={"fingerprint_sha256": DATASET_FINGERPRINT},
        runtime={},
        per_image=[_row()],
    )
    path = tmp_path / "roundtrip.json"
    write_roundtrip_manifest(str(path), payload)

    with pytest.raises(ValueError, match="checkpoint SHA-256"):
        load_verified_roundtrip(
            str(path),
            expected_config_sha256="44" * 32,
            expected_checkpoint_sha256="55" * 32,
            expected_model_type="igpt",
            expected_codec_identity=identity,
            expected_dataset_fingerprint_sha256=DATASET_FINGERPRINT,
        )


def test_roundtrip_attachment_rejects_a_different_dataset(tmp_path):
    identity = _identity()
    payload = build_roundtrip_manifest(
        command=[],
        git={},
        config={"sha256": "44" * 32},
        checkpoint={"sha256": "11" * 32},
        model_type="igpt",
        codec_identity=identity,
        dataset={"fingerprint_sha256": DATASET_FINGERPRINT},
        runtime={},
        per_image=[_row()],
    )
    path = tmp_path / "roundtrip.json"
    write_roundtrip_manifest(str(path), payload)

    with pytest.raises(ValueError, match="dataset fingerprint"):
        load_verified_roundtrip(
            str(path),
            expected_config_sha256="44" * 32,
            expected_checkpoint_sha256="11" * 32,
            expected_model_type="igpt",
            expected_codec_identity=identity,
            expected_dataset_fingerprint_sha256="77" * 32,
        )


def test_roundtrip_attachment_recomputes_pass_status_from_rows(tmp_path):
    identity = _identity()
    payload = build_roundtrip_manifest(
        command=[],
        git={},
        config={"sha256": "44" * 32},
        checkpoint={"sha256": "11" * 32},
        model_type="igpt",
        codec_identity=identity,
        dataset={"fingerprint_sha256": DATASET_FINGERPRINT},
        runtime={},
        per_image=[_row()],
    )
    payload["per_image"][0]["pixel_exact"] = False
    path = tmp_path / "roundtrip.json"
    write_roundtrip_manifest(str(path), payload)

    with pytest.raises(ValueError, match="did not pass"):
        load_verified_roundtrip(
            str(path),
            expected_config_sha256="44" * 32,
            expected_checkpoint_sha256="11" * 32,
            expected_model_type="igpt",
            expected_codec_identity=identity,
            expected_dataset_fingerprint_sha256=DATASET_FINGERPRINT,
        )
