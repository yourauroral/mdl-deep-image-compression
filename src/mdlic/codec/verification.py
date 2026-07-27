"""Schema and validation for real sequential codec roundtrip evidence."""

from __future__ import annotations

import json
import hashlib
import math
import os
import statistics
from collections.abc import Mapping, Sequence

from .container import CURRENT_VERSION, canonical_sha256


ROUNDTRIP_MANIFEST_SCHEMA = "mdlic-sequential-codec-roundtrip-v1"
_RATE_FIELDS = (
    "ideal_bpd",
    "ref_bpd",
    "cdf_ideal_bpd",
    "payload_bpd",
    "packed_payload_bpd",
    "file_bpd",
    "cdf_quantization_delta_bpd",
    "coder_overhead_bpd",
)


def _is_sha256(value) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _validate_roundtrip_rows(rows: Sequence[Mapping]) -> list[dict]:
    normalized = [dict(row) for row in rows]
    if not normalized:
        raise ValueError("roundtrip manifest requires at least one image")

    sample_indices = []
    required_flags = (
        "pixel_exact",
        "integrity_verified",
        "decoded_rgb_checksum_verified",
    )
    for index, row in enumerate(normalized):
        required = (
            "sample_index",
            *required_flags,
            "container_version",
            "source_rgb_sha256",
            *_RATE_FIELDS,
        )
        missing = [key for key in required if key not in row]
        if missing:
            raise ValueError(
                f"roundtrip row {index} is missing: {', '.join(missing)}"
            )
        sample_index = row["sample_index"]
        if (
            not isinstance(sample_index, int)
            or isinstance(sample_index, bool)
            or sample_index < 0
        ):
            raise ValueError(f"roundtrip row {index} has invalid sample_index")
        sample_indices.append(sample_index)
        if any(row[key] is not True for key in required_flags):
            raise ValueError(f"roundtrip row {index} did not pass lossless verification")
        if row["container_version"] != CURRENT_VERSION:
            raise ValueError(
                f"roundtrip row {index} is not an MDLC v{CURRENT_VERSION} result"
            )
        if not _is_sha256(row["source_rgb_sha256"]):
            raise ValueError(f"roundtrip row {index} has invalid source RGB SHA-256")
        for field in _RATE_FIELDS:
            value = row[field]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError(
                    f"roundtrip row {index} has invalid {field}: {value!r}"
                )
    if len(set(sample_indices)) != len(sample_indices):
        raise ValueError("roundtrip sample_indices must be unique")
    return normalized


def build_roundtrip_manifest(
    *,
    command: Sequence[str],
    git: Mapping,
    config: Mapping,
    checkpoint: Mapping,
    model_type: str,
    codec_identity: Mapping,
    dataset: Mapping,
    runtime: Mapping,
    per_image: Sequence[Mapping],
) -> dict:
    """Build a machine-readable record from sequential encode/decode results."""
    rows = _validate_roundtrip_rows(per_image)

    identity = dict(codec_identity)
    summary = {
        f"mean_{field}": statistics.fmean(float(row[field]) for row in rows)
        for field in _RATE_FIELDS
    }
    summary["max_abs_teacher_forced_vs_sequential_ideal_bpd"] = max(
        abs(float(row["ref_bpd"]) - float(row["ideal_bpd"])) for row in rows
    )
    return {
        "schema": ROUNDTRIP_MANIFEST_SCHEMA,
        "protocol": {
            "execution": "full-prefix-zero-suffix-per-token-v1",
            "actual_arithmetic_coding": True,
            "container_version": CURRENT_VERSION,
        },
        "command": list(command),
        "git": dict(git),
        "config": dict(config),
        "checkpoint": dict(checkpoint),
        "model": {
            "type": model_type,
            "codec_identity": identity,
            "codec_identity_sha256": canonical_sha256(identity),
        },
        "dataset": dict(dataset),
        "runtime": dict(runtime),
        "verification": {
            "image_count": len(rows),
            "sample_indices": [int(row["sample_index"]) for row in rows],
            "all_pixel_exact": all(bool(row["pixel_exact"]) for row in rows),
            "all_integrity_verified": all(
                bool(row["integrity_verified"]) for row in rows
            ),
            "all_rgb_checksums_verified": all(
                bool(row["decoded_rgb_checksum_verified"]) for row in rows
            ),
            "all_mdlc_v2": all(
                int(row["container_version"]) == CURRENT_VERSION for row in rows
            ),
        },
        "summary": summary,
        "per_image": rows,
    }


def write_roundtrip_manifest(path: str, payload: Mapping) -> None:
    """Atomically write a roundtrip manifest."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    os.replace(tmp_path, path)


def load_verified_roundtrip(
    path: str,
    *,
    expected_config_sha256: str,
    expected_checkpoint_sha256: str,
    expected_model_type: str,
    expected_codec_identity: Mapping,
    expected_dataset_fingerprint_sha256: str,
) -> dict:
    """Validate attached evidence and return its compact evaluation attachment."""
    absolute = os.path.abspath(path)
    with open(absolute, "rb") as handle:
        raw_manifest = handle.read()
    try:
        payload = json.loads(raw_manifest.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("codec verification manifest is not valid UTF-8 JSON") from exc
    if payload.get("schema") != ROUNDTRIP_MANIFEST_SCHEMA:
        raise ValueError(
            f"codec verification schema must be {ROUNDTRIP_MANIFEST_SCHEMA!r}"
        )
    if payload.get("config", {}).get("sha256") != expected_config_sha256:
        raise ValueError("codec verification config SHA-256 does not match evaluation")
    if payload.get("checkpoint", {}).get("sha256") != expected_checkpoint_sha256:
        raise ValueError("codec verification checkpoint SHA-256 does not match evaluation")
    if payload.get("model", {}).get("type") != expected_model_type:
        raise ValueError("codec verification model type does not match evaluation")
    if (
        payload.get("dataset", {}).get("fingerprint_sha256")
        != expected_dataset_fingerprint_sha256
    ):
        raise ValueError(
            "codec verification dataset fingerprint does not match evaluation"
        )

    protocol = payload.get("protocol", {})
    if protocol != {
        "execution": "full-prefix-zero-suffix-per-token-v1",
        "actual_arithmetic_coding": True,
        "container_version": CURRENT_VERSION,
    }:
        raise ValueError("codec verification protocol is missing or unsupported")

    actual_identity = payload.get("model", {}).get("codec_identity")
    actual_identity_hash = payload.get("model", {}).get("codec_identity_sha256")
    if not isinstance(actual_identity, dict) or (
        canonical_sha256(actual_identity) != actual_identity_hash
    ):
        raise ValueError("codec verification identity hash is internally inconsistent")
    if actual_identity != dict(expected_codec_identity):
        raise ValueError(
            "codec verification identity does not match current checkpoint, source, or runtime"
        )

    rows_value = payload.get("per_image")
    if not isinstance(rows_value, list):
        raise ValueError("codec verification per_image must be a list")
    rows = _validate_roundtrip_rows(rows_value)
    sample_indices = [row["sample_index"] for row in rows]
    image_count = len(rows)
    expected_verification = {
        "image_count": image_count,
        "sample_indices": sample_indices,
        "all_pixel_exact": True,
        "all_integrity_verified": True,
        "all_rgb_checksums_verified": True,
        "all_mdlc_v2": True,
    }
    if payload.get("verification") != expected_verification:
        raise ValueError(
            "codec verification summary does not match its per-image records"
        )

    summary = payload.get("summary")
    if not isinstance(summary, dict):
        raise ValueError("codec verification summary must be a mapping")
    expected_summary = {
        f"mean_{field}": statistics.fmean(float(row[field]) for row in rows)
        for field in _RATE_FIELDS
    }
    expected_summary["max_abs_teacher_forced_vs_sequential_ideal_bpd"] = max(
        abs(float(row["ref_bpd"]) - float(row["ideal_bpd"])) for row in rows
    )
    for key, expected_value in expected_summary.items():
        actual_value = summary.get(key)
        if (
            isinstance(actual_value, bool)
            or not isinstance(actual_value, (int, float))
            or not math.isclose(
                float(actual_value), expected_value, rel_tol=0.0, abs_tol=1e-12
            )
        ):
            raise ValueError(
                "codec verification rate summary does not match per-image records"
            )

    return {
        "status": "verified_on_subset",
        "actual_arithmetic_coding_run": True,
        "image_count": image_count,
        "sample_indices": sample_indices,
        "codec_identity_sha256": actual_identity_hash,
        "manifest": {
            "path": absolute,
            "size_bytes": len(raw_manifest),
            "sha256": hashlib.sha256(raw_manifest).hexdigest(),
        },
    }
