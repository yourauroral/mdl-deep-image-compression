"""Protocol tests for checksummed, model-bound MDLC containers."""

import copy
import random

import pytest

from mdlic.codec.container import (
    CURRENT_VERSION,
    V2_CHECKSUM_SIZE,
    V2_FIXED_HEADER_SIZE,
    build_legacy_v1_container_bytes,
    canonical_sha256,
    make_codec_identity,
    parse_container,
    sha256_rgb_bytes,
    validate_decoded_rgb,
)
from scripts.verify_lossless import (
    _build_container_bytes,
    _parse_container_meta,
    _read_container,
    _read_container_bytes,
    _write_container,
)


SOURCE_RGB = b"synthetic decoded RGB bytes"
SOURCE_SHA256 = sha256_rgb_bytes(SOURCE_RGB)


def _identity(
    *,
    checkpoint="11" * 32,
    config=None,
    protocol=None,
    schedule=None,
    runtime=None,
):
    return make_codec_identity(
        model_type="ccigpt",
        model_config=config or {"type": "ccigpt", "d_model": 16},
        checkpoint_sha256=checkpoint,
        codec_protocol=protocol,
        schedule=schedule,
        runtime=runtime or {"torch": "test", "device_type": "cpu"},
    )


def _rand_bits(n):
    return [random.randint(0, 1) for _ in range(n)]


def _build(*, dual=True):
    random.seed(7)
    coarse = _rand_bits(517) if dual else []
    fine = _rand_bits(8001)
    return _build_container_bytes(
        dual,
        32,
        3,
        coarse,
        fine,
        identity=_identity(),
        source_rgb_sha256=SOURCE_SHA256,
    ), coarse, fine


@pytest.mark.parametrize("dual", [False, True])
def test_v2_default_roundtrip(dual):
    blob, coarse, fine = _build(dual=dual)
    parsed = parse_container(blob, expected_identity=_identity())
    result = _read_container_bytes(blob, expected_identity=_identity())

    assert parsed.version == CURRENT_VERSION
    assert parsed.integrity_verified is True
    assert result == (dual, 32, 3, coarse, fine)


def test_v1_is_read_only_compatible():
    coarse = [1, 0, 1]
    fine = [0, 1] * 7
    blob = build_legacy_v1_container_bytes(True, 32, 3, coarse, fine)

    assert _read_container_bytes(blob) == (True, 32, 3, coarse, fine)
    meta = _parse_container_meta(blob)
    assert meta["version"] == 1
    assert meta["integrity_verified"] is False
    assert meta["integrity"] == "structural-only-legacy-v1"
    assert meta["model_bound"] is False
    with pytest.raises(ValueError, match="no model/protocol identity"):
        _read_container_bytes(blob, expected_identity=_identity())


def test_v2_metadata_and_rate_fields():
    blob, coarse, fine = _build()
    meta = _parse_container_meta(blob)

    assert meta["magic"] == "MDLC"
    assert meta["version"] == 2
    assert (meta["H"], meta["W"], meta["C"]) == (32, 32, 3)
    assert meta["fixed_header_size"] == V2_FIXED_HEADER_SIZE
    assert meta["checksum_bytes"] == V2_CHECKSUM_SIZE
    assert meta["payload_bytes"] == (len(coarse) + 7) // 8 + (len(fine) + 7) // 8
    assert meta["header_size"] == (
        meta["fixed_header_size"] + meta["metadata_bytes"] + meta["checksum_bytes"]
    )
    assert meta["total_bytes"] == meta["header_size"] + meta["payload_bytes"]
    assert meta["payload_bpd"] == pytest.approx(
        (len(coarse) + len(fine)) / (32 * 32 * 3)
    )
    assert meta["file_bpd"] == pytest.approx(len(blob) * 8 / (32 * 32 * 3))
    assert meta["source_rgb_sha256"] == SOURCE_SHA256
    assert meta["codec_identity"] == _identity()


@pytest.mark.parametrize("region", ["header", "metadata", "payload", "checksum"])
def test_v2_rejects_bit_flip_in_every_region(region):
    blob, _, _ = _build()
    parsed = parse_container(blob)
    positions = {
        "header": 6,
        "metadata": V2_FIXED_HEADER_SIZE + 3,
        "payload": V2_FIXED_HEADER_SIZE + parsed.metadata_size,
        "checksum": len(blob) - 1,
    }
    corrupted = bytearray(blob)
    corrupted[positions[region]] ^= 0x01

    with pytest.raises(ValueError):
        parse_container(corrupted)


def test_v2_rejects_wrong_checkpoint_config_protocol_schedule_and_runtime():
    blob, _, _ = _build()
    protocol = {
        "name": "different-coder",
        "first_token_prior": "uniform-256",
        "cdf": {"total": 65536, "quantizer": "different"},
    }
    schedule = {
        "name": "different-order",
        "stream_order": "coarse-then-fine",
        "within_stream": "right-to-left",
    }
    mismatches = {
        "checkpoint_sha256": _identity(checkpoint="22" * 32),
        "model_config_sha256": _identity(config={"type": "ccigpt", "d_model": 32}),
        "codec_protocol_sha256": _identity(protocol=protocol),
        "schedule_sha256": _identity(schedule=schedule),
        "runtime_sha256": _identity(runtime={"torch": "other", "device_type": "cpu"}),
    }

    for field, expected in mismatches.items():
        with pytest.raises(ValueError, match=field):
            parse_container(blob, expected_identity=expected)


def test_identity_rejects_internal_protocol_hash_mismatch():
    identity = copy.deepcopy(_identity())
    identity["codec_protocol"]["name"] = "tampered"
    with pytest.raises(ValueError, match="codec_protocol"):
        _build_container_bytes(
            False,
            32,
            3,
            [],
            [1],
            identity=identity,
            source_rgb_sha256=SOURCE_SHA256,
        )


def test_decoded_rgb_checksum_is_enforced():
    blob, _, _ = _build()
    meta = _parse_container_meta(blob)

    validate_decoded_rgb(SOURCE_RGB, meta)
    with pytest.raises(ValueError, match="decoded RGB checksum mismatch"):
        validate_decoded_rgb(SOURCE_RGB + b"corrupt", meta)


def test_truncation_trailing_bytes_and_bad_magic_are_rejected():
    blob, _, _ = _build()
    for invalid in (blob[:-1], blob + b"extra", b"XXXX" + blob[4:]):
        with pytest.raises(ValueError):
            _parse_container_meta(invalid)


def test_build_rejects_invalid_geometry_streams_and_digest():
    identity = _identity()
    with pytest.raises(ValueError, match="H"):
        _build_container_bytes(
            False, 0, 3, [], [1], identity=identity, source_rgb_sha256=SOURCE_SHA256
        )
    with pytest.raises(ValueError, match="dual"):
        _build_container_bytes(
            False, 32, 3, [1], [1], identity=identity,
            source_rgb_sha256=SOURCE_SHA256,
        )
    with pytest.raises(ValueError, match="source_rgb_sha256"):
        _build_container_bytes(
            False, 32, 3, [], [1], identity=identity, source_rgb_sha256="bad"
        )


def test_write_matches_build_and_enforces_identity_on_read(tmp_path):
    blob, coarse, fine = _build()
    path = tmp_path / "image.mdlc"
    written = _write_container(
        path,
        True,
        32,
        3,
        coarse,
        fine,
        identity=_identity(),
        source_rgb_sha256=SOURCE_SHA256,
    )

    assert path.read_bytes() == blob
    assert written == len(blob)
    assert _read_container(path, expected_identity=_identity()) == (
        True,
        32,
        3,
        coarse,
        fine,
    )


def test_make_identity_hashes_canonical_config():
    first = _identity(config={"type": "ccigpt", "nested": {"a": 1, "b": 2}})
    second = _identity(config={"nested": {"b": 2, "a": 1}, "type": "ccigpt"})
    assert first["model_config_sha256"] == second["model_config_sha256"]
    assert first["model_config_sha256"] == canonical_sha256(
        {"type": "ccigpt", "nested": {"a": 1, "b": 2}}
    )
