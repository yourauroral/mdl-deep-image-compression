"""Versioned MDLC containers for model-bound lossless bitstreams.

MDLC v1 is retained for read compatibility.  MDLC v2 binds a bitstream to
the model and numerical codec protocol, stores the decoded RGB digest, and
protects the complete header/metadata/payload with SHA-256.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ..provenance import runtime_metadata, source_tree_record
from .arithmetic import FREQ_TOTAL, pack_bits, unpack_bits


MAGIC = b"MDLC"
LEGACY_VERSION = 1
CURRENT_VERSION = 2

V1_HEADER = ">4sBBBBII"
V1_HEADER_SIZE = struct.calcsize(V1_HEADER)

# magic, version, flags, H, W, C, reserved, metadata_nbytes, c_nbits, f_nbits
V2_HEADER = ">4sBBHHBBIQQ"
V2_FIXED_HEADER_SIZE = struct.calcsize(V2_HEADER)
V2_CHECKSUM_SIZE = hashlib.sha256().digest_size

FLAG_DUAL = 1 << 0
KNOWN_FLAGS = FLAG_DUAL
LEGACY_IDENTITY_SCHEMA = "mdlic-codec-identity-v1"
IDENTITY_SCHEMA = "mdlic-codec-identity-v2"
METADATA_SCHEMA = "mdlc-v2-metadata"
IMPLEMENTATION_SCHEMA = "mdlic-codec-implementation-v1"
CODEC_SOURCE_PATHS = (
    "src/mdlic",
    "scripts/train.py",
    "scripts/verify_lossless.py",
    "pyproject.toml",
)

DEFAULT_CODEC_PROTOCOL = {
    "name": "mdlic-wnc-arithmetic-v1",
    "first_token_prior": "uniform-256",
    "cdf": {
        "total": FREQ_TOTAL,
        "quantizer": "min-one-largest-remainder-index-tiebreak-v1",
    },
    "probabilities": {
        "logits_dtype": "float32",
        "softmax_dtype": "float64",
    },
    "forward": "full-prefix-zero-suffix-per-token-v1",
}


@dataclass(frozen=True)
class ParsedContainer:
    version: int
    dual: bool
    height: int
    width: int
    channels: int
    coarse_nbits: int
    fine_nbits: int
    coarse_data: bytes
    fine_data: bytes
    fixed_header_size: int
    metadata_size: int
    checksum_size: int
    metadata: dict[str, Any] | None
    integrity_verified: bool

    @property
    def packed_payload_size(self) -> int:
        return len(self.coarse_data) + len(self.fine_data)

    @property
    def non_payload_size(self) -> int:
        return self.fixed_header_size + self.metadata_size + self.checksum_size


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def codec_implementation_fingerprint(repo_root: str | Path | None = None) -> dict[str, Any]:
    """Fingerprint code that can alter model logits or arithmetic coding."""
    source = source_tree_record(repo_root, include_paths=CODEC_SOURCE_PATHS)
    return {
        "schema": IMPLEMENTATION_SCHEMA,
        "execution_source_sha256": source["fingerprint_sha256"],
        "execution_source_file_count": source["file_count"],
    }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_rgb_bytes(rgb_bytes: bytes) -> str:
    return hashlib.sha256(rgb_bytes).hexdigest()


def default_ar_schedule(model_type: str) -> dict[str, str]:
    if model_type == "ccigpt":
        stream_order = "coarse-then-fine"
    elif model_type == "igpt":
        stream_order = "fine-only"
    else:
        raise ValueError(f"unsupported AR codec model_type: {model_type!r}")
    return {
        "name": "strict-raster-pixel-first-ar-v1",
        "stream_order": stream_order,
        "within_stream": "left-to-right",
    }


def runtime_fingerprint(device) -> dict[str, Any]:
    """Describe numerical runtime details that can affect arithmetic decode."""
    details = runtime_metadata(device)
    return {
        "schema": "mdlic-codec-runtime-v2",
        "python": details["python"],
        "platform": details["platform"],
        "packages": {
            "torch": details["packages"]["torch"],
            "triton": details["packages"]["triton"],
        },
        "cuda_runtime": details["cuda_runtime"],
        "cuda_driver": details["cuda_driver"],
        "cudnn": details["cudnn"],
        "device_type": details["device_type"],
        "device_name": details["device_name"],
        "device_capability": details["device_capability"],
        "numerics": details["numerics"],
        "environment": details["environment"],
    }


def make_codec_identity(
    *,
    model_type: str,
    model_config: Mapping[str, Any],
    checkpoint_sha256: str,
    runtime: Mapping[str, Any],
    implementation: Mapping[str, Any] | None = None,
    schedule: Mapping[str, Any] | None = None,
    codec_protocol: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create the identity that encoder and decoder must match exactly."""
    _validate_sha256(checkpoint_sha256, "checkpoint_sha256")
    protocol = dict(codec_protocol or DEFAULT_CODEC_PROTOCOL)
    schedule_dict = dict(schedule or default_ar_schedule(model_type))
    runtime_dict = dict(runtime)
    implementation_dict = dict(
        implementation or codec_implementation_fingerprint()
    )
    return {
        "schema": IDENTITY_SCHEMA,
        "model_type": model_type,
        "checkpoint_sha256": checkpoint_sha256.lower(),
        "model_config_sha256": canonical_sha256(dict(model_config)),
        "codec_protocol": protocol,
        "codec_protocol_sha256": canonical_sha256(protocol),
        "schedule": schedule_dict,
        "schedule_sha256": canonical_sha256(schedule_dict),
        "runtime": runtime_dict,
        "runtime_sha256": canonical_sha256(runtime_dict),
        "implementation": implementation_dict,
        "implementation_sha256": canonical_sha256(implementation_dict),
    }


def in_memory_model_identity(
    model,
    model_type: str,
    device,
    *,
    schedule: Mapping[str, Any] | None = None,
    codec_protocol: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a deterministic identity for tiny tests without checkpoint files."""
    import torch

    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(struct.pack(">I", value.ndim))
        for dim in value.shape:
            digest.update(struct.pack(">Q", dim))
        digest.update(value.view(torch.uint8).numpy().tobytes())
    model_config = {
        "class": f"{model.__class__.__module__}.{model.__class__.__qualname__}",
        "image_size": int(model.image_size),
        "in_channels": int(model.in_channels),
        "state_sha256": digest.hexdigest(),
    }
    return make_codec_identity(
        model_type=model_type,
        model_config=model_config,
        checkpoint_sha256=digest.hexdigest(),
        runtime=runtime_fingerprint(device),
        schedule=schedule,
        codec_protocol=codec_protocol,
    )


def build_container_bytes(
    dual: bool,
    height: int,
    channels: int,
    coarse_bits: list[int],
    fine_bits: list[int],
    *,
    identity: Mapping[str, Any],
    source_rgb_sha256: str,
    width: int | None = None,
) -> bytes:
    """Build a checksummed, model-bound MDLC v2 byte string."""
    width = height if width is None else width
    _validate_geometry(height, width, channels, version=CURRENT_VERSION)
    _validate_streams(dual, coarse_bits, fine_bits)
    _validate_identity(identity)
    _validate_sha256(source_rgb_sha256, "source_rgb_sha256")

    metadata = {
        "schema": METADATA_SCHEMA,
        "identity": dict(identity),
        "source_rgb_sha256": source_rgb_sha256.lower(),
    }
    metadata_bytes = canonical_json_bytes(metadata)
    if len(metadata_bytes) > 0xFFFFFFFF:
        raise ValueError("MDLC v2 metadata exceeds uint32 length")

    coarse_data = pack_bits(coarse_bits) if coarse_bits else b""
    fine_data = pack_bits(fine_bits)
    flags = FLAG_DUAL if dual else 0
    header = struct.pack(
        V2_HEADER,
        MAGIC,
        CURRENT_VERSION,
        flags,
        height,
        width,
        channels,
        0,
        len(metadata_bytes),
        len(coarse_bits),
        len(fine_bits),
    )
    content = header + metadata_bytes + coarse_data + fine_data
    return content + hashlib.sha256(content).digest()


def build_legacy_v1_container_bytes(
    dual: bool,
    height: int,
    channels: int,
    coarse_bits: list[int],
    fine_bits: list[int],
) -> bytes:
    """Build MDLC v1 only for compatibility fixtures and migration tests."""
    _validate_geometry(height, height, channels, version=LEGACY_VERSION)
    _validate_streams(dual, coarse_bits, fine_bits)
    header = struct.pack(
        V1_HEADER,
        MAGIC,
        LEGACY_VERSION,
        1 if dual else 0,
        height,
        channels,
        len(coarse_bits),
        len(fine_bits),
    )
    return (
        header
        + (pack_bits(coarse_bits) if coarse_bits else b"")
        + pack_bits(fine_bits)
    )


def parse_container(
    blob: bytes | bytearray | memoryview,
    *,
    expected_identity: Mapping[str, Any] | None = None,
) -> ParsedContainer:
    if not isinstance(blob, (bytes, bytearray, memoryview)):
        raise TypeError("MDLC blob must be bytes-like")
    data = bytes(blob)
    if len(data) < 5:
        raise ValueError("MDLC container is shorter than magic and version")
    if data[:4] != MAGIC:
        raise ValueError(f"not an MDLC container (magic={data[:4]!r})")
    version = data[4]
    if version == LEGACY_VERSION:
        parsed = _parse_v1(data)
    elif version == CURRENT_VERSION:
        parsed = _parse_v2(data)
    else:
        raise ValueError(f"unsupported MDLC version {version}")

    if expected_identity is not None:
        if parsed.metadata is None:
            raise ValueError("MDLC v1 has no model/protocol identity to verify")
        _validate_expected_identity(parsed.metadata["identity"], expected_identity)
    return parsed


def read_container_bytes(
    blob: bytes | bytearray | memoryview,
    *,
    expected_identity: Mapping[str, Any] | None = None,
) -> tuple[bool, int, int, list[int], list[int]]:
    parsed = parse_container(blob, expected_identity=expected_identity)
    coarse_bits = (
        unpack_bits(parsed.coarse_data, parsed.coarse_nbits)
        if parsed.coarse_nbits
        else []
    )
    fine_bits = unpack_bits(parsed.fine_data, parsed.fine_nbits)
    return parsed.dual, parsed.height, parsed.channels, coarse_bits, fine_bits


def write_container(
    path: str | Path,
    dual: bool,
    height: int,
    channels: int,
    coarse_bits: list[int],
    fine_bits: list[int],
    *,
    identity: Mapping[str, Any],
    source_rgb_sha256: str,
    width: int | None = None,
) -> int:
    """Atomically write a checksummed, model-bound MDLC v2 container."""
    blob = build_container_bytes(
        dual,
        height,
        channels,
        coarse_bits,
        fine_bits,
        identity=identity,
        source_rgb_sha256=source_rgb_sha256,
        width=width,
    )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.name}.tmp.{os.getpid()}")
    try:
        with open(temporary, "wb") as handle:
            handle.write(blob)
        os.replace(temporary, target)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return len(blob)


def read_container(
    path: str | Path,
    *,
    expected_identity: Mapping[str, Any] | None = None,
) -> tuple[bool, int, int, list[int], list[int]]:
    """Read an MDLC file and optionally enforce its codec identity."""
    with open(path, "rb") as handle:
        return read_container_bytes(
            handle.read(),
            expected_identity=expected_identity,
        )


def parse_container_meta(blob: bytes | bytearray | memoryview) -> dict[str, Any]:
    parsed = parse_container(blob)
    total_bytes = len(blob)
    payload_bytes = parsed.packed_payload_size
    total_bits = parsed.coarse_nbits + parsed.fine_nbits
    n_subpix = parsed.height * parsed.width * parsed.channels
    metadata = parsed.metadata or {}
    identity = metadata.get("identity")
    return {
        "total_bytes": total_bytes,
        "header_size": parsed.non_payload_size,
        "fixed_header_size": parsed.fixed_header_size,
        "metadata_bytes": parsed.metadata_size,
        "checksum_bytes": parsed.checksum_size,
        "header_hex": " ".join(
            f"{value:02x}" for value in bytes(blob)[:parsed.fixed_header_size]
        ),
        "magic": MAGIC.decode("ascii"),
        "version": parsed.version,
        "dual": parsed.dual,
        "H": parsed.height,
        "W": parsed.width,
        "C": parsed.channels,
        "n_subpix": n_subpix,
        "coarse_nbits": parsed.coarse_nbits,
        "coarse_nbytes": len(parsed.coarse_data),
        "fine_nbits": parsed.fine_nbits,
        "fine_nbytes": len(parsed.fine_data),
        "payload_bytes": payload_bytes,
        "total_bits": total_bits,
        "bpd": total_bits / n_subpix,
        "payload_bpd": total_bits / n_subpix,
        "packed_payload_bpd": payload_bytes * 8 / n_subpix,
        "file_bpd": total_bytes * 8 / n_subpix,
        "self_consistent": True,
        "integrity_verified": parsed.integrity_verified,
        "integrity": (
            "sha256-complete-container"
            if parsed.integrity_verified
            else "structural-only-legacy-v1"
        ),
        "model_bound": identity is not None,
        "codec_identity": identity,
        "source_rgb_sha256": metadata.get("source_rgb_sha256"),
    }


def validate_decoded_rgb(
    rgb_bytes: bytes,
    metadata_or_meta: Mapping[str, Any],
) -> None:
    expected = metadata_or_meta.get("source_rgb_sha256")
    if expected is None:
        raise ValueError("container has no decoded RGB checksum (legacy MDLC v1)")
    actual = sha256_rgb_bytes(rgb_bytes)
    if not hmac.compare_digest(actual, expected):
        raise ValueError(
            "decoded RGB checksum mismatch; model, protocol, or payload is inconsistent"
        )


def _parse_v1(data: bytes) -> ParsedContainer:
    if len(data) < V1_HEADER_SIZE:
        raise ValueError(
            f"MDLC v1 container is too short: {len(data)} < {V1_HEADER_SIZE}"
        )
    magic, version, dual, height, channels, c_nbits, f_nbits = struct.unpack(
        V1_HEADER, data[:V1_HEADER_SIZE]
    )
    if magic != MAGIC or version != LEGACY_VERSION:
        raise ValueError("invalid MDLC v1 header")
    if dual not in (0, 1):
        raise ValueError(f"invalid dual flag: {dual}")
    _validate_geometry(height, height, channels, version=LEGACY_VERSION)
    _validate_declared_streams(bool(dual), c_nbits, f_nbits)
    c_nbytes = (c_nbits + 7) // 8
    f_nbytes = (f_nbits + 7) // 8
    expected_size = V1_HEADER_SIZE + c_nbytes + f_nbytes
    _validate_exact_size(data, expected_size)
    offset = V1_HEADER_SIZE
    coarse_data = data[offset:offset + c_nbytes]
    fine_data = data[offset + c_nbytes:]
    _validate_padding_zero(coarse_data, c_nbits, "coarse")
    _validate_padding_zero(fine_data, f_nbits, "fine")
    return ParsedContainer(
        version=version,
        dual=bool(dual),
        height=height,
        width=height,
        channels=channels,
        coarse_nbits=c_nbits,
        fine_nbits=f_nbits,
        coarse_data=coarse_data,
        fine_data=fine_data,
        fixed_header_size=V1_HEADER_SIZE,
        metadata_size=0,
        checksum_size=0,
        metadata=None,
        integrity_verified=False,
    )


def _parse_v2(data: bytes) -> ParsedContainer:
    minimum = V2_FIXED_HEADER_SIZE + V2_CHECKSUM_SIZE
    if len(data) < minimum:
        raise ValueError(f"MDLC v2 container is too short: {len(data)} < {minimum}")
    (
        magic,
        version,
        flags,
        height,
        width,
        channels,
        reserved,
        metadata_size,
        c_nbits,
        f_nbits,
    ) = struct.unpack(V2_HEADER, data[:V2_FIXED_HEADER_SIZE])
    if magic != MAGIC or version != CURRENT_VERSION:
        raise ValueError("invalid MDLC v2 header")
    if flags & ~KNOWN_FLAGS:
        raise ValueError(f"unknown MDLC v2 flags: 0x{flags:02x}")
    if reserved != 0:
        raise ValueError("MDLC v2 reserved header byte must be zero")
    dual = bool(flags & FLAG_DUAL)
    _validate_geometry(height, width, channels, version=CURRENT_VERSION)
    _validate_declared_streams(dual, c_nbits, f_nbits)

    c_nbytes = (c_nbits + 7) // 8
    f_nbytes = (f_nbits + 7) // 8
    expected_size = (
        V2_FIXED_HEADER_SIZE
        + metadata_size
        + c_nbytes
        + f_nbytes
        + V2_CHECKSUM_SIZE
    )
    _validate_exact_size(data, expected_size)

    content = data[:-V2_CHECKSUM_SIZE]
    expected_digest = data[-V2_CHECKSUM_SIZE:]
    actual_digest = hashlib.sha256(content).digest()
    if not hmac.compare_digest(actual_digest, expected_digest):
        raise ValueError("MDLC v2 SHA-256 integrity check failed")

    offset = V2_FIXED_HEADER_SIZE
    metadata_bytes = data[offset:offset + metadata_size]
    try:
        metadata = json.loads(metadata_bytes.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid MDLC v2 metadata JSON") from exc
    _validate_metadata(metadata)
    if canonical_json_bytes(metadata) != metadata_bytes:
        raise ValueError("MDLC v2 metadata is not canonical JSON")

    offset += metadata_size
    coarse_data = data[offset:offset + c_nbytes]
    fine_data = data[offset + c_nbytes:offset + c_nbytes + f_nbytes]
    _validate_padding_zero(coarse_data, c_nbits, "coarse")
    _validate_padding_zero(fine_data, f_nbits, "fine")
    return ParsedContainer(
        version=version,
        dual=dual,
        height=height,
        width=width,
        channels=channels,
        coarse_nbits=c_nbits,
        fine_nbits=f_nbits,
        coarse_data=coarse_data,
        fine_data=fine_data,
        fixed_header_size=V2_FIXED_HEADER_SIZE,
        metadata_size=metadata_size,
        checksum_size=V2_CHECKSUM_SIZE,
        metadata=metadata,
        integrity_verified=True,
    )


def _validate_geometry(height: int, width: int, channels: int, *, version: int) -> None:
    limit = 255 if version == LEGACY_VERSION else 65535
    for name, value in (("H", height), ("W", width)):
        if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= limit:
            raise ValueError(f"MDLC v{version} {name} must be in [1,{limit}], got {value!r}")
    if not isinstance(channels, int) or isinstance(channels, bool) or not 1 <= channels <= 255:
        raise ValueError(f"MDLC v{version} C must be in [1,255], got {channels!r}")


def _validate_streams(
    dual: bool,
    coarse_bits: list[int],
    fine_bits: list[int],
) -> None:
    if bool(dual) != bool(coarse_bits):
        raise ValueError("dual flag does not match coarse bitstream presence")
    if not fine_bits:
        raise ValueError("fine bitstream must not be empty")
    for name, bits in (("coarse", coarse_bits), ("fine", fine_bits)):
        if any(bit not in (0, 1) for bit in bits):
            raise ValueError(f"{name} bitstream may contain only 0/1")
        if len(bits) > 0xFFFFFFFFFFFFFFFF:
            raise ValueError(f"{name} bitstream exceeds MDLC v2 uint64 limit")


def _validate_declared_streams(dual: bool, c_nbits: int, f_nbits: int) -> None:
    if not dual and c_nbits != 0:
        raise ValueError("single-scale container must not declare a coarse bitstream")
    if dual and c_nbits == 0:
        raise ValueError("dual-scale container must contain a coarse bitstream")
    if f_nbits == 0:
        raise ValueError("fine bitstream must not be empty")


def _validate_padding_zero(data: bytes, n_bits: int, label: str) -> None:
    remainder = n_bits % 8
    if remainder and data:
        padding_mask = (1 << (8 - remainder)) - 1
        if data[-1] & padding_mask:
            raise ValueError(f"{label} bitstream has non-zero byte padding")


def _validate_exact_size(data: bytes, expected_size: int) -> None:
    if len(data) != expected_size:
        relation = "truncated" if len(data) < expected_size else "has trailing bytes"
        raise ValueError(
            f"MDLC container size mismatch: header declares {expected_size} bytes, "
            f"actual size is {len(data)} ({relation})"
        )


def _validate_sha256(value: str, label: str) -> None:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} must be a 64-character hexadecimal SHA-256")
    try:
        bytes.fromhex(value)
    except ValueError as exc:
        raise ValueError(f"{label} must be hexadecimal") from exc


def _validate_identity(identity: Mapping[str, Any]) -> None:
    if not isinstance(identity, Mapping):
        raise ValueError("codec identity must be a mapping")
    schema = identity.get("schema")
    if schema not in (LEGACY_IDENTITY_SCHEMA, IDENTITY_SCHEMA):
        raise ValueError(
            "codec identity schema must be one of "
            f"{LEGACY_IDENTITY_SCHEMA!r}, {IDENTITY_SCHEMA!r}"
        )
    digest_keys = [
        "checkpoint_sha256",
        "model_config_sha256",
        "codec_protocol_sha256",
        "schedule_sha256",
        "runtime_sha256",
    ]
    value_hash_pairs = [
        ("codec_protocol", "codec_protocol_sha256"),
        ("schedule", "schedule_sha256"),
        ("runtime", "runtime_sha256"),
    ]
    if schema == IDENTITY_SCHEMA:
        implementation = identity.get("implementation")
        if not isinstance(implementation, Mapping) or (
            implementation.get("schema") != IMPLEMENTATION_SCHEMA
        ):
            raise ValueError(
                f"codec implementation schema must be {IMPLEMENTATION_SCHEMA!r}"
            )
        digest_keys.append("implementation_sha256")
        value_hash_pairs.append(("implementation", "implementation_sha256"))
    for key in digest_keys:
        _validate_sha256(identity.get(key), key)
    for value_key, hash_key in value_hash_pairs:
        if canonical_sha256(identity.get(value_key)) != identity[hash_key]:
            raise ValueError(f"codec identity {value_key} does not match {hash_key}")


def _validate_metadata(metadata: Any) -> None:
    if not isinstance(metadata, dict) or metadata.get("schema") != METADATA_SCHEMA:
        raise ValueError(f"MDLC v2 metadata schema must be {METADATA_SCHEMA!r}")
    _validate_identity(metadata.get("identity"))
    _validate_sha256(metadata.get("source_rgb_sha256"), "source_rgb_sha256")


def _validate_expected_identity(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> None:
    _validate_identity(expected)
    keys = [
        "schema",
        "model_type",
        "checkpoint_sha256",
        "model_config_sha256",
        "codec_protocol_sha256",
        "schedule_sha256",
        "runtime_sha256",
    ]
    if expected.get("schema") == IDENTITY_SCHEMA:
        keys.append("implementation_sha256")
    mismatches = [key for key in keys if actual.get(key) != expected.get(key)]
    if mismatches:
        raise ValueError(
            "MDLC codec identity mismatch: " + ", ".join(mismatches)
        )
