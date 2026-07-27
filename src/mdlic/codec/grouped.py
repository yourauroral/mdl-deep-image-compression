"""Grouped masked arithmetic coding with frozen within-group probabilities."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from ..masking_scheduler import MaskSchedule
from .arithmetic import (
    FREQ_TOTAL,
    ArithmeticDecoder,
    ArithmeticEncoder,
    build_cumfreq,
)
from .container import (
    build_container_bytes,
    canonical_sha256,
    parse_container,
    read_container_bytes,
    sha256_rgb_bytes,
    validate_decoded_rgb,
)


ProbabilityFn = Callable[
    [tuple[int, ...], int, tuple[int, ...]],
    Sequence[Sequence[float]],
]


def grouped_codec_protocol() -> dict[str, Any]:
    return {
        "name": "mdlic-grouped-masked-arithmetic-v1",
        "first_token_prior": "none-all-fine-tokens-model-coded",
        "cdf": {
            "total": FREQ_TOTAL,
            "quantizer": "min-one-largest-remainder-index-tiebreak-v1",
        },
        "probabilities": {
            "logits_dtype": "float32",
            "softmax_dtype": "float64",
        },
        "forward": "one-bidirectional-forward-per-group-v1",
        "within_group": "independent-frozen-marginals-index-order-v1",
        "state_update": "reveal-after-entire-group-v1",
    }


@dataclass(frozen=True)
class GroupedEncoding:
    bits: tuple[int, ...]
    teacher_forced_nll_bits: float
    quantized_cdf_nll_bits: float
    forward_calls: int


@dataclass(frozen=True)
class GroupedDecoding:
    tokens: tuple[int, ...]
    forward_calls: int


def _validate_tokens(tokens: Sequence[int], schedule: MaskSchedule, vocab_size: int) -> tuple[int, ...]:
    if not isinstance(vocab_size, int) or isinstance(vocab_size, bool) or vocab_size < 2:
        raise ValueError("vocab_size must be an integer >= 2")
    try:
        values = tuple(tokens)
    except TypeError as exc:
        raise TypeError("tokens must be a sequence") from exc
    if len(values) != schedule.num_tokens:
        raise ValueError(
            f"token length {len(values)} does not match schedule {schedule.num_tokens}"
        )
    for index, value in enumerate(values):
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or not 0 <= value < vocab_size
        ):
            raise ValueError(
                f"tokens[{index}] must be an integer in [0,{vocab_size})"
            )
    return values


def _validate_mask_token_id(mask_token_id: int, vocab_size: int) -> None:
    if (
        not isinstance(mask_token_id, int)
        or isinstance(mask_token_id, bool)
        or mask_token_id < vocab_size
    ):
        raise ValueError(
            f"mask_token_id must be an integer >= vocab_size ({vocab_size})"
        )


def _group_tables(
    probability_fn: ProbabilityFn,
    state: list[int],
    group_index: int,
    positions: tuple[int, ...],
    vocab_size: int,
) -> tuple[list[list[float]], list[list[int]]]:
    rows = list(probability_fn(tuple(state), group_index, positions))
    if len(rows) != len(positions):
        raise ValueError(
            f"probability_fn returned {len(rows)} rows for {len(positions)} positions"
        )

    normalized_rows: list[list[float]] = []
    cdfs: list[list[int]] = []
    for row_index, row in enumerate(rows):
        values = [float(value) for value in row]
        if len(values) != vocab_size:
            raise ValueError(
                f"probability row {row_index} has {len(values)} values, "
                f"expected {vocab_size}"
            )
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("probability rows must contain finite non-negative values")
        total = math.fsum(values)
        if total <= 0:
            normalized = [1.0 / vocab_size] * vocab_size
        else:
            normalized = [value / total for value in values]
        normalized_rows.append(normalized)
        cdfs.append(build_cumfreq(normalized))
    return normalized_rows, cdfs


def encode_grouped(
    tokens: Sequence[int],
    schedule: MaskSchedule,
    probability_fn: ProbabilityFn,
    *,
    vocab_size: int = 256,
    mask_token_id: int | None = None,
) -> GroupedEncoding:
    """Encode tokens with exactly one probability callback per schedule group."""
    tokens = _validate_tokens(tokens, schedule, vocab_size)
    mask_token_id = vocab_size if mask_token_id is None else mask_token_id
    _validate_mask_token_id(mask_token_id, vocab_size)

    state = [mask_token_id] * schedule.num_tokens
    encoder = ArithmeticEncoder()
    model_nll = 0.0
    cdf_nll = 0.0
    for group_index, positions in enumerate(schedule.groups):
        probabilities, cdfs = _group_tables(
            probability_fn,
            state,
            group_index,
            positions,
            vocab_size,
        )
        # Build every table before revealing any symbol from this group.
        for position, probs, cdf in zip(positions, probabilities, cdfs):
            symbol = tokens[position]
            model_nll -= math.log2(max(probs[symbol], 1e-300))
            frequency = cdf[symbol + 1] - cdf[symbol]
            cdf_nll -= math.log2(frequency / cdf[-1])
            encoder.encode(symbol, cdf)
        for position in positions:
            state[position] = tokens[position]

    return GroupedEncoding(
        bits=tuple(encoder.finish()),
        teacher_forced_nll_bits=model_nll,
        quantized_cdf_nll_bits=cdf_nll,
        forward_calls=schedule.num_groups,
    )


def decode_grouped(
    bits: Sequence[int],
    schedule: MaskSchedule,
    probability_fn: ProbabilityFn,
    *,
    vocab_size: int = 256,
    mask_token_id: int | None = None,
) -> GroupedDecoding:
    """Decode one complete grouped stream with frozen within-group CDFs."""
    if not isinstance(vocab_size, int) or isinstance(vocab_size, bool) or vocab_size < 2:
        raise ValueError("vocab_size must be an integer >= 2")
    mask_token_id = vocab_size if mask_token_id is None else mask_token_id
    _validate_mask_token_id(mask_token_id, vocab_size)
    state = [mask_token_id] * schedule.num_tokens
    decoder = ArithmeticDecoder(bits)

    for group_index, positions in enumerate(schedule.groups):
        _, cdfs = _group_tables(
            probability_fn,
            state,
            group_index,
            positions,
            vocab_size,
        )
        decoded_group = [decoder.decode(cdf) for cdf in cdfs]
        for position, symbol in zip(positions, decoded_group):
            state[position] = symbol
    return GroupedDecoding(tokens=tuple(state), forward_calls=schedule.num_groups)


def make_model_probability_fn(model, *, coarse_tokens=None) -> ProbabilityFn:
    """Adapt MaskedIGPT or CCMDLM to the pure grouped codec callback contract."""
    import torch

    from .probability import codec_probabilities

    if model.training:
        raise ValueError("grouped codec requires model.eval()")
    try:
        device = next(model.parameters()).device
    except StopIteration as exc:
        raise ValueError("model must have parameters") from exc

    coarse = None
    if coarse_tokens is not None:
        coarse = torch.as_tensor(coarse_tokens, dtype=torch.long, device=device)
        if coarse.ndim == 1:
            coarse = coarse.unsqueeze(0)
        if coarse.ndim != 2 or coarse.shape[0] != 1:
            raise ValueError("coarse_tokens must describe exactly one sample")

    def probability_fn(state, group_index, positions):
        del group_index
        input_tokens = torch.tensor(
            state,
            dtype=torch.long,
            device=device,
        ).unsqueeze(0)
        device_type = "cuda" if device.type == "cuda" else "cpu"
        with torch.no_grad(), torch.amp.autocast(device_type=device_type, enabled=False):
            if coarse is None:
                output = model.forward_masked(input_tokens)
            else:
                output = model.forward_masked(
                    coarse_tokens=coarse,
                    fine_input_tokens=input_tokens,
                )
            logits = output["logits"]
            if logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[1] != len(state):
                raise ValueError("masked model returned logits with an invalid shape")
            if logits.dtype != torch.float32:
                raise ValueError(
                    "grouped codec protocol requires float32 logits, "
                    f"got {logits.dtype}"
                )
            selected = logits[0, list(positions)]
            return codec_probabilities(selected).cpu().tolist()

    return probability_fn


def verify_grouped_codec(
    tokens: Sequence[int],
    schedule: MaskSchedule,
    probability_fn: ProbabilityFn,
    *,
    identity: dict[str, Any],
    height: int,
    width: int,
    channels: int,
) -> dict[str, Any]:
    """Run encode/container/decode and report all grouped rate conventions."""
    tokens = _validate_tokens(tokens, schedule, vocab_size=256)
    if height * width * channels != schedule.num_tokens:
        raise ValueError("image geometry does not match schedule.num_tokens")
    if identity.get("schedule_sha256") != schedule.sha256:
        raise ValueError("codec identity is bound to a different mask schedule")
    expected_protocol_hash = canonical_sha256(grouped_codec_protocol())
    if identity.get("codec_protocol_sha256") != expected_protocol_hash:
        raise ValueError("codec identity is not bound to grouped codec protocol v1")

    encoded = encode_grouped(tokens, schedule, probability_fn)
    source_bytes = bytes(tokens)
    blob = build_container_bytes(
        False,
        height,
        channels,
        [],
        list(encoded.bits),
        identity=identity,
        source_rgb_sha256=sha256_rgb_bytes(source_bytes),
        width=width,
    )
    dual, parsed_height, parsed_channels, coarse_bits, fine_bits = read_container_bytes(
        blob,
        expected_identity=identity,
    )
    if dual or coarse_bits or parsed_height != height or parsed_channels != channels:
        raise ValueError("grouped container metadata is inconsistent")
    decoded = decode_grouped(fine_bits, schedule, probability_fn)
    decoded_bytes = bytes(decoded.tokens)
    parsed = parse_container(blob, expected_identity=identity)
    if parsed.width != width:
        raise ValueError("grouped container width is inconsistent")
    validate_decoded_rgb(decoded_bytes, parsed.metadata)

    denominator = schedule.num_tokens
    payload_bits = len(encoded.bits)
    packed_payload_bits = parsed.packed_payload_size * 8
    file_bits = len(blob) * 8
    return {
        "pixel_exact": decoded.tokens == tokens,
        "rgb_checksum_verified": True,
        "teacher_forced_nll_bits": encoded.teacher_forced_nll_bits,
        "schedule_nll_bits": encoded.teacher_forced_nll_bits,
        "quantized_cdf_nll_bits": encoded.quantized_cdf_nll_bits,
        "payload_bits": payload_bits,
        "packed_payload_bits": packed_payload_bits,
        "file_bits": file_bits,
        "teacher_forced_nll_bpd": encoded.teacher_forced_nll_bits / denominator,
        "schedule_nll_bpd": encoded.teacher_forced_nll_bits / denominator,
        "quantized_cdf_nll_bpd": encoded.quantized_cdf_nll_bits / denominator,
        "payload_bpd": payload_bits / denominator,
        "packed_payload_bpd": packed_payload_bits / denominator,
        "file_bpd": file_bits / denominator,
        "cdf_quantization_gap_bits": (
            encoded.quantized_cdf_nll_bits - encoded.teacher_forced_nll_bits
        ),
        "coder_overhead_bits": payload_bits - encoded.quantized_cdf_nll_bits,
        "fine_forward_calls_encode": encoded.forward_calls,
        "fine_forward_calls_decode": decoded.forward_calls,
        "fine_forward_calls_total": encoded.forward_calls + decoded.forward_calls,
        "schedule_sha256": schedule.sha256,
        "container_bytes": len(blob),
    }
