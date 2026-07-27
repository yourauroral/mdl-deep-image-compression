import json
import math

import pytest
import torch

from mdlic.codec.container import in_memory_model_identity
from mdlic.codec.grouped import (
    decode_grouped,
    encode_grouped,
    grouped_codec_protocol,
    make_model_probability_fn,
    verify_grouped_codec,
)
from mdlic.masking_scheduler import MaskSchedule
from mdlic.models.masked_igpt import MaskedIGPT


def _uniform_probability_fn(records):
    def predict(state, group_index, positions):
        records.append((state, group_index, positions))
        return [[1.0] * 256 for _ in positions]
    return predict


def test_group_probabilities_are_frozen_until_the_whole_group_is_decoded():
    schedule = MaskSchedule.raster_chunks(6, 3)
    tokens = (10, 11, 12, 13, 14, 15)
    encode_records = []
    encoded = encode_grouped(tokens, schedule, _uniform_probability_fn(encode_records))
    decode_records = []
    decoded = decode_grouped(
        encoded.bits,
        schedule,
        _uniform_probability_fn(decode_records),
    )

    expected_states = [
        (256, 256, 256, 256, 256, 256),
        (10, 11, 256, 256, 256, 256),
        (10, 11, 12, 13, 256, 256),
    ]
    assert [record[0] for record in encode_records] == expected_states
    assert [record[0] for record in decode_records] == expected_states
    assert decoded.tokens == tokens
    assert encoded.forward_calls == schedule.num_groups
    assert decoded.forward_calls == schedule.num_groups
    assert encoded.teacher_forced_nll_bits == pytest.approx(6 * 8.0)
    assert encoded.quantized_cdf_nll_bits == pytest.approx(6 * 8.0)


def test_k_equals_t_roundtrips_one_hundred_synthetic_samples():
    schedule = MaskSchedule.raster_chunks(9, 9)

    for sample_index in range(100):
        tokens = tuple((sample_index * 17 + position * 29) % 256 for position in range(9))
        encoded = encode_grouped(tokens, schedule, lambda state, group, positions: [
            [1.0] * 256 for _ in positions
        ])
        decoded = decode_grouped(encoded.bits, schedule, lambda state, group, positions: [
            [1.0] * 256 for _ in positions
        ])
        assert decoded.tokens == tokens


def test_tiny_masked_model_codec_reconciles_nll_payload_and_file_bpd():
    torch.manual_seed(35)
    model = MaskedIGPT(
        image_size=2,
        in_channels=1,
        vocab_size=256,
        d_model=8,
        N=1,
        h=1,
        d_ff=16,
        dropout=0.0,
    ).eval()
    schedule = MaskSchedule.raster_chunks(model.seq_len, 2)
    protocol = grouped_codec_protocol()
    identity = in_memory_model_identity(
        model,
        "masked_igpt",
        torch.device("cpu"),
        schedule=schedule.to_dict(),
        codec_protocol=protocol,
    )
    probability_fn = make_model_probability_fn(model)
    tokens = (12, 34, 56, 78)

    result = verify_grouped_codec(
        tokens,
        schedule,
        probability_fn,
        identity=identity,
        height=2,
        width=2,
        channels=1,
    )

    target = torch.tensor([tokens], dtype=torch.long)
    independent_teacher_forced_bits = 0.0
    for group_index, positions in enumerate(schedule.groups):
        state = schedule.masked_state(tokens, group_index, mask_token_id=256)
        input_tokens = torch.tensor([state], dtype=torch.long)
        loss_mask = torch.zeros_like(input_tokens, dtype=torch.bool)
        loss_mask[:, list(positions)] = True
        output = model.forward_masked(
            input_tokens,
            target_tokens=target,
            loss_mask=loss_mask,
            z_loss_weight=0.0,
        )
        independent_teacher_forced_bits += (
            output["masked_nll_nats"].item() / math.log(2.0)
        )

    assert result["pixel_exact"] is True
    assert result["rgb_checksum_verified"] is True
    assert result["schedule_nll_bits"] == result["teacher_forced_nll_bits"]
    assert result["teacher_forced_nll_bits"] == pytest.approx(
        independent_teacher_forced_bits,
        abs=1e-5,
    )
    assert abs(result["coder_overhead_bits"]) < 4.0
    assert result["packed_payload_bpd"] >= result["payload_bpd"]
    assert result["file_bpd"] > result["packed_payload_bpd"]
    assert result["fine_forward_calls_encode"] == 2
    assert result["fine_forward_calls_decode"] == 2
    assert result["fine_forward_calls_total"] == 4
    assert result["schedule_sha256"] == schedule.sha256
    assert result["container_bytes"] * 8 == result["file_bits"]
    json.dumps(result, allow_nan=False)


def test_grouped_file_rejects_identity_for_a_different_schedule():
    model = MaskedIGPT(
        image_size=2, in_channels=1, d_model=8, N=1, h=1, d_ff=16,
    ).eval()
    schedule = MaskSchedule.raster_chunks(4, 2)
    wrong_schedule = MaskSchedule.raster_chunks(4, 4)
    identity = in_memory_model_identity(
        model,
        "masked_igpt",
        torch.device("cpu"),
        schedule=wrong_schedule.to_dict(),
        codec_protocol=grouped_codec_protocol(),
    )

    with pytest.raises(ValueError, match="different mask schedule"):
        verify_grouped_codec(
            (1, 2, 3, 4),
            schedule,
            make_model_probability_fn(model),
            identity=identity,
            height=2,
            width=2,
            channels=1,
        )


def test_model_probability_fn_rejects_non_float32_logits():
    class ReducedPrecisionModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))

        def forward_masked(self, input_tokens):
            batch, length = input_tokens.shape
            return {
                "logits": torch.zeros(
                    batch,
                    length,
                    256,
                    dtype=torch.bfloat16,
                    device=input_tokens.device,
                )
            }

    probability_fn = make_model_probability_fn(ReducedPrecisionModel().eval())

    with pytest.raises(ValueError, match="requires float32 logits"):
        probability_fn((256, 256), 0, (0, 1))
