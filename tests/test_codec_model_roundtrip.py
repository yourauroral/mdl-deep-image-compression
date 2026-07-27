import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from mdlic.codec.probability import (
    codec_probabilities,
    codec_target_nll_per_image,
)
from scripts.verify_lossless import _verify_image
from scripts.evaluate import evaluate_model
from mdlic.models.igpt import IGPT


def test_codec_target_nll_matches_fp64_softmax_in_token_chunks():
    torch.manual_seed(4)
    logits = torch.randn(2, 5, 7, dtype=torch.float32)
    targets = torch.randint(0, 7, (2, 5))

    actual = codec_target_nll_per_image(
        logits, targets, token_chunk_size=2,
    )
    probabilities = codec_probabilities(logits)
    expected = -torch.log(
        probabilities.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    ).mean(dim=1)

    assert actual.dtype == torch.float64
    assert torch.allclose(actual, expected, rtol=0.0, atol=1e-15)


def test_codec_probability_helper_rejects_non_fp32_logits():
    with pytest.raises(ValueError, match="float32 logits"):
        codec_probabilities(torch.zeros(1, 4, dtype=torch.float64))


def test_codec_aligned_evaluator_matches_tiny_sequential_codec_nll():
    torch.manual_seed(8)
    model = IGPT(
        image_size=2,
        in_channels=1,
        vocab_size=256,
        d_model=8,
        N=1,
        h=1,
        d_ff=16,
        dropout=0.0,
    ).eval()
    image = torch.rand(1, 1, 2, 2)
    loader = DataLoader(
        TensorDataset(image, torch.zeros(1, dtype=torch.long)),
        batch_size=1,
    )

    mean, _, _, extras = evaluate_model(
        model,
        loader,
        torch.device("cpu"),
        codec_numerics=True,
    )
    codec_result = _verify_image(
        model, "igpt", image, torch.device("cpu"), log_every=0,
    )

    assert mean == pytest.approx(codec_result["ideal_bpd"], abs=1e-6)
    assert extras["score_numerics"]["logits_dtype"] == "float32"
    assert extras["score_numerics"]["softmax_dtype"] == "float64"


def test_tiny_single_stream_codec_reports_all_rate_conventions():
    torch.manual_seed(12)
    model = IGPT(
        image_size=2,
        in_channels=1,
        vocab_size=256,
        d_model=8,
        N=1,
        h=1,
        d_ff=16,
        dropout=0.0,
    ).eval()
    x = torch.rand(1, 1, 2, 2)

    result = _verify_image(
        model, "igpt", x, torch.device("cpu"), log_every=0,
    )

    assert result["pixel_exact"] is True
    assert result["ref_bpd"] == pytest.approx(result["ideal_bpd"], abs=1e-5)
    assert result["payload_bpd"] == result["achieved_bpd"]
    assert result["packed_payload_bpd"] >= result["payload_bpd"]
    assert result["file_bpd"] > result["packed_payload_bpd"]
    assert result["container_version"] == 2
    assert result["integrity_verified"] is True
    assert result["decoded_rgb_checksum_verified"] is True
    assert result["container_bytes"] > 64
    assert abs(result["coder_overhead_bpd"] * result["N_f"]) < 4.0
