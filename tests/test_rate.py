import math

import pytest
import torch

from mdlic.rate import (
    dual_stream_bpd,
    ideal_stream_bits,
    num_model_predictions,
    single_stream_bpd,
)


def test_ideal_stream_bits_counts_only_predicted_tokens():
    ce = math.log(2.0)  # one bit per predicted token
    assert ideal_stream_bits(ce, 5) == pytest.approx(8.0 + 4.0)


def test_model_prediction_steps_exclude_each_streams_first_token():
    assert num_model_predictions(3072) == 3071
    assert num_model_predictions(64, 3072) == 63 + 3071


def test_single_stream_bpd_includes_uniform_first_token():
    ce = math.log(2.0)
    assert single_stream_bpd(ce, 5) == pytest.approx(12.0 / 5.0)


def test_dual_stream_bpd_uses_fine_dimensions_as_denominator():
    ce_c = 2.0 * math.log(2.0)
    ce_f = 1.0 * math.log(2.0)
    # coarse: 8 + 2 predicted tokens * 2 bits; fine: 8 + 4 * 1 bit
    assert dual_stream_bpd(ce_c, 3, ce_f, 5) == pytest.approx(24.0 / 5.0)


def test_rate_helpers_preserve_tensor_gradients():
    ce_c = torch.tensor(2.0, requires_grad=True)
    ce_f = torch.tensor(3.0, requires_grad=True)
    bpd = dual_stream_bpd(ce_c, 3, ce_f, 5)
    bpd.backward()
    assert ce_c.grad.item() == pytest.approx(2 / math.log(2.0) / 5)
    assert ce_f.grad.item() == pytest.approx(4 / math.log(2.0) / 5)


@pytest.mark.parametrize("bad", [0, -1])
def test_rate_helpers_reject_empty_streams(bad):
    with pytest.raises(ValueError):
        ideal_stream_bits(1.0, bad)
