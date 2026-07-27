"""Rate-accounting helpers shared by models, evaluation, and the codec.

Cross entropy is averaged over the ``T - 1`` next-token predictions, while a
complete discrete stream also contains the first token.  The codec encodes
that first token with a uniform 256-way prior, which costs exactly 8 bits.
Keeping this accounting in one module prevents model NLL and bitstream metrics
from silently using different conventions.
"""

import math


LOG_2 = math.log(2.0)
DEFAULT_FIRST_TOKEN_BITS = 8.0


def _validate_num_tokens(num_tokens: int) -> None:
    if not isinstance(num_tokens, int) or isinstance(num_tokens, bool):
        raise TypeError(f"num_tokens must be an int, got {type(num_tokens).__name__}")
    if num_tokens < 1:
        raise ValueError(f"num_tokens must be >= 1, got {num_tokens}")


def num_model_predictions(*stream_num_tokens: int) -> int:
    """Return the number of next-token model calls across independent streams.

    Each stream's first token uses its explicit prior and therefore does not
    require a model prediction.  A dual-stream codec with lengths ``Nc`` and
    ``Nf`` performs ``(Nc - 1) + (Nf - 1)`` predictions.
    """
    if not stream_num_tokens:
        raise ValueError("at least one stream length is required")
    for num_tokens in stream_num_tokens:
        _validate_num_tokens(num_tokens)
    return sum(num_tokens - 1 for num_tokens in stream_num_tokens)


def ideal_stream_bits(
    mean_ce_nats,
    num_tokens: int,
    first_token_bits: float = DEFAULT_FIRST_TOKEN_BITS,
):
    """Return ideal bits for one stream from mean next-token CE in nats.

    ``mean_ce_nats`` may be a Python number or a scalar/per-image tensor.  The
    first-token term is constant and therefore changes reported rate without
    changing gradients.
    """
    _validate_num_tokens(num_tokens)
    if first_token_bits < 0:
        raise ValueError("first_token_bits must be non-negative")
    return first_token_bits + mean_ce_nats * (num_tokens - 1) / LOG_2


def single_stream_bpd(mean_ce_nats, num_tokens: int):
    """Ideal model bits per original token for a single AR stream."""
    return ideal_stream_bits(mean_ce_nats, num_tokens) / num_tokens


def dual_stream_bpd(
    coarse_mean_ce_nats,
    coarse_num_tokens: int,
    fine_mean_ce_nats,
    fine_num_tokens: int,
):
    """Ideal bpd for an independently coded coarse stream plus the fine image.

    The denominator is the number of original fine-image sub-pixels.  Both
    streams have their own uniformly coded first token.
    """
    coarse_bits = ideal_stream_bits(coarse_mean_ce_nats, coarse_num_tokens)
    fine_bits = ideal_stream_bits(fine_mean_ce_nats, fine_num_tokens)
    return (coarse_bits + fine_bits) / fine_num_tokens
