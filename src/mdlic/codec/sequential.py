"""Shared sequential autoregressive arithmetic-coding primitives."""

from __future__ import annotations

import math

from .arithmetic import (
    ArithmeticDecoder,
    ArithmeticEncoder,
    FREQ_TOTAL,
    build_cumfreq,
)
from .probability import codec_probabilities


UNIFORM_CUM = list(range(0, FREQ_TOTAL + 1, FREQ_TOTAL // 256))


def logits_from_tokens(igpt, buffer_tokens, coarse_ctx, pos_index):
    """Return fp32 next-token logits for one position in a padded prefix."""
    import torch

    with torch.no_grad():
        hidden, position_ids = igpt._embed_inputs(
            buffer_tokens,
            coarse_ctx=coarse_ctx,
        )
        for block in igpt.blocks:
            hidden = block(hidden, position_ids=position_ids)
        logits = igpt.head(hidden[:, pos_index:pos_index + 1, :])
    return logits.float().squeeze(1)


def probabilities_from_logits(logits_row):
    """Convert fp32 logits to the codec's deterministic fp64 probabilities."""
    return codec_probabilities(logits_row).squeeze(0).tolist()


def encode_sequence(igpt, tokens, coarse_ctx, device, tag, log_every):
    """Encode a token sequence and return bits plus two ideal-rate totals."""
    import torch

    token_count = tokens.shape[1]
    input_length = igpt.seq_len - 1
    encoder = ArithmeticEncoder()
    model_ideal_bits = 8.0
    cdf_ideal_bits = 8.0

    encoder.encode(int(tokens[0, 0].item()), UNIFORM_CUM)
    buffer = torch.zeros((1, input_length), dtype=torch.long, device=device)
    for position in range(1, token_count):
        buffer[0, position - 1] = tokens[0, position - 1]
        logits = logits_from_tokens(
            igpt,
            buffer,
            coarse_ctx,
            position - 1,
        )
        probabilities = probabilities_from_logits(logits)
        symbol = int(tokens[0, position].item())
        model_ideal_bits += -math.log2(max(probabilities[symbol], 1e-300))
        cumulative = build_cumfreq(probabilities)
        cdf_ideal_bits += -math.log2(
            (cumulative[symbol + 1] - cumulative[symbol]) / cumulative[-1]
        )
        encoder.encode(symbol, cumulative)
        if log_every and position % log_every == 0:
            print(f"    [{tag} encode] {position}/{token_count - 1}")
    return encoder.finish(), model_ideal_bits, cdf_ideal_bits


def encode_sequence_iter(igpt, tokens, coarse_ctx, device):
    """Yield encode progress, then return the completed arithmetic bit list."""
    import torch

    token_count = tokens.shape[1]
    input_length = igpt.seq_len - 1
    encoder = ArithmeticEncoder()
    encoder.encode(int(tokens[0, 0].item()), UNIFORM_CUM)
    buffer = torch.zeros((1, input_length), dtype=torch.long, device=device)
    for position in range(1, token_count):
        buffer[0, position - 1] = tokens[0, position - 1]
        logits = logits_from_tokens(
            igpt,
            buffer,
            coarse_ctx,
            position - 1,
        )
        probabilities = probabilities_from_logits(logits)
        encoder.encode(
            int(tokens[0, position].item()),
            build_cumfreq(probabilities),
        )
        yield position, token_count - 1
    return encoder.finish()


def decode_sequence_iter(igpt, bits, token_count, coarse_ctx, device):
    """Yield decode progress, then return the reconstructed token tensor."""
    import torch

    input_length = igpt.seq_len - 1
    decoder = ArithmeticDecoder(bits)
    output = torch.zeros((1, token_count), dtype=torch.long, device=device)
    output[0, 0] = decoder.decode(UNIFORM_CUM)
    buffer = torch.zeros((1, input_length), dtype=torch.long, device=device)
    for position in range(1, token_count):
        buffer[0, position - 1] = output[0, position - 1]
        logits = logits_from_tokens(
            igpt,
            buffer,
            coarse_ctx,
            position - 1,
        )
        probabilities = probabilities_from_logits(logits)
        output[0, position] = decoder.decode(build_cumfreq(probabilities))
        yield position, token_count - 1
    return output


def decode_sequence(igpt, bits, token_count, coarse_ctx, device, tag, log_every):
    """Decode a token sequence while printing optional CLI progress."""
    generator = decode_sequence_iter(
        igpt,
        bits,
        token_count,
        coarse_ctx,
        device,
    )
    try:
        while True:
            position, total = next(generator)
            if log_every and position % log_every == 0:
                print(f"    [{tag} decode] {position}/{total}")
    except StopIteration as stop:
        return stop.value


def detokenize(tokens, image_size: int, channels: int):
    """Convert one pixel-first token sequence to a CHW uint8 tensor."""
    import torch

    tensor = tokens.view(1, image_size, image_size, channels)
    return tensor.permute(0, 3, 1, 2).contiguous()[0].to(torch.uint8)
