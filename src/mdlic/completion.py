"""Autoregressive image-completion primitives shared by CLI and demo."""

from __future__ import annotations

import math


def row_aligned_keep_tokens(
    keep_frac: float,
    *,
    image_size: int,
    channels: int,
    total_tokens: int,
) -> int:
    """Map a requested fraction to the nearest non-empty full raster row."""
    if (
        isinstance(keep_frac, bool)
        or not isinstance(keep_frac, (int, float))
        or not math.isfinite(float(keep_frac))
        or not 0.0 < float(keep_frac) <= 1.0
    ):
        raise ValueError(f"keep_frac must be finite and in (0, 1], got {keep_frac!r}")
    if image_size < 1 or channels < 1:
        raise ValueError("image_size and channels must be positive")

    tokens_per_row = image_size * channels
    expected_tokens = image_size * tokens_per_row
    if total_tokens != expected_tokens:
        raise ValueError(
            f"total_tokens={total_tokens} does not describe an "
            f"{image_size}x{image_size}x{channels} raster"
        )
    keep_rows = int(math.floor(float(keep_frac) * image_size + 0.5))
    keep_rows = max(1, min(keep_rows, image_size))
    return keep_rows * tokens_per_row


def _logits_at(igpt, buffer, coarse_ctx, position):
    import torch

    with torch.no_grad():
        hidden, position_ids = igpt._embed_inputs(
            buffer,
            coarse_ctx=coarse_ctx,
        )
        for block in igpt.blocks:
            hidden = block(hidden, position_ids=position_ids)
        logits = igpt.head(hidden[:, position:position + 1, :])
    return logits.float().squeeze(1)


def _sample(logits_row, temperature: float, top_k: int) -> int:
    import torch

    logits = logits_row.squeeze(0).double()
    if temperature <= 0:
        return int(logits.argmax().item())
    logits = logits / temperature
    if top_k and top_k < logits.numel():
        threshold = torch.topk(logits, top_k).values[-1]
        logits = logits.masked_fill(logits < threshold, float("-inf"))
    probabilities = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probabilities, 1).item())


def complete_image(model, model_type, image, keep_frac, temperature, top_k, device):
    """Complete one BCHW image and return original, masked, and sampled CHW tensors."""
    import torch

    image = image.to(device).clamp(0, 1).float()
    igpt = model.fine if model_type == "ccigpt" else model
    channels, image_size = igpt.in_channels, igpt.image_size
    token_count = igpt.seq_len
    input_length = token_count - 1

    with torch.amp.autocast(device_type=device.type, enabled=False):
        coarse_ctx = None
        if model_type == "ccigpt":
            coarse_image = model._coarse_input(image)
            coarse_tokens = model.coarse._tokenize(coarse_image)
            coarse_ctx = model.ctx_alpha * model._compute_coarse_ctx(coarse_tokens)

        tokens = igpt._tokenize(image).clone()
        keep = row_aligned_keep_tokens(
            keep_frac,
            image_size=image_size,
            channels=channels,
            total_tokens=token_count,
        )
        generated = tokens.clone()
        buffer = torch.zeros(
            (1, input_length),
            dtype=torch.long,
            device=device,
        )
        for position in range(keep, token_count):
            buffer[0, :position] = generated[0, :position]
            logits = _logits_at(igpt, buffer, coarse_ctx, position - 1)
            generated[0, position] = _sample(logits, temperature, top_k)

    def detokenize(value):
        value = value.view(1, image_size, image_size, channels)
        return value.permute(0, 3, 1, 2).contiguous()[0].to(torch.uint8)

    original = (image.clamp(0, 1) * 255).round().to(torch.uint8)[0]
    completed = detokenize(generated)
    masked_tokens = tokens.clone()
    masked_tokens[0, keep:] = 128
    masked = detokenize(masked_tokens)
    return original.cpu(), masked.cpu(), completed.cpu()
