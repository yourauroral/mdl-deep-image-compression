"""Numerical probability helpers shared by evaluation and arithmetic codecs."""

from __future__ import annotations

import torch


CODEC_LOGITS_DTYPE = torch.float32
CODEC_PROBABILITY_DTYPE = torch.float64
DEFAULT_TOKEN_CHUNK_SIZE = 1024


def codec_probabilities(logits: torch.Tensor) -> torch.Tensor:
    """Return the codec's fp64 softmax probabilities from fp32 logits."""
    if logits.dtype != CODEC_LOGITS_DTYPE:
        raise ValueError(
            "codec probability protocol requires float32 logits, "
            f"got {logits.dtype}"
        )
    return torch.softmax(logits.to(CODEC_PROBABILITY_DTYPE), dim=-1)


def codec_target_nll_per_image(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    token_chunk_size: int = DEFAULT_TOKEN_CHUNK_SIZE,
) -> torch.Tensor:
    """Compute mean target NLL per image under the codec's fp64 softmax.

    Token chunks bound the temporary fp64 tensor for ImageNet64 while retaining
    the exact logits/softmax dtype convention used by the arithmetic codec.
    """
    if logits.dtype != CODEC_LOGITS_DTYPE:
        raise ValueError(
            "codec probability protocol requires float32 logits, "
            f"got {logits.dtype}"
        )
    if logits.ndim != 3 or targets.ndim != 2:
        raise ValueError("expected logits (B,T,V) and targets (B,T)")
    if logits.shape[:2] != targets.shape:
        raise ValueError(
            "logits/targets shape mismatch: "
            f"{tuple(logits.shape[:2])} != {tuple(targets.shape)}"
        )
    if token_chunk_size < 1:
        raise ValueError("token_chunk_size must be >= 1")

    batch_size, num_tokens, _ = logits.shape
    if num_tokens < 1:
        raise ValueError("targets must contain at least one predicted token")
    nll_sum = torch.zeros(
        batch_size,
        dtype=CODEC_PROBABILITY_DTYPE,
        device=logits.device,
    )
    for start in range(0, num_tokens, token_chunk_size):
        stop = min(start + token_chunk_size, num_tokens)
        rows = logits[:, start:stop].to(CODEC_PROBABILITY_DTYPE)
        target_rows = targets[:, start:stop].unsqueeze(-1)
        target_logits = rows.gather(-1, target_rows).squeeze(-1)
        nll_sum += (torch.logsumexp(rows, dim=-1) - target_logits).sum(dim=1)
    return nll_sum / num_tokens


def codec_aligned_score_metadata() -> dict[str, object]:
    """Describe the fast teacher-forced score's numerical convention."""
    return {
        "name": "mdlic-codec-aligned-teacher-forced-nll-v1",
        "forward": "teacher-forced-full-sequence",
        "autocast": False,
        "logits_dtype": "float32",
        "softmax_dtype": "float64",
        "actual_arithmetic_coding": False,
    }


def diagnostic_score_metadata(amp_dtype: torch.dtype | None) -> dict[str, object]:
    """Describe a fast diagnostic score that is not codec-numerics aligned."""
    dtype_name = None if amp_dtype is None else str(amp_dtype).removeprefix("torch.")
    return {
        "name": "mdlic-diagnostic-model-nll-v1",
        "forward": "teacher-forced-full-sequence",
        "autocast": amp_dtype is not None,
        "amp_dtype": dtype_name,
        "logits_dtype": "float32-after-forward-cast",
        "softmax_dtype": "float32",
        "actual_arithmetic_coding": False,
    }
