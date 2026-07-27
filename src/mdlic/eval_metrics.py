"""Shared per-image likelihood metrics for training and standalone evaluation."""

import torch
import torch.nn.functional as F

from .codec.probability import codec_target_nll_per_image
from .rate import dual_stream_bpd, single_stream_bpd


def tokenize_targets(x: torch.Tensor) -> torch.Tensor:
    """Tokenize RGB-like tensors and return the ``T - 1`` NTP targets."""
    tokens = (x.clamp(0, 1) * 255).round().long()
    tokens = tokens.permute(0, 2, 3, 1).reshape(x.size(0), -1)
    return tokens[:, 1:]


def ce_per_image(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean next-token cross entropy per image in nats."""
    batch_size = targets.size(0)
    vocab_size = logits.size(-1)
    ce_tokens = F.cross_entropy(
        logits.float().reshape(-1, vocab_size),
        targets.reshape(-1),
        reduction="none",
    )
    return ce_tokens.view(batch_size, -1).mean(dim=1)


def per_image_rate_components(
    model,
    x: torch.Tensor,
    out: dict,
    *,
    codec_numerics: bool = False,
) -> dict[str, torch.Tensor]:
    """Compute per-image CE components and bpd, including first-token priors."""
    raw_model = model.module if hasattr(model, "module") else model
    raw_model = getattr(raw_model, "_orig_mod", raw_model)
    ce_fn = codec_target_nll_per_image if codec_numerics else ce_per_image

    target_f = tokenize_targets(x).to(out["logits"].device)
    ce_f = ce_fn(out["logits"], target_f)
    if "ce_loss_coarse" not in out or out["ce_loss_coarse"] is None:
        return {
            "bpd": single_stream_bpd(ce_f, raw_model.seq_len),
            "ce_fine": ce_f,
        }

    logits_c = out.get("logits_coarse")
    if logits_c is None:
        raise KeyError("CC-iGPT output must include logits_coarse for per-image metrics")
    x_c = raw_model._coarse_input(x.clamp(0, 1).to(torch.float32))
    target_c = tokenize_targets(x_c).to(logits_c.device)
    ce_c = ce_fn(logits_c, target_c)
    return {
        "bpd": dual_stream_bpd(
            ce_c, raw_model.coarse.seq_len,
            ce_f, raw_model.fine.seq_len,
        ),
        "ce_coarse": ce_c,
        "ce_fine": ce_f,
    }


def per_image_bpd(
    model,
    x: torch.Tensor,
    out: dict,
    *,
    codec_numerics: bool = False,
) -> torch.Tensor:
    """Compute ideal model bpd per image, including each stream's first token."""
    return per_image_rate_components(
        model, x, out, codec_numerics=codec_numerics,
    )["bpd"]
