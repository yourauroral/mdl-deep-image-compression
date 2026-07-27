"""Bidirectional masked-token variant of iGPT for grouped likelihoods."""

from __future__ import annotations

import torch
import torch.nn as nn

from .igpt import IGPT, _categorical_ce_zloss


class MaskedIGPT(IGPT):
    """iGPT backbone with an explicit input-only MASK embedding.

    The categorical output remains ``vocab_size``-way and tied to the original
    token embedding.  MASK has id ``vocab_size`` on input, but is never a target
    class.  Grouped training and coding must call ``forward_masked``; the
    inherited causal entry point is blocked to prevent accidental AR training.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask_token_id = self.vocab_size
        self.mask_embedding = nn.Parameter(torch.empty(self.d_model))
        self.mask_ratio_proj = nn.Linear(1, self.d_model)
        nn.init.normal_(self.mask_embedding, mean=0.0, std=0.02)
        nn.init.normal_(self.mask_ratio_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.mask_ratio_proj.bias)

        positions = torch.arange(self.seq_len)
        self.register_buffer(
            "_masked_channel_indices",
            positions % self.in_channels,
            persistent=False,
        )
        self.register_buffer(
            "_masked_position_ids",
            positions // self.in_channels,
            persistent=False,
        )

    def _embed_masked_inputs(
        self,
        input_tokens: torch.Tensor,
        *,
        coarse_ctx: torch.Tensor | None,
        mask_ratio,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if input_tokens.dtype != torch.long:
            raise TypeError("masked input_tokens must have dtype torch.long")
        if input_tokens.ndim != 2 or input_tokens.shape[1] != self.seq_len:
            raise ValueError(
                f"masked input_tokens must have shape (B,{self.seq_len}), "
                f"got {tuple(input_tokens.shape)}"
            )
        if ((input_tokens < 0) | (input_tokens > self.mask_token_id)).any():
            raise ValueError(
                f"masked inputs must be in [0,{self.mask_token_id}]"
            )

        masked_positions = input_tokens.eq(self.mask_token_id)
        safe_tokens = input_tokens.masked_fill(masked_positions, 0)
        hidden = self.token_embed(safe_tokens)
        mask_value = self.mask_embedding.to(dtype=hidden.dtype).view(1, 1, -1)
        hidden = torch.where(masked_positions.unsqueeze(-1), mask_value, hidden)

        if coarse_ctx is not None:
            if coarse_ctx.shape != hidden.shape:
                raise ValueError(
                    f"coarse_ctx shape {tuple(coarse_ctx.shape)} != "
                    f"hidden {tuple(hidden.shape)}"
                )
            hidden = hidden + coarse_ctx

        hidden = hidden + self.channel_embed(
            self._masked_channel_indices,
        ).unsqueeze(0)

        batch_size = input_tokens.shape[0]
        if mask_ratio is None:
            ratio = masked_positions.float().mean(dim=1, keepdim=True)
        else:
            ratio = torch.as_tensor(
                mask_ratio,
                device=input_tokens.device,
                dtype=torch.float32,
            )
            if ratio.ndim == 0:
                ratio = ratio.expand(batch_size).unsqueeze(1)
            elif ratio.ndim == 1 and ratio.shape[0] == batch_size:
                ratio = ratio.unsqueeze(1)
            elif ratio.shape != (batch_size, 1):
                raise ValueError(
                    f"mask_ratio must be scalar, (B,), or (B,1); got {tuple(ratio.shape)}"
                )
        if not torch.isfinite(ratio).all() or ((ratio < 0) | (ratio > 1)).any():
            raise ValueError("mask_ratio values must be finite and in [0, 1]")
        ratio_embedding = self.mask_ratio_proj(ratio.to(dtype=hidden.dtype))
        hidden = hidden + ratio_embedding.unsqueeze(1)
        return hidden, self._masked_position_ids, masked_positions

    def forward(self, *args, **kwargs):
        del args, kwargs
        raise RuntimeError(
            "MaskedIGPT requires forward_masked(); use IGPT for causal AR forward"
        )

    def forward_masked(
        self,
        input_tokens: torch.Tensor,
        *,
        target_tokens: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
        coarse_ctx: torch.Tensor | None = None,
        mask_ratio=None,
        z_loss_weight: float = 1e-4,
    ) -> dict[str, torch.Tensor | int | None]:
        """Run one full-length bidirectional masked-token forward pass.

        ``loss_mask`` identifies the current schedule group.  It must be a
        subset of input MASK positions, so future groups can remain masked
        without contributing to the current group's conditional CE.
        """
        hidden, position_ids, masked_positions = self._embed_masked_inputs(
            input_tokens,
            coarse_ctx=coarse_ctx,
            mask_ratio=mask_ratio,
        )
        for block in self.blocks:
            hidden = block(
                hidden,
                position_ids=position_ids,
                is_causal=False,
            )
        logits = self.head(hidden)

        result: dict[str, torch.Tensor | int | None] = {
            "loss": None,
            "masked_ce_loss": None,
            "masked_nll_nats": None,
            "num_loss_tokens": 0,
            "logits": logits,
            "masked_positions": masked_positions,
        }
        if target_tokens is None:
            if loss_mask is not None:
                raise ValueError("loss_mask requires target_tokens")
            return result
        if target_tokens.shape != input_tokens.shape or target_tokens.dtype != torch.long:
            raise ValueError(
                "target_tokens must be torch.long with the same shape as input_tokens"
            )

        if loss_mask is None:
            loss_mask = masked_positions
        if loss_mask.shape != input_tokens.shape or loss_mask.dtype != torch.bool:
            raise ValueError("loss_mask must be bool with the same shape as input_tokens")
        if (loss_mask & ~masked_positions).any():
            raise ValueError("loss_mask must be a subset of MASK input positions")
        num_loss_tokens = int(loss_mask.sum().item())
        if num_loss_tokens == 0:
            raise ValueError("loss_mask must select at least one token")

        selected_targets = target_tokens[loss_mask]
        if ((selected_targets < 0) | (selected_targets >= self.vocab_size)).any():
            raise ValueError(f"selected targets must be in [0,{self.vocab_size})")
        loss, ce_loss, _ = _categorical_ce_zloss(
            logits[loss_mask],
            selected_targets,
            z_loss_weight,
        )
        result.update({
            "loss": loss,
            "masked_ce_loss": ce_loss,
            "masked_nll_nats": ce_loss * num_loss_tokens,
            "num_loss_tokens": num_loss_tokens,
        })
        return result

    def extra_repr(self) -> str:
        return (
            f"mask_token_id={self.mask_token_id}, seq_len={self.seq_len}, "
            f"vocab_size={self.vocab_size}"
        )
