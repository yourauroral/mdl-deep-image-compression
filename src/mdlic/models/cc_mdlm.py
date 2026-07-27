"""Coarse-conditioned masked image model with a fixed grouped objective."""

from __future__ import annotations

import torch

from ..masking_scheduler import MaskSchedule
from .cc_igpt import CCIGPT
from .masked_igpt import MaskedIGPT


class CCMDLM(CCIGPT):
    """CC-iGPT coarse stream plus a bidirectional MaskedIGPT fine stream."""

    def __init__(self, *args, **kwargs):
        if "_fine_model_cls" in kwargs:
            raise TypeError("CCMDLM fixes _fine_model_cls to MaskedIGPT")
        super().__init__(*args, _fine_model_cls=MaskedIGPT, **kwargs)

    def forward_masked(
        self,
        *,
        coarse_tokens: torch.Tensor,
        fine_input_tokens: torch.Tensor,
        target_tokens: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
        mask_ratio=None,
        z_loss_weight: float = 1e-4,
    ) -> dict:
        """Run the fine masked model from decoder-available coarse tokens."""
        if coarse_tokens.dtype != torch.long or coarse_tokens.ndim != 2:
            raise ValueError("coarse_tokens must be a rank-2 torch.long tensor")
        if coarse_tokens.shape[1] != self.coarse.seq_len:
            raise ValueError(
                f"coarse token length must be {self.coarse.seq_len}, "
                f"got {coarse_tokens.shape[1]}"
            )
        if fine_input_tokens.shape[0] != coarse_tokens.shape[0]:
            raise ValueError("coarse and fine token batches must match")
        coarse_ctx = self.ctx_alpha * self._compute_coarse_ctx_full(coarse_tokens)
        return self.fine.forward_masked(
            fine_input_tokens,
            target_tokens=target_tokens,
            loss_mask=loss_mask,
            coarse_ctx=coarse_ctx,
            mask_ratio=mask_ratio,
            z_loss_weight=z_loss_weight,
        )

    def forward_group(
        self,
        x: torch.Tensor,
        *,
        schedule: MaskSchedule,
        group_index: int,
        z_loss_weight: float = 1e-4,
    ) -> dict:
        """Teacher-force one fixed schedule group for an unbiased train step.

        Sampling ``group_index`` uniformly makes ``group_weight * group CE`` an
        unbiased estimate of mean fine-token schedule CE.  This is explicitly
        the fixed grouped objective, not an MDLM NELBO.
        """
        if not isinstance(schedule, MaskSchedule):
            raise TypeError("schedule must be a MaskSchedule")
        if schedule.num_tokens != self.fine.seq_len:
            raise ValueError(
                f"schedule has {schedule.num_tokens} tokens, fine model has "
                f"{self.fine.seq_len}"
            )
        revealed = schedule.revealed_before(group_index)
        current_group = schedule.groups[group_index]

        x = x.clamp(0, 1).to(torch.float32)
        x_c_float = self._coarse_input(x)
        out_c = self.coarse(x_c_float, z_loss_weight=z_loss_weight)
        coarse_tokens = self.coarse._tokenize(x_c_float)
        target_tokens = self.fine._tokenize(x)

        fine_input = torch.full_like(target_tokens, self.fine.mask_token_id)
        if revealed:
            revealed_index = torch.tensor(
                revealed,
                device=target_tokens.device,
                dtype=torch.long,
            )
            fine_input[:, revealed_index] = target_tokens[:, revealed_index]
        group_index_tensor = torch.tensor(
            current_group,
            device=target_tokens.device,
            dtype=torch.long,
        )
        loss_mask = torch.zeros_like(target_tokens, dtype=torch.bool)
        loss_mask[:, group_index_tensor] = True

        out_f = self.forward_masked(
            coarse_tokens=coarse_tokens,
            fine_input_tokens=fine_input,
            target_tokens=target_tokens,
            loss_mask=loss_mask,
            z_loss_weight=z_loss_weight,
        )
        group_weight = schedule.num_groups * len(current_group) / schedule.num_tokens
        fixed_group_loss = out_c["loss"] + group_weight * out_f["loss"]
        return {
            "loss": fixed_group_loss,
            "fixed_group_loss": fixed_group_loss,
            "objective": "fixed_group_conditional_ce",
            "ce_loss_coarse": out_c["ce_loss"],
            "masked_group_ce": out_f["masked_ce_loss"],
            "masked_group_nll_nats": out_f["masked_nll_nats"],
            "group_weight": group_weight,
            "group_index": group_index,
            "num_groups": schedule.num_groups,
            "num_group_tokens": len(current_group),
            "bpd": None,
            "logits": out_f["logits"],
            "logits_coarse": out_c["logits"],
        }

    def forward(self, x, *, schedule, group_index, z_loss_weight: float = 1e-4):
        return self.forward_group(
            x,
            schedule=schedule,
            group_index=group_index,
            z_loss_weight=z_loss_weight,
        )
