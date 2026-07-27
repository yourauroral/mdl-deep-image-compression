"""Public model construction helpers shared by training and inference tools."""

from __future__ import annotations

from collections.abc import Mapping

from .models.cc_igpt import CCIGPT
from .models.igpt import IGPT


def _shared_igpt_kwargs(model_config: Mapping[str, object]) -> dict[str, object]:
    return {
        "in_channels": model_config["in_channels"],
        "vocab_size": model_config["vocab_size"],
        "dropout": model_config["dropout"],
        "activation_checkpointing": model_config.get(
            "activation_checkpointing", False
        ),
        "drop_path": model_config.get("drop_path", 0.0),
    }


def build_igpt_from_config(model_config: Mapping[str, object], device) -> IGPT:
    """Build the single-scale iGPT described by a model config mapping."""
    model = IGPT(
        image_size=model_config["image_size"],
        d_model=model_config["d_model"],
        N=model_config["N"],
        h=model_config["h"],
        d_ff=model_config["d_ff"],
        **_shared_igpt_kwargs(model_config),
    )
    return model.to(device)


def build_ccigpt_from_config(model_config: Mapping[str, object], device) -> CCIGPT:
    """Build the coarse-conditioned iGPT described by a model config mapping."""
    model = CCIGPT(
        image_size=model_config["image_size"],
        pool_factor=model_config["pool_factor"],
        fine_d_model=model_config["d_model"],
        fine_N=model_config["N"],
        fine_h=model_config["h"],
        fine_d_ff=model_config["d_ff"],
        coarse_d_model=model_config["coarse_d_model"],
        coarse_N=model_config["coarse_N"],
        coarse_h=model_config["coarse_h"],
        coarse_d_ff=model_config["coarse_d_ff"],
        coarse_in_channels=model_config.get("coarse_in_channels"),
        **_shared_igpt_kwargs(model_config),
    )
    return model.to(device)


def build_model_from_config(model_config: Mapping[str, object], device):
    """Build a supported AR model, rejecting unknown model types explicitly."""
    model_type = model_config.get("type", "igpt")
    if model_type == "igpt":
        return build_igpt_from_config(model_config, device)
    if model_type == "ccigpt":
        return build_ccigpt_from_config(model_config, device)
    raise ValueError(
        f"unsupported AR model type {model_type!r}; expected 'igpt' or 'ccigpt'"
    )
