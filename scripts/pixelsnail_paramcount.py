#!/usr/bin/env python3
"""Audit the PixelSNAIL parameter estimate used by the project tables.

PixelSNAIL does not report a CIFAR-10 parameter count in the paper.  This
script enumerates tensors in the official ``neocxi/pixelsnail-public``
``h12_noup_smallkey`` configuration.  The repository's approximately 91M
figure uses the training CLI default ``nr_resnet=4``.

The formulas include weight-normalization ``V``, ``g``, and bias parameters.
They also reproduce the public PixelCNN++ configuration as a sanity check.
No model checkpoint or third-party package is required.
"""

from __future__ import annotations


def conv_params(taps: int, in_channels: int, out_channels: int, *, weight_norm: bool = True) -> int:
    """Return parameters for a convolution or network-in-network layer."""
    affine_params = 2 * out_channels if weight_norm else out_channels
    return taps * in_channels * out_channels + affine_params


def nin_params(in_channels: int, out_channels: int, *, weight_norm: bool = True) -> int:
    return conv_params(1, in_channels, out_channels, weight_norm=weight_norm)


def gated_resnet_params(
    channels: int,
    taps: int,
    *,
    conditioning_channels: int = 0,
    weight_norm: bool = True,
) -> int:
    """Count a PixelCNN++ gated residual block, including optional input a."""
    first = conv_params(taps, 2 * channels, channels, weight_norm=weight_norm)
    conditioning = (
        nin_params(2 * conditioning_channels, channels, weight_norm=weight_norm)
        if conditioning_channels
        else 0
    )
    second = conv_params(taps, 2 * channels, 2 * channels, weight_norm=weight_norm)
    return first + conditioning + second


def pixelsnail_params(
    *,
    channels: int = 256,
    attention_repetitions: int = 12,
    residual_blocks: int = 4,
    query_size: int = 16,
    logistic_mixtures: int = 10,
    image_channels: int = 3,
    weight_norm: bool = True,
) -> tuple[int, dict[str, int]]:
    """Count ``h12_noup_smallkey`` tensors by architectural component."""
    padded_image_channels = image_channels + 1
    parts = {
        "initial_convolutions": (
            conv_params(3, padded_image_channels, channels, weight_norm=weight_norm)
            + conv_params(2, padded_image_channels, channels, weight_norm=weight_norm)
        )
    }

    inner_block = gated_resnet_params(channels, 4, weight_norm=weight_norm)
    parts["inner_resnets"] = attention_repetitions * residual_blocks * inner_block

    raw_channels = image_channels + channels + 2
    key_mixin = gated_resnet_params(raw_channels, 1, weight_norm=weight_norm)
    key_mixin += nin_params(
        raw_channels, channels // 2 + query_size, weight_norm=weight_norm
    )

    query_channels = channels + 2
    query = gated_resnet_params(query_channels, 1, weight_norm=weight_norm)
    query += nin_params(query_channels, query_size, weight_norm=weight_norm)

    foldback = gated_resnet_params(
        channels,
        1,
        conditioning_channels=channels // 2,
        weight_norm=weight_norm,
    )
    parts["attention_blocks"] = attention_repetitions * (key_mixin + query + foldback)
    parts["output_nin"] = nin_params(
        channels, 10 * logistic_mixtures, weight_norm=weight_norm
    )
    return sum(parts.values()), parts


def pixelcnnpp_params(
    *,
    channels: int = 160,
    residual_blocks: int = 5,
    logistic_mixtures: int = 10,
    image_channels: int = 3,
    weight_norm: bool = True,
) -> int:
    """Reproduce the public PixelCNN++ count as a formula sanity check."""
    padded_image_channels = image_channels + 1
    total = conv_params(3, padded_image_channels, channels, weight_norm=weight_norm)
    total += conv_params(3, padded_image_channels, channels, weight_norm=weight_norm)
    total += conv_params(2, padded_image_channels, channels, weight_norm=weight_norm)

    for stage in range(3):
        for _ in range(residual_blocks):
            total += gated_resnet_params(channels, 6, weight_norm=weight_norm)
            total += gated_resnet_params(
                channels, 4, conditioning_channels=channels, weight_norm=weight_norm
            )
        if stage < 2:
            total += conv_params(6, channels, channels, weight_norm=weight_norm)
            total += conv_params(4, channels, channels, weight_norm=weight_norm)

    for stage in range(3):
        for _ in range(residual_blocks + 1):
            total += gated_resnet_params(
                channels, 6, conditioning_channels=channels, weight_norm=weight_norm
            )
            total += gated_resnet_params(
                channels, 4, conditioning_channels=2 * channels, weight_norm=weight_norm
            )
        if stage < 2:
            total += conv_params(6, channels, channels, weight_norm=weight_norm)
            total += conv_params(4, channels, channels, weight_norm=weight_norm)

    return total + nin_params(channels, 10 * logistic_mixtures, weight_norm=weight_norm)


def main() -> None:
    pixelcnnpp = pixelcnnpp_params()
    print(f"PixelCNN++ sanity check: {pixelcnnpp:,} ({pixelcnnpp / 1e6:.2f}M)")
    print("PixelSNAIL h12_noup_smallkey:")
    for residual_blocks in (2, 4, 5):
        total, parts = pixelsnail_params(residual_blocks=residual_blocks)
        breakdown = ", ".join(f"{name}={value:,}" for name, value in parts.items())
        print(
            f"  nr_resnet={residual_blocks}: {total:,} ({total / 1e6:.2f}M); "
            f"{breakdown}"
        )


if __name__ == "__main__":
    main()
