"""Reproducible Pillow settings for classical lossless codec baselines."""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
from PIL import Image, features
import PIL


@dataclass(frozen=True)
class CodecSpec:
    method: str
    display_name: str
    pil_format: str
    save_kwargs: tuple[tuple[str, object], ...]
    backend: str

    def kwargs(self) -> dict[str, object]:
        return dict(self.save_kwargs)


_CODEC_SPECS = {
    "png": CodecSpec(
        method="png",
        display_name="PNG (lossless)",
        pil_format="PNG",
        save_kwargs=(("optimize", True),),
        backend="zlib",
    ),
    "webp": CodecSpec(
        method="webp",
        display_name="WebP (lossless)",
        pil_format="WEBP",
        save_kwargs=(("lossless", True),),
        backend="webp",
    ),
}


def get_codec_spec(method: str) -> CodecSpec:
    try:
        return _CODEC_SPECS[method.lower()]
    except (AttributeError, KeyError) as exc:
        supported = ", ".join(sorted(_CODEC_SPECS))
        raise ValueError(
            f"unsupported traditional codec {method!r}; expected one of {supported}"
        ) from exc


def encode_rgb_array(image: np.ndarray, method: str) -> bytes:
    """Encode one HWC uint8 RGB image with the project's pinned options."""
    array = np.asarray(image)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(
            f"expected HWC uint8 RGB input, got shape={array.shape}, dtype={array.dtype}"
        )
    spec = get_codec_spec(method)
    buffer = io.BytesIO()
    Image.fromarray(array).save(
        buffer,
        format=spec.pil_format,
        **spec.kwargs(),
    )
    return buffer.getvalue()


def codec_metadata(method: str) -> dict[str, object]:
    """Return JSON-serializable settings and runtime library versions."""
    spec = get_codec_spec(method)
    return {
        "method": spec.method,
        "display_name": spec.display_name,
        "pil_format": spec.pil_format,
        "save_kwargs": spec.kwargs(),
        "pillow_version": PIL.__version__,
        "backend": spec.backend,
        "backend_version": features.version(spec.backend),
    }
