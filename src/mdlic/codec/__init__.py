"""无损算术编码 codec（纯 Python，不依赖 torch）。

把 AR 模型给出的每步条件分布 p(x_i | x_<i) 喂给算术编码器，得到真实可解的
bitstream，码长近似模型条件码长（另计首 token 先验）。用于"真实可解性" roundtrip
验证（scripts/verify_lossless.py），区别于只报 bpd、不验证可解性的工作。
"""
from .arithmetic import ArithmeticEncoder, ArithmeticDecoder, build_cumfreq, FREQ_TOTAL
from .grouped import (
    decode_grouped,
    encode_grouped,
    grouped_codec_protocol,
    make_model_probability_fn,
    verify_grouped_codec,
)

__all__ = [
    "ArithmeticEncoder",
    "ArithmeticDecoder",
    "build_cumfreq",
    "FREQ_TOTAL",
    "encode_grouped",
    "decode_grouped",
    "grouped_codec_protocol",
    "make_model_probability_fn",
    "verify_grouped_codec",
]
