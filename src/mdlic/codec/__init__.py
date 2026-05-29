"""无损算术编码 codec（纯 Python，不依赖 torch）。

把 AR 模型给出的每步条件分布 p(x_i | x_<i) 喂给算术编码器，得到真实可解的
bitstream，码长 ≈ Σ −log2 p = NLL/ln2 = bpd·N。用于"真实可解性" roundtrip
验证（scripts/verify_lossless.py），区别于只报 bpd、不验证可解性的工作。
"""
from .arithmetic import ArithmeticEncoder, ArithmeticDecoder, build_cumfreq, FREQ_TOTAL

__all__ = ["ArithmeticEncoder", "ArithmeticDecoder", "build_cumfreq", "FREQ_TOTAL"]
