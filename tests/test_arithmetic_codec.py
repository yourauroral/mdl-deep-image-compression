"""算术 codec 单元测试（纯 CPU，WSL 可跑）。

验证三件事：
  1. roundtrip 逐符号 bit-exact（任意离散源）；
  2. 实际码长逼近 Shannon 熵（量化 overhead 可控）；
  3. 退化分布（near-deterministic）、均匀分布等边界不崩。
"""
import math
import random

import pytest

from mdlic.codec.arithmetic import (
    ArithmeticEncoder, ArithmeticDecoder, build_cumfreq, pack_bits, unpack_bits,
    FREQ_TOTAL, TOP,
)


def _roundtrip(symbols, prob_tables):
    """用每步给定的概率表编码 symbols，再解码，返回解出的序列与 bit 数。"""
    enc = ArithmeticEncoder()
    for sym, probs in zip(symbols, prob_tables):
        enc.encode(sym, build_cumfreq(probs))
    bits = enc.finish()

    dec = ArithmeticDecoder(bits)
    out = [dec.decode(build_cumfreq(probs)) for probs in prob_tables]
    return out, len(bits)


def test_roundtrip_uniform():
    V = 256
    random.seed(0)
    symbols = [random.randrange(V) for _ in range(2000)]
    probs = [1.0 / V] * V
    tables = [probs] * len(symbols)
    out, nbits = _roundtrip(symbols, tables)
    assert out == symbols
    # 均匀 256-way：每符号理论 8 bit，量化+收尾 overhead 应 < 1%
    assert nbits / len(symbols) < 8.1


def test_roundtrip_skewed_random_tables():
    """每步用不同的随机分布，模拟真实 AR 模型逐步变化的条件分布。"""
    V = 256
    random.seed(1)
    tables = []
    symbols = []
    for _ in range(1500):
        logits = [random.gauss(0, 3) for _ in range(V)]
        m = max(logits)
        exps = [math.exp(l - m) for l in logits]
        z = sum(exps)
        probs = [e / z for e in exps]
        tables.append(probs)
        # 按该分布采样一个真实符号
        r = random.random()
        acc = 0.0
        sym = V - 1
        for i, p in enumerate(probs):
            acc += p
            if r <= acc:
                sym = i
                break
        symbols.append(sym)
    out, _ = _roundtrip(symbols, tables)
    assert out == symbols


def test_code_length_near_entropy():
    """实际码长应逼近 Σ −log2 p(真实符号)（量化 overhead 小）。"""
    V = 256
    random.seed(2)
    tables = []
    symbols = []
    model_ideal_bits = 0.0
    cdf_ideal_bits = 0.0
    for _ in range(3000):
        logits = [random.gauss(0, 2) for _ in range(V)]
        m = max(logits)
        exps = [math.exp(l - m) for l in logits]
        z = sum(exps)
        probs = [e / z for e in exps]
        sym = random.randrange(V)
        tables.append(probs)
        symbols.append(sym)
        model_ideal_bits += -math.log2(probs[sym])
        cum = build_cumfreq(probs)
        cdf_ideal_bits += -math.log2((cum[sym + 1] - cum[sym]) / cum[-1])
    out, nbits = _roundtrip(symbols, tables)
    assert out == symbols
    # 算术 coder 应跟它实际使用的量化 CDF 比，而不是量化前 softmax。
    # 最小频数会抬高极小概率，因此 model_ideal_bits 与 cdf_ideal_bits 的差可正可负。
    assert model_ideal_bits > 0
    assert abs(nbits - cdf_ideal_bits) < 4.0


def test_near_deterministic_distribution():
    """一个符号概率接近 1：码长应极短，且仍可逆。"""
    V = 256
    symbols = [7] * 500
    probs = [1e-6] * V
    probs[7] = 1.0 - 1e-6 * (V - 1)
    tables = [probs] * len(symbols)
    out, nbits = _roundtrip(symbols, tables)
    assert out == symbols
    # 500 个近确定符号，总码长应远小于均匀时的 500·8=4000 bit
    assert nbits < 500


def test_build_cumfreq_sums_to_total():
    V = 256
    random.seed(3)
    for _ in range(50):
        logits = [random.gauss(0, 4) for _ in range(V)]
        m = max(logits)
        exps = [math.exp(l - m) for l in logits]
        z = sum(exps)
        probs = [e / z for e in exps]
        cum = build_cumfreq(probs)
        assert cum[0] == 0
        assert cum[-1] == FREQ_TOTAL
        # 严格单调（每符号至少 1）→ 任意符号子区间非空
        for i in range(V):
            assert cum[i + 1] > cum[i]


def test_build_cumfreq_all_zero_falls_back_to_uniform():
    cum = build_cumfreq([0.0] * 256)
    assert cum[0] == 0
    assert cum[-1] == FREQ_TOTAL
    for i in range(256):
        assert cum[i + 1] > cum[i]


def test_build_cumfreq_rejects_negative_values():
    with pytest.raises(ValueError, match="负数"):
        build_cumfreq([-1.0, 0.0, 0.5, 0.5])


def test_build_cumfreq_rejects_non_finite_values():
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="有限"):
            build_cumfreq([0.5, bad, 0.5])


def test_pack_bits_roundtrip_len():
    bits = [1, 0, 1, 1, 0, 0, 1, 0, 1]
    packed = pack_bits(bits)
    assert len(packed) == 2  # 9 bit → 2 byte


def test_unpack_bits_inverts_pack():
    # pack→unpack(原始 bit 数) 必须逐位还原（丢弃字节对齐 padding）
    for n in [1, 7, 8, 9, 17, 100]:
        bits = [random.randint(0, 1) for _ in range(n)]
        assert unpack_bits(pack_bits(bits), n) == bits


def test_unpack_bits_overflow_raises():
    # 要求的 bit 数超过 data 容量应报错（容器 header n_bits 损坏的保护）
    with pytest.raises(ValueError):
        unpack_bits(pack_bits([1, 0, 1]), 100)


@pytest.mark.parametrize(
    "cum",
    [
        [],
        [1, 2],
        [0, 1, 1],
        [0, 1.5, 2],
        [0, TOP // 4 + 1],
    ],
)
def test_coder_rejects_invalid_cumfreq(cum):
    with pytest.raises((TypeError, ValueError)):
        ArithmeticEncoder().encode(0, cum)
    with pytest.raises((TypeError, ValueError)):
        ArithmeticDecoder([0]).decode(cum)


@pytest.mark.parametrize("symbol", [-1, 2, True, 1.5])
def test_encoder_rejects_invalid_symbol(symbol):
    with pytest.raises(ValueError, match="symbol"):
        ArithmeticEncoder().encode(symbol, [0, 1, 2])


def test_build_cumfreq_rejects_total_above_precision_limit():
    with pytest.raises(ValueError, match="precision limit"):
        build_cumfreq([0.5, 0.5], total=TOP // 4 + 1)


@pytest.mark.parametrize("bits", [[0, 2], [0, -1], [False], [0.0]])
def test_bit_apis_reject_non_binary_integer_values(bits):
    with pytest.raises(ValueError, match="integer 0 or 1"):
        pack_bits(bits)
    with pytest.raises(ValueError, match="integer 0 or 1"):
        ArithmeticDecoder(bits)


@pytest.mark.parametrize("n_bits", [-1, True, 1.5])
def test_unpack_bits_rejects_invalid_length(n_bits):
    with pytest.raises(ValueError, match="non-negative integer"):
        unpack_bits(b"\x00", n_bits)


def test_encode_pack_file_roundtrip(tmp_path):
    # 模拟落盘链路：encode → pack → 写文件 → 读回 → unpack → decode 还原符号
    random.seed(1)
    V = 256
    syms, tables = [], []
    enc = ArithmeticEncoder()
    for _ in range(500):
        logits = [random.gauss(0, 2.0) for _ in range(V)]
        m = max(logits); ex = [math.exp(l - m) for l in logits]; z = sum(ex)
        p = [e / z for e in ex]
        s = random.randrange(V)
        syms.append(s); tables.append(p)
        enc.encode(s, build_cumfreq(p))
    bits = enc.finish()
    n_bits = len(bits)

    f = tmp_path / "stream.bin"
    f.write_bytes(pack_bits(bits))
    bits_back = unpack_bits(f.read_bytes(), n_bits)
    assert bits_back == bits

    dec = ArithmeticDecoder(bits_back)
    out = [dec.decode(build_cumfreq(p)) for p in tables]
    assert out == syms
