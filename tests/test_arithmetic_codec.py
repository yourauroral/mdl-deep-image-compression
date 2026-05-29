"""算术 codec 单元测试（纯 CPU，WSL 可跑）。

验证三件事：
  1. roundtrip 逐符号 bit-exact（任意离散源）；
  2. 实际码长逼近 Shannon 熵（量化 overhead 可控）；
  3. 退化分布（near-deterministic）、均匀分布等边界不崩。
"""
import math
import random

from src.mdlic.codec.arithmetic import (
    ArithmeticEncoder, ArithmeticDecoder, build_cumfreq, pack_bits, FREQ_TOTAL,
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
    ideal_bits = 0.0
    for _ in range(3000):
        logits = [random.gauss(0, 2) for _ in range(V)]
        m = max(logits)
        exps = [math.exp(l - m) for l in logits]
        z = sum(exps)
        probs = [e / z for e in exps]
        sym = random.randrange(V)
        tables.append(probs)
        symbols.append(sym)
        ideal_bits += -math.log2(probs[sym])
    out, nbits = _roundtrip(symbols, tables)
    assert out == symbols
    overhead = (nbits - ideal_bits) / ideal_bits
    # 量化到 2^16 + 每符号保底 1 的 overhead，整体应 < 3%
    assert overhead < 0.03, f"overhead {overhead:.4f} 过大"


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


def test_pack_bits_roundtrip_len():
    bits = [1, 0, 1, 1, 0, 0, 1, 0, 1]
    packed = pack_bits(bits)
    assert len(packed) == 2  # 9 bit → 2 byte
