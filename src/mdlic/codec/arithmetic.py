"""32-bit 整数算术编码器（Witten-Neal-Cleary, CACM 1987 风格，手写）。

纯 Python + 标准库，无 torch / numpy 依赖，方便在 WSL 上单测。

核心不变量
----------
区间 [low, high)（high 不含）始终落在 [0, TOP) 内，TOP = 2^32。
每个符号 i 用累积频率三元组 (cum_lo, cum_hi, total) 划分子区间：

    span      = high - low
    new_low   = low + span * cum_lo // total
    new_high  = low + span * cum_hi // total

随后做 renormalization（E1/E2/E3 三类区间放缩）保证精度不丢：
  - E1: 整段落在下半区 [0, HALF)        → 输出 0 + 暂存位
  - E2: 整段落在上半区 [HALF, TOP)      → 输出 1 + 暂存位（减 HALF）
  - E3: 跨中点但夹在 [QUARTER, 3·QUARTER)→ underflow，pending+1（减 QUARTER）

为保证 span * total 不溢出且 span 永远 > total，要求 4 * FREQ_TOTAL <= TOP。
这里 FREQ_TOTAL = 2^16，远小于 TOP/4 = 2^30，安全。

decoder 复刻完全相同的区间放缩逻辑，靠读入的 code 落在哪个子区间反查符号；
只要 encoder/decoder 在每一步用**完全相同**的 cumfreq 表，就 bit-exact 可逆。
"""
import math
from collections.abc import Sequence
from typing import List, Tuple

PRECISION = 32
TOP = 1 << PRECISION          # 2^32
HALF = TOP >> 1               # 2^31
QUARTER = TOP >> 2            # 2^30
THREE_QUARTER = 3 * QUARTER   # 3·2^30
MASK = TOP - 1

# 概率量化的总频数。需满足 4 * FREQ_TOTAL <= TOP（这里 2^16 << 2^30）。
FREQ_TOTAL = 1 << 16


def _validate_total(total: int) -> None:
    if not isinstance(total, int) or isinstance(total, bool) or total <= 0:
        raise ValueError(f"frequency total must be a positive integer, got {total!r}")
    if 4 * total > TOP:
        raise ValueError(
            f"frequency total {total} exceeds arithmetic precision limit TOP/4={QUARTER}"
        )


def _validate_cumfreq(cum: Sequence[int]) -> tuple[int, int]:
    if not isinstance(cum, Sequence) or isinstance(cum, (str, bytes, bytearray)):
        raise TypeError("cumfreq must be a sequence of integers")
    if len(cum) < 2:
        raise ValueError("cumfreq must contain at least [0, total]")
    if not isinstance(cum[0], int) or isinstance(cum[0], bool) or cum[0] != 0:
        raise ValueError("cumfreq[0] must be integer zero")

    previous = 0
    for index, value in enumerate(cum[1:], start=1):
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"cumfreq[{index}] must be an integer, got {value!r}")
        if value <= previous:
            raise ValueError("cumfreq must be strictly increasing")
        previous = value
    _validate_total(previous)
    return len(cum) - 1, previous


def _validate_bits(bits) -> List[int]:
    try:
        values = list(bits)
    except TypeError as exc:
        raise TypeError("bits must be an iterable of integer 0/1 values") from exc
    for index, bit in enumerate(values):
        if not isinstance(bit, int) or isinstance(bit, bool) or bit not in (0, 1):
            raise ValueError(f"bits[{index}] must be integer 0 or 1, got {bit!r}")
    return values


def build_cumfreq(probs: List[float], total: int = FREQ_TOTAL) -> List[int]:
    """概率向量 → 长度 V+1 的累积频数表 cum，cum[0]=0, cum[V]=total。

    确定性量化（encoder/decoder 必须得到逐位相同的结果）：
      1. 每个符号至少分到 1（保证任意符号可编码，避免 0 宽子区间）；
      2. 剩余 total-V 按 probs 比例分配，向下取整；
      3. 余数按 (小数部分, 索引) 降序补给，保证 Σfreq == total 且可复现。
    """
    V = len(probs)
    if V == 0:
        raise ValueError("概率向量不能为空")
    if not isinstance(total, int) or isinstance(total, bool) or total <= V:
        raise ValueError(f"total ({total}) 必须是大于词表大小 ({V}) 的整数")
    _validate_total(total)

    clean_probs = []
    for i, value in enumerate(probs):
        p = float(value)
        if not math.isfinite(p):
            raise ValueError(f"probs[{i}] 不是有限数: {value!r}")
        if p < 0.0:
            raise ValueError(f"probs[{i}] 为负数: {p}")
        clean_probs.append(p)

    s = math.fsum(clean_probs)
    if s <= 0.0:
        clean_probs = [1.0] * V  # 全 0 兜底为均匀分布
        s = float(V)

    budget = total - V                       # 先给每符号保底 1
    freqs = [1] * V
    rema: List[Tuple[float, int]] = []
    allocated = 0
    for i, p in enumerate(clean_probs):
        share = (p / s) * budget if p > 0.0 else 0.0
        add = int(share)                      # 向下取整
        freqs[i] += add
        allocated += add
        rema.append((share - add, i))

    leftover = budget - allocated             # 因取整丢掉的份额
    # 小数部分大的优先补；索引升序作为稳定 tie-break（与平台无关，可复现）
    rema.sort(key=lambda t: (-t[0], t[1]))
    for k in range(leftover):
        freqs[rema[k][1]] += 1

    cum = [0] * (V + 1)
    acc = 0
    for i in range(V):
        acc += freqs[i]
        cum[i + 1] = acc
    assert cum[V] == total, f"cumfreq 总和 {cum[V]} != {total}"
    return cum


class ArithmeticEncoder:
    """逐符号编码到 bit 列表（0/1）。最后 finish() 收尾并可 pack 成 bytes。"""

    def __init__(self):
        self.low = 0
        self.high = MASK              # 闭区间上界，区间语义 [low, high]
        self.pending = 0              # E3 underflow 暂存位计数
        self.bits: List[int] = []

    def _emit(self, bit: int):
        self.bits.append(bit)
        # 暂存的 underflow 位取反输出（WNC 标准收尾）
        inv = 1 - bit
        for _ in range(self.pending):
            self.bits.append(inv)
        self.pending = 0

    def encode(self, symbol: int, cum: List[int]):
        vocab_size, total = _validate_cumfreq(cum)
        if (
            not isinstance(symbol, int)
            or isinstance(symbol, bool)
            or not 0 <= symbol < vocab_size
        ):
            raise ValueError(
                f"symbol must be an integer in [0,{vocab_size}), got {symbol!r}"
            )
        span = self.high - self.low + 1
        self.high = self.low + (span * cum[symbol + 1]) // total - 1
        self.low = self.low + (span * cum[symbol]) // total

        # renormalization
        while True:
            if self.high < HALF:                      # E1：落下半区
                self._emit(0)
            elif self.low >= HALF:                    # E2：落上半区
                self._emit(1)
                self.low -= HALF
                self.high -= HALF
            elif self.low >= QUARTER and self.high < THREE_QUARTER:  # E3：underflow
                self.pending += 1
                self.low -= QUARTER
                self.high -= QUARTER
            else:
                break
            self.low = (self.low << 1) & MASK
            self.high = ((self.high << 1) & MASK) | 1

    def finish(self) -> List[int]:
        """收尾：再输出 2 位定位最终区间（含 pending 反转）。返回 bit 列表。"""
        self.pending += 1
        if self.low < QUARTER:
            self._emit(0)
        else:
            self._emit(1)
        return self.bits


class ArithmeticDecoder:
    """从 bit 列表逐符号解码。需与 encoder 每步使用相同 cumfreq 表。"""

    def __init__(self, bits: List[int]):
        self.bits = _validate_bits(bits)
        self.pos = 0
        self.low = 0
        self.high = MASK
        self.code = 0
        for _ in range(PRECISION):
            self.code = (self.code << 1) | self._next_bit()

    def _next_bit(self) -> int:
        # 读尽后补 0（encoder 末尾有限位，decoder 可能多读几位定位区间）
        if self.pos < len(self.bits):
            b = self.bits[self.pos]
            self.pos += 1
            return b
        return 0

    def decode(self, cum: List[int]) -> int:
        _, total = _validate_cumfreq(cum)
        span = self.high - self.low + 1
        # 当前 code 落在 [0,total) 的哪个累积区间
        value = ((self.code - self.low + 1) * total - 1) // span

        # 二分查 symbol：最大的 s 使 cum[s] <= value
        lo, hi = 0, len(cum) - 1
        while hi - lo > 1:
            mid = (lo + hi) >> 1
            if cum[mid] <= value:
                lo = mid
            else:
                hi = mid
        symbol = lo

        self.high = self.low + (span * cum[symbol + 1]) // total - 1
        self.low = self.low + (span * cum[symbol]) // total

        while True:
            if self.high < HALF:
                pass
            elif self.low >= HALF:
                self.low -= HALF
                self.high -= HALF
                self.code -= HALF
            elif self.low >= QUARTER and self.high < THREE_QUARTER:
                self.low -= QUARTER
                self.high -= QUARTER
                self.code -= QUARTER
            else:
                break
            self.low = (self.low << 1) & MASK
            self.high = ((self.high << 1) & MASK) | 1
            self.code = ((self.code << 1) & MASK) | self._next_bit()
        return symbol


def pack_bits(bits: List[int]) -> bytes:
    """bit 列表 → bytes（高位在前，末尾补 0 到字节对齐）。"""
    bits = _validate_bits(bits)
    out = bytearray()
    acc = 0
    n = 0
    for b in bits:
        acc = (acc << 1) | (b & 1)
        n += 1
        if n == 8:
            out.append(acc)
            acc = 0
            n = 0
    if n > 0:
        out.append(acc << (8 - n))
    return bytes(out)


def unpack_bits(data: bytes, n_bits: int) -> List[int]:
    """bytes → 前 n_bits 个 bit 列表（pack_bits 的逆，高位在前）。

    pack_bits 末尾补 0 到字节对齐，故 unpack 必须知道真实 bit 数 n_bits 才能
    丢弃 padding。是落盘 .bin → 解码的必要逆操作（容器 header 存 n_bits）。
    """
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("data must be bytes-like")
    if not isinstance(n_bits, int) or isinstance(n_bits, bool) or n_bits < 0:
        raise ValueError(f"n_bits must be a non-negative integer, got {n_bits!r}")
    if n_bits > len(data) * 8:
        raise ValueError(f"n_bits={n_bits} 超过 data 容量 {len(data)*8} bit")
    bits = []
    for i in range(n_bits):
        byte = data[i >> 3]
        bit = (byte >> (7 - (i & 7))) & 1
        bits.append(bit)
    return bits
