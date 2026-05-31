"""MDLC .bin 容器单元测试（纯 CPU，WSL 可跑，无 torch 依赖）。

验证 verify_lossless.py 抽出的容器 helper（demo /api/encode|inspect|decode 复用）：
  1. _build_container_bytes → _read_container_bytes 双/单尺度逐 bit roundtrip；
  2. _parse_container_meta 字段正确（尺寸/码长/bpd/自洽校验）；
  3. 坏 magic / 截断 / 版本不符等损坏输入报错；
  4. 落盘 _write_container 与内存 _build_container_bytes 逐字节一致。
"""
import random
import struct

import pytest

from scripts.verify_lossless import (
    _build_container_bytes, _read_container_bytes, _parse_container_meta,
    _write_container, _read_container, _MAGIC, _HEADER, _HEADER_SIZE,
)


def _rand_bits(n):
    return [random.randint(0, 1) for _ in range(n)]


def test_build_read_roundtrip_dual():
    """双尺度：build → read 必须逐位还原 coarse/fine bit 列表与 H/C。"""
    random.seed(0)
    H, C = 32, 3
    c_bits = _rand_bits(517)      # 非字节对齐，验证 padding 处理
    f_bits = _rand_bits(24690)
    blob = _build_container_bytes(True, H, C, c_bits, f_bits)
    dual, rH, rC, rc, rf = _read_container_bytes(blob)
    assert dual is True
    assert (rH, rC) == (H, C)
    assert rc == c_bits
    assert rf == f_bits


def test_build_read_roundtrip_single():
    """单尺度：coarse 为空，fine 即整串。"""
    random.seed(1)
    H, C = 32, 3
    f_bits = _rand_bits(8001)
    blob = _build_container_bytes(False, H, C, [], f_bits)
    dual, rH, rC, rc, rf = _read_container_bytes(blob)
    assert dual is False
    assert (rH, rC) == (H, C)
    assert rc == []
    assert rf == f_bits


def test_build_read_roundtrip_imagenet64_geometry():
    """大图几何（64×64×3）也应正确 round-trip（H/C 各占 1 字节，64 在范围内）。"""
    random.seed(2)
    H, C = 64, 3
    c_bits = _rand_bits(2048)
    f_bits = _rand_bits(100003)
    blob = _build_container_bytes(True, H, C, c_bits, f_bits)
    dual, rH, rC, rc, rf = _read_container_bytes(blob)
    assert (dual, rH, rC) == (True, H, C)
    assert rc == c_bits and rf == f_bits


def test_parse_meta_fields():
    """_parse_container_meta 的尺寸/码长/bpd/自洽校验字段正确。"""
    random.seed(3)
    H, C = 32, 3
    c_bits = _rand_bits(600)        # 600 bit → 75 byte（对齐）
    f_bits = _rand_bits(24000)      # 24000 bit → 3000 byte（对齐）
    blob = _build_container_bytes(True, H, C, c_bits, f_bits)
    m = _parse_container_meta(blob)

    assert m["magic"] == "MDLC"
    assert m["version"] == 1
    assert m["dual"] is True
    assert (m["H"], m["C"]) == (H, C)
    assert m["n_subpix"] == H * H * C
    assert m["coarse_nbits"] == 600
    assert m["fine_nbits"] == 24000
    assert m["total_bits"] == 24600
    assert m["coarse_nbytes"] == 75
    assert m["fine_nbytes"] == 3000
    # payload = 75 + 3000；header 16B
    assert m["payload_bytes"] == 3075
    assert m["total_bytes"] == _HEADER_SIZE + 3075
    assert m["self_consistent"] is True
    assert m["bpd"] == pytest.approx(24600 / (32 * 32 * 3))
    # header_hex 是 16 字节 = 16 组两位 hex
    assert len(m["header_hex"].split()) == _HEADER_SIZE


def test_parse_meta_non_byte_aligned_bpd():
    """非字节对齐 bit 数下 bpd 仍按真实 bit 数算（而非按字节）。"""
    H, C = 32, 3
    c_bits = [1] * 7         # 7 bit → 1 byte（含 1 bit padding）
    f_bits = [0] * 9         # 9 bit → 2 byte（含 7 bit padding）
    blob = _build_container_bytes(True, H, C, c_bits, f_bits)
    m = _parse_container_meta(blob)
    assert m["total_bits"] == 16          # 真实 bit 数，不含 padding
    assert m["bpd"] == pytest.approx(16 / (32 * 32 * 3))
    assert m["self_consistent"] is True   # payload 3B = 1 + 2


def test_bad_magic_raises():
    blob = b"XXXX" + b"\x00" * 32
    with pytest.raises(ValueError, match="MDLC"):
        _read_container_bytes(blob)
    with pytest.raises(ValueError, match="MDLC"):
        _parse_container_meta(blob)


def test_truncated_header_raises():
    """文件小于 header 大小 → parse 报错（损坏检测）。"""
    blob = b"MDLC" + b"\x01\x01"   # 只有 6 字节 < 16
    with pytest.raises(ValueError):
        _parse_container_meta(blob)


def test_bad_version_raises():
    """版本号不符 → read 报错。"""
    # 手工拼一个 version=99 的 header（其余字段随意但合法）
    header = struct.pack(_HEADER, _MAGIC, 99, 1, 32, 3, 0, 8)
    blob = header + b"\x00"   # fine 8 bit = 1 byte
    with pytest.raises(ValueError, match="版本"):
        _read_container_bytes(blob)


def test_write_matches_build(tmp_path):
    """落盘 _write_container 与内存 _build_container_bytes 逐字节一致。"""
    random.seed(4)
    H, C = 32, 3
    c_bits = _rand_bits(333)
    f_bits = _rand_bits(12345)
    expect = _build_container_bytes(True, H, C, c_bits, f_bits)

    p = tmp_path / "img.bin"
    nbytes = _write_container(str(p), True, H, C, c_bits, f_bits)
    on_disk = p.read_bytes()
    assert on_disk == expect
    assert nbytes == len(expect)

    # 读文件路径与读字节路径解出的内容一致
    dual, rH, rC, rc, rf = _read_container(str(p))
    assert (dual, rH, rC) == (True, H, C)
    assert rc == c_bits and rf == f_bits
