import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..")) 

import torch

import triton
import triton.language as tl
from src.mdlic.ops.flash_attn import TritonAttention, _test_op

@pytest.mark.parametrize(
    "batch, heads, seq_len, head_dim, causal",
    [
        (2, 4, 1024, 64, True),    # 因果，中等规模
        (2, 4, 1024, 64, False),   # 非因果
        (1, 2, 128, 32, True),     # 小尺寸，快速验证
        (2, 8, 2048, 64, True),    # 较大规模（可能耗时）
        (1, 2, 192, 32, True),     # 非 block 对齐（192 = 64*3，对齐）
        (1, 4, 320, 64, True),     # 非 128 对齐但 64 对齐
    ],
    ids=["causal_medium", "non_causal", "causal_small", "causal_large",
         "causal_192", "causal_320"]
)

def test_flash_attn(batch, heads, seq_len, head_dim, causal):
    """测试 Flash Attention 前向和反向与标准实现的等价性"""
    # 调用原始的 test_op 函数，它会自动比较 Triton 实现与标准实现
    _test_op(
        BATCH_SIZE=batch,
        NUM_HEADS=heads,
        SEQ_LEN=seq_len,
        HEAD_DIM=head_dim,
        causal=causal,
        dtype=torch.float16  # 默认使用 float16，可根据需要调整
    )

if __name__ == "__main__":
    _test_op(BATCH_SIZE=4, NUM_HEADS=8, SEQ_LEN=2048, HEAD_DIM=64, causal=True)
    _test_op(BATCH_SIZE=4, NUM_HEADS=8, SEQ_LEN=2048, HEAD_DIM=64, causal=False)
    print("PASSED")