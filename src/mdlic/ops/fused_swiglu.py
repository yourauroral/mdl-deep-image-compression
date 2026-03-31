"""
Fused SwiGLU Triton Kernel — forward + backward 合并为单次 kernel launch。

SwiGLU: out = SiLU(a) ⊙ b
  其中 a = W1·x（gate path），b = W3·x（value path）。

标准 PyTorch 实现需要 3 次内存读写：
  1. 读 a → 计算 silu(a) → 写中间结果
  2. 读 silu(a) + b → 逐元素乘 → 写 out
Fused kernel 只需 1 次：读 a, b → 计算 silu(a)*b → 写 out。

Backward 采用 activation recomputation 策略：
  不存储 silu(a) 中间结果（节省 M×N 的显存），
  而是在 backward 时从保存的 a 重新计算 sigmoid(a) 和 silu(a)。
  额外计算量 ~33%，但对显存受限场景收益大。

参考:
  [1] Shazeer, "GLU Variants Improve Transformers," arXiv:2002.05202, 2020.
      SwiGLU = Swish(a) ⊙ b, Section 2, Eq.(6).
  [2] Hsu et al., "Liger Kernel: Efficient Triton Kernels for LLM Training,"
      arXiv:2410.10989, 2024.
      Fused SwiGLU pattern 参考（手写实现）。
  [3] Chen et al., "Training Deep Nets with Sublinear Memory Cost,"
      arXiv:1604.06174, 2016.
      Activation recomputation（gradient checkpointing）的理论基础。
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_swiglu_fwd_kernel(
    A_ptr,      # gate input: a = W1·x, shape (M, N)
    B_ptr,      # value input: b = W3·x, shape (M, N)
    C_ptr,      # output: silu(a) * b, shape (M, N)
    stride_m,   # row stride
    N: tl.constexpr,          # 列数
    BLOCK_N: tl.constexpr,    # 每个 program 处理的列数
):
    """
    Forward kernel: c_i = silu(a_i) * b_i，逐行处理。

    每个 program 处理一行的一个 BLOCK_N 片段。
    Grid: (M, cdiv(N, BLOCK_N))
    """
    row = tl.program_id(0)
    col_block = tl.program_id(1)
    col_offsets = col_block * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = col_offsets < N

    offset = row * stride_m + col_offsets

    # 读取 a, b（一次 HBM 读）
    a = tl.load(A_ptr + offset, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + offset, mask=mask, other=0.0)

    # SiLU(a) = a * sigmoid(a)  [1] Eq.(5)
    sig_a = tl.sigmoid(a)
    silu_a = a * sig_a

    # out = silu(a) ⊙ b  [1] Eq.(6)
    c = silu_a.to(b.dtype) * b

    # 写入输出（一次 HBM 写）
    tl.store(C_ptr + offset, c, mask=mask)


@triton.jit
def _fused_swiglu_bwd_kernel(
    A_ptr,      # 保存的 gate input a
    B_ptr,      # 保存的 value input b
    DC_ptr,     # 上游梯度 dc = d_loss/d_c
    DA_ptr,     # 输出: d_loss/d_a
    DB_ptr,     # 输出: d_loss/d_b
    stride_m,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Backward kernel: 从保存的 a, b 重新计算 silu(a)（activation recomputation）。

    推导（c = silu(a) * b）:
      dc/da = b * d(silu(a))/da = b * (silu(a) + sigmoid(a) * (1 - silu(a)))
            = b * sigmoid(a) * (1 + a * (1 - sigmoid(a)))
            = b * sigmoid(a) * (1 + a - a * sigmoid(a))
      简化: dc/da = b * (silu(a) * (1 - sigmoid(a)) + sigmoid(a))
                   = b * (sigmoid(a) + silu(a) * (1 - sigmoid(a)))
      dc/db = silu(a)

    最终:
      da = dc * b * (sigmoid(a) + a * sigmoid(a) * (1 - sigmoid(a)))
      db = dc * silu(a)

    Ref: [2] Liger-Kernel SwiGLU backward, activation recomputation pattern [3].
    """
    row = tl.program_id(0)
    col_block = tl.program_id(1)
    col_offsets = col_block * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = col_offsets < N

    offset = row * stride_m + col_offsets

    # Activation recomputation: 重新读 a, b（不存 silu(a)，节省显存）[3]
    a_raw = tl.load(A_ptr + offset, mask=mask, other=0.0)
    orig_dtype = a_raw.dtype
    a = a_raw.to(tl.float32)
    b = tl.load(B_ptr + offset, mask=mask, other=0.0).to(tl.float32)
    dc = tl.load(DC_ptr + offset, mask=mask, other=0.0).to(tl.float32)

    # 重新计算 sigmoid 和 silu
    sig_a = tl.sigmoid(a)
    silu_a = a * sig_a

    # da = dc * b * (sigmoid(a) + a * sigmoid(a) * (1 - sigmoid(a)))
    #    = dc * b * sigmoid(a) * (1 + a * (1 - sigmoid(a)))
    #    = dc * b * sigmoid(a) * (1 + a - a * sigmoid(a))
    da = dc * b * (sig_a + silu_a * (1.0 - sig_a))

    # db = dc * silu(a)
    db = dc * silu_a

    tl.store(DA_ptr + offset, da.to(orig_dtype), mask=mask)
    tl.store(DB_ptr + offset, db.to(orig_dtype), mask=mask)


class FusedSwiGLUFunction(torch.autograd.Function):
    """
    Fused SwiGLU autograd 函数。

    Forward:  c = silu(a) * b，一次 kernel
    Backward: 从 (a, b) 重新计算 silu(a)，一次 kernel

    保存 a, b 用于 backward（activation recomputation 替代存储 silu(a)）。
    """

    @staticmethod
    def forward(ctx, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        assert a.shape == b.shape, f"a.shape={a.shape} != b.shape={b.shape}"
        # 防御性 contiguous()，避免非连续张量导致 Triton kernel 越界
        a = a.contiguous()
        b = b.contiguous()

        orig_shape = a.shape
        a_flat = a.reshape(-1, a.shape[-1])
        b_flat = b.reshape(-1, b.shape[-1])
        M, N = a_flat.shape

        c = torch.empty_like(b_flat)

        BLOCK_N = triton.next_power_of_2(N) if N <= 4096 else 4096
        grid = (M, triton.cdiv(N, BLOCK_N))

        _fused_swiglu_fwd_kernel[grid](
            a_flat, b_flat, c,
            stride_m=N,
            N=N,
            BLOCK_N=BLOCK_N,
        )

        # 保存 a, b 用于 backward activation recomputation
        ctx.save_for_backward(a_flat, b_flat)
        ctx.orig_shape = orig_shape

        return c.reshape(orig_shape)

    @staticmethod
    def backward(ctx, dc: torch.Tensor):
        a_flat, b_flat = ctx.saved_tensors
        dc_flat = dc.reshape(a_flat.shape).contiguous()
        M, N = a_flat.shape

        da = torch.empty_like(a_flat)
        db = torch.empty_like(b_flat)

        BLOCK_N = triton.next_power_of_2(N) if N <= 4096 else 4096
        grid = (M, triton.cdiv(N, BLOCK_N))

        _fused_swiglu_bwd_kernel[grid](
            a_flat, b_flat, dc_flat,
            da, db,
            stride_m=N,
            N=N,
            BLOCK_N=BLOCK_N,
        )

        return da.reshape(ctx.orig_shape), db.reshape(ctx.orig_shape)


def fused_swiglu(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Fused SwiGLU 便捷接口。

    计算 silu(a) ⊙ b，a 和 b 形状相同。
    等价于 F.silu(a) * b，但只需一次 HBM 读写。

    参数:
      a: gate input (W1·x)，任意形状，最后一维为 d_ff
      b: value input (W3·x)，与 a 同形状
    返回:
      silu(a) * b，同形状
    """
    return FusedSwiGLUFunction.apply(a, b)
