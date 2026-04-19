"""
Flash Attention Triton Kernel — forward + backward with causal masking.

=== 概述 ===
手写 Flash Attention v2 在 Triton 上的实现，支持 causal mask（自回归）。
与标准 O(N²) 注意力相比，通过分块计算和 online softmax 将 HBM IO 降至 O(N·d)。

=== Forward Pass ===
- 使用 online softmax 技巧（Milakov & Gimelshein, arXiv:1805.02867）
  避免存储完整注意力矩阵 P ∈ ℝ^(N×N)，只需 O(1) SRAM 存储中间累加值
- 分块计算：每个 thread block 处理查询的一个块 Q ∈ ℝ^(Tr×d)，
  逐个加载 key/value 块并更新累加器
- 维护运行最大值 m_i 和归一化因子 l_i，使用 rescaling 技巧保持数值稳定性
- 在反向时保存 logsumexp 值 M ∈ ℝ^N 用于梯度计算

=== Backward Pass ===
- Recomputation 策略：前向时计算的 attention logits P 在反向时重新计算
  （不存储），仅通过 logsumexp M 和输出 O 恢复
- Early termination 优化（FlashAttention-2）：
  * _attn_bwd_dq: query q 只与 KV 位置 0..q 有梯度，之后的块全被 mask
  * _attn_bwd_dk_dv: KV k 只与 Q 位置 k..T 有梯度
  减少约 50% 的无效计算

=== 关键文献 ===
  [1] Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention
      with IO-Awareness," NeurIPS 2022, arXiv:2205.14135.
      - 原始 Flash Attention，O(N) SRAM, O(N) IOs
  [2] Dao, "FlashAttention-2: Faster Attention with Better Parallelism and
      Work Partitioning," arXiv:2307.08691, 2023.
      - 改进并行度，引入 causal early termination，2x-4x 加速
  [3] Milakov & Gimelshein, "Online normalizer calculation for softmax,"
      arXiv:1805.02867, 2018.
      - Online softmax 的数学基础，使用 max 和 sum 进行稳定的 rescaling
  [4] Triton Tutorials — Fused Attention.
      https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html
"""

import torch

import triton
import triton.language as tl

# ─── 固定 block size ───────────────────────────────────────────
# 不使用 autotune，避免首次运行编译 24 个变体的开销。
# BLOCK_Q=64, BLOCK_KV=64 在 HEAD_DIM<=128 时是通用的平衡选择。
# Ref: [2] Section 3.3 — block size 选择需平衡 SRAM 占用与并行度。
BLOCK_Q_DEFAULT = 64
BLOCK_KV_DEFAULT = 64
# backward 使用较小的 micro block 和较大的 macro block
BLOCK_MICRO = 32
BLOCK_MACRO = 64


@triton.jit
def _attn_fwd_inner(
    O_block,    # 累加器，shape (BLOCK_SIZE_Q, HEAD_DIM)
    l_i,        # 运行归一化因子 l_i = Σ_j exp(s_ij - m_i)，shape (BLOCK_SIZE_Q,)
    m_i,        # 运行最大值 m_i = max_j s_ij，用于数值稳定，shape (BLOCK_SIZE_Q,)
    Q_block,    # 当前查询块 Q，shape (BLOCK_SIZE_Q, HEAD_DIM)
    K_block_ptr, # key 块指针，逐步加载
    V_block_ptr, # value 块指针，逐步加载
    block_index_q,  # 当前处理的 Q 块索引（用于 causal mask）
    softmax_scale,  # 缩放因子 1/√d，在 QK^T 后应用
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    STAGE: tl.constexpr,  # 1: 全非 mask，2: 对角块，3: 全非 mask 或对角块后
    offs_q: tl.constexpr,  # query 的绝对位置偏移
    offs_kv: tl.constexpr, # kv 的相对位置偏移
    SEQ_LEN: tl.constexpr,
):
    """
    Flash Attention 的核心循环，处理一个查询块与所有 KV 块的交互。

    算法（online softmax，Ref [3]）：
      初始化: m_i = -∞, l_i = 0, O = 0
      对每个 KV 块 j:
        1. 加载 K_j, V_j
        2. 计算 QK_j^T ∈ ℝ^(Tr×Tc)，应用 causal mask（若有）和缩放
        3. m_ij ← max(QK_j^T) per row，更新运行最大值
           - 修正因子 α = exp(m_i - m_ij) 用于重新加权历史累积
        4. P_j ← exp(QK_j^T - m_ij)（减 m_ij 避免溢出）
        5. l_ij ← sum(P_j) per row，更新运行归一化因子
        6. O ← α·O_old + P_j·V_j（修正旧累积并添加新贡献）
      最后: O ← O / l_i（归一化）

    关键点：
    - l_i 和 m_i 在块之间被修正，避免显式存储完整 P（O(1) vs O(N²) SRAM）
    - Causal mask 在 STAGE 2 时应用在 QK_block 上（mask 为 -inf）
    """
    # ─── 确定本次迭代的 KV 范围 ───
    # STAGE: 1 = 对角线左侧（非 causal），2 = 对角线上（过渡），3 = 非 causal 或对角线（全）
    if STAGE == 1:
        # 非 causal 或因果下对角线左侧: KV 从 0 到 query 块开始处
        lo, hi = 0, block_index_q * BLOCK_SIZE_Q
    elif STAGE == 2:
        # Causal 对角线块：query 和 KV 在同一块，部分被 mask
        lo, hi = block_index_q * BLOCK_SIZE_Q, (block_index_q + 1) * BLOCK_SIZE_Q
        lo = tl.multiple_of(lo, BLOCK_SIZE_Q)  # 编译器优化提示
    else:
        # 非 causal 全覆盖或 causal 对角线右侧（全被 mask，跳过）
        lo, hi = 0, SEQ_LEN

    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))

    # ─── Online Softmax 主循环 ───
    for start_kv in range(lo, hi, BLOCK_SIZE_KV):
        start_kv = tl.multiple_of(start_kv, BLOCK_SIZE_KV)  # 编译器优化

        # 1. 加载 K 块并计算 QK^T
        K_block = tl.load(K_block_ptr)  # shape (HEAD_DIM, BLOCK_SIZE_KV)
        QK_block = tl.dot(Q_block, K_block)  # (BLOCK_SIZE_Q, HEAD_DIM) @ (HEAD_DIM, BLOCK_SIZE_KV) → (BLOCK_SIZE_Q, BLOCK_SIZE_KV)

        # 2. 应用 causal mask 和缩放
        if STAGE == 2:
            # 对角线块中，应用下三角 mask：mask[i,j] = True if i >= j，否则为 False
            # 被 mask 的位置 (i < j) 对应未来token，设为 -inf
            mask = offs_q[:, None] >= (start_kv + offs_kv[None, :])  # (BLOCK_SIZE_Q, BLOCK_SIZE_KV)
            # QK_block * scale + where(mask, 0, -inf) = {scaled_logits, -inf}[mask]
            QK_block = QK_block * softmax_scale + tl.where(mask, 0, float('-inf'))
            # 更新运行最大值（-inf 项会被忽略）
            m_ij = tl.maximum(m_i, tl.max(QK_block, 1))  # per-row max
            QK_block -= m_ij[:, None]  # 减去最大值避免 exp 溢出
        else:
            # STAGE 1 或 3：无 mask，直接缩放并更新最大值
            m_ij = tl.maximum(m_i, tl.max(QK_block, 1) * softmax_scale)
            QK_block = QK_block * softmax_scale - m_ij[:, None]

        # 3. 计算注意力权重 P 和归一化因子更新
        P_block = tl.math.exp(QK_block)  # exp(logits - m_ij) ∈ [0,1]
        l_ij = tl.sum(P_block, 1)  # 每行求和，shape (BLOCK_SIZE_Q,)

        # 4. 修正历史累积（online softmax 的核心）
        # 新的全局最大值是 m_ij，旧的是 m_i，差值为 m_i - m_ij ≤ 0
        # α = exp(m_i - m_ij) ≤ 1 是修正因子
        alpha = tl.math.exp(m_i - m_ij)
        # 旧的 l_i 需要乘以 α 来适配新的最大值
        # l_i_new = α·l_i_old + l_ij = Σ exp(s_ij - m_ij)（全局）
        l_i = l_i * alpha + l_ij

        # 5. 加载 V 块并更新输出累积
        V_block = tl.load(V_block_ptr)  # shape (BLOCK_SIZE_KV, HEAD_DIM)
        V_block = V_block.to(Q_block.dtype)
        P_block = P_block.to(Q_block.dtype)

        # O_new = α·O_old + P·V
        # 这对应 O = Σ_j α_j·P_j·V_j 的递推更新
        # 其中 α_j 随 j 累积，使得最终 O 被正确归一化
        O_block = O_block * alpha[:, None]
        O_block = tl.dot(P_block, V_block, O_block)  # 累加更新

        m_i = m_ij

        # ─── 移动指针到下一个 KV 块 ───
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_SIZE_KV, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_SIZE_KV))

    return O_block, l_i, m_i


@triton.jit
def _attn_fwd(
    Q,  # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
    K,  # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
    V,  # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
    softmax_scale,
    M,  # BATCH_SIZE, NUM_HEADS, SEQ_LEN
    O,  # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
    stride_Q_batch,
    stride_Q_head,
    stride_Q_seq,
    stride_Q_dim,
    stride_K_batch,
    stride_K_head,
    stride_K_seq,
    stride_K_dim,
    stride_V_batch,
    stride_V_head,
    stride_V_seq,
    stride_V_dim,
    stride_O_batch,
    stride_O_head,
    stride_O_seq,
    stride_O_dim,
    BATCH_SIZE,
    NUM_HEADS: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    STAGE: tl.constexpr,
):
    # This indicate which block in the sequence length to process
    block_index_q = tl.program_id(0)

    # This indicates which head and batch to process. Each program is associated with a single head of a single batch
    index_batch_head = tl.program_id(1)
    # This indicate which batch this program is associated with (each batch has NUM_HEADS heads)
    index_batch = index_batch_head // NUM_HEADS
    # This indicate the position of the head in the batch
    index_head = index_batch_head % NUM_HEADS

    # This allows to get the (N_CTX, HEAD_DIM) block in the Q, K, V by selecting indexing it by batch and head
    qvk_offset = (
        index_batch.to(tl.int64) * stride_Q_batch
        + index_head.to(tl.int64) * stride_Q_head
    )

    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_Q_seq, stride_Q_dim),
        offsets=(block_index_q * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_V_seq, stride_V_dim),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_KV, HEAD_DIM),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(HEAD_DIM, SEQ_LEN),
        strides=(
            stride_K_dim,
            stride_K_seq,
        ),  # We invert the strides w.r.t Q, so we transpose the matrix
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_SIZE_KV),
        order=(0, 1),
    )

    O_block_ptr = tl.make_block_ptr(
        base=O + qvk_offset,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_O_seq, stride_O_dim),
        offsets=(block_index_q * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
        order=(1, 0),
    )

    # offs_q: the offsets for the tokens in the Q to process
    offs_q = block_index_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
    # offs_kv: the offsets for the tokens in the K and V sequence to process
    offs_kv = tl.arange(0, BLOCK_SIZE_KV)

    # m_i: the running maximum. We have one for each query
    m_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) - float("inf")
    # l_i: the running sum. We have one for each query (as we sum the attention scores by rows)
    l_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) + 1.0
    # acc: the accumulator for the output, which is a group of rows of the O matrix
    O_block = tl.zeros([BLOCK_SIZE_Q, HEAD_DIM], dtype=tl.float32)

    # load the blocks of Q: it will stay in SRAM throughout
    Q_block = tl.load(Q_block_ptr)

    # Stage: 3 if causal, else 1

    if STAGE == 1 or STAGE == 3:
        # This step runs for non-causal attention or for the blocks to the left of the diagonal in the causal attention
        O_block, l_i, m_i = _attn_fwd_inner(
            O_block,
            l_i,
            m_i,
            Q_block,
            K_block_ptr,
            V_block_ptr,
            block_index_q,
            softmax_scale,
            BLOCK_SIZE_Q,
            BLOCK_SIZE_KV,
            4 - STAGE,
            offs_q,
            offs_kv,
            SEQ_LEN,
        )

    if STAGE == 3:
        # This step runs for the blocks to the right of the diagonal in the causal attention
        O_block, l_i, m_i = _attn_fwd_inner(
            O_block,
            l_i,
            m_i,
            Q_block,
            K_block_ptr,
            V_block_ptr,
            block_index_q,
            softmax_scale,
            BLOCK_SIZE_Q,
            BLOCK_SIZE_KV,
            2,
            offs_q,
            offs_kv,
            SEQ_LEN,
        )
    # epilogue
    m_i += tl.math.log(
        l_i
    )  # This is needed to compute the logsumexp for the backwards pass
    O_block = O_block / l_i[:, None]
    m_ptrs = M + index_batch_head * SEQ_LEN + offs_q
    tl.store(m_ptrs, m_i)
    tl.store(O_block_ptr, O_block.to(O.type.element_ty))


@triton.jit
def _attn_bwd_preprocess(
    O,
    dO,
    D,
    SEQ_LEN,
    BLOCK_SIZE_Q: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    block_index_q = tl.program_id(0)
    offs_q = block_index_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
    index_batch_head = tl.program_id(1)
    offs_dim = tl.arange(0, HEAD_DIM)
    # Load a single block of BLOCK_SIZE_Q rows of O
    O_block = tl.load(
        O
        + index_batch_head * HEAD_DIM * SEQ_LEN
        + offs_q[:, None] * HEAD_DIM
        + offs_dim[None, :]
    ).to(tl.float32)
    # Load a single block of BLOCK_SIZE_Q rows of dO
    dO_block = tl.load(
        dO
        + index_batch_head * HEAD_DIM * SEQ_LEN
        + offs_q[:, None] * HEAD_DIM
        + offs_dim[None, :]
    ).to(tl.float32)
    # Compute the D block
    D_block = tl.sum(dO_block * O_block, axis=1)  # Shape: (BLOCK_SIZE_Q,)
    # Store the D block
    D_block_ptrs = D + index_batch_head * SEQ_LEN + offs_q
    tl.store(D_block_ptrs, D_block)


@triton.jit
def _attn_bwd_dq(
    Q,
    K,
    V,
    softmax_scale,
    dO,
    dQ,
    dK,
    dV,
    M,
    D,
    stride_batch,
    stride_head,
    stride_seq,
    stride_dim,
    NUM_HEADS,
    SEQ_LEN,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    STAGE: tl.constexpr,
):
    index_batch_head = tl.program_id(2)
    index_batch = index_batch_head // NUM_HEADS
    index_head = index_batch_head % NUM_HEADS
    offset_batch_head = (stride_batch * index_batch + stride_head * index_head).to(
        tl.int64
    )
    # This is the offset that allows us to select the right sequence given the batch and head.
    offset_batch_head_seq = (index_batch_head * SEQ_LEN).to(tl.int64)

    # Make sure the pointers are in the right place w.r.t batch and head
    # The reason we don't access the blocks through make_block_ptr is because we need to use the range of offsets to apply the masking
    Q += offset_batch_head
    K += offset_batch_head
    V += offset_batch_head
    dO += offset_batch_head
    dQ += offset_batch_head
    dK += offset_batch_head
    dV += offset_batch_head

    # Make sure the pointers are in the right place w.r.t batch, head and sequence
    M += offset_batch_head_seq
    D += offset_batch_head_seq

    # load scales
    offs_dim = tl.arange(0, HEAD_DIM)

    index_block_kv = tl.program_id(0)

    start_q = index_block_kv * BLOCK_Q
    offs_q = start_q + tl.arange(0, BLOCK_Q)

    Q_block = tl.load(Q + offs_q[:, None] * stride_seq + offs_dim[None, :] * stride_dim)
    dQ_block = tl.zeros([BLOCK_Q, HEAD_DIM], dtype=tl.float32)
    dO_block = tl.load(
        dO + offs_q[:, None] * stride_seq + offs_dim[None, :] * stride_dim
    )

    M_block = tl.load(M + offs_q)
    M_block = M_block[:, None]

    offs_kv = tl.arange(0, BLOCK_KV)

    # We access the K and V as transposed blocks
    kT_ptrs = K + offs_kv[None, :] * stride_seq + offs_dim[:, None] * stride_dim
    vT_ptrs = V + offs_kv[None, :] * stride_seq + offs_dim[:, None] * stride_dim

    Di = tl.load(D + offs_q)

    curr_kv = 0
    # Causal early termination: query 位置 q 只需 KV 位置 0..q+BLOCK_Q-1，
    # 之后的 KV block 全被 mask 为 -inf，P=0 不贡献梯度。
    # 非 causal 时遍历全部 KV。
    # Ref: Dao et al., "FlashAttention-2," arXiv:2307.08691, Section 3.1.
    if STAGE == 3:
        num_steps = (start_q + BLOCK_Q + BLOCK_KV - 1) // BLOCK_KV
    else:
        num_steps = SEQ_LEN // BLOCK_KV
    for blk_idx in range(num_steps):
        K_T_block = tl.load(kT_ptrs)
        V_T_block = tl.load(vT_ptrs)
        QK_block = softmax_scale * tl.dot(Q_block, K_T_block)

        # Causal mask: 在 exp 之前将未来位置设为 -inf，与 forward 语义一致。
        # 修复: 原代码先算 exp 再 mask，导致死代码和 -1e6 近似误差。
        if STAGE == 3:
            offs_kv = curr_kv + tl.arange(0, BLOCK_KV)
            mask_block = offs_q[:, None] >= offs_kv[None, :]
            QK_block = tl.where(mask_block, QK_block, float('-inf'))

        P_block = tl.math.exp(QK_block - M_block)

        # Compute dP and dS.
        dP_block = tl.dot(dO_block, V_T_block.to(dO_block.dtype)).to(tl.float32)
        dS_block = P_block * (dP_block - Di[:, None])
        dS_block = dS_block.to(Q_block.dtype)
        # Compute dQ.
        # NOTE: We need to de-scale dq in the end, because kT was pre-scaled.
        dQ_block += softmax_scale * tl.dot(dS_block, tl.trans(K_T_block))
        # Increment pointers.
        curr_kv += BLOCK_KV
        kT_ptrs += BLOCK_KV * stride_seq
        vT_ptrs += BLOCK_KV * stride_seq

    dQ_block_ptrs = dQ + offs_q[:, None] * stride_seq + offs_dim[None, :] * stride_dim
    tl.store(dQ_block_ptrs, dQ_block)


@triton.jit
def _attn_bwd_dk_dv(
    Q,
    K,
    V,
    softmax_scale,
    dO,
    dQ,
    dK,
    dV,
    M,
    D,
    stride_batch,
    stride_head,
    stride_seq,
    stride_dim,
    NUM_HEADS,
    SEQ_LEN,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    STAGE: tl.constexpr,
):
    index_batch_head = tl.program_id(2)
    index_batch = index_batch_head // NUM_HEADS
    index_head = index_batch_head % NUM_HEADS
    offset_batch_head = (stride_batch * index_batch + stride_head * index_head).to(
        tl.int64
    )
    # This is the offset that allows us to select the right sequence given the batch and head.
    offset_batch_head_seq = (index_batch_head * SEQ_LEN).to(tl.int64)

    # Make sure the pointers are in the right place w.r.t batch and head
    # The reason we don't access the blocks through make_block_ptr is because we need to use the range of offsets to apply the masking
    Q += offset_batch_head
    K += offset_batch_head
    V += offset_batch_head
    dO += offset_batch_head
    dQ += offset_batch_head
    dK += offset_batch_head
    dV += offset_batch_head

    # Make sure the pointers are in the right place w.r.t batch, head and sequence
    M += offset_batch_head_seq
    D += offset_batch_head_seq

    # load scales
    offs_dim = tl.arange(0, HEAD_DIM)

    index_block_kv = tl.program_id(0)
    start_kv = index_block_kv * BLOCK_KV

    offs_kv = start_kv + tl.arange(0, BLOCK_KV)

    dV_block = tl.zeros([BLOCK_KV, HEAD_DIM], dtype=tl.float32)
    dK_block = tl.zeros([BLOCK_KV, HEAD_DIM], dtype=tl.float32)

    # load K and V: they stay in SRAM throughout the inner loop.
    K_block = tl.load(
        K + offs_kv[:, None] * stride_seq + offs_dim[None, :] * stride_dim
    )  # Shape: (BLOCK_KV1, HEAD_DIM)
    V_block = tl.load(
        V + offs_kv[:, None] * stride_seq + offs_dim[None, :] * stride_dim
    )  # Shape: (BLOCK_KV1, HEAD_DIM)

    offs_q = tl.arange(0, BLOCK_Q)

    # We access the Q as a transposed array, so that's why we treat offs_q as a column vector ans offs_dim as a row vector
    # This is equivalent to doing:
    # q_ptrs = Q + offs_q[:, None] * stride_seq + offs_dim[None, :] * stride_dim
    # qT_ptrs = tl.trans(q_ptrs)
    # We point to the first BLOCK_Q rows of Q for both the qT and dO pointers, inside the for loop we will move forward by BLOCK_Q rows at each iteration.
    qT_ptrs = Q + offs_q[None, :] * stride_seq + offs_dim[:, None] * stride_dim
    dO_ptrs = dO + offs_q[:, None] * stride_seq + offs_dim[None, :] * stride_dim

    # Causal early termination: KV 位置 k 只被 Q 位置 >= k 的 query 注意到，
    # 所以从 start_kv 向下对齐到 BLOCK_Q 边界开始迭代。
    # 非 causal 时从 0 开始遍历全部。
    # Ref: Dao et al., "FlashAttention-2," arXiv:2307.08691, Section 3.1.
    if STAGE == 3:
        lo_q = (start_kv // BLOCK_Q) * BLOCK_Q
    else:
        lo_q = 0
    curr_q = lo_q
    qT_ptrs += lo_q * stride_seq
    dO_ptrs += lo_q * stride_seq
    num_steps = (SEQ_LEN - lo_q) // BLOCK_Q
    for blk_idx in range(num_steps):
        # Load a block of Q
        qT_block = tl.load(qT_ptrs)
        # Load the logsumexp values for the queries in the current block
        offs_q = curr_q + tl.arange(0, BLOCK_Q)
        m = tl.load(M + offs_q)

        # (QK^T)^T = K(Q^T) = P^T
        QK_T_block = softmax_scale * tl.dot(K_block, qT_block)
        # softmax via logsumexp trick
        P_T_block = tl.math.exp(QK_T_block - m[None, :])

        if STAGE == 3:
            # Autoregressive masking: post-exp 置零是正确的，因为 M (logsumexp)
            # 在 forward 时已正确计算（包含 causal mask 信息），
            # 所以 exp(QK - M) 对被 mask 的位置不会产生大值。
            # 直接置零比 pre-exp -inf 更简洁。
            mask_block = (
                offs_q[None, :] >= offs_kv[:, None]
            )  # Shape: (BLOCK_KV, BLOCK_Q)
            P_T_block = tl.where(mask_block, P_T_block, 0.0)

        dO_block = tl.load(dO_ptrs)
        # dV_new = dV_old + P^T x dO
        dV_block += tl.dot(P_T_block.to(dO_block.dtype), dO_block)

        # Delta = rowsum(O * dO)
        Di = tl.load(D + offs_q)

        # dP^T = V x dO^T
        dpT_block = tl.dot(V_block.to(dO_block.dtype), tl.trans(dO_block)).to(tl.float32)

        # dS^T = P^T * (dP^T - Delta^T)
        dS_T_block = P_T_block * (dpT_block - Di[None, :])
        dS_T_block = dS_T_block.to(K_block.dtype)

        # dK_new = dK_old + dS^T x Q
        dK_block += softmax_scale * tl.dot(dS_T_block, tl.trans(qT_block))
        # Increment pointers.
        curr_q += BLOCK_Q
        qT_ptrs += BLOCK_Q * stride_seq
        dO_ptrs += BLOCK_Q * stride_seq

    # Write back dV.
    dV_block_ptrs = dV + offs_kv[:, None] * stride_seq + offs_dim[None, :] * stride_dim
    tl.store(dV_block_ptrs, dV_block)

    # Write back dK.
    dK_block_ptrs = dK + offs_kv[:, None] * stride_seq + offs_dim[None, :] * stride_dim
    tl.store(dK_block_ptrs, dK_block)


class TritonAttention(torch.autograd.Function):
    """
    Flash Attention v2 的 torch.autograd.Function 封装。

    === 功能 ===
    实现双向梯度的 Flash Attention，支持任意 seq_len（自动 padding）和 causal mask。

    === 关键优化 ===
    1. 分块计算：避免 O(N²) 中间注意力矩阵（Ref [1])
       - Forward: O(N·d) HBM IO，O(1) SRAM 相对于 seq_len
       - Backward: recomputation + early termination，O(N·d) HBM IO

    2. Online softmax（Ref [3])：
       - 维护运行最大值和归一化因子，稳定数值计算
       - 修正因子 α = exp(m_i - m_ij) 使累加器适配新的最大值

    3. Causal early termination（Ref [2]）：
       - Backward dQ: query q 只与 KV 0..q 有梯度
       - Backward dK/dV: KV k 只与 Q k..T 有梯度
       - 减少约 50% 无效计算

    4. Seq_len padding：
       - 自动 pad 到 BLOCK_SIZE 倍数（避免 grid 为 0）
       - Forward 后 unpad，backward unpad 梯度
       - 支持 CIFAR-100 等非对齐长度（seq_len=3071）

    === 数学背景 ===
    标准 attention: O = softmax(QK^T / √d)V

    Online softmax 分块：对于块 j ∈ 1..m
      1. 加载 Q_i, K_j, V_j
      2. s_ij = Q_i K_j^T / √d
      3. m_ij = max(s_ij)，更新全局 m_i ← max(m_i, m_ij)
      4. P_ij = exp(s_ij - m_ij)
      5. l_ij = sum(P_ij)，修正 l_i ← α·l_i + l_ij，其中 α = exp(m_i_old - m_i_new)
      6. O_i ← α·O_i + P_ij V_j

    最终 O_i ← O_i / l_i（归一化）

    参考文献：
      [1] Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention
          with IO-Awareness," NeurIPS 2022, arXiv:2205.14135.
      [2] Dao, "FlashAttention-2: Faster Attention with Better Parallelism and
          Work Partitioning," arXiv:2307.08691, 2023.
      [3] Milakov & Gimelshein, "Online normalizer calculation for softmax,"
          arXiv:1805.02867, 2018.
    """

    @staticmethod
    def forward(ctx, Q, K, V, causal, softmax_scale):
        """
        Flash Attention 前向计算。

        参数:
          Q, K, V: (batch, num_heads, seq_len, head_dim)
          causal: bool，是否应用 causal mask（自回归）
          softmax_scale: float = 1/√d，注意力缩放因子

        返回:
          O: (batch, num_heads, seq_len, head_dim)，注意力输出

        === 算法步骤 ===
        1. Padding: seq_len 对齐到 BLOCK_MACRO 倍数
           - 避免 kernel grid 计算为 0
           - Pad 区域值为 0，causal mask 下对结果无影响

        2. Forward kernel（_attn_fwd）：
           - grid = (num_query_blocks, batch*num_heads, 1)
           - 每个 program 处理一个 query 块和所有 KV 块
           - 使用 online softmax 累加注意力
           - 保存 logsumexp M 供 backward 使用

        3. Unpadding: 返回原始 seq_len 的输出

        计算复杂度：
          - 时间: O(N²) ops（标准 attention）
          - HBM IO: O(N·d)（vs 标准 O(N²)）
          - SRAM: O(1) per query block（vs 标准 O(N)）

        数值稳定性：
          - Online softmax 使用最大值修正，避免溢出
          - 支持 float16 和 float32
        """
        HEAD_DIM_Q, HEAD_DIM_K = Q.shape[-1], K.shape[-1]
        HEAD_DIM_V = V.shape[-1]

        BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM = Q.shape

        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V

        # ── Padding: seq_len 对齐到 BLOCK_MACRO 的倍数 ──
        # 避免 kernel grid 计算出 0 或者越界访问。
        # Pad 区域填零，在 causal mask 下不影响非 pad 位置的结果。
        ALIGN = BLOCK_MACRO  # 使用最大 block size 对齐
        PAD = (ALIGN - SEQ_LEN % ALIGN) % ALIGN
        if PAD > 0:
            Q = torch.nn.functional.pad(Q, (0, 0, 0, PAD))  # pad seq dim
            K = torch.nn.functional.pad(K, (0, 0, 0, PAD))
            V = torch.nn.functional.pad(V, (0, 0, 0, PAD))
        SEQ_LEN_PADDED = SEQ_LEN + PAD

        import os
        if os.environ.get("DEBUG_MEM"):
            free, total = torch.cuda.mem_get_info()
            alloc = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"[flash_attn] Q.shape={tuple(Q.shape)} dtype={Q.dtype} "
                  f"Q_size={Q.numel()*Q.element_size()/1024**2:.1f}MB | "
                  f"alloc={alloc:.2f}GB reserved={reserved:.2f}GB "
                  f"free={free/1024**3:.2f}GB/{total/1024**3:.2f}GB", flush=True)
        O = torch.empty_like(Q)
        stage = 3 if causal else 1

        # grid: 不再使用 lambda（autotune 已移除），直接计算
        grid = (
            triton.cdiv(SEQ_LEN_PADDED, BLOCK_Q_DEFAULT),
            BATCH_SIZE * NUM_HEADS,
            1,
        )

        # M 是 logsumexp，backward 需要（每个 query 一个标量）
        M = torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN_PADDED), device=Q.device, dtype=torch.float32
        )

        _attn_fwd[grid](
            Q=Q,
            K=K,
            V=V,
            softmax_scale=softmax_scale,
            M=M,
            O=O,
            stride_Q_batch=Q.stride(0),
            stride_Q_head=Q.stride(1),
            stride_Q_seq=Q.stride(2),
            stride_Q_dim=Q.stride(3),
            stride_K_batch=K.stride(0),
            stride_K_head=K.stride(1),
            stride_K_seq=K.stride(2),
            stride_K_dim=K.stride(3),
            stride_V_batch=V.stride(0),
            stride_V_head=V.stride(1),
            stride_V_seq=V.stride(2),
            stride_V_dim=V.stride(3),
            stride_O_batch=O.stride(0),
            stride_O_head=O.stride(1),
            stride_O_seq=O.stride(2),
            stride_O_dim=O.stride(3),
            BATCH_SIZE=BATCH_SIZE,
            NUM_HEADS=NUM_HEADS,
            SEQ_LEN=SEQ_LEN_PADDED,
            HEAD_DIM=HEAD_DIM,
            BLOCK_SIZE_Q=BLOCK_Q_DEFAULT,
            BLOCK_SIZE_KV=BLOCK_KV_DEFAULT,
            STAGE=stage,
            num_warps=4,
            num_stages=3,
        )

        # 保存 padded 版本供 backward 使用（避免重复 padding）
        ctx.save_for_backward(Q, K, V, O, M)
        ctx.softmax_scale = softmax_scale
        ctx.HEAD_DIM = HEAD_DIM
        ctx.causal = causal
        ctx.SEQ_LEN_ORIG = SEQ_LEN
        ctx.PAD = PAD

        # Unpad 输出
        if PAD > 0:
            O = O[:, :, :SEQ_LEN, :]
        return O

    @staticmethod
    def backward(ctx, dO):
        """
        Flash Attention 反向计算。

        参数:
          dO: (batch, num_heads, seq_len, head_dim)，来自上游的梯度

        返回:
          (dQ, dK, dV, None, None): 对 (Q, K, V, causal, softmax_scale) 的梯度

        === 算法（Ref [2] 和 recomputation）===

        标准反向：
          P_ij = softmax(QK^T / √d)        # forward 保存
          ∂L/∂Q = (∂L/∂P · P) · K^T
          ∂L/∂K = Q^T · (∂L/∂P · P)
          ∂L/∂V = P^T · ∂L/∂O

        Flash Attention v2 优化：
          - P 不保存，而是通过 logsumexp M 和输出 O 重新计算
            P_ij = exp(s_ij - m_i)，其中 m_i = logsumexp(s_i)
          - Recomputation：避免 O(N²) 中间存储

        反向三个内核：
          1. _attn_bwd_preprocess: 计算 D_i = rowsum(O_i ⊙ dO_i)
             用于 dS 计算：dS = P ⊙ (dP - D)
             Ref: [2] eq.(9)

          2. _attn_bwd_dk_dv: 固定 KV，遍历所有 Q 块
             dK += softmax_scale · dS^T · Q
             dV += P^T · dO
             **Causal 优化**: KV k 从 Q 块 k // BLOCK_Q 开始迭代
             （之前的 Q 块对 KV k 无梯度）

          3. _attn_bwd_dq: 固定 Q，遍历所有 KV 块
             dQ += softmax_scale · dS · K^T
             **Causal 优化**: query 块 q 最多迭代到 (q+1) 的 KV 块
             （之后的 KV 对此 Q 块完全被 mask）

        === 关键修复 ===
        - Causal mask：在 exp 之前应用 mask 为 -inf（pre-exp）
          避免死代码和 float32 下的近似误差（-1e6 vs -inf）
        - Early termination 减少无效计算 ~50%

        计算复杂度：
          - 时间: O(N²) ops（标准 attention 反向）
          - HBM IO: O(N·d)（vs 标准 O(N²)）
          - SRAM: O(1) per block
        """
        Q, K, V, O, M = ctx.saved_tensors
        PAD = ctx.PAD
        SEQ_LEN_ORIG = ctx.SEQ_LEN_ORIG

        # Pad dO 以匹配 forward 保存的 padded 张量
        if PAD > 0:
            dO = torch.nn.functional.pad(dO, (0, 0, 0, PAD))

        dO = dO.contiguous()
        assert Q.is_contiguous() and K.is_contiguous() and V.is_contiguous()
        dQ = torch.empty_like(Q)
        dK = torch.empty_like(K)
        dV = torch.empty_like(V)

        BATCH_SIZE, NUM_HEADS, SEQ_LEN = Q.shape[:3]
        NUM_WARPS, NUM_STAGES = 4, 3

        preprocess_grid = (SEQ_LEN // BLOCK_MACRO, BATCH_SIZE * NUM_HEADS)
        D = torch.empty_like(M)  # Shape: (BATCH_SIZE, NUM_HEADS, SEQ_LEN)

        # 预计算 D_i = rowsum(O_i * dO_i)
        _attn_bwd_preprocess[preprocess_grid](
            O=O,
            dO=dO,
            D=D,
            SEQ_LEN=SEQ_LEN,
            BLOCK_SIZE_Q=BLOCK_MACRO,
            HEAD_DIM=ctx.HEAD_DIM,
        )

        grid = (SEQ_LEN // BLOCK_MACRO, 1, BATCH_SIZE * NUM_HEADS)

        stage = 3 if ctx.causal else 1

        # 固定 KV，遍历所有 Q block → 计算 dK, dV
        _attn_bwd_dk_dv[grid](
            Q=Q,
            K=K,
            V=V,
            softmax_scale=ctx.softmax_scale,
            dO=dO,
            dQ=dQ,
            dK=dK,
            dV=dV,
            M=M,
            D=D,
            stride_batch=Q.stride(0),
            stride_head=Q.stride(1),
            stride_seq=Q.stride(2),
            stride_dim=Q.stride(3),
            NUM_HEADS=NUM_HEADS,
            SEQ_LEN=SEQ_LEN,
            BLOCK_Q=BLOCK_MICRO,
            BLOCK_KV=BLOCK_MACRO,
            HEAD_DIM=ctx.HEAD_DIM,
            STAGE=stage,
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES,
        )

        # 固定 Q，遍历所有 KV block → 计算 dQ
        _attn_bwd_dq[grid](
            Q=Q,
            K=K,
            V=V,
            softmax_scale=ctx.softmax_scale,
            dO=dO,
            dQ=dQ,
            dK=dK,
            dV=dV,
            M=M,
            D=D,
            stride_batch=Q.stride(0),
            stride_head=Q.stride(1),
            stride_seq=Q.stride(2),
            stride_dim=Q.stride(3),
            NUM_HEADS=NUM_HEADS,
            SEQ_LEN=SEQ_LEN,
            BLOCK_Q=BLOCK_MACRO,
            BLOCK_KV=BLOCK_MICRO,
            HEAD_DIM=ctx.HEAD_DIM,
            STAGE=stage,
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES,
        )

        # Unpad 梯度
        if PAD > 0:
            dQ = dQ[:, :, :SEQ_LEN_ORIG, :]
            dK = dK[:, :, :SEQ_LEN_ORIG, :]
            dV = dV[:, :, :SEQ_LEN_ORIG, :]

        return dQ, dK, dV, None, None

def _test_op(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, causal, dtype=torch.float16):
    """
    测试 Triton Flash Attention 与 PyTorch 标准实现的等价性。
    支持任意 SEQ_LEN（含非 block 对齐的情况）。
    """
    Q = (
        torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda"
        )
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    K = (
        torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda"
        )
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    V = (
        torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda"
        )
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )

    softmax_scale = 1 / (HEAD_DIM**0.5)
    dO = torch.randn_like(Q)

    # reference implementation
    MASK = torch.tril(torch.ones((SEQ_LEN, SEQ_LEN), device="cuda"))
    P = torch.matmul(Q, K.transpose(2, 3)) * softmax_scale
    if causal:
        P[:, :, MASK == 0] = float("-inf")
    P = torch.softmax(P.float(), dim=-1).to(dtype)
    ref_O = torch.matmul(P, V)
    ref_O.backward(dO)
    ref_dV, V.grad = V.grad.clone(), None
    ref_dK, K.grad = K.grad.clone(), None
    ref_dQ, Q.grad = Q.grad.clone(), None

    # triton implementation
    tri_out = TritonAttention.apply(Q, K, V, causal, softmax_scale).to(dtype)
    tri_out.backward(dO)
    tri_dV, V.grad = V.grad.clone(), None
    tri_dK, K.grad = K.grad.clone(), None
    tri_dQ, Q.grad = Q.grad.clone(), None

    # compare
    rtol = 0.0
    atol = 1e-2
    assert torch.allclose(ref_O, tri_out, atol=atol, rtol=rtol), \
        f"O mismatch: max diff={torch.max(torch.abs(ref_O - tri_out)).item():.6f}"
    assert torch.allclose(ref_dK, tri_dK, atol=atol, rtol=rtol), \
        f"dK mismatch: max diff={torch.max(torch.abs(ref_dK - tri_dK)).item():.6f}"
    assert torch.allclose(ref_dV, tri_dV, atol=atol, rtol=rtol), \
        f"dV mismatch: max diff={torch.max(torch.abs(ref_dV - tri_dV)).item():.6f}"
    assert torch.allclose(ref_dQ, tri_dQ, atol=atol, rtol=rtol), \
        f"dQ mismatch: max diff={torch.max(torch.abs(ref_dQ - tri_dQ)).item():.6f}"