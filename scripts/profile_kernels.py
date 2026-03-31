#!/usr/bin/env python3
"""
Triton Kernel Profiling 工具 — 对比 fused kernel vs PyTorch 实现的性能。

对每个手写 Triton kernel 运行 warmup + benchmark，输出:
  1. 延迟 (ms): fused vs pytorch
  2. 加速比: pytorch_time / fused_time
  3. 显存占用差异（forward 峰值）
  4. Markdown 格式汇总表格

支持 forward-only 和 forward+backward 两种模式。

Usage:
    # 全部 kernel benchmark
    python scripts/profile_kernels.py

    # 指定 kernel
    python scripts/profile_kernels.py --kernel fused_rms_norm fused_swiglu

    # 自定义 shape
    python scripts/profile_kernels.py --batch 64 --seq_len 3072 --d_model 128

    # 包含 backward
    python scripts/profile_kernels.py --backward

参考:
  Triton 官方 benchmark utilities:
  https://triton-lang.org/main/getting-started/tutorials/
"""

import os
import sys
import time
import argparse
import torch
import torch.nn.functional as F
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# ── Benchmark 工具函数 ──────────────────────────────────────────
def benchmark_fn(fn, warmup=10, repeats=50):
    """
    运行 fn() 多次，返回中位延迟(ms)。
    使用 CUDA event 计时，避免 CPU-GPU 同步延迟影响。
    """
    # warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    # 计时
    times = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    times.sort()
    # 返回中位数（避免 outlier 影响）
    median = times[len(times) // 2]
    return median


def measure_memory(fn):
    """
    测量 fn() 的 CUDA 峰值显存分配(MB)。
    """
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 1024 / 1024
    return peak


# ── Kernel Benchmarks ───────────────────────────────────────────

def bench_fused_rms_norm(M, N, dtype, backward=False):
    """RMSNorm: fused triton vs pytorch"""
    from src.mdlic.ops.fused_rms_norm import fused_rms_norm

    x = torch.randn(M, N, device="cuda", dtype=dtype, requires_grad=backward)
    w = torch.ones(N, device="cuda", dtype=dtype, requires_grad=backward)
    eps = 1e-10

    def fused_fn():
        out = fused_rms_norm(x, w, eps)
        if backward:
            out.sum().backward(retain_graph=True)

    def pytorch_fn():
        rms = x.float().pow(2).mean(-1, keepdim=True).add(eps).sqrt()
        out = (w.float() * (x.float() / rms)).to(dtype)
        if backward:
            out.sum().backward(retain_graph=True)

    t_fused = benchmark_fn(fused_fn)
    t_pytorch = benchmark_fn(pytorch_fn)
    mem_fused = measure_memory(fused_fn)
    mem_pytorch = measure_memory(pytorch_fn)

    return t_fused, t_pytorch, mem_fused, mem_pytorch


def bench_fused_ce_zloss(M, V, dtype, backward=False):
    """Cross-Entropy + z-loss: fused triton vs pytorch"""
    from src.mdlic.ops.fused_ce_zloss import fused_cross_entropy_zloss

    logits = torch.randn(M, V, device="cuda", dtype=dtype, requires_grad=backward)
    targets = torch.randint(0, V, (M,), device="cuda")
    z_w = 1e-4

    def fused_fn():
        ce, z = fused_cross_entropy_zloss(logits, targets, z_loss_weight=z_w)
        loss = ce + z_w * z
        if backward:
            loss.backward(retain_graph=True)

    def pytorch_fn():
        ce = F.cross_entropy(logits.float(), targets)
        lse = torch.logsumexp(logits.float(), dim=-1)
        z = (lse ** 2).mean()
        loss = ce + z_w * z
        if backward:
            loss.backward(retain_graph=True)

    t_fused = benchmark_fn(fused_fn)
    t_pytorch = benchmark_fn(pytorch_fn)
    mem_fused = measure_memory(fused_fn)
    mem_pytorch = measure_memory(pytorch_fn)

    return t_fused, t_pytorch, mem_fused, mem_pytorch


def bench_fused_swiglu(M, N, dtype, backward=False):
    """SwiGLU: fused triton vs pytorch"""
    from src.mdlic.ops.fused_swiglu import fused_swiglu

    a = torch.randn(M, N, device="cuda", dtype=dtype, requires_grad=backward)
    b = torch.randn(M, N, device="cuda", dtype=dtype, requires_grad=backward)

    def fused_fn():
        out = fused_swiglu(a, b)
        if backward:
            out.sum().backward(retain_graph=True)

    def pytorch_fn():
        out = F.silu(a.float()).to(dtype) * b
        if backward:
            out.sum().backward(retain_graph=True)

    t_fused = benchmark_fn(fused_fn)
    t_pytorch = benchmark_fn(pytorch_fn)
    mem_fused = measure_memory(fused_fn)
    mem_pytorch = measure_memory(pytorch_fn)

    return t_fused, t_pytorch, mem_fused, mem_pytorch


def bench_fused_rope(B, h, T, d_k, dtype, backward=False):
    """RoPE: fused triton vs pytorch"""
    from src.mdlic.ops.fused_rope import fused_apply_rotary_emb
    from src.mdlic.models.layers import RotaryEmbedding, apply_rotary_emb

    rope = RotaryEmbedding(d_k).to("cuda")
    cos, sin = rope(T, torch.device("cuda"))
    cos, sin = cos.to(dtype), sin.to(dtype)

    def fused_fn():
        q = torch.randn(B, h, T, d_k, device="cuda", dtype=dtype).contiguous()
        k = torch.randn(B, h, T, d_k, device="cuda", dtype=dtype).contiguous()
        fused_apply_rotary_emb(q, k, cos, sin)

    def pytorch_fn():
        q = torch.randn(B, h, T, d_k, device="cuda", dtype=dtype)
        k = torch.randn(B, h, T, d_k, device="cuda", dtype=dtype)
        apply_rotary_emb(q, k, cos, sin)

    t_fused = benchmark_fn(fused_fn)
    t_pytorch = benchmark_fn(pytorch_fn)
    mem_fused = measure_memory(fused_fn)
    mem_pytorch = measure_memory(pytorch_fn)

    return t_fused, t_pytorch, mem_fused, mem_pytorch


def bench_fused_add_rms_norm(M, N, dtype, backward=False):
    """Add+RMSNorm: fused triton vs pytorch"""
    from src.mdlic.ops.fused_add_rms_norm import fused_add_rms_norm

    residual = torch.randn(M, N, device="cuda", dtype=dtype, requires_grad=backward)
    sublayer = torch.randn(M, N, device="cuda", dtype=dtype, requires_grad=backward)
    w = torch.ones(N, device="cuda", dtype=dtype, requires_grad=backward)
    eps = 1e-10

    def fused_fn():
        out = fused_add_rms_norm(residual, sublayer, w, eps)
        if backward:
            out.sum().backward(retain_graph=True)

    def pytorch_fn():
        s = sublayer.float()
        rms = s.pow(2).mean(-1, keepdim=True).add(eps).sqrt()
        out = residual + (w.float() * (s / rms)).to(dtype)
        if backward:
            out.sum().backward(retain_graph=True)

    t_fused = benchmark_fn(fused_fn)
    t_pytorch = benchmark_fn(pytorch_fn)
    mem_fused = measure_memory(fused_fn)
    mem_pytorch = measure_memory(pytorch_fn)

    return t_fused, t_pytorch, mem_fused, mem_pytorch


def bench_flash_attn(B, h, T, d_k, dtype, backward=False):
    """Flash Attention v2: triton vs pytorch SDPA"""
    from src.mdlic.ops.flash_attn import TritonAttention

    scale = 1.0 / math.sqrt(d_k)

    def fused_fn():
        q = torch.randn(B, h, T, d_k, device="cuda", dtype=dtype, requires_grad=backward)
        k = torch.randn(B, h, T, d_k, device="cuda", dtype=dtype, requires_grad=backward)
        v = torch.randn(B, h, T, d_k, device="cuda", dtype=dtype, requires_grad=backward)
        out = TritonAttention.apply(q, k, v, True, scale)
        if backward:
            out.sum().backward()

    def pytorch_fn():
        q = torch.randn(B, h, T, d_k, device="cuda", dtype=dtype, requires_grad=backward)
        k = torch.randn(B, h, T, d_k, device="cuda", dtype=dtype, requires_grad=backward)
        v = torch.randn(B, h, T, d_k, device="cuda", dtype=dtype, requires_grad=backward)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        if backward:
            out.sum().backward()

    t_fused = benchmark_fn(fused_fn)
    t_pytorch = benchmark_fn(pytorch_fn)
    mem_fused = measure_memory(fused_fn)
    mem_pytorch = measure_memory(pytorch_fn)

    return t_fused, t_pytorch, mem_fused, mem_pytorch


def bench_fused_attn_rope(B, h, T, d_k, dtype, backward=False):
    """Fused Attn+RoPE: triton fused vs 分步 RoPE + SDPA"""
    from src.mdlic.ops.fused_attn_rope import fused_attn_rope
    from src.mdlic.models.layers import RotaryEmbedding, apply_rotary_emb

    rope = RotaryEmbedding(d_k).to("cuda")
    cos, sin = rope(T, torch.device("cuda"))
    cos, sin = cos.to(dtype), sin.to(dtype)
    scale = 1.0 / math.sqrt(d_k)

    def fused_fn():
        q = torch.randn(B, h, T, d_k, device="cuda", dtype=dtype,
                         requires_grad=backward).contiguous()
        k = torch.randn(B, h, T, d_k, device="cuda", dtype=dtype,
                         requires_grad=backward).contiguous()
        v = torch.randn(B, h, T, d_k, device="cuda", dtype=dtype,
                         requires_grad=backward)
        out = fused_attn_rope(q, k, v, cos, sin, causal=True,
                              softmax_scale=scale)
        if backward:
            out.sum().backward()

    def pytorch_fn():
        q = torch.randn(B, h, T, d_k, device="cuda", dtype=dtype,
                         requires_grad=backward)
        k = torch.randn(B, h, T, d_k, device="cuda", dtype=dtype,
                         requires_grad=backward)
        v = torch.randn(B, h, T, d_k, device="cuda", dtype=dtype,
                         requires_grad=backward)
        q_r, k_r = apply_rotary_emb(q, k, cos, sin)
        out = F.scaled_dot_product_attention(q_r, k_r, v, is_causal=True)
        if backward:
            out.sum().backward()

    t_fused = benchmark_fn(fused_fn)
    t_pytorch = benchmark_fn(pytorch_fn)
    mem_fused = measure_memory(fused_fn)
    mem_pytorch = measure_memory(pytorch_fn)

    return t_fused, t_pytorch, mem_fused, mem_pytorch


def bench_fused_linear_ce(M, D, V, dtype, backward=False):
    """Fused Linear+CE+z-loss: triton fused vs 分步 linear + CE + logsumexp"""
    from src.mdlic.ops.fused_linear_ce import fused_linear_cross_entropy

    hidden = torch.randn(M, D, device="cuda", dtype=torch.float32,
                          requires_grad=backward)
    weight = torch.randn(V, D, device="cuda", dtype=torch.float32)
    targets = torch.randint(0, V, (M,), device="cuda")
    z_w = 1e-4

    def fused_fn():
        h = hidden.detach().clone().requires_grad_(backward)
        ce, z = fused_linear_cross_entropy(h, weight, targets, z_loss_weight=z_w)
        loss = ce + z_w * z
        if backward:
            loss.backward()

    def pytorch_fn():
        h = hidden.detach().clone().requires_grad_(backward)
        logits = h @ weight.T
        ce = F.cross_entropy(logits, targets)
        lse = torch.logsumexp(logits, dim=-1)
        z = (lse ** 2).mean()
        loss = ce + z_w * z
        if backward:
            loss.backward()

    t_fused = benchmark_fn(fused_fn)
    t_pytorch = benchmark_fn(pytorch_fn)
    mem_fused = measure_memory(fused_fn)
    mem_pytorch = measure_memory(pytorch_fn)

    return t_fused, t_pytorch, mem_fused, mem_pytorch


def estimate_flops_bytes(kernel_name, M, N, B, h, T, d_k, V, dtype_bytes):
    """
    估算各 kernel 的理论 FLOPs 和 HBM 读写字节数（forward-only）。

    用于计算 arithmetic intensity = FLOPs / bytes，
    判断 kernel 是 compute-bound 还是 memory-bound。

    参考:
      [1] Williams et al., "Roofline: An Insightful Visual Performance Model,"
          Communications of the ACM, 2009.

    返回:
      (flops, bytes_rw) — 理论 FLOPs, HBM 读写字节数
    """
    db = dtype_bytes  # 每元素字节数

    if kernel_name == "fused_rms_norm":
        # FLOPs: x² → mean → +eps → sqrt → /rms → *w  ≈ 5*M*N
        # Bytes: read x(M*N) + w(N), write out(M*N)
        flops = 5 * M * N
        bytes_rw = (2 * M * N + N) * db

    elif kernel_name == "fused_ce_zloss":
        # FLOPs: softmax(M*V) + log(M) + CE index(M)  ≈ 3*M*V
        # Bytes: read logits(M*V) + targets(M*4B), write ce+z(2)
        flops = 3 * M * V
        bytes_rw = M * V * db + M * 4

    elif kernel_name == "fused_swiglu":
        # FLOPs: silu(a) = a*sigmoid(a) ≈ 3*M*N, then ⊙b ≈ M*N → 4*M*N
        # Bytes: read a(M*N) + b(M*N), write out(M*N)
        flops = 4 * M * N
        bytes_rw = 3 * M * N * db

    elif kernel_name == "fused_rope":
        # FLOPs: per element 6 ops (2 mul + 2 mul + 2 add) × Q + K
        # Bytes: read/write Q(B*h*T*d_k) + K(B*h*T*d_k) + cos/sin(T*d_k)
        numel = B * h * T * d_k
        flops = 12 * numel  # 6 ops × 2 (Q and K)
        bytes_rw = (4 * numel + 2 * T * d_k) * db  # read Q,K + write Q,K + cos,sin

    elif kernel_name == "fused_add_rms_norm":
        # FLOPs: add(M*N) + rms(5*M*N) ≈ 6*M*N
        # Bytes: read residual(M*N) + sublayer(M*N) + w(N), write out(M*N)
        flops = 6 * M * N
        bytes_rw = (3 * M * N + N) * db

    elif kernel_name == "flash_attn":
        # FLOPs: 2*B*h*T²*d_k (QK^T) + 2*B*h*T²*d_k (softmax·V) ≈ 4*B*h*T²*d_k
        # Causal: 约一半 → 2*B*h*T²*d_k
        # Bytes: read Q,K,V (3*B*h*T*d_k), write O (B*h*T*d_k)
        flops = 2 * B * h * T * T * d_k
        bytes_rw = 4 * B * h * T * d_k * db

    elif kernel_name == "fused_attn_rope":
        # RoPE FLOPs + Flash Attn FLOPs
        numel = B * h * T * d_k
        rope_flops = 12 * numel
        attn_flops = 2 * B * h * T * T * d_k
        flops = rope_flops + attn_flops
        bytes_rw = 4 * B * h * T * d_k * db  # 同 flash_attn（RoPE 就地）

    elif kernel_name == "fused_linear_ce":
        # FLOPs: matmul(M*D*V) + softmax(M*V) ≈ 2*M*D*V + 3*M*V
        D = N  # d_model
        flops = 2 * M * D * V + 3 * M * V
        # Bytes: read hidden(M*D) + weight(V*D), write ce+z(2 scalars)
        bytes_rw = (M * D + V * D) * db

    else:
        flops = 0
        bytes_rw = 1  # 避免除零

    return flops, max(bytes_rw, 1)


# ── 主程序 ──────────────────────────────────────────────────────

ALL_KERNELS = [
    "fused_rms_norm", "fused_ce_zloss", "fused_swiglu",
    "fused_rope", "fused_add_rms_norm", "flash_attn",
    "fused_attn_rope", "fused_linear_ce"
]


def main():
    parser = argparse.ArgumentParser(description="Triton Kernel Profiling")
    parser.add_argument('--kernel', nargs='+', default=ALL_KERNELS,
                        choices=ALL_KERNELS,
                        help='要 benchmark 的 kernel（默认全部）')
    parser.add_argument('--batch', type=int, default=64, help='batch size')
    parser.add_argument('--seq_len', type=int, default=3072,
                        help='序列长度 (默认 3072 = 32×32×3)')
    parser.add_argument('--d_model', type=int, default=128, help='模型维度')
    parser.add_argument('--d_ff', type=int, default=384, help='FFN 维度')
    parser.add_argument('--h', type=int, default=4, help='注意力头数')
    parser.add_argument('--vocab', type=int, default=256, help='词表大小')
    parser.add_argument('--dtype', choices=['fp16', 'bf16', 'fp32'],
                        default='bf16', help='数据类型')
    parser.add_argument('--backward', action='store_true',
                        help='同时 benchmark backward')
    parser.add_argument('--roofline', action='store_true',
                        help='输出 roofline 分析（arithmetic intensity + 瓶颈判断）')
    parser.add_argument('--warmup', type=int, default=10, help='warmup 轮数')
    parser.add_argument('--repeats', type=int, default=50, help='重复轮数')
    args = parser.parse_args()

    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    dtype = dtype_map[args.dtype]

    B = args.batch
    T = args.seq_len
    d_model = args.d_model
    d_ff = args.d_ff
    h = args.h
    d_k = d_model // h
    V = args.vocab
    M = B * T  # 总 token 数
    dtype_bytes = {torch.float16: 2, torch.bfloat16: 2, torch.float32: 4}[dtype]

    mode = "forward+backward" if args.backward else "forward-only"
    print(f"{'='*80}")
    print(f"  Triton Kernel Profiling — {mode}")
    print(f"  B={B}, T={T}, d_model={d_model}, d_ff={d_ff}, h={h}, V={V}")
    print(f"  dtype={args.dtype}, warmup={args.warmup}, repeats={args.repeats}")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"{'='*80}\n")

    results = []

    for kernel_name in args.kernel:
        print(f"Benchmarking {kernel_name}...", end="", flush=True)
        try:
            if kernel_name == "fused_rms_norm":
                shape_desc = f"({M}, {d_model})"
                t_f, t_p, m_f, m_p = bench_fused_rms_norm(
                    M, d_model, dtype, args.backward)

            elif kernel_name == "fused_ce_zloss":
                shape_desc = f"({M}, {V})"
                t_f, t_p, m_f, m_p = bench_fused_ce_zloss(
                    M, V, dtype, args.backward)

            elif kernel_name == "fused_swiglu":
                shape_desc = f"({M}, {d_ff})"
                t_f, t_p, m_f, m_p = bench_fused_swiglu(
                    M, d_ff, dtype, args.backward)

            elif kernel_name == "fused_rope":
                shape_desc = f"({B}, {h}, {T}, {d_k})"
                t_f, t_p, m_f, m_p = bench_fused_rope(
                    B, h, T, d_k, dtype, args.backward)

            elif kernel_name == "fused_add_rms_norm":
                shape_desc = f"({M}, {d_model})"
                t_f, t_p, m_f, m_p = bench_fused_add_rms_norm(
                    M, d_model, dtype, args.backward)

            elif kernel_name == "flash_attn":
                shape_desc = f"({B}, {h}, {T}, {d_k})"
                t_f, t_p, m_f, m_p = bench_flash_attn(
                    B, h, T, d_k, dtype, args.backward)

            elif kernel_name == "fused_attn_rope":
                shape_desc = f"({B}, {h}, {T}, {d_k})"
                t_f, t_p, m_f, m_p = bench_fused_attn_rope(
                    B, h, T, d_k, dtype, args.backward)

            elif kernel_name == "fused_linear_ce":
                shape_desc = f"({M}, {d_model}, {V})"
                t_f, t_p, m_f, m_p = bench_fused_linear_ce(
                    M, d_model, V, dtype, args.backward)

            speedup = t_p / t_f if t_f > 0 else float('inf')
            mem_save = (m_p - m_f) / m_p * 100 if m_p > 0 else 0

            # Roofline 分析
            flops_est, bytes_est = estimate_flops_bytes(
                kernel_name, M, d_model, B, h, T, d_k, V, dtype_bytes)
            arith_intensity = flops_est / bytes_est  # FLOPs/byte
            # 实际吞吐: FLOPs / 时间
            actual_tflops = (flops_est / (t_f * 1e-3)) / 1e12 if t_f > 0 else 0

            results.append({
                "kernel": kernel_name,
                "shape": shape_desc,
                "fused_ms": t_f,
                "pytorch_ms": t_p,
                "speedup": speedup,
                "fused_mem_mb": m_f,
                "pytorch_mem_mb": m_p,
                "mem_save_pct": mem_save,
                "flops": flops_est,
                "bytes": bytes_est,
                "arith_intensity": arith_intensity,
                "actual_tflops": actual_tflops,
            })
            print(f" {speedup:.2f}x")

        except Exception as e:
            print(f" SKIP ({e})")
            results.append({
                "kernel": kernel_name,
                "shape": "N/A",
                "fused_ms": 0, "pytorch_ms": 0,
                "speedup": 0,
                "fused_mem_mb": 0, "pytorch_mem_mb": 0,
                "mem_save_pct": 0,
                "error": str(e),
            })

    # ── 打印 Markdown 汇总表格 ──
    print(f"\n{'='*80}")
    print(f"  性能对比表格 ({mode})")
    print(f"{'='*80}\n")

    print("| Kernel | Shape | Fused (ms) | PyTorch (ms) | Speedup | Fused Mem (MB) | PyTorch Mem (MB) | Mem Save |")
    print("|--------|-------|-----------|-------------|---------|---------------|-----------------|----------|")

    for r in results:
        if "error" in r:
            print(f"| {r['kernel']} | {r['shape']} | — | — | SKIP | — | — | {r['error']} |")
        else:
            print(f"| {r['kernel']} | {r['shape']} | {r['fused_ms']:.3f} | {r['pytorch_ms']:.3f} | **{r['speedup']:.2f}x** | {r['fused_mem_mb']:.1f} | {r['pytorch_mem_mb']:.1f} | {r['mem_save_pct']:.1f}% |")

    print()

    # 总结
    valid = [r for r in results if "error" not in r and r["speedup"] > 0]
    if valid:
        avg_speedup = sum(r["speedup"] for r in valid) / len(valid)
        max_speedup = max(r["speedup"] for r in valid)
        max_kernel = max(valid, key=lambda r: r["speedup"])["kernel"]
        print(f"平均加速比: {avg_speedup:.2f}x")
        print(f"最大加速比: {max_speedup:.2f}x ({max_kernel})")

    # ── Roofline 分析（可选）──
    # Ref: Williams et al., "Roofline: An Insightful Visual Performance Model,"
    #      Communications of the ACM, 2009.
    if args.roofline and valid:
        print(f"\n{'='*80}")
        print(f"  Roofline 分析")
        print(f"{'='*80}\n")

        print("| Kernel | FLOPs | Bytes (R+W) | AI (FLOPs/B) | Actual TFLOPS | Bottleneck |")
        print("|--------|-------|-------------|-------------|---------------|------------|")

        for r in valid:
            flops_str = f"{r['flops']/1e9:.2f}G" if r['flops'] >= 1e9 else f"{r['flops']/1e6:.1f}M"
            bytes_str = f"{r['bytes']/1e9:.2f}G" if r['bytes'] >= 1e9 else f"{r['bytes']/1e6:.1f}M"
            ai = r['arith_intensity']
            # 判断瓶颈: AI < ~10-50 FLOPs/byte 通常是 memory-bound（取决于 GPU）
            # H800: peak ~2000 TFLOPS (fp16), bandwidth ~3.35 TB/s → ridge point ≈ 600
            # 4090: peak ~330 TFLOPS (fp16), bandwidth ~1008 GB/s → ridge point ≈ 327
            # A100: peak ~312 TFLOPS (fp16), bandwidth ~2039 GB/s → ridge point ≈ 153
            # 保守判断: AI < 100 视为 memory-bound
            bottleneck = "Memory" if ai < 100 else "Compute"
            tflops_str = f"{r['actual_tflops']:.3f}"
            print(f"| {r['kernel']} | {flops_str} | {bytes_str} | {ai:.1f} | {tflops_str} | **{bottleneck}** |")

        print()
        print("判断标准: AI < 100 FLOPs/byte → Memory-bound（fusion 优化有效）")
        print("          AI ≥ 100 FLOPs/byte → Compute-bound（算法优化更重要）")
        print()


if __name__ == "__main__":
    main()
