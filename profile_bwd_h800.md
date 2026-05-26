================================================================================
  Triton Kernel Profiling — forward+backward
  B=64, T=3072, d_model=448, d_ff=1216, h=7, V=256
  dtype=bf16, warmup=20, repeats=200
  GPU: NVIDIA H800 PCIe
================================================================================

Benchmarking fused_rms_norm... 13.26x
Benchmarking fused_ce_zloss... 7.79x
Benchmarking fused_swiglu... 2.49x
Benchmarking fused_rope... 1.24x
Benchmarking fused_add_rms_norm... 10.13x
Benchmarking flash_attn... 1.20x
Benchmarking fused_attn_rope... 1.34x
Benchmarking fused_linear_ce... 0.01x

================================================================================
  性能对比表格 (forward+backward)
================================================================================

| Kernel | Shape | Fused (ms) | PyTorch (ms) | Speedup | Fused Mem (MB) | PyTorch Mem (MB) | Mem Save |
|--------|-------|-----------|-------------|---------|---------------|-----------------|----------|
| fused_rms_norm | (196608, 448) | 1.482 | 19.653 | **13.26x** | 1276.8 | 3192.8 | 60.0% |
| fused_ce_zloss | (196608, 256) | 3.171 | 24.690 | **7.79x** | 385.5 | 1155.0 | 66.6% |
| fused_swiglu | (196608, 1216) | 9.325 | 23.213 | **2.49x** | 3648.0 | 5472.0 | 33.3% |
| fused_rope | (64, 7, 3072, 64) | 4.843 | 6.028 | **1.24x** | 674.3 | 1010.3 | 33.3% |
| fused_add_rms_norm | (196608, 448) | 1.959 | 19.837 | **10.13x** | 1780.8 | 3192.8 | 44.2% |
| flash_attn | (64, 7, 3072, 64) | 20.353 | 24.372 | **1.20x** | 1354.5 | 1858.5 | 27.1% |
| fused_attn_rope | (64, 7, 3072, 64) | 28.705 | 38.482 | **1.34x** | 1692.8 | 2196.8 | 22.9% |
| fused_linear_ce | (196608, 448, 256) | 6112.437 | 32.286 | **0.01x** | 1409.9 | 1699.4 | 17.0% |

平均加速比: 5.35x  （7 个生效 kernel，排除反面案例 fused_linear_ce）
              4.68x （8 个 kernel 全量均值，含反面案例）
最大加速比: 13.26x (fused_rms_norm)

================================================================================
  Roofline 分析
================================================================================

| Kernel | FLOPs | Bytes (R+W) | AI (FLOPs/B) | Actual TFLOPS | Bottleneck |
|--------|-------|-------------|-------------|---------------|------------|
| fused_rms_norm | 440.4M | 352.3M | 1.2 | 0.297 | **Memory** |
| fused_ce_zloss | 151.0M | 101.4M | 1.5 | 0.048 | **Memory** |
| fused_swiglu | 352.3M | 528.5M | 0.7 | 0.038 | **Memory** |
| fused_rope | 1.06G | 705.4M | 1.5 | 0.218 | **Memory** |
| fused_add_rms_norm | 528.5M | 528.5M | 1.0 | 0.270 | **Memory** |
| flash_attn | 541.17G | 704.6M | 768.0 | 26.588 | **Compute** |
| fused_attn_rope | 542.22G | 704.6M | 769.5 | 18.889 | **Compute** |
| fused_linear_ce | 45.25G | 176.4M | 256.5 | 0.007 | **Compute** |

判断标准: AI < 100 FLOPs/byte → Memory-bound（fusion 优化有效）
          AI ≥ 100 FLOPs/byte → Compute-bound（算法优化更重要）

