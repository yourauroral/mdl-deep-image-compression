================================================================================
  Triton Kernel Profiling — forward-only
  B=64, T=3072, d_model=448, d_ff=1216, h=7, V=256
  dtype=bf16, warmup=20, repeats=200
  GPU: NVIDIA H800 PCIe
================================================================================

Benchmarking fused_rms_norm... 24.00x
Benchmarking fused_ce_zloss... 63.75x
Benchmarking fused_swiglu... 8.15x
Benchmarking fused_rope... 1.24x
Benchmarking fused_add_rms_norm... 16.82x
Benchmarking flash_attn... 1.08x
Benchmarking fused_attn_rope... 1.42x
Benchmarking fused_linear_ce... 0.01x

================================================================================
  性能对比表格 (forward-only)
================================================================================

| Kernel | Shape | Fused (ms) | PyTorch (ms) | Speedup | Fused Mem (MB) | PyTorch Mem (MB) | Mem Save |
|--------|-------|-----------|-------------|---------|---------------|-----------------|----------|
| fused_rms_norm | (196608, 448) | 0.212 | 5.086 | **24.00x** | 336.8 | 840.8 | 59.9% |
| fused_ce_zloss | (196608, 256) | 0.232 | 14.792 | **63.75x** | 99.0 | 483.0 | 79.5% |
| fused_swiglu | (196608, 1216) | 0.769 | 6.266 | **8.15x** | 1368.0 | 2736.0 | 50.0% |
| fused_rope | (64, 7, 3072, 64) | 4.850 | 6.020 | **1.24x** | 674.3 | 1010.3 | 33.3% |
| fused_add_rms_norm | (196608, 448) | 0.299 | 5.027 | **16.82x** | 504.8 | 1344.8 | 62.5% |
| flash_attn | (64, 7, 3072, 64) | 5.617 | 6.055 | **1.08x** | 677.2 | 677.2 | 0.0% |
| fused_attn_rope | (64, 7, 3072, 64) | 9.813 | 13.909 | **1.42x** | 1015.5 | 1178.3 | 13.8% |
| fused_linear_ce | (196608, 448, 256) | 1538.158 | 18.586 | **0.01x** | 707.4 | 1091.4 | 35.2% |

平均加速比: 14.56x
最大加速比: 63.75x (fused_ce_zloss)

================================================================================
  Roofline 分析
================================================================================

| Kernel | FLOPs | Bytes (R+W) | AI (FLOPs/B) | Actual TFLOPS | Bottleneck |
|--------|-------|-------------|-------------|---------------|------------|
| fused_rms_norm | 440.4M | 352.3M | 1.2 | 2.079 | **Memory** |
| fused_ce_zloss | 151.0M | 101.4M | 1.5 | 0.651 | **Memory** |
| fused_swiglu | 352.3M | 528.5M | 0.7 | 0.458 | **Memory** |
| fused_rope | 1.06G | 705.4M | 1.5 | 0.218 | **Memory** |
| fused_add_rms_norm | 528.5M | 528.5M | 1.0 | 1.769 | **Memory** |
| flash_attn | 541.17G | 704.6M | 768.0 | 96.346 | **Compute** |
| fused_attn_rope | 542.22G | 704.6M | 769.5 | 55.256 | **Compute** |
| fused_linear_ce | 45.25G | 176.4M | 256.5 | 0.029 | **Compute** |

判断标准: AI < 100 FLOPs/byte → Memory-bound（fusion 优化有效）
          AI ≥ 100 FLOPs/byte → Compute-bound（算法优化更重要）

