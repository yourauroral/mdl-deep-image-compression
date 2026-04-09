# MDL Deep Image Compression

基于自回归 Transformer（iGPT）的深度无损图像压缩。将图像视为像素 token 序列，用 GPT Transformer 建模条件概率 p(x_t | x_{<t})，交叉熵损失直接对应 Shannon 最优编码长度，即 BPP = CE_loss / ln(2) × C。

## Baseline 对比

### 学术参考（直接引用论文数字）

| 方法 | CIFAR-10 bits/dim ↓ | 来源 | 建模方式 |
|------|---------------------|------|----------|
| PixelCNN++ | 2.92 | Salimans et al., ICLR 2017 | Discretized logistic mixture |
| Image Transformer | 2.90 | Parmar et al., ICML 2018 | Autoregressive Transformer |
| PixelSNAIL | 2.85 | Chen et al., ICML 2018 | Discretized logistic mixture |
| **iGPT (Ours)** | **3.77** | — | **Categorical CE (vocab=256)** |

> **注**: PixelCNN++ 和 PixelSNAIL 使用 discretized logistic mixture likelihood，
> 本文使用逐通道 categorical CE（vocab=256）。两者都是估计 log-likelihood 的 bits/dim，
> 物理含义一致（Shannon 编码长度），但建模分布不同。
>
> iGPT 原文（Chen et al. 2020）使用 9-bit color palette（512 色 k-means 聚类），
> 与本文的逐通道 256 vocab **不可直接换算**，因此不纳入数值对比。

### 传统方法锚点

| 方法 | CIFAR-10 approx BPP |
|------|---------------------|
| PNG (lossless) | ~5.87 |
| WebP (lossless) | ~5.02 |
| FLIF | ~4.50 |

### 消融实验

以完整模型为 baseline (E0)，逐项去掉单个组件观察 BPP 变化（消融法）。
每个实验通过 config 中的布尔开关控制（如 `model.use_qk_norm: false`）：

| 实验 | 配置 | BPP↓ | ΔBPP |
|------|------|------|------|
| E0 | Full Proposed Model | 3.77 | — |
| E1 | w/o YCbCr | TBD | +? |
| E2 | w/o RoPE (learned PE) | TBD | +? |
| E3 | w/o Post-Norm (Pre-Norm) | TBD | +? |
| E4 | w/o SwiGLU (ReLU FFN) | TBD | +? |
| E5 | w/o QK-Norm | TBD | +? |
| E6 | w/o Depth-Scaled Init | TBD | +? |
| E7 | w/o z-loss | TBD | +? |
| E8 | w/ Sub-pixel AR | TBD | -? |
| E9 | w/ Label Smoothing σ=1.0 | TBD | -? |
| E10 | w/ Sliding Window W=512 | TBD | -? |
| E11 | w/ DMOL Loss (K=10) | TBD | -? |

## 环境要求

| 依赖 | 最低版本 | 说明 |
|------|----------|------|
| Python | 3.10+ | |
| PyTorch | **2.4+** | `torch.amp.autocast` 统一接口；`F.scaled_dot_product_attention` 需 2.0+ |
| torchvision | 0.19+ | 随 PyTorch 2.4 配套 |
| Triton | 3.0+ | PyTorch CUDA 安装自带，Triton kernel 可选 |
| CUDA | 11.8+ | Triton kernel 需要 GPU；CPU 下自动回退到 PyTorch 实现 |

## 安装

依赖手动安装：

```bash
pip install torch torchvision pyyaml numpy pillow tensorboard
pip install scikit-image  # 可选，用于 SSIM 计算
```

无需 `pip install -e`，直接从项目根目录运行脚本即可。

## 快速开始

### 训练

```bash
# CIFAR-10
python scripts/train.py --config configs/igpt_cifar10_baseline.yaml

# CIFAR-100
python scripts/train.py --config configs/igpt_cifar100_baseline.yaml

# 从断点恢复
python scripts/train.py --config configs/igpt_cifar100_baseline.yaml \
    --resume experiments/igpt_cifar100_baseline/checkpoints/epoch_10.pth

# 多卡分布式
torchrun --nproc_per_node=4 scripts/train.py --config configs/igpt_cifar10_baseline.yaml

# 多次独立运行（不同 seed）
python scripts/train.py --config configs/igpt_cifar100_baseline.yaml --seed 0
python scripts/train.py --config configs/igpt_cifar100_baseline.yaml --seed 1

# 导出训练曲线为 CSV（方便 matplotlib 画论文图）
python scripts/train.py --config configs/igpt_cifar100_baseline.yaml --export_csv
```

### 测试

```bash
python tests/test_dataloader.py      # 验证数据加载和 YCbCr 转换
python tests/test_flash_attn.py      # 验证 Flash Attention kernel
python scripts/dryrun_forward.py     # 快速前向 sanity check（含 MTP/soft-cap/muP/DMOL 测试）

# Triton kernel 单元测试（需 GPU + Triton）
pytest tests/test_fused_rms_norm.py      # Fused RMSNorm fwd+bwd 精度
pytest tests/test_fused_ce_zloss.py      # Fused CE+z-loss fwd+bwd 精度
pytest tests/test_fused_swiglu.py        # Fused SwiGLU fwd+bwd 精度
pytest tests/test_fused_rope.py          # Fused RoPE vs PyTorch apply_rotary_emb
pytest tests/test_fused_add_rms_norm.py  # Fused Add+RMSNorm fwd+bwd 精度
pytest tests/test_fused_attn_rope.py     # Fused Attn+RoPE vs 分步 RoPE+SDPA
pytest tests/test_fused_linear_ce.py     # Fused Linear+CE+z-loss vs 分步实现

# 或一次跑全部
pytest tests/ -v
```

### 评测

```bash
# 单模型评测
python scripts/evaluate.py --config configs/igpt_cifar10_baseline.yaml \
    --checkpoint experiments/igpt_cifar10_baseline/checkpoints/best.pth

# Per-channel BPP 分解（Y/Cb/Cr）
python scripts/evaluate.py --config configs/igpt_cifar10_baseline.yaml \
    --checkpoint best.pth --per_channel

# 消融批量评测（扫描实验目录）
python scripts/evaluate.py --config configs/igpt_cifar10_baseline.yaml \
    --ablation_dir experiments/

# SWA vs best 对比
python scripts/evaluate.py --config configs/igpt_cifar10_baseline.yaml \
    --checkpoint experiments/exp/checkpoints/best.pth --swa

# 传统方法（PNG/WebP）对比
python scripts/evaluate.py --config configs/igpt_cifar10_baseline.yaml \
    --checkpoint best.pth --traditional

# Per-position BPP 热力图（保存 PNG，分析压缩难度的空间分布）
python scripts/evaluate.py --config configs/igpt_cifar10_baseline.yaml \
    --checkpoint best.pth --heatmap
```

### Kernel Profiling

```bash
# 全部 kernel 加速比对比
python scripts/profile_kernels.py

# 指定 kernel + 包含 backward
python scripts/profile_kernels.py --kernel fused_rms_norm fused_swiglu --backward

# 自定义 shape
python scripts/profile_kernels.py --batch 64 --seq_len 3072 --d_model 128

# Roofline 分析（arithmetic intensity + 瓶颈判断）
python scripts/profile_kernels.py --roofline
```

### 监控

```bash
tensorboard --logdir experiments/igpt_cifar100_baseline/logs --port 6006
```

### 消融实验

```bash
# E0: Full Proposed Model (baseline)
python scripts/train.py --config configs/igpt_cifar100_baseline.yaml --seed 42 --export_csv

# E1-E7: 修改 config 中对应开关（如 model.use_ycbcr: false），其余不变
# 例：E1 w/o YCbCr
# 复制 config → 修改 use_ycbcr: false, exp_name: E1_no_ycbcr → 训练
python scripts/train.py --config configs/ablation/E1_no_ycbcr.yaml --seed 42 --export_csv

# 消融批量评测（目录下所有实验一键对比）
python scripts/evaluate.py --config configs/igpt_cifar100_baseline.yaml \
    --ablation_dir experiments/

# 多 seed 取均值（3 次独立运行）
for seed in 0 1 2; do
    python scripts/train.py --config configs/igpt_cifar100_baseline.yaml --seed $seed
done

# 完整评测流水线（单模型）
python scripts/evaluate.py --config configs/igpt_cifar100_baseline.yaml \
    --checkpoint experiments/igpt_cifar100_baseline/checkpoints/best.pth \
    --per_channel --traditional --heatmap
```

## 架构

### 模型架构图

```
                    RGB Image [B, 3, 32, 32]
                              |
                    +---------v-----------+
                    |    RGB -> YCbCr     |
                    |   (ITU-R BT.601)   |
                    |  round() 量化 [0,255]|
                    +---------+-----------+
                              |
                    +---------v-----------+
                    |  Flatten to Tokens  |
                    |  seq_len = 3×32×32  |
                    |     = 3072 tokens   |
                    +---------+-----------+
                              |
            +-----------------+-----------------+
            |  channel-first (默认)              |  pixel-first (子像素自回归)
            |  [Y_all | Cb_all | Cr_all]       |  [Y₀,Cb₀,Cr₀, Y₁,Cb₁,Cr₁, ...]
            +-----------------+-----------------+
                              |
                    +---------v-----------+
                    | NTP Shift: x[:-1]   |  input_tokens = x[0..T-1]
                    |   target = x[1:]    |  target_tokens = x[1..T]
                    +---------+-----------+
                              |
                    +---------v-----------+
                    |   Token Embedding   |<--- Weight Tying (CE 模式)
                    |   (vocab = 256,     |          |
                    |    d_model)         |          |
                    +---------+-----------+          |
                              |                      |
                    (+ Channel Embedding,             |
                     子像素自回归模式)                  |
                              |                      |
                              |                      |
            +=========================================+    |
            |       N x GPT Block (Post-Norm)         |    |
            |                                         |    |
            |  input                                  |    |
            |    |                                    |    |
            |    +------->+                           |    |
            |    |        |                           |    |
            |    v        |                           |    |
            |  MHA        | (skip)                    |    |
            |  (RoPE,     |                           |    |
            |   QK-Norm,  |                           |    |
            |   Causal)   |                           |    |
            |    |        |                           |    |
            |    v        |                           |    |
            |  RMSNorm    |                           |    |
            |    |        |                           |    |
            |    +<-------+                           |    |
            |    | (add)                              |    |
            |    |                                    |    |
            |    +------->+                           |    |
            |    |        |                           |    |
            |    v        |                           |    |
            |  SwiGLU     | (skip)                    |    |
            |  FFN        |                           |    |
            |    |        |                           |    |
            |    v        |                           |    |
            |  RMSNorm    |                           |    |
            |    |        |                           |    |
            |    +<-------+                           |    |
            |    | (add)                              |    |
            |    v                                    |    |
            |  output                                 |    |
            +=========================================+    |
                              |                      |
                    +---------v-----------+          |
                    |    Output Head      |--- Weight Tying (CE 模式)
                    |  CE:   Linear(→256) |
                    |  DMOL: Linear(→3K   |
                    |        或 →10K)     |
                    +---------+-----------+
                              |
            +-----------------+-----------------+
            |                                   |
  +---------v-----------+         +-------------v-----------+
  |    CE Loss Path     |         |      DMOL Loss Path     |
  |  Cross-Entropy      |         |  Discretized Mixture of |
  |  + z-loss (1e-4)    |         |  Logistics (K=10)       |
  |  + Fused Linear+CE  |         |  跨通道条件化:           |
  |  (可选 Triton)      |         |  μ_cb += α·Y            |
  +---------+-----------+         |  μ_cr += β·Y + γ·Cb     |
            |                     +-------------+-----------+
            +-----------------+-----------------+
                              |
                     BPP = NLL / ln(2) × C
```

**Post-Norm 残差模式** (OLMo 2 风格):
```
  x = x + RMSNorm(Attention(x))
  x = x + RMSNorm(FFN(x))
```

**子像素自回归序列排列**:
```
  channel-first:  [Y₀ Y₁ Y₂ ... Y₁₀₂₃ | Cb₀ Cb₁ ... Cb₁₀₂₃ | Cr₀ Cr₁ ... Cr₁₀₂₃]
  pixel-first:    [Y₀ Cb₀ Cr₀ | Y₁ Cb₁ Cr₁ | Y₂ Cb₂ Cr₂ | ... | Y₁₀₂₃ Cb₁₀₂₃ Cr₁₀₂₃]
                       ↑ causal mask 使 Cb₀ 可以看到 Y₀，Cr₀ 可以看到 Y₀ 和 Cb₀
```

模型代码位于 `src/mdlic/`：

### iGPT (`models/igpt.py`)

自回归图像压缩。图像展平为像素序列，用 GPT Transformer 建模像素的联合概率分布。

- **输入预处理**：RGB → YCbCr（ITU-R BT.601），`round()` 量化到 [0,255]
- **Tokenization**：每像素每通道作为独立 token（vocab=256，seq_len=3072 for 32×32×3）
- **子像素自回归**（可选）：pixel-first 序列排列 [Y₀,Cb₀,Cr₀, Y₁,Cb₁,Cr₁, ...]，
  使 Cb 条件于同像素的 Y，Cr 条件于 Y+Cb，配合 channel embedding 和像素级 RoPE。
  Ref: PixelCNN++ (Salimans 2017), PixelSNAIL (Chen 2018)
- **模型结构**：SwiGLU FFN、RoPE（base=500000）、QK-Norm、RMSNorm、OLMo 2 post-norm、Weight Tying
- **损失**：Cross-Entropy + z-loss 正则化（权重 1e-4）
- **BPP 计算**：`BPP = CE_loss / ln(2) × C`（C 为通道数，只用 CE，不含 z-loss）
- **可选**：MTP 辅助预测头（DeepSeek-V3 风格，默认关闭）
- **可选**：Logit soft-capping（Gemma 2 风格，默认关闭）
- **可选**：Gaussian Label Smoothing（σ-高斯软化 target，保留像素序数关系，默认关闭）
- **可选**：Sliding Window Attention（Longformer/Mistral 风格混合 full/windowed，默认关闭）
- **可选**：DMOL Loss（Discretized Mixture of Logistics，K 组 logistic 混合分布替代 256-class CE，
  利用像素值序数结构；子像素 AR 模式下支持跨通道条件化 μ_cb←Y, μ_cr←Y+Cb，PixelCNN++ 风格）

### 共享层 (`models/layers.py`)

| 模块 | 说明 |
|------|------|
| `RotaryEmbedding` | RoPE 位置编码，base=500000（LLaMA 3 / OLMo 2） |
| `MultiHeadAttentionBlock` | QK-Norm + RoPE (可选 Fused) + 可选 Triton Flash Attention |
| `RMSNorm` | Root Mean Square Normalization，可选 Fused Triton kernel |
| `FeedForwardBlock` | SwiGLU 三线性门控 FFN，可选 Fused SwiGLU kernel |
| `ReLUFeedForwardBlock` | ReLU FFN（消融 baseline） |
| `GPTBlock` | MHA + FFN + OLMo 2 post-norm (可选 Fused Add+RMSNorm) |

### 手写 Triton Kernels (`ops/`)

所有 kernel 均为毕设手写实现（不使用 liger-kernel 等第三方库），体现工程量。

| 文件 | 行数 | 说明 | 参考 |
|------|------|------|------|
| `flash_attn.py` | ~960 | Flash Attention v2，causal early termination + seq_len padding | Dao 2023 |
| `fused_rms_norm.py` | ~236 | Fused RMSNorm fwd+bwd | Zhang & Sennrich 2019 |
| `fused_ce_zloss.py` | ~276 | Fused CE + z-loss，online softmax 避免 O(V) 中间矩阵 | PaLM 2022, Liger 2024 |
| `fused_swiglu.py` | ~204 | Fused SwiGLU fwd+bwd，activation recomputation | Shazeer 2020, Liger 2024 |
| `fused_rope.py` | ~123 | Fused RoPE，就地旋转 Q/K，零中间张量 | Su et al. 2021, Liger 2024 |
| `fused_add_rms_norm.py` | ~217 | Fused Add+RMSNorm，post-norm 残差+归一化合并 | OLMo 2 2025, Liger 2024 |
| `fused_attn_rope.py` | ~100 | Fused Attention+RoPE，合并 RoPE 旋转与 Flash Attention | Su 2021, Dao 2023 |
| `fused_linear_ce.py` | ~280 | Fused Linear+CE+z-loss，output head 投影与 CE 合并，避免实例化 logits | Liger 2024, PaLM 2022 |
| `fused_dmol.py` | ~380 | Fused DMOL，discretized logistic mixture CDF + logsumexp，单通道/跨通道两种模式 | PixelCNN++ 2017 |

所有 Triton kernel 均有 PyTorch 优雅回退，CPU 环境或无 Triton 时自动切换。
训练启动时会打印各 kernel 的 ON/OFF 状态。
总 Triton 代码量 ~2,800 行（不含测试）。

### Triton Kernel 单元测试 (`tests/`)

每个手写 kernel 均有独立的单元测试，验证 fused kernel 与 PyTorch 参考实现的数值等价性：

| 测试文件 | 验证内容 | 参数覆盖 |
|----------|----------|----------|
| `test_fused_rms_norm.py` | fwd + bwd(dx, dw) vs PyTorch RMSNorm | 7 shape × 3 dtype + 3D |
| `test_fused_ce_zloss.py` | CE + z-loss vs `F.cross_entropy` + logsumexp² | 6 shape × 3 z_weight + fp16 |
| `test_fused_swiglu.py` | fwd + bwd(da, db) vs `F.silu(a)*b` | 7 shape × 3 dtype + 3D |
| `test_fused_rope.py` | Q/K 旋转 vs `apply_rotary_emb()` | 5 shape × 3 dtype + 就地语义 |
| `test_fused_add_rms_norm.py` | fwd + bwd(d_residual, d_sublayer, dw) | 7 shape × 3 dtype + 3D |
| `test_flash_attn.py` | fwd + bwd vs `F.scaled_dot_product_attention` | 6 shape + causal/non-causal |
| `test_fused_attn_rope.py` | fused RoPE+Attn vs 分步 RoPE + SDPA | 4 shape × 2 dtype + 单 token |
| `test_fused_linear_ce.py` | fused Linear+CE+z-loss vs 分步 linear + CE | 5 shape × 3 z_weight + weight tying |

### Kernel Profiling (`scripts/profile_kernels.py`)

对比 8 个 fused kernel 与 PyTorch 等价实现的性能，输出 Markdown 汇总表格：

- **延迟** (ms)：CUDA Event 精确计时（中位数）
- **加速比**：PyTorch time / Fused time
- **显存**：峰值 CUDA 显存对比
- **Roofline**：arithmetic intensity (FLOPs/byte) + compute/memory 瓶颈判断
- 支持 `--backward` 模式、自定义 shape、指定 kernel

### 优化器 (`optim/`)

| 文件 | 说明 | 参考 |
|------|------|------|
| `muon.py` | Muon 优化器：Newton-Schulz 正交化 SGD for 2D weights + AdamW for rest | Jordan 2024 |

## 配置系统

所有超参数在 `configs/*.yaml` 中定义：

| 字段 | 说明 |
|------|------|
| `model.d_model` | 隐层维度（默认 128） |
| `model.N` | Transformer 层数（默认 2） |
| `model.h` | 注意力头数（默认 4） |
| `model.d_ff` | FFN 隐层维度（默认 384） |
| `model.use_mtp` | 是否启用 MTP 辅助头（默认 false） |
| `model.use_ycbcr` | YCbCr 预处理（默认 true） |
| `model.use_rope` | RoPE 位置编码（默认 true） |
| `model.use_post_norm` | OLMo 2 post-norm（默认 true） |
| `model.use_swiglu` | SwiGLU FFN（默认 true） |
| `model.use_qk_norm` | QK-Norm（默认 true） |
| `model.use_subpixel_ar` | 子像素自回归 pixel-first 序列（默认 false） |
| `model.sliding_window_size` | Sliding window 大小（-1=full causal，推荐 512/1024） |
| `model.full_attn_every_n` | 每 N 层 1 层 full attention（0=全部 windowed） |
| `model.loss_type` | 损失类型：`ce`（categorical CE，默认）或 `dmol`（Discretized Mixture of Logistics） |
| `model.num_mixtures` | DMOL 混合分量数 K（默认 10，仅 loss_type=dmol 时生效） |
| `model.logit_soft_cap` | Logit soft-capping 上界（0=禁用，推荐 30.0） |
| `train.amp_dtype` | `fp16` 或 `bf16` |
| `train.lr_schedule` | `cosine` / `wsd` / `multistep` |
| `train.z_loss_weight` | z-loss 正则化权重（默认 1e-4） |
| `train.label_smoothing_sigma` | Gaussian label smoothing σ（0=禁用，推荐 1.0） |
| `train.muon.enabled` | 是否启用 Muon 优化器 |
| `train.swa.enabled` | 是否启用 SWA（默认 false） |
| `data.dataset` | 数据集名称（`cifar10` 或 `cifar100`） |

实验输出到 `experiments/{exp_name}/{logs,checkpoints}/`。检查点文件：`best.pth`、`epoch_*.pth`、`swa.pth`。

## 数据集

通过 `data.dataset` 字段选择数据集，由 `torchvision.datasets` 自动下载。

| 数据集 | 训练集 | 测试集 | seq_len | 配置文件 |
|--------|--------|--------|---------|----------|
| CIFAR-10 | 50,000 | 10,000 | 3,072 | `configs/igpt_cifar10_baseline.yaml` |
| CIFAR-100 | 50,000 | 10,000 | 3,072 | `configs/igpt_cifar100_baseline.yaml` |

## 特性

**模型架构**
- 自回归 iGPT（SwiGLU FFN、RoPE base=500k、QK-Norm、OLMo 2 post-norm）
- YCbCr 色彩空间输入（ITU-R BT.601），降低通道间冗余
- Weight Tying（embedding ↔ output head 共享权重）
- 子像素自回归（可选，pixel-first 序列 + channel embedding + 像素级 RoPE）
- Sliding Window Attention（可选，Longformer/Mistral 风格，可混合 full attention）
- DMOL Loss（可选，K 组 logistic 混合替代 CE，跨通道条件化 μ_cb←Y, μ_cr←Y+Cb）
- MTP 辅助预测头（可选，DeepSeek-V3 风格）
- Logit soft-capping（可选，Gemma 2 风格）
- 7 项消融开关 + 子像素自回归 + Label Smoothing + Sliding Window 实验，config 驱动

**训练策略**
- Gaussian Label Smoothing（可选，高斯软化 target 保留像素值序数关系）
- Cosine / WSD LR schedule + linear warmup
- SWA 权重平均（手写 lerp 实现）
- muP 初始化 + LR 缩放
- Muon 优化器（Newton-Schulz 正交化）
- 混合精度（bf16/fp16）+ 多 GPU DDP + no_sync 梯度累积
- Selective activation checkpointing + torch.compile

**手写 Triton Kernels（9 个，~2,800 行）**
- Fused RMSNorm · Fused SwiGLU · Fused CE+z-loss
- Flash Attention v2 · Fused RoPE · Fused Add+RMSNorm
- Fused Attention+RoPE · Fused Linear+CE+z-loss · Fused DMOL
- 全部自动降级到 PyTorch 实现
- 全部有独立单元测试

**工程**
- Config 验证（必填字段 + 类型 + 取值范围检查）
- NaN/Inf 检测 + SWA NaN 防护
- Checkpoint 兼容性检查（non-strict 加载 + 诊断）
- Fused kernel 状态日志（8 个 kernel ON/OFF）
- TensorBoard 日志，验证集 BPP mean±std
- CSV 训练曲线导出（`--export_csv`）
- Per-position BPP 热力图（`--heatmap`）
- Kernel Roofline 分析（`--roofline`）
- `--resume` 断点续训，`--seed` 多次独立运行

## 项目文件结构

```
mdl-deep-image-compression/
├── configs/
│   ├── igpt_cifar10_baseline.yaml     # CIFAR-10 配置
│   └── igpt_cifar100_baseline.yaml    # CIFAR-100 配置
├── src/mdlic/
│   ├── models/
│   │   ├── igpt.py                    # iGPT 模型（~550 行）
│   │   └── layers.py                  # GPTBlock/MHA/FFN/RMSNorm（412 行）
│   ├── ops/
│   │   ├── __init__.py                # Kernel 懒加载注册
│   │   ├── flash_attn.py              # Flash Attention v2（~960 行）
│   │   ├── fused_rms_norm.py          # Fused RMSNorm（~236 行）
│   │   ├── fused_ce_zloss.py          # Fused CE+z-loss（~276 行）
│   │   ├── fused_swiglu.py            # Fused SwiGLU（~204 行）
│   │   ├── fused_rope.py              # Fused RoPE（~123 行）
│   │   ├── fused_add_rms_norm.py      # Fused Add+RMSNorm（~217 行）
│   │   ├── fused_attn_rope.py         # Fused Attn+RoPE（~100 行）
│   │   ├── fused_linear_ce.py         # Fused Linear+CE（~280 行）
│   │   └── fused_dmol.py              # Fused DMOL（~380 行）
│   ├── optim/
│   │   └── muon.py                    # Muon 优化器（~203 行）
│   └── utils/
│       └── __init__.py                # seed_everything, compute_bpp, DMOL loss
├── scripts/
│   ├── train.py                       # 训练主循环（~680 行）
│   ├── evaluate.py                    # 评测工具集（~700 行）
│   ├── profile_kernels.py             # Kernel benchmark（~500 行）
│   └── dryrun_forward.py              # Forward sanity check（~150 行）
├── tests/
│   ├── test_dataloader.py             # 数据加载测试
│   ├── test_flash_attn.py             # Flash Attention 测试
│   ├── test_fused_rms_norm.py         # Fused RMSNorm 测试
│   ├── test_fused_ce_zloss.py         # Fused CE+z-loss 测试
│   ├── test_fused_swiglu.py           # Fused SwiGLU 测试
│   ├── test_fused_rope.py             # Fused RoPE 测试
│   ├── test_fused_add_rms_norm.py     # Fused Add+RMSNorm 测试
│   ├── test_fused_attn_rope.py        # Fused Attn+RoPE 测试
│   └── test_fused_linear_ce.py        # Fused Linear+CE 测试
├── future.md                          # 项目总览与实验计划
├── README.md
└── CLAUDE.md
```

## 参考文献

### 模型架构
- [Chen et al. 2020] "Generative Pretraining from Pixels," ICML 2020 (iGPT)
- [Su et al. 2021] "RoFormer," arXiv:2104.09864 (RoPE)
- [Zhang & Sennrich 2019] "Root Mean Square Layer Normalization," NeurIPS 2019 (RMSNorm)
- [Shazeer 2020] "GLU Variants Improve Transformers," arXiv:2002.05202 (SwiGLU)
- [OLMo 2 2025] arXiv:2501.00656 (Post-Norm, RoPE base=500k, z-loss, Weight Tying)
- [DeepSeek-V3 2024] arXiv:2412.19437 (MTP head)
- [Gemma 2 2024] arXiv:2408.00118 (Logit soft-capping)
- [PaLM 2022] arXiv:2204.02311 (z-loss)
- [Press & Wolf 2017] "Using the Output Embedding to Improve Language Models," EACL 2017 (Weight Tying)
- [Dehghani et al. 2023] "Scaling Vision Transformers," arXiv:2302.05442 (QK-Norm)
- [Radford et al. 2019] "GPT-2" (Depth-scaled init, Weight Tying)
- [Vaswani et al. 2017] "Attention Is All You Need," NeurIPS 2017

### 子像素自回归
- [Salimans et al. 2017] "PixelCNN++," ICLR 2017 — 通道间条件依赖, CIFAR-10: 2.92 bits/dim
- [van den Oord et al. 2016] "Conditional Image Generation with PixelCNN Decoders," NeurIPS 2016 — 子像素条件分解
- [Chen et al. 2018] "PixelSNAIL," ICML 2018 — 通道自回归, CIFAR-10: 2.85 bits/dim

### Baseline
- [Salimans et al. 2017] "PixelCNN++," ICLR 2017 — CIFAR-10: 2.92 bits/dim
- [Chen et al. 2018] "PixelSNAIL," ICML 2018 — CIFAR-10: 2.85 bits/dim
- [Parmar et al. 2018] "Image Transformer," ICML 2018 — CIFAR-10: 2.90 bits/dim
- [Shannon 1948] "A Mathematical Theory of Communication" — BPP 最优编码长度

### 训练策略
- [Hu et al. 2024] "MiniCPM," arXiv:2404.06395 (WSD LR schedule)
- [Hagele et al. 2024] arXiv:2405.18392 (WSD + SWA 组合)
- [Izmailov et al. 2018] "Averaging Weights Leads to Wider Optima," UAI 2018 (SWA)
- [Yang et al. 2022] "Tensor Programs V," arXiv:2203.03466 (muP)
- [Jordan 2024] "Muon," arXiv:2502.16982 (Newton-Schulz optimizer)
- [Xiong et al. 2020] "On Layer Normalization in the Transformer Architecture," ICML 2020 (Pre/Post-norm)
- [Loshchilov & Hutter 2016] "SGDR," arXiv:1608.03983 (Cosine annealing)
- [Chen et al. 2016] arXiv:1604.06174 (Activation checkpointing)

### Triton Kernels
- [Dao 2023] "FlashAttention-2," arXiv:2307.08691
- [Dao et al. 2022] "FlashAttention," NeurIPS 2022, arXiv:2205.14135
- [Milakov & Gimelshein 2018] "Online normalizer calculation for softmax," arXiv:1805.02867
- [Hsu et al. 2024] "Liger Kernel," arXiv:2410.10989 (Fused CE/SwiGLU/RMSNorm/RoPE pattern)
- [Williams et al. 2009] "Roofline," CACM (Roofline model)

### 色彩空间
- [ITU-R BT.601] YCbCr 标准
- [Wallace 1992] "The JPEG Still Picture Compression Standard"
- [Wiegand et al. 2003] "Overview of the H.264/AVC Video Coding Standard"

### 理论框架
- [Shannon 1948] "A Mathematical Theory of Communication" — 源编码定理
- [Tishby & Zaslavsky 2015] "Deep Learning and the Information Bottleneck Principle," arXiv:1503.02406
- [Delétang et al. 2024] "Language Modeling Is Compression," ICLR 2024, arXiv:2309.10668

### Sliding Window Attention & Label Smoothing
- [Child et al. 2019] "Generating Long Sequences with Sparse Transformers," arXiv:1904.10509 — CIFAR-10: 2.80 bits/dim
- [Beltagy et al. 2020] "Longformer," arXiv:2004.05150 (Sliding window attention)
- [Jiang et al. 2023] "Mistral 7B," arXiv:2310.06825 (Mixed sliding window + full attention)
- [Szegedy et al. 2016] "Rethinking the Inception Architecture," arXiv:1512.00567 (Label smoothing)

### Discretized Mixture of Logistics (DMOL)
- [Salimans et al. 2017] "PixelCNN++: Improving the PixelCNN with Discretized Logistic Mixture Likelihood," ICLR 2017 — DMOL loss, cross-channel conditioning, CIFAR-10: 2.92 bits/dim
