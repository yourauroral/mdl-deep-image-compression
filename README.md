# MDL Deep Image Compression

基于 **Minimum Description Length (MDL)** 原则的深度图像压缩。核心命题：**压缩即预测** — CE loss 直接对应 Shannon 最优编码长度 (Shannon 1948, Delétang et al. 2024)。

- **Phase A (完成)**: iGPT token-level 自回归压缩 + 8 个手写 Triton Kernel (~2,400 行)
- **Phase B (当前)**: MSPA 多尺度像素无损自回归 (借鉴 VAR next-scale prediction) + Linear Probe 表征评估

## Baseline 对比

| 方法 | CIFAR-10 bits/dim ↓ | 来源 |
|------|---------------------|------|
| PixelCNN++ | 2.92 | Salimans et al., ICLR 2017 |
| Image Transformer | 2.90 | Parmar et al., ICML 2018 |
| PixelSNAIL | 2.85 | Chen et al., ICML 2018 |
| **iGPT (Ours)** | **3.77** | — |
| PNG (lossless) | ~5.87 | 传统方法 |
| WebP (lossless) | ~5.02 | 传统方法 |

### 消融实验

| 实验 | 配置 | BPP↓ | ΔBPP |
|------|------|------|------|
| E0 | Full Proposed Model | 3.77 | — |
| E1 | w/o YCbCr | TBD | +? |
| E2 | w/o SwiGLU (ReLU FFN) | TBD | +? |
| E3 | w/ Sub-pixel AR | TBD | -? |

## 快速开始

```bash
pip install torch torchvision pyyaml numpy pillow tensorboard
```

```bash
# 训练 iGPT
python scripts/train.py --config configs/igpt_cifar10_baseline.yaml

# 多卡分布式
torchrun --nproc_per_node=4 scripts/train.py --config configs/igpt_cifar10_baseline.yaml

# 评测
python scripts/evaluate.py --config configs/igpt_cifar10_baseline.yaml \
    --checkpoint experiments/igpt_cifar10_baseline/checkpoints/best.pth

# Linear Probe (各层表征分类准确率)
python scripts/linear_probe.py --config configs/igpt_cifar10_baseline.yaml \
    --checkpoint best.pth --layers all

# Kernel Profiling
python scripts/profile_kernels.py --roofline

# 测试
pytest tests/ -v
```

### AutoDL 训练流程

本地 WSL 只做开发和 dry-run，训练一律在 AutoDL GPU 实例上执行。

```bash
# ========== 本地 WSL: 推送代码 ==========
git add -A && git commit -m "sync to autodl" && git push origin dev

# ========== AutoDL 实例首次部署 ==========
# SSH 登录后：
cd /root/autodl-tmp
git clone <repo-url> mdl-deep-image-compression
cd mdl-deep-image-compression
git checkout dev
pip install torch torchvision pyyaml numpy pillow tensorboard triton

# 前向 dry-run，确认改动未破坏模型
python scripts/dryrun_forward.py

# 单元测试，确认 8 个 Triton kernel 正确
pytest tests/ -v

# ========== AutoDL: 训练 ==========
# 单卡
python scripts/train.py --config configs/igpt_cifar10_baseline.yaml

# 多卡 (按实例 GPU 数调整 nproc_per_node)
torchrun --nproc_per_node=4 scripts/train.py --config configs/igpt_cifar10_baseline.yaml

# 后台挂起 (断开 SSH 不中断)
nohup python scripts/train.py --config configs/igpt_cifar10_baseline.yaml \
    > train.log 2>&1 &
tail -f train.log

# 监控 GPU
watch -n 1 nvidia-smi

# TensorBoard (AutoDL 自定义端口转发)
tensorboard --logdir experiments/ --port 6006 --host 0.0.0.0

# ========== AutoDL → 本地: 回收 checkpoint ==========
# 本地 WSL 执行：
scp -P <port> root@<autodl-host>:/root/autodl-tmp/mdl-deep-image-compression/experiments/igpt_cifar10_baseline/checkpoints/best.pth \
    ./experiments/igpt_cifar10_baseline/checkpoints/

# ========== 断点续训 ==========
python scripts/train.py --config configs/igpt_cifar10_baseline.yaml \
    --resume experiments/igpt_cifar10_baseline/checkpoints/last.pth
```

## 架构

### 模型架构图

```
                    RGB Image [B, 3, 32, 32]
                              |
                    +---------v-----------+
                    |    RGB -> YCbCr     |  ITU-R BT.601
                    |  round() 量化 [0,255]|
                    +---------+-----------+
                              |
          +-------------------+-------------------+
          |                                       |
    iGPT (Phase A)                        MSPA (Phase B)
          |                                       |
  +-------v--------+                  +-----------v-----------+
  | Flatten: 3072  |                  | Multi-Scale Pyramid   |
  | tokens         |                  | S0(1×1)  → 3 tokens  |
  | (3×32×32)      |                  | S1(2×2)  → 12        |
  +-------+--------+                  | S2(4×4)  → 48        |
          |                           | S3(8×8)  → 192       |
          |                           | S4(16×16)→ 768       |
          |                           | S5(32×32)→ 3072      |
          |                           | 总计: 4095 tokens     |
          |                           +-----------+-----------+
          |                                       |
          +-------------------+-------------------+
                              |
            +-----------------+-----------------+
            |  channel-first (默认)              |  pixel-first (子像素自回归)
            |  [Y_all | Cb_all | Cr_all]       |  [Y₀,Cb₀,Cr₀, Y₁,Cb₁,Cr₁, ...]
            +-----------------+-----------------+
                              |
                    +---------v-----------+
                    | 自回归移位:          |  input  = x[0..T-1]
                    |  input = x[:-1]    |  target = x[1..T]
                    |  target = x[1:]    |  (用 x₀ 预测 x₁, 用 x₀x₁ 预测 x₂, ...)
                    +---------+-----------+
                              |
                    +---------v-----------+
                    |   Token Embedding   |<--- Weight Tying
                    |   (vocab = 256,     |     (embedding 和 output head
                    |    d_model)         |      共享同一权重矩阵)
                    +---------+-----------+          |
                              |                      |
                    (+ Channel Embedding,            |
                    子像素自回归模式)                  |
                    (+ Scale Embedding,              |
                     MSPA 模式)                       |
                              |                      |
            ┌─────────────────────────────────────────────────────┐
            │         N × GPT Block (OLMo 2 reordered norm)       │
            │                                                     │
            │   ──────────────────●───────────────────●─────►     │
            │   x             ╱   │  x'           ╱   │  x''      │
            │                ╱    ▼              ╱    ▼           │
            │             ADD   MHA           ADD   SwiGLU        │
            │              ▲   (RoPE·QK-Norm)  ▲    FFN           │
            │              │   (Flash·Causal)  │                  │
            │              │    │              │    │             │
            │              │    ▼              │    ▼             │
            │              └── RMSNorm         └── RMSNorm        │
            │                                                     │
            │   x'  = x  + RMSNorm(MHA(x))                        │
            │   x'' = x' + RMSNorm(FFN(x'))                       │
            │                                                     │
            │   ↓ Linear Probe 可从任一 Block 输出取 hidden        │
            │     IGPT.encode(x, max_layer) → GAP → 线性分类器     │
            └─────────────────────────────────────────────────────┘
                              |                      |
                    +---------v-----------+          |
                    |    Output Head      |--- Weight Tying
                    |    Linear(→256)     |   (fused path: backward
                    +---------+-----------+    显式回算 d_W, 保证
                              |                 tied head 正确梯度)
                    +---------v-----------+
                    | Cross-Entropy Loss  |
                    | + z-loss 正则 (1e-4) |
                    | + Fused Linear+CE   |
                    | (Triton kernel)     |
                    +---------+-----------+
                              |
              iGPT: BPP = CE × T / ln(2) / (H·W·C)
              MSPA: BPP = Σ_k CE_k · N_k / ln(2) / (H·W·C)
                    （所有 scale 合计，S0..S4 虽少仍需传输）
```

**OLMo 2 Reordered Norm**:  `x = x + RMSNorm(Attention(x))`, `x = x + RMSNorm(FFN(x))`
（区别于 Vaswani 2017 原始 post-norm `LN(x + sublayer(x))`，norm 放在残差内、sublayer 之后）

**子像素自回归序列**:
```
channel-first:  [Y₀ Y₁ ... Y₁₀₂₃ | Cb₀ Cb₁ ... Cb₁₀₂₃ | Cr₀ Cr₁ ... Cr₁₀₂₃]
pixel-first:    [Y₀ Cb₀ Cr₀ | Y₁ Cb₁ Cr₁ | ... | Y₁₀₂₃ Cb₁₀₂₃ Cr₁₀₂₃]
                     ↑ causal mask 使 Cb₀ 看到 Y₀, Cr₀ 看到 Y₀+Cb₀
```

### iGPT (`models/igpt.py`)

自回归像素压缩，将图像展平为 token 序列建模 p(x_t | x_{<t})。

**模型架构与技术**:

| 技术 | 说明 | 参考 |
|------|------|------|
| RGB → YCbCr | ITU-R BT.601 色彩空间变换，降低通道间冗余（亮度/色度分离），`round()` 量化到 [0,255] | JPEG (Wallace 1992) |
| RoPE | Rotary Position Embedding，base=500000，编码相对位置；子像素 AR 模式下为像素级 RoPE（同像素的 Y/Cb/Cr 共享 position_id） | Su et al. 2021, LLaMA 3 |
| QK-Norm | 对 Q、K 做 per-head RMSNorm，防止注意力 logits 爆炸，稳定大模型训练 | Dehghani et al. 2023 |
| RMSNorm (Post-Norm) | OLMo 2 风格后归一化：`x = x + RMSNorm(sublayer(x))`，训练更稳定，无需 final norm | OLMo 2 (2025) |
| SwiGLU FFN | 三线性门控 FFN：`out = (xW_gate ⊙ SiLU(xW_up)) W_down`，d_ff = (8/3)×d_model | Shazeer 2020, LLaMA |
| Weight Tying | Token embedding 与 output head 共享权重矩阵，减少参数量 | Press & Wolf 2017, GPT-2 |
| z-loss | 正则化项 `z_loss = λ·(logsumexp(logits))²`，防止 logits 幅度失控，λ=1e-4 | PaLM (Chowdhery 2022) |
| 深度缩放初始化 | 输出投影层 std = 1/√(2N)，N 为层数，防止深层残差累积过大 | GPT-2, OLMo 2 |
| 子像素自回归 | pixel-first 序列 [Y₀,Cb₀,Cr₀, Y₁,Cb₁,Cr₁,...]，causal mask 自然实现 p(Cb\|Y), p(Cr\|Y,Cb) 通道间条件依赖。额外的 channel embedding 标识通道身份 | PixelCNN++ (Salimans 2017) |

**训练策略**:

| 技术 | 说明 | 参考 |
|------|------|------|
| WSD Schedule | Warmup-Stable-Decay 三阶段学习率调度，比 cosine 更适合长训练 | MiniCPM (Hu et al. 2024) |
| SWA | Stochastic Weight Averaging，训练后期对权重做指数移动平均，获得更平坦的 loss landscape | Izmailov et al. 2018 |
| muP | Maximal Update Parameterization，按宽度比例缩放初始化和学习率，小模型调参可迁移到大模型 | Yang et al. 2022 |
| Muon Optimizer | Newton-Schulz 正交化 SGD，对 2D 权重用正交化更新方向，其余用 AdamW | Jordan 2024 |
| DDP + no_sync | 多 GPU 分布式训练，梯度累积中间步跳过 AllReduce 通信 | PyTorch DDP |
| Selective Checkpointing | 只对 Attention 层做 activation checkpointing，平衡显存和速度 | Chen et al. 2016 |
| Mixed Precision | bf16/fp16 自动混合精度训练 | PyTorch AMP |
| torch.compile | PyTorch 2.0+ 图编译优化 (fullgraph=True) | PyTorch 2.0+ |

### VQVAE (`models/vqvae.py`) + VAR (`models/var.py`)

多尺度 VQVAE (6 scales, K=512, EMA codebook) + VAR next-scale prediction (block-causal attention: scale 内双向、跨 scale 因果)。

### 共享层 (`models/layers.py`)

GPTBlock, MultiHeadAttentionBlock (RoPE + QK-Norm + Flash Attention + attn_mask), RMSNorm, SwiGLU FFN。RoPE 的 `cos/sin` 按 `(seq_len, device)` 缓存，避免逐层逐步重算。

### 手写 Triton Kernels (`ops/`, ~2,400 行)

| Kernel | 行数 | 说明 |
|--------|------|------|
| Flash Attention v2 | ~960 | causal early termination, online softmax |
| Fused RMSNorm | ~236 | fwd+bwd |
| Fused CE+z-loss | ~276 | online softmax, 避免 O(V) 中间矩阵 |
| Fused SwiGLU | ~204 | activation recomputation |
| Fused RoPE | ~123 | 就地旋转 Q/K |
| Fused Add+RMSNorm | ~217 | post-norm 残差+归一化合并 |
| Fused Attn+RoPE | ~100 | RoPE + Flash Attention 合并 |
| Fused Linear+CE | ~280 | output head + CE, 避免实例化 logits；backward 显式回算 d_W 保证 weight-tying 下 head 权重正确更新 |

全部自动降级到 PyTorch，全部有独立单元测试。

### Linear Probe (`scripts/linear_probe.py`)

冻结预训练模型，通过 `IGPT.encode(x, max_layer)` 直接取各层 hidden state（跳过 output head 与 loss），全局平均池化 → 线性分类器 → 报告分类准确率。
支持 iGPT 和 MSPA，对比多尺度上下文对表征质量的提升 (Chen et al. 2020)。

## 项目结构

```
src/mdlic/
├── models/    igpt.py, vqvae.py, var.py, layers.py
├── ops/       8 个 Triton kernels (flash_attn, fused_rms_norm, ...)
├── optim/     muon.py (Muon optimizer)
└── utils/     seed, BPP
scripts/       train.py, evaluate.py, linear_probe.py, profile_kernels.py
configs/       igpt_cifar10_baseline.yaml, vqvae_cifar10.yaml, var_cifar10.yaml
tests/         8 个 kernel 单元测试
```

**参考文献**

**模型**: iGPT (Chen 2020), RoPE (Su 2021), RMSNorm (Zhang 2019), SwiGLU (Shazeer 2020), OLMo 2 (2025), QK-Norm (Dehghani 2023), Weight Tying (Press 2017), Linear Probe (Alain & Bengio 2017)

**像素自回归**: PixelCNN++ (Salimans 2017), PixelSNAIL (Chen 2018), PixelCNN (van den Oord 2016)

**多尺度**: VAR (Tian 2024), VQ-VAE (van den Oord 2017)

**Triton**: FlashAttention v1/v2 (Dao 2022/2023), Online Softmax (Milakov 2018), Liger Kernel (Hsu 2024)

**训练**: WSD (MiniCPM 2024), SWA (Izmailov 2018), muP (Yang 2022), Muon (Jordan 2024)

**理论**: Shannon (1948), MDL (Rissanen 1978), Language Modeling Is Compression (Delétang 2024)
