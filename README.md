# MDL Deep Image Compression

基于 **Minimum Description Length (MDL)** 原则的深度图像压缩系统设计与实现。核心命题：**压缩即预测** — CE loss 直接对应 Shannon 最优编码长度 (Shannon 1948, Delétang et al. 2024)。

- **Phase A (完成)**: iGPT token-level 自回归压缩 + 7 个手写 Triton Kernel (~2,000 行)，iGPT-S CIFAR-10 SWA **2.9739 bits/dim**
- **Phase B (训练中)**: CC-iGPT（Coarse-Conditioned iGPT）双尺度条件自回归 — 浅层 coarse iGPT (8×8, 192 token) 独立编码进 bitstream，UP + 量化后通过 additive embedding（可学习 α）注入 fine iGPT (32×32, 3072 token)，目标 BPP_total < iGPT-S 2.9739
- **Phase C (完成)**: Demo 前端可视化系统 (FastAPI + Chart.js, 5 个展示面板)

## Baseline 对比

| 方法 | Params | CIFAR-10 bits/dim ↓ | 来源 |
|------|--------|---------------------|------|
| PixelCNN++ | 52M | 2.92 | Salimans et al., ICLR 2017 |
| Image Transformer | 95M | 2.90 | Parmar et al., ICML 2018 |
| PixelSNAIL | 380M | 2.85 | Chen et al., ICML 2018 |
| **iGPT-S (Ours, best)** | **76.05M** | **2.9792** | d_model=512, N=24, 200 epochs |
| **iGPT-S (Ours, SWA)** | **76.05M** | **2.9739** | SWA averaged over 21 checkpoints |
| **CC-iGPT (Ours)** | **~95M** | 训练中 | 双尺度条件 AR (fine iGPT-S + 浅层 coarse iGPT, pool=4×) |
| PNG (lossless) | — | ~5.87 | 传统方法 |
| WebP (lossless) | — | ~5.02 | 传统方法 |

## 快速开始

```bash
pip install torch torchvision pyyaml numpy pillow tensorboard triton
```

```bash
# 单元测试 + 前向 dry-run
pytest tests/ -v
python scripts/dryrun_forward.py

# 训练 — 单卡
torchrun --nproc_per_node=1 scripts/train.py --config configs/igpt_cifar10_s.yaml

# 训练 — 多卡 DDP (按 GPU 数调整 nproc_per_node)
torchrun --nproc_per_node=2 scripts/train.py --config configs/igpt_cifar10_s.yaml
torchrun --nproc_per_node=2 scripts/train.py --config configs/ccigpt_cifar10_s.yaml

# 训练 — ImageNet 32×32
torchrun --nproc_per_node=2 scripts/train.py --config configs/igpt_imagenet32_s.yaml

# 断点续训 (resume 会自动用 config 中的 lr 覆盖 checkpoint 旧值)
torchrun --nproc_per_node=2 scripts/train.py \
    --config configs/igpt_cifar10_s.yaml \
    --resume experiments/igpt_cifar10_s/checkpoints/epoch_100.pth

# 评测 — iGPT
python scripts/evaluate.py --config configs/igpt_cifar10_s.yaml \
    --checkpoint experiments/igpt_cifar10_s/checkpoints/best.pth

# 评测 — CC-iGPT (含 coarse / fine CE 分解 + BPP_total)
python scripts/evaluate.py --config configs/ccigpt_cifar10_s.yaml \
    --checkpoint experiments/ccigpt_cifar10_s/checkpoints/best.pth
# SWA 权重
python scripts/evaluate.py --config configs/ccigpt_cifar10_s.yaml \
    --checkpoint experiments/ccigpt_cifar10_s/checkpoints/swa.pth

# Linear Probe (各层表征分类准确率)
python scripts/linear_probe.py --config configs/igpt_cifar10_s.yaml \
    --checkpoint experiments/igpt_cifar10_s/checkpoints/best.pth --layers all

# Kernel Profiling
python scripts/profile_kernels.py --roofline

# Demo 前端
pip install fastapi uvicorn python-multipart
uvicorn demo.server:app --reload --port 8000
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

# 验证环境
python scripts/dryrun_forward.py
pytest tests/ -v

# ========== AutoDL: 训练 ==========
torchrun --nproc_per_node=2 scripts/train.py --config configs/igpt_cifar10_s.yaml
torchrun --nproc_per_node=2 scripts/train.py --config configs/ccigpt_cifar10_s.yaml

# 后台挂起 (断开 SSH 不中断)
nohup torchrun --nproc_per_node=2 scripts/train.py \
    --config configs/ccigpt_cifar10_s.yaml \
    > train_ccigpt.log 2>&1 &
tail -f train_ccigpt.log

# 监控 GPU
watch -n 1 nvidia-smi

# TensorBoard (AutoDL 自定义端口转发)
tensorboard --logdir experiments/ --port 6006 --host 0.0.0.0

# ========== AutoDL → 本地: 回收 checkpoint ==========
# 本地 WSL 执行：
scp -P <port> root@<autodl-host>:/root/autodl-tmp/mdl-deep-image-compression/experiments/igpt_cifar10_s/checkpoints/best.pth \
    ./experiments/igpt_cifar10_s/checkpoints/
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
    iGPT (Phase A)                       CC-iGPT (Phase B)
          |                                       |
  +-------v--------+                  +-----------v-----------+
  | Flatten: 3072  |                  | DOWN avg_pool 8×8     |
  | tokens         |                  |   → Coarse iGPT       |
  | (3×32×32)      |                  |     192 tokens, 进 bs |
  +-------+--------+                  | UP bilinear → quantize|
          |                           |   → fine.token_embed  |
          |                           |   → α · coarse_ctx    |
          |                           |     (additive 注入)    |
          |                           | Fine iGPT 3072 tokens |
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
                    (+ α · coarse_ctx,               |
                     CC-iGPT fine 分支)              |
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
                    |    Linear(→256)     |
                    +---------+-----------+
                              |
                    +---------v-----------+
                    | Cross-Entropy Loss  |
                    | + z-loss 正则 (1e-4) |
                    | (Fused CE Triton)   |
                    +---------+-----------+
                              |
              iGPT:    BPP = CE × T / ln(2) / (H·W·C)
              CC-iGPT: BPP_total = (CE_c · N_c + CE_f · N_f) / ln(2) / N_f
                       (coarse + fine 联合压缩率，N_f = H·W·C)
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

### VQVAE/VAR 路线（已删除）

早期曾考虑 VQVAE + VAR 有损路线，现已删除（毕设聚焦 YCbCr 域无损）。多尺度方向曾尝试 MSPA 实现，因训练稳定性问题未采用，转而采用更简洁的 CC-iGPT。

### CC-iGPT (`models/cc_igpt.py`)

Coarse-Conditioned iGPT —— MSPA 的 2-scale 退化版本。回避了多尺度 loss 平衡难题，复用全部 iGPT 训练栈与 Triton kernels。

| 组件 | 说明 |
|------|------|
| DOWN | `F.adaptive_avg_pool2d(x, 8)` 在 float 域下采样到 8×8 |
| Coarse iGPT | 浅层（d_model=256, N=6），独立 NTP 训练，CE 进 bitstream（192 token，~6% overhead） |
| UP | `F.interpolate(bilinear)` 回到 32×32，`rgb_to_ycbcr_int` 量化（与 fine encoder 一致） |
| Ctx 注入 | `fine.token_embed(UP(x_c)_tokens)` → AR shift → `α · coarse_ctx`（additive，无新增可学习参数） |
| 可学习 α | `nn.Parameter(torch.ones(1))`，初始 1.0；模型自适应注入强度，避免 ctx 过强压制 fine token embed |
| 联合 BPP | `BPP_total = (CE_c · 192 + CE_f · 3072) / ln(2) / 3072`（按 H·W·C 归一化） |
| 训练 | 端到端联合 `loss = loss_coarse + loss_fine`，无尺度间加权 |
| 关闭 ctx | `fine(x, coarse_ctx=None)` 严格等价 vanilla iGPT（unit test 校验 CE diff < 1e-6） |

设计参考: Burt & Adelson "Laplacian Pyramid" (1983)、van den Oord "Conditional PixelCNN" (NeurIPS 2016, additive 条件)、Tian "VAR" (NeurIPS 2024)。

### 共享层 (`models/layers.py`)

GPTBlock, MultiHeadAttentionBlock (RoPE + QK-Norm + Flash Attention + attn_mask), RMSNorm, SwiGLU FFN。RoPE 的 `cos/sin` 按 `(seq_len, device)` 缓存，避免逐层逐步重算。

### 手写 Triton Kernels (`ops/`, ~2,000 行, 7 个 kernel)

| Kernel | 行数 | 说明 |
|--------|------|------|
| Flash Attention v2 | ~960 | causal early termination, online softmax |
| Fused CE+z-loss | ~276 | online softmax, 避免 O(V) 中间矩阵 |
| Fused RMSNorm | ~236 | fwd+bwd |
| Fused Add+RMSNorm | ~217 | post-norm 残差+归一化合并 |
| Fused SwiGLU | ~204 | activation recomputation |
| Fused RoPE | ~123 | 就地旋转 Q/K |
| Fused Attn+RoPE | ~100 | RoPE + Flash Attention 合并 |

全部自动降级到 PyTorch，全部有独立单元测试。

### Linear Probe (`scripts/linear_probe.py`)

冻结预训练模型，通过 `IGPT.encode(x, max_layer)` 直接取各层 hidden state（跳过 output head 与 loss），全局平均池化 → 线性分类器 → 报告**训练结束时**的测试准确率（避免按 epoch 选 max 造成 test-set peeking）。
对比预训练表征质量 (Chen et al. 2020)。

## 项目结构

```
src/mdlic/
├── models/    igpt.py, cc_igpt.py, layers.py
├── ops/       7 个 Triton kernels (flash_attn, fused_rms_norm, ...)
├── optim/     muon.py (Muon optimizer)
└── utils/     seed, BPP
scripts/       train.py, evaluate.py, linear_probe.py, dryrun_forward.py, profile_kernels.py
configs/       igpt_cifar10_s, igpt_cifar100_s, igpt_imagenet32_s, ccigpt_cifar10_s
tests/         7 个 kernel 单元测试
demo/
├── server.py          FastAPI 后端 (predict / metrics / probe / kernels / scales)
├── static/            HTML + JS (Chart.js) + CSS 前端，5 个面板：上传预测、BPP 对比、
│                       Linear Probe、Triton kernel 加速比、CC-iGPT coarse/fine token 分配
└── data/              预计算 JSON 数据 (训练后替换为真实结果)
```

**参考文献**

**模型**: iGPT (Chen 2020), RoPE (Su 2021), RMSNorm (Zhang 2019), SwiGLU (Shazeer 2020), OLMo 2 (2025), QK-Norm (Dehghani 2023), Weight Tying (Press 2017), Linear Probe (Alain & Bengio 2017)

**像素自回归**: PixelCNN++ (Salimans 2017), PixelSNAIL (Chen 2018), PixelCNN (van den Oord 2016)

**多尺度**: VAR (Tian 2024), VQ-VAE (van den Oord 2017)

**Triton**: FlashAttention v1/v2 (Dao 2022/2023), Online Softmax (Milakov 2018), Liger Kernel (Hsu 2024)

**训练**: WSD (MiniCPM 2024), SWA (Izmailov 2018), muP (Yang 2022), Muon (Jordan 2024)

**理论**: Shannon (1948), MDL (Rissanen 1978), Language Modeling Is Compression (Delétang 2024)
