# MDL Deep Image Compression

基于 **Minimum Description Length (MDL)** 原则的深度图像压缩系统设计与实现。核心命题：**压缩即预测** — CE loss 直接对应 Shannon 最优编码长度 (Shannon 1948, Delétang et al. 2024)。

> **无损口径**：本系统所有主结果都在 **RGB-bit-exact** 域报告 — 直接在 RGB uint8 上建模，与 PixelCNN++ / Sparse Transformer 等基线同域可比。

- **Phase A (完成)**: iGPT token-level 自回归压缩 + 8 个手写 Triton Kernel（7 个进入训练栈，1 个 `fused_linear_ce` 在 V=256 下经 roofline 分析证伪、保留作反面案例，~2,400 行）
- **Phase B (进行中)**: CC-iGPT（Coarse-Conditioned iGPT）双尺度条件自回归 — 浅层 coarse iGPT (8×8, 192 token) 独立编码进 bitstream，UP + 量化后通过 additive embedding（可学习标量 α）注入 fine iGPT (32×32, 3072 token)
- **Phase C (完成)**: Demo 前端可视化系统 (FastAPI + Chart.js, 5 个展示面板)

## Baseline 对比

| 方法 | Params | CIFAR-10 bits/dim ↓ | 域 | 来源 |
|------|--------|---------------------|----|------|
| PixelCNN++ | 52M | 2.92 | RGB-bit-exact | Salimans et al., ICLR 2017 |
| Image Transformer | 95M | 2.90 | RGB-bit-exact | Parmar et al., ICML 2018 |
| PixelSNAIL | 380M | 2.85 | RGB-bit-exact | Chen et al., ICML 2018 |
| Sparse Transformer | 59M | 2.80 | RGB-bit-exact | Child et al., 2019 (128 层 strided sparse attention) |
| **CC-iGPT (Ours, channel-first)** | ~81M | 3.2540 | RGB-bit-exact | `[R…G…B…]` 平铺，消融起点 |
| **CC-iGPT (Ours, R-only)** | ~81M | **3.1074** | RGB-bit-exact | `coarse_in_channels=1`，coarse 仅压 R 8×8（64 token），fine 通过 sub-pixel AR 自学 G/B，**主表 SOTA** |
| **CC-iGPT (Ours, sub-pixel)** | ~81M | 3.1953 | RGB-bit-exact | `use_subpixel_ar=true`，序列布局 `[R₀G₀B₀ R₁G₁B₁ ...]`，coarse 192 token RGB |
| PNG | — | ~5.87 | RGB-bit-exact | 传统方法 |
| WebP (lossless mode) | — | ~5.02 | RGB-bit-exact | 传统方法 |

**消融三角内部论证**（不依赖跨基线绝对数字）：
- **双尺度方法本身有效**：R-only 3.11 vs channel-first 3.25 = **−0.14 bpd**
- **更多 coarse 信号未必更好**：sub-pixel 3.20 > R-only 3.11（差 +0.09），验证 R-only 灰度先验是双尺度容量配置的**甜点** — fine CE 在两种配置下都触底到 ~2.0，coarse 开销线性增长但 fine 边际收益饱和

CC-iGPT 主表与 Sparse Transformer 2.80 的差距来自参数预算（~81M vs 59M）/ 深度（24 层 dense vs 128 层 strided sparse）/ 训练 epoch（50 vs 200+）/ DMoL 输出头已尝试失败 git revert（详见 [`future.md`](future.md)）。CIFAR-10 主表的方法有效性靠**消融三角内部差值**论证；**SOTA 对比**靠 ImageNet 64×64 < 3.44（Sparse Transformer 152M strided）路线支撑。

### 创新点定位 & 与 SOTA 的关系

本工作并非以击败 Sparse Transformer 为目标。CC-iGPT 在 24 层 / ~81M 的参数预算下，沿三个维度构建差异化：

1. **方法 — 零新参数的双尺度条件注入**：coarse iGPT 量化 token 经 bit-exact 反量化/上采样/重 tokenize 后，复用 `fine.token_embed` 得到 `coarse_ctx`，再以可学习标量 α 做 additive 注入。整个 ctx 通路只引入 1 个标量参数；encoder/decoder 共用同一函数，bitstream 真实可解码。回避了多尺度联合 AR (MSPA) 的 loss 平衡难题。
2. **工程 — 8 个手写 Triton kernel + 1 个 roofline 证伪的反面案例**：7 个进入训练栈，1 个 `fused_linear_ce` 在 V=256 下经 roofline 分析判定为负收益（compute-bound + 三重循环失去 cuBLAS GEMM 利用率），保留在 `ops/` 作工程严谨性的反向证据，详见 [`experiments/kernel_negative_finding.md`](experiments/kernel_negative_finding.md)。
3. **分析 — RGB-bit-exact 三档消融 + 跨数据集 SOTA 对比**：CIFAR-10 RGB-bit-exact 三档消融（channel-first / R-only / sub-pixel）证明双尺度方法本身有效（−0.14 bpd）与 pixel-first AR 红利；ImageNet 64×64 跨数据集验证（目标 < 3.44 超 Sparse Transformer 152M）；Linear Probe 逐层表征曲线；roofline forward 与 fwd+bwd 双视角。

## 快速开始

```bash
pip install torch torchvision pyyaml numpy pillow tensorboard triton
```

```bash
# 单元测试 + 前向 dry-run
pytest tests/ -v
python scripts/dryrun_forward.py

# 训练 — 多卡 DDP (按 GPU 数调整 nproc_per_node)
torchrun --nproc_per_node=2 scripts/train.py --config configs/igpt_cifar10_s_rgb.yaml

# CC-iGPT RGB-bit-exact 三档
torchrun --nproc_per_node=2 scripts/train.py --config configs/ccigpt_cifar10_s_rgb.yaml          # channel-first
torchrun --nproc_per_node=2 scripts/train.py --config configs/ccigpt_cifar10_s_rgb_ronly.yaml    # R-only 灰度先验
torchrun --nproc_per_node=2 scripts/train.py --config configs/ccigpt_cifar10_s_rgb_subpixel.yaml # sub-pixel AR (主表)

# 断点续训 (resume 会自动用 config 中的 lr 覆盖 checkpoint 旧值)
torchrun --nproc_per_node=2 scripts/train.py \
    --config configs/ccigpt_cifar10_s_rgb_subpixel.yaml \
    --resume experiments/ccigpt_cifar10_s_rgb_subpixel/checkpoints/epoch_30.pth

# 评测 — CC-iGPT (含 coarse / fine CE 分解 + bpd_total)
python scripts/evaluate.py --config configs/ccigpt_cifar10_s_rgb_subpixel.yaml \
    --checkpoint experiments/ccigpt_cifar10_s_rgb_subpixel/checkpoints/best.pth
# SWA vs best 对比（同时评测 best.pth 和 swa.pth）
python scripts/evaluate.py --config configs/ccigpt_cifar10_s_rgb_subpixel.yaml \
    --checkpoint experiments/ccigpt_cifar10_s_rgb_subpixel/checkpoints/best.pth --swa

# Linear Probe (各层表征分类准确率)
python scripts/linear_probe.py --config configs/igpt_cifar10_s_rgb.yaml \
    --checkpoint experiments/igpt_cifar10_s_rgb/checkpoints/best.pth --layers all

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
cd /root/autodl-tmp
git clone <repo-url> mdl-deep-image-compression
cd mdl-deep-image-compression
git checkout dev
pip install torch torchvision pyyaml numpy pillow tensorboard triton

python scripts/dryrun_forward.py
pytest tests/ -v

# ========== AutoDL: 训练 ==========
# ⚠️ AutoDL 多卡 DDP 必须设置 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
nohup torchrun --nproc_per_node=2 scripts/train.py \
    --config configs/ccigpt_cifar10_s_rgb_subpixel.yaml --export_csv \
  > experiments/ccigpt_cifar10_s_rgb_subpixel_train.log 2>&1 &

watch -n 1 nvidia-smi
tensorboard --logdir experiments/ --port 6006 --host 0.0.0.0

# ========== AutoDL → 本地: 回收 checkpoint ==========
scp -P <port> root@<autodl-host>:/root/autodl-tmp/mdl-deep-image-compression/experiments/ccigpt_cifar10_s_rgb_subpixel/checkpoints/best.pth \
    ./experiments/ccigpt_cifar10_s_rgb_subpixel/checkpoints/
```

## 架构

### 模型架构图

```
                    RGB Image [B, 3, 32, 32]   (uint8, RGB-bit-exact)
                              |
          +-------------------+-------------------+
          |                                       |
    iGPT (Phase A)                       CC-iGPT (Phase B)
          |                                       |
  +-------v--------+                  +-----------v-----------+
  | Flatten: 3072  |                  | DOWN avg_pool 8×8     |
  | tokens         |                  |   → Coarse iGPT       |
  | (3×32×32)      |                  |     192 tokens, 进 bs |
  +-------+--------+                  | bit-exact ctx 路径:    |
          |                           |   coarse 量化 token    |
          |                           |   → 反量化 (RGB /255)  |
          |                           |   → bilinear UP 32×32  |
          |                           |   → re-tokenize        |
          |                           |   → fine.token_embed   |
          |                           |   → α · coarse_ctx     |
          |                           |     (additive 注入)    |
          |                           | Fine iGPT 3072 tokens |
          |                           +-----------+-----------+
          |                                       |
          +-------------------+-------------------+
                              |
            +-----------------+-----------------+
            |  channel-first                     |  pixel-first (子像素自回归)
            |  [R_all | G_all | B_all]           |  [R₀,G₀,B₀, R₁,G₁,B₁, ...]
            |  (消融起点)                         |  (主表配置)
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
              iGPT:    bpd = CE × T / ln(2) / (H·W·C)
              CC-iGPT: bpd_total = (CE_c · N_c + CE_f · N_f) / ln(2) / N_f
                       (coarse + fine 联合压缩率，N_f = H·W·C)
```

**OLMo 2 Reordered Norm**:  `x = x + RMSNorm(Attention(x))`, `x = x + RMSNorm(FFN(x))`

**子像素自回归序列**:
```
channel-first:  [R₀ R₁ ... R₁₀₂₃ | G₀ G₁ ... G₁₀₂₃ | B₀ B₁ ... B₁₀₂₃]
pixel-first:    [R₀ G₀ B₀ | R₁ G₁ B₁ | ... | R₁₀₂₃ G₁₀₂₃ B₁₀₂₃]
                     ↑ causal mask 使 G₀ 看到 R₀, B₀ 看到 R₀+G₀
```

### iGPT (`models/igpt.py`)

自回归像素压缩，将图像展平为 token 序列建模 p(x_t | x_{<t})。

**模型架构与技术**:

| 技术 | 说明 | 参考 |
|------|------|------|
| RoPE | Rotary Position Embedding，base=500000，编码相对位置；子像素 AR 模式下为像素级 RoPE（同像素的 R/G/B 共享 position_id） | Su et al. 2021, LLaMA 3 |
| QK-Norm | 对 Q、K 做 per-head RMSNorm，防止注意力 logits 爆炸，稳定大模型训练 | Dehghani et al. 2023 |
| RMSNorm (Post-Norm) | OLMo 2 风格后归一化：`x = x + RMSNorm(sublayer(x))`，训练更稳定，无需 final norm | OLMo 2 (2025) |
| SwiGLU FFN | 三线性门控 FFN：`out = (xW_gate ⊙ SiLU(xW_up)) W_down`，d_ff = (8/3)×d_model | Shazeer 2020, LLaMA |
| Weight Tying | Token embedding 与 output head 共享权重矩阵，减少参数量 | Press & Wolf 2017, GPT-2 |
| z-loss | 正则化项 `z_loss = λ·(logsumexp(logits))²`，防止 logits 幅度失控，λ=1e-4 | PaLM (Chowdhery 2022) |
| 深度缩放初始化 | 输出投影层 std = 1/√(2N)，N 为层数 | GPT-2, OLMo 2 |
| 子像素自回归 | pixel-first 序列 [R₀,G₀,B₀, R₁,G₁,B₁,...]，causal mask 自然实现 p(G\|R), p(B\|R,G) 通道间条件依赖。额外的 channel embedding 标识通道身份 | PixelCNN++ (Salimans 2017) |

**训练策略**:

| 技术 | 说明 | 参考 |
|------|------|------|
| Cosine + Warmup LR | 线性 warmup → cosine 衰减；可选 `min_lr_ratio` 末段 LR 下限避免 SWA "假平均" | OLMo 2 (2025) |
| SWA | Stochastic Weight Averaging，训练后期对权重做指数移动平均 | Izmailov et al. 2018 |
| DDP + no_sync | 多 GPU 分布式训练，梯度累积中间步跳过 AllReduce 通信 | PyTorch DDP |
| Selective Checkpointing | 只对 Attention 层做 activation checkpointing | Chen et al. 2016 |
| Mixed Precision | bf16/fp16 自动混合精度训练 | PyTorch AMP |

### CC-iGPT (`models/cc_igpt.py`)

Coarse-Conditioned iGPT —— 双尺度条件式自回归。回避了多尺度 loss 平衡难题，复用全部 iGPT 训练栈与 Triton kernels。

| 组件 | 说明 |
|------|------|
| DOWN | `F.adaptive_avg_pool2d(x, 8)` 在 float 域下采样到 8×8 |
| Coarse iGPT | 浅层（d_model=256, N=6），独立 NTP 训练，CE 进 bitstream（192 token，~6% overhead；R-only 配置下为 64 token，~2% overhead） |
| **Bit-exact ctx 路径** | encoder/decoder 必须看到**同一个** `coarse_ctx`，否则 fine 端算术编码不可解。统一管线：coarse 量化 token → `/255` 反量化为 RGB float → bilinear UP 到 32×32 → 与 fine encoder 同规则 re-tokenize → `fine.token_embed` |
| Ctx 注入 | AR shift `coarse_ctx[:, 1:]`（ctx[i] 对应 fine 被预测位置 i，PixelCNN++ conditional 标准语义）→ `α · coarse_ctx`（additive，仅引入 1 个标量参数 α） |
| 可学习 α | `nn.Parameter(torch.ones(1))`，初始 1.0；模型自适应注入强度 |
| 联合 bits/dim | `bpd_total = (CE_c · N_c + CE_f · N_f) / ln(2) / N_f` |
| 训练 | 端到端联合 `loss = loss_coarse + loss_fine`，无尺度间加权 |
| 关闭 ctx | `fine(x, coarse_ctx=None)` 严格等价 vanilla iGPT（unit test 校验 CE diff < 1e-6） |

设计参考: Burt & Adelson "Laplacian Pyramid" (1983)、van den Oord "Conditional PixelCNN" (NeurIPS 2016, additive 条件)、Tian "VAR" (NeurIPS 2024)。

### 共享层 (`models/layers.py`)

GPTBlock, MultiHeadAttentionBlock (RoPE + QK-Norm + Flash Attention + attn_mask), RMSNorm, SwiGLU FFN。RoPE 的 `cos/sin` 按 `(seq_len, device)` 缓存。

### 手写 Triton Kernels (`ops/`, ~2,400 行, 8 个 kernel)

7 个进入训练栈，1 个 (`fused_linear_ce`) 经 roofline 分析在 V=256 下负收益、保留作反面案例（详见 [`experiments/kernel_negative_finding.md`](experiments/kernel_negative_finding.md)）。

| Kernel | 行数 | 说明 |
|--------|------|------|
| Flash Attention v2 | ~960 | causal early termination, online softmax |
| Fused CE+z-loss | ~276 | online softmax, 避免 O(V) 中间矩阵 |
| Fused RMSNorm | ~236 | fwd+bwd |
| Fused Add+RMSNorm | ~217 | post-norm 残差+归一化合并 |
| Fused SwiGLU | ~204 | activation recomputation |
| Fused RoPE | ~123 | 就地旋转 Q/K |
| Fused Attn+RoPE | ~62 | 薄包装层：组合 fused_rope + flash_attn |
| ~~Fused Linear+CE~~ | ~393 | **反面案例** — V=256 下三重循环失去 cuBLAS GEMM 利用率 |

全部 7 个生效 kernel 自动降级到 PyTorch，全部有独立单元测试。

### Linear Probe (`scripts/linear_probe.py`)

冻结预训练模型，通过 `IGPT.encode(x, max_layer)` 直接取各层 hidden state，全局平均池化 → 线性分类器 → 报告**训练结束时**的测试准确率。对比预训练表征质量 (Chen et al. 2020)。

## 项目结构

```
src/mdlic/
├── models/    igpt.py, cc_igpt.py, layers.py
├── ops/       7 个 Triton kernels + 1 反面案例 (fused_linear_ce)
├── data/      imagenet32_npy.py (mmap-backed Dataset)
└── utils/     seed, bpd, clean_state_dict
scripts/       train.py, evaluate.py, linear_probe.py, dryrun_forward.py, profile_kernels.py, prepare_imagenet32.py
configs/       igpt_cifar10_s_rgb,
               ccigpt_cifar10_s_rgb (channel-first),
               ccigpt_cifar10_s_rgb_ronly (R-only B1, 主表 SOTA),
               ccigpt_cifar10_s_rgb_subpixel (sub-pixel B1)
tests/         8 个 kernel/模型 单元测试 (含 test_ccigpt_smoke)
demo/
├── server.py          FastAPI 后端 (predict / metrics / probe / kernels / scales)
├── static/            HTML + JS (Chart.js) + CSS 前端，5 个面板
└── data/              预计算 JSON 数据
```

**参考文献**

**模型**: iGPT (Chen 2020), RoPE (Su 2021), RMSNorm (Zhang 2019), SwiGLU (Shazeer 2020), OLMo 2 (2025), QK-Norm (Dehghani 2023), Weight Tying (Press 2017), Linear Probe (Alain & Bengio 2017)

**像素自回归**: PixelCNN++ (Salimans 2017), PixelSNAIL (Chen 2018), PixelCNN (van den Oord 2016), Sparse Transformer (Child 2019)

**多尺度**: VAR (Tian 2024), VQ-VAE (van den Oord 2017), Multi-Scale PixelCNN (Reed 2017), Subscale Pixel Networks (Menick 2018)

**Triton**: FlashAttention v1/v2 (Dao 2022/2023), Online Softmax (Milakov 2018), Liger Kernel (Hsu 2024)

**训练**: SWA (Izmailov 2018), Cosine + Warmup (OLMo 2 2025)

**理论**: Shannon (1948), MDL (Rissanen 1978), Language Modeling Is Compression (Delétang 2024)
