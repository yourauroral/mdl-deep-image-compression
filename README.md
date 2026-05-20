# MDL Deep Image Compression

基于 **Minimum Description Length (MDL)** 原则的深度图像压缩系统设计与实现。核心命题：**压缩即预测** — CE loss 直接对应 Shannon 最优编码长度 (Shannon 1948, Delétang et al. 2024)。

> **关于"无损"口径**：本系统提供两档配置 ——
> (a) `color_transform=bt601`（iGPT-S baseline 默认，对齐工业链路 JPEG/H.26x/VVC）：**YCbCr-int 域无损**。RGB 经 BT.601 + `round()` 进入 YCbCr-int 后，建模/编码/解码链严格无损；但 `round()` 是多对一映射，相对原始 RGB 是**近无损**。
> (b) `color_transform=none`（CC-iGPT 主表）：**RGB-bit-exact 无损**。直接在 RGB uint8 上建模，与 PixelCNN++ / Sparse Transformer 等基线同域可比。CC-iGPT 主表（R-only 灰度先验 + sub-pixel AR）走此档；YCbCr-int 域 CC-iGPT 在 B1 切片 fix 后不再追，仅作脚注保留旧切片语义数字。

- **Phase A (完成)**: iGPT token-level 自回归压缩 + 8 个手写 Triton Kernel（7 个进入训练栈，1 个 `fused_linear_ce` 在 V=256 下经 roofline 分析证伪、保留作反面案例，~2,400 行），iGPT-S CIFAR-10 SWA **2.9739 bits/dim**
- **Phase B (代码完成，主表 RGB-bit-exact 域，重训中)**: CC-iGPT（Coarse-Conditioned iGPT）双尺度条件自回归 — 浅层 coarse iGPT (8×8, 192 token) 独立编码进 bitstream，UP + 量化后通过 additive embedding（可学习标量 α）注入 fine iGPT (32×32, 3072 token)。⚠️ 2026-05-20 修复 `_compute_coarse_ctx` 的 AR 切片对齐（`[:, :-1] → [:, 1:]`，ctx 与被预测位置同位对齐 PixelCNN++ conditional 标准语义）后，**主表收敛到 RGB-bit-exact 两档**（R-only 灰度先验 + sub-pixel AR），不再追 YCbCr-int 域 CC-iGPT 主表与 RGB channel-first 序列实验（旧切片语义结果作脚注保留，不重训覆盖）。
- **Phase C (完成)**: Demo 前端可视化系统 (FastAPI + Chart.js, 5 个展示面板)
- **YCbCr credit 测量 (旧切片语义脚注)**: `configs/ccigpt_cifar10_s_rgb.yaml`（`color_transform=none`, channel-first）在旧切片语义下训练 50 epoch + SWA 实测 **3.2540 bits/dim**，配合 YCbCr-int 主路径 2.8047 给出 YCbCr credit = `0.4493 bits/dim`。该数字作为论文"两域同骨干实测桥"的支撑保留为脚注，但因 channel-first 已被 sub-pixel AR 实测超越（−0.092 bpd）、且 YCbCr-int 主表不再做 B1 切片重训，两个数字仅作旧切片语义对照，不进入主结果。

## Baseline 对比

> ⚠️ **B1 切片对齐 fix (2026-05-20) 与主表范围收敛**：`_compute_coarse_ctx` 末尾 `[:, :-1] → [:, 1:]`，ctx 与被预测位置同位对齐（PixelCNN++ conditional / VAR multi-scale 标准语义）。同时主表收敛到 **RGB-bit-exact 两档**（R-only 灰度先验 + sub-pixel AR），不再做 YCbCr-int 域 CC-iGPT 与 RGB channel-first 重训。下表中 YCbCr-int CC-iGPT 2.8047 与 RGB channel-first 3.2540 是旧切片语义结果，作为对照与"为什么需要 sub-pixel AR"的失败信号脚注保留，**不进入新切片语义下的主结果**。R-only 与 sub-pixel 行将在 B1 切片下重训后回填新数字。重训顺序与决策见 [`future.md`](future.md) §3。

| 方法 | Params | CIFAR-10 bits/dim ↓ | 域 | 来源 |
|------|--------|---------------------|----|------|
| PixelCNN++ | 52M | 2.92 | RGB-bit-exact | Salimans et al., ICLR 2017 |
| Image Transformer | 95M | 2.90 | RGB-bit-exact | Parmar et al., ICML 2018 |
| PixelSNAIL | 380M | 2.85 | RGB-bit-exact | Chen et al., ICML 2018 |
| Sparse Transformer | 59M | 2.80 | RGB-bit-exact | Child et al., 2019 (128 层 strided sparse attention) |
| **iGPT-S (Ours, best)** | **76.05M** | **2.9792** | YCbCr-int | d_model=512, N=24, 200 epochs |
| **iGPT-S (Ours, SWA)** | **76.05M** | **2.9739** | YCbCr-int | SWA averaged over 21 checkpoints |
| **CC-iGPT (Ours, RGB R-only)** | ~81M | **待 B1 重训** | RGB-bit-exact | `color_transform=none` + `coarse_in_channels=1`，coarse 仅压 R 8×8（64 token, ~2% overhead），fine 通过 sub-pixel AR 自学 G/B 偏色（B1 切片下首个完整训练，进行中） |
| **CC-iGPT (Ours, RGB sub-pixel AR)** | ~81M | **待 B1 重训** | RGB-bit-exact | `color_transform=none` + `use_subpixel_ar=true`，序列布局 `[R₀G₀B₀ R₁G₁B₁ ...]` 把通道相关吃进 AR 结构（主表关键数字，B1 切片下重训） |
| ~~CC-iGPT (YCbCr, 旧切片语义)~~ | ~81M | 2.8047 ± 0.0747 (脚注) | YCbCr-int | early-stop @ ep20，旧切片语义；B1 切片下不再重训，作 YCbCr credit 测量配套数字保留 |
| ~~CC-iGPT (RGB channel-first, 旧切片语义)~~ | ~81M | 3.2540 (脚注) | RGB-bit-exact | `color_transform=none`，SWA (50 ep)，旧切片语义；channel-first vs sub-pixel −0.092 bpd 已显示效果不佳，不重训 |
| PNG | — | ~5.87 | RGB-bit-exact | 传统方法 |
| WebP (lossless mode) | — | ~5.02 | RGB-bit-exact | 传统方法 |

> **域口径与脚注数字说明**：上表 iGPT-S 仍以 YCbCr-int 域报告（Phase A 主路径，未涉及 coarse_ctx，B1 不影响）。CC-iGPT **主表两行（R-only / sub-pixel）只走 RGB-bit-exact 域**，与 PixelCNN++ / Sparse Transformer 同域可比。两行划线脚注：YCbCr-int 2.8047 与 RGB channel-first 3.2540 是 B1 切片 fix 之前的旧切片语义结果，仅作以下两个用途保留：
>
> 1. **YCbCr credit 测量**：bpd_RGB − bpd_YCbCr = 3.2540 − 2.8047 = 0.4493 bits/dim，反映 BT.601 通道解相关 + `round()` 量化共同贡献的"可被压缩信息"。据作者所知，文献中此前未见对该差额的系统测量 —— 学界基线（PixelCNN++ / PixelSNAIL / Sparse Transformer）一律在 RGB-bit-exact 上评估，工业编解码器（JPEG/H.26x/VVC）一律在 YCbCr 上操作，本工作以**旧切片语义实测**首次给出两域同骨干的桥（B1 切片下两个数字预期同向变化，credit 差额方向不变）。
> 2. **RGB channel-first vs sub-pixel 对照**：3.2540 − 3.1625 = 0.092 bits/dim，证实"通道相关靠 AR 序列布局吃比靠 coarse_ctx 注入更有效"，sub-pixel 是三档 RGB 配置里唯一真正下降的方案，作为论文 §3.x"为什么需要 sub-pixel AR"的失败信号。
>
> CC-iGPT 主表（B1 切片下 RGB sub-pixel）vs Sparse Transformer 2.80 的差距，归因于：(a) 50 epoch vs 文献基线 200+ epoch 训练预算差；(b) 24 层 dense attention vs Sparse Transformer 128 层 strided sparse attention；(c) PixelCNN++ 系 mixture-of-logistics (DMoL) 已尝试 2026-05 失败 git revert（本仓库 AdamW + DDP + bf16 训练栈与 PixelCNN++ 原版 Adamax + WN + fp32 结构性不兼容，详见 [`future.md`](future.md) §3.5 / §7 F1）；RCT 等 RGB-domain 通道相关化方向未尝试。

### ImageNet 32×32 训练动力学验证（早期 checkpoint，旧切片语义脚注）

受时间预算限制 ImageNet32 完整训练未完成，且 B1 切片 fix 后主表收敛到 CIFAR-10 RGB-bit-exact 两档，ImageNet32 不在 B1 切片下重训。下表为旧切片语义下 epoch 7 best 中间 checkpoint 实测数字，作为**训练动力学外推证据**保留（不进 CIFAR-10 主表，避免与文献基线被误读为同一基准；定性结论"CC-iGPT 在更大数据集上训练动力学保持正常"不依赖切片选择）：

| 方法 | epoch | bits/dim (YCbCr-int) | CE_coarse | CE_fine | α | coarse bit share | 参考 |
|------|-------|----|----|----|---|---|------|
| **CC-iGPT (Ours, ImageNet32, training, 旧切片)** | 7 / 120 | **3.0951 ± 0.0967** (脚注) | 2.8300 | 1.9685 | 0.476 | 8.2% | 本文（旧切片语义，不在 B1 下重训） |
| Sparse Transformer | 充分收敛 | 3.44 | — | — | — | — | Child et al. 2019 (RGB-bit-exact) |

CC-iGPT 7 epoch 的 CE_coarse 已**低于**其 CIFAR-10 ep20 收敛值（2.8300 < 2.9329），说明 ImageNet32 ~1.28M 训练样本使 coarse 模型获得更广分布；CE_fine 仍高于 CIFAR-10（1.9685 > 1.7591），符合 fine 76M 在 7 epoch 远未收敛的预期。α 与 coarse bit share 落在设计预期范围内，验证 CC-iGPT 双尺度条件式注入在更大数据集上**训练动力学保持正常**。完整收敛实验留作后续工作。


### 创新点定位 & 与 SOTA 的关系

本工作并非以击败 Sparse Transformer (CIFAR-10 2.80, 59M, 128 层 strided sparse attention) 为目标。
CC-iGPT 在 24 层 / ~81M 的参数预算下，沿三个维度构建差异化：

1. **方法 — 零新参数的双尺度条件注入**：coarse iGPT 量化 token 经 bit-exact 反量化/上采样/重 tokenize 后，复用 `fine.token_embed` 得到 `coarse_ctx`，再以可学习标量 α 做 additive 注入。整个 ctx 通路只引入 1 个标量参数；encoder/decoder 共用同一函数，bitstream 真实可解码。回避了多尺度联合 AR (MSPA) 的 loss 平衡难题。
2. **工程 — 8 个手写 Triton kernel + 1 个 roofline 证伪的反面案例**：7 个进入训练栈，1 个 `fused_linear_ce` 在 V=256 下经 roofline 分析判定为负收益（compute-bound + 三重循环失去 cuBLAS GEMM 利用率），保留在 `ops/` 作工程严谨性的反向证据，详见 [`experiments/kernel_negative_finding.md`](experiments/kernel_negative_finding.md)。
3. **分析 — 域口径诚实标注 + 多视角评估 + RGB-bit-exact 三档实测对照**：明确区分 **YCbCr-int 域无损**（iGPT-S baseline 主路径，相对原始 RGB 近无损）与 **RGB-bit-exact 无损**（CC-iGPT 主表路径，相对原始 RGB 严格无损）两档语义；同时报告两域 bpd（YCbCr credit 测量作为旧切片语义脚注），Linear Probe 逐层表征曲线，Roofline forward 与 fwd+bwd 双视角。RGB-bit-exact 域内进一步给出三档实测对照（channel-first 3.2540 / YCoCg-R lifting 3.2630 / **sub-pixel AR 3.1625**，均旧切片语义），证实"通道相关靠 AR 序列布局吃比靠 coarse_ctx 注入更有效"，sub-pixel 是三档里唯一真正下降的方案（−0.092 bpd）；B1 切片 fix 后主表只重训 R-only + sub-pixel 两档（YCbCr-int CC-iGPT 主表与 channel-first 不再追，作脚注）。

与 Sparse Transformer 的差距（B1 重训后 RGB sub-pixel vs 2.80）来自参数预算（81M vs 59M）/ 深度（24 vs 128 层）/ 训练 epoch（50 vs 200+）/ DMoL 输出头已尝试失败 git revert（详见 [`future.md`](future.md) §3.5 / §7 F1）/ 未实现 strided sparse attention 与 RCT 等 RGB-domain 通道相关化，而非方法路线缺陷。

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

# 训练 — CC-iGPT RGB-domain ablation (color_transform=none)
torchrun --nproc_per_node=2 scripts/train.py --config configs/ccigpt_cifar10_s_rgb.yaml

# 断点续训 (resume 会自动用 config 中的 lr 覆盖 checkpoint 旧值)
torchrun --nproc_per_node=2 scripts/train.py \
    --config configs/igpt_cifar10_s.yaml \
    --resume experiments/igpt_cifar10_s/checkpoints/epoch_100.pth

# 评测 — iGPT
python scripts/evaluate.py --config configs/igpt_cifar10_s.yaml \
    --checkpoint experiments/igpt_cifar10_s/checkpoints/best.pth

# 评测 — CC-iGPT (含 coarse / fine CE 分解 + bpd_total)
python scripts/evaluate.py --config configs/ccigpt_cifar10_s.yaml \
    --checkpoint experiments/ccigpt_cifar10_s/checkpoints/best.pth
# SWA vs best 对比（同时评测 best.pth 和 swa.pth）
python scripts/evaluate.py --config configs/ccigpt_cifar10_s.yaml \
    --checkpoint experiments/ccigpt_cifar10_s/checkpoints/best.pth --swa

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
# ⚠️ AutoDL 多卡 DDP 必须设置 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1，
#    否则 NCCL init 会卡死或报 "unhandled cuda error"（共享 GPU 实例无 P2P/IB）。
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
    torchrun --nproc_per_node=2 scripts/train.py --config configs/igpt_cifar10_s.yaml
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
    torchrun --nproc_per_node=2 scripts/train.py --config configs/ccigpt_cifar10_s.yaml

# 后台挂起 (断开 SSH 不中断) — 推荐写法，NCCL flag 必带
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
nohup torchrun --nproc_per_node=2 scripts/train.py \
    --config configs/ccigpt_cifar10_s.yaml \
    > train_ccigpt.log 2>&1 &
tail -f train_ccigpt.log

# CC-iGPT RGB-bit-exact R-only coarse 改造 (sub-pixel AR + R-only 灰度先验)
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
nohup torchrun --nproc_per_node=2 scripts/train.py \
    --config configs/ccigpt_cifar10_s_rgb_ronly.yaml --export_csv \
  > experiments/ccigpt_cifar10_s_rgb_ronly_train.log 2>&1 &

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
  +-------+--------+                  | bit-exact ctx 路径:    |
          |                           |   coarse 量化 token    |
          |                           |   → 反量化 (BT.601⁻¹)  |
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
            |  [Y_all | Cb_all | Cr_all]         |  [Y₀,Cb₀,Cr₀, Y₁,Cb₁,Cr₁, ...]
            |  (YCbCr-int 主路径默认)             |  (RGB-bit-exact 路径 + iGPT-S 默认)
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
                       (coarse + fine 联合压缩率，N_f = H·W·C；
                        bpd = bits per dimension/sub-pixel；
                        bpp = bpd × C)
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
| Cosine + Warmup LR | 线性 warmup → cosine 衰减；可选 `min_lr_ratio` 末段 LR 下限避免 SWA "假平均" | OLMo 2 (2025) |
| SWA | Stochastic Weight Averaging，训练后期对权重做指数移动平均，获得更平坦的 loss landscape | Izmailov et al. 2018 |
| DDP + no_sync | 多 GPU 分布式训练，梯度累积中间步跳过 AllReduce 通信 | PyTorch DDP |
| Selective Checkpointing | 只对 Attention 层做 activation checkpointing，平衡显存和速度 | Chen et al. 2016 |
| Mixed Precision | bf16/fp16 自动混合精度训练 | PyTorch AMP |

### CC-iGPT (`models/cc_igpt.py`)

Coarse-Conditioned iGPT —— 双尺度条件式自回归。回避了多尺度 loss 平衡难题，复用全部 iGPT 训练栈与 Triton kernels。

| 组件 | 说明 |
|------|------|
| DOWN | `F.adaptive_avg_pool2d(x, 8)` 在 float 域下采样到 8×8 |
| Coarse iGPT | 浅层（d_model=256, N=6），独立 NTP 训练，CE 进 bitstream（192 token，~6% overhead） |
| **Bit-exact ctx 路径** | encoder/decoder 必须看到**同一个** `coarse_ctx`，否则 fine 端算术编码不可解。统一管线：coarse 量化 token → 反量化 RGB（`bt601` 走 BT.601 inverse，`ycocg_r` 走整数 lifting 逆变换，`none` 直接 /255）→ bilinear UP 到 32×32 → 与 fine encoder 同规则 re-tokenize → `fine.token_embed` |
| Ctx 注入 | AR shift `coarse_ctx[:, 1:]`（ctx[i] 对应 fine 被预测位置 i，PixelCNN++ conditional 标准语义）→ `α · coarse_ctx`（additive，仅引入 1 个标量参数 α） |
| 可学习 α | `nn.Parameter(torch.ones(1))`，初始 1.0；模型自适应注入强度，避免 ctx 过强压制 fine token embed |
| 联合 bits/dim | `bpd_total = (CE_c · 192 + CE_f · 3072) / ln(2) / 3072`（按 H·W·C 子像素数归一化） |
| 训练 | 端到端联合 `loss = loss_coarse + loss_fine`，无尺度间加权 |
| 关闭 ctx | `fine(x, coarse_ctx=None)` 严格等价 vanilla iGPT（unit test 校验 CE diff < 1e-6） |
| 一致性回归 | `tests/test_ccigpt_smoke.py` 同时断言 (a) encoder/decoder ctx max diff < 1e-6 (b) `coarse_tokens.view(B,C,S,S)` 与 `rgb_to_ycbcr_int(x_c_float)` byte-exact，覆盖 YCbCr/RGB 两条路径 |

设计参考: Burt & Adelson "Laplacian Pyramid" (1983)、van den Oord "Conditional PixelCNN" (NeurIPS 2016, additive 条件)、Tian "VAR" (NeurIPS 2024)。

### 共享层 (`models/layers.py`)

GPTBlock, MultiHeadAttentionBlock (RoPE + QK-Norm + Flash Attention + attn_mask), RMSNorm, SwiGLU FFN。RoPE 的 `cos/sin` 按 `(seq_len, device)` 缓存，避免逐层逐步重算。

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
| Fused Attn+RoPE | ~62 | 薄包装层：组合 fused_rope + flash_attn，简化 layers.py 调用 |
| ~~Fused Linear+CE~~ | ~393 | **反面案例** — V=256 下三重循环失去 cuBLAS GEMM 利用率，比 PyTorch 慢 100× |

全部 7 个生效 kernel 自动降级到 PyTorch，全部有独立单元测试。

### Linear Probe (`scripts/linear_probe.py`)

冻结预训练模型，通过 `IGPT.encode(x, max_layer)` 直接取各层 hidden state（跳过 output head 与 loss），全局平均池化 → 线性分类器 → 报告**训练结束时**的测试准确率（避免按 epoch 选 max 造成 test-set peeking）。
对比预训练表征质量 (Chen et al. 2020)。

## 项目结构

```
src/mdlic/
├── models/    igpt.py, cc_igpt.py, layers.py
├── ops/       7 个 Triton kernels (flash_attn, fused_rms_norm, ...) + 1 反面案例 (fused_linear_ce)
├── data/      imagenet32_npy.py (mmap-backed Dataset)
└── utils/     seed, bpd, clean_state_dict
scripts/       train.py, evaluate.py, linear_probe.py, dryrun_forward.py, profile_kernels.py, prepare_imagenet32.py
configs/       igpt_cifar10_s, igpt_cifar10_s_rgb, igpt_cifar100_s, igpt_imagenet32_s,
               ccigpt_cifar10_s, ccigpt_cifar10_s_rgb, ccigpt_cifar10_s_rgb_subpixel,
               ccigpt_cifar10_s_ycocg, ccigpt_imagenet32_s
tests/         8 个 kernel/模型 单元测试 (含 test_ccigpt_smoke 16 项)
demo/
├── server.py          FastAPI 后端 (predict / metrics / probe / kernels / scales)
├── static/            HTML + JS (Chart.js) + CSS 前端，5 个面板：上传预测、bits/dim 对比、
│                       Linear Probe、Triton kernel 加速比、CC-iGPT coarse/fine token 分配
└── data/              预计算 JSON 数据 (训练后替换为真实结果)
```

**参考文献**

**模型**: iGPT (Chen 2020), RoPE (Su 2021), RMSNorm (Zhang 2019), SwiGLU (Shazeer 2020), OLMo 2 (2025), QK-Norm (Dehghani 2023), Weight Tying (Press 2017), Linear Probe (Alain & Bengio 2017)

**像素自回归**: PixelCNN++ (Salimans 2017), PixelSNAIL (Chen 2018), PixelCNN (van den Oord 2016), Sparse Transformer (Child 2019)

**多尺度**: VAR (Tian 2024), VQ-VAE (van den Oord 2017)

**Triton**: FlashAttention v1/v2 (Dao 2022/2023), Online Softmax (Milakov 2018), Liger Kernel (Hsu 2024)

**训练**: SWA (Izmailov 2018), Cosine + Warmup (OLMo 2 2025)

**理论**: Shannon (1948), MDL (Rissanen 1978), Language Modeling Is Compression (Delétang 2024)
