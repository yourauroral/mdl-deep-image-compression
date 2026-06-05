# MDL Deep Image Compression

基于 **Minimum Description Length (MDL)** 原则的深度图像压缩系统设计与实现。核心命题：**压缩即预测** — CE loss 直接对应 Shannon 最优编码长度 (Shannon 1948, Delétang et al. 2024)。

> **无损口径**：本系统所有主结果都在 **RGB-bit-exact** 域报告 — 直接在 RGB uint8 上建模，与 PixelCNN++ / Sparse Transformer 等基线同域可比。

- **Phase A (完成)**: iGPT token-level 自回归压缩 + 8 个手写 Triton Kernel（7 个进入训练栈，1 个 `fused_linear_ce` 在 V=256 下经 roofline 分析证伪、保留作反面案例）
- **Phase B (v1 历史主表完成)**: CC-iGPT（Coarse-Conditioned iGPT）双尺度条件自回归 — 浅层 coarse iGPT (R-only 配置 8×8×1, 64 token, ~2% overhead) 独立编码进 bitstream，UP + 量化后通过 additive embedding（可学习标量 α）注入 fine iGPT (32×32×3, 3072 token)。CIFAR-10 RGB-bit-exact R-only v1 历史主表 **2.9035 bpd**（softmax head, 100ep + 全套正则 + TTA hflip，已被 v2 替代）
- **Phase C (完成)**: Demo 前端可视化系统 (FastAPI + Chart.js, 9 个展示面板，含交互式无损 codec 图像⇄.bin 真实可解性验证 + 下游 OOD typicality / 跨数据集 bpd / 图像补全 AR inpainting 实时面板)
- **Phase D (完成, 2026-05-27)**: 深窄 + ensemble — fine N=24/d=512 → N=32/d=448 (82.95M)、epoch 100→200、`min_lr_ratio=0.05` + SWA last 31 ckpts (start ep170) + EMA 0.9998；`evaluate.py --ensemble` 多 ckpt probability-mixture ensemble (best+SWA+EMA)。**主表 ensemble + TTA hflip = 2.8296 ± 0.0854**（超越 PixelSNAIL 380M 2.85，逼近 Sparse Transformer 59M 2.80）。详见 [future.md §4](future.md)

## Baseline 对比

| 方法 | 类别 | Params | CIFAR-10 bits/dim ↓ | 域 | 来源 |
|------|------|--------|---------------------|----|------|
| PixelCNN++ | Autoregressive | 52M | 2.92 | RGB-bit-exact | Salimans et al., ICLR 2017 |
| Image Transformer | Autoregressive | 95M | 2.90 | RGB-bit-exact | Parmar et al., ICML 2018 |
| PixelSNAIL | Autoregressive | 380M | 2.85 | RGB-bit-exact | Chen et al., ICML 2018 |
| Sparse Transformer | Autoregressive | 59M | 2.80 | RGB-bit-exact | Child et al., 2019 (128 层 strided sparse attention) |
| **CC-iGPT v2 (Ours, R-only)** | Autoregressive | 82.95M | **2.8296** ± 0.0854 | RGB-bit-exact | `coarse_in_channels=1`，coarse 仅压 R 8×8（64 token），fine 通过 sub-pixel AR 自学 G/B，**主表**（200ep + RandomCrop + DropPath 0.1 + EMA 0.9998 + SWA last 31 ckpts + ensemble best/SWA/EMA + TTA hflip）|
| CC-iGPT v1 (Ours, R-only) | Autoregressive | ~81M | 2.9035 ± 0.0854 | RGB-bit-exact | v1 历史主表（100ep + EMA 0.9995 + TTA hflip）|
| PNG | Classical codec | — | 5.87 | RGB-bit-exact | Hoogeboom et al., NeurIPS 2019 报告 |
| WebP (lossless) | Classical codec | — | 4.61 | RGB-bit-exact | Hoogeboom et al., NeurIPS 2019 报告 |

CC-iGPT v2 R-only **超越 PixelSNAIL 380M (2.85)、逼近 Sparse Transformer 59M (2.80)**，参数预算 82.95M。主表 ensemble (best+SWA+EMA) + TTA hflip = **2.8296 ± 0.0854**（vs PixelSNAIL gap -0.020 / vs Sparse Trans gap +0.030）。SOTA 路线靠 ImageNet 64×64 < 3.44（Sparse Transformer 152M strided）跨数据集验证支撑。

### 创新点定位

1. **方法 — 零新参数的双尺度条件注入**：coarse iGPT 量化 token 经 bit-exact 反量化/上采样/重 tokenize 后，复用 `fine.token_embed` 得到 `coarse_ctx`，再以可学习标量 α 做 additive 注入。整个 ctx 通路只引入 1 个标量参数；encoder/decoder 共用同一函数，bitstream 真实可解码。回避了多尺度联合 AR 的 loss 平衡难题。
2. **工程 — 8 个手写 Triton kernel + 1 个 roofline 证伪的反面案例**：7 个进入训练栈，1 个 `fused_linear_ce` 在 V=256 下经 `scripts/profile_kernels.py --kernel fused_linear_ce --roofline` 分析判定为负收益（compute-bound + 三重循环失去 cuBLAS GEMM 利用率），保留作工程严谨性的反向证据。
3. **分析 — RGB-bit-exact 主表 + 跨数据集 SOTA 对比**：CIFAR-10 R-only v2 主表 **2.8296 bpd**（超越 PixelSNAIL 380M 2.85）；ImageNet 64×64 跨数据集（目标 < 3.44）；Linear Probe 逐层表征曲线；roofline forward 与 fwd+bwd 双视角。

## 快速开始

```bash
pip install -e ".[dev]"
# AutoDL / CUDA kernel profiling:
pip install -e ".[cuda,dev]"
# Demo / ImageNet64 parquet preprocessing:
pip install -e ".[demo,imagenet-prep]"
```

```bash
# 单元测试（WSL CPU 即可）
pytest -m cpu -v
# AutoDL CUDA kernel tests:
pytest -m cuda -v

# 训练 — 多卡 DDP (按 GPU 数调整 nproc_per_node)
torchrun --nproc_per_node=2 scripts/train.py --config configs/igpt_cifar10_s_rgb.yaml

# CC-iGPT RGB-bit-exact v1 (R-only 历史主表 — softmax head, 2.9035 bpd, 100ep, 已被 v2 替代)
torchrun --nproc_per_node=2 scripts/train.py --config configs/ccigpt_cifar10_s_rgb_ronly.yaml

# CC-iGPT v2 — 深窄 N=32/d=448 + 200ep (当前主表 ensemble+TTA = 2.8296, 详见 future.md §4)
torchrun --nproc_per_node=2 scripts/train.py --config configs/ccigpt_cifar10_s_rgb_ronly_v2.yaml

# 评测 — CC-iGPT v2 主表数字 (ensemble best+SWA+EMA + TTA hflip)
python scripts/evaluate.py --config configs/ccigpt_cifar10_s_rgb_ronly_v2.yaml \
    --ensemble experiments/ccigpt_cifar10_s_rgb_ronly_v2/checkpoints/best.pth,\
experiments/ccigpt_cifar10_s_rgb_ronly_v2/checkpoints/swa.pth,\
experiments/ccigpt_cifar10_s_rgb_ronly_v2/checkpoints/ema.pth --tta_hflip

# 评测 — CC-iGPT 单 ckpt (含 coarse / fine CE 分解 + bpd_total)
python scripts/evaluate.py --config configs/ccigpt_cifar10_s_rgb_ronly_v2.yaml \
    --checkpoint experiments/ccigpt_cifar10_s_rgb_ronly_v2/checkpoints/best.pth --tta_hflip
# 单 ckpt per-image 统计（std / stderr / bootstrap CI；可导出 JSON）
python scripts/evaluate.py --config configs/ccigpt_cifar10_s_rgb_ronly_v2.yaml \
    --checkpoint experiments/ccigpt_cifar10_s_rgb_ronly_v2/checkpoints/best.pth \
    --tta_hflip --per_image_stats --per_image_json experiments/per_image_bpd.json

# Linear Probe (各层表征分类准确率 — CC-iGPT v2 fine + α·coarse_ctx，32 层 L19 best 79.33%)
python scripts/linear_probe.py --config configs/ccigpt_cifar10_s_rgb_ronly_v2.yaml \
    --checkpoint experiments/ccigpt_cifar10_s_rgb_ronly_v2/checkpoints/best.pth --layers all

# ── 下游任务（论文 §5 MDL 主线；批量执行见 downstream/runbook.md）──
# 一键跑全套（self_test 预检 + 分步计时 + 失败隔离）
bash downstream/run_downstream.sh

# OOD 检测 (typicality；三 scorer raw_bpd/typ_total/typ_dualscale，绕 Nalisnick 坑)
python scripts/ood_detect.py --config configs/ccigpt_cifar10_s_rgb_ronly_v2.yaml \
    --checkpoint experiments/ccigpt_cifar10_s_rgb_ronly_v2/checkpoints/best.pth \
    --ood svhn,cifar100 --ref_images 2000

# 跨数据集 bpd 泛化 (cifar100 / svhn / stl10；resize 到模型分辨率，仅看相对值)
python scripts/evaluate.py --config configs/ccigpt_cifar10_s_rgb_ronly_v2.yaml \
    --checkpoint experiments/ccigpt_cifar10_s_rgb_ronly_v2/checkpoints/best.pth --dataset_override cifar100

# 图像补全 (AR inpainting；存 原图|已知上半|补全 网格 PNG)
python scripts/complete_image.py --config configs/ccigpt_cifar10_s_rgb_ronly_v2.yaml \
    --checkpoint experiments/ccigpt_cifar10_s_rgb_ronly_v2/checkpoints/best.pth \
    --num_images 4 --keep_frac 0.5 --out experiments/completion_grid.png

# 真实可解性 roundtrip (算术编解码，断言逐像素 bit-identical)
python scripts/verify_lossless.py --config configs/ccigpt_cifar10_s_rgb_ronly_v2.yaml \
    --checkpoint experiments/ccigpt_cifar10_s_rgb_ronly_v2/checkpoints/best.pth --num_images 2
python scripts/verify_lossless.py --self_test   # 仅 coder roundtrip，无需 GPU/ckpt
# bitstream 落盘为自包含 MDLC .bin（图像→bits→文件→bits→图像 全链路），再只读解析
python scripts/verify_lossless.py --config <yaml> --checkpoint <best.pth> --num_images 2 --dump_dir experiments/bitstreams
python scripts/verify_lossless.py --inspect experiments/bitstreams/img0.bin   # 只读：结构/hex/码长/bpd，无需 GPU/ckpt

# Kernel Profiling
python scripts/profile_kernels.py --roofline

# ImageNet64 传统 codec baseline（PNG/WebP lossless bpd，默认抽样 2000 张）
python scripts/traditional_codec_bpd.py /root/autodl-tmp/imagenet64_png/val.npy --limit 2000

# Demo 前端 (9 面板可视化：①上传→bpd 热力图 / ②baseline 对比 / ③Linear Probe / ④Kernel 性能 / ⑤coarse+fine 双尺度 / ⑥OOD typicality / ⑦跨数据集 bpd / ⑧图像补全 AR inpainting / ⑨交互式无损 codec 图像⇄.bin 真实可解性验证)
# 下游 Panel 7/8 数据由 AutoDL 跑 `bash downstream/run_downstream.sh` 带 --json_out 回填 demo/data/{ood,transfer}.json；Panel 9 实时调 /api/complete
# ckpt 优先级: v2 (2.8296 主表) → v1 历史 (2.9035)
pip install -e ".[demo]"

# 本地 / WSL: localhost 默认 8000
uvicorn demo.server:app --reload --port 8000      # http://localhost:8000

# AutoDL: 仅 6006 / 6008 端口可被公网映射，必须 --host 0.0.0.0
uvicorn demo.server:app --host 0.0.0.0 --port 6006 --reload   # 见实例详情公网映射地址
```

### AutoDL 训练流程

本地 WSL 只做开发和 dry-run，训练一律在 AutoDL GPU 实例上执行。

```bash
# 本地: 推送
git add -A && git commit -m "sync to autodl" && git push origin dev

# AutoDL: 训练 (多卡 DDP 必须设置 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1)
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
nohup torchrun --nproc_per_node=2 scripts/train.py \
    --config configs/ccigpt_cifar10_s_rgb_ronly_v2.yaml --export_csv \
  > experiments/ccigpt_cifar10_s_rgb_ronly_v2_train.log 2>&1 &
# 新启动的训练会在 checkpoints/ 旁路写 best.meta.json / ema.meta.json / swa.meta.json，
# 记录 epoch、config、seed、bpd/std 等 provenance；不改变 .pth 格式或 --resume 行为。

# AutoDL → 本地: 回收 checkpoint
scp -P <port> root@<autodl-host>:/root/autodl-tmp/mdl-deep-image-compression/experiments/<exp>/checkpoints/best.pth \
    ./experiments/<exp>/checkpoints/
```

## 架构

### 模型架构图

```
                    RGB Image [B, 3, 32, 32]   (uint8, RGB-bit-exact)
                              |
          +-------------------+-------------------+
          |                                       |
    iGPT (Phase A)                       CC-iGPT (Phase B / D 完成)
          |                                       |
  +-------v--------+                  +-----------v-----------+
  | Flatten: 3072  |                  | DOWN avg_pool 8×8     |
  | tokens         |                  |   → Coarse iGPT       |
  | (3×32×32)      |                  |     R-only 8×8×1=64   |
  +-------+--------+                  |     tokens, 进 bs     |
          |                           | bit-exact ctx 路径:    |
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
                    +---------v-----------+
                    | 序列布局: R-only 主表 |
                    | coarse: R 8×8 (64)  |
                    | fine: sub-pixel AR  |
                    | [R₀,G₀,B₀, R₁,...] |
                    +---------+-----------+
                              |
                    +---------v-----------+
                    | 自回归移位:          |
                    |  input  = x[:-1]    |
                    |  target = x[1:]     |
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
                    |    Output Head      |--- Weight Tying (softmax)
                    |   Linear → 256-way  |    (head.weight ↔ token_embed)
                    +---------+-----------+
                              |
                    +---------v-----------+
                    |     CE + z-loss     |
                    |  (Fused CE Triton)  |
                    +---------+-----------+
                              |
              iGPT:    bpd = CE × T / ln(2) / (H·W·C)
              CC-iGPT: bpd_total = (CE_c · N_c + CE_f · N_f) / ln(2) / N_f
                       (coarse + fine 联合压缩率，N_f = H·W·C)
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
| Coarse iGPT | 浅层（d_model=256, N=6），独立 NTP 训练，CE 进 bitstream。R-only 主表配置仅压 R 通道 8×8（64 token, ~2% overhead） |
| **Bit-exact ctx 路径** | encoder/decoder 必须看到**同一个** `coarse_ctx`，否则 fine 端算术编码不可解。统一管线：coarse 量化 token → `/255` 反量化为 RGB float → bilinear UP 到 32×32 → 与 fine encoder 同规则 re-tokenize → `fine.token_embed` |
| Ctx 注入 | AR shift `coarse_ctx[:, 1:]`（ctx[i] 对应 fine 被预测位置 i，PixelCNN++ conditional 标准语义）→ `α · coarse_ctx`（additive，仅引入 1 个标量参数 α） |
| 可学习 α | `nn.Parameter(torch.ones(1))`，初始 1.0；模型自适应注入强度 |
| 联合 bits/dim | `bpd_total = (CE_c · N_c + CE_f · N_f) / ln(2) / N_f` |
| 训练 | 端到端联合 `loss = loss_coarse + loss_fine`，无尺度间加权 |

设计参考: Burt & Adelson "Laplacian Pyramid" (1983)、van den Oord "Conditional PixelCNN" (NeurIPS 2016, additive 条件)、Tian "VAR" (NeurIPS 2024)。

### 共享层 (`models/layers.py`)

GPTBlock, MultiHeadAttentionBlock (RoPE + QK-Norm + Flash Attention + attn_mask), RMSNorm, SwiGLU FFN。RoPE 的 `cos/sin` 按 `(seq_len, device)` 缓存。

### 手写 Triton Kernels (`ops/`, ~2,400 行, 8 个 kernel)

7 个进入训练栈，1 个 (`fused_linear_ce`) 经 `scripts/profile_kernels.py --kernel fused_linear_ce --roofline` 分析在 V=256 下负收益、保留作反面案例。

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

### Linear Probe (`scripts/linear_probe.py`)

冻结预训练模型，通过 `IGPT.encode(x, max_layer)` 直接取各层 hidden state，全局平均池化 → 线性分类器 → 报告**训练结束时**的测试准确率。对比预训练表征质量 (Chen et al. 2020)。

## 项目结构

```
src/mdlic/
├── models/    igpt.py, cc_igpt.py, layers.py
├── ops/       7 个 Triton kernels + 1 反面案例 (fused_linear_ce)
├── data/      imagenet64_npy.py (mmap-backed Dataset)
└── utils/     seed, bpd, clean_state_dict
scripts/       train.py (checkpoint sidecar *.meta.json), evaluate.py (含 --ensemble / --dataset_override / --json_out / --per_image_stats), linear_probe.py (含 --probe_dataset transfer),
               ood_detect.py (--json_out), complete_image.py (--scale), verify_lossless.py (--dump_dir 落盘 .bin / --inspect 只读解析), dryrun_forward.py, profile_kernels.py,
               traditional_codec_bpd.py, prepare_imagenet64_png.py
configs/       igpt_cifar10_s_rgb,
               ccigpt_cifar10_s_rgb_ronly      (R-only v1 历史主表 2.9035 bpd, 100ep, 已被 v2 替代),
               ccigpt_cifar10_s_rgb_ronly_v2   (深窄 N=32/d=448 + 200ep, 当前主表 ensemble+TTA 2.8296 bpd)
downstream/    runbook.md (下游任务执行手册) + run_downstream.sh (AutoDL 批量执行)
tests/         单元测试（含 test_ccigpt_smoke / test_arithmetic_codec / test_ood_math）
demo/
├── server.py          FastAPI 后端 (predict / encode / inspect / decode / complete / metrics / probe / kernels / scales / ood / transfer)
│                      ckpt 加载优先级: v2 → ronly softmax
├── static/            HTML + JS (Chart.js) + CSS 前端，9 个面板
└── data/              预计算 JSON 数据（含 ood.json / transfer.json，下游 --json_out 回填）
```

**参考文献**

**模型**: iGPT (Chen 2020), RoPE (Su 2021), RMSNorm (Zhang 2019), SwiGLU (Shazeer 2020), OLMo 2 (2025), QK-Norm (Dehghani 2023), Weight Tying (Press 2017), Linear Probe (Alain & Bengio 2017)

**像素自回归**: PixelCNN++ (Salimans 2017), PixelSNAIL (Chen 2018), PixelCNN (van den Oord 2016), Sparse Transformer (Child 2019)

**多尺度**: VAR (Tian 2024), VQ-VAE (van den Oord 2017), Multi-Scale PixelCNN (Reed 2017), Subscale Pixel Networks (Menick 2018)

**Triton**: FlashAttention v1/v2 (Dao 2022/2023), Online Softmax (Milakov 2018), Liger Kernel (Hsu 2024)

**训练**: SWA (Izmailov 2018), Cosine + Warmup (OLMo 2 2025)

**理论**: Shannon (1948), MDL (Rissanen 1978), Language Modeling Is Compression (Delétang 2024)
