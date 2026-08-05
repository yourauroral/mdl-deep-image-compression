# MDL Deep Image Compression

基于 **Minimum Description Length (MDL)** 原则的深度图像压缩系统设计与实现。核心命题：**压缩即预测** — 在双方预共享模型与概率协议时，next-token CE 对应条件数据码长 (Shannon 1948, Delétang et al. 2024)。当前 bpd 不包含模型权重描述长度；严格两部 MDL 还需计入 `L(model)`。

> **无损口径**：本系统所有主结果都在 **RGB-bit-exact** 域报告 — 直接在 RGB uint8 上建模，与 PixelCNN++ / Sparse Transformer 等基线同域可比。

- **Phase A (完成)**: iGPT token-level 自回归压缩 + 6 个进入训练栈的手写 Triton primitive；RoPE→FlashAttention 是两个 kernel 的 pipeline，`fused_linear_ce` 是 V=256 下的负收益反面案例
- **Phase B (历史实验完成)**: CC-iGPT（Coarse-Conditioned iGPT）双尺度条件自回归 — 浅层 coarse iGPT (R-only 配置 8×8×1, 64 token) 独立编码进 bitstream，UP + 量化后通过 additive embedding（可学习标量 α）注入 fine iGPT (32×32×3, 3072 token)。v1 的 **2.9035 bpd** 是 TTA 诊断协议下的历史实验结果，已被 v2 架构替代
- **Phase C (完成)**: Demo 前端可视化系统 (FastAPI + Chart.js, 7 个展示面板，含交互式无损 codec 图像⇄.bin 真实可解性验证 + 图像补全 AR inpainting 实时面板)
- **Phase D (历史实验完成, 2026-05-27)**: fine N=24/d=512 → N=32/d=448；每个 checkpoint 对应同一套 `82,949,441` 参数架构，训练 200 epoch 后产出 best/SWA/EMA 三组权重。`2.8296` 是三成员 probability-mixture ensemble + hflip TTA 协议下的历史实验结果：评测时加载 3 个 checkpoint，并对每张图执行 6 次 forward；在新 evaluator 中归类为诊断协议，旧 `±0.0854` 统计口径已作废
- **CIFAR-10 formal benchmark (正式 teacher-forced 评测完成, 2026-08-04)**: 现有 `best.pth` 的单 checkpoint、no-TTA 正式结果为 **2.8328 bpd**（10,000 张，逐图 std `0.6719`，bootstrap 95% CI `[2.8201, 2.8456]`）。同一 codec identity 下的 2 张 sequential roundtrip 已 `verified_on_subset`：两张均 pixel-exact，manifest 已保存于 `results/formal/cifar10/`
- **ImageNet64 benchmark (正式 teacher-forced 评测完成, 2026-08-03)**: 历史 `3.4800 bpd` 仍是三成员 ensemble + hflip TTA 诊断结果；现有 `best.pth` 的单 checkpoint、no-TTA 正式结果为 **3.4812 bpd**（49,999 张，逐图 std `0.9040`，bootstrap 95% CI `[3.4734, 3.4898]`）。schema v5 manifest 记录实际 `.npy` 样本数、数据文件 SHA-256 和运行环境；sequential roundtrip 尚未执行
- **ImageNet64 传统 codec baseline (完整复核完成, 2026-08-06)**: 验证集完整 49,999 张的 PNG `optimize=True` 为 **5.7063 bpd**，WebP `lossless=True` 为 **4.6365 bpd**（均为包含文件头的 complete-file bpd）；运行时为 Pillow 10.3.0 / zlib 1.2.13 / libwebp 1.3.2，manifest 见 [`results/formal/imagenet64/traditional_codecs_val_full.json`](results/formal/imagenet64/traditional_codecs_val_full.json)

## 当前进度

| 模块 | 状态 | 备注 |
|---|---|---|
| CIFAR-10 v2 正式主表 | 已重评（teacher-forced） | 单 `best.pth`、no-TTA：**2.8328 bpd**；10,000 张，95% bootstrap CI `[2.8201, 2.8456]`；2 张 sequential `verified_on_subset`，均 pixel-exact |
| Linear probe | 待重跑 | 旧 `79.33%/73.19%` 曲线曾用 test 选层；新脚本改为 validation 选层、多 classifier seeds、完整 train 重训、最终单层 test |
| Demo 前端 | 完成 | 7 面板；保留 upload / metrics / probe / kernels / scales / completion / codec |
| 辅助实验 | CLI 已收敛 | 保留 `linear_probe.py`、`complete_image.py`、`verify_lossless.py`；直接调用脚本，不再维护批量包装层 |
| ImageNet64 正式主表 | 已重评（teacher-forced） | 单 `best.pth`、no-TTA：**3.4812 bpd**；49,999 张，95% bootstrap CI `[3.4734, 3.4898]`；真实 sequential roundtrip 仍 pending |
| ImageNet64 传统 codec baseline | 已完成（完整验证集） | PNG **5.7063 bpd**；WebP lossless **4.6365 bpd**；49,999 张；complete-file bpd |
| CC-MDLM 研究线 | 协议地基完成 | deterministic `MaskSchedule`、独立双向 `forward_masked`、group-frozen arithmetic codec 与 tiny CPU roundtrip；尚未训练、无 bpd 结论 |
| 表征理论 | 文档完成 | [theory.md](theory.md)：linear probe 为何有效（MDL + LRH + MDL probing 三段论），含阅读清单与实验菜单 |
| 本地验证 | 通过 | CPU 回归、`compileall`、coder self-test 与 `git diff --check` 通过；CUDA/Triton 仍待 GPU 环境验证 |

## Baseline 对比

| 方法 | 类别 | Params | CIFAR-10 bits/dim ↓ | 域 | 来源 |
|------|------|--------|---------------------|----|------|
| PixelCNN++ | Autoregressive | ≈53.5M | 2.92 | RGB-bit-exact | Salimans et al., ICLR 2017 |
| Image Transformer | Autoregressive | ≈40M | 2.90 | RGB-bit-exact | Parmar et al., ICML 2018 |
| PixelSNAIL | Autoregressive | ≈91M | 2.85 | RGB-bit-exact | Chen et al., ICML 2018 |
| Sparse Transformer | Autoregressive | 59M | 2.80 | RGB-bit-exact | Child et al., 2019 (128 层 strided sparse attention) |
| CC-iGPT v2 (Ours, historical experiment; diagnostic protocol) | Autoregressive ensemble | 82.95M/member (K=3) | 2.8296 | RGB-bit-exact | best/SWA/EMA probability mixture + hflip TTA；6 forwards/image；非正式单模型主表 |
| **CC-iGPT v2 (Ours, formal protocol)** | Autoregressive | 82.95M | **2.8328** | RGB-bit-exact | 单 `best.pth`、no-TTA、10,000 张；schema v5 manifest；teacher-forced ideal-model NLL |
| PNG | Classical codec | — | 5.87 | RGB-bit-exact | 本项目全量 CIFAR-10 test 实测；Pillow PNG `optimize=True` |
| WebP (lossless) | Classical codec | — | 4.61 | RGB-bit-exact | 本项目全量 CIFAR-10 test 实测；Pillow WebP `lossless=True` |

> **注**：Image Transformer 原论文未报告总参数量；表中 ≈40M 系据官方实现 tensor2tensor `imagetransformer_cifar10_base`（12 层 / d=512 / ff=2048）估算。PixelSNAIL 原论文亦未报告参数量；表中 ≈91M 系据官方实现 `neocxi/pixelsnail-public` 的 CIFAR-10 配置（`h12_noup_smallkey`, `nr_filters=256`, `nr_resnet=4` 默认值 — 官方 README 的 CIFAR 训练命令未覆盖 `--nr_resnet`）逐层核算，脚本见 [`scripts/pixelsnail_paramcount.py`](scripts/pixelsnail_paramcount.py)（PixelCNN++ 校验锚 55.35M 对齐公开 ~53.5M）；原所列 380M 无出处、已撤下。Sparse Transformer 的 59M（CIFAR-10, 128L/d256）与 152M（ImageNet64, 48L/d512）均为原论文自报。

> **传统 codec 复核（2026-07-26）**：CIFAR-10 test 10,000 张、分母 `32×32×3`。Pillow 12.2.0 / zlib 1.3 下 PNG `optimize=True` 为 `5.866397656 bpd`；Pillow 12.2.0 / libwebp 1.6.0 下 WebP `lossless=True` 为 `4.606462500 bpd`。两者解码后均为 0 张像素不一致；完整协议与数据哈希见 [`demo/data/traditional_codecs.json`](demo/data/traditional_codecs.json)。

`82.95M` 是一个 checkpoint 内去除 tied embedding/head 重复引用后的可训练参数量。best/SWA/EMA 是同一架构、同一训练轨迹导出的三组权重，不应表述成一个“248.85M 参数模型”；但复现 `2.8296` 时仍需加载三个完整 checkpoint，并执行 3 members × 2 flips = 6 次 forward。该结果可以引用，但必须连同 probability-mixture ensemble + hflip TTA 协议一起引用。它不是单模型结果，也不是当前 MDLC codec 的实际文件码率。CIFAR-10 正式单模型 teacher-forced 结果现为 `2.8328 bpd`，完整 manifest 见 [`results/formal/cifar10/teacher_forced.json`](results/formal/cifar10/teacher_forced.json)；ImageNet64 正式结果见下表。

### ImageNet 64×64 对比

| 方法 | 类别 | Params | ImageNet64 bits/dim ↓ | 域 | 来源 |
|------|------|--------|-----------------------|----|------|
| PixelCNN | Autoregressive | — | 3.57 | RGB-bit-exact | van den Oord et al., 2016 |
| SPN (Subscale Pixel Network) | Autoregressive | — | 3.52 | RGB-bit-exact | Menick & Kalchbrenner, ICLR 2019 |
| Sparse Transformer | Autoregressive | 152M | 3.44 | RGB-bit-exact | Child et al., 2019 (strided sparse attention) |
| CC-iGPT (Ours, historical experiment; diagnostic protocol) | Autoregressive ensemble | 82.95M/member (K=3) | 3.4800 | RGB-bit-exact | best/SWA/EMA + hflip TTA；6 forwards/image；旧 batch-level std 作废 |
| **CC-iGPT (Ours, formal protocol)** | Autoregressive | 82.95M | **3.4812** | RGB-bit-exact | 单 `best.pth`、no-TTA、49,999 张；schema v5 manifest；teacher-forced ideal-model NLL |
| PNG | Classical codec | — | 5.7063 | RGB-bit-exact | 本项目 ImageNet64 val 49,999 张实测；完整文件 bpd；Pillow PNG `optimize=True` |
| WebP (lossless) | Classical codec | — | 4.6365 | RGB-bit-exact | 本项目 ImageNet64 val 49,999 张实测；完整文件 bpd；Pillow WebP `lossless=True` |

历史 `3.4800` 是该 ensemble+TTA 配置的诊断 NLL，不等于单模型结果或实际 bitstream bpd。正式单模型 teacher-forced 结果现为 `3.4812 bpd`，完整 manifest 见 [`results/formal/imagenet64/teacher_forced.json`](results/formal/imagenet64/teacher_forced.json)。该 manifest 的 `sequential_roundtrip.status` 为 `not_run`，因此不能把 `3.4812` 表述为已实测 arithmetic payload/file bpd。

> **ImageNet64 传统 codec baseline（2026-08-06）**：AutoDL 上验证集完整 49,999 张的 PNG `optimize=True` 为 **5.7063 bpd**，WebP `lossless=True` 为 **4.6365 bpd**（Pillow 10.3.0 / zlib 1.2.13 / libwebp 1.3.2）。指标分母为 `64×64×3`，包含 PNG/WebP 文件头；完整 manifest 见 [`results/formal/imagenet64/traditional_codecs_val_full.json`](results/formal/imagenet64/traditional_codecs_val_full.json)。此前 2,000 张前缀诊断值 PNG `5.718` / WebP `4.640` 仍保留作抽样记录。传统 codec 的 complete-file bpd 与 CC-iGPT 的 teacher-forced ideal-model NLL 分属不同码率口径，不能将 `3.4812` 表述为已实测 arithmetic payload/file bpd。

### 创新点定位

1. **方法 — 单标量参数的双尺度条件注入**：coarse iGPT 量化 token 经 bit-exact 反量化/上采样/重 tokenize 后，复用 `fine.token_embed` 得到 `coarse_ctx`，再以可学习标量 α 做 additive 注入。整个 ctx 通路只引入 1 个标量参数；encoder/decoder 共用同一函数，bitstream 真实可解码。历史 checkpoint 使用双流等权目标 `loss_coarse + loss_fine`；它不是严格码长加权，本轮不改变该目标。
2. **工程 — 6 个训练用手写 Triton primitive + pipeline + 反面案例**：RoPE→FlashAttention 是两个 kernel 的组合 pipeline，不另计融合 primitive；`fused_linear_ce` 在 V=256 下保留作负收益案例。性能结论待用修正后的 benchmark harness 重跑。
3. **分析 — RGB-bit-exact 评估与真实 codec 对账**：历史 ensemble/TTA 和 probe 曲线作为探索结果保留；正式产物要求单模型 manifest、逐图统计、真实 payload/file bpd 与无损 roundtrip。

## 快速开始

以下命令均从仓库根目录执行。WSL 用于开发和 CPU 验证；训练、正式评测与
CUDA/Triton profiling 放在 AutoDL 执行。

### 安装

```bash
# WSL：核心依赖与测试工具
python3 -m pip install -e '.[dev]'

# AutoDL：增加 Triton 与 ImageNet64 parquet 预处理依赖
python3 -m pip install -e '.[cuda,dev,imagenet-prep]'

# 需要运行 Demo 时再安装
python3 -m pip install -e '.[demo]'
```

### 验证

```bash
# WSL：CUDA 用例会自动 skip，其余测试全部执行
python3 -m pytest -q
python3 -O scripts/verify_lossless.py --self_test
python3 -m compileall -q src scripts demo tests
git diff --check
```

AutoDL 上再执行 CUDA 用例和 profiling：

```bash
python3 -m pytest -m cuda -q
python3 scripts/profile_kernels.py --roofline
```

### 正式评测

```bash
# WSL：预览命令，不加载 checkpoint 或数据
python3 scripts/run_formal_evaluations.py cifar10 \
    --verify_cifar_images 2 --dry_run
python3 scripts/run_formal_evaluations.py imagenet64 \
    --nproc_per_node 4 --dry_run

# AutoDL：用已有 best.pth 重评 CIFAR-10；不需要重训
python3 scripts/run_formal_evaluations.py cifar10 \
    --nproc_per_node 1 --cifar_batch_size 48 --verify_cifar_images 2

# AutoDL：ImageNet64 full-set teacher-forced；单 H800 示例
python3 scripts/run_formal_evaluations.py imagenet64 \
    --nproc_per_node 1 --imagenet64_batch_size 2
```

正式结果写入 `results/formal/<dataset>/teacher_forced.json`。schema v5 把三件事分开记录：

- `rate_accounting.metric=teacher_forced_ideal_model_bpd`：全测试集理想模型 NLL；即使使用 codec 的 fp32 logits/fp64 softmax，也不声称运行了算术编码。
- `codec_protocol_support`：当前单 checkpoint、raster AR、no-TTA codec 是否实现同一概率协议。
- `sequential_roundtrip`：仅在加载同 config/checkpoint、同数据 fingerprint、同源码和同 runtime 的 `verify_lossless.py --result_json` 产物后，才标记明确子集已执行真实 arithmetic encode/decode。

manifest 的 `evaluation_execution` 另存 DDP world size、collective backend、无 padding stride 分片、每 rank batch size 与 DataLoader worker 数；多卡启动信息不会因 worker 进程只看到 `evaluate.py` 参数而丢失。

ImageNet64 的逐 token roundtrip 在无 KV-cache 下非常昂贵，可先只生成 full-set teacher-forced manifest；没有实际运行就保持 `sequential_roundtrip.status=not_run`，不能补写或推断该证明。生成产物说明见 [`results/formal/README.md`](results/formal/README.md)。

历史 `2.8296` 诊断协议只在需要复核时运行：

```bash
CIFAR_RUN=experiments/ccigpt_cifar10_s_rgb_ronly_v2
python3 scripts/evaluate.py \
    --config configs/ccigpt_cifar10_s_rgb_ronly_v2.yaml \
    --ensemble "$CIFAR_RUN/checkpoints/best.pth,$CIFAR_RUN/checkpoints/swa.pth,$CIFAR_RUN/checkpoints/ema.pth" \
    --tta_hflip --per_image_stats
```

### Linear Probe

```bash
# CIFAR-10 native probe
python3 scripts/linear_probe.py \
    --config configs/ccigpt_cifar10_s_rgb_ronly_v2.yaml \
    --checkpoint experiments/ccigpt_cifar10_s_rgb_ronly_v2/checkpoints/best.pth \
    --layers all

# IN64 -> CIFAR-10 transfer probe（32->64 resize；旧 73.19% 需按新协议重跑）
python3 scripts/linear_probe.py \
    --config configs/ccigpt_imagenet64_v1.yaml \
    --checkpoint experiments/ccigpt_imagenet64_v1/checkpoints/best.pth \
    --probe_dataset cifar10 --probe_data_root datasets/ --layers all
```

> 为什么 linear probe 能从纯压缩模型里线性读出语义？理论见 [theory.md](theory.md)（MDL → LRH → MDL probing 三段论 + 阅读清单 + E1–E6 实验菜单）。

### 补全与无损验证

```bash
# 图像补全
python3 scripts/complete_image.py --config configs/ccigpt_cifar10_s_rgb_ronly_v2.yaml \
    --checkpoint experiments/ccigpt_cifar10_s_rgb_ronly_v2/checkpoints/best.pth \
    --num_images 4 --keep_frac 0.5 --temperature 1.0 --top_k 100 \
    --out experiments/completion_grid.png

# 单独调试 codec：执行真实 roundtrip，并检查生成的 MDLC v2 容器
python3 scripts/verify_lossless.py --config configs/ccigpt_cifar10_s_rgb_ronly_v2.yaml \
    --checkpoint experiments/ccigpt_cifar10_s_rgb_ronly_v2/checkpoints/best.pth \
    --num_images 1 --dump_dir results/formal/cifar10/bitstreams \
    --result_json results/formal/cifar10/sequential_roundtrip.json
python3 scripts/verify_lossless.py \
    --inspect results/formal/cifar10/bitstreams/img0.bin
```

MDLC v2 使用 32B fixed header、canonical JSON metadata、coarse/fine payload
和覆盖整个容器内容的 SHA-256。当前 codec identity-v2 绑定 checkpoint、model config、
CDF、AR schedule、实现源码 fingerprint，以及 Python/PyTorch/Triton/CUDA driver、GPU
capability、TF32/SDP/determinism 等数值 runtime，并保存预处理后 RGB 的 SHA-256。
`--inspect` 不需要模型；真正解码必须提供 identity 完全匹配的外部模型。旧 MDLC v1
以及带 legacy identity-v1 的早期 MDLC v2 仅保留只读检查路径；新解码不会把它们误当成
当前 runtime 已验证的可移植 bitstream。

### ImageNet64

以下两条预处理路径二选一，输出都会包含实际 shape、文件 hash 与工具版本的
`dataset_manifest.json`：

```bash
# 已有 64×64 PNG
python3 scripts/prepare_imagenet64_png.py \
    --train_dir /root/autodl-tmp/imagenet64/train_64x64 \
    --val_dir /root/autodl-tmp/imagenet64/valid_64x64 \
    --out_dir /root/autodl-tmp/imagenet64_png --workers 16

# 原始 parquet shard；默认保留已处理文件，便于审计和恢复
python3 scripts/prepare_imagenet64_streaming.py \
    --raw_dir datasets/imagenet64_hf/raw \
    --out_dir /root/autodl-tmp/imagenet64_png

# 传统 codec baseline 诊断抽样（PNG/WebP lossless bpd，前 2000 张）
OMP_NUM_THREADS=1 python3 scripts/traditional_codec_bpd.py \
    /root/autodl-tmp/imagenet64_png/val.npy --limit 2000 \
    --result_json results/formal/imagenet64/traditional_codecs_val_prefix2000.json

# 正式 ImageNet64 传统 codec 基线（完整 49,999 张验证集，运行较慢）
OMP_NUM_THREADS=1 python3 scripts/traditional_codec_bpd.py \
    /root/autodl-tmp/imagenet64_png/val.npy --limit 0 \
    --result_json results/formal/imagenet64/traditional_codecs_val_full.json
```

磁盘确实受限且已有其他原始数据副本时，可显式增加 `--delete_processed`。脚本只会在
输出 fsync、回读验证和 state 提交完成后删除对应 shard，但该选项仍不可恢复。

### 训练新实验

先复制一个配置、修改 `exp_name`，再设置下面两个变量。不要用历史实验的 `exp_name`
直接启动新训练，以免把产物写进已有目录。

```bash
cp -i configs/ccigpt_imagenet64_v1.yaml configs/my_experiment.yaml
```

编辑 `configs/my_experiment.yaml`，为 `exp_name` 和实验设置赋新值。确认后再启动：

```bash
CONFIG_PATH=configs/my_experiment.yaml
NUM_GPUS=4
torchrun --standalone --nproc_per_node="$NUM_GPUS" scripts/train.py \
    --config "$CONFIG_PATH" --export_csv
```

`--resume` 用于同一 config、seed、world size 下的完整 training-state 精确续训；
`--init_from` 用于裸权重、旧 checkpoint 或修改过结构的初始化，两者互斥。`--resume`
会恢复模型、optimizer、scheduler、scaler、SWA/EMA、每个 rank 的 RNG、epoch 与 best
metric，并验证源码、runtime 和 train/validation 数据指纹。DDP 验证采用无 padding
分片，每个验证样本恰好统计一次。NCCL 环境变量应按 AutoDL 实例拓扑设置，不在通用
命令中硬编码。

### Demo 前端

7 个面板：上传 bpd 热力图、baseline 对比、Linear Probe、Kernel 性能、coarse/fine 双尺度、图像补全、交互式无损 codec。补全面板调用 `/api/complete`；codec 面板调用 `/api/{encode,inspect,decode}`。
模型请求固定串行，后端只缓存当前数据集的一份模型；CIFAR/ImageNet64 切换时先释放旧模型再加载新模型，避免两份权重同时常驻 GPU。

```bash
# 本地 / WSL
uvicorn demo.server:app --reload --port 8000

# AutoDL / 公网映射：不要启用 --reload；API key 从服务环境注入
export MDLIC_DEMO_API_KEY="$(openssl rand -hex 32)"
export MDLIC_DEMO_RATE_LIMIT=6
uvicorn demo.server:app --host 0.0.0.0 --port 6006
```

公网部署还应在 TLS 反向代理层配置认证、请求体上限和共享限流。应用内限流仅在
单进程内生效；只有确认反向代理会覆盖而不是追加客户端提供的
`X-Forwarded-For` 时，才设置 `MDLIC_DEMO_TRUST_PROXY=1`。CORS 白名单不是访问控制。

## WSL 与 AutoDL 同步

本地 WSL 只做开发和 CPU 验证，训练一律在 AutoDL GPU 实例上执行。`git pull` 不会影响已经启动的 Python 训练进程；它只影响后续新启动的命令。

```bash
# WSL：完成提交后推送
git push origin dev

# AutoDL：同步代码
cd /root/autodl-tmp/mdl-deep-image-compression
git pull --ff-only origin dev
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
                    | 序列布局: R-only 配置 |
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
              iGPT:    bpd = [8 + CE × (T-1) / ln(2)] / T
              CC-iGPT: bpd_total = [8 + CE_c·(N_c-1)/ln(2)
                                   + 8 + CE_f·(N_f-1)/ln(2)] / N_f
                       (每个独立 AR stream 的首 token 使用均匀 256-way 先验)
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

Coarse-Conditioned iGPT —— 双尺度条件式自回归，复用 iGPT 训练栈与 Triton kernels。历史训练采用 coarse/fine 两流平均 CE 等权；正式 bpd 则按实际 token 数和首 token 先验计算，两者不要混为同一口径。

| 组件 | 说明 |
|------|------|
| DOWN | `F.adaptive_avg_pool2d(x, 8)` 在 float 域下采样到 8×8 |
| Coarse iGPT | 浅层（d_model=256, N=6），独立 NTP 训练，CE 进 bitstream。R-only 配置仅压 R 通道 8×8（64 token） |
| **Bit-exact ctx 路径** | encoder/decoder 必须看到**同一个** `coarse_ctx`，否则 fine 端算术编码不可解。统一管线：coarse 量化 token → `/255` 反量化为 RGB float → bilinear UP 到 32×32 → 与 fine encoder 同规则 re-tokenize → `fine.token_embed` |
| Ctx 注入 | AR shift `coarse_ctx[:, 1:]`（ctx[i] 对应 fine 被预测位置 i，PixelCNN++ conditional 标准语义）→ `α · coarse_ctx`（additive，仅引入 1 个标量参数 α） |
| 可学习 α | `nn.Parameter(torch.ones(1))`，初始 1.0；模型自适应注入强度 |
| 联合 bits/dim | `[8 + CE_c·(N_c-1)/ln(2) + 8 + CE_f·(N_f-1)/ln(2)] / N_f`；两个 stream 的首 token 均按 8 bit 计入 |
| 训练 | 历史 checkpoint 使用 `loss = loss_coarse + loss_fine`（两流平均 CE 等权）；不是严格 rate weighting，本轮保持不变 |

设计参考: Burt & Adelson "Laplacian Pyramid" (1983)、van den Oord "Conditional PixelCNN" (NeurIPS 2016, additive 条件)、Tian "VAR" (NeurIPS 2024)。

### CC-MDLM 协议地基 (`models/masked_igpt.py`, `models/cc_mdlm.py`)

masked 研究线与原 AR forward 分离：fine 输入是完整未移位序列，显式 input-only `[MASK]` embedding，输出仍为 256-way tied categorical head，attention 使用 `is_causal=False`。`MaskSchedule` 把精确 group 顺序序列化并哈希；grouped codec 每组只 forward 一次，先冻结整组 CDF，再编码/解码该组全部 token。测试覆盖 `K=T` 的 100 个合成 roundtrip，以及 tiny 模型的 schedule NLL、量化 CDF NLL、payload、packed payload 和 MDLC file bpd 对账。

这只是 E1/E2 的 CPU 协议地基，不是训练结果。尚未接入 CIFAR 训练配置，也不报告 masked CE/NELBO 为 bpd；旧 AR checkpoint 与新类参数结构不同，未来只能用 `--init_from` 做 shape-compatible 初始化，不能 `--resume`。

### 共享层 (`models/layers.py`)

GPTBlock, MultiHeadAttentionBlock (RoPE + QK-Norm + Flash Attention + attn_mask), RMSNorm, SwiGLU FFN。RoPE 的 `cos/sin` 按 `(seq_len, device)` 缓存。

### 手写 Triton Kernels (`ops/`, ~2,400 行)

训练栈包含 6 类手写 primitive。`fused_attn_rope` 是组合 RoPE 与 FlashAttention 的 Python pipeline，不作为额外融合 kernel 计数；`fused_linear_ce` 是独立负收益案例。

| Kernel | 行数 | 说明 |
|--------|------|------|
| Flash Attention v2 | ~960 | causal early termination, online softmax |
| Fused CE+z-loss | ~276 | online softmax, 避免 O(V) 中间矩阵 |
| Fused RMSNorm | ~236 | fwd+bwd |
| Fused Add+RMSNorm | ~217 | post-norm 残差+归一化合并 |
| Fused SwiGLU | ~204 | activation recomputation |
| Fused RoPE | ~123 | 就地旋转 Q/K |
| RoPE→FlashAttention pipeline | ~62 | 薄包装层：依次组合 `fused_rope` 与 `flash_attn`，不是单一融合 op |
| ~~Fused Linear+CE~~ | ~393 | **反面案例** — V=256 下三重循环失去 cuBLAS GEMM 利用率 |

### Linear Probe (`scripts/linear_probe.py`)

冻结预训练模型，通过 `IGPT.encode(x, max_layer)` 取各层 hidden state，全局平均池化后训练线性分类器。层选择只使用训练集划出的 validation；选定层在完整训练集重训后仅评估一次 test。脚本把 classifier-seed 标准差作为优化波动的描述性统计，并另对最终 test 样本做 percentile bootstrap；两者都不替代多预训练 seed。

## 项目结构

```
src/mdlic/
├── models/    igpt.py, cc_igpt.py, masked_igpt.py, cc_mdlm.py, layers.py
├── codec/     arithmetic.py, sequential.py, container.py, verification.py, grouped.py
├── completion.py, model_factory.py, eval_metrics.py, masking_scheduler.py, rate.py
├── provenance.py, request_limits.py
├── ops/       6 类训练 primitive + 1 组合 pipeline + 1 反面案例
├── data/      evaluation.py, imagenet64_npy.py (mmap-backed Dataset), manifest.py
└── utils/     seed, bpd, clean_state_dict
scripts/       train.py, evaluate.py, run_formal_evaluations.py, linear_probe.py,
               complete_image.py, verify_lossless.py, profile_kernels.py,
               traditional_codec_bpd.py, pixelsnail_paramcount.py, render_report_figures.py,
               prepare_imagenet64_png.py, prepare_imagenet64_streaming.py
configs/       igpt_cifar10_s_rgb,
               ccigpt_cifar10_s_rgb_ronly      (R-only v1 历史实验/diagnostic protocol, 100ep, 已被 v2 替代),
               ccigpt_cifar10_s_rgb_ronly_v2   (深窄 N=32/d=448 + 200ep；formal 2.8328 bpd，2-image roundtrip verified),
               ccigpt_imagenet64_v1            (ImageNet64 12ep；formal 3.4812 bpd，roundtrip pending)
tests/         单元测试（含 test_ccigpt_smoke / test_arithmetic_codec）
demo/
├── server.py          FastAPI 后端 (predict / encode / inspect / decode / complete / metrics / probe / kernels / scales)
│                      ckpt 加载优先级: v2 → ronly softmax
├── static/            HTML + JS (Chart.js) + CSS 前端，7 个面板
└── data/              预计算 JSON 数据（metrics / probe / kernels / scales）
```

**参考文献**

**模型**: iGPT (Chen 2020), RoPE (Su 2021), RMSNorm (Zhang 2019), SwiGLU (Shazeer 2020), OLMo 2 (2025), QK-Norm (Dehghani 2023), Weight Tying (Press 2017), Linear Probe (Alain & Bengio 2017)

**像素自回归**: PixelCNN++ (Salimans 2017), PixelSNAIL (Chen 2018), PixelCNN (van den Oord 2016), Sparse Transformer (Child 2019)

**多尺度**: VAR (Tian 2024), VQ-VAE (van den Oord 2017), Multi-Scale PixelCNN (Reed 2017), Subscale Pixel Networks (Menick 2018)

**Triton**: FlashAttention v1/v2 (Dao 2022/2023), Online Softmax (Milakov 2018), Liger Kernel (Hsu 2024)

**训练**: SWA (Izmailov 2018), Cosine + Warmup (OLMo 2 2025)

**理论**: Shannon (1948), MDL (Rissanen 1978), Language Modeling Is Compression (Delétang 2024)

**表征 / 可解释性**: 详见 [theory.md](theory.md)（MDL + LRH + MDL probing 三段论 + 阅读清单）

> 完整阅读路径（7 阶，从信息论地基到「超越压缩」）见版本化路线文档 [future.md](future.md)。本地执行清单记录在 ignored 的 `plan.md`。
