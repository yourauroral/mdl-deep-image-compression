# 项目总览

> 毕业设计：基于自回归 Transformer 的图像压缩。所有实验在 **RGB-bit-exact 域**（直接在 RGB uint8 上建模，与 PixelCNN++ / Sparse Transformer 等基线同域可比）。所有优化手写实现。

## 0. 2026-07-27 执行口径（优先级高于后文历史备忘）

1. **当前主线不换模型、不改训练损失**：CC-iGPT 继续使用 `CE_coarse + CE_fine`。这是一种双流平衡训练目标，不等同于严格按真实码长加权；本轮代码修正不要求重训，也不使现有 checkpoint 失效。
2. **正式结果已完成**：CIFAR-10 `2.8296` 与 ImageNet64 `3.4800` 都是有效的历史实验结果，但其协议是三成员 ensemble + hflip TTA，在新 evaluator 中归入 diagnostic，而不是正式单模型主表。每个 CC-iGPT v2 checkpoint 是 `82,949,441` 参数；该协议加载 best/SWA/EMA 三组权重（K=3），而不是一个 248.85M 参数架构，hflip 后每张图共 6 次 forward。现有 `best.pth` 已完成两个数据集的 single-checkpoint/no-TTA teacher-forced 重评：CIFAR-10 **2.8328 bpd**（10,000 张，逐图 std 0.6719，95% bootstrap CI `[2.8201, 2.8456]`），ImageNet64 **3.4812 bpd**（49,999 张，逐图 std 0.9040，95% bootstrap CI `[3.4734, 3.4898]`）。CIFAR-10 另有 2 张 `verified_on_subset` sequential roundtrip，均 pixel-exact；两者都不需要重训。
3. **Phase E 是并行研究分支，不替换 AR 无损主线**：目标是研究 Masked Diffusion Language Model (MDLM) / grouped ARDM 的 `实际码率 vs forward calls`，不是先承诺更低 bpd 或实时加速。Phase E 需要新的训练；现有 AR checkpoint 最多用于初始化兼容权重，不能直接当作 masked model 结果。
4. **codec-first**：在任何大规模训练前，先用 tiny CIFAR/合成图完成真实 `encode -> file -> decode` 闭环。没有固定 schedule、可解码概率因式分解、量化 CDF 协议和逐像素一致测试，就不把 masked CE/NELBO 称为压缩率。
5. **六层指标分开报告**：训练 masked CE/NELBO、固定 schedule 的理想模型 NLL、量化 CDF NLL、算术 payload bpd、字节 padding 后的 packed payload bpd、完整文件 bpd。另报 fine/total forward calls、wall-clock、峰值显存和硬件环境。
6. **执行环境分工**：WSL 负责代码、CPU 测试和 `run_formal_evaluations.py --dry_run`；两个数据集的正式评测及 codec roundtrip 在持有数据与 checkpoint 的 AutoDL 上运行。CIFAR-10/ImageNet64 teacher-forced manifest 已完成；无泄漏 probe/matched controls 和 CUDA backend matrix/profiling 暂缓，未执行前保持 pending，不阻塞本轮 evaluator/manifest 修复，也不据此推进 CC-MDLM 正式训练。

### Phase E 的最小概率协议

令确定性、数据无关的 schedule 把 fine token 划分为有序且互斥的组 `G_1,...,G_K`。第一版明确采用组内条件独立近似：

```text
q_theta(x_fine | x_coarse)
  = product_k product_{i in G_k}
      p_theta(x_i | x_coarse, x_{G_1}, ..., x_{G_{k-1}}, MASK_elsewhere)
```

这是一个可归一化、可直接逐组算术编码的概率模型。encoder 和 decoder 在组开始时拥有相同的已揭示 token；一次 forward 产生该组所有边缘分布，组内 token 按固定索引依次送入 coder，但后解出的组内 token 不能反过来改变本组分布。schedule 的名称、版本、`K`、分组哈希、模型哈希和 CDF 精度必须进入容器或结果 manifest。

**重要区分**：通用 MDLM 的随机时间训练 NELBO 是 likelihood bound/训练指标，不会自动等于上述固定 schedule 的实际码长。只有把 latent path/schedule 与编码算法完整定义后，才能讨论对应 code length；若想以 NELBO 码长为目标，可能还需显式编码扩散轨迹或 bits-back 类协议。第一版因此先实现 grouped likelihood codec，再把 MDLM-NELBO 作为可对照的训练目标，而不是把二者写成等号。

### Go/No-Go 门槛

| Gate | 必须通过的证据 | 未通过时 |
|---|---|---|
| E0 AR 基线冻结 | 新 evaluator 的单 checkpoint/no-TTA manifest；现有 MDLC roundtrip | 不开始 masked 分支对比 |
| E1 协议原型 | tiny 模型、至少 100 张/随机样本 RGB token 逐值 roundtrip；坏 header/schedule/hash 会拒绝 | 不训练正式模型 |
| E2 因式分解验证 | teacher-forced schedule NLL、量化 CDF NLL、payload/file bpd 可逐 token 对账 | 不报告 NELBO 为 bpd |
| E3 CIFAR toy | `K in {T, 128, 64, 32, 16}` 的实测 rate/calls/time；至少 3 seeds | 不扩到 ImageNet64 |
| E4 结构研究 | raster/checkerboard/pyramid/channel-refine 在相同参数、训练预算和 codec 下比较 | 不做稀疏 attention/蒸馏 |

`T -> K` 只表示 fine 分支 forward-call 数减少。CIFAR fine 从 3071 calls 到 `K=32/64` 是约 96.0x/48.0x 的 **fine-call reduction**；保留 coarse AR 的 63 次条件 forward 后，总调用数从 3134 变为 95/127，即约 33.0x/24.7x。wall-clock 还受双向整序列 forward、算术 coder、数据搬运和 kernel 效率影响，必须实测，不能写成 50–100 倍速度提升。

---

## 1. 当前状态（截至 2026-08-04；Phase D 完成 2026-05-27，下游任务 + 前端无损面板 2026-05-29，IN64 训练完成 2026-06-07，CIFAR/IN64 正式 teacher-forced 重评完成 2026-08-04）

- **CC-iGPT v2 历史实验结果（Phase D 完成，diagnostic protocol）**：ensemble (best+SWA+EMA) + TTA hflip = **2.8296 bpd**；旧 `±0.0854` 口径不作为新逐图 std。CIFAR-10 正式单模型/no-TTA 数字现为 **2.8328 bpd**（详 §0）
- **CIFAR-10 正式 teacher-forced 结果（2026-08-04）**：single `best.pth` + no TTA = **2.8328 bpd**；10,000 张，逐图 std 0.6719，95% bootstrap CI `[2.8201, 2.8456]`。同一 codec identity 的 2 张 sequential roundtrip 已 `verified_on_subset`，均 pixel-exact；子集 mean payload `2.7074 bpd`、packed payload `2.7109 bpd`、完整 file `8.0703 bpd`，不外推到全量测试集。
- **ImageNet64 正式 teacher-forced 结果（2026-08-03）**：single `best.pth` + no TTA = **3.4812 bpd**；49,999 张，逐图 std 0.9040，95% bootstrap CI `[3.4734, 3.4898]`，manifest 见 `results/formal/imagenet64/teacher_forced.json`。sequential roundtrip 尚未执行。
- **ImageNet64 传统 codec 诊断（2026-08-05）**：验证集前 2,000 张 PNG = **5.718 bpd**、WebP lossless = **4.640 bpd**；这是固定前缀抽样，完整 49,999 张基线仍待 `--limit 0` 重跑后再进入正式比较。
- v1 历史 TTA 诊断：CC-iGPT R-only 100ep best+TTA 2.9035；旧 std 口径仅留作日志，不进入新协议比较
- Linear probe 历史曲线 best L19 = **79.33%**；旧脚本曾用 test 选层，需按“训练集分层 validation 选层、完整 train 重训、selected layer test、多 seeds”协议重跑后再作为正式结果
- 总参数量 **82.95M**（fine 78.14M + coarse 4.81M）
- **下游任务（§6）**：保留 linear probe / 图像补全 / 真实可解性 codec 三条主线；前端保留补全与无损 codec 交互面板。IN64 收敛后优先跑 IN64→CIFAR transfer probe。

---

## 2. 项目阶段

| Phase | 状态 | 关键产物 |
|---|---|---|
| A — iGPT raster-scan AR + 6 个训练 primitive + 1 组合 pipeline + 1 反面案例 | ✅ | `src/mdlic/ops/` |
| B — CC-iGPT v1 双尺度条件式 AR (100ep, R-only) | ✅ | best+TTA 2.9035 |
| C — Demo 前端可视化 (FastAPI + Chart.js, 7 面板，含交互式无损 codec 图像⇄.bin 真实可解性验证) | ✅ | `demo/` |
| D — 深窄 (N=32/d=448, 200ep) + 历史 ensemble/TTA 实验 | ✅ | diagnostic protocol 下 2.8296；CIFAR-10 formal 2.8328；ImageNet64 formal 3.4812 |
| E — Masked diffusion / grouped ARDM codec | CPU 协议地基完成，训练 gated | 先复核 tiny roundtrip，再训练 CIFAR toy |

详细组件、Triton kernel 列表、架构图见 README。

---

## 3. CC-iGPT 设计

### 核心思想

iGPT 光栅扫描中早期 token 缺全局上下文 → BPP 高。CC-iGPT 用一层粗尺度全局上下文作 additive embedding 注入 fine 模型，关键约束是 **encoder/decoder 必须见到 bit-exact 一致的 coarse_ctx**。

### 数据流

```
        原图 x  [B, 3, 32, 32]   (clamp 0-1, fp32)
       ┌────┴────────────────────────────┐
       │                                 │
       │                          avg_pool 8×8
       │                                 │
       │                       x_c_full [B, 3, 8, 8]
       │                          │
       │                          │ R-only 配置：取 [:, :1]
       │                          ▼
       │                       x_c_float [B, 1, 8, 8]
       │                          ┌──────┴──────┐
       │                          │             │
       │                  self.coarse(...)    self.coarse._tokenize(...)
       │                  (transformer       (round 量化进 bitstream)
       │                   前向)
       │                          │             │
       │                          ▼             ▼
       │                   out_c {ce,...}   coarse_tokens
       │                                    [B, 64] int 0-255
       │                                         │
       │                                         ▼
       │                                _compute_coarse_ctx
       │                                         │
       │                                         ▼
       │                                coarse_ctx [B, 3071, d_model]
       │                                         │
       │                                  × ctx_alpha (可学习标量)
       │                                         │
       └─────────────────────────►◄──────────────┘
                                  ▼
                          self.fine(x, coarse_ctx=α·coarse_ctx)
                                  │
                                  ▼
                          out_f {ce, loss, logits}

loss      = out_c.loss + out_f.loss          (端到端联合, 无加权，保持不变)
ideal_bits = (8 + CE_c · (N_c-1) / ln2) + (8 + CE_f · (N_f-1) / ln2)
bpd_total  = ideal_bits / N_f
```

要点：fine 看到的是**原图 32×32**（不是重建图）；coarse 给的只是 `α · coarse_ctx` 这一份"低频先验"。

### `_compute_coarse_ctx` 管线（带 shape 注释）

```
coarse_tokens  [B, 64]                       ← R-only int 0-255
        │ view(B, 1, 8, 8) / 255.0           ← 反量化到 [0,1]
        ▼
   [B, 1, 8, 8]   R float
        │ F.interpolate bilinear UP
        ▼
   [B, 1, 32, 32]
        │ expand 到 (B, 3, 32, 32)            ← 灰度先验复制三通道
        ▼
   [B, 3, 32, 32]
        │ × 255 → round → long                ← 与 fine encoder 同量化规则
        ▼
   [B, 3, 32, 32]   RGB int 0-255
        │ permute(0,2,3,1).reshape(B, -1)     ← pixel-first
        ▼
   [B, 3072]
        │ self.fine.token_embed               ← 复用 fine 的 token embedding, 无新参数
        ▼
   [B, 3072, d_model]
        │ [:, 1:]                              ← AR shift, ctx[i] 与 fine 被预测位置 i 同位对齐
        ▼
   coarse_ctx  [B, 3071, d_model]
```

约束：encoder 和 decoder 调用**同一个** `_compute_coarse_ctx`，输入的 `coarse_tokens` 也是同一份（decoder 从 bitstream 段 A 解出来），所以两边重建的 ctx **bit-exact 相等**，fine 端的算术编码才可解。

### 设计要点

| 决策 | 选择 | 原因 |
|------|------|------|
| 条件 vs 残差 | **条件式注入**（fine 仍预测 raw token） | 复用现有 token_embed/CE kernel |
| 注入方式 | additive embedding（复用 embedding，仅新增 α 标量） | 简单、可关闭做消融对照 |
| α scaling | `nn.Parameter(torch.ones(1))`，端到端学 | 避免 ctx 过强压制 fine token embed |
| coarse 编码 | 独立小 iGPT (256d×6L)，CE 进 bitstream | 解码端可还原（side information；R-only 占联合 token 数约 2%，历史 bpd 占比约 4.6%） |
| **ctx 输入** | **量化 token，非 float** | encoder/decoder bit-exact 一致；bitstream 真实可解 |
| **反量化路径** | `view(B,C,S,S) → /255 → UP → re-tokenize` | 与 IGPT._tokenize 互逆，与 fine encoder 同规则 |
| AR shift | `coarse_ctx[:, 1:]` | PixelCNN++ conditional / VAR multi-scale 标准语义 |
| 联合 loss | `loss = loss_coarse + loss_fine`（无加权） | 平衡双流优化；不是严格 rate weighting，本轮不改以保持 checkpoint 连续性 |

### 一致性回归测试（tests/test_ccigpt_smoke.py）

- `test_encoder_decoder_ctx_consistency`：encoder 与 decoder 调用 `_compute_coarse_ctx` 得到的 ctx max diff < 1e-6
- `test_disable_ctx_equivalent_to_vanilla_igpt`：`coarse_ctx=None` 时 fine ≡ 单 iGPT
- `test_coarse_ctx_ar_shift_alignment`：ctx 长度严格等于 fine.seq_len-1（防止丢 `[:, 1:]`）

### 信息论基础

- H(X_fine | X_coarse) ≤ H(X_fine) — 条件熵不增 (Cover & Thomas 2006)
- 自然图像 1/f² 功率谱 → 信息集中在粗尺度 (Field 1987)
- coarse 是 side information（必须传输，进 bitstream），与 VAR next-scale 同源

---

## 4. Phase D 设计速查（深窄 + ensemble）

参考 Sparse Transformer (Child 2019, N=128/d=256, 59M — CIFAR-10 配置) 的"深窄"哲学，一次训练同时吃两份收益：架构深窄 + epoch 翻倍。

| 字段 | v1 | **v2** | 变化原因 |
|---|---|---|---|
| `N` (fine) | 24 | **32** | 深 33%（深窄哲学，NLL 增益 -20~-50 mbpd）|
| `d_model` (fine) | 512 | **448** | 窄 12.5%（总参数控在 ~78M）|
| `d_ff` (fine) | 1376 | **1216** | 同步等比 ((8/3)·d_model) |
| `h` (fine) | 8 | **7** | 维持 d_k=64 |
| `epochs` | 100 | **200** | token/param 1.3→2.6，吃欠拟合 |
| `min_lr_ratio` | 0.0 | **0.05** | cosine 末段 LR ≥ 5%·peak，避免 SWA "假平均" |
| `swa.start_epoch` | 90 | **170** | last 31 ckpts 平均（ep170..200, interval=1）|
| `ema.decay` | 0.9995 | **0.9998** | 200ep 长训需更长半衰期 |
| `warmup_epochs` | 3 | **5** | 200ep + 深窄初期更平稳 |
| `batch_size × accum` | 64×2 | **48×3** | 32 层显存压力 +33%，effective batch=144 |

**Lever 2 — 多 checkpoint 概率 mixture**：`scripts/evaluate.py --ensemble best,swa,ema` 先对各成员 logits 做 `log_softmax`，再用 `logsumexp - log(K)` 得到逐 token 的算术平均概率。三档是同一训练轨迹的三种平滑（best=val 最优瞬点 / EMA=指数 / SWA=均匀），各自捕获不同 loss-landscape 邻域。

**历史实验日志**：in-training best ep196 = 2.8312；ensemble + TTA hflip = 2.8296。数字可作为对应旧协议的实验观测引用，但不能替代 single-checkpoint/no-TTA 正式主表；正式 CIFAR-10 数字现为 2.8328 bpd，完整统计和 roundtrip 证据以新 evaluator manifest 为准。

---

## 5. 退役方案（保留作论文反面案例）

### DMoL 输出头（2026-05-25 彻底退场）

2026-04 至 2026-05 共 6 次实现尝试，每次失败根因都被归纳到具体数值边界（init / log_scale bias / optimizer / fp32 / 通道耦合 / 域坐标系）。但第 6 次（2026-05-25 修复 [-1,1] 域对齐）Stage 2 (80M 主网) 训练中 CE_f 仍持续 flat ≥ 10 nat / sub-pixel，无法收敛到 softmax baseline。

**最终诊断**：DMoL 的连续混合 + 离散化设计与 sub-pixel pixel-first AR 架构存在结构性不兼容 — sub-pixel AR 已在 token 级建模 R/G/B 通道依赖，DMoL 的混合分布引入冗余优化目标；80M 主网 hidden activation 方差让 boundary 项进入 sigmoid 饱和区。本架构下 **softmax categorical** 是唯一稳定路径。

**论文叙事**：作为反面案例写入 §3.X，强调"在 mini setup (0.7M) 上验证的数值健康指标在大模型 (80M) 上可能因 hidden activation 方差变化失效；DMoL 这类高度数值敏感的输出头必须严格对齐文献原版坐标系，且在目标参数规模下端到端验证"。

**代码处置**：DMoL 专属文件 (`src/mdlic/losses/dmol.py`, `tests/test_dmol.py`, 4 个 dmol*.yaml configs) 全部从仓库删除；模型层 DMoL 分支整段移除。

### 已删除特性

MSPA / YCbCr 色彩前端 / logit soft-capping / Gaussian label smoothing / sliding window attention / MTP head — 训练崩溃或与本架构无收益，已从代码移除，只保留在 `MEMORY.md` 红线列表。

---

## 6. 下游任务（论文 §5，MDL 主线）

**主线假设**：MDL（最小描述长度）联系压缩与表征学习。压缩结果、linear probe、条件生成和真实 codec 是四类相关证据，但不能仅凭相关性写成因果定理；正式数字均按各自无泄漏协议重跑。

| # | 任务 | 验证的 MDL 命题 | 状态 | 成本 | 抢 GPU |
|---|---|---|---|---|---|
| 1 | Linear probe | 压得越好 -> 表征越好 | ✅ 已有 (+IN64->CIFAR transfer 已实现) | `scripts/linear_probe.py` | 否 |
| 2 | 图像补全 (image completion) | AR 条件生成能力，Sparse Trans 同款定性展示 | ✅ 已实测 + 前端实时版 | `scripts/complete_image.py` / `POST /api/complete` | 轻微 |
| 3 | 真实可解性 demo | 严格无损 + bitstream 真实可解 | ✅ 已实测 2/2 bit-identical + 落盘 .bin + 前端 codec 面板 | `scripts/verify_lossless.py` | 轻微 |

### 6.1 Linear probe（已完成 + transfer probe 已实现）

旧脚本日志为 with α·coarse_ctx 的 L19 `79.33%`，但该层由 test curve 选择，只能视为探索性结果。新脚本使用 validation 选层、多 seeds，并仅对选定层做最终 test；重跑后再与 iGPT-S 比较。

**IN64->CIFAR-10 transfer（旧协议探索曲线）**：`--probe_dataset cifar10 --probe_data_root datasets/` 让 ImageNet64 预训练权重探在 CIFAR-10 上，图像 resize 32->64。旧 test curve 在 L16 为 `73.19%`，但同样存在 test 选层问题；不得据此宣称胜过 iGPT-S，也不能与 native 32x32 横比。待新脚本重跑后只报告 validation-selected layer 的最终 test。**未选 IN64 1000-way**（需 label-保留重跑 prepare + 无 iGPT baseline，2026-05-29 决策）。

### 6.2 图像补全（✅ 已实现 + AutoDL 已冒烟）

给前 keep_frac 比例的 token（raster pixel-first → 上半若干行），AR 续采样补全其余。Sparse Transformer 唯一展示的"下游"即此（无数字）。AR 模型天然支持、不重训。

**命令**：`python scripts/complete_image.py --config configs/ccigpt_cifar10_s_rgb_ronly_v2.yaml --checkpoint <best.pth> --num_images 4 --keep_frac 0.5 --temperature 1.0 --top_k 100 --out experiments/completion_grid.png`。输出网格每行 = 原图 | 已知上半(灰=待补) | 补全。**诚实声明**：CC-iGPT 的 coarse ctx 由整图缩略图算，语义是"低分缩略图 + 上半真实像素 → 补下半"，coarse 是显式 side-channel（与压缩时独立 bitstream 同源），非偷看；纯补全需让 coarse 也只看上半（未做，注释存档）。仅 CIFAR v2 实用（fine 3072 token），IN64 太慢。

### 6.3 真实可解性 demo（算术编解码 roundtrip，✅ 已实测 2026-05-30）

证明"严格无损 + bitstream 真实可解"——区别于只报 bpd、不验证可逆性的工作。把 CC-iGPT 当真实无损 codec：图像 → coarse/fine token → 逐步条件分布 → 算术编码 → 真实 bitstream → 解码 → 断言**逐像素 bit-identical**，报告实际码长(bit)/bpd 与模型 teacher-forced NLL 对照。

- **代码**：`src/mdlic/codec/arithmetic.py`（手写 32-bit WNC 算术编码器，纯 Python，`pack_bits`/`unpack_bits` 字节打包逆操作）+ `tests/test_arithmetic_codec.py` + `src/mdlic/codec/container.py` + `tests/test_mdlc_container.py` + `scripts/verify_lossless.py`（roundtrip 主脚本）
- **关键设计**：模型无 KV-cache → decode 每步跑完整 forward（前缀真实 token + 0 后缀，causal mask 保证不泄漏）；encode/decode 都走 per-step 同一函数 → logits 逐位相同 → 保证可逆；全程 fp32；token 0 用均匀先验；双尺度先解 coarse 独立 bitstream 重建 ctx 再解 fine
- **范围**：CIFAR v2 已完成当前 identity-v2/schema v5 manifest 的 2/2 GPU roundtrip；IN64 checkpoint 已存在，但 12288-token 逐步 codec 成本很高，未完成真实文件 roundtrip，不据此宣称可用速度
- **实测（2026-05-30，AutoDL，865s）**：**2/2 图 bit-identical**。img0 achieved 3.0931 bpd（NLL 3.0896，overhead 0.11%）/ img1 2.3216（NLL 2.3161，overhead 0.24%）。逐图 bpd 随图像复杂度变，关键是 **overhead vs NLL 仅 0.1–0.2%**（小正数）→ 报的 bpd = 真实可逆码长，算术编码近最优
- **bitstream 落盘（MDLC v2，2026-07-26；identity-v2 于 2026-07-27）**：新编码默认写 32B fixed header + canonical JSON metadata + coarse/fine payload + 32B SHA-256。identity 绑定 checkpoint、model config、CDF、AR schedule、实现源码 fingerprint，以及 Python/PyTorch/Triton/CUDA/设备与 backend 数值 runtime；metadata 另存预处理 RGB SHA-256，解码结束强制核对。header/metadata/payload/checksum 任一区域位翻转，或 checkpoint/config/protocol/schedule/source/runtime 不同，均会拒绝。
- **验证边界**：历史 2/2 AutoDL 真实图 roundtrip 使用 v1 容器；当前 MDLC v2 + identity-v2 已通过 CPU tiny-model、legacy inspect、协议破坏测试，并在 CIFAR-10 上完成 2/2 GPU 子集验证。正式 evaluator schema v5 只在 config/checkpoint、数据 fingerprint、源码和 runtime 全匹配时附加 `verify_lossless --result_json` 的子集证明；ImageNet64 的 GPU roundtrip 仍未执行。

### 6.4 前端下游任务面板（codec / 补全，✅ 实现 2026-05-29~31 + AutoDL 已冒烟）

把保留的下游任务搬上 demo 前端，均为实时交互；没有静态下游 JSON 回填。

| Panel | 任务 | 端点 | 类型 | 数据回填 |
|---|---|---|---|---|
| codec | 真实可解性 图像⇄.bin | `POST /api/encode` + `/api/inspect` + `/api/decode` | 实时（逐 token gold 算术编/解码，流式 NDJSON，~60–120s） | — |
| 补全 | 图像补全 AR inpainting | `POST /api/complete` | 实时（无 KV-cache，~20–40s/图） | — |

- **补全面板**：上传图 + keep%/温度滑块 + "运行补全"按钮（避免误触长采样）；复用 `scripts/complete_image._complete_one`（与 §6.2 完全同源），出 原图 / 已知上半(灰=待补) / AR 补全 三联图。诚实声明（coarse 是显式 side-channel）写进面板描述。
- **补全面板与 §6.3 取舍同 codec 面板**：补全是真实逐步采样（无作弊），但占 GPU ~数十秒；与 IN64 训练共享卡，挑空窗跑。下限 `keep_frac≥0.3` 防超长单请求占住 worker。
- **WSL 侧已静态验证**：py_compile（server.py + 相关脚本）、HTML↔JS id 1:1、app.js `node --check`。**首跑 forward 在 AutoDL**（无 fastapi + GPU + 纪律）。
- **code review 修复（2026-05-30）**：`/api/complete` 用 `torch.no_grad()` 包 `_complete_one` —— `ctx_alpha`（可训练 `Parameter`）原会在每请求建 autograd 图并占住整个请求周期，浪费显存且与 `predict` 不一致。复验 py_compile + pytest 全过。
- **code review 修复（2026-06-01）**：codec 面板 `/api/encode` 由单次 teacher-forced forward（快路径）改为逐 token gold（与 `/api/decode` 共用 `_logits_from_tokens`）—— 两路径 logits 在 GPU 上差 ~1 ULP（实测 max\|Δ\|=3.8e-5），算术编码零容忍 → 翻符号 → 解码失步成噪点；同源后 bit-exact 可解。顺带删掉 `/api/encode` 误导性的 `编码自检 ✅ bit-identical` 徽章（token 化无损是恒真的，真正证据是 decode 端指纹交叉校验）及死字段 `orig_png`/`pixel_exact`。pytest 59 过。
- **AutoDL 已冒烟**：补全面板上传任意图调 keep%/温度跑补全；codec 面板「编码→下载 .bin→解析→解码」复核盲解码 + 指纹交叉校验，均通过。

#### 历史：codec 面板从「快路径」到「逐 token gold」（2026-05-29 设计 → 2026-06-01 修正）

把 §6.3 的"真实可解"搬上 demo 前端（用户最看重的卖点），做成**交互式**：上传图 → 实时算术编码出模型绑定的 `.bin` → 下载 → 上传解析结构/identity（`/api/inspect`，无 GPU）→ 用匹配模型盲解码还原图 + RGB SHA-256 校验。

- **代码**：`demo/server.py` `/api/encode` + `/api/inspect` + `/api/decode`（复用 `src/mdlic/codec/arithmetic` 与 `scripts/verify_lossless` 的 gold 路径）+ `demo/static/{index.html,app.js,style.css}` codec 面板
- **初版（已废弃）的取舍**：为浏览器秒级响应，`/api/encode` 曾用**一次 teacher-forced forward** 拿全部位置条件分布（而非逐 token 重跑 T 次）。理论上模型 causal → 位置 i 分布只依赖 token[0..i]，应与逐 token 路径一致。
- **实测翻车（2026-06-01）**：单次 forward 与逐 token forward 的 logits 在 GPU 上差 ~1 ULP（实测 max\|Δ\|=3.8e-5，归约顺序差异）。teacher-forced NLL/bpd 容忍这点误差，但**算术编码零容忍** —— 某 token 跨累积频数边界翻符号 → 解码自此失步 → 还原图成彩色噪点。
- **修正**：`/api/encode` 改走 `verify_lossless._encode_sequence_iter`，与 `/api/decode` 的 `_decode_sequence_iter` **共用 `_logits_from_tokens`**（prefix+0 缓冲逐位相同）→ logits 逐位相同 → cumfreq 表相同 → bit-exact 可解。代价是 encode 也要 ~3100 次 forward（与 decode 同量级），故同样流式吐 NDJSON 进度绕开反代超时。
- **诚实声明**：CC-iGPT 从 **DECODED** coarse token 重建 fine ctx（decoder 视角，不作弊）；编码侧不再显示 `bit-identical` 徽章（token 化无损恒真，无证据价值），真正的可解性证据是 decode 端「指纹与刚编码的 .bin 逐 token 一致」交叉校验。严格逐步版同源由 §6.3 `verify_lossless.py` CLI 背书（2/2 bit-identical）。

### 执行时序（gating）

**§6 保留任务代码已就绪**：§6.1 IN64->CIFAR transfer probe（待 IN64 最终 ckpt）、§6.2 补全、§6.3 可解性（+ 落盘 .bin）。复用 CIFAR v2 现成 ckpt，不写训练目录；但补全和 codec 会抢 GPU，IN64 训练期间不要同时发起。

### 6.5 IN64 训练完成后用 IN64 权重重跑下游（2026-06-07 训练完成；ep12 best.pth 为最终权重）

**计划**：IN64 训出最终 `best.pth` 后，优先用 IN64 权重跑 transfer probe，作为"大数据集预训练 -> 表征更强"的第二轨道证据。补全和可解性可离线小规模跑，但不进实时 demo。

| 下游 | IN64 重跑可行性 | 坑 / 改动 |
|---|---|---|
| ① transfer probe | ✅ **本就是为 IN64 设计的**（§6.1）| `linear_probe.py --probe_dataset cifar10`，CIFAR 图 resize 32→64，与 32-native 66.93/79.33 **非同协议**，只作 transfer 趋势 |
| ② 图像补全 | ❌ **慢到不可用** | fine 12288 token，无 KV-cache 单图采样 ~十几分钟（§6.2 已标）；IN64 补全仅能离线出极少量网格图，**不能上 demo 实时** |
| ③ 可解性 roundtrip | ❌ **太慢** | 12288 token，encode+decode 各一遍逐 token forward，单图可能数十分钟（§6.3 已标"IN64 太慢"）；最多离线验 1 张证可逆，不进交互 |

**结论**：IN64 收敛后，**transfer probe 是主目标**（天然为它设计、最有叙事价值）；补全与可解性受 12288 token 拖累，IN64 版仅能离线小规模做、不上 demo。**CIFAR v2 仍是补全/codec 的主力载体**，IN64 下游是补充轨道而非替换。

### 6.6 "为什么 linear probe works" — 理论阅读清单（LRH / 信息论 probing）

研究方向：把 §6.1 现有的"压得越好 → 线性可分表征越强"从**断言**升级为**机制解释** —— 即 LLM 可解释性里的 linear representation hypothesis (LRH) 在 AR 像素压缩模型上的对应。核心问题拆成两半：(A) 为什么压缩目标会逼出语义（MDL 半，§6.1 已有）；(B) 为什么语义是**线性可读**的（LRH 半，是真正有贡献的一半）。

本模型上 (B) 的四条压力（前三条是架构特异性，正是可写成贡献的点）：
1. **线性 weight-tied readout**（`igpt.py:74` head = 单个 `nn.Linear(d,256)` tied token_embed，末 hidden→logits 零非线性）→ 预测相关结构被梯度推进线性方向。
2. **加性残差流**（OLMo2 post-norm `x = x + RMSNorm(sublayer(x))`）→ 表征是各层贡献的 running sum，线性方向是残差流的"母语"（Anthropic residual-stream/superposition 图景，与 LM 同构）。
3. **Superposition**（d=448 但概念数 ≫ 448，高维近正交 → 概念打包成近正交线性方向）。
4. **`α·coarse_ctx` 是加性上下文向量**（`cc_igpt.py:164` 注入 layer-0 残差流），适合研究线性干预；但旧 `+12.4pp` 比较混入架构和训练预算差异，不能作为 ctx 的因果提升，需 matched retraining ablation。
中层峰 (L0→L19↑→L31↓)：早层局部像素统计、中层语义抽象峰、末层为 256-way 输出分布再特化丢弃语义（信息瓶颈/tunnel 效应）。同样的倒 U 在 iGPT 与 LLM 都出现 → 几何由**目标形式 (NTP + 线性 head) + 架构 (残差流)** 决定，与模态无关，这是"类似 LLM interpretation"的答案。

**阅读清单**（⭐ = 直接回答本问题的核心 5 篇；标题/年份可靠，引用前复核 venue/arXiv ID）：

A. probe 到底测什么（方法论，先读）
- ⭐ Alain & Bengio 2017, *Understanding intermediate layers using linear classifier probes* (ICLR-W) —— 技术起源，`linear_probe.py` 已引；确立"逐层线性可分性"视角。
- ⭐ Hewitt & Liang 2019, *Designing and Interpreting Probes with Control Tasks* (EMNLP) —— 必答的质疑：高 acc ≠ encoded；引入 **selectivity / control task**。答辩护盾。
- Pimentel et al. 2020, *Information-Theoretic Probing for Linguistic Structure* (ACL) —— probing = 估互信息；data-processing-inequality（信息恒在，probe 测的是**可提取性**）→ 本问题真正是"为什么线性可提取"。
- Voita & Titov 2020, *Information-Theoretic Probing with MDL* (EMNLP) —— probing 即压缩；probe 质量本身是 MDL 量，与本课题 MDL 主线天然融合。
- Xu et al. 2020, *A Theory of Usable Information (V-information)* (ICLR) —— linear probe acc = 线性预测族下的 V-usable info；形式化挂钩。

B. 为什么概念是线性的（LRH 半，核心）
- ⭐ Elhage et al. 2021, *A Mathematical Framework for Transformer Circuits* + 2022 *Toy Models of Superposition* (Anthropic) —— 残差流即线性 superposition；本模型加性流与之同构（压力 2+3）。
- ⭐ Park, Choe, Veitch 2024, *The Linear Representation Hypothesis and the Geometry of LLMs* (ICML) —— 形式化 LRH：embedding vs unembedding、causal inner product、概念=正交方向；本模型 weight-tying 令 embedding≡unembedding 是其特例。
- Jiang et al. 2024, *On the Origins of Linear Representations in LLMs* —— 线性性源自 NTP 目标 + log-odds 结构；最直接的"为什么 NTP→线性"，平移到像素 NTP。
- Engels et al. 2024, *Not All Language Model Features Are Linear* —— 诚实反例（部分特征是 circular/manifold）；引它让结论读作"强近似"而非定律，护交叉质询。

C. 中层峰（倒 U L0→L19↑→L31↓）
- ⭐ Chen et al. 2020, *Generative Pretraining from Pixels (iGPT)* (ICML) —— 直接血脉，已引；其 mid-layer probe 峰图 = 本工作同款曲线，开篇即引。
- Tishby & Zaslavsky 2015 / Shwartz-Ziv & Tishby 2017, Information Bottleneck —— 倒 U 背后的"压缩-再特化"动力学。
- 2024 多篇 "intermediate layers best representations LLM"（搜该关键词）—— 确认中层峰跨模态，正是"类似 LLM"卖点。

D. 线性方向是因果的（steering 故事）
- Turner et al. 2023, *Activation Addition (ActAdd)*；Zou et al. 2023, *Representation Engineering*；Anthropic 2024, *Scaling Monosemanticity* (SAE 因果 steering) —— 把 `α·coarse_ctx` 论证成**工程化 steering vector**，并支撑"沿 probe 方向 steer 图像补全"的因果实验。

**只读四篇破题**：Jiang 2024（NTP→线性）+ Elhage 2022（残差流/superposition）+ iGPT 2020（像素域中层峰）+ Hewitt & Liang 2019（证明真实非记忆）。

**可选实验**（把断言变证据，复用 `linear_probe.py`，WSL 写 / AutoDL 跑）：
- E2 跨 checkpoint 画 val bpd vs best-layer probe acc（有 `epoch_6..12.pth`）—— 检验二者相关性的直接证据；因果结论仍需匹配训练消融。
- E1 加 1 隐层 MLP probe 对比 —— gap 小 ⇒ 信息真线性编码 (LRH 成立)；LRH 头条证据。
- E3 control task / selectivity (Hewitt & Liang) —— 随机标签基线，证 probe 读结构非记忆。
- E5 coarse_ctx 消融（`--no_coarse_ctx` 已存在）—— 量化加性方向逐层贡献，验压力 4。
- E4 用 probe 权重向量做残差流 steering —— 验方向因果性，图像域版 activation steering（工作量最大、最像 interpretation）。

---

## 7. 未来方向（不在毕设范围内）

- **ImageNet 64×64 benchmark**：历史实验日志为单 ckpt+TTA `3.4810`、三成员 ensemble+TTA `3.4800`；二者可按各自协议引用。现有预训练 `best.pth` 的正式 single-checkpoint/no-TTA teacher-forced 结果为 **3.4812 bpd**（49,999 张；逐图 std 0.9040；95% bootstrap CI `[3.4734, 3.4898]`），详见 schema v5 manifest。旧 `±0.1161` 是 batch-level std，作废为正式不确定性指标；sequential roundtrip 和实际 file bpd 仍未测量。

### 7.1 Diffusion / VDM 路线判断（2026-06-05；零基础阅读顺序一并列出）

**结论先行**：如果主线仍是 **RGB-bit-exact / bpd / arithmetic coding**，不要直接把当前 CC-iGPT 换成连续 VDM 或 Stable-Diffusion 式 decoder。当前项目最强的链路是 `categorical CE → bpd → 算术编码真实可解`；连续 VDM 需要 latent / bits-back / discretization 处理，工程上会打散这条优势。更合理的入口是 **discrete ARDM / masked diffusion**：仍然预测 uint8 token，仍然用 softmax CE 和算术编码，但把逐 token AR 改成分组/多步 schedule，从而显著减少 encode/decode forward 次数。

#### A. 推荐技术路线：CC-iGPT → CC-MDLM（主推）

目标不是马上刷新 `2.8296 bpd`，而是做出一条新的 trade-off 曲线：

```
full AR:      3072 / 12288 次条件预测，bpd 最优但解码慢
CC-MDLM-GK:     16 / 32 / 64 个 deterministic groups，码率与运行时间均待实测
```

实现原则：

- **保留 RGB-bit-exact token**：状态空间仍是 `0..255` categorical token，不进入连续像素扩散。
- **fine 分支改 bidirectional transformer**：去掉 causal mask，输入已知 token + `[MASK]` token + timestep/mask-ratio embedding。
- **训练 objective**：先实现与固定 group factorization 对齐的 masked CE；再把随机时间 MDLM-NELBO 作为独立 objective ablation。coarse 分支仍独立进 bitstream。
- **压缩/解压 schedule 必须确定**：encoder/decoder 共享同一个 group 顺序，不能依赖随机采样。
- **编码方式**：每一组内假设条件独立，用模型给该组所有 masked token 的 categorical 分布，再逐 token 送入 arithmetic coder。
- **评估主图**：`payload/file bpd` vs `fine/total forward calls`，另报 encode/decode wall-clock。任何 gap 或速度数字都必须来自真实 codec；`0.03~0.08 bpd` 只能作为实验假设，不能预写成结论。

首批 schedule：

| schedule | 作用 | 预期 |
|---|---|---|
| `raster_chunks` | sanity baseline；按原 raster 序列切成 K 段 | 最容易实现，但空间条件弱 |
| `checkerboard` | 先预测棋盘 anchor，再补洞 | 图像局部条件更强 |
| `pyramid` | `8×8 coarse → 16×16 mid → 32×32 fine groups` | 最贴合 CC-iGPT 叙事 |
| `channel_refine` | 先 R/Y，再 G/B 或 chroma | 可能降低色彩条件熵 |

命名统一为 `CC-MDLM`（Coarse-Conditioned Masked Diffusion Language Model）；固定 K 组 codec 变体记为 `CC-MDLM-GK`，例如 `CC-MDLM-G32`。ARDM 是 grouped compression protocol 的来源，不再作为项目总名。第一版只做 CIFAR-10。

#### B. 中期结构升级：加 16×16 mid-scale

当前是：

```
8×8 coarse -> 32×32 fine
```

建议扩成：

```
8×8 coarse -> 16×16 mid -> 32×32 fine
```

到 ImageNet64 时自然变成：

```
16×16 coarse -> 32×32 mid -> 64×64 fine
```

优先让 mid-scale 只压 R / Y 一类低频主通道，fine 继续 RGB sub-pixel。这样比直接把 64×64 fine attention 硬扛到 12288 token 更稳，也给 CC-MDLM 的 `pyramid schedule` 一个天然落点。

#### C. 低风险结构 ablation

- **per-layer ctx_alpha**：把单个 `ctx_alpha` 改成每层一个标量，成本最低，解释清楚。
- **channel / token gate**：`hidden = hidden + gate * coarse_ctx`，gate 可由 layer 或 channel 学。
- **tiny FiLM**：`hidden = hidden * (1 + gamma(ctx)) + beta(ctx)`，表达力更强，但要防止过拟合和破坏 bit-exact ctx 一致性。
- **reversible color transform**：理论上 RCT / YCoCg-R 是整数可逆、可保持 RGB-bit-exact；但当前红线里明确禁了 YCbCr/YCoCg-R（历史失败项），所以只能作为独立分支重开，不能混进毕设主线。

#### D. VDM / continuous diffusion 的定位

VDM 可以读，但不建议先实现。原因：

- VDM 是连续 latent variable likelihood model，和当前 `uint8 categorical CE` 口径不同。
- 若要把 VDM 变成真正无损 codec，需要 bits-back / ANS 一类隐变量编码，单图压缩还有初始化 bits 和 amortization 问题。
- 作为论文讨论可以说：VDM 提供 diffusion likelihood 的理论上界，但本项目选择 ARDM 是因为它保留了单图 arithmetic coding 的工程闭环。

如果一定要做 diffusion codec，建议单开 **有损/感知压缩** 支线：

```
entropy bottleneck / quantized latent -> conditional diffusion decoder
metrics: bpp, PSNR, MS-SSIM, LPIPS, FID
```

这条线不要和 RGB-bit-exact bpd 主表混在一起；它回答的是“低码率感知质量”，不是“严格无损 MDL”。

#### E. 从零基础开始的论文阅读顺序

→ diffusion 核心阅读路径已统一到 **§2 论文阅读 Roadmap**（Stage 0–5，最小关键路径 D3PM→MDLM→ARDM→MaskGIT），照那个读即可，不在此重复维护（避免两份清单漂移）。

本 §7.1 备忘特有的"为何不走连续 / 有损"对照篇（仅作路线判断背景，**不在 CC-MDLM 实现路径上**）：
- Kingma & Welling, **VAE** (2013, arXiv:1312.6114) —— variational bound / latent overhead，理解连续 likelihood 模型为何有 latent 开销。
- *(VDM 已升入 §2.1 理论主线 作核心理论读物 —— 它是 diffusion↔最大似然的桥，非仅"对照篇"；连续实现仍在红线外，但理论必读。)*
- Mentzer et al., **HiFiC** (2020, arXiv:2006.09965) / Ballé hyperprior 系列 —— 有损生成式压缩的 rate-distortion 评价体系，与无损 bpd 是两条线（避免混为一谈）。

结论不变：第一份训练代码是 `CC-MDLM-GK` 的 CIFAR-10 toy；在它之前先完成 §0 的无训练 tiny codec protocol。比较的是真实 file bpd、calls 与 wall-clock，不反过来先写连续 VDM。

---

## 8. 红线（DO NOT 项）

| 事项 | 原因 |
|---|---|
| 重启 DMoL / VQ / 任何输出头改造 | DMoL 6 次失败彻底退场（§5）|
| 加 Mixup / CutMix / label smoothing | NLL 任务文献支持有限 |
| 改色彩前端 (YCbCr / YCoCg-R) | `MEMORY.md` 全禁 |
| 加 logit soft-capping / Gaussian LS / sliding window attn | 已删除项 |
| WSL 上跑 CUDA 训练 / profiling | 本地只做 CPU 验证；训练、CUDA kernel 测试与 profiling 放到 AutoDL |

## 9. Phase E 架构演进指南：从 CC-iGPT 到 CC-MDLM

> **历史版本锚点**：commit `debc3af`、branch `dev`、2026-06-12 全仓通读核对。§E3 的旧 file:line 引用仅作线索；实现以当前代码和测试为准。
> **命名约定**：研究总名 **CC-MDLM**；固定 K 组 codec 为 **CC-MDLM-GK**。分支 `phase_e_cc_mdlm`、不移位前向 `forward_masked`、调度模块 `masking_scheduler.py`。ARDM 仅指借鉴的 grouped factorization/coding protocol。

> **核心动机**：探索用离散 masked/grouped factorization 减少传统 AR 的串行模型调用。2025–2026 工作只作为待核检索线索，不作为方案成立的前提或正式引用。
> **战略底线**：首版保留 RGB uint8 离散 token，并为 CC-MDLM-GK 单独证明算术编码闭环；旧 AR codec 的可解性不能自动继承给新模型。连续 latent diffusion 只作独立研究线。

---

### E1. 战略定位与核心原则

#### E1.1 为什么选择 CC-MDLM 而不是连续 VDM/Latent Diffusion？
如果您直接转向类似 COLA DLM 或 Stable Diffusion 的**连续潜在空间扩散**，将面临灾难性的工程重构：
1. **丧失 Bit-Exact**：连续空间必须经过量化或复杂的 Bits-back 编码才能无损存储，这将彻底打破您目前优雅的 `CE loss = bpd = 算术编码长度` 的直接对应关系。
2. **Triton 资产作废**：您精心优化的 Categorical CE Triton Kernels（核心是 `fused_ce_zloss`，**不是** `fused_linear_ce` —— 后者在 V=256 下经 roofline 证伪、是反面案例，不在训练栈）将无法用于连续空间的 MSE/Score Matching。

**结论**：采用 **CC-MDLM (Coarse-Conditioned Masked Diffusion Language Model)** 作为研究方向；首个可解码实例是使用固定 grouped factorization 的 `CC-MDLM-GK`。
*   **状态空间**：严格保持 `V=256` 的 `uint8` categorical tokens。
*   **待验证收益**：将 fine 分支模型调用从 3071 次降至 `K=32/64`（fine-call reduction 约 96.0x/48.0x）；保留 coarse AR 后总调用 reduction 约 33.0x/24.7x。运行加速和码率损失均未知，目标是实测 **`file bpd vs. total forward calls vs. wall-clock`** 前沿。

#### E1.2 MDL、NELBO 与实际码长的边界
扩散 ELBO/NELBO 是对负对数似然的变分上界，不应直接写成现成文件的算术码长。要把它实现为无损 code，必须指定 latent trajectory 或固定 grouped factorization、双端共享的 schedule、CDF 量化和容器协议；部分变分方案还涉及 bits-back 及初始 bits。本文第一版采用 §0 的显式 grouped factorization，因此其 schedule NLL 可直接对应理想码长；MDLM-NELBO 只作为训练/理论指标单列。只有实际 roundtrip 后，才能说 MDL 命题扩展到了该 masked codec。

---

### E2. 论文阅读 Roadmap

> **组织原则（VDM 教训，2026-06-12）**：旧版按"实现流水线 Stage"单轴排，差点把 **VDM 这种"实现红线外、却理论必读"的论文当对照篇丢掉**（Bits-Back 当时也犯同样错）。改为**双维标注**——每篇先写 ①核心贡献，再标 ②角色 ∈ {**理论主线 T** / **实现主线 I** / **边界对照 B**}。**判断"读不读"看理论维，"抄不抄代码"看实现维，两维独立**；"我们不实现 X" ≠ "X 不值得读"。**先读完再动代码**（用户决策 2026-06-12）。

> ⚠ **引用核对**：标题/作者/venue 凭知识给出（cutoff 2025-08），**arXiv 编号离线重建、引用前逐个核对**（项目"verify venue/arXiv before citing"纪律）。

#### E2.1 理论主线 T：NELBO 何时能转化为码长

这条线用于判断训练 bound、模型 likelihood 与可实现 code length 何时一致、何时只是不等式。**VDM 是理论枢纽，但不能代替 codec 协议。**
- *(根)* Shannon, *A Mathematical Theory of Communication*, 1948, Bell System Tech. J.（无 arXiv）。source coding theorem：熵 = 最优无损码长下界。一切"CE = bpd"的源头，开篇一句话引。
- *(项目命名之源)* Rissanen, *Modeling by Shortest Data Description*, 1978, Automatica（无 arXiv）+ Grünwald, *A Tutorial Introduction to the MDL Principle*, 2007, arXiv:math/0406077（选读，体系化）。项目命名源于 MDL；严格边界见 §E1.2。
- Hinton & van Camp, *Keeping Neural Networks Simple by Minimizing the Description Length of the Weights*, COLT 1993（无 arXiv）。**bits-back 思想的起点** + 把 MDL 接到神经网络权重上；与下文 Bits-Back-ANS 配对读（一个讲思想、一个讲落地编码）。
- VAE — Kingma & Welling, 2013, arXiv:1312.6114。ELBO / variational bound 的根；理解 likelihood 模型的码长下界与 latent overhead 从哪来。
- *(护盾)* Theis, van den Oord, Bethge, *A Note on the Evaluation of Generative Models*, ICLR 2016, arXiv:1511.01844。**为何用 bpd/NLL 而非 FID/PSNR/SSIM**：三者无单调关系、对无损压缩唯有 NLL 有意义。直接撑起项目"不做有损指标"红线，答辩护盾。
- DDPM — Ho, Jain, Abbeel, NeurIPS 2020, arXiv:2006.11239（**亦在实现线**）。denoising objective = 后续 masked CE 的祖宗，只读 objective。（score/SDE 统一视角见 Song et al. 2021, arXiv:2011.13456，选读）
- ⭐ **VDM** — Kingma, Salimans, Poole, Ho, NeurIPS 2021, arXiv:2107.00630。**diffusion ↔ 最大似然的桥**：用于理解 variational bound 与 likelihood 的关系，也用于提醒“bound”不等于无需协议即可落盘的 code length。实现红线外，理论必读。
- ⭐ **Bits-Back with ANS** — Townsend, Bird, Barber, ICLR 2019, arXiv:1901.04866。隐变量模型接近 −ELBO 码长所需的 bits-back 机制；用来检查 masked diffusion 是否隐含需要传输/回收 latent path，而不是先假定“不需要 bits-back”。显式固定 grouped factorization 无 latent path 时才可直接按其 NLL 编码。
- ⭐ **Language Modeling Is Compression** — Delétang et al., ICLR 2024, arXiv:2309.10668（项目支柱）。直接支持已定义顺序概率分解的 arithmetic coding；对 masked next-set 必须先定义联合分布/组内独立假设，不能仅凭 masked CE 自动推出码长。

#### E2.2 实现主线 I：怎么搭 CC-MDLM

- *(前身)* **Multinomial Diffusion** — Hoogeboom et al., *Argmax Flows and Multinomial Diffusion*, NeurIPS 2021, arXiv:2102.05379。D3PM 的直接前身（同作者线）：categorical 变量怎么扩散、uniform vs absorbing 转移核；先读它再读 D3PM 更顺。
- *(祖宗·目标)* **BERT** — Devlin et al., NAACL 2019, arXiv:1810.04805。masked token 预测目标的源头；masked diffusion 取单步 = BERT-style MLM。读它认清"我们的 fine 分支 = 把 BERT 的随机掩码接上算术编码"。
- *(祖宗·任意序)* **XLNet** — Yang et al., NeurIPS 2019, arXiv:1906.08237。permutation / order-agnostic AR；ARDM "任意序"立论的直接祖宗，理解为何任意分解顺序仍是合法 likelihood。
- ⭐ **D3PM** — Austin, Johnson, Ho, Tarlow, van den Berg, NeurIPS 2021, arXiv:2107.03006（**亦在理论线**）。**absorbing state** = 我们的 `[MASK]`；token 也能 diffusion 且保留 categorical CE。
- ⭐ **MDLM** — Sahoo et al., *Simple and Effective Masked Diffusion Language Models*, NeurIPS 2024, arXiv:2406.07524 + **Shi et al.**, *Simplified and Generalized Masked Diffusion for Discrete Data*, arXiv:2406.04329（**两篇不同论文，勿混**）。带走：masked-diffusion NELBO 的重加权 masked CE 形式及其 likelihood-bound 语义。它是训练目标候选，不自动给出本项目固定 K-step 文件码长；必须与 ARDM compression protocol 对照。
- ⭐ **ARDM** — Hoogeboom et al., *Autoregressive Diffusion Models*, ICLR 2022, arXiv:2110.02037。**重点读 compression section**：order-agnostic AR / 并行分组解码做无损编码 —— masked diffusion ↔ 算术编码闭环的桥，**训练目标决策的右选项（group-AR）**。带走：按组编码、#network-calls vs bpd。
  → **读完 MDLM + ARDM 敲定训练目标**（任意序 vs group-AR；详 §E3.1/§E3.3）。
- ⭐ **MaskGIT** — Chang, Zhang, Jiang, Liu, Freeman, CVPR 2022, arXiv:2202.04200。masked 图像 token + K 步 confidence 并行解码；checkerboard/cosine schedule 的直接先例。confidence 排序数据相关，第一版 bit-exact 路径不上（§E3.1）。
- *(schedule)* **Improved DDPM** — Nichol & Dhariwal, ICML 2021, arXiv:2102.09672。cosine noise schedule 的出处；参照 §E3.4。
- *(MaskGIT 采样升级，选)* **Token-Critic** — Lezama et al., ECCV 2022, arXiv:2209.04439。给 MaskGIT 并行解码加 critic 重排，提采样质量；若第一版 confidence schedule 效果差时的备选（仍在 bit-exact 红线外，只作启发）。
- **VAR** — Tian et al., NeurIPS 2024 (best paper), arXiv:2404.02905。next-scale prediction；coarse-AR + fine-masked 的 `pyramid` schedule（§E3.4）来源。
- *(选 / 工程)* LLaDA（Nie et al., 2025, arXiv:2502.09992）大规模 masked-diffusion 的 schedule/sampling 实践；Muse（Chang et al., ICML 2023）放大版 MaskGIT。

#### E2.3 边界对照 B：知道我们不做什么

- SEDD（Score Entropy Discrete Diffusion）— Lou, Meng, Ermon, ICML 2024, arXiv:2310.16834。离散扩散的**替代目标**（score-entropy / 比率匹配），理论漂亮但非我们的 masked-CE 路线；选读其与 likelihood 的关系。
- Discrete Flow Matching（Gat et al., NeurIPS 2024, arXiv:2407.15595）/ Generative Flows on Discrete State-Spaces（Campbell et al., ICML 2024, arXiv:2402.04997，**两篇不同，勿混**）。无需连续 latent 的离散生成替代路径，与 masked diffusion 互补、非首选。
- *(主表对手·AR 图像)* PixelRNN/PixelCNN（van den Oord et al., ICML 2016, arXiv:1601.06759）→ Gated PixelCNN（arXiv:1606.05328）→ **PixelSNAIL**（Chen et al., ICML 2018, arXiv:1712.09763）→ **Image Transformer**（Parmar et al., ICML 2018, arXiv:1802.05751）→ **Sparse Transformer**（Child et al., 2019, arXiv:1904.10509）。**主表逐个对比的 AR 无损基线**（CIFAR 2.80–2.90 区间、IN64 3.44）；读懂它们的 bpd 口径与 raster-scan 因式分解，才能说明哪些结果真正同协议可比。
- HiFiC（Mentzer et al., 2020, arXiv:2006.09965）/ Ballé hyperprior / Cascaded Diffusion（Saharia et al., JMLR 2022）。**有损 / 感知压缩**（rate-distortion），与无损 bpd 是两条线，读到能区分指标即可。
- 连续 VDM 的**实现** / COLA DLM（ByteDance Seed 2026，连续 latent）—— **红线外**，只反向参考其"打破 AR 瓶颈"的机制思想（VDM 的*理论*在 §2.1，勿与其*实现*混为一谈）。
- IDF（Hoogeboom et al., NeurIPS 2019, arXiv:1905.07376）/ IDF++（van den Berg et al., ICLR 2021, arXiv:2006.12459）/ L3C（Mentzer et al., CVPR 2019, arXiv:1811.12817）—— 学习式无损 baseline，对比用。
- *(bits-back 无损实测)* **Bit-Swap**（Kingma, Abbeel, Ho, ICML 2019, arXiv:1905.06845）/ **HiLLoC**（Townsend et al., ICLR 2020, arXiv:1912.09953）。把 §2.1 的 bits-back 理论落到**真实无损 codec**：分层 latent + ANS 的码长实测。**反衬**我们离散无 latent 路线"无 bits-back 开销、算术编码闭环"的工程简洁性。

#### E2.4 最小关键路径

**实现**：D3PM → MDLM → ARDM → MaskGIT　＋　**理论**：VDM → Language Modeling is Compression。
六篇破题：前四篇帮助定义 CC-MDLM，后两篇帮助讲清 likelihood bound、概率分解与可实现 code length 的边界。

#### E2.5 博客 / 教程 / 讲解

> 都是二手讲解，**只用来建直觉、不进引用**；正式叙事一律回到 §2.1–2.3 原始论文。URL 凭记忆给，点开前自行核对域名是否仍有效。

- **Lilian Weng**, *What are Diffusion Models?*（lilianweng.github.io，2021；含 2024 离散/扩散 LM 更新）—— 连续 diffusion 数学推导第一站，把 DDPM/VDM 的 ELBO 推一遍，配 §2.1 读。
- **Sander Dieleman**, *Diffusion models are autoencoders*（2022）+ *Diffusion language models*（2023，sander.ai）—— 离散 / masked diffusion 的直觉性串讲，专门讲"为何离散也能 diffusion"，配 §2.2 D3PM/MDLM 读，强烈推荐。
- **Yang Song**, *Generative Modeling by Estimating Gradients of the Data Distribution*（yang-song.net，2021）—— score-based / SDE 统一视角的作者亲笔讲解，§2.1 的 Song 2021 选读篇的友好版。
- **HuggingFace**, *The Annotated Diffusion Model*（huggingface.co/blog/annotated-diffusion）—— 逐行 PyTorch 注释实现 DDPM；想看"objective 怎么落成代码"时对照，但我们不照抄（红线外连续路线）。
- *(MDL / 压缩直觉)* **Marcus Hutter**, *Hutter Prize* 主页 + Delétang "Language Modeling Is Compression" 的官方 blog/talk（DeepMind）—— "智能 = 压缩"论点的科普入口，§1.2 立论的通俗背书。
- *(算术编码工程)* Matt Mahoney, *Data Compression Explained*（mattmahoney.net/dc）—— 算术编码 / range coder 的工程实现细节，回看 `codec/arithmetic.py` 的 WNC 实现与 ULP 确定性坑时的参考。

#### E2.6 前沿 / 收尾参考（2024–2026）

> 上面四节是地基。这一节是**隔离的检索队列**：2025–2026 条目在核对论文主页、作者、版本、venue、arXiv/DOI 并记录访问日期前，不进入正式引用、不支撑方法选择、不写“方向已验证”。标为待核的条目甚至可能不存在或元数据有误。

- *(待核)* **Block Diffusion / Interpolating Between Autoregressive and Diffusion Language Models** — 作者、venue 与 arXiv 元数据待在线核验。若论文确实定义块间 AR、块内 diffusion，再评估其与 coarse-AR + fine-masked 的关系；当前不称为“本项目实例”。
- *(实现·并行加速)* 离散扩散**蒸馏**——SDTT（*Score Distillation Through Time*, 2025）/ Di4C（多步→少步蒸馏，2025）等 *(待核 arXiv)*。把 K 步并行解码进一步压到个位数 forward；与 §3.2 的 `bpd vs #forward` Pareto 同一目标，作"还能更快吗"的延伸读物。
- *(实现·采样加速·清华 TSAIL)* **DPM-Solver / DPM-Solver++ / -v3**（Lu, Zhou, …, Zhu；NeurIPS 2022 / 2023, arXiv:2206.00927 / 2211.01095 / 2310.13268）+ **Analytic-DPM**（Bao et al., ICLR 2022 outstanding, arXiv:2201.06503）。**与本项目 headline 同轴**：把（连续）diffusion 采样从上千步压到 10–20 步 = 我们 K 步并行解码的连续版对照。**和清华 TSAIL 交流的接口论文**（见 §9）；离散 vs 连续是两条线，借其"少步采样"思路、不抄其连续实现。
- ⭐ *(理论·absorbing=任意序条件)* **RADD** — *Your Absorbing Discrete Diffusion Secretly Models the Conditional Distributions of Clean Data*, Ou, Li 等, 2024, arXiv:2406.03736 *(核对)*。证明 absorbing/masked 离散扩散的 concrete score 可**重参数化为干净数据的任意序条件分布** → masked diffusion ≈ any-order AR。**直接撑起 CC-MDLM fine 分支的理论合法性**，与 §2.2 MDLM 同线但更直接，建议精读。
- *(实现·大规模 masked diffusion LM)* **LLaDA** — *Large Language Diffusion Models*, Nie et al., 2025, arXiv:2502.09992（§2.2 已列）。8B masked-diffusion LLM；证明 masked diffusion 能与 AR-LLM 同量级竞争，"方向被验证"的有力佐证。
- *(理论·目标统一)* MD4 / GenMD4（Shi 等后续的"广义 masked diffusion"，2024–2025）*(与 §2.2 Shi et al. 同线，待核)* —— state-dependent masking 等推广，看 masked-CE 目标族的边界，选读。
- *(边界·连续主导范式)* **Flow Matching**（Lipman et al., ICLR 2023, arXiv:2210.02747）+ **Rectified Flow**（Liu et al., ICLR 2023, arXiv:2209.03003）—— 2024–2026 连续生成的主导训练范式（SD3 等用它）。**红线外**（连续），但要知道"主流去哪了"，读到能一句话说清它与离散 masked 的分野即可。
- *(边界·LLM 无损压缩落地)* `ts_zip` / `cmix` / NNCP（Bellard 等的 LLM-as-compressor 工程）与 Delétang 后续实测 —— "大模型当无损压缩器"的真实 bpd 数字，作我们 codec 工程口径的旁证（非首选基线）。
- *(传闻/待核·工业 masked diffusion)* Gemini Diffusion / 商用 diffusion-LLM（2025）—— 证明 masked-diffusion 已能在大规模工作；仅作"方向被验证"的叙事佐证，**存在性自行核实**，不进正式引用。

#### E2.7 稀疏性与 CC-MDLM 的结合

> **关键转折（2026-06-20 讨论）**：固定 mask schedule 允许使用数据无关的静态稀疏结构，encoder/decoder 在结构层面容易复现。它仍不“天然 bit-exact”：浮点 kernel、硬件、软件版本和 CDF 量化都可能造成分布差异，必须沿用真实 roundtrip 与 manifest 验证。稀疏 attention 只在 E1–E4 通过后研究。

**三根稀疏轴，按与 CC-MDLM 契合度排序：**

- **轴 1 · 时间/步稀疏（#forward-calls）—— headline，契合度最高、最该先做。** 就是 CC-MDLM 核心卖点本身：K 步并行解码 = 在"步"轴上稀疏化（N token → K 次 forward）。工作：**ARDM**（§2.2，group-AR 并行解码 + #network-calls vs bpd）、**离散扩散蒸馏 SDTT / Di4C**（§2.6，*待核*，把 K 步再蒸到个位数 forward）。**对 CC-MDLM 最有用的稀疏是"时间稀疏"，不是"空间稀疏"。**
- **轴 2 · 结构/块稀疏（静态、确定性、图像原生）—— 潜在契合，但需要独立 kernel 工程与验证。**
  - **Block Diffusion**（§2.6，引用待核）：块间 AR、块内 diffusion 可作为结构参照。block-sparse attention 需要新的正确性、数值和性能实现，不能视为给现有 kernel “加一个 mask 即可”。
  - **Axial Attention**（Ho et al., 2019, arXiv:1912.12180, *核对*）：2D 注意力分解成行/列，O(n·√n)，确定性、图像原生；Axial-DDPM 已验证配 diffusion 可行。**IN64 12288 序列（已 batch 48→12 抗 OOM）的真实显存解药。**
  - **Sparse Transformer**（Child 2019, arXiv:1904.10509）：主表**已引**，strided/fixed pattern，确定性；稀疏注意力顺这条血脉走，**引用自洽**，不必硬塞 2025 LLM 技术。
- **轴 3 · mask-conditioned 注意力稀疏 —— 候选研究点。** masked 位置只 attend “已揭示位置 + 局部窗口”，已揭示位置 attend 全局；揭示集由协议共享，所以结构可复现。它仍需验证对 likelihood、CDF 一致性和运行时间的影响；Longformer/BigBird 的 global+local 只能作结构参照。

**图谱（两轴标注，延续 §2 体例）：**

| 工作 | 稀疏轴 | 协议状态 | 对 CC-MDLM |
|---|---|---|---|
| ARDM / grouped masked | 时间（步） | 首版必须验证 | headline，先做 |
| Block diffusion 类方法 | 结构（块） | 文献与实现待核 | 后续参照 |
| Axial / Sparse Transformer | 结构（图像 pattern） | 静态 mask 可复现，数值待测 | 高分辨率候选 |
| Longformer/BigBird（via coarse） | 结构（global+local） | 静态 mask 可复现，数值待测 | 中期候选 |
| mask-conditioned sparse | 协议/状态 | schedule 可复现，codec 待测 | 研究假设 |
| 数据相关 top-k sparse | 内容路由 | 首版风险过高 | 暂不做 |

**红线 / 为何排除 NSA·MLA·MoE**（与全注意力问题同源，见 [[project_sparse_attention_assessment]]）：
- **NSA / DSA 类学习式稀疏**：数据相关路由把数值确定性和协议复杂度显著放大，第一版排除；不是数学上绝对不可编码，而是当前 codec 风险/收益不成立。
- **GQA / MLA**：主要价值常在 KV-cache，但即使无 cache 也可能减少 K/V 投影和注意力流量；当前瓶颈未测，暂不优先，而不是断言收益为零。
- **MoE**：稀疏的是参数（另一根轴），加路由非确定性 + 负载均衡，对 82M 小模型是工程黑洞、离叙事。**不适用。**
- **causal AR 上的空间稀疏**：3072 序列全注意力养得起，丢连接 = 丢预测信息 = bpd 升，反向交易。**仅当上 ≥64px（IN64 12288 / 128px 49152）显存逼人时，才上轴 2 的静态稀疏。**

**一句话**：先验证时间轴上的 K-step grouped codec；静态结构和 mask-conditioned 稀疏属于后续性能研究，数据相关路由暂不进入首版协议。

---

### E3. CC-MDLM 架构设计与工程改造路线

#### E3.0 代码地基现状（2026-06-12 全仓通读后核对）

实现 Phase E 前先把"计划假设"对到"代码现实"，逐文件核对结论：

| 计划假设 | 代码现实 | 影响 |
|---|---|---|
| 需为 bidirectional 重写 attention kernel | 基础能力部分具备：`flash_attn.py` 有 `causal=False` 路径；RoPE→FlashAttention 是两个 kernel 的 pipeline，并非融合 op | 仍需把 `is_causal` 透传到各后端，并在目标 shape 上做前向/反向数值测试与显存测试；旧 file:line 仅作线索 |
| "100% 复用 `fused_linear_ce`" | 训练栈实际用 **`fused_ce_zloss`**；`fused_linear_ce` 是 V=256 roofline 证伪的反面案例（`ops/__init__.py:35` 明确不在栈内） | 复用的是 `fused_ce_zloss`；且它内部 `mean()` 全部 M 行，masked CE 要先 gather 被 mask 行或加 per-row mask |
| 改 attention mask 即可 | NTP shift 渗透全栈：`igpt.forward` 的 `tokens[:,:-1]/[:,1:]`、`_channel_indices`/`_position_ids` buffer（长度 `seq_len-1`，`igpt.py:80-82`）、`_embed_inputs` 定长断言（`igpt.py:139`）、`coarse_ctx[:,1:]`（`cc_igpt.py:141`）、demo `_make_heatmap_b64`、`evaluate._per_image_bpd` | masked 路径必须是**不移位的并行 forward 分叉**（`forward_masked`），而非在 AR forward 上打补丁 |

**一个隐藏对齐陷阱**：`flash_attn.py:698` 断言 `causal or PAD==0` —— 非因果注意力要求 seq_len 已对齐到 64（否则 pad key 无 mask 屏蔽会污染 softmax）。CIFAR fine=3072(=48×64)、IN64 fine=12288(=192×64) 本身对齐，**但前提是 masked 模型吃完整未移位序列**；AR 模型吃的是 `seq_len-1=3071`（需 PAD=1）。这恰是"必须不移位分叉"的又一理由：做对了对齐自动满足，沿用 `-1` 旧习惯则断言命中。

#### E3.1 核心机制：从 Causal AR 到 Bidirectional Masking

1. **保留离散 Token 空间**：状态空间严格保持 `V=256` 的 `uint8` categorical tokens。
2. **双向注意力**：fine 分支走 `is_causal=False`（kernel 已支持，见 §3.0），coarse 分支保持 AR（仅 64 token，AR 解码很便宜；加速目标是 fine 3072/12288 → K）。
3. **`[MASK]` token 与权重共享**：新增第 257 行输入 embedding 作 `[MASK]`，**输出头仍 256-way softmax**（MASK 永不作 target）—— 这是输入 embedding 改动，**不是输出头改造，不触 DMoL 红线**。但 head 与 `token_embed.weight` 的 weight-tying 需提前决定：tie 到 `token_embed.weight[:256]` 切片在 PyTorch 里别扭，或用固定（非学习）MASK 向量。叙事坚持"256-way categorical softmax 不变"。
4. **确定性 masking schedule**：encoder/decoder 共享同一去 mask 顺序（`raster_chunks`/`checkerboard`/`pyramid` 均为数据无关固定顺序 → OK）。LLaDA 式 confidence-based 去 mask 是数据相关的，只有 encode/decode 共享逐位相同 logits 才可解 —— 第一版 bit-exact 路径先不上 confidence 排序。

#### E3.2 Bit-exact K-step 解码

这是整条路线成败的核心，`verify_lossless`（§6.3）的经验直接迁移：

- **每组一次 forward**：解一组 G 个 masked token 时，对 `[已揭示 token + MASK]` 跑**一次** forward 得到 G 个**边缘分布** `p(x_i | 已揭示)`，逐 token 送算术编码器（编各自的边缘）。decoder 给定相同的已揭示集合跑**同一次** forward → 得到相同 G 个边缘 → 全部解出。**这就是加速来源（1 forward/组）与 bpd 代价（组内条件独立性 gap）的来源**。
- **数值纪律照旧但不作跨平台保证**：encode/decode 共用同一个 masked logits 路径、禁用 autocast，并记录模型/软件/硬件 hash。相同输入和代码仍需实测 CDF 是否逐项一致；最终可移植格式应考虑显式 logits 量化/整数 CDF 规范或 CPU reference。不能从“无 atomics”直接推出所有 GPU/版本 bit-exact。
- **bpd 口径诚实**：固定 schedule 下 grouped factorization 的理想 NLL 与真实 coder rate单列；随机时间 MDLM-NELBO另列。K=T 只表示每组一个 token、消除组内独立近似，不保证新双向模型与旧 AR checkpoint 拥有相同 NLL。K 小的码率 gap 与 wall-clock 一律实测。**贡献候选是 frontier 形状，不是预设更低 bpd。**

#### E3.3 代码改造清单

> **实现状态（2026-08-04）**：CPU 协议地基已完成。现有代码包含可序列化/哈希的 `MaskSchedule.raster_chunks`、独立 `MaskedIGPT.forward_masked`、`CCMDLM` fixed-group objective、组内概率冻结的 arithmetic codec，以及 100 个合成样本和 tiny 模型的文件级 roundtrip/NLL 对账。AR schema v5 evaluator、identity-v2 roundtrip manifest 和 AutoDL 正式重评 runner 已就绪；CIFAR-10/ImageNet64 teacher-forced 评测已完成，CIFAR-10 2 张 GPU sequential roundtrip 已 `verified_on_subset`，ImageNet64 roundtrip 仍未执行。无泄漏 probe 与 CUDA backend/profiling 仍保持 pending。因此没有 CC-MDLM bpd/速度结论，也未启动 E3 训练。

仅在 CIFAR-10 上，按低风险顺序：

1. **模型层 (`models/igpt.py` / `cc_igpt.py`)**
    *   新增 `[MASK]` embedding（257 行）+ 决定 weight-tying 处理。
    *   写**不移位**的 `forward_masked`（输入完整 T、`is_causal=False`、可选 timestep/mask-ratio embedding）；不要改 AR `forward`。
    *   `MultiHeadAttentionBlock` / `GPTBlock` 透传 `is_causal` 到三后端（RoPE→FlashAttention pipeline / `TritonAttention.apply` / `F.sdpa`）。
2. **训练脚本 (`scripts/train.py`)**
    *   新增 `masking_scheduler.py`：`raster_chunks`（3072→K 组）先行作 sanity baseline。
    *   先实现与固定 groups 对齐的 conditional CE；MDLM 的 mask-ratio 重加权 NELBO 作为独立配置，输出字段不得与 schedule NLL 混名。
3. **解码与验证 (`scripts/verify_lossless.py`)**
    *   **先做 K=T sanity**：验证新 grouped codec 自身 encode/decode token 完全一致；不要求它与旧 AR 模型码率相等。
    *   再实现 K-step 分组解码，输出 §0 六层 rate 指标与 fine/total forward 次数，画 `file bpd vs calls vs wall-clock`。
    *   全程 WSL 静态可验证（pytest cpu + py_compile），首跑 forward 在 AutoDL。

#### E3.4 推荐的 Masking Schedules
| Schedule | 机制描述 | 预期优势 |
| :--- | :--- | :--- |
| **raster_chunks** | 将 3072 tokens 按原 AR 序列切成 K 段，逐段 Mask | 最容易实现，作为 Sanity Baseline |
| **checkerboard** | 先预测棋盘 anchor，再补洞 | 图像局部空间条件更强，视觉连贯性好 |
| **pyramid** | 8×8 coarse → 16×16 mid → 32×32 fine groups | 最贴合 CC-iGPT 叙事，天然支持多尺度 |
| **channel_refine** | 先 R/Y，再 G/B 或 chroma | 可能降低色彩条件熵，利用通道相关性 |

---

### E4. 论文叙事与答辩边界

只有 E0–E4 的真实实验通过后，论文才可以采用以下叙事：

1. **从“极致压缩”到“帕累托最优”**
    *   **旧结果**：`2.8296` 是 ensemble+TTA 诊断协议下的历史实验结果，可以按该协议引用，但不能与单模型 codec 成本混写；当前 CIFAR-10 single/no-TTA teacher-forced 结果为 `2.8328 bpd`。
    *   **候选新叙事**：报告离散图像无损压缩的 **`完整文件 bpd / 总 forward calls / 实测延迟`** 前沿，并逐点给出真实 roundtrip。只有测得的 rate gap 和 speedup 才能进入摘要。
2. **MDL 理论的泛化**
    *   展示显式 grouped factorization 如何把 masked prediction 转成可解码的离散 code；NELBO 与实际文件码长的差异本身也是结果。
3. **工程严谨性的延续**
    *   对 RGB 预处理后像素做真实算术文件 roundtrip，并完整报告容器开销、CDF 量化和运行环境；不对未实现的连续 codec 作价值判断。

---

### E5. 时间线与 Gating 策略

*   **Week 1**: 精读 MDLM / ARDM compression 段；写出 §0 的概率因式分解、schedule/container schema 和指标定义，完成 design review。
*   **Week 2**: 不训练大模型，先用 tiny model 完成 `raster_chunks` K=T 与 K<T 的文件 roundtrip、错误注入和逐 bit 对账（Gate E1/E2）。
*   **Week 3**: 仅在前两 gate 通过后训练 CIFAR toy，对比 fixed-group CE 与 MDLM-NELBO；跑 K sweep 和至少 3 seeds。
*   **Week 4**: 依据实测结果决定 checkerboard/pyramid；生成 rate/calls/time 图与 result manifests。未通过 gate 则记录负结果，不扩到 IN64。

> **⚠️ 红线警告**：
> 1. 连续 VDM/latent codec 必须作为指标与协议不同的独立分支，不能复用无损主表结论。
> 2. Phase E 首版保持 RGB uint8 token；“pixel exact”限定为输入转 RGB 后的像素，不是原 PNG/JPEG 文件字节。
> 3. 所有 diffusion 实验先通过 tiny codec 与 CIFAR gate，不直接上 ImageNet64。

---

## 10. 读博方向衔接（本项目 → 研究方向；2026-06-20）

> 用途：报考 / 交流时把本项目讲到研究方向的语言里。**只保留方向衔接，不点名具体老师 / 组**（任职、招生、篇目均需自行在 Scholar / 组主页核实，不在此断言）。

### 10.1 三个可让不同方向眼前一亮的卖点

1. **diffusion / 生成模型方向** → 当前可诚实表述为："已有可解码 AR 无损基线，CIFAR-10 与 ImageNet64 分别有 2.8328/3.4812 bpd 的正式 teacher-forced 单模型 manifest；下一阶段研究显式 grouped masked factorization，先验证 NELBO、schedule NLL 与实际文件 bpd 的差异，再画 rate/calls/time 前沿。" `2.8296/3.4800` 作为明确标注协议的历史 ensemble+TTA 实验结果，不冒充单模型或 MDLM 实测；`2.8328/3.4812` 也不冒充实际 arithmetic file bpd。
2. **高效系统 / 编译 / 量化方向** → “训练栈含 6 个手写 Triton kernel；RoPE→FlashAttention 是两个 kernel 的 pipeline，不冒充额外 fusion；`fused_linear_ce` 是独立反面案例。基准 harness 已修正，旧 H800 数字待重跑后再作性能结论。”
3. **信息论 / 编码方向** → "项目名源于 MDL；当前指标是预共享模型下的条件码长。手写 WNC 算术编解码做到 **bit-exact roundtrip**，并显式计入首 token、CDF 量化、padding 与容器开销。"

### 10.2 衔接论文清单

- **diffusion 加速线**：DPM-Solver / -++ / -v3、Analytic-DPM（§2.6）—— 理解"少步采样"，对接 K 步并行解码同轴思路。
- **离散 / masked diffusion 理论线**：RADD、MDLM、D3PM、VDM（§2.1/2.2/2.6）—— absorbing=any-order，本项目 fine 分支的合法性根基。
- **大规模验证线**：LLaDA、Block Diffusion（§2.6）—— 方向被验证 + 与本项目结构同形。
- **信息论 / 压缩线**：Shannon、Rissanen/Grünwald、Language-Modeling-is-Compression、Bits-Back（§2.1）—— MDL 立论。

---

## 11. 从「压缩即预测」到「超越压缩」阅读路径

> 本节原为 README 中的「论文阅读路径」，迁入 future.md 保持 README 公开页面简洁。  
> 路径按「从你已经在做的事，一步步往外走」排列，难度 ★（好读）到 ★★★★★（硬核）。  
> **怎么用**：第 1–3 阶是项目地基，建议读透；第 4–5 阶回答「压缩不够在哪」；第 6 阶是远景选读，不影响项目推进。看不下去可以停在任何一阶——往后每一阶都是可选的，不是欠账。

### 11.1 地基：何时 CE 对应条件码长

| 论文 | 它回答什么 | 难度 |
|---|---|---|
| Shannon 1948, *A Mathematical Theory of Communication* | 熵 = 最优码长；CE 就是「比最优多花的比特」 | ★★ |
| Rissanen 1978, *Modeling by Shortest Data Description* (MDL) | 「最短描述」为什么是模型选择原则——本项目命名的由来 | ★★ |
| Delétang et al. 2024, *Language Modeling Is Compression* (arXiv:2309.10668) | 把命题做成实证：LM 当算术编码器压 PNG/FLAC。**读这篇就懂了整个项目的世界观** | ★★ |
| Blier & Ollivier 2018, *The Description Length of Deep Learning Models* | 把 MDL 和 SGD 训练接起来（theory.md 已引） | ★★★ |

### 11.2 架构谱系：自回归像素建模与扩散替代路线

> iGPT / CC-iGPT 是 AR 线的直接延续；CC-MDLM 是离散 masked likelihood 的并行替代路线。它与连续 score/SDE diffusion 有共同的逐步去噪直觉，但不是把连续 SDE 简单离散化。

| 论文 | 它回答什么 | 难度 |
|---|---|---|
| van den Oord et al. 2016, PixelRNN / PixelCNN | 逐像素 p(x_t \| x_{<t}) 的开山 | ★★ |
| Salimans et al. 2017, PixelCNN++ | 子像素 AR + 通道条件的直接来源 | ★★★ |
| Chen et al. 2020, iGPT | 「拿 GPT 压像素」= Phase A | ★★ |
| Child et al. 2019, Sparse Transformer | 主表一直在追的 2.80 baseline | ★★★ |
| Song et al. 2021, *Score-Based Generative Modeling through SDEs* (arXiv:2011.13456) | AR 的非自回归替代路线：连续扩散 = 去噪过程；读此篇理解「为什么选离散 masked diffusion 而非连续 SDE」——是 Phase E 出发点的反面参照 | ★★★★ |

### 11.3 压缩与智能：正方与反方

> 项目押注「压缩 = 预测 = 智能」。这一阶是这个命题的正反辩论——也是 linear probe 结果为什么有意义。

| 论文 | 它回答什么 | 难度 |
|---|---|---|
| Sutskever 2023, *An Observation on Generalization*（Simons 讲座）| 无监督学习就是压缩——正方，最好懂的入口 | ★ |
| Huang et al. 2024, *Compression Represents Intelligence Linearly* (arXiv:2404.09937) | 31 模型 × 12 benchmark：能力 ≈ 线性正比压缩效率。**和 bpd 指标直接对话** | ★★ |
| Ted Chiang 2023, *ChatGPT Is a Blurry JPEG of the Web* | 反方：有损压缩漂亮的插值正是它不可靠的根源（通俗）| ★ |
| Legg & Hutter 2007, *Universal Intelligence* (arXiv:0712.3329) | 把「压缩先验 = 智能」写成一个方程——正方的理论顶点 | ★★★★ |
| Voita & Titov 2020, *Information-Theoretic Probing with MDL* | linear probe「压缩逼出结构」的直接证据（theory.md 已引）| ★★★ |

### 11.4 压缩的边界一：典型性不等于有效性

> 模型最小化 CE = 匹配像素分布。这一阶讲这个目标何时会骗你——对**图像补全 / inpainting 面板**尤其直接。

| 论文 | 它回答什么 | 难度 |
|---|---|---|
| Holtzman et al. 2020, *The Curious Case of Neural Text Degeneration* (arXiv:1904.09751) | 最大化模型自身似然 → 平淡重复；必须截掉高概率尾巴。**MLE 优化「典型」不是「对」的最干净证据** | ★★ |
| Gudibande et al. 2023, *The False Promise of Imitating Proprietary LLMs* (arXiv:2305.15717) | 模仿一个分布只抓住表面、能力会封顶（选读）| ★★ |

### 11.5 压缩的边界二：一次前向的计算上限

> 模型是**固定深度 Transformer、每 token 一次前向**，有没有算力上限？这一阶是架构表达力的理论边界。

| 论文 | 它回答什么 | 难度 |
|---|---|---|
| Merrill & Sabharwal 2022, *Saturated Transformers are Constant-Depth Threshold Circuits* (arXiv:2106.16213) | 现实 Transformer ⊆ TC⁰：定深做不了本质串行的问题 | ★★★★ |
| Feng et al. 2023 (arXiv:2305.15408) · Li et al. 2024 (arXiv:2402.12875) | 给了「中间步 (CoT)」就能突破——深度省不掉，只能在别处付账 | ★★★ |

### 11.6 远景选读：分布匹配与计算的物理

> 到这里已离图像压缩很远。这些是「如果压缩不是终点，判据该长什么样」——**读着玩，别有压力**。

| 论文 | 它回答什么 | 难度 |
|---|---|---|
| LeCun 2022, *A Path Towards Autonomous Machine Intelligence* | 用能量最小化 + 世界模型替代逐像素预测（JEPA）| ★★★ |
| Bengio et al. 2023, *GFlowNet Foundations* (arXiv:2111.09266) | 不喂数据分布，而是给判据 (reward) 按「流」采样——「非 MLE」的操作化 | ★★★★ |
| Mézard & Montanari 2009, *Information, Physics, and Computation* + Zdeborová & Krzakala 2016 (arXiv:1511.02476) | 把学习 / 推理重述为物理相变（统计物理视角）| ★★★★★ |
| Aaronson 2005, *NP-complete Problems and Physical Reality* · Bonifaci et al. 2012, *Physarum can compute shortest paths* | 「物理能不能替你算」的乐趣与冷水 | ★★★ |

**一句总纲**：第 1–3 阶巩固「压缩即智能」，第 4–5 阶看清它的两条边界（**典型性、深度**），第 6 阶探索边界之外。项目稳稳站在 1–3 阶——往后每一阶都是可选的，不是欠账。
