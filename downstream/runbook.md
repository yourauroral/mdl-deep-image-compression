# CC-iGPT 下游任务 Runbook

论文 §5 下游任务的执行手册。一条 MDL 主线把四件事串起来：

> **压得越好 ⇒ 学得越好。** 同一个 CC-iGPT 权重，从四个侧面验证它学到的不是
> 数据集记忆而是自然图像的通用统计：
> - **表征质量** → linear probe（各层 hidden 的线性可分性）
> - **密度估计** → OOD typicality（能否识别分布外）
> - **泛化** → 跨数据集 bpd（换一个图像集仍压得动）
> - **生成** → 图像补全（条件采样补全缺失像素）
>
> 而 **`verify_lossless`** 是这条线的地基：它证明我们报的 bpd 不是 proxy，而是
> 一段真实可逆的 bitstream 码长（算术编解码 roundtrip，逐像素 bit-identical）。

---

## 前提

- **在 AutoDL GPU 上跑。** WSL 无 GPU，只有两个 `--self_test` 能本地跑（纯数学/coder）。
- 全部复用 CIFAR v2 的 `best.pth`，**不修改、不重训**任何模型。
- 这些任务与 IN64 训练**共享同一张卡的显存/算力**。稳妥起见在 IN64 某个 epoch 落盘
  的间隙跑，或先确认显存余量够再开。它们不会写训练目录，但会抢资源。

## 环境变量

```bash
cd ~/work/mdl-deep-image-compression          # 改成你的 AutoDL repo 路径
CFG=configs/ccigpt_cifar10_s_rgb_ronly_v2.yaml
CKDIR=experiments/ccigpt_cifar10_s_rgb_ronly_v2/checkpoints
BEST=$CKDIR/best.pth
```

## 执行顺序（cheap → expensive，让失败尽早暴露）

`[0] self_test → [1] OOD → [2] 跨数据集 → [3] 补全 → [4] roundtrip`

一键全跑：`bash downstream/run_downstream.sh`
单跑某步：`bash downstream/run_downstream.sh ood`（见脚本顶部用法）

---

## [0] 预检 self_test（秒级，无需 GPU/ckpt）

```bash
python scripts/verify_lossless.py --self_test
python scripts/ood_detect.py --self_test
```

**预期：**
```
[self_test] roundtrip bit-exact: True
[self_test] coded 23510 bit / ideal 24478.4 bit / overhead -3.96%
[self_test] AUROC/typicality 数学 OK (裸 bpd AUROC=0.002 → typicality=0.995)
```

**关于那个负 overhead（-3.96%）—— 不是 bug，别误读：**
self_test 的符号是**均匀随机**抽的（`random.randrange(V)`），大量落在 `p` 的尾部，
`-log2(p[s])` 高达 30–50 bit；而 `build_cumfreq` 给每个符号兜底 `freq≥1=1/2¹⁶`，
coder 对尾部符号最多花 16 bit。于是 `ideal`（按全精度 `p` 算）被尾部抬高，coded
被量化地板压低 → coded < ideal → 负数。它其实是个**好压力测试**：证明 roundtrip
连近零概率符号都能 bit-exact 还原。

⚠ **真实图像 [4] 不会出现这个**：真实像素 token 由训练好的模型赋予正常概率，overhead
是小**正**数。答辩别拿这行 -3.96% 当效率指标（看着像违反 Shannon），用 [4] 的数字。

---

## [1] OOD typicality（~1–3 min）

```bash
python scripts/ood_detect.py --config $CFG --checkpoint $BEST \
    --ood svhn,cifar100 --ref_images 2000 \
    --json_out demo/data/ood.json
```

**预期：** 一张三列 AUROC 表（`raw_bpd` / `typ_total` / `typ_dualscale`）。
- `raw_bpd` 对 **SVHN 很可能 < 0.5** —— 复现 Nalisnick（CIFAR 训的模型给 SVHN 更低 bpd），
  脚本会打印 `⚠ 复现 Nalisnick` 提示。
- `typ_total`、`typ_dualscale` 把它救回 > 0.5；**`typ_dualscale`（双尺度联合）是本工作差异化**，
  单尺度模型给不出 coarse/fine 两个独立信号。
- `--json_out` 回填前端 **Panel 7（/api/ood）** 的 AUROC 分组柱状图（含 0.5 随机基线虚线）。

**论文用途：** 论证"密度估计"侧面，且顺带展示一个反直觉的文献坑 + 你的修正。

---

## [2] 跨数据集泛化 bpd（各 ~1–2 min，in-domain + 三个 override）

```bash
# in-domain 基线（单 ckpt 无 ensemble/TTA，与各 override 同协议可比，作前端参照柱）
python scripts/evaluate.py --config $CFG --checkpoint $BEST --json_out demo/data/transfer.json
python scripts/evaluate.py --config $CFG --checkpoint $BEST --dataset_override cifar100 --json_out demo/data/transfer.json
python scripts/evaluate.py --config $CFG --checkpoint $BEST --dataset_override svhn     --json_out demo/data/transfer.json
python scripts/evaluate.py --config $CFG --checkpoint $BEST --dataset_override stl10    --json_out demo/data/transfer.json
```

**预期：** 每次一张 bpd 表 + `[跨数据集] override → …` 的可比性 caveat 行
（SVHN `split='test'`；STL-10 96→32 下采样）。`--json_out` 按数据集 key upsert 到同一
JSON，四次跑共同填满前端 **Panel 8（/api/transfer）** 的水平条形图（in-domain 蓝柱为参照）。

⚠ **可比性：** 含重采样，**勿与各集官方 bpd 横比**，只看"同一模型在不同集上的相对值"——
CIFAR-10 训的模型在别的自然图像集上 bpd 仍合理，即学到的是自然图像统计而非记忆。

**论文用途：** 论证"泛化"侧面。

---

## [3] 图像补全 demo（~30s/图 → 4 张约 2 min）

```bash
python scripts/complete_image.py --config $CFG --checkpoint $BEST \
    --num_images 4 --keep_frac 0.5 --temperature 1.0 --top_k 100 \
    --out experiments/completion_grid.png
```

**预期：** `image k: done` ×4，存 PNG 网格，每行 = **原图 | 已知上半(灰=待补) | 补全**。

⚠ **诚实声明（写进论文，别藏）：** CC-iGPT 的 coarse ctx 由**整图**缩略图算，所以本 demo
语义是"低分缩略图 + 上半真实像素 → 补下半"，coarse 是显式 side-channel（与压缩时
独立 bitstream 同源），**不是偷看答案**。纯补全需让 coarse 也只看上半（未做，注释存档）。
仅 CIFAR v2 实用（fine 3072 token）；IN64 12288 token 太慢。

**论文用途：** 论证"生成"侧面（定性面板，Sparse Transformer 同款下游）。

---

## [4] 真实可解性 roundtrip（最慢，~2 min/图 → 2 张约 5 min；实测 865s）

```bash
python scripts/verify_lossless.py --config $CFG --checkpoint $BEST --num_images 2

# 可选：bitstream 落盘为自包含 MDLC .bin（写后读回断言 bit 一致）
python scripts/verify_lossless.py --config $CFG --checkpoint $BEST --num_images 2 \
    --dump_dir experiments/bitstreams
# 只读解析某个 .bin（结构/hex/码长/bpd，无需 GPU/ckpt/模型）
python scripts/verify_lossless.py --inspect experiments/bitstreams/img0.bin
```

**预期（2026-05-30 实测）：**
```
✅ bit-identical
码长: 9502 bit  (1188 byte)  [coarse 445 + fine 9057]
achieved bpd = 3.0931  (理论 NLL bpd 3.0896, 模型 teacher-forced bpd 3.0876)
量化+收尾 overhead vs NLL: 0.11%        # 小正数
（img1: 2.3216 bpd, overhead 0.24%）
== 结果：2/2 张 bit-identical 还原 ==
# --dump_dir 时额外：bitstream → experiments/bitstreams/img0.bin (1205 byte) 文件读回 bit 一致 ✅
```

⚠ **逐图 bpd 随图像复杂度变**（img0 3.09 / img1 2.32），不是某张图的绝对值有意义，关键是
**2/2 可逆 + overhead vs NLL 仅 0.1–0.2%** → 证明报的 bpd = 真实可逆码长、算术编码近最优。

这里慢是因为模型**无 KV-cache**，每个 token 跑一次完整 forward（encode + decode 各一遍）。
属预期，不是卡死；`--log_every 512` 会打印进度。

**`.bin` 容器（`--dump_dir`/`--inspect`）：** 自包含 MDLC 格式（16B header + coarse/fine 算术编码字节），
仅凭文件即可读出尺寸/码长/bpd（`--inspect` 不依赖模型）。答辩三连：`ls -l` 看真文件 →
`--inspect` 无模型读元数据 → roundtrip decode 回 bit-identical 原图。

**论文用途：** 整条 MDL 主线的地基 —— 证明 bpd = 真实可逆码长，区别于只报 bpd 的工作。

---

## [5] transfer probe（**待 IN64 ckpt 就绪再跑**）

```bash
# IN64 训练出 best.pth 后：
python scripts/linear_probe.py \
    --config configs/ccigpt_imagenet64_v1.yaml \
    --checkpoint experiments/ccigpt_imagenet64_v1/checkpoints/best.pth \
    --probe_dataset cifar10 --probe_data_root datasets/ --layers all
```

⚠ IN64 模型 `image_size=64`，CIFAR 图被 resize **32→64**，**与 32-native 的
66.93/79.33 不是同协议**（输入分辨率不同），只能作 transfer 趋势看，勿直接横比。
（IN64→IN64 的 1000-way native probe 需 label-保留的 prepare，当前数据管线丢标签，
见 `future.md §6.1`，列为 IN64 训完后的任务。）

---

## [6] Demo 前端面板（下游任务可视化，复用上面跑出的数字）

下游任务的前端在 demo 里分两类（详见 `future.md §6.6`）：

| Panel | 数据来源 | 类型 |
|---|---|---|
| 7 OOD typicality | `/api/ood` ← `demo/data/ood.json`（[1] `--json_out` 回填） | 静态 JSON |
| 8 跨数据集 bpd | `/api/transfer` ← `demo/data/transfer.json`（[2] `--json_out` upsert） | 静态 JSON |
| 9 图像补全 | `POST /api/complete`（实时采样，~20–40s/图） | 实时交互 |

- **Panel 7 / 8**：跑完 [1][2] 后 `demo/data/{ood,transfer}.json` 的 `generated`
  时间戳被填上、各行数值就位；前端自动从"待 AutoDL 跑"占位切到图表。**不跑也不报错**
  （占位 JSON 已 checkin，面板显示 pending 提示）。
- **Panel 9**：与 [3] 同源（复用 `complete_image._complete_one`），但上传任意图即时补全，
  可调 keep%/温度。无 KV-cache → 单次 ~20–40s，**与 IN64 训练共享 GPU，挑空窗用**。

启动（AutoDL，仅 6006/6008 端口可公网映射）：

```bash
uvicorn demo.server:app --host 0.0.0.0 --port 6006 --reload
```

---

## 结果归档建议

- [1][2] 控制台表格 → 截图 / 复制进论文 §5 表；`--json_out` 同时落 demo 前端 Panel 7/8。
- [3] `experiments/completion_grid.png` → 论文定性面板 / 答辩 demo；Panel 9 是其交互版。
- [4] 控制台 `✅ 2/2 bit-identical` + achieved bpd → §5 可解性硬证据（前端 Panel 2 交互版）。
- 跑完把关键数字回填到论文 §5 与 `future.md §6`。
