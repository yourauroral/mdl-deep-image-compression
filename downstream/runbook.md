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
    --ood svhn,cifar100 --ref_images 2000
```

**预期：** 一张三列 AUROC 表（`raw_bpd` / `typ_total` / `typ_dualscale`）。
- `raw_bpd` 对 **SVHN 很可能 < 0.5** —— 复现 Nalisnick（CIFAR 训的模型给 SVHN 更低 bpd），
  脚本会打印 `⚠ 复现 Nalisnick` 提示。
- `typ_total`、`typ_dualscale` 把它救回 > 0.5；**`typ_dualscale`（双尺度联合）是本工作差异化**，
  单尺度模型给不出 coarse/fine 两个独立信号。

**论文用途：** 论证"密度估计"侧面，且顺带展示一个反直觉的文献坑 + 你的修正。

---

## [2] 跨数据集泛化 bpd（各 ~1–2 min，跑三次）

```bash
python scripts/evaluate.py --config $CFG --checkpoint $BEST --dataset_override cifar100
python scripts/evaluate.py --config $CFG --checkpoint $BEST --dataset_override svhn
python scripts/evaluate.py --config $CFG --checkpoint $BEST --dataset_override stl10
```

**预期：** 每次一张 bpd 表 + `[跨数据集] override → …` 的可比性 caveat 行
（SVHN `split='test'`；STL-10 96→32 下采样）。

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

## [4] 真实可解性 roundtrip（最慢，~2 min/图 → 2 张约 5 min）

```bash
python scripts/verify_lossless.py --config $CFG --checkpoint $BEST --num_images 2
```

**预期：**
```
✅ bit-identical
码长: ... bit  (... byte)  [coarse ... + fine ...]
achieved bpd = 2.83xx  (理论 NLL bpd 2.83xx, 模型 teacher-forced bpd 2.83xx)
量化+收尾 overhead vs NLL: 0.xx%        # 小正数
== 结果：2/2 张 bit-identical 还原 ==
```

这里慢是因为模型**无 KV-cache**，每个 token 跑一次完整 forward（encode + decode 各一遍）。
属预期，不是卡死；`--log_every 512` 会打印进度。

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

## 结果归档建议

- [1][2] 控制台表格 → 截图 / 复制进论文 §5 表。
- [3] `experiments/completion_grid.png` → 论文定性面板 / 答辩 demo。
- [4] 控制台 `✅ 2/2 bit-identical` + achieved bpd → §5 可解性硬证据。
- 跑完把关键数字回填到论文 §5 与 `future.md §6`。
