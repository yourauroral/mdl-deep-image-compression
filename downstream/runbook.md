# CC-iGPT 下游任务 Runbook

论文下游任务执行手册。当前保留的下游代码只覆盖三类证据：

> **压得越好 -> 学得越好。**
> - **表征质量**：linear probe；IN64 权重可用 `--probe_dataset cifar10` 做 transfer probe。
> - **生成能力**：图像补全，展示条件采样能力。
> - **真实可解性**：算术编解码 roundtrip，证明 bpd 对应真实可逆 bitstream。

`run_downstream.sh` 只调度保留的 `pre / complete / verify` 步骤，不再生成静态下游 JSON。

---

## 前提

- 在 AutoDL GPU 上跑补全和 roundtrip；WSL 只适合跑 `--self_test`。
- 全部复用 CIFAR v2 的 `best.pth`，不修改、不重训任何模型。
- 这些任务不会写训练目录或 checkpoint；只会占用同一张卡的显存/算力。你不暂停 IN64 训练也可以同步代码，但不要在训练进程旁边额外启动这些 GPU 任务。

## 环境变量

```bash
cd ~/work/mdl-deep-image-compression          # 改成你的 AutoDL repo 路径
CFG=configs/ccigpt_cifar10_s_rgb_ronly_v2.yaml
CKDIR=experiments/ccigpt_cifar10_s_rgb_ronly_v2/checkpoints
BEST=$CKDIR/best.pth
```

## 执行顺序

```bash
# 一键跑保留的 ready 步骤：pre -> complete -> verify
bash downstream/run_downstream.sh

# 单跑某步
bash downstream/run_downstream.sh pre
bash downstream/run_downstream.sh complete
bash downstream/run_downstream.sh verify
```

`pre` 是 coder 自检；显式跑 `complete` 或 `verify` 时脚本也会先执行一次 `pre`。

---

## [0] 预检 self_test（秒级，无需 GPU/ckpt）

```bash
python scripts/verify_lossless.py --self_test
```

预期包含：

```text
[self_test] roundtrip bit-exact: True
```

如果看到负 overhead，不要把它当真实压缩效率。self_test 的符号是均匀随机抽样，大量落在模型概率尾部，而 coder 频数有量化地板；它只是压力测试。真实图像 roundtrip 应看 `[2]` 的 achieved bpd 与 NLL overhead。

## [1] 图像补全 demo

```bash
python scripts/complete_image.py --config $CFG --checkpoint $BEST \
    --num_images 4 --keep_frac 0.5 --temperature 1.0 --top_k 100 \
    --out experiments/completion_grid.png
```

输出 PNG 网格，每行是 **原图 | 已知上半(灰=待补) | 补全**。

说明：CC-iGPT 的 coarse ctx 由整图缩略图算，所以语义是“低分缩略图 + 上半真实像素 -> 补下半”。coarse 是压缩时独立 bitstream 的显式 side-channel，不是纯粹只看上半的 inpainting。仅 CIFAR v2 实用；IN64 fine token 太长，不适合实时补全。

## [2] 真实可解性 roundtrip

```bash
python scripts/verify_lossless.py --config $CFG --checkpoint $BEST --num_images 2

# 可选：落盘自包含 MDLC .bin，再只读解析
python scripts/verify_lossless.py --config $CFG --checkpoint $BEST --num_images 2 \
    --dump_dir experiments/bitstreams
python scripts/verify_lossless.py --inspect experiments/bitstreams/img0.bin
```

预期看到 bit-identical 断言通过，并报告实际码长、achieved bpd、teacher-forced NLL bpd 与量化/收尾 overhead。论文里优先引用“逐像素 bit-identical + overhead vs NLL 很小”作为真实无损证据。

这里慢是因为模型没有 KV-cache，每个 token 都要跑一次完整 forward。属预期，不是卡死；`--log_every 512` 会打印进度。

## [3] IN64 -> CIFAR transfer probe（✅ 完成 2026-06-07，完整 32 层 best L16 = 73.19%）

```bash
python scripts/linear_probe.py \
    --config configs/ccigpt_imagenet64_v1.yaml \
    --checkpoint experiments/ccigpt_imagenet64_v1/checkpoints/best.pth \
    --probe_dataset cifar10 --probe_data_root datasets/ --layers all
```

IN64 模型 `image_size=64`，CIFAR 图会 resize **32->64**。这个结果只能看 transfer 趋势，不能和 CIFAR-10 native 32x32 的 66.93/79.33 直接横比。
**实测：完整 32 层 best L16 = 73.19%**（中层峰，L15–L17 为 73.12/73.19/73.18 平台；L0 41.69%→L16 73.19%→L31 64.79%），胜 iGPT-S native 66.93% (+6.3pp)、低于 CC-iGPT v2 native 79.33%（transfer + resize 域偏移代价）。
注：probe `--batch_size` 默认已改自适应（IN64 fine seq=12288 → batch 64），不指定即可，避免长序列下 batch 256 触发 int32 偏移溢出 → CUDA illegal memory access。

---

## Demo 前端

保留的相关端点：

| 功能 | 端点 | 类型 |
|---|---|---|
| 图像补全 | `POST /api/complete` | 实时采样 |
| 真实编码 | `POST /api/encode` | 流式 NDJSON |
| 只读解析 | `POST /api/inspect` | 无需 GPU |
| 真实解码 | `POST /api/decode` | 流式 NDJSON |

启动：

```bash
uvicorn demo.server:app --host 0.0.0.0 --port 6006 --reload
```

AutoDL 训练不中断时，建议只访问静态面板；`/api/complete`、`/api/encode`、`/api/decode` 都会额外占 GPU。

## 结果归档建议

- `experiments/completion_grid.png`：放论文定性补全面板。
- `verify_lossless.py` 控制台输出：引用 bit-identical、achieved bpd 与 overhead。
- `.bin` 文件 + `--inspect` 输出：答辩时展示真实文件可自描述解析。
