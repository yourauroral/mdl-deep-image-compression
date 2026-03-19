# MDL Deep Image Compression

基于自回归建模（iGPT）的深度图像无损压缩，使用 Transformer 学习像素序列的联合概率分布。

## 安装

```bash
pip install torch torchvision pyyaml numpy pillow tensorboard
```

## 训练

```bash
# 基础训练
python scripts/train.py --config configs/igpt_cifar100_baseline.yaml

# 从断点恢复
python scripts/train.py --config configs/igpt_cifar100_baseline.yaml \
    --resume experiments/igpt_cifar100_baseline/checkpoints/epoch_10.pth

# 多卡分布式
torchrun --nproc_per_node=4 scripts/train.py --config configs/igpt_cifar100_baseline.yaml
```

## 监控

```bash
tensorboard --logdir experiments/igpt_cifar100_baseline/logs --port 6006
```

## 实验对比

开启 MTP 辅助头时，复制 config 并修改以下字段：

```yaml
exp_name: "igpt_cifar100_mtp"
model:
  use_mtp: true
train:
  mtp_weight: 0.1
```

```bash
python scripts/train.py --config configs/igpt_cifar100_mtp.yaml
```

## 数据集

CIFAR-100 由 torchvision 自动下载，放置于 config 中 `data.train` 指定的目录。

## 特性

- 自回归 iGPT 模型（SwiGLU FFN、RoPE、QK-Norm、OLMo 2 post-norm）
- YCbCr 色彩空间输入，降低通道间冗余
- MTP 辅助预测头（可选，DeepSeek-V3 风格）
- Cosine decay + linear warmup 学习率调度
- 混合精度训练（bf16 / fp16）与多 GPU 分布式
- TensorBoard 日志，验证集 BPP 均值±标准差
- 配置文件驱动，`--resume` 断点续训
