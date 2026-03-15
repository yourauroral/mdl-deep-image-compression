# MDL Deep Image Compression

基于最小描述长度（MDL）原则的深度图像压缩，支持 Hyperprior 与 iGPT 两种模型。

## 安装

```bash
pip install torch torchvision pyyaml numpy pillow tensorboard
```

## 使用

```bash
python scripts/train.py --config configs/igpt_cifar100_baseline.yaml

python scripts/train.py --config configs/hyperprior_mse.yaml
```

配置文件示例见 `configs/`，数据集放置于 `datasets/`。

## 特性

- Hyperprior / iGPT 双模型支持
- 混合精度训练（fp16 / bf16）与多 GPU 分布式
- TensorBoard 日志 + PSNR / SSIM / BPP 指标
- 配置文件驱动，零代码改动切换实验

## 状态

基础训练流程已验证，持续迭代中。欢迎 Issue 或 PR。