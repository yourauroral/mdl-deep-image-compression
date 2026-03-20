# MDL Deep Image Compression

基于自回归 Transformer（iGPT）的深度图像压缩，对像素序列建模联合概率分布，使用交叉熵损失实现无损/近无损压缩。

## 环境要求

| 依赖 | 最低版本 | 说明 |
|------|----------|------|
| Python | 3.10+ | |
| PyTorch | **2.4+** | `torch.amp.autocast` 统一接口；`F.scaled_dot_product_attention` 需 2.0+ |
| torchvision | 0.19+ | 随 PyTorch 2.4 配套 |
| Triton | 3.0+ | PyTorch CUDA 安装自带，Triton kernel 可选 |
| CUDA | 11.8+ | Triton kernel 需要 GPU；CPU 下自动回退到 PyTorch 实现 |

## 安装

依赖手动安装：

```bash
pip install torch torchvision pyyaml numpy pillow tensorboard
pip install scikit-image  # 可选，用于 SSIM 计算
```

无需 `pip install -e`，直接从项目根目录运行脚本即可。

## 快速开始

### 训练

```bash
# 基础训练（CIFAR-100）
python scripts/train.py --config configs/igpt_cifar100_baseline.yaml

# 从断点恢复
python scripts/train.py --config configs/igpt_cifar100_baseline.yaml \
    --resume experiments/igpt_cifar100_baseline/checkpoints/epoch_10.pth

# 多卡分布式
torchrun --nproc_per_node=4 scripts/train.py --config configs/igpt_cifar100_baseline.yaml
```

### 测试

```bash
python tests/test_dataloader.py      # 验证数据加载和 YCbCr 转换
python tests/test_flash_attn.py      # 验证 Flash Attention kernel
python scripts/dryrun_forward.py     # 快速前向 sanity check
```

### 监控

```bash
tensorboard --logdir experiments/igpt_cifar100_baseline/logs --port 6006
```

## 架构

模型代码位于 `src/mdlic/`：

### iGPT (`models/igpt.py`)

自回归图像压缩。图像展平为像素序列，用 GPT Transformer 建模像素的联合概率分布。

- **输入预处理**：RGB → YCbCr（ITU-R BT.601），降低通道间冗余；每像素每通道作为独立 token（值域 0–255）
- **模型结构**：SwiGLU FFN、RoPE（base=500000）、QK-Norm、RMSNorm、OLMo 2 post-norm
- **损失**：Cross-Entropy + z-loss 正则化（权重 1e-4）
- **BPP 计算**：`BPP = CE_loss / ln(2) × C`（C 为通道数）
- **可选**：MTP 辅助预测头（DeepSeek-V3 风格，默认关闭）

### 共享层 (`models/layers.py`)

| 模块 | 说明 |
|------|------|
| `RotaryEmbedding` | RoPE 位置编码，base=500000（LLaMA 3 风格） |
| `MultiHeadAttentionBlock` | QK-Norm + RoPE + 可选 Triton Flash Attention |
| `RMSNorm` | Root Mean Square Normalization，可选 fused Triton kernel |
| `FeedForwardBlock` | SwiGLU 三线性门控 FFN |
| `GPTBlock` | MHA + FFN + OLMo 2 post-norm |

### Triton Kernels (`ops/`)

| 文件 | 说明 |
|------|------|
| `flash_attn.py` | 手写 Flash Attention v2，支持 causal mask、任意 seq_len、causal early termination |
| `ops/fused_rms_norm.py` | 手写 Fused RMSNorm，forward+backward 合并为单次 kernel launch |

所有 Triton kernel 均有 PyTorch 优雅回退，CPU 环境下自动切换。

## 配置系统

所有超参数在 `configs/*.yaml` 中定义：

| 字段 | 说明 |
|------|------|
| `model.type` | `igpt` |
| `model.d_model` | 隐层维度（默认 128） |
| `model.N` | Transformer 层数（默认 2） |
| `model.h` | 注意力头数（默认 4） |
| `model.d_ff` | FFN 隐层维度（默认 384） |
| `model.use_mtp` | 是否启用 MTP 辅助头（默认 false） |
| `train.lmbda` | Rate-distortion 权衡系数 |
| `train.amp_dtype` | `fp16` 或 `bf16` |
| `train.z_loss_weight` | z-loss 正则化权重（默认 1e-4） |
| `data.train` / `data.valid` | 数据集路径 |
| `checkpoint.save_dir` | 检查点保存目录 |

实验输出到 `experiments/{exp_name}/{logs,checkpoints}/`。检查点文件：`best_igpt.pth`、`epoch_*.pth`。

### 实验对比示例

开启 MTP 辅助头：

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

CIFAR-100 由 `torchvision.datasets.CIFAR100` 自动下载，放置于 config 中 `data.train` 指定的目录。

## 特性

- 自回归 iGPT 模型（SwiGLU FFN、RoPE、QK-Norm、OLMo 2 post-norm）
- YCbCr 色彩空间输入，降低通道间冗余
- MTP 辅助预测头（可选，DeepSeek-V3 风格）
- Cosine decay + linear warmup 学习率调度
- 混合精度训练（bf16 / fp16）与多 GPU 分布式
- TensorBoard 日志，验证集 BPP 均值±标准差
- 配置文件驱动，`--resume` 断点续训
- 手写 Flash Attention Triton kernel（causal early termination，任意 seq_len）
- 手写 Fused RMSNorm Triton kernel
