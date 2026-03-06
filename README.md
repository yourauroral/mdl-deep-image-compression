# MDL Deep Image Compression

基于最小描述长度（MDL）原则的深度学习图像压缩系统。  
实现超先验（Hyperprior）和 iGPT 两种模型，支持训练、验证与测试。

## 快速开始

1. **安装依赖**  
   ```bash
   pip install torch torchvision pyyaml numpy pillow tensorboard
   ```

2. **准备数据**  
   - 超先验：图像文件夹（如 `datasets/div2k/train`）  
   - iGPT：CIFAR-100（指向 `datasets/`）

3. **修改配置**  
   `configs/` 下已有示例（`hyperprior_mse.yaml`、`igpt_small.yaml`）

4. **训练**  
   ```bash
   python scripts/train.py --config configs/你的配置.yaml
   ```

## 主要特点
- 超先验 + iGPT 模型  
- 模块化设计，配置文件驱动  
- 支持混合精度、分布式训练  
- 自动日志（TensorBoard）和指标（PSNR/SSIM/BPP）

## 项目状态
持续开发中，基础训练流程已跑通。欢迎 Issue 或 PR。
