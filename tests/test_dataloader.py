import sys 
import os
import yaml 
import torch 
from torch.utils.data import Dataset, DataLoader 
from PIL import Image 
from torchvision import transforms 
import numpy as np

# 将项目根目录添加到python路径下（脚本在tests/下，根目录在上一级）
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")) 
sys.path.insert(0, project_root)

class ImageFolderDataset(Dataset):
  """从文件夹读取图像，支持随即裁剪和归一化"""
  def __init__(self, root, patch_size=None, transform=None):
    self.filenames = [os.path.join(root, f) for f in os.listdir(root) \
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))] 
    self.patch_size = patch_size
    self.transform = transform or transforms.ToTensor()
  
  def __len__(self):
    return len(self.filenames)
  
  def __getitem__(self, idx):
    img = Image.open(self.filenames[idx]).convert('RGB') 
    if self.patch_size is not None:
      w, h = img.size
      if w < self.patch_size or h < self.patch_size:
        scale = self.patch_size / min(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.BICUBIC)
        w, h = new_w, new_h
      i = np.random.randint(0, w - self.patch_size + 1)
      j = np.random.randint(0, h - self.patch_size + 1)
      img = img.crop((i, j, i + self.patch_size, j + self.patch_size)) 
    # 转换成tensor并归一化到[0, 1]
    x = self.transform(img) 
    return x

if __name__ == "__main__":
  # 配置文件路径(相对于项目根目录)
  config_path = os.path.join(project_root, "configs/hyperprior_mse.yaml")
  print(f"Using config file: {config_path}") 

  with open(config_path, "r") as f:
    config = yaml.safe_load(f)
  
  # 检查配置文件中的键名是否与脚本兼容
  print("Config keys:", config.keys())
  print("Data keys:", config['data'].keys()) 

  # 创建数据集和数据加载器
  train_dataset = ImageFolderDataset(
    root = config['data']['train'],
    patch_size = config['data']['patch_size']
  )
  print(f"Number of training samples: {len(train_dataset)}") 

  train_loader = DataLoader(
    train_dataset,
    batch_size = config['train']['batch_size'],
    shuffle = True,
    num_workers = config['data']['num_workers'],
  )

  for i, x in enumerate(train_loader):
    print(f"Batch {i}: image tensor shape: {x.shape}") 
    print(f"min: {x.min():.4f}, max: {x.max():.4g}") 
    break