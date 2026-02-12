import torch.nn as nn 

def conv(in_ch, out_ch, k=5, s=2):
  p = k // 2 
  return nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p) 

def deconv(in_ch, out_ch, k=5, s=2):
  p = k // 2
  op = s - 1 
  return nn.ConvTranspose2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, output_padding=op) 

class ResidualBlock(nn.Module):
  def __init__(self, ch:int):
    super().__init__()
    self.net = nn.Sequential(
      nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1),
    )
    self.act = nn.ReLU(inplace=True) 
  def forward(self, x):
    return self.act(x + self.net(x)) 