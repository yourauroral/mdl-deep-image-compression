import torch 
from mdlic.models import HyperpriorModel 

def main():
  device = "cuda" if torch.cuda.is_available() else "cpu"
  net = HyperpriorModel().to(device).eval()
  x = torch.rand(1,3,512,512, device=device)
  out = net(x)
  for k,v in out.items():
      print(k, tuple(v.shape))
  assert out["x_hat"].shape == x.shape
  print("OK")

if __name__ == '__main__':
  main()