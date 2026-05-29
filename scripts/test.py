import io, numpy as np
from PIL import Image
v = np.load("/root/autodl-tmp/val_64.npy", mmap_mode="r")
N = min(2000, v.shape[0])                      # sample for speed
png = webp = 0.0
for i in range(N):
    im = Image.fromarray(np.asarray(v[i], dtype=np.uint8))
    b = io.BytesIO(); im.save(b, "PNG");            png  += b.tell()*8
    b = io.BytesIO(); im.save(b, "WEBP", lossless=True); webp += b.tell()*8
dims = 64*64*3
print(f"PNG  bpd = {png/N/dims:.3f}")
print(f"WebP bpd = {webp/N/dims:.3f}")