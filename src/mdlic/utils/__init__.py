# 延迟导入：metrics 模块尚未实现（计划包含 psnr, ssim, bpp 工具函数）。
# 在模块实现之前，使用 try/except 避免 import mdlic.utils 时崩溃。
try:
    from .metrics import psnr, compute_ssim, compute_bpp
except ImportError:
    psnr = None
    compute_ssim = None
    compute_bpp = None
