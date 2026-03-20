# 延迟导入：fused_ce_zloss 为可选的 Triton 自定义 kernel，
# 仅在安装 triton 并且实现文件存在时可用。
# 若 kernel 不可用，训练脚本会回退到标准 PyTorch 的
# F.cross_entropy + logsumexp 实现。
try:
    from .fused_ce_zloss import fused_cross_entropy_zloss
except ImportError:
    fused_cross_entropy_zloss = None

try:
    from .fused_rms_norm import FusedRMSNorm, fused_rms_norm
except ImportError:
    FusedRMSNorm = None
    fused_rms_norm = None
