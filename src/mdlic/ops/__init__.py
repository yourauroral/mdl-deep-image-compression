# 延迟导入：fused kernel 为可选的 Triton 自定义 kernel，
# 仅在安装 triton 并且实现文件存在时可用。
# 若 kernel 不可用，训练脚本会回退到标准 PyTorch 实现。
try:
    from .fused_ce_zloss import fused_cross_entropy_zloss
except ImportError:
    fused_cross_entropy_zloss = None

try:
    from .fused_rms_norm import FusedRMSNorm, fused_rms_norm
except ImportError:
    FusedRMSNorm = None
    fused_rms_norm = None

try:
    from .fused_swiglu import fused_swiglu
except ImportError:
    fused_swiglu = None

try:
    from .fused_rope import fused_apply_rotary_emb
except ImportError:
    fused_apply_rotary_emb = None

try:
    from .fused_add_rms_norm import fused_add_rms_norm
except ImportError:
    fused_add_rms_norm = None

try:
    from .fused_attn_rope import fused_attn_rope
except ImportError:
    fused_attn_rope = None

try:
    from .fused_linear_ce import fused_linear_cross_entropy
except ImportError:
    fused_linear_cross_entropy = None
