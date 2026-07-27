"""pytest 全局配置。

WSL 等纯 CPU 环境没有 triton，但 fused kernel 测试文件顶层
`import triton` 会让 pytest collect 阶段直接 ImportError，整个 pytest 退出
非零。collect_ignore_glob 在 collect 之前判定是否跳过文件，比写在每个
测试文件里的 try-import 守卫更干净。
"""

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

collect_ignore_glob = []
try:
    import triton  # noqa: F401
except ImportError:
    collect_ignore_glob = [
        "test_flash_attn.py",
        "test_fused_*.py",
    ]


CUDA_TEST_FILES = {
    "test_flash_attn.py",
    "test_fused_add_rms_norm.py",
    "test_fused_attn_rope.py",
    "test_fused_ce_zloss.py",
    "test_fused_linear_ce.py",
    "test_fused_rms_norm.py",
    "test_fused_rope.py",
    "test_fused_swiglu.py",
}

SLOW_TEST_FILES = {
    "test_flash_attn.py",
    "test_fused_attn_rope.py",
    "test_fused_linear_ce.py",
}


def pytest_collection_modifyitems(config, items):
    """Attach coarse resource markers for quick local / AutoDL selections."""
    for item in items:
        filename = item.path.name
        if filename in CUDA_TEST_FILES or item.get_closest_marker("cuda") is not None:
            item.add_marker("cuda")
        else:
            item.add_marker("cpu")
        if filename in SLOW_TEST_FILES:
            item.add_marker("slow")
