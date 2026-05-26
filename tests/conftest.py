"""pytest 全局配置 — triton 不可用时跳过所有 fused kernel 测试。

WSL 等纯 CPU 环境没有 triton，但 fused kernel 测试文件顶层
`import triton` 会让 pytest collect 阶段直接 ImportError，整个 pytest 退出
非零。collect_ignore_glob 在 collect 之前判定是否跳过文件，比写在每个
测试文件里的 try-import 守卫更干净。
"""

collect_ignore_glob = []
try:
    import triton  # noqa: F401
except ImportError:
    collect_ignore_glob = [
        "test_flash_attn.py",
        "test_fused_*.py",
    ]
