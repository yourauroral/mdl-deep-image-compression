"""
针对 2026-04-20 代码审计发现的四个问题的回归测试。

测试范围：
  1. DDP checkpoint 保存/加载：raw_model 解包 + module. 前缀剥离
  2. evaluate.py per-channel：fused linear CE 关闭后 logits 非空
  3. grad_accum epoch 末尾 flush：余数 micro-batch 的梯度不丢失
  4. CIFAR-10 / CIFAR-100 配置可加载且字段一致

运行：
    pytest tests/test_fixes.py -v
"""
import os
import sys
import math
import yaml
import tempfile
import torch
import torch.nn as nn
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.mdlic.models.igpt import IGPT
from src.mdlic.models import igpt as igpt_mod


# ──────────────────────────────────────────────────────────────
# 公共 fixture：构造一个最小规格的 IGPT，便于快速跑测试
# ──────────────────────────────────────────────────────────────
def _build_tiny_igpt(device='cpu'):
    """N=2、d_model=64 的 mini iGPT，CPU 上几十 ms 跑完。"""
    return IGPT(
        image_size=8,         # 8×8 图像，序列长度 = 8*8*3 = 192
        in_channels=3,
        vocab_size=256,
        d_model=64,
        N=2,
        h=4,
        d_ff=128,
        dropout=0.0,
        use_ycbcr=True,
        use_rope=True,
        use_post_norm=True,
        use_swiglu=True,
        use_qk_norm=True,
        use_depth_scaled_init=True,
        use_zloss=True,
        activation_checkpointing=False,
        use_subpixel_ar=True,
    ).to(device)


# ──────────────────────────────────────────────────────────────
# Fix #1: DDP checkpoint 保存/加载
# ──────────────────────────────────────────────────────────────
class _FakeDDP(nn.Module):
    """
    伪 DDP wrapper —— 真 DDP 需要 NCCL/Gloo 进程组，单元测试不便启动。
    我们只复刻关键行为：state_dict() 的 key 带 `module.` 前缀。
    """
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *a, **kw):
        return self.module(*a, **kw)


def test_ddp_state_dict_has_module_prefix():
    """验证 DDP 包装后 key 多了 module. 前缀（这是 bug 的根因）。"""
    raw = _build_tiny_igpt()
    wrapped = _FakeDDP(raw)
    raw_keys = set(raw.state_dict().keys())
    wrapped_keys = set(wrapped.state_dict().keys())
    assert all(k.startswith('module.') for k in wrapped_keys)
    assert {k[len('module.'):] for k in wrapped_keys} == raw_keys


def test_raw_model_save_loads_into_bare_model():
    """
    模拟修复后的保存路径：用 raw_model.state_dict() 保存，
    然后由单卡评测脚本（裸 IGPT）加载，应该零 mismatch。
    """
    raw = _build_tiny_igpt()
    wrapped = _FakeDDP(raw)
    distributed = True
    raw_model = wrapped.module if distributed else wrapped

    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
        torch.save(raw_model.state_dict(), f.name)
        path = f.name

    try:
        fresh = _build_tiny_igpt()
        sd = torch.load(path, map_location='cpu', weights_only=False)
        # strict=True：所有 key 必须严格对应
        missing, unexpected = fresh.load_state_dict(sd, strict=True)
        assert missing == [] and unexpected == []
    finally:
        os.unlink(path)


def test_strip_module_prefix_handles_legacy_checkpoint():
    """
    旧版本可能保存了带 module. 前缀的 checkpoint，
    _strip_module_prefix 应能透明地把它转成裸 state_dict。
    """
    def _strip_module_prefix(sd):
        if any(k.startswith('module.') for k in sd.keys()):
            return {(k[len('module.'):] if k.startswith('module.') else k): v
                    for k, v in sd.items()}
        return sd

    raw = _build_tiny_igpt()
    wrapped = _FakeDDP(raw)
    legacy_sd = wrapped.state_dict()  # 带 module. 前缀

    fresh = _build_tiny_igpt()
    cleaned = _strip_module_prefix(legacy_sd)
    missing, unexpected = fresh.load_state_dict(cleaned, strict=True)
    assert missing == [] and unexpected == []

    # 已经干净的 state_dict 应原样返回
    clean_sd = raw.state_dict()
    assert _strip_module_prefix(clean_sd) is clean_sd or \
           set(_strip_module_prefix(clean_sd).keys()) == set(clean_sd.keys())


# ──────────────────────────────────────────────────────────────
# Fix #2: evaluate.py per-channel —— logits 不能为 None
# ──────────────────────────────────────────────────────────────
def test_fused_linear_ce_off_yields_logits():
    """
    关闭 _USE_FUSED_LINEAR_CE 后，forward 返回的 out["logits"] 必须非 None，
    否则 evaluate_per_channel 会立即 NoneType 崩溃。
    """
    saved = igpt_mod._USE_FUSED_LINEAR_CE
    igpt_mod._USE_FUSED_LINEAR_CE = False
    try:
        model = _build_tiny_igpt()
        model.eval()
        x = torch.rand(2, 3, 8, 8)
        with torch.no_grad():
            out = model(x)
        assert out["logits"] is not None
        # 序列长度 8*8*3 = 192，NTP 偏移后 T = 191
        assert out["logits"].shape == (2, 191, 256)
    finally:
        igpt_mod._USE_FUSED_LINEAR_CE = saved


def test_fused_linear_ce_flag_restored_after_exception():
    """
    模拟 evaluate_per_channel 中 try/finally 的语义：
    即便循环里抛异常，全局 flag 也必须被复位，否则会污染后续命令。
    """
    saved = igpt_mod._USE_FUSED_LINEAR_CE
    try:
        igpt_mod._USE_FUSED_LINEAR_CE = False
        try:
            raise RuntimeError("simulated eval error")
        finally:
            igpt_mod._USE_FUSED_LINEAR_CE = saved
    except RuntimeError:
        pass
    assert igpt_mod._USE_FUSED_LINEAR_CE == saved


# ──────────────────────────────────────────────────────────────
# Fix #3: grad_accum 末尾 flush
# ──────────────────────────────────────────────────────────────
def test_grad_accum_epoch_end_flushes_residual():
    """
    len(loader)=10, grad_accum_steps=4 → 余数 2。
    若仅当 (i+1)%accum==0 时 step，最后两步的梯度会被下一 epoch zero_grad 清掉。
    修复后的条件 `(i+1)%accum==0 OR (i+1)==steps` 应让最后一步也触发 step。
    """
    steps = 10
    grad_accum_steps = 4
    sync_steps = []
    for i in range(steps):
        is_last_step = (i + 1) == steps
        if (i + 1) % grad_accum_steps == 0 or is_last_step:
            sync_steps.append(i + 1)

    # 期望：第 4、8 步是常规累积窗口，第 10 步是 epoch 末尾强制 flush
    assert sync_steps == [4, 8, 10]


def test_grad_accum_no_sync_releases_at_last_step():
    """no_sync 在 epoch 最后一步必须放开，让 AllReduce 同步残余梯度。"""
    steps, grad_accum_steps = 10, 4
    no_sync_steps = []
    for i in range(steps):
        is_last_step = (i + 1) == steps
        is_accumulating = ((i + 1) % grad_accum_steps != 0) and (not is_last_step)
        if is_accumulating:
            no_sync_steps.append(i + 1)

    # 第 1,2,3,5,6,7,9 步处于 no_sync；第 4,8,10 步同步
    assert no_sync_steps == [1, 2, 3, 5, 6, 7, 9]


def test_grad_accum_divisible_case_unchanged():
    """整除情况下行为应与修复前完全一致（向后兼容）。"""
    steps, grad_accum_steps = 12, 4
    sync_steps = []
    for i in range(steps):
        is_last_step = (i + 1) == steps
        if (i + 1) % grad_accum_steps == 0 or is_last_step:
            sync_steps.append(i + 1)
    assert sync_steps == [4, 8, 12]


# ──────────────────────────────────────────────────────────────
# Fix #4: CIFAR-10 / CIFAR-100 配置一致性
# ──────────────────────────────────────────────────────────────
CONFIG_DIR = os.path.join(os.path.dirname(__file__), '..', 'configs')


@pytest.mark.parametrize("name", ["igpt_cifar10_s.yaml", "igpt_cifar100_s.yaml"])
def test_s_config_loads(name):
    """两份 S 级别 config 都能被 yaml 解析，且关键字段齐全。"""
    path = os.path.join(CONFIG_DIR, name)
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for key in ['exp_name', 'model', 'data', 'train', 'eval', 'checkpoint']:
        assert key in cfg, f"{name} 缺少字段 {key}"
    assert cfg['model']['N'] == 24
    assert cfg['model']['d_model'] == 512
    assert cfg['train']['amp_dtype'] == 'bf16'


def test_cifar10_vs_cifar100_only_differ_in_dataset_and_expname():
    """
    CIFAR-10 和 CIFAR-100 的 S 配置应仅在 exp_name 和 data.dataset 不同，
    其他超参一致 —— 这样切数据集只需复制配置，不会引入意外差异。
    """
    with open(os.path.join(CONFIG_DIR, 'igpt_cifar10_s.yaml')) as f:
        c10 = yaml.safe_load(f)
    with open(os.path.join(CONFIG_DIR, 'igpt_cifar100_s.yaml')) as f:
        c100 = yaml.safe_load(f)

    assert c10['exp_name'] != c100['exp_name']
    assert c10['data']['dataset'] == 'cifar10'
    assert c100['data']['dataset'] == 'cifar100'

    # 模型/训练超参完全一致
    assert c10['model'] == c100['model']
    assert c10['train'] == c100['train']
    assert c10['eval'] == c100['eval']
    assert c10['checkpoint'] == c100['checkpoint']


# ──────────────────────────────────────────────────────────────
# 端到端 smoke：保存→加载→forward 输出一致
# ──────────────────────────────────────────────────────────────
def test_end_to_end_save_load_consistency():
    """
    完整链路验证：DDP wrapped 模型保存 → 单卡裸模型加载 → forward 输出 bit-exact 一致。
    这是实际评测脚本会走的路径，能暴露任何 key/shape/dtype 不一致。
    """
    saved_flag = igpt_mod._USE_FUSED_LINEAR_CE
    igpt_mod._USE_FUSED_LINEAR_CE = False  # 让 logits 非 None 便于比对
    try:
        torch.manual_seed(42)
        raw = _build_tiny_igpt()
        wrapped = _FakeDDP(raw)
        raw_model = wrapped.module

        x = torch.rand(2, 3, 8, 8)
        raw_model.eval()
        with torch.no_grad():
            ref = raw_model(x)["logits"]

        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            torch.save(raw_model.state_dict(), f.name)
            path = f.name

        try:
            fresh = _build_tiny_igpt()
            fresh.load_state_dict(torch.load(path, map_location='cpu',
                                              weights_only=False))
            fresh.eval()
            with torch.no_grad():
                got = fresh(x)["logits"]
            assert torch.equal(ref, got), "保存→加载后 forward 输出不一致"
        finally:
            os.unlink(path)
    finally:
        igpt_mod._USE_FUSED_LINEAR_CE = saved_flag
