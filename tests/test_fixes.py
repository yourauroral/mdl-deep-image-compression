"""
DDP checkpoint / grad_accum / config validation 回归测试。

测试范围：
  1. DDP checkpoint 保存/加载：raw_model 解包 + module. 前缀剥离
  2. grad_accum epoch 末尾 flush：余数 micro-batch 的梯度不丢失
  3. configs 目录所有 yaml 通过 _validate_config
  4. 端到端：保存→加载→forward 一致

运行：
    pytest tests/test_fixes.py -v
"""
import os
import sys
import math
import csv
import random
import yaml
import tempfile
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn as nn
import pytest
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mdlic.models.igpt import IGPT


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
        activation_checkpointing=False,
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
# Fix #2: forward logits 可用于评测
# ──────────────────────────────────────────────────────────────
def test_forward_yields_logits():
    """forward 返回的 out["logits"] 必须非 None，evaluate_per_channel 依赖它。"""
    model = _build_tiny_igpt()
    model.eval()
    x = torch.rand(2, 3, 8, 8)
    with torch.no_grad():
        out = model(x)
    assert out["logits"] is not None
    # 序列长度 8*8*3 = 192，NTP 偏移后 T = 191
    assert out["logits"].shape == (2, 191, 256)


def test_per_image_bpd_matches_batch_igpt():
    """per-image bpd 均值应与含首 token 的 IGPT scalar bpd 同口径。"""
    from scripts.evaluate import _per_image_bpd

    model = _build_tiny_igpt()
    model.eval()
    x = torch.rand(3, 3, 8, 8)
    with torch.no_grad():
        out = model(x)
        per_image = _per_image_bpd(model, x, out)
    assert torch.allclose(per_image.mean(), out["bpd"], atol=1e-6, rtol=1e-6)


def test_per_image_summary_bootstrap_optional():
    """bootstrap_samples=0 时只报确定性汇总，便于快速评测。"""
    from scripts.evaluate import _summarize_per_image_bpd

    summary = _summarize_per_image_bpd([1.0, 2.0, 3.0], bootstrap_samples=0)
    assert summary["n"] == 3
    assert summary["mean"] == pytest.approx(2.0)
    assert summary["std"] == pytest.approx(1.0)
    assert "ci95_bootstrap" not in summary


def test_ensemble_log_probs_uses_probability_mixture():
    """Ensemble must average probabilities, not per-model NLL values."""
    from scripts.evaluate import _ensemble_log_probs

    probs_a = torch.tensor([[[0.9, 0.1]]], dtype=torch.float32)
    probs_b = torch.tensor([[[0.1, 0.9]]], dtype=torch.float32)
    log_probs = _ensemble_log_probs([probs_a.log(), probs_b.log()])

    expected = ((probs_a + probs_b) * 0.5).log()
    assert torch.allclose(log_probs, expected, atol=1e-7, rtol=1e-7)

    target = torch.tensor([[0]])
    mixture_nll = -log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    mean_member_nll = -torch.stack([probs_a.log(), probs_b.log()], dim=0).mean(dim=0) \
        .gather(-1, target.unsqueeze(-1)).squeeze(-1)
    assert mixture_nll.item() < mean_member_nll.item()


def test_ensemble_per_image_stats_are_batch_size_invariant():
    """Ensemble mean/std must come from images, not squared batch means."""
    from scripts.evaluate import evaluate_ensemble

    torch.manual_seed(9)
    model_a = _build_tiny_igpt().eval()
    torch.manual_seed(10)
    model_b = _build_tiny_igpt().eval()
    images = torch.rand(5, 3, 8, 8)
    labels = torch.zeros(5, dtype=torch.long)
    dataset = TensorDataset(images, labels)

    results = []
    for batch_size in (2, 3):
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        mean, std, _, extras = evaluate_ensemble(
            [model_a, model_b], loader, torch.device("cpu"),
            collect_per_image=True,
        )
        values = extras["per_image_bpd"]
        assert len(values) == len(dataset)
        assert [row["sample_id"] for row in extras["per_image_records"]] == list(range(5))
        assert mean == pytest.approx(float(torch.tensor(values).mean()), abs=1e-6)
        assert std == pytest.approx(float(torch.tensor(values).std(unbiased=True)), abs=1e-6)
        results.append((mean, std, values))

    assert results[0][0] == pytest.approx(results[1][0], abs=1e-7)
    assert results[0][1] == pytest.approx(results[1][1], abs=1e-7)
    assert results[0][2] == pytest.approx(results[1][2], abs=1e-6)


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


def test_grad_accum_residual_window_uses_residual_denominator():
    """残余窗口里的每个 micro-batch 都应除以余数，而不是固定 accum 长度。"""
    from scripts.train import _grad_accum_window_size

    steps, grad_accum_steps = 10, 4
    denominators = [
        _grad_accum_window_size(i, steps, grad_accum_steps)
        for i in range(steps)
    ]
    assert denominators == [4, 4, 4, 4, 4, 4, 4, 4, 2, 2]


def test_grad_accum_divisible_case_unchanged():
    """整除情况下行为应与修复前完全一致（向后兼容）。"""
    steps, grad_accum_steps = 12, 4
    from scripts.train import _grad_accum_window_size

    denominators = [
        _grad_accum_window_size(i, steps, grad_accum_steps)
        for i in range(steps)
    ]
    assert denominators == [4] * 12

    sync_steps = []
    for i in range(steps):
        is_last_step = (i + 1) == steps
        if (i + 1) % grad_accum_steps == 0 or is_last_step:
            sync_steps.append(i + 1)
    assert sync_steps == [4, 8, 12]


def test_train_epoch_metrics_are_sample_weighted_for_short_last_batch():
    """Epoch metrics must not give a short final batch the same weight as a full batch."""
    from scripts.train import train_one_epoch

    class BatchMeanModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = nn.Parameter(torch.tensor(0.0))

        def forward(self, x, z_loss_weight=0.0):
            value = x.mean() + self.anchor * 0.0
            return {"loss": value, "ce_loss": value, "bpd": value}

    model = BatchMeanModel()
    loader = DataLoader(
        TensorDataset(torch.tensor([[0.0], [0.0], [6.0]])),
        batch_size=2,
        shuffle=False,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)

    avg_loss, avg_bpd = train_one_epoch(
        model,
        loader,
        [optimizer],
        scaler=None,
        device=torch.device("cpu"),
        epoch=0,
        log_freq=100,
        writer=None,
        clip_max_norm=1.0,
        amp_dtype=None,
    )

    assert avg_loss == pytest.approx(2.0)
    assert avg_bpd == pytest.approx(2.0)


def test_rng_state_roundtrip_restores_all_cpu_generators():
    from scripts.train import _capture_rng_state, _restore_rng_state

    random.seed(101)
    np.random.seed(102)
    torch.manual_seed(103)
    state = _capture_rng_state()
    expected = (
        random.random(),
        float(np.random.random()),
        torch.rand(4),
    )

    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    _restore_rng_state(state)
    actual = (
        random.random(),
        float(np.random.random()),
        torch.rand(4),
    )
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])


def test_training_csv_resume_appends_without_duplicate_header(tmp_path):
    from scripts.train import _CSV_HEADER, _open_training_csv

    path = tmp_path / "training_curves.csv"
    fh, writer = _open_training_csv(str(path), resume=False, start_epoch=1)
    writer.writerow([2, "1", "2", "3", "4", "5", "6"])
    fh.close()

    fh, writer = _open_training_csv(str(path), resume=True, start_epoch=3)
    writer.writerow([3, "1", "2", "3", "4", "5", "6"])
    fh.close()

    with path.open(newline="") as source:
        rows = list(csv.reader(source))
    assert rows[0] == _CSV_HEADER
    assert [int(row[0]) for row in rows[1:]] == [2, 3]

    with pytest.raises(ValueError, match="重复/倒序"):
        _open_training_csv(str(path), resume=True, start_epoch=3)


def test_canonical_config_hash_is_order_independent():
    from scripts.train import _canonical_config_hash

    left = {"model": {"N": 2, "d_model": 64}, "train": {"lr": 1e-3}}
    right = {"train": {"lr": 1e-3}, "model": {"d_model": 64, "N": 2}}
    assert _canonical_config_hash(left) == _canonical_config_hash(right)
    right["model"]["N"] = 3
    assert _canonical_config_hash(left) != _canonical_config_hash(right)


def test_distributed_eval_sampler_partitions_without_padding_or_drop():
    from scripts.train import DistributedEvalSampler

    dataset = list(range(11))
    shards = [
        list(DistributedEvalSampler(dataset, num_replicas=3, rank=rank))
        for rank in range(3)
    ]
    flattened = [index for shard in shards for index in shard]

    assert [len(shard) for shard in shards] == [4, 4, 3]
    assert sorted(flattened) == list(range(len(dataset)))
    assert len(flattened) == len(set(flattened))


def _resume_fixture(config):
    from scripts.train import TRAINING_STATE_SCHEMA, _canonical_config_hash

    provenance_sha256 = "ab" * 32
    return {
        "schema": TRAINING_STATE_SCHEMA,
        "epoch": 3,
        "model_state_dict": {},
        "optimizer_state_dicts": [{}],
        "scheduler_state_dict": {},
        "best_bpd": 2.9,
        "config_sha256": _canonical_config_hash(config),
        "model_config_sha256": _canonical_config_hash(config["model"]),
        "seed": 42,
        "world_size": 2,
        "rng_states_by_rank": [{}, {}],
        "provenance": {"fingerprint_sha256": provenance_sha256},
        "provenance_sha256": provenance_sha256,
    }


def test_strict_resume_checkpoint_requires_complete_matching_state():
    from scripts.train import _validate_resume_checkpoint

    config = {"model": {"type": "igpt", "N": 2}, "train": {"seed": 42}}
    checkpoint = _resume_fixture(config)
    _validate_resume_checkpoint(
        checkpoint,
        config=config,
        seed=42,
        world_size=2,
        provenance_fingerprint=checkpoint["provenance_sha256"],
        optimizer_count=1,
        scheduler_required=True,
        scaler_required=False,
        swa_enabled=False,
        ema_enabled=False,
    )

    with pytest.raises(RuntimeError, match="--init_from"):
        _validate_resume_checkpoint(
            {"model_state_dict": {}}, config=config, seed=42, world_size=2,
            provenance_fingerprint=checkpoint["provenance_sha256"],
            optimizer_count=1, scheduler_required=True, scaler_required=False,
            swa_enabled=False, ema_enabled=False,
        )

    incomplete = dict(checkpoint)
    del incomplete["optimizer_state_dicts"]
    with pytest.raises(RuntimeError, match="incomplete"):
        _validate_resume_checkpoint(
            incomplete, config=config, seed=42, world_size=2,
            provenance_fingerprint=checkpoint["provenance_sha256"],
            optimizer_count=1, scheduler_required=True, scaler_required=False,
            swa_enabled=False, ema_enabled=False,
        )


def test_strict_resume_rejects_config_seed_and_world_size_changes():
    from scripts.train import _validate_resume_checkpoint

    config = {"model": {"type": "igpt", "N": 2}, "train": {"seed": 42}}
    checkpoint = _resume_fixture(config)
    common = dict(
        provenance_fingerprint=checkpoint["provenance_sha256"],
        optimizer_count=1,
        scheduler_required=True,
        scaler_required=False,
        swa_enabled=False,
        ema_enabled=False,
    )
    changed = {"model": dict(config["model"]), "train": {"seed": 42, "lr": 1e-3}}
    with pytest.raises(RuntimeError, match="full config hash"):
        _validate_resume_checkpoint(
            checkpoint, config=changed, seed=42, world_size=2, **common,
        )
    with pytest.raises(RuntimeError, match="seed mismatch"):
        _validate_resume_checkpoint(
            checkpoint, config=config, seed=7, world_size=2, **common,
        )
    with pytest.raises(RuntimeError, match="world_size mismatch"):
        _validate_resume_checkpoint(
            checkpoint, config=config, seed=42, world_size=1, **common,
        )
    with pytest.raises(RuntimeError, match="provenance mismatch"):
        _validate_resume_checkpoint(
            checkpoint, config=config, seed=42, world_size=2,
            **{**common, "provenance_fingerprint": "cd" * 32},
        )


def test_legacy_resume_requires_explicit_acknowledgement():
    from scripts.train import (
        LEGACY_TRAINING_STATE_SCHEMA,
        _validate_resume_checkpoint,
    )

    config = {"model": {"type": "igpt", "N": 2}, "train": {"seed": 42}}
    checkpoint = _resume_fixture(config)
    checkpoint["schema"] = LEGACY_TRAINING_STATE_SCHEMA
    del checkpoint["provenance"]
    del checkpoint["provenance_sha256"]
    kwargs = dict(
        config=config,
        seed=42,
        world_size=2,
        optimizer_count=1,
        scheduler_required=True,
        scaler_required=False,
        swa_enabled=False,
        ema_enabled=False,
    )

    with pytest.raises(RuntimeError, match="allow_legacy_resume"):
        _validate_resume_checkpoint(checkpoint, **kwargs)
    _validate_resume_checkpoint(
        checkpoint, allow_legacy_resume=True, **kwargs,
    )


def test_init_from_loads_only_shape_compatible_model_weights():
    from scripts.train import _load_init_from_state_dict

    source = _build_tiny_igpt()
    target = _build_tiny_igpt()
    transferred = torch.full_like(source.token_embed.weight, 0.25)
    checkpoint = {
        "model_state_dict": {
            "module.token_embed.weight": transferred,
            "module.head.weight": torch.zeros(1),
            "module.not_a_real_parameter": torch.zeros(1),
        }
    }

    stats = _load_init_from_state_dict(target, checkpoint)
    assert torch.equal(target.token_embed.weight, transferred)
    assert 0.0 < stats["parameter_coverage"] < 1.0
    assert stats["shape_mismatch_keys"] == 1
    assert stats["unexpected_keys"] == 1
    with pytest.raises(RuntimeError, match="matched no trainable"):
        _load_init_from_state_dict(target, {"not_a_key": torch.zeros(1)})


def test_checkpoint_meta_records_initialization_semantics():
    from scripts.train import _checkpoint_meta

    config = {
        "exp_name": "test",
        "model": {"type": "igpt"},
        "train": {"seed": 42},
    }
    args = SimpleNamespace(config="config.yaml", resume=None, init_from="weights.pth")
    meta = _checkpoint_meta(config, args, 1, "best", 42, metrics={})
    assert meta["initialization_mode"] == "init_from"
    assert meta["init_from_path"] == "weights.pth"


# ──────────────────────────────────────────────────────────────
# Fix #4: configs 目录所有 yaml 都能加载且过 _validate_config
# ──────────────────────────────────────────────────────────────
import glob

CONFIG_DIR = os.path.join(os.path.dirname(__file__), '..', 'configs')
ALL_CONFIGS = sorted(os.path.basename(p) for p in glob.glob(os.path.join(CONFIG_DIR, '*.yaml')))


@pytest.mark.parametrize("name", ALL_CONFIGS)
def test_config_validates(name):
    """每个 yaml 必填字段齐全且通过 _validate_config，新增 yaml 自动覆盖。"""
    from scripts.train import _validate_config
    path = os.path.join(CONFIG_DIR, name)
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for key in ['exp_name', 'model', 'data', 'train', 'eval', 'checkpoint']:
        assert key in cfg, f"{name} 缺少字段 {key}"
    _validate_config(cfg)


# ──────────────────────────────────────────────────────────────
# 端到端 smoke：保存→加载→forward 输出一致
# ──────────────────────────────────────────────────────────────
def test_end_to_end_save_load_consistency():
    """
    完整链路验证：DDP wrapped 模型保存 → 单卡裸模型加载 → forward 输出 bit-exact 一致。
    这是实际评测脚本会走的路径，能暴露任何 key/shape/dtype 不一致。
    """
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
