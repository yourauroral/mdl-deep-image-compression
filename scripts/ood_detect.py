"""OOD 检测（typicality test）— 论文 §5 / future.md §6.2，跑在 AutoDL（推理，不碰训练）。

MDL 命题：压得越好 → 越能识别分布外。把 CIFAR-10 训练的 CC-iGPT 当密度模型，
用 per-image bpd 做异常分。

为什么不能用裸 bpd 阈值（必须读）
--------------------------------
Nalisnick et al. ICLR 2019 *Do Deep Generative Models Know What They Don't Know?*：
CIFAR-10 训的似然模型会给 SVHN **更低** bpd（更"像"训练分布），裸 bpd 阈值得
AUROC < 0.5。本脚本同时报"裸 bpd"和"typicality"两个 scorer，前者用来**复现并
展示**这个反直觉现象，后者才是正确做法。

typicality test（Nalisnick 2019b, *Detecting OOD Inputs Using Typicality*）：
高维高斯的样本几乎不落在概率最高的中心，而集中在一层"典型集"薄壳上。所以 OOD
判据不是"bpd 高/低"，而是"bpd 偏离训练分布的均值多远"——双向异常 |z|。

本工作差异化（双尺度）
--------------------
CC-iGPT 天然给两个独立信号：coarse bpd 偏离 + fine bpd 偏离。joint score
= sqrt(z_coarse² + z_fine²)，是单尺度模型给不出的。脚本对比三种 scorer：
  raw_bpd       —— 裸 bpd（演示 Nalisnick 失败）
  typ_total     —— |z_total|（单信号 typicality）
  typ_dualscale —— sqrt(z_c²+z_f²)（双尺度联合，本工作）

用法（AutoDL）:
    python scripts/ood_detect.py \
        --config configs/ccigpt_cifar10_s_rgb_ronly_v2.yaml \
        --checkpoint experiments/ccigpt_cifar10_s_rgb_ronly_v2/checkpoints/best.pth \
        --ood svhn,cifar100 --ref_images 2000

ID = CIFAR-10 test，OOD = SVHN / CIFAR-100 test（均 32×32 原生，无需 resize）。
typicality 参考统计 (μ, σ) 取自 CIFAR-10 **训练集** 子样本（模型学到的分布）。
"""
import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── 纯函数（numpy，WSL 可单测，不依赖 torch / GPU）─────────────────────

def rankdata_avg(a):
    """1-indexed 排名，平手取平均（等价 scipy.stats.rankdata 'average'）。"""
    a = np.asarray(a, dtype=float)
    n = len(a)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1, dtype=float)
    sorted_a = a[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return ranks


def auroc(scores_id, scores_ood):
    """AUROC，约定"分数越高越 OOD"。= P(score_ood > score_id)（Mann-Whitney U）。

    若某 scorer 让 OOD 分数系统性更低（如裸 bpd 遇 SVHN），结果 < 0.5，
    正好量化 Nalisnick 失败。
    """
    s_id = np.asarray(scores_id, dtype=float)
    s_ood = np.asarray(scores_ood, dtype=float)
    n_id, n_ood = len(s_id), len(s_ood)
    if n_id == 0 or n_ood == 0:
        return float("nan")
    ranks = rankdata_avg(np.concatenate([s_id, s_ood]))
    r_ood = ranks[n_id:].sum()
    return (r_ood - n_ood * (n_ood + 1) / 2.0) / (n_id * n_ood)


def zscore(vals, mu, sigma):
    return (np.asarray(vals, dtype=float) - mu) / max(float(sigma), 1e-8)


def dual_scale_score(z_coarse, z_fine):
    """双尺度联合 typicality：欧氏范数 sqrt(z_c² + z_f²)。"""
    return np.sqrt(np.asarray(z_coarse) ** 2 + np.asarray(z_fine) ** 2)


# ── 模型路径（仅真实评测时 import torch）────────────────────────────

def _build_model(config, checkpoint, device):
    import torch
    from src.mdlic.utils import clean_state_dict
    from scripts.train import _build_ccigpt_from_config, _build_model_from_config
    mcfg = config["model"]
    model_type = mcfg.get("type", "igpt")
    model = (_build_ccigpt_from_config(mcfg, device) if model_type == "ccigpt"
             else _build_model_from_config(mcfg, device))
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(clean_state_dict(sd))
    model.eval()
    return model, model_type


def _per_image_ce(logits, full_tokens):
    """logits (B,T-1,V) + full_tokens (B,T) → per-image 平均 CE (nats/token) (B,)。"""
    import torch
    import torch.nn.functional as F
    B, Tm1, V = logits.shape
    targets = full_tokens[:, 1:]
    ce = F.cross_entropy(logits.reshape(-1, V).float(), targets.reshape(-1),
                         reduction="none")
    return ce.view(B, Tm1).mean(dim=1)


def per_image_signals(model, model_type, x, device):
    """(B,C,H,W) → per-image (bpd_total, ce_coarse, ce_fine)，与 cc_igpt.forward 同公式。"""
    import torch
    import torch.nn.functional as F
    x = x.to(device).clamp(0, 1).float()
    ln2 = math.log(2.0)
    with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=False):
        if model_type == "ccigpt":
            x_c = F.adaptive_avg_pool2d(x, model.coarse_size)[:, :model.coarse.in_channels]
            coarse_tokens = model.coarse._tokenize(x_c)
            out_c = model.coarse(x_c)
            ce_c = _per_image_ce(out_c["logits"], coarse_tokens)
            coarse_ctx = model.ctx_alpha * model._compute_coarse_ctx(coarse_tokens)
            out_f = model.fine(x, coarse_ctx=coarse_ctx)
            ce_f = _per_image_ce(out_f["logits"], model.fine._tokenize(x))
            Nc, Nf = model.coarse.seq_len, model.fine.seq_len
            bpd = (ce_c * Nc + ce_f * Nf) / ln2 / Nf
            return (bpd.cpu().numpy(), ce_c.cpu().numpy(), ce_f.cpu().numpy())
        else:
            out = model(x)
            ce = _per_image_ce(out["logits"], model._tokenize(x))
            bpd = ce / ln2
            b = bpd.cpu().numpy()
            return (b, b.copy(), b.copy())   # 单尺度退化：coarse=fine=total


def _collect(model, model_type, loader, device, max_images, tag):
    bpds, cs, fs, n = [], [], [], 0
    for batch in loader:
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        b, c, f = per_image_signals(model, model_type, x, device)
        bpds.append(b); cs.append(c); fs.append(f); n += len(b)
        if max_images and n >= max_images:
            print(f"  [{tag}] 截断于 {n} 张 (--max_images={max_images})")
            break
    print(f"  [{tag}] 收集 {n} 张")
    return (np.concatenate(bpds), np.concatenate(cs), np.concatenate(fs))


def _loader(ds, batch_size):
    from torch.utils.data import DataLoader
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4,
                      pin_memory=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=str)
    ap.add_argument("--checkpoint", type=str)
    ap.add_argument("--ood", type=str, default="svhn,cifar100",
                    help="逗号分隔 OOD 集：svhn / cifar100")
    ap.add_argument("--ref_images", type=int, default=2000,
                    help="估 μ/σ 的 CIFAR-10 训练子样本数")
    ap.add_argument("--max_images", type=int, default=None,
                    help="每个集最多评多少张（默认全量；截断会打印）")
    ap.add_argument("--batch_size", type=int, default=50)
    ap.add_argument("--self_test", action="store_true",
                    help="仅验证 AUROC/typicality 数学（合成数据，无需 GPU/ckpt）")
    args = ap.parse_args()

    if args.self_test:
        _self_test()
        return

    if not args.config or not args.checkpoint:
        ap.error("需要 --config 和 --checkpoint（或用 --self_test 仅验证数学）")

    import torch
    import yaml
    from torchvision import transforms
    from torchvision.datasets import CIFAR10, CIFAR100, SVHN

    print("== 预检：OOD 数学 self_test ==")
    _self_test()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_type = _build_model(config, args.checkpoint, device)
    root = config["data"].get("valid", "datasets/")
    tf = transforms.ToTensor()                      # CIFAR v2 = 32×32，OOD 均原生 32×32
    print(f"== 模型加载 (type={model_type}, device={device}) ==\n")

    # 参考统计：CIFAR-10 训练子样本
    print("[1] 估 typicality 参考 μ/σ (CIFAR-10 train) ...")
    ref_ds = CIFAR10(root=root, train=True, download=True, transform=tf)
    from torch.utils.data import Subset
    ref_ds = Subset(ref_ds, list(range(min(args.ref_images, len(ref_ds)))))
    ref_bpd, ref_c, ref_f = _collect(model, model_type, _loader(ref_ds, args.batch_size),
                                     device, None, "ref")
    mu_t, sd_t = ref_bpd.mean(), ref_bpd.std()
    mu_c, sd_c = ref_c.mean(), ref_c.std()
    mu_f, sd_f = ref_f.mean(), ref_f.std()
    print(f"    μ_bpd={mu_t:.4f} σ={sd_t:.4f} | μ_c={mu_c:.4f} σ={sd_c:.4f} "
          f"| μ_f={mu_f:.4f} σ={sd_f:.4f}\n")

    # ID = CIFAR-10 test
    print("[2] ID = CIFAR-10 test ...")
    id_ds = CIFAR10(root=root, train=False, download=True, transform=tf)
    id_bpd, id_c, id_f = _collect(model, model_type, _loader(id_ds, args.batch_size),
                                  device, args.max_images, "cifar10-test")

    def scorers(bpd, c, f):
        return {
            "raw_bpd": bpd,
            "typ_total": np.abs(zscore(bpd, mu_t, sd_t)),
            "typ_dualscale": dual_scale_score(zscore(c, mu_c, sd_c),
                                              zscore(f, mu_f, sd_f)),
        }
    id_s = scorers(id_bpd, id_c, id_f)

    print(f"\n{'='*64}")
    print(f"{'OOD 集':>14s} | {'raw_bpd':>10s} | {'typ_total':>10s} | {'typ_dualscale':>13s}")
    print(f"{'-'*64}")
    for name in [s.strip() for s in args.ood.split(",") if s.strip()]:
        if name == "svhn":
            ds = SVHN(root=root, split="test", download=True, transform=tf)
        elif name == "cifar100":
            ds = CIFAR100(root=root, train=False, download=True, transform=tf)
        else:
            print(f"  跳过未知 OOD: {name}")
            continue
        print(f"\n[3] OOD = {name} ...")
        o_bpd, o_c, o_f = _collect(model, model_type, _loader(ds, args.batch_size),
                                   device, args.max_images, name)
        o_s = scorers(o_bpd, o_c, o_f)
        a_raw = auroc(id_s["raw_bpd"], o_s["raw_bpd"])
        a_tot = auroc(id_s["typ_total"], o_s["typ_total"])
        a_dual = auroc(id_s["typ_dualscale"], o_s["typ_dualscale"])
        print(f"{name:>14s} | {a_raw:>10.4f} | {a_tot:>10.4f} | {a_dual:>13.4f}")
        if a_raw < 0.5:
            print(f"    ⚠ raw_bpd AUROC={a_raw:.3f}<0.5 → 复现 Nalisnick：裸 bpd 给 "
                  f"{name} 更低，typicality 修正之")
    print(f"{'='*64}")
    print("scorer 越高越 OOD；AUROC>0.5 有判别力。typ_dualscale 是本工作双尺度差异化。")


def _self_test():
    """合成数据验证 AUROC + typicality 数学（WSL 可跑）。"""
    rng = np.random.RandomState(0)
    # 完美分离：OOD 全高 → AUROC=1
    assert abs(auroc([0, 1, 2], [3, 4, 5]) - 1.0) < 1e-9
    # OOD 全低 → 0（模拟 Nalisnick 裸 bpd）
    assert abs(auroc([3, 4, 5], [0, 1, 2]) - 0.0) < 1e-9
    # 平手居中 → 0.5
    assert abs(auroc([1, 1], [1, 1]) - 0.5) < 1e-9
    # 随机大样本同分布 → ≈0.5
    a = auroc(rng.randn(5000), rng.randn(5000))
    assert 0.45 < a < 0.55, a
    # typicality：ID 近 μ → |z| 小；OOD 远离（双向，含更低）→ |z| 大 → AUROC 高
    mu, sd = 3.0, 0.5
    id_v = rng.normal(mu, sd, 3000)
    ood_v = rng.normal(mu - 2.0, 0.5, 3000)          # bpd 更低（Nalisnick 式）
    a_raw = auroc(id_v, ood_v)                        # 裸 bpd：OOD 更低 → <0.5
    a_typ = auroc(np.abs(zscore(id_v, mu, sd)), np.abs(zscore(ood_v, mu, sd)))
    assert a_raw < 0.5, a_raw
    assert a_typ > 0.9, a_typ                         # typicality 救回
    # dual_scale 单调性
    assert dual_scale_score([3], [4])[0] == 5.0
    print(f"[self_test] AUROC/typicality 数学 OK "
          f"(裸 bpd AUROC={a_raw:.3f} → typicality={a_typ:.3f})")


if __name__ == "__main__":
    main()
