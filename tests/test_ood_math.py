"""OOD typicality 数学单测（纯 numpy，WSL 可跑，不依赖 torch/GPU）。"""
import numpy as np

from scripts.ood_detect import rankdata_avg, auroc, zscore, dual_scale_score


def test_auroc_perfect_separation():
    assert abs(auroc([0, 1, 2], [3, 4, 5]) - 1.0) < 1e-9


def test_auroc_inverted_is_zero():
    # OOD 系统性更低（Nalisnick 裸 bpd 现象）→ AUROC < 0.5（这里=0）
    assert abs(auroc([3, 4, 5], [0, 1, 2]) - 0.0) < 1e-9


def test_auroc_ties_half():
    assert abs(auroc([1, 1, 1], [1, 1, 1]) - 0.5) < 1e-9


def test_auroc_random_near_half():
    rng = np.random.RandomState(1)
    a = auroc(rng.randn(8000), rng.randn(8000))
    assert 0.46 < a < 0.54


def test_auroc_matches_mannwhitney_bruteforce():
    rng = np.random.RandomState(2)
    s_id = rng.randn(60)
    s_ood = rng.randn(60) + 0.5
    # 暴力 P(ood>id) + 0.5·P(==)
    wins = ties = 0
    for o in s_ood:
        for i in s_id:
            if o > i:
                wins += 1
            elif o == i:
                ties += 1
    brute = (wins + 0.5 * ties) / (len(s_id) * len(s_ood))
    assert abs(auroc(s_id, s_ood) - brute) < 1e-9


def test_rankdata_average_ties():
    # [10,10,20] → 平手取平均：(1+2)/2=1.5, 1.5, 3
    r = rankdata_avg([10, 10, 20])
    assert list(r) == [1.5, 1.5, 3.0]


def test_typicality_rescues_nalisnick():
    """裸 bpd 对"更低 bpd 的 OOD"失败 (<0.5)，|z| typicality 救回 (>0.9)。"""
    rng = np.random.RandomState(3)
    mu, sd = 3.0, 0.5
    id_v = rng.normal(mu, sd, 4000)
    ood_v = rng.normal(mu - 2.0, 0.5, 4000)
    assert auroc(id_v, ood_v) < 0.5
    a_typ = auroc(np.abs(zscore(id_v, mu, sd)), np.abs(zscore(ood_v, mu, sd)))
    assert a_typ > 0.9


def test_dual_scale_euclidean():
    out = dual_scale_score([3.0, 0.0], [4.0, 0.0])
    assert out[0] == 5.0 and out[1] == 0.0
