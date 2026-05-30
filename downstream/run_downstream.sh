#!/usr/bin/env bash
# CC-iGPT v2 (CIFAR-10) 下游任务批量执行 —— 在 AutoDL GPU 上跑。
# 复用 best.pth，不改训练；与 IN64 训练共享显存，建议在 epoch 间隙跑。
# 详见 downstream/runbook.md。
#
# 用法：
#   bash downstream/run_downstream.sh                # 跑全部 ready 步骤 [0]-[4]
#   bash downstream/run_downstream.sh ood            # 只跑 OOD
#   bash downstream/run_downstream.sh cross complete # 跑指定若干步
# 可选步骤名：pre(自检) | ood | cross | complete | verify
# 环境变量：PY 覆盖解释器（默认 python），BEST 覆盖 ckpt 路径。

set -uo pipefail

# 定位 repo 根（脚本在 downstream/ 下），保证相对路径稳定
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY=${PY:-python}
CFG=configs/ccigpt_cifar10_s_rgb_ronly_v2.yaml
CKDIR=experiments/ccigpt_cifar10_s_rgb_ronly_v2/checkpoints
BEST=${BEST:-$CKDIR/best.pth}
COMPLETION_OUT=experiments/completion_grid.png

# ── 步骤定义 ──────────────────────────────────────────────────────

step_pre() {
    echo "== [0] 预检 self_test（coder + OOD 数学）=="
    "$PY" scripts/verify_lossless.py --self_test
    "$PY" scripts/ood_detect.py --self_test
}

step_ood() {
    echo "== [1] OOD typicality（svhn,cifar100）=="
    "$PY" scripts/ood_detect.py --config "$CFG" --checkpoint "$BEST" \
        --ood svhn,cifar100 --ref_images 2000 \
        --json_out demo/data/ood.json
}

step_cross() {
    echo "== [2] 跨数据集泛化 bpd（cifar10 in-domain + cifar100 / svhn / stl10）=="
    # cifar10 in-domain 单 ckpt 基线（无 ensemble/TTA），与各 override 同协议可比，
    # 作前端 /api/transfer 面板的对照行
    echo "-- in-domain → cifar10 --"
    "$PY" scripts/evaluate.py --config "$CFG" --checkpoint "$BEST" \
        --json_out demo/data/transfer.json
    for ds in cifar100 svhn stl10; do
        echo "-- override → $ds --"
        "$PY" scripts/evaluate.py --config "$CFG" --checkpoint "$BEST" \
            --dataset_override "$ds" --json_out demo/data/transfer.json
    done
}

step_complete() {
    echo "== [3] 图像补全 demo（4 张, keep_frac=0.5）=="
    "$PY" scripts/complete_image.py --config "$CFG" --checkpoint "$BEST" \
        --num_images 4 --keep_frac 0.5 --temperature 1.0 --top_k 100 \
        --out "$COMPLETION_OUT"
}

step_verify() {
    echo "== [4] 真实可解性 roundtrip（2 张, bit-identical 断言）=="
    "$PY" scripts/verify_lossless.py --config "$CFG" --checkpoint "$BEST" \
        --num_images 2
}

# ── 调度 ─────────────────────────────────────────────────────────

# self_test 是地基：先确认 coder/数学没问题，失败就别浪费 GPU 跑后面
preflight() {
    if ! step_pre; then
        echo "!! 预检 self_test 失败，终止（coder/数学有问题，先修）"
        exit 1
    fi
}

run_step() {
    local name="$1"
    local t0=$SECONDS
    local rc=0
    case "$name" in
        pre)      step_pre || rc=$? ;;
        ood)      step_ood || rc=$? ;;
        cross)    step_cross || rc=$? ;;
        complete) step_complete || rc=$? ;;
        verify)   step_verify || rc=$? ;;
        *) echo "!! 未知步骤: $name（可选: pre ood cross complete verify）"; return 2 ;;
    esac
    local dt=$((SECONDS - t0))
    if [ "$rc" -eq 0 ]; then
        echo ">> [$name] OK（用时 ${dt}s）"
    else
        echo ">> [$name] 失败 rc=$rc（用时 ${dt}s）—— 继续后续步骤"
    fi
    FAILED["$name"]=$rc
    return 0
}

declare -A FAILED

if [ "$#" -eq 0 ]; then
    STEPS=(ood cross complete verify)     # 默认 ready 全套（pre 由 preflight 单独跑）
    preflight
else
    STEPS=("$@")
    # 显式指定步骤时，若没点 pre 也先做一次轻量自检兜底
    case " $* " in *" pre "*) : ;; *) preflight ;; esac
fi

for s in "${STEPS[@]}"; do
    run_step "$s"
done

# ── 汇总 ─────────────────────────────────────────────────────────
echo
echo "================ 汇总 ================"
nfail=0
for s in "${STEPS[@]}"; do
    rc=${FAILED[$s]:-NA}
    if [ "$rc" = "0" ]; then
        echo "  [$s] ✅"
    else
        echo "  [$s] ❌ rc=$rc"
        nfail=$((nfail + 1))
    fi
done
echo "====================================="
[ "$nfail" -eq 0 ] && echo "全部通过。" || { echo "$nfail 个步骤失败。"; exit 1; }
