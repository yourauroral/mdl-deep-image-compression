#!/usr/bin/env python3
"""Render tracked report figures from versioned JSON and CSV inputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
DEMO_DATA = ROOT / "demo" / "data"


def _method_label(name: str) -> str:
    replacements = {
        "Sparse Transformer": "Sparse\nTransformer",
        "Image Transformer": "Image\nTransformer",
        "CC-iGPT R-only v2 (Ours)": "CC-iGPT v2\n(hist. ens.+TTA)",
    }
    return replacements.get(name, name.replace(" (Ours)", ""))


def render_cifar_baselines() -> None:
    with open(DEMO_DATA / "metrics.json", encoding="utf-8") as handle:
        payload = json.load(handle)
    methods = [
        method for method in payload["methods"]
        if method["bpd"] is not None and method["name"] != "CC-iGPT R-only v1 (Ours)"
    ]
    methods.sort(key=lambda method: method["bpd"])

    labels = [_method_label(method["name"]) for method in methods]
    values = [method["bpd"] for method in methods]
    colors = [
        "#f2b134" if "(Ours)" in method["name"] else "#b9c0e8"
        for method in methods
    ]

    fig, axis = plt.subplots(figsize=(10.5, 5.2))
    bars = axis.bar(labels, values, color=colors, edgecolor="#4854b8", width=0.55)
    axis.set_title("CIFAR-10 RGB-bit-exact", fontsize=16, weight="bold")
    axis.set_ylabel("bits/dim")
    axis.set_ylim(min(values) - 0.07, max(values) + 0.08)
    axis.grid(axis="y", linestyle="--", alpha=0.25)
    axis.set_axisbelow(True)

    for bar, method in zip(bars, methods):
        params = method.get("params", "not reported")
        bpd = (
            f'{method["bpd"]:.4f}'
            if method["name"] == "CC-iGPT R-only v2 (Ours)"
            else f'{method["bpd"]:.2f}'
        )
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            method["bpd"] + 0.008,
            f"{bpd}\n{params}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.text(
        0.5,
        0.02,
        "CC-iGPT 2.8296 is a historical 3-checkpoint probability mixture + hflip "
        "(6 forwards/image), not the pending formal single-model result.",
        ha="center",
        fontsize=9,
        color="#9a5d00",
    )
    fig.subplots_adjust(left=0.09, right=0.98, top=0.88, bottom=0.22)
    fig.savefig(FIGURES / "baseline.png", dpi=180, facecolor="white")
    plt.close(fig)


def _optional_float(value: str) -> float | None:
    return float(value) if value.strip() else None


def render_cifar_training_curve() -> None:
    rows = []
    with open(FIGURES / "training_curves_v2.csv", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append({
                "epoch": int(row["epoch"]),
                "train_bpd": _optional_float(row["train_bpd"]),
                "val_bpd": _optional_float(row["val_bpd"]),
            })

    train_rows = [row for row in rows if row["train_bpd"] is not None]
    val_rows = [row for row in rows if row["val_bpd"] is not None]
    best = min(val_rows, key=lambda row: row["val_bpd"])

    fig, axis = plt.subplots(figsize=(10.5, 4.8))
    axis.plot(
        [row["epoch"] for row in train_rows],
        [row["train_bpd"] for row in train_rows],
        color="#4854b8",
        linewidth=1.4,
        label="train bpd",
    )
    axis.plot(
        [row["epoch"] for row in val_rows],
        [row["val_bpd"] for row in val_rows],
        color="#d97706",
        marker="o",
        markersize=2.5,
        linewidth=1.2,
        label="validation bpd",
    )
    axis.axvline(170, color="#d97706", linestyle="--", linewidth=1, alpha=0.7)
    axis.scatter([best["epoch"]], [best["val_bpd"]], color="#c2410c", zorder=3)
    axis.annotate(
        f'best in-training val: ep{best["epoch"]} = {best["val_bpd"]:.4f}',
        xy=(best["epoch"], best["val_bpd"]),
        xytext=(105, 3.05),
        arrowprops={"arrowstyle": "->", "color": "#c2410c"},
        color="#9a3412",
        fontsize=9,
    )
    axis.text(171, 5.2, "SWA start", rotation=90, va="top", color="#9a5d00")
    axis.text(2, 5.25, "ep1 train bpd = 12.074 (clipped)", color="#4854b8", fontsize=9)
    axis.set_title("CC-iGPT v2 R-only training curve", fontsize=15, weight="bold")
    axis.set_xlabel("epoch")
    axis.set_ylabel("bits/dim")
    axis.set_xlim(1, 200)
    axis.set_ylim(2.4, 5.4)
    axis.grid(linestyle="--", alpha=0.22)
    axis.legend(loc="upper right")

    fig.text(
        0.5,
        0.015,
        "In-training metrics only. CIFAR-10 formal single-checkpoint/no-TTA evaluation is pending; "
        "the historical ensemble+TTA diagnostic is not plotted as a training metric.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    fig.subplots_adjust(left=0.09, right=0.98, top=0.88, bottom=0.2)
    fig.savefig(
        FIGURES / "CC-iGPT-R-only-training-curves.png",
        dpi=180,
        facecolor="white",
    )
    plt.close(fig)


def main() -> None:
    render_cifar_baselines()
    render_cifar_training_curve()


if __name__ == "__main__":
    main()
