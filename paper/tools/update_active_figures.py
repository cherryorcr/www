"""Redraw the two figures used by the active manuscript narrative.

The source values come from the immutable paired-evidence CSV.  This script
only changes presentation labels and the method-flow diagram; it does not
rewrite experiment reports or select a new result.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parents[1] / "figures"
PAIRS = ROOT / "results" / "uq_final_evidence_report_20261013_132047" / "pairs.csv"


def _box(ax, x, y, width, height, label, color):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.03,rounding_size=0.08",
            linewidth=1.1,
            edgecolor="#3b4a5a",
            facecolor=color,
        )
    )
    ax.text(x + width / 2, y + height / 2, label, ha="center", va="center", fontsize=9)


def _arrow(ax, start, end, connectionstyle="arc3"):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.1,
            color="#52606d",
            connectionstyle=connectionstyle,
        )
    )


def draw_flow():
    fig, ax = plt.subplots(figsize=(12.2, 4.7), dpi=220)
    ax.set_xlim(0, 12.2)
    ax.set_ylim(0, 4.7)
    ax.axis("off")

    _box(ax, 0.15, 2.45, 1.85, 1.05, "Current delta\n+ frozen prefix", "#e8f1fb")
    _box(ax, 2.35, 2.45, 1.85, 1.05, "Class slots\nand prototypes", "#edf7ed")
    _box(ax, 4.55, 2.45, 1.85, 1.05, "Masked\nSinkhorn Q", "#fff3df")
    _box(ax, 6.75, 2.45, 2.0, 1.05, "Pooled X\n+ routed text", "#f5e9f7")
    _box(ax, 9.1, 2.45, 2.35, 1.05, "GM + DM + prop\nconstruction losses", "#fbe9e7")
    _box(ax, 4.55, 0.45, 1.85, 0.9, "Hard source IDs\n+ real text", "#e9f5f5")
    _box(ax, 6.75, 0.45, 2.0, 0.9, "Bounded residual R\nfeature correction", "#e9f5f5")
    _box(ax, 9.1, 0.45, 2.35, 0.9, "Append edge-free\nmemory rows", "#e9f5f5")

    _arrow(ax, (2.0, 2.98), (2.35, 2.98))
    _arrow(ax, (4.2, 2.98), (4.55, 2.98))
    _arrow(ax, (6.4, 2.98), (6.75, 2.98))
    _arrow(ax, (8.75, 2.98), (9.1, 2.98))
    _arrow(ax, (5.48, 2.45), (5.48, 1.35))
    _arrow(ax, (6.4, 0.9), (6.75, 0.9))
    _arrow(ax, (8.75, 0.9), (9.1, 0.9))
    _arrow(ax, (10.15, 2.45), (7.75, 1.5), connectionstyle="arc3,rad=0.16")
    _arrow(ax, (7.75, 1.5), (5.48, 2.45), connectionstyle="arc3,rad=-0.16")

    ax.text(
        6.1,
        4.23,
        "CoTAG-Q: one assignment couples graph-aware feature support and real-text routing",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
    )
    ax.text(
        6.1,
        3.86,
        "Current graph messages enter construction; saved memory stores no synthetic adjacency",
        ha="center",
        va="center",
        fontsize=8.5,
        color="#52606d",
    )
    ax.text(8.4, 1.88, "differentiate and update Q, R", ha="center", va="center", fontsize=8)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"unified_q_flow.{suffix}", bbox_inches="tight")
    plt.close(fig)


def draw_paired():
    rows = []
    with PAIRS.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if (
                row["category"] == "confirmation"
                and row["split"] == "sealed_test"
                and row["metric"] == "final_f1"
            ):
                rows.append(row)

    baseline_names = {"cat_cctp": "CaT-CCTP", "gecc_hard": "GECC-hard-TAG"}
    dataset_names = {"arxiv": "ArXiv", "sportsfit": "SportsFit", "bookhis": "BookHis"}
    rows.sort(key=lambda row: (list(dataset_names).index(row["dataset"]), row["right"]))

    fig, ax = plt.subplots(figsize=(8.2, 3.6), layout="constrained")
    for index, row in enumerate(rows):
        color = "#2166ac" if row["right"] == "cat_cctp" else "#c07023"
        mean = float(row["mean_delta"])
        low = float(row["ci_low"])
        high = float(row["ci_high"])
        ax.scatter(mean, index, color=color, zorder=3, s=42)
        ax.hlines(index, low, high, color=color, linewidth=2)
        ax.text(high, index + 0.16, f"  {row['positive_cells']}/{row['cells']} positive", fontsize=8.5)

    ax.axvline(0, color="#555555", linestyle="--", linewidth=1)
    ax.set_yticks(
        range(len(rows)),
        [f"{dataset_names[row['dataset']]}: CoTAG-Q - {baseline_names[row['right']]}" for row in rows],
    )
    ax.set_xlabel("Final Macro-F1 difference (percentage points); paired 95% t CI")
    ax.set_title("Fixed-mask exact-5% paired comparison")
    ax.margins(x=0.3, y=0.45)
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"confirmation_paired.{suffix}", bbox_inches="tight")
    plt.close(fig)


def draw_archived_panels():
    """Refresh archived panels with the paper-facing method name."""
    datasets = ["ArXiv", "SportsFit", "BookHis"]
    methods = ["CoTAG-Q", "GECC-hard-TAG", "CaT-CCTP"]
    f1 = np.array([[52.25, 51.09, 36.74], [78.60, 77.32, 73.75], [41.97, 41.23, 24.83]])
    acc = np.array([[71.59, 70.27, 63.62], [83.93, 85.52, 81.94], [80.34, 79.97, 75.45]])
    colors = ["#1f77b4", "#888888", "#888888"]
    for values, ylabel, filename in ((f1, "Final Macro-F1 (%)", "large_baseline_f1.png"),
                                     (acc, "Accuracy (%)", "large_baseline_accuracy.png")):
        fig, axes = plt.subplots(1, len(datasets), figsize=(13.5, 4.0), sharey=False)
        for axis, dataset, row in zip(axes, datasets, values):
            bars = axis.bar(np.arange(len(methods)), row, color=colors)
            axis.bar_label(bars, fmt="%.1f", padding=2, fontsize=7)
            axis.set_xticks(np.arange(len(methods)), methods, rotation=45, ha="right", fontsize=8)
            axis.set_title(dataset)
            axis.set_ylabel(ylabel)
            axis.grid(axis="y", alpha=0.25)
        fig.suptitle(f"Archived nominal-rate Prompt Student panel: {ylabel}")
        fig.tight_layout()
        fig.savefig(OUT / filename, dpi=180, bbox_inches="tight")
        plt.close(fig)

    source = ROOT / "results" / "unified_q_overnight_20261012" / "construction_summary.csv"
    rows = []
    with source.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row["metric"] == "final_f1" and row["dataset"] in {"sportsfit", "bookhis", "wikics"}:
                rows.append(row)
    display_datasets = [("sportsfit", "SportsFit"), ("bookhis", "BookHis"), ("wikics", "WikiCS")]
    source_methods = ["Unified-Q-EF", "GECC-hard-TAG", "CaT-CCTP"]
    display_methods = ["CoTAG-Q", "GECC-hard", "CaT-CCTP"]
    palette = ["#2f6f9f", "#8a8f98", "#c58c3a"]
    fig, axes = plt.subplots(1, len(display_datasets), figsize=(13.5, 4.0), sharey=False)
    for axis, (key, title) in zip(axes, display_datasets):
        selected = [next(r for r in rows if r["dataset"] == key and r["method"] == method) for method in source_methods]
        means = [float(r["mean"]) for r in selected]
        sds = [float(r["sd"]) for r in selected]
        axis.bar(np.arange(3), means, yerr=sds, capsize=3, color=palette)
        axis.set_xticks(np.arange(3), display_methods, rotation=25)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Final Macro-F1 (%)")
    fig.suptitle("Archived construction/order stability under the matched protocol")
    fig.tight_layout()
    fig.savefig(OUT / "overnight_construction_f1.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    draw_flow()
    draw_paired()
    draw_archived_panels()
