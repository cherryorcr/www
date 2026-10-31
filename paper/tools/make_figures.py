from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


OUT = Path(__file__).resolve().parents[1] / "figures"
OUT.mkdir(parents=True, exist_ok=True)


DATASETS = ["WikiCS", "SportsFit", "BookHis"]
METHODS = ["CoTAG-Q", "GECC-hard-TAG", "CaT-CCTP"]
COLORS = ["#1f77b4", "#d62728", "#2ca02c"]

F1 = np.array(
    [
        [79.86, 79.08, 80.35],
        [78.60, 77.32, 73.75],
        [41.97, 41.23, 24.83],
    ]
)
ACC = np.array(
    [
        [82.85, 82.22, 83.44],
        [83.93, 85.52, 81.94],
        [80.34, 79.97, 75.45],
    ]
)


def grouped(values, ylabel, filename):
    x = np.arange(len(DATASETS))
    width = 0.24
    fig, ax = plt.subplots(figsize=(8.4, 4.4), dpi=220)
    for idx, (method, color) in enumerate(zip(METHODS, COLORS)):
        bars = ax.bar(x + (idx - 1) * width, values[:, idx], width, label=method, color=color)
        ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=7)
    ax.set_xticks(x, DATASETS)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=3, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / f"{filename}.png", bbox_inches="tight")
    fig.savefig(OUT / f"{filename}.pdf", bbox_inches="tight")
    plt.close(fig)


grouped(F1, "Final Macro-F1 (%)", "primary_final_f1")
grouped(ACC, "Accuracy (%)", "primary_accuracy")


SAGE_DATASETS = ["BookHis", "SportsFit", "WikiCS"]
SAGE = np.array(
    [
        [36.83, 35.50, 29.24],
        [75.27, 74.91, 77.39],
        [79.66, 78.74, 80.91],
    ]
)

fig, ax = plt.subplots(figsize=(8.4, 4.4), dpi=220)
x = np.arange(len(SAGE_DATASETS))
width = 0.24
for idx, (method, color) in enumerate(zip(METHODS, COLORS)):
    bars = ax.bar(x + (idx - 1) * width, SAGE[:, idx], width, label=method, color=color)
    ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=7)
ax.set_xticks(x, SAGE_DATASETS)
ax.set_ylabel("GraphSAGE transfer Final Macro-F1 (%)")
ax.set_ylim(0, 100)
ax.grid(axis="y", alpha=0.25)
ax.legend(frameon=False, ncol=3, loc="upper left")
fig.tight_layout()
fig.savefig(OUT / "sage_transfer_f1.png", bbox_inches="tight")
fig.savefig(OUT / "sage_transfer_f1.pdf", bbox_inches="tight")
plt.close(fig)


def flow_chart():
    fig, ax = plt.subplots(figsize=(12, 4.2), dpi=220)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.2)
    ax.axis("off")
    boxes = [
        (0.2, 2.0, 1.8, 1.1, "Current delta\n+ frozen prefix", "#e8f1fb"),
        (2.5, 2.0, 1.8, 1.1, "Class slots\nand prototypes", "#edf7ed"),
        (4.8, 2.0, 1.8, 1.1, "Masked\nSinkhorn Q", "#fff3df"),
        (7.1, 2.0, 1.8, 1.1, "X pooling\n+ text route", "#f5e9f7"),
        (9.4, 2.0, 2.2, 1.1, "In-loop losses\nGM + DM + prop", "#fbe9e7"),
        (4.8, 0.35, 1.8, 0.9, "Hard text\nmaterialization", "#e9f5f5"),
        (7.1, 0.35, 1.8, 0.9, "Append new\nmemory rows", "#e9f5f5"),
    ]
    for x, y, w, h, label, color in boxes:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.03,rounding_size=0.08",
                linewidth=1.1,
                edgecolor="#3b4a5a",
                facecolor=color,
            )
        )
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=9)
    arrows = [
        ((2.0, 2.55), (2.5, 2.55)),
        ((4.3, 2.55), (4.8, 2.55)),
        ((6.6, 2.55), (7.1, 2.55)),
        ((8.9, 2.55), (9.4, 2.55)),
        ((10.5, 2.0), (5.7, 1.25)),
        ((6.6, 0.8), (7.1, 0.8)),
    ]
    for start, end in arrows:
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=12,
                linewidth=1.1,
                color="#52606d",
            )
        )
    ax.text(6.0, 3.55, "CoTAG-Q: one assignment couples graph-aware features and real-text routing",
            ha="center", va="center", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "unified_q_flow.png", bbox_inches="tight")
    fig.savefig(OUT / "unified_q_flow.pdf", bbox_inches="tight")
    plt.close(fig)


flow_chart()


EXTRA_DATASETS = ["ArXiv", "Cora"]
EXTRA = np.array(
    [
        [52.25, 51.09, 36.74],
        [82.44, 83.08, 82.76],
    ]
)

fig, ax = plt.subplots(figsize=(7.2, 4.1), dpi=220)
x = np.arange(len(EXTRA_DATASETS))
for idx, (method, color) in enumerate(zip(METHODS, COLORS)):
    bars = ax.bar(x + (idx - 1) * width, EXTRA[:, idx], width, label=method, color=color)
    ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=7)
ax.set_xticks(x, EXTRA_DATASETS)
ax.set_ylabel("Final Macro-F1 (%)")
ax.set_ylim(0, 100)
ax.grid(axis="y", alpha=0.25)
ax.legend(frameon=False, ncol=3, loc="upper left")
fig.tight_layout()
fig.savefig(OUT / "additional_final_f1.png", bbox_inches="tight")
fig.savefig(OUT / "additional_final_f1.pdf", bbox_inches="tight")
plt.close(fig)


# Internal component ablations are kept separate from external baselines.
ABLATION_DATASETS = ["WikiCS", "SportsFit", "BookHis"]
ABLATION_METHODS = ["Shared-Q", "Split-Q", "No residual"]
ABLATION_COLORS = ["#1f77b4", "#ff7f0e", "#7f7f7f"]
ABLATION_F1 = np.array(
    [
        [80.38, 80.93, 80.96],
        [78.92, 78.82, 78.52],
        [41.97, 42.50, 42.39],
    ]
)
ABLATION_ACC = np.array(
    [
        [83.09, 83.47, 83.39],
        [84.53, 84.56, 83.96],
        [79.91, 80.19, 80.44],
    ]
)


def ablation_grouped(values, ylabel, filename):
    x = np.arange(len(ABLATION_DATASETS))
    width = 0.24
    fig, ax = plt.subplots(figsize=(8.4, 4.4), dpi=220)
    for idx, (method, color) in enumerate(zip(ABLATION_METHODS, ABLATION_COLORS)):
        bars = ax.bar(x + (idx - 1) * width, values[:, idx], width, label=method, color=color)
        ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=7)
    ax.set_xticks(x, ABLATION_DATASETS)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=3, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / f"{filename}.png", bbox_inches="tight")
    fig.savefig(OUT / f"{filename}.pdf", bbox_inches="tight")
    plt.close(fig)


ablation_grouped(ABLATION_F1, "Final Macro-F1 (%)", "ablation_q_f1")
ablation_grouped(ABLATION_ACC, "Accuracy (%)", "ablation_q_accuracy")


RETENTION_DATASETS = ["ArXiv", "SportsFit", "BookHis"]
RETENTION = np.array(
    [
        [109.50, 100.03],
        [101.50, 96.36],
        [100.45, 99.35],
    ]
)

fig, ax = plt.subplots(figsize=(8.4, 4.2), dpi=220)
x = np.arange(len(RETENTION_DATASETS))
width = 0.32
for idx, (metric, color) in enumerate(zip(["F1 retention", "Accuracy retention"], ["#1f77b4", "#2ca02c"])):
    bars = ax.bar(x + (idx - 0.5) * width, RETENTION[:, idx], width, label=metric, color=color)
    ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=7)
ax.axhline(100.0, color="#555555", linewidth=1.0, linestyle="--")
ax.set_xticks(x, RETENTION_DATASETS)
ax.set_ylabel("5% memory / full-history (%)")
ax.set_ylim(90, 115)
ax.grid(axis="y", alpha=0.25)
ax.legend(frameon=False, ncol=2, loc="upper left")
fig.tight_layout()
fig.savefig(OUT / "full_history_retention.png", bbox_inches="tight")
fig.savefig(OUT / "full_history_retention.pdf", bbox_inches="tight")
plt.close(fig)
