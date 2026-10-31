from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUT = Path(__file__).resolve().parents[1] / "figures"
OUT.mkdir(parents=True, exist_ok=True)

METHODS = ["CoTAG-Q", "GECC-hard-TAG", "CaT-CCTP"]
COLORS = ["#1f77b4", "#d62728", "#2ca02c"]
DATASETS = ["ArXiv", "SportsFit", "BookHis"]


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


# Exact-5% primary table: nine order/construction cells.
f1 = np.array([
    [52.91, 52.68, 34.98],
    [80.08, 80.57, 74.50],
    [48.81, 45.34, 29.16],
])
f1_sd = np.array([
    [0.73, 0.45, 1.79],
    [0.60, 1.16, 2.14],
    [1.57, 0.74, 1.05],
])
acc = np.array([
    [71.15, 70.94, 60.41],
    [85.52, 89.01, 85.56],
    [81.05, 81.26, 79.23],
])

fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), sharex=True)
x = np.arange(len(DATASETS))
width = 0.24
for j, (method, color) in enumerate(zip(METHODS, COLORS)):
    offset = (j - 1) * width
    bars = axes[0].bar(x + offset, f1[:, j], width, yerr=f1_sd[:, j],
                       capsize=2.5, label=method, color=color,
                       error_kw={"linewidth": 0.8})
    axes[0].bar_label(bars, fmt="%.1f", padding=2, fontsize=7)
    bars = axes[1].bar(x + offset, acc[:, j], width, label=method, color=color)
    axes[1].bar_label(bars, fmt="%.1f", padding=2, fontsize=7)
for ax, ylabel in zip(axes, ["Final Macro-F1 (%)", "Accuracy (%)"]):
    ax.set_xticks(x, DATASETS)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
axes[0].set_title("Macro-F1 (mean $\\pm$ cell SD)")
axes[1].set_title("Accuracy (mean)")
axes[0].legend(frameon=False, fontsize=8, loc="upper left")
fig.suptitle("Exact-5% primary comparison", fontsize=12, fontweight="bold")
fig.tight_layout()
save(fig, "primary_exact5_comparison")


# Same ArXiv memory under the two tested Student graph towers.
students = ["Prompt GCN", "Prompt GraphSAGE"]
transfer_f1 = np.array([
    [52.91, 52.68, 34.98],
    [53.93, 52.60, 50.24],
])
transfer_acc = np.array([
    [71.15, 70.94, 60.41],
    [71.27, 70.62, 70.78],
])
student_colors = ["#4c78a8", "#72b7b2"]
fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.0), sharex=True)
x = np.arange(len(METHODS))
width = 0.34
for i, (student, color) in enumerate(zip(students, student_colors)):
    offset = (i - 0.5) * width
    bars = axes[0].bar(x + offset, transfer_f1[i], width, label=student, color=color)
    axes[0].bar_label(bars, fmt="%.1f", padding=2, fontsize=7)
    bars = axes[1].bar(x + offset, transfer_acc[i], width, label=student, color=color)
    axes[1].bar_label(bars, fmt="%.1f", padding=2, fontsize=7)
for ax, ylabel in zip(axes, ["Final Macro-F1 (%)", "Accuracy (%)"]):
    ax.set_xticks(x, METHODS, rotation=18, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
axes[0].set_title("Macro-F1")
axes[1].set_title("Accuracy")
axes[0].legend(frameon=False, fontsize=8, loc="upper left")
fig.suptitle("ArXiv same-memory transfer", fontsize=11, fontweight="bold")
fig.tight_layout()
save(fig, "arxiv_student_transfer_current")


# Rate sensitivity used to motivate a shared 5% operating point.
rates = [1, 5, 10]
rate_data = {
    "ArXiv": {
        "CoTAG-Q": [51.09, 52.95, 51.47],
        "CaT-CCTP": [37.35, 33.88, 39.61],
        "GECC-hard-TAG": [51.35, 51.93, 51.67],
    },
    "SportsFit": {
        "CoTAG-Q": [79.28, 79.25, np.nan],
        "CaT-CCTP": [72.46, 74.76, np.nan],
        "GECC-hard-TAG": [79.48, 79.17, np.nan],
    },
    "BookHis": {
        "CoTAG-Q": [46.54, 47.39, 45.87],
        "CaT-CCTP": [28.79, 29.25, 28.46],
        "GECC-hard-TAG": [45.48, 44.23, 46.45],
    },
}
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5), sharey=False)
for ax, dataset in zip(axes, DATASETS):
    for method, color in zip(METHODS, COLORS):
        ax.plot(rates, rate_data[dataset][method], marker="o", linewidth=1.8,
                label=method, color=color)
    ax.axvline(5, color="#555555", linestyle="--", linewidth=0.9, alpha=0.8)
    ax.set_title(dataset)
    ax.set_xticks(rates)
    ax.set_xlabel("Cumulative rate (%)")
    ax.set_ylim(20 if dataset == "BookHis" else 30, 85)
    ax.grid(alpha=0.25)
    if dataset == "SportsFit":
        ax.text(10, 42, "OOM", ha="center", va="center", fontsize=8,
                color="#444444")
axes[0].set_ylabel("Final Macro-F1 (%)")
axes[0].legend(frameon=False, fontsize=7, loc="lower left")
fig.suptitle("Retention-rate sensitivity under the frozen configuration",
             fontsize=11, fontweight="bold")
fig.tight_layout()
save(fig, "rate_sensitivity_current")
