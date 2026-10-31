"""Plot the final Student x constructor factorial."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def grouped(metric: str, dataset: str):
    rows = read(DATA / "method_means.csv")
    rows = [r for r in rows if r["dataset"] == dataset]
    methods = list(dict.fromkeys(r["method"] for r in rows))
    students = ["Prompt Student", "Legacy Student"]
    values = np.array([[float(next(r for r in rows if r["method"] == m and r["student_name"] == s)[f"{metric}_mean"])
                        for s in students] for m in methods])
    errors = np.array([[float(next(r for r in rows if r["method"] == m and r["student_name"] == s)[f"{metric}_sd_order"])
                        for s in students] for m in methods])
    return methods, students, values, errors


COLORS = ["#1f77b4", "#d95f02", "#2ca02c", "#756bb1"]


def plot_metric(metric: str, filename: str, title: str):
    datasets = ["wikics", "sportsfit", "bookhis"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), sharey=True)
    for ax, dataset in zip(axes, datasets):
        methods, students, values, errors = grouped(metric, dataset)
        x = np.arange(len(methods))
        width = 0.36
        for idx, student in enumerate(students):
            ax.bar(x + (idx - 0.5) * width, values[:, idx], width,
                   yerr=errors[:, idx], capsize=3, label=student if dataset == datasets[0] else None,
                   color=["#4c78a8", "#f58518"][idx], edgecolor="white", linewidth=0.5)
        ax.set_title(dataset)
        ax.set_xticks(x, ["Unified-Q", "GECC", "CaT-CCTP", "Adaptive"], rotation=28, ha="right")
        ax.grid(axis="y", alpha=0.25)
        ax.set_ylim(0, 100)
    axes[0].set_ylabel("percent")
    fig.suptitle(title)
    axes[0].legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_interaction():
    path = DATA / "constructor_interactions.csv"
    rows = read(path if path.exists() else DATA / "unified_q_interactions.csv")
    datasets = ["wikics", "sportsfit", "bookhis"]
    x = np.arange(len(datasets))
    fig, ax = plt.subplots(figsize=(9, 5))
    arms = [("uq_dm_edgefree", "Unified-Q", "#1f77b4"),
            ("gecc_hard", "GECC", "#2ca02c"),
            ("adaptive_coherent", "Adaptive", "#d95f02")]
    for arm, label, color in arms:
        subset = [r for r in rows if r["arm"] == arm]
        values = [float(next(r for r in subset if r["dataset"] == d)["interaction_mean"])
                  for d in datasets]
        lows = [float(next(r for r in subset if r["dataset"] == d)["ci95_low"])
                for d in datasets]
        highs = [float(next(r for r in subset if r["dataset"] == d)["ci95_high"])
                 for d in datasets]
        ax.errorbar(x, values, yerr=[np.array(values) - np.array(lows),
                                     np.array(highs) - np.array(values)],
                    fmt="o-", color=color, capsize=4, linewidth=1.8,
                    label=label)
    ax.axhline(0, color="#555", linewidth=1)
    ax.set_xticks(x, ["WikiCS", "SportsFit", "BookHis"])
    ax.set_ylabel("(method-CaT) Prompt - (method-CaT) Legacy, pp")
    ax.set_title("Student × constructor interaction")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "student_constructor_interaction_f1.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path,
                        default=ROOT / "results/student_method_factorial_20261010")
    args = parser.parse_args()
    DATA = args.data
    OUT = DATA / "figures"
    OUT.mkdir(parents=True, exist_ok=True)
    plot_metric("final_f1", "final_f1_student_method.png", "Final Macro-F1: Student × construction method")
    plot_metric("accuracy", "accuracy_student_method.png", "Final Accuracy: Student × construction method")
    plot_interaction()
