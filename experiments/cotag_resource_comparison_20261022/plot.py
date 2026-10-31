"""Standalone figures from verified construction-level CSV rows."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main(args):
    runs = read(args.analysis / "runs.csv")
    old = read(args.previous / "runs.csv")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.spines.left": False, "pdf.fonttype": 42})
    arms = ("original", "incremental_q", "full_history", "cat_cctp", "gecc_hard")
    labels = ("CoTAG-Q, original storage", "CoTAG-Q, class blocks",
              "CoTAG-Q, full history", "CaT adaptation", "GECC adaptation")
    colors = ("#a1a8b0", "#157f78", "#477cb3", "#bc7048", "#815e9f")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True, layout="constrained")
    for ax, dataset in zip(axes, ("arxiv", "sportsfit")):
        for y, (arm, color) in enumerate(zip(arms, colors)):
            selected = ([r for r in old if r["dataset"] == dataset and r["variant"] == "original"]
                        if arm == "original" else [r for r in runs if r["dataset"] == dataset and r["arm"] == arm])
            assert len(selected) == 3
            values = np.array([float(r["peak_gib"]) for r in selected])
            mean = values.mean()
            ax.barh(y, mean, color=color, height=.58, alpha=.88)
            ax.scatter(values, y + np.array([-.13, 0, .13]), color="#20252a", s=15, zorder=3)
            ax.text(values.max() + .28, y, f"{mean:.2f}", va="center", fontsize=10)
        ax.set_yticks(range(len(arms)), labels)
        ax.set_xlim(0, 28)
        ax.set_xlabel("Peak allocated GPU memory (GiB)")
        ax.set_title("ArXiv: three class stages" if dataset == "arxiv" else "SportsFit: five class stages", loc="left", pad=12)
        ax.grid(axis="x", color="#e6e9ec", linewidth=.7)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
    axes[0].invert_yaxis()
    axes[1].tick_params(labelleft=True)
    fig.suptitle("Construction memory: implementation and access comparisons", fontsize=13, weight="bold")
    fig.supxlabel("Bars: mean over three constructions. Dots: individual constructions. Full history relaxes raw-data access.", fontsize=9)
    fig.savefig(args.analysis / "construction_memory.png", dpi=180, bbox_inches="tight")
    fig.savefig(args.analysis / "construction_memory.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for ax, dataset in zip(axes, ("arxiv", "sportsfit")):
        for arm, label, color in zip(arms[1:], labels[1:], colors[1:]):
            selected = [r for r in runs if r["dataset"] == dataset and r["arm"] == arm]
            x = np.array([float(r["peak_gib"]) for r in selected])
            y = np.array([float(r["f1"]) for r in selected])
            ax.scatter(x, y, color=color, alpha=.4, s=25)
            ax.errorbar(x.mean(), y.mean(), yerr=y.std(ddof=1), marker="o", capsize=4,
                        color=color, label=label, markersize=7)
        ax.set_xscale("log")
        ax.set_xlabel("Construction peak (GiB, log scale)")
        ax.set_ylabel("Final report-dev Macro-F1 (%)")
        ax.set_title("ArXiv" if dataset == "arxiv" else "SportsFit", loc="left")
        ax.grid(color="#e6e9ec", linewidth=.7)
        ax.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False, fontsize=9)
    fig.suptitle("Construction memory and downstream prediction quality\n"
                 "Dots: construction means over three students. Error bars: construction SD.",
                 fontsize=11, weight="bold")
    fig.savefig(args.analysis / "quality_memory.png", dpi=180, bbox_inches="tight")
    fig.savefig(args.analysis / "quality_memory.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    main(parser.parse_args())
