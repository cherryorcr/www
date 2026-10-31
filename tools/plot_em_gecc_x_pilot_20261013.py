from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.csv.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    methods = ["gecc_hard", "unified_q_ef", "em_gecc_x", "em_no_gm", "em_no_geometry"]
    labels = ["GECC-hard + nearest-5 TAG*", "Unified-Q pilot (DM off)*",
              "EM-GECC-X", "EM-no-GM", "EM-no-geometry"]
    datasets = ["cora", "citeseer"]
    colors = ["#4c78a8", "#f58518", "#e45756", "#72b7b2", "#54a24b"]
    indexed = {(r["dataset"], r["method"]): r for r in rows}
    expected = {(d, m) for d in datasets for m in methods}
    if len(indexed) != len(rows) or set(indexed) != expected:
        raise ValueError("plot requires exactly one final row per dataset/method")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.9))
    fig.subplots_adjust(top=0.73, bottom=0.24, left=0.06, right=0.98, wspace=0.18)
    x = np.arange(len(datasets))
    width = 0.15
    for j, (method, label, color) in enumerate(zip(methods, labels, colors)):
        for ax, metric in zip(axes, ("f1", "accuracy")):
            values = [float(indexed[(d, method)][metric]) for d in datasets]
            errors = [float(indexed[(d, method)][metric + "_std"]) for d in datasets]
            if any(not np.isfinite(v) or v < 0 for v in errors):
                raise ValueError("sample SD must be finite and nonnegative")
            bars = ax.bar(x + (j - 2) * width, values, width, label=label,
                          color=color, yerr=errors, capsize=3,
                          error_kw={"elinewidth": 1, "ecolor": "#303030"}, zorder=3)
            ax.bar_label(bars, labels=[f"{v:.1f}" for v in values], padding=3, fontsize=8)
    for ax, title, ylabel in zip(axes, ["Final report-dev Macro-F1", "Final report-dev Accuracy"], ["Macro-F1 (%)", "Accuracy (%)"]):
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels([d.title() for d in datasets])
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.25, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 0.93),
               ncol=3, frameon=False, fontsize=10)
    fig.suptitle("EM-GECC-X screening pilot: final-stage report-dev results", fontsize=14, y=0.98)
    fig.text(0.06, 0.10,
             "Bars: mean; error bars: sample SD across 3 paired Student seeds. One class order and one construction seed.\n"
             "* GECC uses adapted nearest-5 text attachment; Unified-Q uses legacy code with both DM weights off.\n"
             "Not an official reproduction or a frozen-main-method comparison; no sealed test evaluated.",
             fontsize=10, va="center", color="#303030")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
