"""Plot the supplementary current-protocol dataset comparison."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results/extended_evidence_20261011"))
    args = parser.parse_args()
    path = args.root / "additional_dataset_summary.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    rows = list(csv.DictReader(path.open(encoding="utf-8", newline="")))
    if not rows:
        raise ValueError("no complete supplementary rows")
    datasets = sorted({row["dataset"] for row in rows})
    methods = sorted({row["method"] for row in rows})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    width = 0.8 / max(1, len(methods))
    x = np.arange(len(datasets))
    for idx, method in enumerate(methods):
        selected = {row["dataset"]: row for row in rows if row["method"] == method}
        f1 = [float(selected[d]["final_f1_mean"]) for d in datasets]
        acc = [float(selected[d]["accuracy_mean"]) for d in datasets]
        offset = (idx - (len(methods) - 1) / 2) * width
        axes[0].bar(x + offset, f1, width, label=method)
        axes[1].bar(x + offset, acc, width, label=method)
    for ax, title, ylabel in ((axes[0], "Additional datasets: Final Macro-F1", "Macro-F1 (%)"),
                              (axes[1], "Additional datasets: Accuracy", "Accuracy (%)")):
        ax.set_xticks(x, datasets)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    axes[1].legend(fontsize=8, frameon=False)
    out = args.root / "additional_dataset_comparison.png"
    fig.savefig(out, dpi=180)
    print(out)


if __name__ == "__main__":
    main()
