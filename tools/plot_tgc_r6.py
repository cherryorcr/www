"""Standalone R6 scientific figures from the audited summary tables."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def save(fig, output, name):
    fig.savefig(output / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(output / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def main(args):
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42})
    summary = json.loads((args.confirmation / "summary.json").read_text())
    assert len(summary["audits"]) == 6 and all(a["status"] == "passed" for a in summary["audits"])
    records = read_csv(args.confirmation / "cells.csv")
    cells = [(d, o) for d in ("cora", "citeseer") for o in ("order19601", "order19602", "order19603")]
    labels = [d.capitalize() + " / " + o.removeprefix("order") for d, o in cells]
    fig, ax = plt.subplots(figsize=(9.5, 5.3), layout="constrained")
    y = np.arange(len(cells))
    for arm, label, color, offset in (("recursive", "R5 recursive", "#ac3344", -.20),
                                      ("k_center", "k-center", "#277da8", 0),
                                      ("kmeans_medoid", "kmeans-medoid", "#52854c", .20)):
        group = [next(r for r in records if (r["dataset"], r["order"], r["stage"], r["arm"]) == (d, o, "2", arm)) for d, o in cells]
        values = np.array([float(r["delta_f1"]) for r in group])
        low = np.array([float(r["paired_ci_low"]) for r in group])
        high = np.array([float(r["paired_ci_high"]) for r in group])
        ax.errorbar(values, y+offset, xerr=[values-low, high-values], fmt="o", markersize=4,
                    capsize=3, linewidth=1.2, label=label, color=color)
    ax.axvline(0, color="#454545", linewidth=1)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Final Macro-F1 difference from C0 (percentage points)")
    ax.set_title("Independent confirmation: all three methods vary across class orders", loc="left", pad=38)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=3, frameon=False)
    ax.grid(axis="x", alpha=.17)
    fig.text(.01, -.02, "Paired 95% t intervals over six student seeds per fixed memory/order. Not construction-seed uncertainty.", fontsize=9)
    save(fig, args.output, "confirmation_paired")
    if args.mechanisms is None:
        return
    mechanism_summary = json.loads((args.mechanisms / "summary.json").read_text())
    assert all(a["status"] == "passed" for a in mechanism_summary["audits"])
    records = read_csv(args.mechanisms / "cells.csv")
    cells = [(d, o) for d in ("cora", "citeseer") for o in ("order19341", "order19342")]
    arms = ("parent_000", "geometry", "random_teacher", "fixed_assignment", "decoupled64", "decoupled128",
            "symmetric_fixed_parent", "symmetric_recursive", "pooled_text", "weighted_edges", "pooled_weighted",
            "frozen_recursive")
    labels = ("C0 parent", "Raw geometry", "Untrained teachers", "Initial assignment",
              "Separate P: 64+64", "Separate P: 128+128", "Symmetric / fixed parent", "Symmetric / recursive",
              "Pooled text embedding", "Weighted edges", "Pooled + weighted", "Frozen R5 artifact")
    idx = {(r["dataset"], r["order"], r["stage"], r["arm"]): float(r["f1"]) for r in records}
    matrix = np.array([[idx[d, o, "2", a] - idx[d, o, "2", "parent_111"] for d, o in cells] for a in arms])
    bound = max(1., float(np.max(np.abs(matrix))))
    fig, ax = plt.subplots(figsize=(8.9, 7.5), layout="constrained")
    image = ax.imshow(matrix, cmap="RdBu", vmin=-bound, vmax=bound, aspect="auto")
    ax.set_xticks(range(len(cells)), [d.capitalize()+"\n"+o.removeprefix("order") for d, o in cells])
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(arms)):
        for j in range(len(cells)):
            ax.text(j, i, f"{matrix[i,j]:+.2f}", ha="center", va="center",
                    color="white" if abs(matrix[i,j]) > .64*bound else "#171717")
    ax.set_title("Development diagnostics: final F1 relative to fresh recursive endpoint", loc="left", pad=12)
    fig.colorbar(image, ax=ax, shrink=.68, label="Macro-F1 difference (pp)")
    fig.text(.01, -.025, "Six-seed means. Single-transition diagnostics except symmetric rollout; pooled text is embedding-only.", fontsize=9)
    save(fig, args.output, "mechanism_matrix")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--mechanisms", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
