"""Plot source-backed paired effects and finite-budget retention."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


PAPER = Path(__file__).resolve().parents[1]
PROJECT = PAPER.parent
DEFAULT_PAIRS = PROJECT / "results/uq_final_evidence_report_20261013_132047/pairs.csv"
DEFAULT_RETENTION = (
    PROJECT / "results/cotag_q_four_gap_20261015_v1"
    / "report_final_20261015T190000/fullhistory_retention.csv"
)
DATASETS = (("arxiv", "ArXiv"), ("sportsfit", "SportsFit"), ("bookhis", "BookHis"))
BASELINES = (
    ("gecc_hard", "(a) vs. GECC-hard-TAG", "#2978A3"),
    ("cat_cctp", "(b) vs. CaT-CCTP", "#B76735"),
)


def read_effects(source: Path) -> dict:
    effects = {}
    with source.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if (
                row["category"] != "confirmation"
                or row["split"] != "sealed_test"
                or row["left"] != "uq_dm_edgefree"
                or row["metric"] != "final_f1"
                or row["right"] not in {item[0] for item in BASELINES}
            ):
                continue
            key = (row["dataset"], row["right"])
            if key in effects:
                raise ValueError(f"Duplicate effect: {key}")
            effect = {
                "dataset": row["dataset"],
                "baseline": row["right"],
                "cells": int(row["cells"]),
                "mean": float(row["mean_delta"]),
                "low": float(row["ci_low"]),
                "high": float(row["ci_high"]),
                "positive_cells": int(row["positive_cells"]),
            }
            if effect["cells"] != 9 or not effect["low"] <= effect["mean"] <= effect["high"]:
                raise ValueError(f"Invalid paired effect: {key}")
            effects[key] = effect
    expected = {(name, item[0]) for name, _ in DATASETS for item in BASELINES}
    if effects.keys() != expected:
        raise ValueError(f"Missing or extra paired effects: {effects.keys() ^ expected}")
    return effects


def read_retention(source: Path) -> dict:
    retention = {}
    with source.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if (
                row["split"] != "sealed_test"
                or row["reference"] != "full_topology"
                or row["metric"] not in {"final_f1", "accuracy"}
            ):
                continue
            key = (row["dataset"], row["metric"])
            if key in retention:
                raise ValueError(f"Duplicate retention record: {key}")
            retention[key] = float(row["ratio_of_means"])
    expected = {(name, metric) for name, _ in DATASETS for metric in ("final_f1", "accuracy")}
    if retention.keys() != expected:
        raise ValueError(f"Missing or extra retention records: {retention.keys() ^ expected}")
    return retention


def style(ax) -> None:
    ax.set_ylim(-0.48, 2.60)
    ax.set_yticks([2, 1, 0], [label for _, label in DATASETS])
    ax.grid(axis="x", color="#E1E5E8", linewidth=0.55, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=0, pad=4)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#AAB1B8")


def draw(effects: dict, retention: dict, outdir: Path) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7.6,
        "axes.labelsize": 7.6, "axes.titlesize": 8,
        "xtick.labelsize": 7.2, "ytick.labelsize": 7.4,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.40))
    fig.patch.set_facecolor("white")
    for ax, (baseline, title, color) in zip(axes[:2], BASELINES):
        records = [effects[(name, baseline)] for name, _ in DATASETS]
        low = min(0.0, *(item["low"] for item in records))
        high = max(0.0, *(item["high"] for item in records))
        span = high - low
        ax.set_xlim(low - span * .10, high + span * .13)
        ax.axvline(0, color="#68727B", linewidth=.8, linestyle="--")
        for index, effect in enumerate(records):
            mean, y = effect["mean"], 2 - index
            ax.errorbar(
                mean, y,
                xerr=[[mean - effect["low"]], [effect["high"] - mean]],
                fmt="o", markersize=4.4, color=color,
                linewidth=1.2, capsize=3, zorder=3,
            )
            ax.annotate(
                f"{mean:+.2f}", (mean, y), xytext=(0, 8),
                textcoords="offset points", ha="center", va="bottom",
                fontsize=7.3, color=color,
            )
        ax.set_title(title, loc="left", pad=10, fontweight="bold")
        ax.set_xlabel("Macro-F1 difference (points)", labelpad=6)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        style(ax)
    ax = axes[2]
    ax.axvline(100, color="#68727B", linewidth=.8, linestyle="--")
    for metric, label, marker, offset, color in (
        ("final_f1", "Macro-F1", "o", .16, "#237D77"),
        ("accuracy", "Accuracy", "s", -.16, "#9B6524"),
    ):
        for index, (name, _) in enumerate(DATASETS):
            value, y = retention[(name, metric)], 2 - index + offset
            ax.scatter(value, y, s=20, marker=marker, color=color,
                       label=label if index == 0 else None, zorder=3)
            ax.annotate(
                f"{value:.2f}", (value, y),
                xytext=(5, 0), textcoords="offset points", ha="left", va="center",
                fontsize=6.9, color=color,
            )
    ax.set_xlim(93.5, 108.0)
    ax.set_xticks([95, 100, 105])
    ax.set_title("(c) Full-history retention", loc="left", pad=10, fontweight="bold")
    ax.set_xlabel("Condensed / full-history (%)", labelpad=6)
    style(ax)
    ax.legend(
        loc="upper center", bbox_to_anchor=(.5, -.31),
        frameon=False, fontsize=7, ncol=2,
        handletextpad=.35, columnspacing=.7,
    )
    fig.subplots_adjust(left=.085, right=.99, bottom=.32, top=.80, wspace=.64)
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / "main_results_overview.pdf", bbox_inches="tight",
                metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(outdir / "main_results_overview.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--retention", type=Path, default=DEFAULT_RETENTION)
    parser.add_argument("--outdir", type=Path, default=PAPER / "figures")
    args = parser.parse_args()
    effects = read_effects(args.pairs)
    retention = read_retention(args.retention)
    draw(effects, retention, args.outdir)
    provenance = {
        "sources": [
            {"path": path.relative_to(PROJECT).as_posix(),
             "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in (args.pairs, args.retention)
        ],
        "effects": list(effects.values()),
        "retention": [
            {"dataset": name, "metric": metric, "ratio_of_means_percent": value}
            for (name, metric), value in retention.items()
        ],
        "interpretation": (
            "Historical sealed_test label denotes post-hoc fixed-mask evidence, "
            "not a fresh blind test. Effects use 9 paired order/construction cells; "
            "each cell averages 3 Student seeds. Retention uses one fixed order "
            "and a common finite Student budget; no lines join unrelated datasets."
        ),
    }
    (args.outdir / "main_results_overview.source.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print("Saved source-backed paired CI and retention panels with provenance.")


if __name__ == "__main__":
    main()
