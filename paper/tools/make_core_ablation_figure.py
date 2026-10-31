"""Plot frozen paired component effects without changing experiment reports."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PAPER = Path(__file__).resolve().parents[1]
PROJECT = PAPER.parent
DEFAULT_SOURCE = (
    PROJECT
    / "results"
    / "uq_component_ablations_20261014"
    / "report_utc_20261013_201811_267213"
    / "pairs.csv"
)
DATASETS = (("arxiv", "ArXiv"), ("sportsfit", "SportsFit"), ("bookhis", "BookHis"))
PANELS = (
    ("uq_fixed_q", "(a) Learning Q", "Full - fixed-Q (F1 points)", "#087F8C"),
    ("uq_no_dm", "(b) Distribution matching", "Full - no-DM (F1 points)", "#B75330"),
    ("uq_no_gm", "(c) Gradient matching", "Full - no-GM (F1 points)", "#586579"),
)


def read_effects(source: Path) -> dict[tuple[str, str], dict]:
    effects = {}
    with source.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if (
                row["family"] != "original_loss"
                or row["left"] != "uq_dm_edgefree"
                or row["split"] != "sealed_test"
                or row["metric"] != "final_f1"
                or row["right"] not in {panel[0] for panel in PANELS}
            ):
                continue
            key = (row["dataset"], row["right"])
            if key in effects:
                raise ValueError(f"Duplicate component effect: {key}")
            if int(row["cells"]) != 3:
                raise ValueError(f"Expected three construction cells: {key}")
            effect = {
                "dataset": row["dataset"],
                "control": row["right"],
                "cells": int(row["cells"]),
                "mean": float(row["mean_delta"]),
                "low": float(row["ci_low"]),
                "high": float(row["ci_high"]),
                "positive_cells": int(row["positive_cells"]),
                "scope": row["scope"],
            }
            if not effect["low"] <= effect["mean"] <= effect["high"]:
                raise ValueError(f"Invalid interval: {key}")
            effects[key] = effect
    expected = {(name, panel[0]) for name, _ in DATASETS for panel in PANELS}
    if effects.keys() != expected:
        raise ValueError(f"Unexpected component records: {effects.keys() ^ expected}")
    return effects


def draw(effects: dict, outdir: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.15), sharey=True)
    for ax, (control, title, xlabel, color) in zip(axes, PANELS):
        records = [effects[(name, control)] for name, _ in DATASETS]
        lower = min(0.0, *(effect["low"] for effect in records))
        upper = max(0.0, *(effect["high"] for effect in records))
        pad = (upper - lower) * 0.12
        ax.set_xlim(lower - pad, upper + pad)
        ax.set_ylim(-0.42, 2.65)
        ax.axvline(0, color="#626B73", linewidth=0.8, linestyle="--", zorder=1)
        for index, effect in enumerate(records):
            y = 2 - index
            mean = effect["mean"]
            ax.errorbar(
                mean,
                y,
                xerr=[[mean - effect["low"]], [effect["high"] - mean]],
                fmt="o",
                markersize=4.5,
                color=color,
                linewidth=1.2,
                capsize=3,
                zorder=3,
            )
            ax.annotate(
                f"{mean:+.2f}",
                (mean, y),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.5,
                color=color,
            )
        ax.set_title(title, loc="left", pad=10, fontweight="bold")
        ax.set_xlabel(xlabel, labelpad=6)
        ax.grid(axis="x", color="#E1E4E8", linewidth=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.set_yticks([2, 1, 0], [label for _, label in DATASETS])
        ax.tick_params(axis="both", length=0, pad=4)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color("#AAB1B8")
    fig.subplots_adjust(left=0.102, right=0.995, bottom=0.27, top=0.80, wspace=0.24)
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        outdir / "core_ablation_effects.pdf",
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    fig.savefig(outdir / "core_ablation_effects.png", bbox_inches="tight", dpi=240)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--outdir", type=Path, default=PAPER / "figures")
    args = parser.parse_args()
    effects = read_effects(args.source)
    draw(effects, args.outdir)
    provenance = {
        "source": args.source.resolve().as_posix(),
        "sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "filter": {
            "family": "original_loss",
            "left": "uq_dm_edgefree",
            "split": "sealed_test",
            "metric": "final_f1",
        },
        "interpretation": "Post-hoc fixed-mask controls; one order, three construction cells; paired t CIs.",
        "effects": list(effects.values()),
    }
    (args.outdir / "core_ablation_effects.source.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved core ablation PDF/PNG and provenance from {len(effects)} paired effects.")


if __name__ == "__main__":
    main()
