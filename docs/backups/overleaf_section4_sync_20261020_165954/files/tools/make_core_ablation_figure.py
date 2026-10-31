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
CONTROLS = (
    ("uq_fixed_q", "Fixed-Q", "#087F8C", "o"),
    ("uq_no_dm", "No-DM", "#B75330", "s"),
    ("uq_no_gm", "No-GM", "#586579", "D"),
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
                or row["right"] not in {control[0] for control in CONTROLS}
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
    expected = {(name, control[0]) for name, _ in DATASETS for control in CONTROLS}
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
    fig, ax = plt.subplots(figsize=(7.1, 2.55))
    all_effects = list(effects.values())
    lower = min(0.0, *(effect["low"] for effect in all_effects))
    upper = max(0.0, *(effect["high"] for effect in all_effects))
    pad = (upper - lower) * 0.08
    ax.set_xlim(lower - pad, upper + pad)
    ax.set_ylim(-0.48, 2.48)
    ax.axvline(0, color="#626B73", linewidth=0.9, linestyle="--", zorder=1)
    offsets = {"uq_fixed_q": 0.18, "uq_no_dm": 0.0, "uq_no_gm": -0.18}
    for control, label, color, marker in CONTROLS:
        for index, (name, dataset_label) in enumerate(DATASETS):
            effect = effects[(name, control)]
            y = 2 - index + offsets[control]
            mean = effect["mean"]
            ax.errorbar(
                mean,
                y,
                xerr=[[mean - effect["low"]], [effect["high"] - mean]],
                fmt=marker,
                markersize=4.2,
                color=color,
                markerfacecolor=color,
                markeredgewidth=0.7,
                linewidth=1.15,
                capsize=2.8,
                label=label if index == 0 else None,
                zorder=3,
            )
            ha = "left" if mean >= 0 else "right"
            dx = 4 if mean >= 0 else -4
            ax.annotate(
                f"{mean:+.2f}",
                (mean, y),
                xytext=(dx, 5),
                textcoords="offset points",
                ha=ha,
                va="bottom",
                fontsize=7.2,
                color=color,
            )
    ax.grid(axis="x", color="#E1E4E8", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.set_yticks([2, 1, 0], [label for _, label in DATASETS])
    ax.set_xlabel("Full constructor minus control (Macro-F1 points)", labelpad=6)
    ax.legend(loc="upper right", ncol=3, frameon=False, handletextpad=0.4,
              columnspacing=1.0, borderaxespad=0.0)
    ax.tick_params(axis="both", length=0, pad=4)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#AAB1B8")
    fig.subplots_adjust(left=0.105, right=0.995, bottom=0.24, top=0.88)
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
        "interpretation": "Post-hoc fixed-mask controls; one order, three construction cells; paired t CIs. One compact Macro-F1 plot; Accuracy remains in the table.",
        "effects": list(effects.values()),
    }
    (args.outdir / "core_ablation_effects.source.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved core ablation PDF/PNG and provenance from {len(effects)} paired effects.")


if __name__ == "__main__":
    main()
