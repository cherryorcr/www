"""Summarize the unified Semantic-Recursive small-graph comparison."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats


LABELS = {
    "c0": "C0 internal control",
    "semantic_recursive": "Semantic-Recursive (ours)",
    "random_real": "Random real-node coreset",
    "k_center_fps": "k-center / greedy FPS",
    "herding": "Herding real-node coreset",
    "medoid": "k-means medoid coreset",
    "raw_full_scratch": "Full raw graph scratch",
    "student_delta_finetune": "Student delta-only fine-tune",
    "student_c0_replay_delta": "Student C0-replay + delta fine-tune",
}


def ci(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    if len(arr) < 2:
        return {"mean": float(arr.mean()) if len(arr) else None,
                "low": None, "high": None, "n": int(len(arr))}
    mean = float(arr.mean())
    half = float(stats.t.ppf(0.975, len(arr) - 1)
                 * arr.std(ddof=1) / math.sqrt(len(arr)))
    return {"mean": mean, "low": mean - half, "high": mean + half,
            "n": int(len(arr))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads((args.input / "results.json").read_text(encoding="utf-8"))
    rows = data["rows"]
    methods = list(dict.fromkeys(r["arm"] for r in rows))
    cells = sorted(set((r["dataset"], r["order"]) for r in rows))

    def row(arm: str, dataset: str, order: str, stage: int = 2) -> dict:
        return next(r for r in rows if r["arm"] == arm
                    and r["dataset"] == dataset
                    and r["order"] == order
                    and r["stage"] == stage)

    cell_final = []
    for dataset, order in cells:
        for arm in methods:
            r = row(arm, dataset, order)
            a = r["aggregate"]
            stage_rows = [row(arm, dataset, order, stage)
                          for stage in range(3)]
            cell_final.append({
                "dataset": dataset,
                "order": order,
                "arm": arm,
                "label": LABELS.get(arm, arm),
                "final_f1": a["mean_f1"],
                "final_accuracy": a["mean_accuracy"],
                "stage_average_f1": float(np.mean([
                    s["aggregate"]["mean_f1"] for s in stage_rows])),
                "stage_average_accuracy": float(np.mean([
                    s["aggregate"]["mean_accuracy"] for s in stage_rows])),
                "old_f1": a["old_f1"],
                "new_f1": a["new_f1"],
                "memory_rows": r["memory_rows"],
                "student_seconds": r["student_wall_seconds"],
            })

    def cell_metric(arm: str, key: str) -> list[float]:
        return [float(next(x for x in cell_final
                           if x["arm"] == arm and x["dataset"] == dataset
                           and x["order"] == order)[key])
                for dataset, order in cells]

    overall = []
    for arm in methods:
        records = [x for x in cell_final if x["arm"] == arm]
        c0_records = [x for x in cell_final if x["arm"] == "c0"]
        deltas = [x["final_f1"] - y["final_f1"]
                  for x, y in zip(records, c0_records)]
        acc_deltas = [x["final_accuracy"] - y["final_accuracy"]
                      for x, y in zip(records, c0_records)]
        seed_diffs = []
        seed_acc_diffs = []
        for dataset, order in cells:
            base = row("semantic_recursive", dataset, order)["per_student"]
            other = row(arm, dataset, order)["per_student"]
            base_by_seed = {int(x["seed"]): x for x in base}
            other_by_seed = {int(x["seed"]): x for x in other}
            for seed in sorted(set(base_by_seed) & set(other_by_seed)):
                seed_diffs.append(other_by_seed[seed]["cumulative_f1"]
                                  - base_by_seed[seed]["cumulative_f1"])
                seed_acc_diffs.append(other_by_seed[seed]["accuracy"]
                                      - base_by_seed[seed]["accuracy"])
        overall.append({
            "arm": arm,
            "label": LABELS.get(arm, arm),
            "final_f1": float(np.mean([x["final_f1"] for x in records])),
            "final_accuracy": float(np.mean([x["final_accuracy"] for x in records])),
            "stage_average_f1": float(np.mean([x["stage_average_f1"] for x in records])),
            "stage_average_accuracy": float(np.mean([x["stage_average_accuracy"] for x in records])),
            "old_f1": float(np.mean([x["old_f1"] for x in records
                                      if x["old_f1"] is not None])),
            "new_f1": float(np.mean([x["new_f1"] for x in records
                                      if x["new_f1"] is not None])),
            "final_f1_cell_std": float(np.std([x["final_f1"] for x in records])),
            "delta_vs_c0_f1": float(np.mean(deltas)),
            "delta_vs_c0_accuracy": float(np.mean(acc_deltas)),
            "positive_cells_vs_c0": int(sum(x > 0 for x in deltas)),
            "cell_count": int(len(deltas)),
            "worst_cell_delta_vs_c0": float(min(deltas)),
            "paired_delta_vs_semantic_f1_ci95": ci(seed_diffs),
            "paired_delta_vs_semantic_accuracy_ci95": ci(seed_acc_diffs),
            "memory_rows": sorted(set(int(x["memory_rows"]) for x in records)),
            "student_seconds_final_mean": float(np.mean([
                x["student_seconds"] for x in records])),
        })

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps({
        "protocol": data["protocol"],
        "cells": cell_final,
        "overall": overall,
    }, indent=2), encoding="utf-8")
    with (args.output / "overall.csv").open("w", newline="", encoding="utf-8") as f:
        fields = list(overall[0].keys())
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(overall)

    lines = [
        "# Semantic-Recursive Unified Baseline Summary",
        "",
        "## Overall",
        "",
        "| Method | Final F1 | Final Acc. | Stage-avg F1 | Stage-avg Acc. | Old F1 | New F1 | Delta vs C0 | Positive cells | Paired delta vs Semantic (95% CI) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for x in overall:
        c = x["paired_delta_vs_semantic_f1_ci95"]
        ci_text = ("n/a" if c["mean"] is None else
                   f"{c['mean']:+.2f} [{c['low']:+.2f}, {c['high']:+.2f}]")
        lines.append(
            f"| {x['label']} | {x['final_f1']:.2f} | {x['final_accuracy']:.2f} | "
            f"{x['stage_average_f1']:.2f} | {x['stage_average_accuracy']:.2f} | "
            f"{x['old_f1']:.2f} | {x['new_f1']:.2f} | {x['delta_vs_c0_f1']:+.2f} | "
            f"{x['positive_cells_vs_c0']}/{x['cell_count']} | {ci_text} |"
        )
    lines += ["", "## Final stage by cell", "",
              "| Dataset/order | " + " | ".join(LABELS.get(m, m) + " F1/Acc"
                                               for m in methods) + " |",
              "|---|" + "---:|" * len(methods)]
    for dataset, order in cells:
        values = []
        for arm in methods:
            x = next(v for v in cell_final if v["dataset"] == dataset
                     and v["order"] == order and v["arm"] == arm)
            values.append(f"{x['final_f1']:.2f}/{x['final_accuracy']:.2f}")
        lines.append("| " + f"{dataset}/{order}" + " | " + " | ".join(values) + " |")
    (args.output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
