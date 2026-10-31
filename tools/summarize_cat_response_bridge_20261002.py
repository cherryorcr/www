"""Summarize the CaT Student-response bridge paired experiment.

The first response-bridge matrix compared ``support_x_only`` with ``cat_x``.
The follow-up MMD-controlled matrix uses ``mmd_support_x_only`` instead, so
the arm names are command-line arguments rather than hard-coded.  The report
also includes paired per-seed deltas and per-order stage averages; these are
useful guards against over-interpreting a single final-stage cell.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _paired_seed_rows(base: dict, candidate: dict) -> list[dict]:
    base_scores = {int(item["seed"]): item for item in base["per_student"]}
    candidate_scores = {
        int(item["seed"]): item for item in candidate["per_student"]
    }
    if set(base_scores) != set(candidate_scores):
        raise ValueError("baseline and candidate student seeds differ")
    return [{
        "seed": seed,
        "baseline_f1": base_scores[seed]["cumulative_f1"],
        "candidate_f1": candidate_scores[seed]["cumulative_f1"],
        "delta_f1": (candidate_scores[seed]["cumulative_f1"]
                      - base_scores[seed]["cumulative_f1"]),
        "baseline_accuracy": base_scores[seed]["accuracy"],
        "candidate_accuracy": candidate_scores[seed]["accuracy"],
        "delta_accuracy": (candidate_scores[seed]["accuracy"]
                           - base_scores[seed]["accuracy"]),
    } for seed in sorted(base_scores)]


def summarize(root: Path, baseline_arm: str, candidate_arm: str) -> dict:
    cells: list[dict] = []
    datasets = [p.name for p in root.iterdir() if p.is_dir()]
    for dataset in sorted(datasets):
        payload = load(root / dataset / "results.json")
        by_key = {
            (row["order"], int(row["stage"]), row["arm"]): row
            for row in payload["rows"]
        }
        keys = sorted({(row["order"], int(row["stage"]))
                       for row in payload["rows"]})
        for order, stage in keys:
            cat = by_key[(order, stage, "cat_x")]
            candidate = by_key[(order, stage, candidate_arm)]
            paired = _paired_seed_rows(cat, candidate)
            cells.append({
                "dataset": dataset,
                "order": order,
                "stage": stage,
                "cat_f1": cat["aggregate"]["mean_f1"],
                "candidate_f1": candidate["aggregate"]["mean_f1"],
                "delta_f1": candidate["aggregate"]["mean_f1"]
                - cat["aggregate"]["mean_f1"],
                "cat_accuracy": cat["aggregate"]["mean_accuracy"],
                "candidate_accuracy": candidate["aggregate"]["mean_accuracy"],
                "delta_accuracy": candidate["aggregate"]["mean_accuracy"]
                - cat["aggregate"]["mean_accuracy"],
                "cat_construction_seconds": cat["construction_seconds"],
                "candidate_construction_seconds": candidate["construction_seconds"],
                "paired_seeds": paired,
            })

    summary = []
    for dataset in sorted(datasets):
        values = [row for row in cells if row["dataset"] == dataset]
        order_means = []
        for order in sorted({row["order"] for row in values}):
            order_rows = [row for row in values if row["order"] == order]
            order_means.append({
                "order": order,
                "cat_f1": mean(row["cat_f1"] for row in order_rows),
                "candidate_f1": mean(row["candidate_f1"] for row in order_rows),
                "delta_f1": mean(row["delta_f1"] for row in order_rows),
                "cat_accuracy": mean(row["cat_accuracy"] for row in order_rows),
                "candidate_accuracy": mean(row["candidate_accuracy"]
                                            for row in order_rows),
                "delta_accuracy": mean(row["delta_accuracy"]
                                        for row in order_rows),
                "cat_stage_average_f1": mean(row["cat_f1"]
                                             for row in order_rows),
                "candidate_stage_average_f1": mean(row["candidate_f1"]
                                                   for row in order_rows),
            })
        seed_rows = [seed_row
                     for cell in values
                     for seed_row in cell["paired_seeds"]]
        order_deltas = [row["delta_f1"] for row in order_means]
        summary.append({
            "dataset": dataset,
            "cells": len(values),
            "orders": len(order_means),
            "mean_cell_delta_f1": mean(row["delta_f1"] for row in values),
            "std_cell_delta_f1": pstdev(row["delta_f1"] for row in values),
            "mean_order_delta_f1": mean(row["delta_f1"] for row in order_means),
            "positive_f1_cells": sum(row["delta_f1"] > 0 for row in values),
            "positive_f1_ratio": mean(row["delta_f1"] > 0 for row in values),
            "worst_cell_delta_f1": min(row["delta_f1"] for row in values),
            "best_cell_delta_f1": max(row["delta_f1"] for row in values),
            "median_cell_delta_f1": sorted(
                row["delta_f1"] for row in values)[len(values) // 2],
            "mean_cell_delta_accuracy": mean(
                row["delta_accuracy"] for row in values),
            "mean_cat_construction_seconds": mean(
                row["cat_construction_seconds"] for row in values),
            "mean_candidate_construction_seconds": mean(
                row["candidate_construction_seconds"] for row in values),
            "construction_cost_ratio": (
                mean(row["candidate_construction_seconds"] for row in values)
                / mean(row["cat_construction_seconds"] for row in values)),
            "stage_average_delta_f1": mean(order_deltas),
            "positive_order_ratio": mean(delta > 0 for delta in order_deltas),
            "worst_order_stage_average_delta_f1": min(order_deltas),
            "paired_seed_mean_delta_f1": mean(
                row["delta_f1"] for row in seed_rows),
            "paired_seed_std_delta_f1": pstdev(
                row["delta_f1"] for row in seed_rows),
            "paired_seed_positive_ratio": mean(
                row["delta_f1"] > 0 for row in seed_rows),
            "order_means": order_means,
        })
    return {"baseline_arm": baseline_arm, "candidate_arm": candidate_arm,
            "cells": cells, "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-arm", default="cat_x")
    parser.add_argument("--candidate-arm", default="support_x_only")
    args = parser.parse_args()
    result = summarize(args.root, args.baseline_arm, args.candidate_arm)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=True),
                           encoding="utf-8")
    for row in result["summary"]:
        print(json.dumps({key: value for key, value in row.items()
                          if key != "order_means"}, ensure_ascii=True))


if __name__ == "__main__":
    main()

