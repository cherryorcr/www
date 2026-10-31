"""Summarize the audited small-graph 9-cell queue without mixing Student SD."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev


ARM_LABELS = {
    "uq_dm_edgefree": "CoTAG-Q",
    "cat_cctp": "CaT-CCTP",
    "gecc_hard": "GECC-hard-TAG",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    rows = []
    for path in sorted((args.root / "cells").glob("*/results.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "complete":
            continue
        protocol = payload["protocol"]
        # Each result file stores one row per continual stage, not one row per
        # Student seed.  The final stage is the final-memory evaluation used
        # by the main table; selecting it avoids counting intermediate stages
        # as extra construction cells.
        final_stage = max(int(row["stage"]) for row in payload["rows"])
        for row in payload["rows"]:
            if int(row["stage"]) != final_stage:
                continue
            score = row["scores"]["sealed_test"]
            rows.append({
                "dataset": protocol["dataset"],
                "order_seed": protocol["order_seed"],
                "construction_seed": protocol["construction_seed"],
                "arm": row["arm"],
                "method": ARM_LABELS.get(row["arm"], row["arm"]),
                "final_f1": float(score["final_f1"]),
                "accuracy": float(score["accuracy"]),
                "student_f1_sd": float(score["final_f1_std"]),
                "student_accuracy_sd": float(score["accuracy_std"]),
            })
    expected = {"cora", "citeseer"}
    observed = {row["dataset"] for row in rows}
    if not expected.issubset(observed):
        raise SystemExit(f"missing datasets: {sorted(expected - observed)}")
    summary = []
    for dataset in sorted(observed):
        for arm in ARM_LABELS:
            values = [row for row in rows if row["dataset"] == dataset and row["arm"] == arm]
            if not values:
                continue
            f1 = [row["final_f1"] for row in values]
            acc = [row["accuracy"] for row in values]
            summary.append({
                "dataset": dataset,
                "arm": arm,
                "method": ARM_LABELS[arm],
                "n_cells": len(values),
                "n_student_runs": len(values) * 3,
                "final_f1_mean": mean(f1),
                "final_f1_cell_sd": stdev(f1) if len(f1) > 1 else 0.0,
                "accuracy_mean": mean(acc),
                "accuracy_cell_sd": stdev(acc) if len(acc) > 1 else 0.0,
                "order_seeds": sorted({row["order_seed"] for row in values}),
                "construction_seeds": sorted({row["construction_seed"] for row in values}),
            })
    summary_path = args.root / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        fields = list(summary[0])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in summary:
            writer.writerow({**row, "order_seeds": ",".join(map(str, row["order_seeds"])),
                             "construction_seeds": ",".join(map(str, row["construction_seeds"]))})
    lines = [
        "# CoTAG-Q small-graph 9-cell summary",
        "",
        "Each cell is one class order x construction seed; three Student seeds are averaged inside each cell.",
        "Cell SD is computed across the nine order-by-construction cell means and is not the Student SD.",
        "",
        "| Dataset | Method | Cells | Student runs | Macro-F1 | Accuracy |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['dataset']} | {row['method']} | {row['n_cells']} | {row['n_student_runs']} "
            f"| {row['final_f1_mean']:.2f} +/- {row['final_f1_cell_sd']:.2f} "
            f"| {row['accuracy_mean']:.2f} +/- {row['accuracy_cell_sd']:.2f} |"
        )
    (args.root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
