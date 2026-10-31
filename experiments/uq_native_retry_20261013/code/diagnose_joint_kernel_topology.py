"""No-training topology audit for the completed Phase-12 joint-kernel matrix."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

import run_safe_joint_kernel_experiments as S


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def correlation(rows: list[dict], x: str, y: str) -> float | None:
    values = [(float(row[x]), float(row[y])) for row in rows
              if row.get(x) is not None and row.get(y) is not None]
    if len(values) < 3:
        return None
    a, b = np.asarray(values, dtype=float).T
    if np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True,
                        help="phase12_frozen_20260920/matrix")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    stage_rows, final_rows = [], []
    nested = any((p / "results.json").exists() for p in args.root.iterdir() if p.is_dir())
    dataset_dirs = [args.root] if nested else sorted(p for p in args.root.iterdir() if p.is_dir())
    for dataset_dir in dataset_dirs:
        for cell in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
            if not (cell / "results.json").exists():
                continue
            manifest = json.loads((cell / "manifest.json").read_text())
            result = json.loads((cell / "results.json").read_text())
            per_arm = {}
            for row in result["results"]:
                per_arm.setdefault(row["arm"], {})[int(row["stage"])] = row
            c0 = []
            joint = []
            stage_count = len(manifest["stage_classes"])
            order = cell.name.split("_", 1)[1].rsplit("_r", 1)[0]
            for stage in range(stage_count):
                c0.append(torch.load(
                    cell / f"c0_s{stage}_memory.pt",
                    map_location="cpu", weights_only=False))
                joint.append(torch.load(
                    cell / f"joint_kernel_s{stage}_memory.pt",
                    map_location="cpu", weights_only=False))
            for stage in range(stage_count):
                for arm, memories in (("c0", c0), ("joint_kernel", joint)):
                    previous = None if stage == 0 else memories[stage - 1]
                    metrics = S.topology_metrics(previous, memories[stage])
                    row = {
                        "dataset": dataset_dir.name,
                        "cell": cell.name,
                        "order": order,
                        "stage_count": stage_count,
                        "rate": manifest["rate"],
                        "arm": arm,
                        "stage": stage,
                        "rows": len(memories[stage]["x"]),
                        **metrics,
                    }
                    stage_rows.append(row)
            c0_final = per_arm["c0"][stage_count - 1]["metrics"]["all"]["cumulative_f1"]["mean"]
            joint_final = per_arm["joint_kernel"][stage_count - 1]["metrics"]["all"]["cumulative_f1"]["mean"]
            final_topology = next(
                row for row in stage_rows
                if row["dataset"] == dataset_dir.name and row["cell"] == cell.name
                and row["arm"] == "joint_kernel"
                and row["stage"] == stage_count - 1)
            final_rows.append({
                "dataset": dataset_dir.name,
                "cell": cell.name,
                "order": order,
                "stage_count": stage_count,
                "rate": manifest["rate"],
                "joint_minus_c0_pp": joint_final - c0_final,
                **{f"final_{key}": value for key, value in final_topology.items()
                   if key not in {"dataset", "cell", "order", "stage_count", "rate",
                                  "arm", "stage", "rows"}},
            })

    summary = {
        "stage_rows": len(stage_rows),
        "final_cells": len(final_rows),
        "correlations": {
            metric: correlation(final_rows, metric, "joint_minus_c0_pp")
            for metric in ("final_old_normalized_drift",
                           "final_old_propagation_drift",
                           "final_cross_old_endpoint_gini",
                           "final_new_internal_degree_gini",
                           "final_max_old_cross_degree",
                           "final_max_new_degree")
        },
        "scope": "diagnostic correlation only; no causal claim or report-based selection",
        "test_labels_evaluated": False,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "stage_topology.csv", stage_rows)
    write_csv(args.output / "final_topology_with_f1.csv", final_rows)
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
