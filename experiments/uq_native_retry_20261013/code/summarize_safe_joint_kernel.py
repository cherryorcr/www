"""Summarize safe joint-kernel V2 without dropping stage-0 null metrics."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def value(metrics, scope, key):
    item = metrics.get(scope, {}).get(key)
    if item is None:
        return None
    return item.get("mean") if isinstance(item, dict) else item


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    cells, stages, per_seed, audits = [], [], [], []
    for cell in sorted(p for p in args.root.iterdir() if p.is_dir()
                       and (p / "results.json").exists()):
        result = json.loads((cell / "results.json").read_text())
        manifest = result["manifest"]
        by_arm = {}
        for row in result["results"]:
            by_arm.setdefault(row["arm"], []).append(row)
            stages.append({
                "dataset": manifest["dataset"], "cell": cell.name,
                "order": manifest["order"], "stage_count": manifest["stage_count"],
                "rate": manifest["rate"], "arm": row["arm"], "stage": row["stage"],
                "rows": row["rows"], "steps": row["steps"],
                "final_f1": value(row["metrics"], "all", "cumulative_f1"),
                "old_f1": value(row["metrics"], "all", "old_f1"),
                "new_f1": value(row["metrics"], "all", "new_f1"),
                "old_f1_std": (row["metrics"]["all"]["old_f1"] or {}).get("std"),
                "new_f1_std": (row["metrics"]["all"]["new_f1"] or {}).get("std"),
                **{f"topology_{k}": v for k, v in row.get("topology", {}).items()},
            })
            for seed in row["per_seed"]:
                per_seed.append({
                    "dataset": manifest["dataset"], "cell": cell.name,
                    "order": manifest["order"], "stage_count": manifest["stage_count"],
                    "rate": manifest["rate"], "arm": row["arm"], "stage": row["stage"],
                    "seed": seed["seed"], "final_f1": value(seed["metrics"], "all", "cumulative_f1"),
                    "old_f1": value(seed["metrics"], "all", "old_f1"),
                    "new_f1": value(seed["metrics"], "all", "new_f1"),
                })
        last = manifest["stage_count"] - 1
        c0 = next(r for r in by_arm["c0"] if r["stage"] == last)
        for arm, rows in by_arm.items():
            final = next(r for r in rows if r["stage"] == last)
            audit = result.get("audits", {}).get(arm, [])[last]
            cells.append({
                "dataset": manifest["dataset"], "cell": cell.name,
                "order": manifest["order"], "stage_count": manifest["stage_count"],
                "rate": manifest["rate"], "arm": arm,
                "final_f1": value(final["metrics"], "all", "cumulative_f1"),
                "final_old_f1": value(final["metrics"], "all", "old_f1"),
                "final_new_f1": value(final["metrics"], "all", "new_f1"),
                "stage_average_f1": float(np.mean([value(r["metrics"], "all", "cumulative_f1")
                                                     for r in rows])),
                "final_minus_c0_pp": value(final["metrics"], "all", "cumulative_f1")
                - value(c0["metrics"], "all", "cumulative_f1"),
                "final_old_minus_c0_pp": (value(final["metrics"], "all", "old_f1")
                    - value(c0["metrics"], "all", "old_f1")) if value(final["metrics"], "all", "old_f1") is not None else None,
                "final_new_minus_c0_pp": (value(final["metrics"], "all", "new_f1")
                    - value(c0["metrics"], "all", "new_f1")) if value(final["metrics"], "all", "new_f1") is not None else None,
                "rows_final": audit.get("payload", {}).get("rows"),
                "logical_bytes_final": audit.get("payload", {}).get("logical_bytes"),
                "serialized_file_bytes_final": audit.get("serialized_file_bytes"),
                "build_seconds": audit.get("build_seconds"),
            })
        audits.extend({"dataset": manifest["dataset"], "cell": cell.name,
                       "arm": arm, "stage": stage, **audit}
                      for arm, values in result.get("audits", {}).items()
                      for stage, audit in enumerate(values))
    summary = args.root / "summary"
    write_csv(summary / "cells.csv", cells)
    write_csv(summary / "stages.csv", stages)
    write_csv(summary / "per_seed.csv", per_seed)
    write_csv(summary / "audits.csv", audits)
    aggregate = []
    for dataset in sorted({r["dataset"] for r in cells}):
        for arm in sorted({r["arm"] for r in cells if r["dataset"] == dataset}):
            selected = [r for r in cells if r["dataset"] == dataset and r["arm"] == arm]
            aggregate.append({"dataset": dataset, "arm": arm, "cells": len(selected),
                "final_f1": float(np.mean([r["final_f1"] for r in selected])),
                "stage_average_f1": float(np.mean([r["stage_average_f1"] for r in selected])),
                "final_old_f1": float(np.nanmean([r["final_old_f1"] for r in selected])),
                "final_new_f1": float(np.nanmean([r["final_new_f1"] for r in selected])),
                "mean_minus_c0_pp": float(np.mean([r["final_minus_c0_pp"] for r in selected])),
                "positive_cells": int(sum(r["final_minus_c0_pp"] > 0 for r in selected)),
                "mean_logical_bytes": float(np.mean([r["logical_bytes_final"] for r in selected
                                                       if r["logical_bytes_final"] is not None]))})
    write_csv(summary / "aggregate.csv", aggregate)
    (summary / "summary.json").write_text(json.dumps({"aggregate": aggregate,
        "cells": len(cells), "test_labels_evaluated": False}, indent=2), encoding="utf-8")
    print(json.dumps({"aggregate": aggregate, "cells": len(cells)}, indent=2))


if __name__ == "__main__":
    main()
