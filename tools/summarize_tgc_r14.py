"""Summarize R14 using paired report-seed metrics."""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import statistics as stats
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def mean(values):
    values = [float(v) for v in values if v is not None]
    return stats.mean(values) if values else None


def ci(values):
    values = [float(v) for v in values]
    if len(values) < 2:
        return [values[0], values[0]] if values else [None, None]
    return [stats.mean(values) - 2.571 * stats.stdev(values) / math.sqrt(len(values)),
            stats.mean(values) + 2.571 * stats.stdev(values) / math.sqrt(len(values))]


def write_csv(path, rows):
    if not rows:
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(root):
    spec = read(root / "protocol.json")
    rows, cells = [], []
    for dataset, order in itertools.product(spec["datasets"], spec["orders"]):
        folder = root / dataset / order
        audit = read(folder / "audit.json")
        if audit.get("status") != "passed":
            raise ValueError(f"audit failed: {dataset}/{order}")
        result = {(int(item["stage"]), item["arm"]): item
                  for item in read(folder / "results.json")["results"]}
        for stage in spec["evaluation_stages"]:
            for arm in spec["arms"]:
                current = result[stage, arm]
                base = result[stage, "c0"]
                current_seeds = {int(x["seed"]): x for x in current["per_seed"]}
                base_seeds = {int(x["seed"]): x for x in base["per_seed"]}
                deltas = []
                old_deltas = []
                new_deltas = []
                for seed in spec["report_seeds"]:
                    now = current_seeds[seed]["metrics"]["all"]
                    ref = base_seeds[seed]["metrics"]["all"]
                    row = {
                        "dataset": dataset, "order": order, "stage": stage,
                        "arm": arm, "seed": seed,
                        "f1": now["cumulative_f1"],
                        "old_f1": now.get("old_f1"),
                        "new_f1": now.get("new_f1"),
                        "delta_f1": now["cumulative_f1"] - ref["cumulative_f1"],
                        "delta_old": (now["old_f1"] - ref["old_f1"]
                                      if now.get("old_f1") is not None and ref.get("old_f1") is not None else None),
                        "delta_new": now["new_f1"] - ref["new_f1"],
                        "seconds": current_seeds[seed].get("seconds"),
                        "steps": current_seeds[seed].get("steps"),
                        "rows": current_seeds[seed].get("rows"),
                    }
                    rows.append(row)
                    deltas.append(row["delta_f1"])
                    new_deltas.append(row["delta_new"])
                    if row["delta_old"] is not None:
                        old_deltas.append(row["delta_old"])
                cells.append({
                    "dataset": dataset, "order": order, "stage": stage, "arm": arm,
                    "f1": current["metrics"]["all"]["cumulative_f1"]["mean"],
                    "f1_sd": current["metrics"]["all"]["cumulative_f1"]["std"],
                    "old_f1": current["metrics"]["all"].get("old_f1", {}).get("mean")
                              if current["metrics"]["all"].get("old_f1") else None,
                    "new_f1": current["metrics"]["all"]["new_f1"]["mean"],
                    "delta_f1": mean(deltas), "delta_old": mean(old_deltas),
                    "delta_new": mean(new_deltas),
                    "delta_ci_low": ci(deltas)[0], "delta_ci_high": ci(deltas)[1],
                    "seconds": mean([v["seconds"] for v in rows[-len(deltas):]
                                     if v["seconds"] is not None]),
                    "rows": rows[-len(deltas)]["rows"],
                    "steps": rows[-len(deltas)]["steps"],
                })

    final_stage = max(spec["evaluation_stages"])
    methods = []
    for arm in spec["arms"]:
        final = [x for x in cells if x["arm"] == arm and x["stage"] == final_stage]
        all_cells = [x for x in cells if x["arm"] == arm]
        methods.append({
            "arm": arm,
            "final_f1": mean(x["f1"] for x in final),
            "final_delta_f1": mean(x["delta_f1"] for x in final),
            "final_delta_old": mean(x["delta_old"] for x in final),
            "final_delta_new": mean(x["delta_new"] for x in final),
            "stage_mean_delta_f1": mean(x["delta_f1"] for x in all_cells),
            "worst_final_delta_f1": min(x["delta_f1"] for x in final),
            "worst_final_delta_new": min(x["delta_new"] for x in final),
            "positive_final_cells": sum(x["delta_f1"] > 0 for x in final),
            "final_cells": len(final),
            "mean_student_seconds": mean(x["seconds"] for x in final),
        })

    contrasts = []
    for arm in spec["arms"]:
        if arm == "c0":
            continue
        for dataset, order in itertools.product(spec["datasets"], spec["orders"]):
            for stage in spec["evaluation_stages"]:
                left = {(x["seed"]): x for x in rows if x["dataset"] == dataset and x["order"] == order
                        and x["stage"] == stage and x["arm"] == arm}
                right = {(x["seed"]): x for x in rows if x["dataset"] == dataset and x["order"] == order
                         and x["stage"] == stage and x["arm"] == "c0"}
                diffs = [left[s]["f1"] - right[s]["f1"] for s in spec["report_seeds"]]
                contrasts.append({"dataset": dataset, "order": order, "stage": stage,
                                  "candidate": arm, "baseline": "c0",
                                  "mean_f1_delta": mean(diffs),
                                  "paired_ci_low": ci(diffs)[0], "paired_ci_high": ci(diffs)[1]})

    result = {"methods": methods, "contrasts": contrasts,
              "fit_slots": len(rows),
              "executed_fits": sum(1 for x in rows if x.get("seconds") is not None),
              "development_only": True,
              "interpretation": "R14 is a development experiment; no arm is promoted by this summary."}
    write_csv(root / "per_seed.csv", rows)
    write_csv(root / "cells.csv", cells)
    write_csv(root / "methods.csv", methods)
    write_csv(root / "contrasts.csv", contrasts)
    (root / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"methods": methods, "fit_slots": len(rows)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    summarize(parser.parse_args().root)
