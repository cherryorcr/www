"""Summarize audited TGC-R Phase A results."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean

ARMS = ("c0", "naive_refresh", "c0m", "tgc_r", "tgc_no_preserve", "tgc_shuffled")


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def row(results: dict, arm: str, stage: int) -> dict:
    return next(x for x in results["results"]
                if x["arm"] == arm and int(x["stage"]) == stage)


def metric(item: dict, scope: str, key: str) -> float | None:
    value = item["metrics"][scope][key]
    return None if value is None else float(value["mean"])


def summarize_cell(cell: Path) -> dict:
    results = read(cell / "results.json")
    stages = len(results["manifest"]["stage_classes"])
    final_stage = stages - 1
    records = {}
    arms = results["manifest"]["arms"]
    for arm in arms:
        final = row(results, arm, final_stage)
        trajectory = [row(results, arm, stage) for stage in range(stages)]
        records[arm] = {
            "final_f1": metric(final, "all", "cumulative_f1"),
            "final_old_f1": metric(final, "seen", "old_f1"),
            "final_new_f1": metric(final, "seen", "new_f1"),
            "stage_avg_f1": mean(metric(item, "all", "cumulative_f1")
                                   for item in trajectory),
        }
    c0 = records["c0"]
    deltas = {}
    for arm in arms[1:]:
        deltas[arm] = {
            "final_f1_pp": records[arm]["final_f1"] - c0["final_f1"],
            "final_old_f1_pp": records[arm]["final_old_f1"] - c0["final_old_f1"],
            "final_new_f1_pp": records[arm]["final_new_f1"] - c0["final_new_f1"],
            "stage_avg_f1_pp": records[arm]["stage_avg_f1"] - c0["stage_avg_f1"],
        }
    revisions = []
    tgc_build = c0_build = 0.0
    for stage in range(stages):
        for arm in arms:
            build = read(cell / f"{arm}_s{stage}_construction.json")["build"]
            value = float(build.get("build_seconds", 0.0))
            if arm == "c0":
                c0_build += value
            if arm == "tgc_r":
                tgc_build += value
            if stage > 0 and arm == "tgc_r":
                revision = build.get("revision", {})
                revisions.append(revision)
    return {
        "dataset": results["manifest"]["dataset"],
        "order": results["manifest"]["order"],
        "arms": records,
        "deltas_vs_c0": deltas,
        "tgc_r_updated_rows": sum(len(item.get("updated_rows", []))
                                   for item in revisions),
        "tgc_r_revision_quota": sum(int(item.get("revision_quota", 0))
                                     for item in revisions),
        "tgc_r_build_cost_ratio": tgc_build / max(c0_build, 1e-12),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--audit", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    audit = read(Path(args.audit))
    cells = [Path(item["cell"]) for item in audit["cells"] if item["passed"]]
    failed = [item for item in audit["cells"] if not item["passed"]]
    if not audit.get("passed") or not cells or failed:
        raise ValueError(f"audit failed for {len(failed)} cells")
    rows = [summarize_cell(cell) for cell in cells]
    aggregate = {}
    arms = list(rows[0]["arms"])
    if any(list(r["arms"]) != arms for r in rows):
        raise ValueError("cells use different arms")
    for arm in arms[1:]:
        values = [r["deltas_vs_c0"][arm]["final_f1_pp"] for r in rows]
        aggregate[arm] = {
            "mean_final_delta_pp": mean(values) if values else None,
            "min_final_delta_pp": min(values) if values else None,
            "cells_at_least_minus_1pp": sum(v >= -1.0 for v in values),
            "cells_positive": sum(v > 0.0 for v in values),
            "mean_old_delta_pp": mean(r["deltas_vs_c0"][arm]["final_old_f1_pp"]
                                       for r in rows) if rows else None,
            "mean_new_delta_pp": mean(r["deltas_vs_c0"][arm]["final_new_f1_pp"]
                                       for r in rows) if rows else None,
        }
    tgc = aggregate.get("tgc_r", {})
    promotion = {
        "mean_final_at_least_1pp": bool(tgc and tgc["mean_final_delta_pp"] >= 1.0),
        "worst_final_at_least_minus_2pp": bool(tgc and tgc["min_final_delta_pp"] >= -2.0),
        "stable_cells_at_least_minus_1pp": bool(
            tgc and tgc["cells_at_least_minus_1pp"] >= math.ceil(2 * len(rows) / 3)),
        "mean_old_not_below_minus_2pp": bool(tgc and tgc["mean_old_delta_pp"] >= -2.0),
        "mean_new_positive": bool(tgc and tgc["mean_new_delta_pp"] > 0.0),
        "build_cost_all_below_2x": bool(tgc and all(r["tgc_r_build_cost_ratio"] <= 2.0
                                             for r in rows)),
    }
    output = {"root": args.root, "cells": rows, "aggregate": aggregate,
              "promotion_gate": promotion, "promoted": all(promotion.values())}
    Path(args.output).write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(json.dumps({"promoted": output["promoted"], "aggregate": aggregate},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
