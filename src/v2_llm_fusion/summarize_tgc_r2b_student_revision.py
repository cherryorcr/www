"""Paired R2B diagnostic report; never assemble independent stages into a rollout."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev

DATASETS = ("cora", "citeseer")
ORDERS = ("canonical", "order19101")
SEEDS = tuple(range(100, 106))
ARMS = ("c0", "teacher") + tuple(
    f"{kind}_k{k}" for k in (1, 4, 8)
    for kind in ("student", "stop", "random", "guarded"))


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


def policy_source(arm, record):
    if arm.startswith("stop_"):
        return "c0"
    if arm.startswith("guarded_"):
        return "student_" + arm.split("_", 1)[1] if record["audit"]["accepted"] else "c0"
    return arm


def values(record, scope, metric):
    rows = {int(r["seed"]): r for r in record["per_seed"]}
    if set(rows) != set(SEEDS) or len(record["per_seed"]) != len(SEEDS):
        raise ValueError("missing or duplicate report seeds")
    return [float(rows[s]["metrics"][scope][metric]) for s in SEEDS]


def paired_summary(candidate, baseline):
    deltas = [a - b for a, b in zip(candidate, baseline)]
    half = 2.570581835636305 * stdev(deltas) / len(deltas) ** .5
    return mean(deltas), mean(deltas) - half, mean(deltas) + half


def promotion(s1, s2):
    criteria = {
        "s2_mean_ge_0p5": s2["delta_all_f1_pp"] >= .5,
        "s2_cells_ge_minus1_at_least3": s2["cells_ge_minus1"] >= 3,
        "s2_new_nonnegative": s2["delta_seen_new_pp"] >= 0,
        "s1_mean_nonnegative": s1["delta_all_f1_pp"] >= 0,
        "s2_better_than_matched_random": s2["delta_vs_random_pp"] > 0,
        "s2_not_below_teacher": s2["delta_vs_teacher_pp"] >= 0,
    }
    return {"passed": all(criteria.values()), "criteria": criteria}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    args = ap.parse_args()
    audit = read(args.root / "audit.json")
    if not audit["passed"] or audit["records_checked"] != 672:
        raise ValueError("complete prediction/invariant audit required")
    metrics, mechanism, costs = [], [], []
    for dataset in DATASETS:
        for order in ORDERS:
            results = read(args.root / dataset / order / "results.json")["results"]
            records = {(r["stage"], r["arm"]): r for r in results}
            expected = {(s, a) for s in (1, 2) for a in ARMS}
            if set(records) != expected or len(results) != len(expected):
                raise ValueError("incomplete or duplicated matrix")
            for stage in (1, 2):
                for view in ("raw_executed", "paired_policy"):
                    for arm in ARMS:
                        source = policy_source(arm, records[stage, arm]) if view == "paired_policy" else arm
                        record = records[stage, source]
                        if record["memory_sha256"] != records[stage, arm]["memory_sha256"]:
                            raise ValueError("policy reused different memory")
                        base = records[stage, "c0"]
                        final = values(record, "all", "cumulative_f1")
                        delta, low, high = paired_summary(final, values(base, "all", "cumulative_f1"))
                        row = {"dataset": dataset, "order": order, "stage": stage, "view": view,
                               "arm": arm, "source_arm": source, "rows": record["rows"],
                               "mean_all_f1": mean(final), "std_all_f1": stdev(final),
                               "mean_seen_f1": mean(values(record, "seen", "cumulative_f1")),
                               "mean_seen_old_f1": mean(values(record, "seen", "old_f1")),
                               "mean_seen_new_f1": mean(values(record, "seen", "new_f1")),
                               "delta_all_f1_pp": delta, "ci95_low_pp": low, "ci95_high_pp": high,
                               "delta_seen_old_pp": mean(values(record, "seen", "old_f1")) - mean(values(base, "seen", "old_f1")),
                               "delta_seen_new_pp": mean(values(record, "seen", "new_f1")) - mean(values(base, "seen", "new_f1")),
                               "delta_vs_teacher_pp": mean(final) - mean(values(records[stage, "teacher"], "all", "cumulative_f1")),
                               "delta_vs_random_pp": None}
                        if "_k" in arm:
                            random = records[stage, "random_" + arm.split("_", 1)[1]]
                            row["delta_vs_random_pp"] = mean(final) - mean(values(random, "all", "cumulative_f1"))
                        metrics.append(row)
                for arm in ARMS:
                    record = records[stage, arm]
                    opt = record["audit"].get("optimization")
                    costs.append({"dataset": dataset, "order": order, "stage": stage, "arm": arm,
                                  "executed_fits": sum(int(r["executed_here"]) for r in record["per_seed"]),
                                  "actual_fit_seconds": sum(r["seconds"] for r in record["per_seed"] if r["executed_here"]),
                                  "historical_reused_fit_seconds": sum(r["seconds"] for r in record["per_seed"] if not r["executed_here"]),
                                  "optimization_seconds": opt["seconds"] if opt else None,
                                  "temporary_student_steps": sum(opt[k] for k in ("outer_train_student_steps", "validation_student_steps", "baseline_student_steps")) if opt else None})
                for k in (1, 4, 8):
                    record = records[stage, f"student_k{k}"]
                    opt, inv = record["audit"]["optimization"], record["audit"]["invariants"]
                    before, after = opt["baseline_probe"]["validation"], opt["selected_validation"]
                    train_old_changes = [p["old"] - b["old"] for step in opt["curve"]
                                         for p, b in zip(step["train_probe"], opt["baseline_probe"]["train"])]
                    mechanism.append({"dataset": dataset, "order": order, "stage": stage, "horizon": k,
                                      "accepted": opt["accepted"], "selected_outer_step": opt["selected_outer_step"],
                                      "changed_rows": len(inv["changed_rows"]),
                                      "max_relative_displacement": inv["max_relative_displacement"],
                                      "validation_new_loss_gain": mean(b["new"] - a["new"] for a, b in zip(after, before)),
                                      "validation_old_loss_gain": mean(b["old"] - a["old"] for a, b in zip(after, before)),
                                      "new_gain_both_seeds": all(a["new"] <= b["new"] - 1e-4 for a, b in zip(after, before)),
                                      "old_tolerance_both_seeds": all(a["old"] <= b["old"] + .01 for a, b in zip(after, before)),
                                      "train_old_probe_evaluations": len(train_old_changes),
                                      "train_old_hinge_active": sum(v > .01 for v in train_old_changes),
                                      "train_old_loss_increase_max": max(train_old_changes),
                                      "optimization_seconds": opt["seconds"]})
    aggregate = []
    for view in ("raw_executed", "paired_policy"):
        for stage in (1, 2):
            for arm in ARMS:
                rows = [r for r in metrics if r["view"] == view and r["stage"] == stage and r["arm"] == arm]
                row = {"view": view, "stage": stage, "arm": arm, "cells": len(rows),
                       "positive_cells": sum(r["delta_all_f1_pp"] > 0 for r in rows),
                       "cells_ge_minus1": sum(r["delta_all_f1_pp"] >= -1 for r in rows),
                       "worst_delta_pp": min(r["delta_all_f1_pp"] for r in rows)}
                for key in ("mean_all_f1", "delta_all_f1_pp", "delta_seen_old_pp", "delta_seen_new_pp",
                            "delta_vs_teacher_pp", "delta_vs_random_pp"):
                    vals = [r[key] for r in rows if r[key] is not None]
                    row[key] = mean(vals) if vals else None
                aggregate.append(row)
    gates = {}
    for view in ("raw_executed", "paired_policy"):
        gates[view] = {}
        for k in (1, 4, 8):
            stages = [next(r for r in aggregate if r["view"] == view and r["stage"] == s and r["arm"] == f"guarded_k{k}")
                      for s in (1, 2)]
            gates[view][str(k)] = promotion(*stages)
    output = {
        "scope": "8 shared-C0-parent single-stage diagnostics, not a recursive continual rollout",
        "audit_passed": audit["passed"], "records": 672,
        "executed_fits": sum(r["executed_fits"] for r in costs),
        "reused_records": 672 - sum(r["executed_fits"] for r in costs),
        "actual_canonical_training_seconds": sum(r["actual_fit_seconds"] for r in costs),
        "student_optimization_seconds": sum(r["optimization_seconds"] for r in costs if r["arm"].startswith("student_")),
        "stop_optimization_seconds": sum(r["optimization_seconds"] for r in costs if r["arm"].startswith("stop_")),
        "teacher_original_elapsed_seconds": None,
        "cost_scope": "actual fit and student/stop optimization timers; teacher/build/cold-start/audit not included",
        "cells": metrics, "aggregate": aggregate, "mechanism": mechanism,
        "costs": costs, "promotion": gates,
        "policy_reporting": "guard uses pre-report frozen decision and the exact selected student/C0 fit; raw redundant refits retained separately",
        "limitations": ["all conditions previously exposed development cells", "construction seed not replicated",
                        "only two data sets and two orders", "per-cell paired t intervals over 6 seeds, no multiple-test correction",
                        "selection validation shares legal current graph context, not unseen raw-data generalization",
                        "1/4/8-step objective versus much longer row-dependent canonical training",
                        "no historical raw graph given to revision operator; full graph still used by downstream evaluator"],
    }
    save(args.root / "summary.json", output)
    for name, rows in (("cell_metrics", metrics), ("aggregate", aggregate), ("mechanism", mechanism), ("costs", costs)):
        with (args.root / f"{name}.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({"executed_fits": output["executed_fits"], "promotion": gates,
                      "paired_s2": [r for r in aggregate if r["view"] == "paired_policy" and r["stage"] == 2]}))


if __name__ == "__main__":
    main()
