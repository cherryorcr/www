"""Report the frozen warm-prefix matrix without selecting arms on report labels."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean, stdev

from summarize_tgc_r2b_student_revision import (
    DATASETS, ORDERS, SEEDS, paired_summary, promotion, read, save, values,
)

WARMUPS, HORIZONS = (0, 64, 256), (1, 4)
ARMS = ("c0", "teacher") + tuple(
    f"{kind}_w{w}_k{k}" for w in WARMUPS for k in HORIZONS
    for kind in ("student", "random", "stop", "guarded"))
SLOTS = len(DATASETS) * len(ORDERS) * 2 * len(ARMS) * len(SEEDS)


def matrix(results):
    records = {(r["stage"], r["arm"]): r for r in results}
    expected = {(s, a) for s in (1, 2) for a in ARMS}
    if len(results) != len(expected) or set(records) != expected:
        raise ValueError("incomplete or duplicated warm-prefix matrix")
    return records


def gate_diagnosis(accepted, delta, low, high):
    return {
        "accepted_negative_mean": bool(accepted and delta < 0),
        "rejected_positive_mean": bool(not accepted and delta > 0),
        "accepted_ci_below_zero": bool(accepted and high < 0),
        "rejected_ci_above_zero": bool(not accepted and low > 0),
    }


def temporary_steps(opt):
    return opt.get("suffix_student_steps", sum(opt.get(k, 0) for k in (
        "outer_train_student_steps", "validation_student_steps", "baseline_student_steps")))


def summarize(root):
    audit = read(root / "audit.json")
    if not audit["passed"] or audit["records_checked"] != SLOTS:
        raise ValueError("complete prediction and memory audit required")
    cells, mechanism, costs, prefixes = [], [], [], []
    preflight_seconds = 0.
    for dataset in DATASETS:
        for order in ORDERS:
            folder = root / dataset / order
            records = matrix(read(folder / "results.json")["results"])
            for stage in (1, 2):
                decision = read(folder / f"s{stage}_decisions.json")
                preflight_seconds += decision["preflight_seconds"]
                for warmup, prefix in decision["prefix_audits"].items():
                    prefixes.append({"dataset": dataset, "order": order, "stage": stage,
                                     "warmup": int(warmup), "seconds": prefix["seconds"],
                                     "student_steps": prefix["student_steps"]})
                for arm in ARMS:
                    record, base = records[stage, arm], records[stage, "c0"]
                    final = values(record, "all", "cumulative_f1")
                    delta, low, high = paired_summary(final, values(base, "all", "cumulative_f1"))
                    row = {"dataset": dataset, "order": order, "stage": stage, "arm": arm,
                           "rows": record["rows"], "mean_all_f1": mean(final),
                           "std_all_f1": stdev(final), "delta_all_f1_pp": delta,
                           "ci95_low_pp": low, "ci95_high_pp": high,
                           "mean_seen_f1": mean(values(record, "seen", "cumulative_f1")),
                           "mean_seen_old_f1": mean(values(record, "seen", "old_f1")),
                           "mean_seen_new_f1": mean(values(record, "seen", "new_f1")),
                           "delta_seen_old_pp": mean(values(record, "seen", "old_f1")) - mean(values(base, "seen", "old_f1")),
                           "delta_seen_new_pp": mean(values(record, "seen", "new_f1")) - mean(values(base, "seen", "new_f1")),
                           "delta_vs_teacher_pp": mean(final) - mean(values(records[stage, "teacher"], "all", "cumulative_f1")),
                           "delta_vs_random_pp": None, "random_ci95_low_pp": None, "random_ci95_high_pp": None,
                           "delta_vs_w0_pp": None, "w0_ci95_low_pp": None, "w0_ci95_high_pp": None}
                    if "_w" in arm:
                        kind, suffix = arm.split("_", 1)
                        horizon = suffix.split("_k")[1]
                        comparisons = (("random", f"random_{suffix}"), ("w0", f"{kind}_w0_k{horizon}"))
                        for label, other in comparisons:
                            diff, lo, hi = paired_summary(final, values(records[stage, other], "all", "cumulative_f1"))
                            row.update({f"delta_vs_{label}_pp": diff,
                                        f"{label}_ci95_low_pp": lo, f"{label}_ci95_high_pp": hi})
                    cells.append(row)
                    opt = record["audit"].get("optimization")
                    warmup = record["audit"].get("warmup", 0)
                    prefix = decision["prefix_audits"].get(str(warmup), {})
                    costs.append({"dataset": dataset, "order": order, "stage": stage, "arm": arm,
                                  "executed_fits": sum(int(r["executed_here"]) for r in record["per_seed"]),
                                  "actual_fit_seconds": sum(r["seconds"] for r in record["per_seed"] if r["executed_here"]),
                                  "actual_fit_wall_seconds": sum(r.get("wall_fit_seconds", 0.) for r in record["per_seed"] if r["executed_here"]),
                                  "optimization_seconds": opt["seconds"] if opt else 0.,
                                  "optimization_executed_here": bool(opt and warmup > 0),
                                  "temporary_suffix_steps": temporary_steps(opt) if opt else 0,
                                  "standalone_revision_seconds": prefix.get("seconds", 0.) + opt["seconds"] if opt else None})
                    if not arm.startswith("student_"):
                        continue
                    before, after = opt["baseline_probe"]["validation"], opt["selected_validation"]
                    changes = [p["old"] - b["old"] for step in opt["curve"]
                               for p, b in zip(step["train_probe"], opt["baseline_probe"]["train"])]
                    inv = next(a["invariants"] for a in decision["arms"] if a["arm"] == arm)
                    mechanism.append({"dataset": dataset, "order": order, "stage": stage,
                                      "warmup": warmup, "horizon": record["audit"]["horizon"],
                                      "accepted": opt["accepted"], "selected_outer_step": opt["selected_outer_step"],
                                      "changed_rows": len(inv["changed_rows"]),
                                      "max_relative_displacement": inv["max_relative_displacement"],
                                      "validation_new_loss_gain": mean(b["new"] - a["new"] for a, b in zip(after, before)),
                                      "validation_old_loss_gain": mean(b["old"] - a["old"] for a, b in zip(after, before)),
                                      "new_gain_each_seed": [b["new"] - a["new"] for a, b in zip(after, before)],
                                      "train_old_probe_evaluations": len(changes),
                                      "train_old_hinge_active": sum(v > .01 for v in changes),
                                      "train_old_loss_increase_max": max(changes),
                                      "report_delta_pp": delta, "ci95_low_pp": low, "ci95_high_pp": high,
                                      **gate_diagnosis(opt["accepted"], delta, low, high)})
    aggregate = []
    for stage in (1, 2):
        for arm in ARMS:
            rows = [r for r in cells if r["stage"] == stage and r["arm"] == arm]
            row = {"stage": stage, "arm": arm, "cells": len(rows),
                   "positive_cells": sum(r["delta_all_f1_pp"] > 0 for r in rows),
                   "cells_ge_minus1": sum(r["delta_all_f1_pp"] >= -1 for r in rows),
                   "worst_delta_pp": min(r["delta_all_f1_pp"] for r in rows)}
            for key in ("mean_all_f1", "delta_all_f1_pp", "delta_seen_old_pp", "delta_seen_new_pp",
                        "delta_vs_teacher_pp", "delta_vs_random_pp", "delta_vs_w0_pp"):
                vals = [r[key] for r in rows if r[key] is not None]
                row[key] = mean(vals) if vals else None
            aggregate.append(row)
    gates = {}
    for w in WARMUPS:
        for k in HORIZONS:
            name = f"guarded_w{w}_k{k}"
            stages = [next(r for r in aggregate if r["stage"] == s and r["arm"] == name) for s in (1, 2)]
            gates[name] = promotion(*stages)
    executed = sum(r["executed_fits"] for r in costs)
    return {"scope": "8 shared-C0-parent diagnostics; not a recursive continual rollout",
            "audit_passed": True, "records": SLOTS, "executed_fits": executed,
            "reused_records": SLOTS - executed, "cells": cells, "aggregate": aggregate,
            "mechanism": mechanism, "costs": costs, "prefixes": prefixes, "promotion": gates,
            "actual_canonical_training_seconds": sum(r["actual_fit_seconds"] for r in costs),
            "actual_fit_wall_seconds": sum(r["actual_fit_wall_seconds"] for r in costs),
            "actual_shared_prefix_seconds": sum(r["seconds"] for r in prefixes),
            "actual_student_optimization_seconds": sum(r["optimization_seconds"] for r in costs if r["optimization_executed_here"] and r["arm"].startswith("student_")),
            "actual_stop_optimization_seconds": sum(r["optimization_seconds"] for r in costs if r["optimization_executed_here"] and r["arm"].startswith("stop_")),
            "preflight_seconds": preflight_seconds,
            "cost_scope": "New execution only in totals; standalone revision includes one full prefix bundle and optimization, excludes C0/teacher/build/cold-start/IO/audit. w0 optimization timers are historical.",
            "limitations": ["Previously exposed development cells; no independent confirmation",
                            "No replicated construction seed; paired t intervals use six report seeds only",
                            "No multiple-comparison correction; mean-sign gate errors are descriptive, not ground truth",
                            "Prefix is fixed C0 and detached; suffix is a truncated training surrogate",
                            "No historical raw input to revision; downstream scoring still uses the real graph",
                            "No cell-wise best-arm selection and no claim of recursive continual improvement"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    args = ap.parse_args()
    output = summarize(args.root)
    save(args.root / "summary.json", output)
    for name in ("cells", "aggregate", "mechanism", "costs", "prefixes"):
        rows = output[name]
        with (args.root / f"{name}.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print({k: output[k] for k in ("records", "executed_fits", "reused_records", "promotion")})


if __name__ == "__main__":
    main()
