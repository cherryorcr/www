"""Paired-seed R3 reporting; never select a policy using report metrics."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean, stdev

import run_tgc_selector_confirmation as B

M, R = B.M, B.R
SLOTS = len(B.DATASETS) * len(B.ORDERS) * 3 * len(B.ARMS) * len(B.REPORT_SEEDS)


def matrix(payload):
    records = {(r["stage"], r["arm"]): r for r in payload["results"]}
    expected = {(s, a) for s in (0, 1, 2) for a in B.ARMS}
    if len(payload["results"]) != len(expected) or set(records) != expected:
        raise ValueError("incomplete or duplicated cell matrix")
    for r in records.values():
        seeds = [x["seed"] for x in r["per_seed"]]
        if seeds != list(B.REPORT_SEEDS):
            raise ValueError("incomplete, duplicated, or unpaired report seeds")
    return records


def values(record, scope="all", metric="cumulative_f1"):
    return [r["metrics"][scope][metric] for r in record["per_seed"]]


def paired(left, right):
    if len(left) != 6 or len(right) != 6:
        raise ValueError("six paired report seeds required")
    differences = [a - b for a, b in zip(left, right)]
    avg = mean(differences)
    half = 2.5705818356 * stdev(differences) / len(differences) ** .5
    return avg, avg - half, avg + half


def summarize(root):
    audit = B.read(root / "audit.json")
    if not audit["passed"] or audit["records_checked"] != SLOTS:
        raise ValueError("complete independent artifact/prediction audit required")
    B.verify_seal(root)
    cells, selectors, costs = [], [], []
    executed = 0
    for dataset in B.DATASETS:
        for order in B.ORDERS:
            out = root / dataset / order
            records = matrix(B.read(out / "results.json"))
            for stage in (0, 1, 2):
                decision = B.read(out / f"s{stage}_decision.json")
                base = values(records[stage, "c0"])
                candidate_gain, candidate_low, candidate_high = paired(values(records[stage, "candidate"]), base)
                for arm in B.ARMS:
                    rec = records[stage, arm]
                    scores = values(rec)
                    delta, low, high = paired(scores, base)
                    row = {"dataset": dataset, "order": order, "stage": stage, "arm": arm,
                           "rows": rec["rows"], "mean_f1": mean(scores), "std_f1": stdev(scores),
                           "delta_pp": delta, "ci95_low_pp": low, "ci95_high_pp": high,
                           "delta_vs_teacher_pp": mean(scores) - mean(values(records[stage, "teacher"])),
                           "delta_vs_random_pp": mean(scores) - mean(values(records[stage, "random"]))}
                    for kind in ("old", "new"):
                        v = values(rec, "seen", kind + "_f1")
                        b = values(records[stage, "c0"], "seen", kind + "_f1")
                        row[f"mean_{kind}_f1"] = mean(v) if v[0] is not None else None
                        row[f"delta_{kind}_pp"] = mean(v) - mean(b) if v[0] is not None else None
                    cells.append(row)
                    actual = [r for r in rec["per_seed"] if r["executed_here"]]
                    executed += len(actual)
                    costs.append({"dataset": dataset, "order": order, "stage": stage, "arm": arm,
                                  "executed_fits": len(actual), "actual_fit_seconds": sum(r["seconds"] for r in actual),
                                  "mean_student_fit_seconds": mean(r["seconds"] for r in rec["per_seed"]),
                                  "c0_build_seconds": decision["construction"]["seconds"],
                                  "teacher_seconds": decision.get("teacher_seconds", 0.),
                                  "prefix_seconds": decision.get("prefix_audit", {}).get("seconds", 0.),
                                  "optimization_seconds": decision.get("optimization", {}).get("seconds", 0.),
                                  "selector_seconds": decision.get("curve", {}).get("seconds", 0.)})
                    if stage and arm in M.POLICIES:
                        p = decision["policies"][arm]
                        selectors.append({"dataset": dataset, "order": order, "stage": stage, "policy": arm,
                                          "accepted": p["accepted"], "selected_arm": p["selected_arm"],
                                          "candidate_gain_pp": candidate_gain, "selected_gain_pp": delta,
                                          "regret_pp": max(0., candidate_gain) - delta,
                                          "accepted_negative_mean": p["accepted"] and candidate_gain < 0,
                                          "rejected_positive_mean": not p["accepted"] and candidate_gain > 0,
                                          "accepted_ci_below_zero": p["accepted"] and candidate_high < 0,
                                          "rejected_ci_above_zero": not p["accepted"] and candidate_low > 0})
    aggregate = []
    for stage in (0, 1, 2):
        for arm in B.ARMS:
            subset = [r for r in cells if r["stage"] == stage and r["arm"] == arm]
            row = {"stage": stage, "arm": arm, "cells": len(subset),
                   "positive_cells": sum(r["delta_pp"] > 0 for r in subset),
                   "cells_ge_minus1": sum(r["delta_pp"] >= -1 for r in subset),
                   "worst_delta_pp": min(r["delta_pp"] for r in subset),
                   "worst_old_pp": min((r["delta_old_pp"] for r in subset if r["delta_old_pp"] is not None), default=None)}
            for key in ("mean_f1", "delta_pp", "delta_old_pp", "delta_new_pp", "delta_vs_teacher_pp", "delta_vs_random_pp"):
                valid = [r[key] for r in subset if r[key] is not None]
                row[key] = mean(valid) if valid else None
            aggregate.append(row)
    primary = {r["stage"]: r for r in aggregate if r["arm"] == "selected_curve"}
    last = primary[2]
    checks = {"final_mean_ge_0_5": last["delta_pp"] >= .5,
              "at_least_three_cells_ge_minus1": last["cells_ge_minus1"] >= 3,
              "worst_cell_ge_minus2": last["worst_delta_pp"] >= -2,
              "s1_mean_nonnegative": primary[1]["delta_pp"] >= 0,
              "new_mean_nonnegative": last["delta_new_pp"] >= 0,
              "worst_old_ge_minus2": last["worst_old_pp"] >= -2,
              "better_than_random": last["delta_vs_random_pp"] > 0,
              "not_below_teacher": last["delta_vs_teacher_pp"] >= 0}
    return {"scope": "4 new order/dataset cells; paired new seeds; shared C0 parents, not recursive",
            "audit_passed": True, "records": SLOTS, "executed_fits": executed,
            "reused_records": SLOTS - executed, "cells": cells, "aggregate": aggregate,
            "selectors": selectors, "costs": costs,
            "primary_promotion": {"passed": all(checks.values()), "checks": checks},
            "limitations": ["Same datasets/report split; new orders and random seeds are not new datasets",
                            "Deterministic C0 construction, no replicated clustering seed",
                            "Six-seed t intervals are within-cell uncertainty, not dataset-level inference",
                            "False accept/reject by mean sign and hindsight regret are descriptive",
                            "No selection using report/test labels and no cell-wise best-policy table",
                            "Cost fields are per-cell components repeated for convenience; do not sum across arms",
                            "Primary policy fixed before report; diagnostic selectors cannot replace it post hoc"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    root = ap.parse_args().root
    result = summarize(root)
    B.save(root / "summary.json", result)
    for name in ("cells", "aggregate", "selectors", "costs"):
        with (root / f"{name}.csv").open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(result[name][0]))
            writer.writeheader()
            writer.writerows(result[name])
    print({k: result[k] for k in ("records", "executed_fits", "reused_records", "primary_promotion")})


if __name__ == "__main__":
    main()
