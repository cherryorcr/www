"""Combine audited V2 datasets; do not treat development cells as confirmation."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from statistics import mean, stdev

from summarize_safe_joint_kernel import value, write_csv

MAIN_ARMS = ("c0", "k_center", "joint_kernel", "degree_balanced", "coverage",
             "operator_stable", "drift_gated")
METRICS = {"f1": "cumulative_f1", "old": "old_f1", "new": "new_f1"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paired(a, b, key):
    left = {r["seed"]: value(r["metrics"], "all", key) for r in a}
    right = {r["seed"]: value(r["metrics"], "all", key) for r in b}
    if left.keys() != right.keys() or len(left) != len(a) or len(right) != len(b):
        raise ValueError("unequal or duplicate paired seeds")
    return {seed: left[seed] - right[seed] for seed in sorted(left)
            if left[seed] is not None and right[seed] is not None}


def collect(root, require_reevaluation=True, allow_missing_serialized=False):
    frozen = json.loads((root / "freeze.json").read_text())
    completion = json.loads((root / "completion.json").read_text())
    audit = json.loads((root / "audit_v2.json").read_text())
    if (not completion["complete"] or completion["fits"] != frozen["expected_fits"]
            or completion["freeze_sha256"] != sha(root / "freeze.json")
            or audit["error_count"] or audit["checked_fits"] != frozen["expected_fits"]
            or require_reevaluation and not audit["independent_model_reevaluation"]):
        raise ValueError(f"incomplete or unaudited root: {root}")
    cells, stages, seeds, provenance = [], [], [], []
    fits = 0
    for spec in frozen["cells"]:
        path = root / spec["name"] / "results.json"
        result = json.loads(path.read_text())
        manifest = result["manifest"]
        by_arm = defaultdict(list)
        for r in result["results"]:
            by_arm[r["arm"]].append(r)
        base = {"dataset": frozen["dataset"], "cell": spec["name"],
                "order": manifest["order"], "rate": manifest["rate"],
                "stage_count": manifest["stage_count"]}
        for arm, rows in by_arm.items():
            rows.sort(key=lambda r: r["stage"])
            if [r["stage"] for r in rows] != list(range(manifest["stage_count"])):
                raise ValueError("incomplete stages")
            audits = result["audits"][arm]
            for row, construction in zip(rows, audits):
                if [s["seed"] for s in row["per_seed"]] != frozen["seeds"]:
                    raise ValueError("unexpected report seeds")
                fits += len(row["per_seed"])
                stage = dict(base, arm=arm, stage=row["stage"], rows=row["rows"],
                             steps=row["steps"], logical_bytes=construction["payload"]["logical_bytes"],
                             build_seconds=construction["build_seconds"],
                             train_mean_seconds=mean(s["seconds"] for s in row["per_seed"]),
                             fit_wall_mean_seconds=mean(s["fit_wall_seconds_including_report_io"] for s in row["per_seed"]),
                             gate_fallback=construction.get("variant_audit", {}).get("gate_fallback"))
                stage.update({name: value(row["metrics"], "all", key) for name, key in METRICS.items()})
                stage.update(row["topology"])
                stages.append(stage)
                for s in row["per_seed"]:
                    seeds.append(dict(base, arm=arm, stage=row["stage"], seed=s["seed"],
                        seconds=s["seconds"], **{name: value(s["metrics"], "all", key) for name, key in METRICS.items()}))
            final = rows[-1]
            control = sorted(by_arm["c0"], key=lambda r: r["stage"])[-1]
            serialized_path = root / spec["name"] / f"{arm}_s{final['stage']}_memory.pt"
            if serialized_path.exists():
                serialized_bytes = serialized_path.stat().st_size
            elif allow_missing_serialized:
                serialized_bytes = None
            else:
                raise FileNotFoundError(serialized_path)
            record = dict(base, arm=arm, final_rows=final["rows"],
                stage_average_f1=mean(value(r["metrics"], "all", "cumulative_f1") for r in rows),
                logical_bytes=audits[-1]["payload"]["logical_bytes"],
                serialized_bytes=serialized_bytes,
                chain_build_seconds=sum(a["build_seconds"] for a in audits),
                chain_train_mean_seconds=sum(mean(s["seconds"] for s in r["per_seed"]) for r in rows),
                chain_fit_wall_mean_seconds=sum(mean(s["fit_wall_seconds_including_report_io"] for s in r["per_seed"]) for r in rows))
            record["chain_total_seconds"] = record["chain_build_seconds"] + record["chain_train_mean_seconds"]
            for name, key in METRICS.items():
                delta = paired(final["per_seed"], control["per_seed"], key)
                record[name] = value(final["metrics"], "all", key)
                record[f"{name}_std_population"] = final["metrics"]["all"][key]["std"]
                record[f"{name}_delta"] = mean(delta.values())
                record[f"{name}_paired_sd_sample"] = stdev(delta.values()) if len(delta) > 1 else None
                record[f"{name}_positive_seeds"] = sum(d > 0 for d in delta.values())
            cells.append(record)
        provenance.append({**base, "results_sha256": sha(path)})
    if fits != frozen["expected_fits"]:
        raise ValueError(f"fit count mismatch: {fits}")
    return cells, stages, seeds, {"dataset": frozen["dataset"], "fits": fits,
        "freeze_sha256": sha(root / "freeze.json"), "audit_sha256": sha(root / "audit_v2.json"),
        "source_hashes": frozen["source_hashes"], "inputs": provenance}


def relative_cost(cells):
    controls = {(r["dataset"], r["cell"]): r for r in cells if r["arm"] == "c0"}
    for r in cells:
        c0 = controls[r["dataset"], r["cell"]]
        for key in ("logical_bytes", "serialized_bytes", "chain_build_seconds",
                    "chain_train_mean_seconds", "chain_total_seconds"):
            if r[key] is None or c0[key] is None:
                r[f"{key}_ratio_c0"] = None
            else:
                r[f"{key}_ratio_c0"] = r[key] / c0[key]


def aggregate(cells, fields):
    groups = defaultdict(list)
    for row in cells:
        groups[tuple(row[k] for k in fields)].append(row)
    output = []
    for key, rows in sorted(groups.items()):
        record = dict(zip(fields, key), cells=len(rows), positive_cells=sum(r["f1_delta"] > 0 for r in rows),
                      worst_cell_delta=min(r["f1_delta"] for r in rows),
                      worst_old_cell_delta=min(r["old_delta"] for r in rows))
        for field in ("f1", "old", "new", "f1_delta", "old_delta", "new_delta", "stage_average_f1",
                      "logical_bytes_ratio_c0", "chain_build_seconds", "chain_train_mean_seconds",
                      "chain_total_seconds", "chain_train_mean_seconds_ratio_c0", "chain_total_seconds_ratio_c0"):
            record[field] = mean(r[field] for r in rows)
        output.append(record)
    return output


def gates(cells):
    main = [r for r in cells if r["arm"] in MAIN_ARMS]
    groups = aggregate(main, ("arm", "dataset", "order"))
    result = []
    for arm in MAIN_ARMS[2:]:
        selected = [r for r in main if r["arm"] == arm]
        combinations = [r for r in groups if r["arm"] == arm]
        if len(selected) != 16 or len(combinations) != 4:
            raise ValueError("promotion gates require the full 16-cell development matrix")
        checks = {
            "three_of_four_combinations_nonnegative": sum(r["f1_delta"] >= 0 for r in combinations) >= 3,
            "two_combinations_at_least_1pp": sum(r["f1_delta"] >= 1 for r in combinations) >= 2,
            "mean_gain_at_least_1pp": mean(r["f1_delta"] for r in selected) >= 1,
            "worst_combination_at_least_minus2pp": min(r["f1_delta"] for r in combinations) >= -2,
            "worst_old_cell_at_least_minus2pp": min(r["old_delta"] for r in selected) >= -2,
            "all_cell_training_cost_at_most_2x": max(r["chain_train_mean_seconds_ratio_c0"] for r in selected) <= 2,
        }
        result.append({"arm": arm, "checks": checks,
            "failed_checks": [k for k, v in checks.items() if not v],
            "quantitative_gates_pass": all(checks.values()),
            "positive_new_combinations": sum(r["new_delta"] > 0 for r in combinations),
            "mean_new_gain_pp": mean(r["new_delta"] for r in selected),
            "worst_old_combination_pp": min(r["old_delta"] for r in combinations),
            "note": "Old-F1 guard shown at both cell and combination levels; new-class repeatability still needs scientific review if all quantitative gates pass."})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-missing-serialized",
        action="store_true",
        help="recompute from audited JSON without local final .pt files; serialized-byte fields remain null",
    )
    args = parser.parse_args()
    cells, stages, seeds, provenance = [], [], [], []
    for root in args.roots:
        c, s, p, source = collect(root, allow_missing_serialized=args.allow_missing_serialized)
        cells.extend(c)
        stages.extend(s)
        seeds.extend(p)
        provenance.append(source)
    if {p["dataset"] for p in provenance} != {"cora", "citeseer"} or len(provenance) != 2:
        raise ValueError("expected two distinct formal datasets")
    relative_cost(cells)
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    main_cells = [r for r in cells if r["arm"] in MAIN_ARMS]
    write_csv(out / "cells.csv", cells)
    write_csv(out / "stages.csv", stages)
    write_csv(out / "per_seed.csv", seeds)
    for name, fields in (("by_dataset", ("arm", "dataset")), ("by_order", ("arm", "dataset", "order")),
                         ("by_rate", ("arm", "rate")), ("by_stage_count", ("arm", "stage_count")),
                         ("main_aggregate", ("arm",))):
        write_csv(out / f"{name}.csv", aggregate(main_cells, fields))
    canonical = [r for r in cells if r["order"] == "canonical" and r["rate"] == .05 and r["stage_count"] == 3]
    write_csv(out / "canonical_components_and_storage.csv", canonical)
    gate = gates(cells)
    summary = {"fits": sum(p["fits"] for p in provenance), "arm_cells": len(cells),
        "main_cells": len(main_cells) // len(MAIN_ARMS), "stage_records": len(stages),
        "main": aggregate(main_cells, ("arm",)), "gates": gate,
        "quantitative_candidates": [r["arm"] for r in gate if r["quantitative_gates_pass"]],
        "cost_scope": "sum across stages of construction plus seed-mean training; pool text preencoding, common load, and reporting IO excluded; IO-inclusive fit time is separate",
        "uncertainty": "std fields are within-cell report-seed variability; cells overlap and are development conditions, not independent replications",
        "test_labels_evaluated": False, "provenance": provenance}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "provenance"}, indent=2))


if __name__ == "__main__":
    main()
