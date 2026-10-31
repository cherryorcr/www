"""Reproducible capacity-sweep tables; never select a per-cell winner."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev

from scipy.stats import spearmanr


CONFIGS = ("radius002", "radius005", "radius020", "protect060", "quota025",
           "protect040", "quota050")
CELLS = [(d, o) for d in ("cora", "citeseer") for o in ("canonical", "order19101")]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def entry(result, arm, stage=2):
    return next(r for r in result["results"] if r["arm"] == arm and r["stage"] == stage)


def value(row, key="cumulative_f1", scope="all"):
    return float(row["metrics"][scope][key]["mean"])


def pair_statistics(left, right):
    a = {r["seed"]: r["metrics"]["all"]["cumulative_f1"] for r in left["per_seed"]}
    b = {r["seed"]: r["metrics"]["all"]["cumulative_f1"] for r in right["per_seed"]}
    if set(a) != set(b) or len(a) != 6:
        raise ValueError("paired six-seed comparison required")
    differences = [a[s] - b[s] for s in sorted(a)]
    sd = stdev(differences)
    return {"paired_delta_pp": mean(differences), "paired_sd_pp": sd,
            "paired_ci95_low_pp": mean(differences) - 2.5705818366 * sd / 6**0.5,
            "paired_ci95_high_pp": mean(differences) + 2.5705818366 * sd / 6**0.5,
            "paired_seed_wins": sum(v > 0 for v in differences)}


def summarize(root: Path, reference: Path) -> dict:
    if not read(reference / "audit.json")["passed"]:
        raise ValueError("Phase A reference audit failed")
    rows = []
    executed = reused = 0
    for config in ("baseline",) + CONFIGS:
        for dataset, order in CELLS:
            if config == "baseline":
                cell = reference / dataset / order
            else:
                run = root / f"{dataset}_{config}"
                audit = read(run / "audit.json")
                if not audit["passed"] or len(audit["cells"]) != 2:
                    raise ValueError(f"capacity audit incomplete: {run}")
                cell = run / dataset / order
                completed = read(cell / "completion.json")
                executed += completed["executed_fits"]
                reused += completed["reused_fits"]
            result = read(cell / "results.json")
            base, revised = entry(result, "c0"), entry(result, "tgc_r")
            if result["manifest"]["student_seeds"] != [100, 101, 102, 103, 104, 105]:
                raise ValueError("report seed protocol mismatch")
            constructions = {a: [read(cell / f"{a}_s{s}_construction.json")["build"]
                                   for s in range(3)] for a in ("c0", "tgc_r")}
            build0 = sum(x["build_seconds"] for x in constructions["c0"])
            build1 = sum(x["build_seconds"] for x in constructions["tgc_r"])
            revisions = [b["revision"] for b in constructions["tgc_r"][1:]]
            record = {"config": config, "dataset": dataset, "order": order,
                      "c0_final_f1": value(base), "final_f1": value(revised),
                      "delta_pp": value(revised) - value(base),
                      "old_delta_pp": value(revised, "old_f1", "seen") - value(base, "old_f1", "seen"),
                      "new_delta_pp": value(revised, "new_f1", "seen") - value(base, "new_f1", "seen"),
                      "stage_avg_delta_pp": mean(value(entry(result, "tgc_r", s)) - value(entry(result, "c0", s)) for s in range(3)),
                      "c0_build_seconds": build0, "build_seconds": build1,
                      "build_ratio": build1 / build0,
                      "quota": sum(r["revision_quota"] for r in revisions),
                      "updated": sum(len(r["updated_rows"]) for r in revisions),
                      "teacher_relative_gain_mean": mean((r["baseline_loss"] - r["final_loss"]) / max(abs(r["baseline_loss"]), 1e-12) for r in revisions),
                      "teacher_relative_gain_final": (revisions[-1]["baseline_loss"] - revisions[-1]["final_loss"]) / max(abs(revisions[-1]["baseline_loss"]), 1e-12),
                      "source": str(cell / "results.json")}
            record.update(pair_statistics(revised, base))
            rows.append(record)
    aggregates = []
    for config in ("baseline",) + CONFIGS:
        group = [r for r in rows if r["config"] == config]
        average = mean(r["delta_pp"] for r in group)
        stable = sum(r["delta_pp"] >= -1 for r in group)
        new = mean(r["new_delta_pp"] for r in group)
        cost = max(r["build_ratio"] for r in group)
        gate = {"mean_final_ge_0p5": average >= .5, "at_least_3_of_4_ge_minus1": stable >= 3,
                "mean_new_nonnegative": new >= 0, "max_build_le_3x": cost <= 3}
        aggregates.append({"config": config, "mean_delta_pp": average,
                           "worst_delta_pp": min(r["delta_pp"] for r in group),
                           "mean_old_delta_pp": mean(r["old_delta_pp"] for r in group),
                           "mean_new_delta_pp": new, "mean_stage_avg_delta_pp": mean(r["stage_avg_delta_pp"] for r in group),
                           "positive_cells": sum(r["delta_pp"] > 0 for r in group),
                           "mean_build_ratio": mean(r["build_ratio"] for r in group),
                           "max_build_ratio": cost, "accepted_rows": sum(r["updated"] for r in group),
                           "attempted_rows": sum(r["quota"] for r in group),
                           "development_gate": gate, "development_pass": all(gate.values())})
    correlations = []
    for dataset, order in CELLS:
        group = [r for r in rows if r["dataset"] == dataset and r["order"] == order]
        corr = spearmanr([r["teacher_relative_gain_final"] for r in group],
                         [r["delta_pp"] for r in group]).statistic
        correlations.append({"dataset": dataset, "order": order,
                             "teacher_gain_vs_final_delta_spearman": float(corr)})
    return {"scope": "four previously exposed development cells, not independent generalization",
            "new_executed_fits": executed, "reused_fit_slots": reused,
            "reference_phase_a_fits": 648, "rows": rows, "aggregate": aggregates,
            "teacher_correlations": correlations,
            "development_winners": [r["config"] for r in aggregates if r["development_pass"]],
            "best_uniform_config": max(aggregates, key=lambda r: r["mean_delta_pp"])["config"],
            "uncertainty": "CI is paired across six student seeds within a cell only; orders and cells are dependent; no multiple-testing correction."}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    result = summarize(args.root, args.reference)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    for key in ("rows", "aggregate"):
        with (args.output / f"{key}.csv").open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(result[key][0]))
            writer.writeheader()
            writer.writerows(result[key])
    print(json.dumps({k: result[k] for k in ("new_executed_fits", "reused_fit_slots", "aggregate", "development_winners")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
