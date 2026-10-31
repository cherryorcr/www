"""Aggregate offline diagnostics without presenting them as continual-policy runs."""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    args = ap.parse_args()
    audit = read(args.root / "audit.json")
    if not audit["passed"]:
        raise ValueError("response audit failed")
    results = [r for d in ("cora", "citeseer") for r in read(args.root / d / "summary.json")["results"]]
    correlations, policies = [], []
    for stage in (1, 2):
        stages = [r for r in results if r["stage"] == stage]
        for horizon in (32, 64, 128, "curve"):
            for score in ("balanced_loss", "old_loss", "new_loss", "train_loss"):
                values = [next(c for c in r["correlations"] if c["horizon"] == horizon)["spearman"][score] for r in stages]
                correlations.append({"stage": stage, "horizon": horizon, "score": score,
                                     "mean_spearman": mean(values), "worst_spearman": min(values),
                                     "positive_cells": sum(x > 0 for x in values),
                                     "cells_ge_0p3": sum(x >= .3 for x in values)})
            for k in (2, 4):
                for policy in ("balanced", "guarded", "own_training_loss", "random", "shuffled", "teacher"):
                    rows = [next(c for c in r["outcomes"] if c["horizon"] == horizon and c["k"] == k and c["policy"] == policy) for r in stages]
                    policies.append({"stage": stage, "horizon": horizon, "k": k, "policy": policy,
                                     "mean_delta_vs_c0_pp": mean(r["delta_vs_c0_pp"] for r in rows),
                                     "worst_delta_vs_c0_pp": min(r["delta_vs_c0_pp"] for r in rows),
                                     "mean_regret_pp": mean(r["regret_pp"] for r in rows),
                                     "fallback_cells": sum(r["fallback_to_c0"] for r in rows)})
    result = {"audit": audit, "scope": "offline ranking only; not a sequential selector benchmark",
              "correlations": correlations, "policies": policies,
              "selection_seconds": sum(r["selection_seconds_all_candidates"] for r in results),
              "prefix_verification_seconds": sum(p["verification_seconds"] for r in results for p in r["prefix_audits"]),
              "selection_trajectories": sum(r["trajectories"] for r in results),
              "new_canonical_fits": sum(r["new_canonical_fits"] for r in results),
              "limitations": ["all cells are exposed development conditions", "stage-2 candidate histories differ",
                              "report scores read only after choices were saved", "correlations and per-stage oracle choices are not deployment gains"]}
    (args.root / "aggregate.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    for key in ("correlations", "policies"):
        with (args.root / f"{key}.csv").open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(result[key][0]))
            writer.writeheader()
            writer.writerows(result[key])
    print(json.dumps({"stage2_balanced_correlations": [r for r in correlations if r["stage"] == 2 and r["score"] == "balanced_loss"],
                      "stage2_k4_balanced_guarded": [r for r in policies if r["stage"] == 2 and r["k"] == 4 and r["policy"] in ("balanced", "guarded")],
                      "selection_seconds": result["selection_seconds"]}))


if __name__ == "__main__":
    main()
