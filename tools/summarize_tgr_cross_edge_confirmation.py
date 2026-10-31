"""Summarize paired transport results without confusing stage and group means."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics as stats


METRICS = ("cumulative_macro_f1", "cumulative_accuracy", "current_f1", "old_mean_f1")


def paired(reference, candidate, key):
    left = {int(row["seed"]): row for row in reference}
    right = {int(row["seed"]): row for row in candidate}
    if set(left) != set(right) or len(left) != len(reference) or len(right) != len(candidate):
        raise ValueError("missing, duplicate, or mismatched paired seeds")
    differences = [right[seed][key] - left[seed][key] for seed in sorted(left)]
    if len(differences) != 6:
        raise ValueError("this frozen protocol requires six paired Student seeds")
    mean = stats.mean(differences)
    sd = stats.stdev(differences)
    # Two-sided Student t quantile at 0.975, df=5; conditional on fixed memory/order.
    radius = 2.570581835636305 * sd / math.sqrt(6)
    return {"delta_pp": mean, "paired_seed_sd": sd, "ci95": [mean - radius, mean + radius],
            "positive_seeds": sum(v > 0 for v in differences), "n_seeds": 6,
            "per_seed_differences": differences,
            "ci_scope": "Student seeds conditional on this constructed memory and order"}


def metric_summary(rows, key):
    values = [r[key] for r in rows]
    return {"mean": stats.mean(values), "seed_sd": stats.stdev(values)}


def summarize_cell(obj):
    arms = obj["arms"]
    baseline = "semantic_recursive" if "semantic_recursive" in arms else "c0"
    ref, cand = arms[baseline]["rows"], arms["message_balanced"]["rows"]
    metrics = [key for key in METRICS if key in ref[0]]
    return {"dataset": obj.get("dataset", obj.get("protocol", {}).get("dataset")),
            "order": obj.get("order", obj.get("protocol", {}).get("order")),
            "stage": obj.get("stage", obj.get("protocol", {}).get("stage")),
            "semantic_recursive": {key: metric_summary(ref, key) for key in metrics},
            "message_balanced": {key: metric_summary(cand, key) for key in metrics},
            "paired": {key: paired(ref, cand, key) for key in metrics}}


def summarize_trajectory(stages):
    if sorted(int(s["stage"]) for s in stages) != [0, 1, 2]:
        raise ValueError("trajectory mean requires stage0, stage1, and stage2")
    result = {}
    stage_means = {}
    for arm in ("semantic_recursive", "message_balanced"):
        stage_means[arm] = []
        for seed in range(150, 156):
            values = []
            for stage in stages:
                rows = stage["arms"][arm]["rows"]
                matches = [r for r in rows if r["seed"] == seed]
                if len(matches) != 1:
                    raise ValueError("missing or duplicate trajectory seed")
                values.append(matches[0])
            stage_means[arm].append({"seed": seed,
                "cumulative_macro_f1": stats.mean(r["cumulative_macro_f1"] for r in values),
                "cumulative_accuracy": stats.mean(r["cumulative_accuracy"] for r in values)})
        result[arm] = {key: metric_summary(stage_means[arm], key)
                      for key in METRICS[:2]}
        result[arm]["worst_stage_mean_f1"] = min(
            stats.mean(r["cumulative_macro_f1"] for r in s["arms"][arm]["rows"])
            for s in stages)
        result[arm]["training_only_seconds"] = sum(
            s["arms"][arm]["training_only_seconds"] for s in stages)
    result["paired"] = {key: paired(stage_means["semantic_recursive"],
                                     stage_means["message_balanced"], key) for key in METRICS[:2]}
    return result


def cohort(cells):
    def aggregate(rows):
        deltas = [r["paired"]["cumulative_macro_f1"]["delta_pp"] for r in rows]
        return {"n_stage_cells": len(rows), "positive_stage_cells": sum(d > 0 for d in deltas),
                "worst_stage_cell_delta_pp": min(deltas),
                "mean_delta_pp": {key: stats.mean(r["paired"][key]["delta_pp"] for r in rows)
                                  for key in METRICS}}
    return {"incremental_stage_cells": aggregate([c for c in cells if c["stage"] > 0]),
            "final_stage_only": aggregate([c for c in cells if c["stage"] == 2])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirmation-root", type=Path, required=True)
    parser.add_argument("--development-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    development = [summarize_cell(json.loads(p.read_text(encoding="utf-8")))
                   for p in sorted(args.development_root.glob("*_s[12].json"))]
    if len(development) != 8:
        raise ValueError("expected eight development stage cells")
    confirmation, trajectories, audits = [], {}, {}
    for dataset in ("cora", "citeseer"):
        root = args.confirmation_root / dataset
        audit = json.loads((root / "reload_audit.json").read_text())
        if audit["status"] != "passed" or audit["unique_checkpoints"] != 30:
            raise ValueError("incomplete checkpoint audit")
        stages = [json.loads((root / f"eval_s{s}.json").read_text()) for s in range(3)]
        confirmation.extend(summarize_cell(s) for s in stages)
        trajectories[dataset] = summarize_trajectory(stages)
        audits[dataset] = {"reload": audit, "construction": [
            json.loads((root / f"s{s}_construction_audit.json").read_text()) for s in (1, 2)]}
    final = [c for c in confirmation if c["stage"] == 2]
    incremental = [c for c in confirmation if c["stage"] > 0]
    summary = {"comparison": "SR + message_balanced versus Semantic-Recursive",
               "development": {"cells": development, "summary": cohort(development)},
               "confirmation": {"cells": confirmation, "summary": cohort(confirmation),
                                "trajectories": trajectories, "audits": audits},
               "combined_descriptive_only": cohort(development + incremental),
               "progression": {
                   "both_dataset_final_deltas_positive": all(
                       c["paired"]["cumulative_macro_f1"]["delta_pp"] > 0 for c in final),
                   "worst_incremental_cell_ge_minus2": min(
                       c["paired"]["cumulative_macro_f1"]["delta_pp"] for c in incremental) >= -2.0}}
    summary["progression"]["passed"] = all(summary["progression"].values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"development": summary["development"]["summary"],
                      "confirmation": summary["confirmation"]["summary"],
                      "progression": summary["progression"]}, indent=2))


if __name__ == "__main__":
    main()
