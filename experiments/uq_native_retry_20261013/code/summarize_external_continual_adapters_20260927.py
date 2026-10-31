"""Aggregate audited external adapters, preserving dataset/order grouping."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats

NAMES = {"semantic_recursive": "Semantic-Recursive",
         "cat_mmd_tag": "CaT-MMD-TAG adaptation",
         "gecc_hard_tag": "GECC-hard-TAG adaptation",
         "k_center_native": "k-center native append",
         "medoid_native": "medoid native append"}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def ci(values):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    if len(values) < 2:
        return {"mean": mean, "low": None, "high": None, "n": len(values)}
    half = float(stats.t.ppf(.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))
    return {"mean": mean, "low": mean - half, "high": mean + half, "n": len(values)}


def fmt(value):
    return "N/A" if value is None else f"{value:.2f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--roots", nargs="+", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--previous", type=Path)
    args = p.parse_args()
    rows, manifests, audits = [], [], []
    expected_keys = set()
    for root in args.roots:
        result = read(root / "results.json")
        manifest = result["protocol"]
        if manifest["student_steps_override"] is not None or manifest["config"]["cat_steps"] != 3000:
            raise ValueError("not the registered formal configuration")
        manifests.append(manifest)
        for d in manifest["datasets"]:
            for o in manifest["orders"]:
                audit = read(root / d / o / "audit.json")
                if not audit["passed"]:
                    raise ValueError("audit failed")
                expected_fits = len(manifest["arms"]) * 3 * len(manifest["report_seeds"])
                if audit["fit_count"] != expected_fits:
                    raise ValueError("audit fit coverage mismatch")
                audits.append({"dataset": d, "order": o, **audit})
                for arm in manifest["arms"]:
                    for s in range(3):
                        expected_keys.add((d, o, arm, s))
        rows.extend(result["rows"])
    keys = [(r["dataset"], r["order"], r["arm"], r["stage"]) for r in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected_keys:
        raise ValueError("missing or duplicate method/stage records")
    for m in manifests[1:]:
        for key in ("orders", "arms", "report_seeds", "config", "source_hashes"):
            if m[key] != manifests[0][key]:
                raise ValueError(f"inconsistent common protocol: {key}")
    seeds = manifests[0]["report_seeds"]
    if (seeds != list(range(150, 156))
            or set(manifests[0]["arms"]) != set(NAMES)
            or manifests[0]["config"]["gecc_restarts"] != 20):
        raise ValueError("not the complete preregistered formal matrix")
    for r in rows:
        if [s["seed"] for s in r["per_student"]] != seeds:
            raise ValueError("unpaired or out-of-order seeds")
        for field, aggregate in (("cumulative_f1", "mean_f1"), ("accuracy", "mean_accuracy")):
            values = [s[field] for s in r["per_student"]]
            if not np.isfinite(values).all() or abs(np.mean(values) - r["aggregate"][aggregate]) > 1e-9:
                raise ValueError("nonfinite or inconsistent saved stage aggregate")
    cells = sorted(set((r["dataset"], r["order"]) for r in rows))
    if set(cells) != {(d, o) for d in ("cora", "citeseer") for o in ("order19611", "order19612")}:
        raise ValueError("formal dataset/order coverage mismatch")
    arms = manifests[0]["arms"]
    table, pairs, overall = [], [], []
    by_key = dict(zip(keys, rows))
    for d, o in cells:
        for stage in range(3):
            budgets = {by_key[d, o, a, stage]["memory_rows"] for a in arms}
            steps = {tuple(s["steps"] for s in by_key[d, o, a, stage]["per_student"]) for a in arms}
            if len(budgets) != 1 or len(steps) != 1:
                raise ValueError("budget or Student-step mismatch")
        for a in arms:
            trajectory = [by_key[d, o, a, stage] for stage in range(3)]
            final = trajectory[-1]
            scores = final["per_student"]
            total_build = [r["construction_seconds"] for r in trajectory]
            table.append({"dataset": d, "order": o, "arm": a,
                          "final_f1": float(np.mean([s["cumulative_f1"] for s in scores])),
                          "final_f1_std": float(np.std([s["cumulative_f1"] for s in scores], ddof=1)),
                          "final_accuracy": float(np.mean([s["accuracy"] for s in scores])),
                          "final_accuracy_std": float(np.std([s["accuracy"] for s in scores], ddof=1)),
                          "stage_avg_f1": float(np.mean([r["aggregate"]["mean_f1"] for r in trajectory])),
                          "stage_avg_accuracy": float(np.mean([r["aggregate"]["mean_accuracy"] for r in trajectory])),
                          "incremental_avg_f1": float(np.mean([r["aggregate"]["mean_f1"] for r in trajectory[1:]])),
                          "old_f1": float(np.mean([s["old_f1"] for s in scores])),
                          "new_f1": float(np.mean([s["new_f1"] for s in scores])),
                          "worst_stage_f1": float(min(r["aggregate"]["mean_f1"] for r in trajectory)),
                          "rows": final["memory_rows"],
                          "construction_seconds_total": None if any(v is None for v in total_build) else sum(total_build),
                          "student_seconds_all_stages_per_seed": sum(r["student_wall_seconds"] for r in trajectory) / len(seeds),
                          **final["resources"]})
            if a != "semantic_recursive":
                reference = by_key[d, o, "semantic_recursive", 2]["per_student"]
                delta = [s["cumulative_f1"] - b["cumulative_f1"] for s, b in zip(reference, scores)]
                acc_delta = [s["accuracy"] - b["accuracy"] for s, b in zip(reference, scores)]
                pairs.append({"dataset": d, "order": o, "baseline": a,
                              "difference": "SR minus external baseline",
                              "conditional_seed_ci": ci(delta),
                              "accuracy_conditional_seed_ci": ci(acc_delta),
                              "seed_deltas": dict(zip(seeds, delta)),
                              "accuracy_seed_deltas": dict(zip(seeds, acc_delta))})
    numeric = ["final_f1", "final_accuracy", "stage_avg_f1", "stage_avg_accuracy", "old_f1", "new_f1",
               "incremental_avg_f1", "student_seconds_all_stages_per_seed", "tensor_bytes", "text_utf8_bytes", "text_tokens"]
    for arm in arms:
        items = [r for r in table if r["arm"] == arm]
        overall.append({"arm": arm, **{k: float(np.mean([r[k] for r in items])) for k in numeric},
                        "construction_seconds_total": None if arm == "semantic_recursive" else float(np.mean([r["construction_seconds_total"] for r in items])),
                        "worst_stage_f1": min(r["worst_stage_f1"] for r in items)})
    comparisons = []
    for arm in arms:
        if arm == "semantic_recursive":
            continue
        values = [r["conditional_seed_ci"]["mean"] for r in pairs if r["baseline"] == arm]
        acc_values = [r["accuracy_conditional_seed_ci"]["mean"] for r in pairs if r["baseline"] == arm]
        comparisons.append({"baseline": arm, "difference": "SR minus baseline",
                            "group_descriptive_ci": ci(values), "positive_cells": sum(v > 0 for v in values),
                            "accuracy_group_descriptive_ci": ci(acc_values),
                            "accuracy_positive_cells": sum(v > 0 for v in acc_values),
                            "worst_cell_delta": min(values), "total_cells": len(values)})
    replay = []
    references = []
    if args.previous:
        previous = read(args.previous)
        for r in previous["rows"]:
            if r["arm"] == "semantic_recursive":
                current = by_key[r["dataset"], r["order"], "semantic_recursive", r["stage"]]
                if [s["seed"] for s in r["per_student"]] != seeds:
                    raise ValueError("previous SR seed mismatch")
                replay.append(max(abs(a[k] - b[k]) for a, b in zip(r["per_student"], current["per_student"])
                                  for k in ("cumulative_f1", "accuracy")))
            if r["arm"] in ("raw_full_scratch", "student_delta_finetune", "student_c0_replay_delta") and r["stage"] == 2:
                references.append(r)
    result = {"protocols": manifests, "audits": audits, "cells": table, "overall": overall,
              "paired_cell_results": pairs, "paired_group_results": comparisons,
              "fit_count": sum(a["fit_count"] for a in audits),
              "previous_sr_max_metric_error_pp": max(replay) if replay else None,
              "previous_reference_results": references,
              "std_definition": "sample standard deviation across six Student seeds within one fixed dataset/order (ddof=1)",
              "limits": "2 datasets x 2 existing orders; one construction seed. Group CI descriptive: four cells share two datasets, not four independent datasets or 24 independent orders. Native external adaptations, not official end-to-end reproductions."}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (args.output / "cells.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    lines = ["# External Continual TAG Adapter Results", "",
             f"Audited fits: {result['fit_count']}. All scores are percentages; differences are percentage points. Std uses ddof=1.", "",
             "## Final Cell Scores", "",
             "|Dataset/order|Method|F1 mean +/- std|Accuracy mean +/- std|Old F1|New F1|", "|---|---|---:|---:|---:|---:|"]
    for r in table:
        lines.append(f"|{r['dataset']}/{r['order']}|{NAMES[r['arm']]}|{r['final_f1']:.2f} +/- {r['final_f1_std']:.2f}|{r['final_accuracy']:.2f} +/- {r['final_accuracy_std']:.2f}|{r['old_f1']:.2f}|{r['new_f1']:.2f}|")
    lines += ["", "## Equal Cell Means", "", "|Method|Final F1|Final Acc|Stage-avg F1|Stage-avg Acc|Old F1|New F1|Worst stage|", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in overall:
        lines.append("|" + NAMES[r["arm"]] + "|" + "|".join(fmt(r[k]) for k in ["final_f1", "final_accuracy", "stage_avg_f1", "stage_avg_accuracy", "old_f1", "new_f1", "worst_stage_f1"]) + "|")
    lines += ["", "## Paired Comparisons", "", "Group intervals use four dataset/order means, not 24 independent seeds.", "", "|Baseline|SR minus baseline|Descriptive 95% group CI|Positive cells|Worst delta|", "|---|---:|---|---:|---:|"]
    for r in comparisons:
        c = r["group_descriptive_ci"]
        lines.append(f"|{NAMES[r['baseline']]}|{c['mean']:+.2f}|[{c['low']:+.2f}, {c['high']:+.2f}]|{r['positive_cells']}/{r['total_cells']}|{r['worst_cell_delta']:+.2f}|")
    lines += ["", "## Paired Cell Intervals", "", "These intervals quantify Student-seed variation conditional on each fixed memory/order, not order generalization.", "", "|Dataset/order|Baseline|SR F1 delta [95% CI]|SR Accuracy delta [95% CI]|", "|---|---|---:|---:|"]
    for r in pairs:
        f, a = r["conditional_seed_ci"], r["accuracy_conditional_seed_ci"]
        lines.append(f"|{r['dataset']}/{r['order']}|{NAMES[r['baseline']]}|{f['mean']:+.2f} [{f['low']:+.2f}, {f['high']:+.2f}]|{a['mean']:+.2f} [{a['low']:+.2f}, {a['high']:+.2f}]|")
    lines += ["", "## Costs", "", "SR construction is N/A: sealed artifact reused, not a zero-cost constructor. Time per seed includes all three stages. Text tokens count stored text before encoder truncation. Tensor bytes cover exported Student memory, not constructor auxiliary state.", "", "|Method|Construction seconds|Student seconds per seed|Final tensor bytes|Final UTF-8 bytes|Final text tokens|", "|---|---:|---:|---:|---:|---:|"]
    for r in overall:
        lines.append("|" + NAMES[r["arm"]] + "|" + "|".join(fmt(r[k]) for k in ["construction_seconds_total", "student_seconds_all_stages_per_seed", "tensor_bytes", "text_utf8_bytes", "text_tokens"]) + "|")
    lines += ["", f"Previous SR maximum per-seed metric difference: {result['previous_sr_max_metric_error_pp']} pp.", "", result["limits"], ""]
    (args.output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"fits": result["fit_count"], "overall": overall, "paired": comparisons,
                      "sr_replay_error": result["previous_sr_max_metric_error_pp"]}, indent=2))


if __name__ == "__main__":
    main()
