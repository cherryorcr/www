"""Compare frozen v4 and external TAG baselines using source per-seed results."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t


CORE_ARMS = ["v4", "random_real", "k_center", "k_center_class", "herding",
             "kmeans_medoid", "full_history_syn", "raw_equal", "raw_long"]
LABELS = {
    "v4": "v4 (frozen continual synthetic memory)",
    "stad_local_canonical": "STAD-local-canonical (static TAG adaptation)",
    "tagsam_static_selfloop": "TAGSAM-ASM self-loop (static adaptation)",
    "tagsam_static_semantic": "TAGSAM-ASM semantic (static adaptation)",
    "random_real": "Random real-node memory", "k_center": "k-center real-node memory",
    "k_center_class": "Class-aware k-center real-node memory", "herding": "Herding real-node memory",
    "kmeans_medoid": "K-means medoid real-node memory",
    "full_history_syn": "Heuristic global recondensation",
    "raw_equal": "Full data, equal optimizer steps",
    "raw_long_selected": "Full data, 8192 steps with selector checkpoint",
}


def read(path, sources):
    sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text(encoding="utf-8"))


def collect(dataset, arm, stages, selected=False):
    stages = sorted(stages, key=lambda r: r["stage"])
    if [r["stage"] for r in stages] != [0, 1, 2]:
        raise ValueError(f"incomplete stage coverage: {dataset}/{arm}")
    key = "selected_metrics" if selected else "metrics"
    values = []
    for stage in stages:
        fits = sorted(stage["per_seed"], key=lambda r: r["seed"])
        if [r["seed"] for r in fits] != list(range(100, 106)):
            raise ValueError(f"incomplete seeds: {dataset}/{arm}")
        values.append([r[key]["all"]["cumulative_f1"] for r in fits])
    final = np.array(values[-1])
    return {"dataset": dataset, "arm": arm, "label": LABELS[arm],
            "rows": stages[-1]["rows"], "final_f1": float(final.mean()),
            "std_population": float(final.std()), "final_per_seed": final.tolist(),
            "final_old_f1": float(np.mean([r[key]["all"]["old_f1"] for r in fits])),
            "final_new_f1": float(np.mean([r[key]["all"]["new_f1"] for r in fits])),
            "stage_f1": [float(np.mean(v)) for v in values],
            "stage_average_f1": float(np.mean(values)),
            "student_seconds": sum(r["seconds"] for s in stages for r in s["per_seed"]),
            "student_steps": sum(r["steps"] for s in stages for r in s["per_seed"])}


def add_reference(rows):
    for dataset in ("cora", "citeseer"):
        local = [r for r in rows if r["dataset"] == dataset]
        ref = next(r for r in local if r["arm"] == "raw_long_selected")
        v4 = next(r for r in local if r["arm"] == "v4")
        for row in local:
            row["retention_percent"] = 100 * row["final_f1"] / ref["final_f1"]
            if "stage_average_f1" in row:
                row["stage_average_retention_percent"] = (100 * row["stage_average_f1"]
                                                           / ref["stage_average_f1"])
            row["node_ratio_percent"] = 100 * row["rows"] / ref["rows"]
            row["delta_vs_full_pp"] = row["final_f1"] - ref["final_f1"]
            diff = np.array(v4["final_per_seed"]) - np.array(row["final_per_seed"])
            margin = student_t.ppf(.975, len(diff) - 1) * diff.std(ddof=1) / np.sqrt(len(diff))
            row["v4_minus_method_pp"] = float(diff.mean())
            row["v4_minus_method_seed_ci95"] = [float(diff.mean() - margin), float(diff.mean() + margin)]
    return rows


def collect_supervised_sanity(root, sources):
    results = []
    for dataset in ("cora", "citeseer"):
        base = root / "stad_supervised_sanity_2026_09_17" / dataset
        audit = read(base / "acceptance.json", sources)
        if audit["errors"] or audit["fits_restored"] != 12 or audit["report_metrics_recomputed"] != 24:
            raise ValueError("incomplete supervised sanity audit")
        payload = read(base / "results.json", sources)
        if sorted(r["arm"] for r in payload["results"]) != ["raw_supervised_matched", "stad_supervised"]:
            raise ValueError("incomplete supervised sanity arms")
        local = []
        for row in payload["results"]:
            fits = sorted(row["per_seed"], key=lambda r: r["seed"])
            if [r["seed"] for r in fits] != list(range(100, 106)) or any(
                    r["stage"] != 2 or r["steps"] != 600 for r in fits):
                raise ValueError("incorrect supervised sanity stage/seeds/steps")
            values = [r["selected_metrics"]["cumulative_f1"] for r in fits]
            local.append({"dataset": dataset, "arm": row["arm"], "rows": row["rows"],
                          "final_f1": float(np.mean(values)), "std_population": float(np.std(values)),
                          "accuracy": float(np.mean([r["selected_metrics"]["accuracy"] for r in fits])),
                          "student_seconds": sum(r["seconds"] for r in fits)})
        ref = next(r for r in local if r["arm"] == "raw_supervised_matched")
        for row in local:
            row["retention_within_supervised_task_percent"] = 100 * row["final_f1"] / ref["final_f1"]
        results.extend(local)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    rows, sources, acceptance, costs = [], {}, {}, {}
    independent = read(args.results_root / "external_tag_stad_2026_09_17" /
                       "independent_acceptance.json", sources)
    if (sorted(r["dataset"] for r in independent["datasets"]) != ["citeseer", "cora"]
            or any(r["errors"] or not r["v4_seed100_state_and_metrics_bitwise_regression"]
                   or [s["stage"] for s in r["stages"]] != [0, 1, 2]
                   for r in independent["datasets"])):
        raise ValueError("STAD independent artifact/regression audit is incomplete")
    for dataset in ("cora", "citeseer"):
        core = read(args.results_root / "small_comparison_audited_r2_2026_09_16" / dataset / "results.json", sources)
        static = read(args.results_root / "five_level_extension_2026_09_17/static" / dataset / "results.json", sources)
        stad_dir = args.results_root / "external_tag_stad_2026_09_17" / dataset
        stad = read(stad_dir / "results.json", sources)
        accepted = read(stad_dir / "acceptance.json", sources)
        if accepted["errors"] or accepted["recomputed_reporting_fits"] != 18:
            raise ValueError(f"STAD has not passed model-restore audit: {dataset}")
        acceptance[dataset] = accepted
        for arm in CORE_ARMS:
            stages = [r for r in core["results"] if r["arm"] == arm]
            rows.append(collect(dataset, "raw_long_selected" if arm == "raw_long" else arm,
                                stages, selected=arm == "raw_long"))
        for arm in ("tagsam_static_selfloop", "tagsam_static_semantic"):
            rows.append(collect(dataset, arm, [r for r in static["results"] if r["arm"] == arm]))
        rows.append(collect(dataset, "stad_local_canonical", stad["results"]))
        costs[dataset] = {
            "stage_rows": [a["rows"] for a in stad["audits"]],
            "stage_edges": [a["edges"] for a in stad["audits"]],
            "stage_build_seconds": [a["build_seconds_this_attempt"] for a in stad["audits"]],
            "stage_selected_minilm_tokens": [a["minilm_tokens"] for a in stad["audits"]],
            "llm_candidates": sum(a["llm_candidates"] for a in stad["audits"]),
            "llm_prompt_tokens": sum(a["llm_usage"]["prompt_tokens"] for a in stad["audits"]),
            "llm_completion_tokens": sum(a["llm_usage"]["completion_tokens"] for a in stad["audits"]),
        }
    rows = add_reference(rows)
    native_sanity = collect_supervised_sanity(args.results_root, sources)
    args.output.mkdir(parents=True, exist_ok=True)
    payload = {"rows": rows, "sources": sources, "stad_model_audits": acceptance,
               "stad_independent_audit": independent,
               "supervised_sanity_separate_task": native_sanity,
               "stad_costs": costs, "retention_definition": "ratio of means, not mean of per-seed ratios",
               "ci_scope": "paired student seeds conditional on one constructor; no construction variance",
               "test_labels_evaluated": False}
    (args.output / "summary.json").write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    with (args.output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("| Method | Cora F1 | Retention | Citeseer F1 | Retention |")
    print("|---|---:|---:|---:|---:|")
    for arm in LABELS:
        first = next(r for r in rows if r["arm"] == arm and r["dataset"] == "cora")
        second = next(r for r in rows if r["arm"] == arm and r["dataset"] == "citeseer")
        print(f"| {LABELS[arm]} | {first['final_f1']:.2f} +/- {first['std_population']:.2f} "
              f"| {first['retention_percent']:.2f}% | {second['final_f1']:.2f} +/- {second['std_population']:.2f} "
              f"| {second['retention_percent']:.2f}% |")


if __name__ == "__main__":
    main()
