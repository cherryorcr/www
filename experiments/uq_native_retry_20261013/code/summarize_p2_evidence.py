"""Build source-backed P2 tables and a final experiment-completeness ledger."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from audit_p2_token_controls import effect


BASE = "concat_t64_g5_r0.5_a0.5"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scores(row, key="per_student"):
    return [s["cumulative_macro_f1"] for s in row[key]]


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--output", default="results/p2_acceptance_2026_09_16")
    args = ap.parse_args()
    root, out = Path(args.results), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    matrix = root / "p2_token_controls_2026_09_16"
    confirmation = root / "p2_token_confirmation_2026_09_16"
    numeric = root / "p2_numerical_guard_2026_09_16"
    metrics = root / "p2_metric_diagnostics_2026_09_16"
    rate = root / "p2_token_rate0025_2026_09_16"
    errors, confirmation_rows, numeric_rows, metric_rows, stage_rows = [], [], [], [], []
    audit = read(matrix / "audit.json")
    if not audit["integrity_passed"] or not audit["actual_tokenizer_checks"]:
        errors.append("strict matrix remote artifact/tokenizer audit failed")
    matrices = {}
    total_training_runs = 0
    for dataset in ("cora", "citeseer"):
        matrices[dataset] = {p.parent.name: read(p) for p in (matrix / dataset).glob("*/results.json")}
        for name, d in matrices[dataset].items():
            for row in d["results"]:
                total_training_runs += row["student_training_runs"]
                stage_rows.append({"dataset": dataset, "name": name, "stage": row["stage"],
                    "rows": row["roots"], "steps": row["student_steps"],
                    "f1": row["cum_dev_f1_mean"], "old_task_slice_f1": row["old_mean_f1_mean"],
                    "new_task_slice_f1": row["current_f1_mean"],
                    "tokens": row["text_audit"]["encoder_content_token_total"],
                    "build_seconds": row["build_seconds"], "train_seconds": row["student_train_seconds"]})
        for cohort, expected in (("independent", 5), ("batch8", 5), ("batch32", 5), ("reverse", 3)):
            files = list((confirmation / dataset / cohort).glob("*/results.json"))
            if len(files) != expected:
                errors.append(f"{dataset}/{cohort}: expected {expected}, got {len(files)}")
                continue
            base = read(confirmation / dataset / cohort / BASE / "results.json")
            for path in files:
                d = read(path)
                source = matrix / dataset / d["name"]
                if d["source_results_sha256"] != digest(source / "results.json") or d["test_accessed"]:
                    errors.append(f"{dataset}/{cohort}/{d['name']}: source/sealing check")
                expected_seeds = list(range(400, 406)) if cohort == "independent" else list(range(100, 106))
                if d["student_seeds"] != expected_seeds or len(d["results"]) != 3:
                    errors.append(f"{dataset}/{cohort}/{d['name']}: seed/stage check")
                for r, b in zip(d["results"], base["results"]):
                    total_training_runs += r["student_training_runs"]
                    if any(r[k] != b[k] for k in ("roots", "student_steps", "student_batch", "student_seeds")):
                        errors.append(f"{dataset}/{cohort}/{d['name']}: matched training check")
                    if r["source_memory_sha256"] != digest(source / f"append_fixed_s{r['stage']}_memory.pt"):
                        errors.append(f"{dataset}/{cohort}/{d['name']}: memory hash check")
                r, b = d["results"][-1], base["results"][-1]
                row = {"dataset": dataset, "cohort": cohort, "name": d["name"],
                    "final_f1": r["cum_dev_f1_mean"], **effect(scores(r), scores(b)),
                    "disjoint_dev_f1": r["disjoint_dev_f1_mean"],
                    "disjoint_dev_gain": r["disjoint_dev_f1_mean"] - b["disjoint_dev_f1_mean"],
                    "stage_average_gain": float(np.mean([a["cum_dev_f1_mean"] - z["cum_dev_f1_mean"]
                                                         for a, z in zip(d["results"], base["results"])])),
                    "max_embedding_change": max(x["text_order_max_embedding_change"] for x in d["results"])}
                confirmation_rows.append(row)
        for path in sorted((numeric / dataset).glob("*/results.json")):
            d = read(path)
            source_name = f"{d['operator']}_t64_g5_r0.5_a0.5"
            source = matrix / dataset / source_name
            for r in d["results"]:
                total_training_runs += r["student_training_runs"]
                if d["test_accessed"] or r["source_memory_sha256"] != digest(source / f"append_fixed_s{r['stage']}_memory.pt"):
                    errors.append(f"{dataset}/{path.parent.name}: source/sealing check")
                if d["mode"] in {"identity", "same_order"} and (not r["matches_original_scores"] or r["max_embedding_change"] != 0):
                    errors.append(f"{dataset}/{path.parent.name}: identity replay failure")
                if d["mode"].startswith("canonical") and r["canonical_order_invariance_error"] != 0:
                    errors.append(f"{dataset}/{path.parent.name}: non-invariant canonicalization")
                mem = torch.load(path.parent / f"append_fixed_s{r['stage']}_memory.pt", map_location="cpu", weights_only=False)
                ref = torch.load(source / f"append_fixed_s{r['stage']}_memory.pt", map_location="cpu", weights_only=False)
                if any(not torch.equal(mem[k], ref[k]) for k in ("x", "mass", "depth", "stage", "edge_index")):
                    errors.append(f"{dataset}/{path.parent.name}: fixed-memory failure")
            if d["mode"] == "canonical_independent":
                baseline = read(confirmation / dataset / "independent" / BASE / "results.json")
            else:
                baseline = matrices[dataset][BASE]
            r = d["results"][-1]
            numeric_rows.append({"dataset": dataset, "operator": d["operator"], "mode": d["mode"],
                "final_f1": r["cum_dev_f1_mean"], **effect(scores(r), scores(baseline["results"][-1])),
                "max_embedding_change": max(z["max_embedding_change"] for z in d["results"]),
                "identity_match_all_stages": all(z["matches_original_scores"] for z in d["results"]),
                "canonical_invariance_error": max(z["canonical_order_invariance_error"] for z in d["results"])})
        if len(list((numeric / dataset).glob("*/results.json"))) != 10:
            errors.append(f"{dataset}: numerical guard incomplete")
        for path in sorted((metrics / dataset).glob("*.json")):
            d = read(path)
            if d["test_accessed"] or len(d["rows"]) != 3:
                errors.append(f"{dataset}/{path.name}: metric diagnostic failed")
            for r in d["rows"]:
                total_training_runs += len(d["student_seeds"])
                if not r["canonical_replay_match"]:
                    errors.append(f"{dataset}/{path.name}: metric identity replay failed")
                metric_rows.append({"dataset": dataset, "operator": d["operator"], "stage": r["stage"],
                    "all_known_prompts_f1": r["cumulative_f1"], "seen_only_prompts_f1": r["seen_prompt_only_f1"],
                    "old_f1_global_support": r["old_class_f1_global_support"],
                    "new_f1_global_support": r["new_class_f1_global_support"]})
        if len(list((metrics / dataset).glob("*.json"))) != 3:
            errors.append(f"{dataset}: metric diagnostics incomplete")
    rate_audit = read(rate / "audit.json")
    if not rate_audit["integrity_passed"] or not rate_audit["actual_tokenizer_checks"]:
        errors.append("secondary rate artifact/tokenizer audit failed")
    for dataset in ("cora", "citeseer"):
        total_training_runs += rate_audit["datasets"][dataset]["training_runs"]
    regressions = []
    for path in (root / "p2_legacy_regression_2026_09_16").glob("*/results.json"):
        d = read(path)
        total_training_runs += sum(r["student_training_runs"] for r in d["results"])
        final = d["results"][-1]["cum_dev_f1_mean"]
        if d["rate"] == 0.05 and abs(final - {"cora": 49.85, "citeseer": 59.17}[d["dataset"]]) > 0.01:
            errors.append(f"{d['dataset']}: legacy default regression")
        regressions.append({"dataset": d["dataset"], "rate": d["rate"], "final_f1": final})
    if len(regressions) != 4:
        errors.append("legacy default/rate references incomplete")
    independent_pass = []
    for name in {r["name"] for r in confirmation_rows if r["cohort"] == "independent" and r["name"] != BASE}:
        rs = [r for r in confirmation_rows if r["name"] == name and r["cohort"] == "independent"]
        if len(rs) == 2 and all(r["ci_low"] > 0 and r["stage_average_gain"] >= 0 for r in rs):
            independent_pass.append(name)
    summary = {"integrity_passed": not errors, "errors": errors,
        "strict_matrix_runs": sum(d["complete_runs"] for d in audit["datasets"].values()),
        "confirmation_runs": len(confirmation_rows), "numerical_runs": len(numeric_rows),
        "secondary_rate_runs": sum(d["complete_runs"] for d in rate_audit["datasets"].values()),
        "metric_diagnostic_runs": len(metric_rows) // 3, "legacy_reference_runs": len(regressions),
        "total_archived_student_fits_excluding_character_pilot": total_training_runs,
        "independent_confirmation_pass": sorted(independent_pass),
        "large_graph_gate_passed": False,
        "gate_note": "Requires cross-dataset, independent-seed, stage-retention and rate evidence; no automatic promotion from a maximum dev score.",
        "reverse_v1_mean_has_reduction_confound": True,
        "regressions": regressions}
    (out / "acceptance.json").write_text(json.dumps(summary, indent=2))
    write_csv(out / "stage_curves.csv", stage_rows)
    write_csv(out / "confirmation.csv", confirmation_rows)
    write_csv(out / "numerical_guard.csv", numeric_rows)
    write_csv(out / "metric_scopes.csv", metric_rows)
    print(json.dumps(summary))
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
