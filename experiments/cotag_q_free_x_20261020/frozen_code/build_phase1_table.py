"""Source-backed canonical baseline table for the frozen C0 experiment."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

import run_frozen_baseline_joint_kernel as E

C, P = E.C, E.P


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in r.items()} for r in rows)


def summarize(stages, selected=False):
    stages = sorted(stages, key=lambda s: s["stage"])
    field = "selected_metrics" if selected else "metrics"
    values = [[r[field]["all"]["cumulative_f1"] for r in s["per_seed"]] for s in stages]
    old = [r[field]["all"]["old_f1"] for r in stages[-1]["per_seed"]]
    new = [r[field]["all"]["new_f1"] for r in stages[-1]["per_seed"]]
    times = [sum(r["seconds"] for r in s["per_seed"]) for s in stages]
    return {"final_f1": float(np.mean(values[-1])), "final_std": float(np.std(values[-1])),
            "stage_average_f1": float(np.mean(values)), "final_old_f1": float(np.mean(old)),
            "final_new_f1": float(np.mean(new)), "stage_f1": np.mean(values, axis=1).tolist(),
            "final_per_seed": values[-1], "student_seconds_six_seeds": sum(times),
            "student_seconds_one_seed_chain": sum(times) / len(stages[0]["per_seed"]),
            "student_steps_six_seeds": sum(r["steps"] for s in stages for r in s["per_seed"]),
            "stage_rows": [s["rows"] for s in stages], "final_rows": stages[-1]["rows"],
            "peak_allocated_mb": max(r["peak_allocated_mb"] for s in stages for r in s["per_seed"])}


def access(arm):
    if arm.startswith("raw"):
        return "full_seen_raw", "real", "full_data_reference"
    if arm.startswith("tagsam") or arm == "full_history_syn":
        return "full_seen_raw", "synthetic", "global_recondensation"
    if arm == "recluster":
        return "old_synthetic_plus_delta", "synthetic", "rewrite_old_memory"
    real = arm in ("random_real", "k_center", "k_center_class", "herding", "kmeans_medoid")
    return "old_memory_plus_delta", "real" if real else "synthetic", "append_only"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--new", type=Path, required=True)
    args = ap.parse_args()
    out = args.new / "summary"
    out.mkdir(exist_ok=True)
    rows, costs, errors, sources, replication = [], [], [], {}, []
    checked = 0
    prior_path = args.results / "five_level_summary_2026_09_17/acceptance.json"
    prior = json.loads(prior_path.read_text())
    for source, digest in prior["sources"].items():
        path = args.results / Path(source).relative_to("/home/pc/research/TAG_condensation/results")
        if C.sha(path) != digest:
            errors.append(f"historical accepted source changed: {path}")
    if prior["error_count"]:
        errors.append("historical acceptance has errors")
    sources[str(prior_path)] = C.sha(prior_path)

    for dataset in ("cora", "citeseer"):
        core_path = args.results / f"small_comparison_audited_r2_2026_09_16/{dataset}/results.json"
        static_path = args.results / f"five_level_extension_2026_09_17/static/{dataset}/results.json"
        fresh_path = args.new / f"matrix/{dataset}/s3_canonical_r05/results.json"
        random_path = args.new / f"baseline/{dataset}/s3_canonical_r05/results.json"
        paths = [core_path, static_path, fresh_path, random_path]
        loaded = [json.loads(p.read_text()) for p in paths]
        core, static, fresh, random_control = loaded
        for path in paths:
            sources[str(path)] = C.sha(path)
        canonical = fresh["manifest"]
        for historic in (core, static):
            m = historic["manifest"]
            if (m["budgets"] != canonical["budgets"] or m["stage_classes"] != canonical["stage_classes"]
                    or m.get("seeds", m.get("student_seeds")) != list(range(100,106))
                    or m["test_labels_evaluated"]):
                errors.append(f"{dataset}: incompatible historical protocol")
        specs = [("c0", fresh_path, fresh, False), ("random_synthetic", random_path, random_control, False)]
        specs += [(a, fresh_path, fresh, False) for a in ("k_center", "recluster")]
        specs += [(a, core_path, core, False) for a in (
            "random_real", "k_center_class", "kmeans_medoid", "herding", "full_history_syn", "raw_equal", "raw_long")]
        specs.append(("raw_long", core_path, core, True))
        specs += [(a, static_path, static, False) for a in ("tagsam_static_selfloop", "tagsam_static_semantic")]

        for arm, path, data, selected in specs:
            label = "raw_long_selected" if selected else arm
            stages = sorted((s for s in data["results"] if s["arm"] == arm), key=lambda s: s["stage"])
            if len(stages) != 3:
                raise ValueError(f"incomplete baseline: {dataset}/{label}")
            for s in stages:
                if [r["seed"] for r in s["per_seed"]] != list(range(100,106)):
                    errors.append(f"{dataset}/{label}: seed list")
                memory_path = path.parent / f"{arm}_s{s['stage']}_memory.pt"
                memory = torch.load(memory_path, map_location="cpu", weights_only=False)
                audit = json.loads(memory_path.with_name(f"{arm}_s{s['stage']}_construction.json").read_text())
                if C.tensor_hash(memory) != audit["tensor_sha256"]:
                    errors.append(f"{dataset}/{label}: memory hash")
                if not arm.startswith("raw") and len(memory["x"]) != canonical["budgets"][s["stage"]]:
                    errors.append(f"{dataset}/{label}: budget")
                for row in s["per_seed"]:
                    cp_path = path.parent / f"{arm}_s{s['stage']}_seed{row['seed']}.pt"
                    saved = torch.load(cp_path, map_location="cpu", weights_only=False)
                    if C.tensor_hash(saved["state"]) != row["end_state_sha256"] or row["support_sha256"] != C.tensor_hash(memory):
                        errors.append(f"{dataset}/{label}: checkpoint state/support")
                    if row["steps"] != (8192 if arm == "raw_long" else P.student_steps(canonical["budgets"][s["stage"]])):
                        errors.append(f"{dataset}/{label}: training steps")
                    checked += int(not selected)
                costs.append({"dataset": dataset, "arm": label, "stage": s["stage"],
                    "build_seconds": audit.get("build_seconds"), "historical": path in (core_path, static_path),
                    "payload_bytes": E.payload(memory)["logical_bytes"], "edge_entries": memory["edge_index"].shape[1],
                    "memory_file_sha256": C.sha(memory_path), "memory_tensor_sha256": C.tensor_hash(memory),
                    "source_results_sha256": C.sha(path)})
            memory_bytes = [c["payload_bytes"] for c in costs if c["dataset"] == dataset and c["arm"] == label]
            builds = [c["build_seconds"] for c in costs if c["dataset"] == dataset and c["arm"] == label]
            read, node, management = access(arm)
            row = {"dataset": dataset, "arm": label, **summarize(stages, selected),
                   "source_results": str(path), "source_sha256": C.sha(path), "data_access": read,
                   "node_type": node, "memory_management": management,
                   "construction_class_labels": arm in ("c0", "random_synthetic", "k_center_class", "full_history_syn") or arm.startswith("tagsam"),
                   "timing_source": "historical" if path in (core_path, static_path) else "fresh",
                   "build_seconds_chain": sum(builds) if all(v is not None for v in builds) else None,
                   "final_memory_bytes": memory_bytes[-1]}
            rows.append(row)

        for arm, archived_arm in (("c0", "dispersion"), ("k_center", "k_center"), ("recluster", "recluster")):
            stages = sorted((s for s in fresh["results"] if s["arm"] == arm), key=lambda s: s["stage"])
            archived = sorted((s for s in core["results"] if s["arm"] == archived_arm), key=lambda s: s["stage"])
            for current, previous in zip(stages, archived):
                new_m = torch.load(fresh_path.parent / f"{arm}_s{current['stage']}_memory.pt", map_location="cpu", weights_only=False)
                old_m = torch.load(core_path.parent / f"{archived_arm}_s{current['stage']}_memory.pt", map_location="cpu", weights_only=False)
                differences = [r["metrics"]["all"]["cumulative_f1"] - p["metrics"]["all"]["cumulative_f1"]
                               for r, p in zip(current["per_seed"], previous["per_seed"])]
                replication.append({"dataset": dataset, "arm": arm, "stage": current["stage"],
                    "same_training_tensors_and_text": E.J_hash(new_m) == E.J_hash(old_m),
                    "per_seed_f1_delta": differences})
        full = next(r for r in rows if r["dataset"] == dataset and r["arm"] == "raw_long_selected")
        c0 = next(r for r in rows if r["dataset"] == dataset and r["arm"] == "c0")
        for row in (r for r in rows if r["dataset"] == dataset):
            row["retention_vs_raw_selected_percent"] = 100 * row["final_f1"] / full["final_f1"]
            row["stage_avg_retention_percent"] = 100 * row["stage_average_f1"] / full["stage_average_f1"]
            row["c0_minus_method_pp"] = c0["final_f1"] - row["final_f1"]
            row["node_ratio_percent"] = 100 * row["final_rows"] / full["final_rows"]
    result = {"rows": rows, "replication": replication, "sources": sources,
              "historical_acceptance": prior, "scope": "common-student report-dev; TAGSAM partial adaptation; historical costs not synchronized"}
    C.save_json(out / "phase1_summary.json", result)
    write_csv(out / "phase1_summary.csv", rows)
    write_csv(out / "phase1_costs.csv", costs)
    acceptance = {"errors": errors, "error_count": len(errors), "checkpoint_support_pairs_checked": checked,
                  "source_hashes": sources, "script_sha256": C.sha(__file__), "test_labels_evaluated": False,
                  "prior_model_reevaluations_reused": prior["extension_saved_models_reevaluated"],
                  "fresh_models_covered_by_separate_matrix_audit": True}
    C.save_json(out / "phase1_acceptance.json", acceptance)
    print(json.dumps({k:v for k,v in acceptance.items() if k != "source_hashes"}), flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
