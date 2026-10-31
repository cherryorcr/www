"""Recompute R4 paired comparisons from sealed per-seed records, not console logs."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics as st


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def interval(values):
    critical = {4: 3.182446305, 6: 2.570581836}.get(len(values))
    mean = st.mean(values)
    if critical is None:
        raise ValueError("unregistered interval size")
    half = critical * st.stdev(values) / math.sqrt(len(values))
    return [mean - half, mean + half]


def csv_file(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def promotion_decision(main, random, costs, limits):
    return {"mean_final": main["final_delta_pp"] >= limits["mean_final_min_pp"],
            "cells_ge_minus1": main["cells_ge_minus1"] >= limits["cells_ge_minus1_min"],
            "worst_final": main["worst_final_delta_pp"] >= limits["worst_final_min_pp"],
            "worst_old": main["worst_final_old_delta_pp"] >= limits["worst_old_min_pp"],
            "mean_new": main["final_new_delta_pp"] >= limits["mean_new_min_pp"],
            "mean_vs_random": main["final_mean_f1"] - random["final_mean_f1"] >= limits["mean_vs_random_min_pp"],
            "build_seconds": max(c["executed_primary_build_with_diagnostics_seconds"] for c in costs)
                             <= limits["max_build_seconds_per_stage"]}


def summarize(root):
    p = read(root / "protocol.json")
    seal = read(root / "selection_seal.json")
    assert seal["protocol_sha256"] == digest(root / "protocol.json")
    for name, sha in seal["decisions"].items():
        assert digest(root / name) == sha, name
    per_seed, cells, supports, optimizations, costs, audits = [], [], [], [], [], []
    for dataset in p["datasets"]:
        for order in p["orders"]:
            folder = root / dataset / order
            audits.append({"dataset": dataset, "order": order, **read(folder / "audit.json")})
            assert audits[-1]["status"] == "passed"
            manifest = read(folder / "manifest.json")
            for stage in range(3):
                base = {seed: read(folder / f"c0_s{stage}_seed{seed}.json") for seed in p["report_seeds"]}
                for arm in p["arms"]:
                    values = []
                    for seed in p["report_seeds"]:
                        item = read(folder / f"{arm}_s{stage}_seed{seed}.json")
                        assert item["seed"] == seed and item["stage"] == stage and item["arm"] == arm
                        assert item["start_state_sha256"] == base[seed]["start_state_sha256"]
                        assert item["steps"] == base[seed]["steps"] and item["rows"] == manifest["budgets"][stage]
                        row = {"dataset": dataset, "order": order, "stage": stage, "arm": arm, "seed": seed,
                               "rows": item["rows"], "steps": item["steps"],
                               "f1": item["metrics"]["all"]["cumulative_f1"],
                               "seen_f1": item["metrics"]["seen"]["cumulative_f1"],
                               "old_f1": item["metrics"]["all"]["old_f1"],
                               "new_f1": item["metrics"]["all"]["new_f1"],
                               "delta_f1": item["metrics"]["all"]["cumulative_f1"] - base[seed]["metrics"]["all"]["cumulative_f1"],
                               "delta_old": (item["metrics"]["all"]["old_f1"] - base[seed]["metrics"]["all"]["old_f1"]) if stage else None,
                               "delta_new": item["metrics"]["all"]["new_f1"] - base[seed]["metrics"]["all"]["new_f1"],
                               "seconds": item["seconds"], "executed": item["executed_here"],
                               "peak_mb": item["peak_allocated_mb"], "source_sha256": digest(folder / f"{arm}_s{stage}_seed{seed}.json")}
                        per_seed.append(row)
                        values.append(row)
                    ci = interval([v["delta_f1"] for v in values])
                    cells.append({"dataset": dataset, "order": order, "stage": stage, "arm": arm,
                                  "f1": st.mean(v["f1"] for v in values), "f1_sd": st.stdev(v["f1"] for v in values),
                                  "delta_f1": st.mean(v["delta_f1"] for v in values),
                                  "paired_ci_low": ci[0], "paired_ci_high": ci[1],
                                  "old_f1": st.mean(v["old_f1"] for v in values) if stage else None,
                                  "new_f1": st.mean(v["new_f1"] for v in values),
                                  "delta_old": st.mean(v["delta_old"] for v in values) if stage else None,
                                  "delta_new": st.mean(v["delta_new"] for v in values),
                                  "seconds": st.mean(v["seconds"] for v in values)})
                if not stage:
                    continue
                ctx = read(folder / f"s{stage}_support_diagnostics.json")
                for condition, details in ctx["supports"].items():
                    curve = details["direct_distillation"]["curve"]
                    supports.append({"dataset": dataset, "order": order, "stage": stage, "condition": condition,
                                     "old_rows": ctx["old_rows"], "raw_delta_rows": ctx["current_raw_rows"],
                                     "compact_rows": ctx["compact_rows"],
                                     "sampled_old_fraction": st.mean(t["sampled_old_fraction"] for t in details["teachers"]),
                                     "old_normalized_block_error": details["old_normalized_block_max_error"],
                                     "old_retrieval": details["ensemble_probes"]["validation"]["old_retrieval"],
                                     "new_retrieval": details["ensemble_probes"]["validation"]["new_retrieval"],
                                     "teacher_disagreement_kl": details["teacher_disagreement_kl"]["validation"],
                                     "direct_kl_step1": curve[0]["validation_kl"],
                                     "direct_kl_last": curve[-1]["validation_kl"],
                                     "teacher_seconds": sum(v["seconds"] for v in details["teachers"])})
                for arm in (*p["optimization_specs"], "stop"):
                    opt = read(folder / f"{arm}_s{stage}_optimization.json")
                    optimizations.append({"dataset": dataset, "order": order, "stage": stage, "arm": arm,
                                          "train_improvement": opt["initial"]["train"]["new"] - opt["final"]["train"]["new"],
                                          "validation_improvement": opt["initial"]["validation"]["new"] - opt["final"]["validation"]["new"],
                                          "final_old_train_kl": opt["final"]["train"]["old_kl"],
                                          "final_old_validation_kl": opt["final"]["validation"]["old_kl"],
                                          "active_steps": sum(v["constraint_active"] for v in opt["curve"]),
                                          "final_scale": opt["feasibility"][-1]["scale"],
                                          "changed_rows": len(opt["invariants"]["changed_rows"]),
                                          "max_displacement": opt["invariants"]["max_relative_displacement"],
                                          "seconds": opt["seconds"], "peak_mb": opt["peak_cuda_bytes"] / 2**20})
                c0 = read(folder / f"c0_s{stage}_construction.json")
                rel = read(folder / f"relation_new_s{stage}_optimization.json")
                selected_teacher = ctx["supports"][p["support"]]["teachers"]
                payload = read(folder / f"s{stage}_decision.json")["memories"]["relation_new"]["payload"]
                cost = {"dataset": dataset, "order": order, "stage": stage,
                        "c0_seconds": c0["seconds"], "support_seconds": ctx["support_materialization_seconds"],
                        "all_support_diagnostics_seconds": ctx["seconds"],
                        "selected_teacher_seconds": sum(t["seconds"] for t in selected_teacher),
                        "optimization_seconds": rel["seconds"],
                        "executed_primary_build_with_diagnostics_seconds": c0["seconds"] + ctx["support_materialization_seconds"] + ctx["seconds"] + rel["seconds"],
                        "selected_components_lower_bound_seconds": c0["seconds"] + ctx["support_materialization_seconds"] + sum(t["seconds"] for t in selected_teacher) + rel["seconds"],
                        "logical_memory_bytes": payload["logical_bytes"], "peak_optimization_mb": rel["peak_cuda_bytes"] / 2**20}
                costs.append(cost)
    assert len(per_seed) == 792 and sum(r["executed"] for r in per_seed) == 504
    methods = []
    for arm in p["arms"]:
        final = [v for v in cells if v["arm"] == arm and v["stage"] == 2]
        s1 = [v for v in cells if v["arm"] == arm and v["stage"] == 1]
        methods.append({"arm": arm, "final_mean_f1": st.mean(v["f1"] for v in final),
                        "final_delta_pp": st.mean(v["delta_f1"] for v in final),
                        "final_cell_descriptive_ci": interval([v["delta_f1"] for v in final]),
                        "stage1_delta_pp": st.mean(v["delta_f1"] for v in s1),
                        "stage_mean_delta_pp": st.mean(v["delta_f1"] for v in cells if v["arm"] == arm),
                        "final_old_delta_pp": st.mean(v["delta_old"] for v in final),
                        "final_new_delta_pp": st.mean(v["delta_new"] for v in final),
                        "worst_final_delta_pp": min(v["delta_f1"] for v in final),
                        "worst_final_old_delta_pp": min(v["delta_old"] for v in final),
                        "cells_ge_minus1": sum(v["delta_f1"] >= -1 for v in final)})
    by_arm = {r["arm"]: r for r in methods}
    main = by_arm["relation_new"]
    gates = promotion_decision(main, by_arm["random_new"], costs, p["promotion"])
    result = {"protocol_sha256": digest(root / "protocol.json"), "selection_seal_sha256": digest(root / "selection_seal.json"),
              "fit_slots": len(per_seed), "executed_fits": sum(r["executed"] for r in per_seed),
              "methods": methods, "promotion_gates": gates, "primary_eligible_for_confirmation": all(gates.values()),
              "main_mean_vs_random_pp": main["final_mean_f1"] - by_arm["random_new"]["final_mean_f1"],
              "audits": audits, "costs": costs,
              "ci_scope": "paired t across six student seeds per fixed cell; four-cell t is descriptive, not iid population inference",
              "dataset_repeats_are_not_new_datasets": True,
              "total_actual_student_seconds": sum(r["seconds"] for r in per_seed if r["executed"])}
    for name, records in (("per_seed", per_seed), ("cells", cells), ("supports", supports),
                           ("optimizations", optimizations), ("costs", costs)):
        csv_file(root / f"{name}.csv", records)
    (root / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(summarize(args.root), indent=2))
