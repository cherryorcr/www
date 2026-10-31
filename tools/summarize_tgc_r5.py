"""Paired R5 results, factorial interactions, recursive contrasts and costs."""
import argparse
import json
from pathlib import Path
import statistics as st

from summarize_tgc_r4_relational import read, digest, interval, csv_file


def summarize(root):
    protocol = read(root / "protocol.json")
    seal = read(root / "selection_seal.json")
    assert seal["protocol_sha256"] == digest(root / "protocol.json")
    for path, sha in seal["decisions"].items():
        assert digest(root / path) == sha, path
    assert read(root / "pipeline_status.json")["status"] == "complete"
    rows, cells, audits, costs, mechanisms = [], [], [], [], []
    joint = "joint" in protocol["arms"]
    for dataset in protocol["datasets"]:
        for order in protocol["orders"]:
            folder = root / dataset / order
            audit = read(folder / "audit.json")
            assert audit["status"] == "passed"
            audits.append({"dataset": dataset, "order": order, **audit})
            for stage in range(3):
                baseline = {s: read(folder / f"c0_s{stage}_seed{s}.json") for s in protocol["report_seeds"]}
                for arm in protocol["arms"]:
                    group = []
                    for seed in protocol["report_seeds"]:
                        value = read(folder / f"{arm}_s{stage}_seed{seed}.json")
                        base = baseline[seed]
                        assert value["start_state_sha256"] == base["start_state_sha256"]
                        assert value["steps"] == base["steps"] and value["rows"] == base["rows"]
                        assert value["stage"] == stage and value["seed"] == seed and value["arm"] == arm
                        metrics, bm = value["metrics"]["all"], base["metrics"]["all"]
                        item = {"dataset": dataset, "order": order, "stage": stage, "arm": arm, "seed": seed,
                                "f1": metrics["cumulative_f1"], "old_f1": metrics["old_f1"], "new_f1": metrics["new_f1"],
                                "seen_f1": value["metrics"]["seen"]["cumulative_f1"],
                                "delta_f1": metrics["cumulative_f1"] - bm["cumulative_f1"],
                                "delta_old": metrics["old_f1"] - bm["old_f1"] if stage else None,
                                "delta_new": metrics["new_f1"] - bm["new_f1"],
                                "rows": value["rows"], "steps": value["steps"], "seconds": value["seconds"],
                                "executed": value["executed_here"], "source_sha256": digest(folder / f"{arm}_s{stage}_seed{seed}.json")}
                        rows.append(item)
                        group.append(item)
                    ci = interval([v["delta_f1"] for v in group])
                    cells.append({"dataset": dataset, "order": order, "stage": stage, "arm": arm,
                                  **{k: st.mean(v[k] for v in group) if group[0][k] is not None else None
                                     for k in ("f1", "old_f1", "new_f1", "seen_f1", "delta_f1", "delta_old", "delta_new", "seconds")},
                                  "f1_sd": st.stdev(v["f1"] for v in group), "paired_ci_low": ci[0], "paired_ci_high": ci[1]})
                c0_seconds = read(folder / f"c0_s{stage}_construction.json")["seconds"]
                for arm in protocol["arms"]:
                    extra = 0.
                    if stage and joint and arm != "c0":
                        prefix = arm if arm in ("recursive", "maintenance", "capacity") else "shared"
                        build = read(folder / f"{prefix}_s{stage}_construction.json")
                        extra += build["seconds"]
                        if arm == "maintenance":
                            extra += read(folder / f"maintenance_s{stage}_context_audit.json")["seconds"]
                            extra += read(folder / f"maintenance_s{stage}_optimization.json")["seconds"]
                        assignment = build["assignment"]
                        mechanisms.append({"dataset": dataset, "order": order, "stage": stage, "arm": arm,
                                           "initial_loss": assignment["curve"][0]["loss"],
                                           "final_loss": assignment["curve"][-1]["loss"],
                                           "min_effective_sources": assignment["min_effective_sources"],
                                           "mean_new_x_relative_displacement": build["materialization"]["new_x_relative_displacement_mean"],
                                           "rows_transferred": build["capacity"]["rows_transferred"] if build["capacity"] else 0,
                                           "text_tokens": build["materialization"]["text"]["token_total"]})
                    elif stage and not joint and arm != "c0":
                        opt_arm = "relation_old" if arm == "random_old" else arm
                        opt = read(folder / f"{opt_arm}_s{stage}_optimization.json")
                        extra += read(folder / f"s{stage}_support_diagnostics.json")["seconds"] + opt["seconds"]
                        mechanisms.append({"dataset": dataset, "order": order, "stage": stage, "arm": arm,
                                           "rows_changed": len(opt["invariants"]["changed_rows"]),
                                           "scale": opt["feasibility"][-1]["scale"],
                                           "max_relative_displacement": opt["invariants"]["max_relative_displacement"],
                                           "new_train_gain": opt["initial"]["train"]["new"] - opt["final"]["train"]["new"],
                                           "new_validation_gain": opt["initial"]["validation"]["new"] - opt["final"]["validation"]["new"],
                                           "final_old_train_kl": opt["final"]["train"]["old_kl"]})
                    costs.append({"dataset": dataset, "order": order, "stage": stage, "arm": arm,
                                  "c0_seconds": c0_seconds, "extra_seconds": extra,
                                  "build_components_seconds": c0_seconds + extra,
                                  "student_seconds": next(v["seconds"] for v in cells if
                                                          (v["dataset"], v["order"], v["stage"], v["arm"]) == (dataset, order, stage, arm))})
    expected = len(protocol["datasets"]) * len(protocol["orders"]) * 3 * len(protocol["arms"]) * len(protocol["report_seeds"])
    assert len(rows) == expected
    methods = []
    for arm in protocol["arms"]:
        final = [v for v in cells if v["arm"] == arm and v["stage"] == 2]
        methods.append({"arm": arm, "final_mean_f1": st.mean(v["f1"] for v in final),
                        "final_delta_pp": st.mean(v["delta_f1"] for v in final),
                        "final_cell_descriptive_ci": interval([v["delta_f1"] for v in final]),
                        "stage1_delta_pp": st.mean(v["delta_f1"] for v in cells if v["arm"] == arm and v["stage"] == 1),
                        "stage_mean_delta_pp": st.mean(v["delta_f1"] for v in cells if v["arm"] == arm),
                        "final_old_delta_pp": st.mean(v["delta_old"] for v in final),
                        "final_new_delta_pp": st.mean(v["delta_new"] for v in final),
                        "worst_final_delta_pp": min(v["delta_f1"] for v in final),
                        "worst_final_old_delta_pp": min(v["delta_old"] for v in final),
                        "cells_ge_minus1": sum(v["delta_f1"] >= -1 for v in final),
                        "mean_incremental_build_seconds": st.mean(v["build_components_seconds"] for v in costs if v["arm"] == arm and v["stage"] > 0),
                        "mean_student_seconds": st.mean(v["seconds"] for v in cells if v["arm"] == arm)})
    contrasts = []
    definitions = ({"interaction": {"joint": 1, "content": -1, "topology": -1, "c0": 1},
                    "recursion_vs_shared": {"recursive": 1, "joint": -1},
                    "maintenance_vs_recursive": {"maintenance": 1, "recursive": -1},
                    "capacity_vs_recursive": {"capacity": 1, "recursive": -1}} if joint else
                   {"relation_vs_random": {"relation_old": 1, "random_old": -1},
                    "relation_vs_shuffled": {"relation_old": 1, "shuffled_old": -1}})
    indexed = {(r["dataset"], r["order"], r["stage"], r["seed"], r["arm"]): r for r in rows}
    for name, weights in definitions.items():
        for dataset in protocol["datasets"]:
            for order in protocol["orders"]:
                for stage in (1, 2):
                    diff = [sum(weight * indexed[(dataset, order, stage, seed, arm)]["f1"] for arm, weight in weights.items())
                            for seed in protocol["report_seeds"]]
                    ci = interval(diff)
                    contrasts.append({"contrast": name, "dataset": dataset, "order": order, "stage": stage,
                                      "mean_pp": st.mean(diff), "paired_ci_low": ci[0], "paired_ci_high": ci[1]})
    gates = None
    if not joint:
        by = {v["arm"]: v for v in methods}
        main, random, limits = by["relation_old"], by["random_old"], protocol["progression"]
        gates = {"mean_final": main["final_delta_pp"] >= limits["mean_final_min_pp"],
                 "cells_ge_minus1": main["cells_ge_minus1"] >= limits["cells_ge_minus1_min"],
                 "worst_final": main["worst_final_delta_pp"] >= limits["worst_final_min_pp"],
                 "worst_old": main["worst_final_old_delta_pp"] >= limits["worst_old_min_pp"],
                 "mean_new": main["final_new_delta_pp"] >= limits["mean_new_min_pp"],
                 "mean_vs_random": main["final_mean_f1"] - random["final_mean_f1"] >= limits["mean_vs_random_min_pp"]}
    result = {"protocol_sha256": digest(root / "protocol.json"), "fit_slots": len(rows),
              "executed_fits": sum(r["executed"] for r in rows), "methods": methods, "audits": audits,
              "progression_checks": gates, "all_progression_checks_passed": all(gates.values()) if gates else None,
              "contrasts": contrasts, "student_actual_seconds": sum(r["seconds"] for r in rows if r["executed"]),
              "ci_scope": "6 paired seeds per fixed cell; 4-cell interval descriptive, not IID population inference",
              "cost_scope": "recorded construction components; excludes common cold encoder/cache loading; factorial materialization shared; random cost includes generating matched revision"}
    for name, records in (("per_seed", rows), ("cells", cells), ("costs", costs), ("mechanisms", mechanisms), ("contrasts", contrasts)):
        csv_file(root / f"{name}.csv", records)
    (root / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.root)
    print(json.dumps({k: result[k] for k in ("fit_slots", "executed_fits", "methods", "progression_checks")}, indent=2))
