"""Recompute R6 confirmation and diagnostic contrasts from sealed fit records."""
import argparse
import itertools
import json
from pathlib import Path
import statistics as st

from summarize_tgc_r4_relational import read, digest, interval, csv_file


def arms_at(spec, stage):
    return spec.get("stage_arms", {}).get(str(stage), spec["arms"])


def mean_nullable(values):
    values = list(values)
    return st.mean(values) if all(v is not None for v in values) else None


def contrast(rows, spec, name, weights, stages=(2,)):
    indexed = {(v["dataset"], v["order"], v["stage"], v["seed"], v["arm"]): v for v in rows}
    result = []
    for dataset, order, stage in itertools.product(spec["datasets"], spec["orders"], stages):
        item = {"contrast": name, "dataset": dataset, "order": order, "stage": stage}
        for metric in ("f1", "old_f1", "new_f1"):
            diffs = [sum(w * indexed[dataset, order, stage, seed, a][metric]
                         for a, w in weights.items()) for seed in spec["report_seeds"]]
            ci = interval(diffs)
            item.update({metric + "_mean_pp": st.mean(diffs), metric + "_ci_low": ci[0], metric + "_ci_high": ci[1]})
        result.append(item)
    return result


def confirmation_cost(folder, arm, stage):
    audit = read(folder / f"{arm}_s{stage}_build_audit.json")
    c0_seconds = read(folder / f"c0_s{stage}_build_audit.json")["seconds"]
    seconds = audit["seconds"] + (c0_seconds if arm == "recursive" and stage else 0.)
    return {"recorded_build_seconds": seconds,
            "teacher_seconds": sum(t["seconds"] for t in audit.get("teachers", [])),
            "logical_bytes": audit["payload"]["logical_bytes"],
            "text_utf8_bytes": audit["payload"]["text_utf8_bytes"],
            "edge_entries": audit["payload"]["edge_entries"],
            "text": audit["text"]}


def summarize(root):
    spec, seal = read(root / "protocol.json"), read(root / "selection_seal.json")
    seal_sha = digest(root / "selection_seal.json")
    assert seal["protocol_sha256"] == digest(root / "protocol.json")
    assert read(root / "pipeline_status.json")["status"] == "complete"
    for path, sha in seal["decisions"].items():
        assert digest(root / path) == sha, path
    confirmation = "confirmation" in spec["name"]
    rows, cells, audits, costs, diagnostics = [], [], [], [], []
    for dataset, order in itertools.product(spec["datasets"], spec["orders"]):
        folder = root / dataset / order
        audit, manifest = read(folder / "audit.json"), read(folder / "manifest.json")
        assert audit["status"] == "passed" and audit["maximum_metric_error_pp"] < 1e-7
        audits.append({"dataset": dataset, "order": order, **audit})
        count, executed = 0, 0
        for stage in spec["evaluation_stages"]:
            base = {seed: read(folder / f"c0_s{stage}_seed{seed}.json") for seed in spec["report_seeds"]}
            for arm in arms_at(spec, stage):
                group = []
                for seed in spec["report_seeds"]:
                    path = folder / f"{arm}_s{stage}_seed{seed}.json"
                    value, control = read(path), base[seed]
                    assert (value["arm"], value["stage"], value["seed"]) == (arm, stage, seed)
                    assert value["start_state_sha256"] == control["start_state_sha256"]
                    assert value["steps"] == control["steps"] and value["rows"] == manifest["budgets"][stage]
                    assert value["selection_seal_sha256"] == seal_sha
                    m, b = value["metrics"]["all"], control["metrics"]["all"]
                    item = {"dataset": dataset, "order": order, "stage": stage, "arm": arm, "seed": seed,
                            "f1": m["cumulative_f1"], "old_f1": m["old_f1"], "new_f1": m["new_f1"],
                            "seen_f1": value["metrics"]["seen"]["cumulative_f1"],
                            "delta_f1": m["cumulative_f1"] - b["cumulative_f1"],
                            "delta_old": m["old_f1"] - b["old_f1"] if stage else None,
                            "delta_new": m["new_f1"] - b["new_f1"],
                            "rows": value["rows"], "steps": value["steps"], "seconds": value["seconds"],
                            "peak_mb": value["peak_allocated_mb"], "executed": value["executed_here"],
                            "checkpoint_sha256": value["checkpoint_sha256"], "source_sha256": digest(path)}
                    rows.append(item)
                    group.append(item)
                    count += 1
                    executed += value["executed_here"]
                ci = interval([v["delta_f1"] for v in group])
                cell = {"dataset": dataset, "order": order, "stage": stage, "arm": arm,
                        **{k: mean_nullable(v[k] for v in group) for k in (
                            "f1", "old_f1", "new_f1", "seen_f1", "delta_f1", "delta_old", "delta_new", "seconds")},
                        "f1_sd": st.stdev(v["f1"] for v in group), "paired_ci_low": ci[0], "paired_ci_high": ci[1]}
                cells.append(cell)
                if confirmation:
                    costs.append({"dataset": dataset, "order": order, "stage": stage, "arm": arm,
                                  **confirmation_cost(folder, arm, stage), "mean_student_seconds": cell["seconds"]})
                elif (folder / f"{arm}_s{stage}_build_audit.json").exists():
                    details = read(folder / f"{arm}_s{stage}_build_audit.json")
                    assn, gap, ref = details.get("assignment", {}), details.get("gap", {}), details.get("reference", {})
                    curve = assn.get("curve", [])
                    diagnostics.append({"dataset": dataset, "order": order, "stage": stage, "arm": arm,
                        "f1": cell["f1"], "parent_bits": details.get("parent_bits"),
                        "initial_loss": curve[0]["loss"] if curve else None,
                        "final_loss": curve[-1]["loss"] if curve else None,
                        "recorded_seconds": details.get("seconds", details.get("variant_seconds")),
                        "reference_seconds": ref.get("seconds"),
                        "teacher_seconds": sum(t.get("seconds", 0.) for t in ref.get("teachers", [])),
                        "logical_bytes": details.get("payload", {}).get("logical_bytes"),
                        **{k: gap.get(k, {}).get("balanced") for k in (
                            "pooled_reconstruction", "materialized_reconstruction", "materialized_symmetric_message")},
                        "compact_teacher_vs_pooled_mse": gap.get("compact_teacher_vs_pooled_mse")})
        completed = read(folder / "completion.json")
        assert count == audit["fit_slots"] == completed["fit_slots"]
        assert executed == completed["executed_fits"]
    assert len(rows) == len(spec["datasets"]) * len(spec["orders"]) * len(spec["report_seeds"]) * sum(
        len(arms_at(spec, s)) for s in spec["evaluation_stages"])
    methods = []
    for arm in spec["arms"]:
        final = [c for c in cells if c["arm"] == arm and c["stage"] == 2]
        evaluated = [c for c in cells if c["arm"] == arm]
        entry = {"arm": arm, "final_mean_f1": st.mean(v["f1"] for v in final),
                 "final_delta_pp": st.mean(v["delta_f1"] for v in final),
                 "final_cell_descriptive_ci": interval([v["delta_f1"] for v in final]),
                 "evaluated_stages": sorted({v["stage"] for v in evaluated}),
                 "evaluated_stage_mean_delta_pp": st.mean(v["delta_f1"] for v in evaluated),
                 "final_old_delta_pp": st.mean(v["delta_old"] for v in final),
                 "final_new_delta_pp": st.mean(v["delta_new"] for v in final),
                 "worst_final_delta_pp": min(v["delta_f1"] for v in final),
                 "worst_final_old_delta_pp": min(v["delta_old"] for v in final),
                 "cells_ge_minus1": sum(v["delta_f1"] >= -1 for v in final),
                 "positive_final_cells": sum(v["delta_f1"] > 0 for v in final),
                 "mean_student_seconds": st.mean(v["seconds"] for v in evaluated)}
        if confirmation:
            arm_costs = [v for v in costs if v["arm"] == arm and v["stage"] > 0]
            entry.update(mean_incremental_build_seconds=st.mean(v["recorded_build_seconds"] for v in arm_costs),
                         mean_final_logical_bytes=st.mean(v["logical_bytes"] for v in arm_costs if v["stage"] == 2))
        methods.append(entry)
    contrasts = []
    if confirmation:
        for baseline in ("c0", "k_center", "kmeans_medoid"):
            contrasts += contrast(rows, spec, "recursive_vs_" + baseline, {"recursive": 1., baseline: -1.}, stages=(1, 2))
        limits = spec["progression"]
        main = next(v for v in methods if v["arm"] == "recursive")
        gates = {"mean_final": main["final_delta_pp"] >= limits["mean_final_min_pp"],
                 "stage_mean": main["evaluated_stage_mean_delta_pp"] >= limits["stage_mean_min_pp"],
                 "worst_final": main["worst_final_delta_pp"] >= limits["worst_final_min_pp"],
                 "worst_old": main["worst_final_old_delta_pp"] >= limits["worst_old_min_pp"],
                 "cells_ge_minus1": main["cells_ge_minus1"] >= limits["min_cells_ge_minus1"],
                 "mean_new": main["final_new_delta_pp"] >= limits["new_mean_min_pp"]}
    else:
        gates = None
        bits = list(itertools.product((0, 1), repeat=3))
        # Difference of factorial averages; interactions use differences of differences.
        for size in (1, 2, 3):
            for factors in itertools.combinations(range(3), size):
                weights = {"parent_" + "".join(map(str, b)):
                           (1. if sum(b[i] for i in factors) % 2 == size % 2 else -1.) / 2 ** (3 - size)
                           for b in bits}
                contrasts += contrast(rows, spec, "parent_factorial_" + "_".join(("x", "text", "edges")[i] for i in factors), weights)
        definitions = {
            "recursive_vs_shared_parent": {"parent_111": 1, "parent_000": -1},
            "output_effect_reference0": {"reference0_output1": 1, "parent_000": -1},
            "output_effect_reference1": {"parent_111": 1, "reference1_output0": -1},
            "reference_effect_output0": {"reference1_output0": 1, "parent_000": -1},
            "reference_effect_output1": {"parent_111": 1, "reference0_output1": -1},
            "reference_output_interaction": {"parent_111": 1, "reference0_output1": -1, "reference1_output0": -1, "parent_000": 1},
            "pooled_effect_binary": {"pooled_text": 1, "parent_111": -1},
            "pooled_effect_weighted": {"pooled_weighted": 1, "weighted_edges": -1},
            "weighted_effect_reencoded": {"weighted_edges": 1, "parent_111": -1},
            "weighted_effect_pooled": {"pooled_weighted": 1, "pooled_text": -1},
            "materialization_interaction": {"pooled_weighted": 1, "pooled_text": -1, "weighted_edges": -1, "parent_111": 1}}
        if "frozen_joint" in spec["arms"]:
            definitions.update(fresh_shared_vs_frozen={"parent_000": 1, "frozen_joint": -1},
                               fresh_recursive_vs_frozen={"parent_111": 1, "frozen_recursive": -1})
        for arm in ("geometry", "random_teacher", "fixed_assignment", "decoupled64", "decoupled128", "symmetric_fixed_parent", "symmetric_recursive"):
            definitions[arm + "_vs_recursive"] = {arm: 1, "parent_111": -1}
        for name, weights in definitions.items():
            contrasts += contrast(rows, spec, name, weights, stages=(1, 2) if name == "symmetric_recursive_vs_recursive" else (2,))
    contrast_summary = []
    for name, stage in sorted({(v["contrast"], v["stage"]) for v in contrasts}):
        group = [v for v in contrasts if (v["contrast"], v["stage"]) == (name, stage)]
        contrast_summary.append({"contrast": name, "stage": stage, "mean_pp": st.mean(v["f1_mean_pp"] for v in group),
                                 "cell_descriptive_ci": interval([v["f1_mean_pp"] for v in group]),
                                 "positive_cells": sum(v["f1_mean_pp"] > 0 for v in group),
                                 "old_mean_pp": st.mean(v["old_f1_mean_pp"] for v in group),
                                 "new_mean_pp": st.mean(v["new_f1_mean_pp"] for v in group)})
    correlations = []
    if not confirmation:
        for dataset, order in itertools.product(spec["datasets"], spec["orders"]):
            for metric in ("materialized_reconstruction", "materialized_symmetric_message", "compact_teacher_vs_pooled_mse"):
                group = [v for v in diagnostics if v["dataset"] == dataset and v["order"] == order and v["stage"] == 2
                         and v["arm"] in ("parent_111", "pooled_text", "weighted_edges", "pooled_weighted")]
                correlations.append({"dataset": dataset, "order": order, "metric": metric, "n": len(group),
                                     "pearson": st.correlation([v[metric] for v in group], [v["f1"] for v in group]),
                                     "scope": "descriptive four representations at fixed P/teacher; no causal or significance claim"})
    dataset_methods = []
    for dataset, arm in itertools.product(spec["datasets"], spec["arms"]):
        group = [v for v in cells if v["dataset"] == dataset and v["arm"] == arm and v["stage"] == 2]
        dataset_methods.append({"dataset": dataset, "arm": arm,
                                **{k: st.mean(v[k] for v in group) for k in (
                                    "f1", "delta_f1", "old_f1", "new_f1", "delta_old", "delta_new")}})
    student_repeat = []
    if not confirmation and "frozen_joint" in spec["arms"]:
        historical_root = root.parent / Path(spec["source_root"]).name
        assert historical_root.exists(), "R5 sealed fit records required to quantify same-memory numerical repeats"
        old_seal = read(historical_root / "selection_seal.json")
        assert old_seal["protocol_sha256"] == digest(historical_root / "protocol.json")
        for dataset, order, arm in itertools.product(spec["datasets"], spec["orders"], ("c0", "joint", "recursive")):
            assert read(historical_root / dataset / order / "audit.json")["status"] == "passed"
            fresh_arm = "c0" if arm == "c0" else "frozen_" + arm
            diffs = []
            for seed in spec["report_seeds"]:
                old = read(historical_root / dataset / order / f"{arm}_s2_seed{seed}.json")
                new = read(root / dataset / order / f"{fresh_arm}_s2_seed{seed}.json")
                for key in ("support_memory_sha256", "start_state_sha256", "steps", "rows", "effective_batch"):
                    assert old[key] == new[key], (dataset, order, arm, seed, key)
                diffs.append(new["metrics"]["all"]["cumulative_f1"]-old["metrics"]["all"]["cumulative_f1"])
            ci = interval(diffs)
            student_repeat.append({"dataset": dataset, "order": order, "arm": arm,
                                   "mean_pp": st.mean(diffs), "mean_absolute_seed_pp": st.mean(abs(v) for v in diffs),
                                   "max_absolute_seed_pp": max(abs(v) for v in diffs), "paired_ci": ci})
    result = {"protocol_sha256": digest(root / "protocol.json"), "selection_seal_sha256": digest(root / "selection_seal.json"),
              "fit_slots": len(rows), "executed_fits": sum(v["executed"] for v in rows),
              "independent_reload_scores": sum(v["unique_reload_scores"] for v in audits),
              "methods": methods, "dataset_methods": dataset_methods, "audits": audits, "contrasts": contrast_summary,
              "progression_checks": gates, "all_progression_checks_passed": all(gates.values()) if gates is not None else None,
              "student_actual_seconds": sum(v["seconds"] for v in rows if v["executed"]),
              "same_memory_student_repeat_vs_r5": student_repeat,
              "correlations": correlations,
              "ci_scope": "paired t across six student seeds per fixed condition; 4/6-cell intervals descriptive; orders are not new datasets; unadjusted multiple diagnostics",
              "cost_scope": "recorded components exclude common cold encoder/cache load; reused teacher/base costs require explicit addition; diagnostic time is not end-to-end algorithm cost"}
    for name, values in (("per_seed", rows), ("cells", cells), ("contrasts", contrasts), ("costs", costs), ("diagnostics", diagnostics)):
        if values:
            csv_file(root / f"{name}.csv", values)
    (root / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    result = summarize(args.root)
    print(json.dumps({k: result[k] for k in ("fit_slots", "executed_fits", "independent_reload_scores", "methods", "progression_checks", "contrasts")}, indent=2))
