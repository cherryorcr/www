"""Recompute v5 results from saved predictions and apply preregistered gates."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t
import torch

import retention_constrained_append as V
import run_v5_retention as R

C, M = R.C, R.M
METRICS = ("cumulative_f1", "old_f1", "new_f1", "accuracy")


def normalized(obj):
    return json.loads(json.dumps(obj))


def stats(values):
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()), "std": float(values.std(ddof=1)) if len(values) > 1 else 0.,
            "n": len(values)}


def paired_stats(values):
    result = stats(values)
    sem = result["std"] / np.sqrt(result["n"])
    result["ci95_low"] = result["mean"] - float(student_t.ppf(.975, result["n"] - 1)) * sem
    result["ci95_high"] = result["mean"] + float(student_t.ppf(.975, result["n"] - 1)) * sem
    result["simultaneous_lower"] = result["mean"] + float(student_t.ppf(.05 / 12, result["n"] - 1)) * sem
    return result


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def compare_construction_checks(actual, expected):
    # Independent CPU backends may differ in the last bits of double norm
    # reductions. Serialized tensors, hashes and discrete checks stay exact.
    discrete = lambda value: {k: v for k, v in value.items() if k != "drift"}
    require(normalized(discrete(actual)) == discrete(expected), "construction hash or invariant changed")
    require(set(actual["drift"]) == set(expected["drift"]), "drift fields changed")
    errors = []
    for key in actual["drift"]:
        np.testing.assert_allclose(actual["drift"][key], expected["drift"][key], rtol=1e-10, atol=1e-12,
                                   err_msg=f"recomputed {key} drift differs")
        errors.append(abs(actual["drift"][key] - expected["drift"][key]))
    return max(errors)


def audit_order(root, dataset, order, registration, verify_predictions=True):
    out = root / dataset / order
    result = json.loads((out / "results.json").read_text())
    require(result["registration_sha256"] == C.sha(root / dataset / "registration.json"), "registration hash mismatch")
    groups = registration["orders"][order]
    seeds = registration["seeds"][order]
    arms = result["arms"]
    expected = {(arm, stage, seed) for arm in arms for stage in range(len(groups)) for seed in seeds}
    actual = {(r["arm"], r["stage"], r["seed"]) for r in result["fits"]}
    require(expected == actual and len(actual) == len(result["fits"]), "missing or duplicate fits")
    blocks = [torch.load(out / f"block_s{s}.pt", map_location="cpu", weights_only=False) for s in range(len(groups))]
    memories, audit_count, prediction_checks = {}, 0, 0
    drift_recompute_errors = []
    for arm in arms:
        old = None
        for stage, block in enumerate(blocks):
            mem_path = out / f"{arm}_s{stage}_memory.pt"
            memory = torch.load(mem_path, map_location="cpu", weights_only=False)
            construction = json.loads((out / f"{arm}_s{stage}_construction.json").read_text())
            require(C.sha(mem_path) == construction["file_sha256"], "memory file mismatch")
            require(C.tensor_hash(memory) == construction["tensor_sha256"], "memory tensor mismatch")
            checks = V.audit_append(old, block, memory, arm)
            drift_recompute_errors.append(compare_construction_checks(checks, construction["checks"]))
            audit_count += 1
            memories[arm, stage] = memory
            old = memory
    stage0, predictions = {}, {}
    for row in result["fits"]:
        key = f"{row['arm']}_s{row['stage']}_seed{row['seed']}"
        saved_row = json.loads((out / f"{key}.json").read_text())
        require(saved_row == row, "aggregate fit row mismatch")
        memory = memories[row["arm"], row["stage"]]
        require(row["support_sha256"] == C.tensor_hash(memory), "fit memory hash mismatch")
        require(row["training_sha256"] == V.training_hash(memory), "training support mismatch")
        require(row["steps"] == R.P.student_steps(len(memory["x"])), "student step mismatch")
        path = out / f"{key}.pt"
        require(C.sha(path) == row["checkpoint_sha256"], "checkpoint file changed")
        saved = torch.load(path, map_location="cpu", weights_only=False)
        require(C.tensor_hash(saved["state"]) == row["end_state_sha256"], "state hash mismatch")
        pred = saved["prediction"]
        predictions[row["arm"], row["stage"], row["seed"]] = pred
        require(not pred["test_evaluated"] and pred["split"] == "report_dev", "wrong prediction split")
        require(bool(((pred["original_ids"] % 10) == 3).all()), "prediction contains non-report IDs")
        require(normalized(pred["stage_classes"]) == normalized(groups), "prediction class order mismatch")
        if verify_predictions:
            metrics = M.score_logits(pred["logits"], pred["target"].numpy(), groups, row["stage"])
            legacy = R.legacy_from_predictions(pred["target"], pred["logits"].argmax(1), groups, row["stage"])
            require(normalized(metrics) == row["metrics"], "pooled metrics mismatch")
            require(normalized(legacy) == row["legacy_task_slice"], "task-slice metrics mismatch")
            prediction_checks += 1
        if row["stage"] == 0:
            stage0.setdefault(row["seed"], []).append(row)
    numerical = []
    for seed, rows in stage0.items():
        hashes = {r["training_sha256"] for r in rows}
        initial = {r["start_state_sha256"] for r in rows}
        require(len(hashes) == 1 and len(initial) == 1, "stage-zero input/init mismatch")
        scores = [r["metrics"]["all"]["cumulative_f1"] for r in rows]
        numerical.append({"dataset": dataset, "order": order, "seed": seed,
            "identical_input": True, "identical_end_state": len({r["end_state_sha256"] for r in rows}) == 1,
            "f1_range": max(scores) - min(scores)})
    transitions = []
    for arm in arms:
        for stage in range(1, len(groups)):
            old_classes = torch.tensor([c for group in groups[:stage] for c in group])
            for seed in seeds:
                before, after = predictions[arm, stage - 1, seed], predictions[arm, stage, seed]
                require(torch.equal(before["original_ids"], after["original_ids"]), "prediction ID order changed")
                require(torch.equal(before["target"], after["target"]), "prediction targets changed")
                mask = torch.isin(after["target"], old_classes)
                target = after["target"][mask]
                old_p = before["logits"][mask].argmax(1)
                new_p = after["logits"][mask].argmax(1)
                was_correct, now_correct = old_p == target, new_p == target
                percent = lambda value: float(value.float().mean() * 100)
                transitions.append({"dataset": dataset, "order": order, "arm": arm,
                    "stage": stage, "seed": seed, "old_nodes": int(mask.sum()),
                    "old_prediction_flip_percent": percent(old_p != new_p),
                    "correct_to_wrong_percent": percent(was_correct & ~now_correct),
                    "wrong_to_correct_percent": percent(~was_correct & now_correct),
                    "previous_old_accuracy": percent(was_correct),
                    "current_old_accuracy": percent(now_correct)})
    return result, {"construction_checks": audit_count, "prediction_checks": prediction_checks,
                    "stage_zero": numerical, "posthoc_prediction_transitions": transitions,
                    "max_drift_recompute_absolute_error": max(drift_recompute_errors)}


def metric_arrays(fits, arm, seeds, stages):
    by = {(r["stage"], r["seed"]): r for r in fits if r["arm"] == arm}
    final = [by[stages - 1, seed] for seed in seeds]
    return {"final": np.array([r["metrics"]["all"]["cumulative_f1"] for r in final]),
            "stage_average": np.array([np.mean([by[s, seed]["metrics"]["all"]["cumulative_f1"] for s in range(stages)]) for seed in seeds]),
            "old": np.array([r["metrics"]["all"]["old_f1"] for r in final]),
            "new": np.array([r["metrics"]["all"]["new_f1"] for r in final]),
            "seen_stage_average": np.array([np.mean([by[s, seed]["metrics"]["seen"]["cumulative_f1"] for s in range(stages)]) for seed in seeds])}


def evaluate_gates(paired, costs, audit_ok):
    decisions = {}
    for arm in V.ARMS[1:]:
        rows = [r for r in paired if r["arm"] == arm]
        final = [r for r in rows if r["metric"] == "final"]
        stage = [r for r in rows if r["metric"] == "stage_average"]
        old = [r for r in rows if r["metric"] == "old"]
        new = [r for r in rows if r["metric"] == "new"]
        require(len(final) == 4, "promotion requires all four cells")
        gains = [r for r in final if r["mean"] >= 1.5]
        gates = {
            "final_mean_no_material_loss": all(r["mean"] >= -1 for r in final),
            "two_gains_including_confirmation": len(gains) >= 2 and any(r["order"] == "matched_unseen" for r in gains),
            "stage_average_no_material_loss": all(r["mean"] >= -1 for r in stage),
            "pooled_old_retention": all(r["mean"] >= -2 for r in old),
            "pooled_new_positive": float(np.mean([r["mean"] for r in new])) > 0 and sum(r["mean"] > 0 for r in new) >= 2,
            "construction_at_most_twice": all(r["construction_ratio_to_c0"] <= 2 for r in costs if r["arm"] == arm),
            "simultaneous_noninferiority_evidence": all(r["simultaneous_lower"] > -2 for r in final),
            "artifact_and_numerical_audit": audit_ok,
        }
        decisions[arm] = {"passed": all(gates.values()), "gates": gates,
                          "failed_gates": [k for k, v in gates.items() if not v],
                          "mean_final_gain": float(np.mean([r["mean"] for r in final])),
                          "mean_new_gain": float(np.mean([r["mean"] for r in new]))}
    passed = [arm for arm, decision in decisions.items() if decision["passed"]]
    selected = min(passed, key=lambda a: (-decisions[a]["mean_final_gain"],
               sum(r["construction_seconds"] for r in costs if r["arm"] == a), a)) if passed else None
    return {"candidates": decisions, "large_candidate": selected,
            "large_status": "authorized_by_small_gate" if selected else "not_started_small_gate_failed",
            "promotion": "pending_large_confirmation" if selected else "reject_v5_retain_c0",
            "scope": "two small datasets, two orders, six seeds per cell; report-dev reused historically"}


def summarize(root, output):
    output.mkdir(parents=True, exist_ok=True)
    cells, paired, costs, stages_out, mechanisms, interactions, numerical = [], [], [], [], [], [], []
    audits, transitions = [], []
    for dataset in ("cora", "citeseer"):
        registration = json.loads((root / dataset / "registration.json").read_text())
        freeze = json.loads((root / dataset / "freeze.json").read_text())
        require(registration["freeze_sha256"] == C.sha(root / dataset / "freeze.json"), "freeze changed")
        require(freeze["config"] == V.RetentionConfig().dict(), "constraint config changed")
        require(freeze["code_sha256"] == R.source_hashes(), "running audit code differs from training snapshot")
        completion = json.loads((root / dataset / "completion.json").read_text())
        require(completion["status"] == "complete" and completion["fits"] == 144, "dataset incomplete")
        prep = json.loads((root / dataset / "preprocessing.json").read_text())
        for order, groups in registration["orders"].items():
            result, audit = audit_order(root, dataset, order, registration)
            audits.append({"dataset": dataset, "order": order, **audit})
            transitions.extend(audit["posthoc_prediction_transitions"])
            numerical.extend(audit["stage_zero"])
            seeds = registration["seeds"][order]
            arrays = {arm: metric_arrays(result["fits"], arm, seeds, len(groups)) for arm in V.ARMS}
            common_build = sum(a["build_seconds"] for a in result["shared_blocks"])
            construction = {arm: common_build + sum(a["append_seconds"] for a in result["construction"][arm]) for arm in V.ARMS}
            baseline_training = sum(r["seconds"] for r in result["fits"] if r["arm"] == "c0")
            for arm in V.ARMS:
                final_memory = result["construction"][arm][-1]
                final_artifact = torch.load(root / dataset / order / f"{arm}_s{len(groups) - 1}_memory.pt",
                                            map_location="cpu", weights_only=False)
                row = {"dataset": dataset, "order": order, "arm": arm,
                       "final_rows": final_memory["payload"]["rows"]}
                for metric, values in arrays[arm].items():
                    row[metric] = stats(values)["mean"]
                    row[metric + "_std"] = stats(values)["std"]
                cells.append(row)
                fits = [r for r in result["fits"] if r["arm"] == arm]
                train = sum(r["seconds"] for r in fits)
                cost = {"dataset": dataset, "order": order, "arm": arm,
                        "source_encoding_seconds_shared_once_per_dataset": prep["source_encoding_seconds"],
                        "shared_block_seconds_charged_per_arm": common_build,
                        "append_seconds": construction[arm] - common_build,
                        "construction_seconds": construction[arm],
                        "construction_ratio_to_c0": construction[arm] / construction["c0"],
                        "training_seconds_all_18_fits": train,
                        "training_ratio_to_c0": train / baseline_training,
                        "evaluation_io_seconds": sum(r["nontraining_seconds"] for r in fits),
                        "construction_audit_io_seconds": sum(a["audit_and_memory_io_seconds"] for a in result["construction"][arm]),
                        "max_training_peak_allocated_mb": max(r["peak_allocated_mb"] for r in fits),
                        "final_payload_bytes": final_memory["payload"]["logical_payload_bytes"],
                        "final_serialized_bytes": final_memory["serialized_bytes"],
                        "final_edge_entries": final_memory["payload"]["edge_entries"],
                        "final_raw_edge_entries": final_artifact.get("raw_edge_index", final_artifact["edge_index"]).shape[1],
                        "final_normalized_support_entries": V.support(final_artifact)._nnz(),
                        "selection_runs": 0}
                costs.append(cost)
                if arm != "c0":
                    for metric in arrays[arm]:
                        paired.append({"dataset": dataset, "order": order, "arm": arm, "metric": metric,
                                       **paired_stats(arrays[arm][metric] - arrays["c0"][metric])})
                for stage in range(len(groups)):
                    subset = [r for r in fits if r["stage"] == stage]
                    summary = {"dataset": dataset, "order": order, "arm": arm, "stage": stage,
                               "rows": subset[0]["rows"], "steps": subset[0]["steps"]}
                    for scope in ("all", "seen"):
                        for metric in METRICS:
                            values = [r["metrics"][scope][metric] for r in subset if r["metrics"][scope][metric] is not None]
                            summary[scope + "_" + metric] = float(np.mean(values)) if values else None
                    summary["task_slice_old"] = float(np.mean([r["legacy_task_slice"]["old_mean_f1"] for r in subset])) if stage else None
                    summary["task_slice_new"] = float(np.mean([r["legacy_task_slice"]["current_f1"] for r in subset]))
                    stages_out.append(summary)
                    build = result["construction"][arm][stage]
                    top = build["details"]["topology"]
                    refresh = build["details"]["refresh"]
                    mechanisms.append({"dataset": dataset, "order": order, "arm": arm, "stage": stage,
                        **build["checks"]["drift"],
                        "cross_mix_lambda": top["lambda"] if top else None,
                        "refresh_scale": refresh["scale"] if refresh else None,
                        "max_anchor_relative": refresh["max_anchor_relative"] if refresh else None,
                        "refreshed_rows": len(refresh["refreshed_rows"]) if refresh else 0,
                        "trained_embedding_drift_mean": float(np.mean([r["trained_old_embedding_relative_drift"] for r in subset]))})
            for metric in arrays["full"]:
                interaction = arrays["full"][metric] - arrays["core"][metric] - arrays["topology"][metric] + arrays["c0"][metric]
                interactions.append({"dataset": dataset, "order": order, "metric": metric, **paired_stats(interaction)})
    audit_ok = all(r["f1_range"] == 0 for r in numerical)
    audit = {"status": "passed" if audit_ok else "numerical_control_failed",
             "construction_checks": sum(a["construction_checks"] for a in audits),
             "prediction_checks": sum(a["prediction_checks"] for a in audits),
             "max_drift_recompute_absolute_error": max(a["max_drift_recompute_absolute_error"] for a in audits),
             "drift_recompute_tolerance": {"absolute": 1e-12, "relative": 1e-10},
             "stage_zero_repeats": numerical, "all_stage_zero_states_bitwise_equal": all(r["identical_end_state"] for r in numerical),
             "code_sha256": R.source_hashes(), "auditor_sha256": C.sha(__file__),
             "audited_cells": audits}
    decision = evaluate_gates(paired, costs, audit_ok)
    for filename, rows in (("cell_summary.csv", cells), ("paired_differences.csv", paired),
                           ("costs.csv", costs), ("stage_summary.csv", stages_out),
                           ("mechanism.csv", mechanisms), ("interaction.csv", interactions),
                           ("posthoc_prediction_transitions.csv", transitions)):
        write_csv(output / filename, rows)
    C.save_json(output / "audit.json", audit)
    C.save_json(output / "decision.json", decision)
    C.save_json(output / "summary.json", {"cells": cells, "paired": paired, "costs": costs,
        "stages": stages_out, "mechanism": mechanisms, "interaction": interactions,
        "decision": decision, "audit": audit, "posthoc_prediction_transitions": transitions})
    lines = ["# v5 小图预注册实验汇总", "", "主指标：all-prompt、pooled seen support 的节点分类 Macro-F1；mean ± seed sample std。", "",
             "|数据集|顺序|组别|最终 F1|阶段均值|pooled old|pooled new|", "|---|---|---|---:|---:|---:|---:|"]
    for row in cells:
        lines.append(f"|{row['dataset']}|{row['order']}|{row['arm']}|{row['final']:.2f} ± {row['final_std']:.2f}|{row['stage_average']:.2f}|{row['old']:.2f}|{row['new']:.2f}|")
    lines += ["", "## 晋升结论", "", f"`{decision['promotion']}`；大图状态：`{decision['large_status']}`。", ""]
    for arm, value in decision["candidates"].items():
        lines.append(f"- {arm}: {'通过' if value['passed'] else '未通过'}；未通过条件：{', '.join(value['failed_gates']) or '无'}。")
    lines += ["", f"审计：{audit['construction_checks']} 张 memory、{audit['prediction_checks']} 份保存预测；{audit['status']}。", "",
              "配对区间、阶段明细、成本、漂移和 interaction 见同目录 CSV/JSON。共享编码仅按每个数据集一次计费，单 arm/单顺序独立使用时仍需支付该成本。", ""]
    (output / "V5_SMALL_RESULTS_ZH.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"audit": audit["status"], "fits": audit["prediction_checks"], "decision": decision}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    torch.set_num_threads(2)
    summarize(Path(args.root), Path(args.output))


if __name__ == "__main__":
    main()
