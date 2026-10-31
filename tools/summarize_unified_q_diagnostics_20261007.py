"""Separate Student attribution, constructor diagnostics and order confirmation."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev


def dump_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def std(values):
    return stdev(values) if len(values) > 1 else 0.0


def collect(root):
    cells, stages, ab, failures, numerical = [], [], [], [], []
    for path in sorted(root.glob("*/student_ab.json")):
        value = json.loads(path.read_text())
        protocol = value["protocol"]["source_protocol"]
        groups = {}
        for row in value["rows"]:
            groups.setdefault((row["arm"], row["student"]), []).append(row)
        for (arm, student), rows in groups.items():
            final_stage = len(protocol["groups"]) - 1
            final = [r for r in rows if r["stage"] == final_stage]
            if len(final) != 3 or len(rows) != 3 * len(protocol["groups"]):
                continue
            ab.append({"dataset": protocol["dataset"], "budget_mode": protocol["budget_mode"],
                "arm": arm, "student": student, "final_f1": mean(r["final_f1"] for r in final),
                "final_f1_std": std([r["final_f1"] for r in final]),
                "accuracy": mean(r["accuracy"] for r in final),
                "stage_avg_f1": mean(r["final_f1"] for r in rows),
                "stage_avg_accuracy": mean(r["accuracy"] for r in rows),
                "old_f1": mean(r["old_f1"] for r in final),
                "new_f1": mean(r["new_f1"] for r in final),
                "max_archived_f1_difference": max(abs(r.get("archived_f1_difference", 0)) for r in rows)})
    for path in sorted(root.glob("*/results.json")):
        value = json.loads(path.read_text())
        protocol = value["protocol"]
        for arm in sorted({r["arm"] for r in value["rows"]}):
            rows = sorted([r for r in value["rows"] if r["arm"] == arm], key=lambda r: r["stage"])
            for row in rows:
                a = row["audit"]
                score = row["score"]
                stages.append({"run": path.parent.name, "dataset": protocol["dataset"],
                    "run_status": value["status"],
                    "arm": arm, "stage": row["stage"], "memory_rows": row["memory_rows"],
                    **{k: score[k] for k in ("final_f1", "accuracy", "old_f1", "new_f1")}})
                if "q_row_sum_max_error" in a:
                    numerical.append({"run": path.parent.name, "dataset": protocol["dataset"],
                        "run_status": value["status"],
                        "arm": arm, "stage": row["stage"],
                        "row_error_final": a["q_row_sum_max_error"],
                        "column_relative_error_final": a["q_column_relative_error"],
                        "max_row_error": a["sinkhorn_max_row_error"],
                        "max_column_relative_error": a["sinkhorn_max_column_relative_error"],
                        "max_iterations": a["sinkhorn_max_iterations"],
                        "tolerance": a["config"]["sinkhorn_tolerance"],
                        "q_updates": a["q_update_count"], "r_updates": a["residual_update_count"],
                        "old_updates": a["old_update_count"],
                        "old_x_relative_change": a["old_x_relative_change_max"],
                        "accepted_swaps": a["topology"].get("accepted_swaps", 0)})
            if value["status"] != "complete" or len(rows) != len(protocol["groups"]):
                continue
            final = rows[-1]["score"]
            cells.append({"run": path.parent.name, "dataset": protocol["dataset"],
                "arm": arm, "budget_mode": protocol["budget_mode"],
                "order_seed": protocol["order_seed"], "construction_seed": protocol["construction_seed"],
                "protocol_family": "old-x-adaptation" if arm.startswith("uq_old") else "frozen-prefix",
                "final_f1": final["final_f1"], "final_f1_std": final["final_f1_std"],
                "accuracy": final["accuracy"], "accuracy_std": final["accuracy_std"],
                "old_f1": final["old_f1"], "new_f1": final["new_f1"],
                "stage_avg_f1": mean(r["score"]["final_f1"] for r in rows),
                "stage_avg_accuracy": mean(r["score"]["accuracy"] for r in rows),
                "worst_stage": min(r["score"]["final_f1"] for r in rows),
                "memory_rows": rows[-1]["memory_rows"],
                "construction_seconds": sum(r["construction_seconds"] for r in rows),
                "student_seconds": sum(r["student_seconds"] for r in rows),
                "peak_mb": max(r["construction_peak_mb"] or 0 for r in rows)})
    resolved_tags = set()
    for path in root.glob("memretry_*/results.json"):
        try:
            if json.loads(path.read_text()).get("status") == "complete":
                resolved_tags.add(path.parent.name.replace("memretry_", "", 1))
        except (OSError, json.JSONDecodeError):
            pass
    all_failures = []
    for path in root.glob("*_exit_codes.json"):
        all_failures.extend(r for r in json.loads(path.read_text()) if r["exit_code"] != 0)
    failures = [r for r in all_failures if r.get("tag") not in resolved_tags]
    resolved_failures = [r for r in all_failures if r.get("tag") in resolved_tags]
    return cells, stages, ab, failures, numerical, resolved_failures


def paired(cells):
    result = []
    for dataset in ("wikics", "bookhis"):
        subset = [r for r in cells if r["dataset"] == dataset and r["arm"] in ("cat_cctp", "uq_v3")]
        index = {(r["order_seed"], r["construction_seed"], r["arm"]): r for r in subset}
        identities = sorted({(o, s) for o, s, a in index})
        diffs, order_diffs = [], {}
        for order, seed in identities:
            if (order, seed, "cat_cctp") not in index or (order, seed, "uq_v3") not in index:
                continue
            delta = index[order, seed, "uq_v3"]["final_f1"] - index[order, seed, "cat_cctp"]["final_f1"]
            diffs.append(delta)
            order_diffs.setdefault(order, []).append(delta)
        if not diffs:
            continue
        order_means = [mean(v) for v in order_diffs.values()]
        half = 4.3026527299 * std(order_means) / math.sqrt(3) if len(order_means) == 3 else None
        result.append({"dataset": dataset, "paired_cells": len(diffs),
            "orders": len(order_means), "mean_delta_f1": mean(diffs),
            "order_mean_delta": mean(order_means),
            "descriptive_order_t_ci_low": mean(order_means) - half if half is not None else None,
            "descriptive_order_t_ci_high": mean(order_means) + half if half is not None else None,
            "positive_cells": sum(d > 0 for d in diffs), "worst_delta": min(diffs),
            "reference_order_included": True})
    return result


def operator_attribution(root):
    summary, replay = [], []
    for path in sorted(root.glob("*/operator_ab.json")):
        data = json.loads(path.read_text())
        if data.get("status") != "complete":
            continue
        for arm in ("cat_cctp", "uq_semantic"):
            for student in ("legacy", "prompt"):
                for operator in ("legacy", "common"):
                    rows = [r for r in data["rows"] if (r["arm"], r["student"], r["operator"]) == (arm, student, operator)]
                    primary = [r for r in rows if r["repeat"] == 0]
                    summary.append({"dataset": data["dataset"], "budget_mode": data["budget_mode"],
                        "arm": arm, "student": student, "operator": operator,
                        "final_f1": mean(r["final_f1"] for r in primary),
                        "accuracy": mean(r["accuracy"] for r in primary)})
                    first = next(r for r in primary if r["seed"] == 150)
                    second = next(r for r in rows if r["repeat"] == 1)
                    replay.append({"dataset": data["dataset"], "arm": arm,
                        "student": student, "operator": operator,
                        "same_memory": first["memory_hash"] == second["memory_hash"],
                        "same_predictions": first["prediction_hash"] == second["prediction_hash"],
                        "same_seed_f1_difference": second["final_f1"] - first["final_f1"]})
    return summary, replay


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("results/unified_q_diagnostics_20261007"))
    args = p.parse_args()
    cells, stages, ab, failures, numerical, resolved_failures = collect(args.root)
    pairs = paired(cells)
    operators, replay = operator_attribution(args.root)
    for name, rows in (("method_cells", cells), ("stage_metrics", stages), ("student_ab", ab),
                       ("numerical_audit", numerical), ("paired_confirmation", pairs),
                       ("operator_attribution", operators), ("same_seed_replay", replay)):
        dump_csv(args.root / f"{name}.csv", rows)
    complete_stages = sum(r["run_status"] == "complete" for r in stages)
    summary = {"cells": cells, "student_ab": ab, "paired": pairs, "failures": failures,
               "resolved_failures": resolved_failures,
               "completed_stage_count": complete_stages,
               "stage_count": len(stages), "numerical_checks": numerical,
               "operators": operators, "same_seed_replay": replay}
    (args.root / "summary.json").write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    lines = ["# Unified-Q controlled diagnostics", "",
        "Report-dev only. Do not mix Student/budget/old-memory protocols. No test-driven selection.", "",
        "## Same-memory Student A/B", "",
        "|Dataset|Budget|Memory|Student|Final F1|Accuracy|Stage F1|Old F1|New F1|",
        "|---|---|---|---|---:|---:|---:|---:|---:|"]
    for r in ab:
        lines.append("|{dataset}|{budget_mode}|{arm}|{student}|{final_f1:.2f}|{accuracy:.2f}|{stage_avg_f1:.2f}|{old_f1:.2f}|{new_f1:.2f}|".format(**r))
    lines += ["", "## Training objective versus inference operator", "",
        "Same trained model evaluated with both raw-graph operators; repeats excluded from means.", "",
        "|Dataset|Memory|Training Student|Evaluation operator|Final F1|Accuracy|",
        "|---|---|---|---|---:|---:|"]
    for r in operators:
        lines.append("|{dataset}|{arm}|{student}|{operator}|{final_f1:.2f}|{accuracy:.2f}|".format(**r))
    for dataset in ("wikics", "bookhis", "sportsfit"):
        selected = [r for r in cells if r["dataset"] == dataset and not r["run"].startswith("confirm_")]
        if not selected:
            continue
        lines += ["", f"## {dataset}: reference order, one construction seed", "",
            "All rows below use the SAME Prompt Student. Old-X rows are a separate adaptation protocol.", "",
            "|Method|Protocol|Final F1|Accuracy|Stage F1|Stage Acc|Old F1|New F1|Construction s|Peak MiB|",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for r in selected:
            lines.append("|{arm}|{protocol_family}|{final_f1:.2f}|{accuracy:.2f}|{stage_avg_f1:.2f}|{stage_avg_accuracy:.2f}|{old_f1:.2f}|{new_f1:.2f}|{construction_seconds:.1f}|{peak_mb:.0f}|".format(**r))
    lines += ["", "## Independent order / construction confirmation", "",
              "CI treats orders, not Student seeds, as units; only three orders and one is the reference."]
    for r in pairs:
        lines += ["", json.dumps(r, ensure_ascii=True)]
    strict = [r for r in numerical if r["tolerance"] > 0]
    violations = [r for r in strict if max(r["max_row_error"], r["max_column_relative_error"]) > r["tolerance"]]
    lines += ["", "## Audit", "", f"Completed method cells: {len(cells)}; completed-run stages: {complete_stages}; recorded stages including partial attempts: {len(stages)}.",
              f"Strict solver stages: {len(strict)}; tolerance violations: {len(violations)}.",
              f"Unresolved failed queued tasks: {len(failures)}; resolved retries excluded from failure count: {len(resolved_failures)}.",
              "Shared-GPU timings are not exclusive throughput measurements."]
    for r in failures:
        lines.append(json.dumps(r, ensure_ascii=True))
    (args.root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"method_cells": len(cells), "stages": len(stages),
                      "ab_rows": len(ab), "failures": failures, "strict_violations": len(violations)}))


if __name__ == "__main__":
    main()
