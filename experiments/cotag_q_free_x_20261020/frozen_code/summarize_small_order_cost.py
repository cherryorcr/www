"""Validate and summarize frozen small-graph order/cost confirmation artifacts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def read(path, sources):
    sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_arm(dataset, order, arm, stages, cost, audits, costs, manifest):
    stages = sorted(stages, key=lambda s: s["stage"])
    if [s["stage"] for s in stages] != [0, 1, 2]:
        raise ValueError("incomplete stage coverage")
    key = "selected_metrics" if arm == "raw_long" else "metrics"
    grid, metrics, seen_grid = [], [], []
    for s, stage in enumerate(stages):
        fits = sorted(stage["per_seed"], key=lambda r: r["seed"])
        if [r["seed"] for r in fits] != list(range(100, 106)):
            raise ValueError("incomplete seed coverage")
        if any(r["steps"] != (8192 if arm == "raw_long" else manifest["student_steps"][s]) for r in fits):
            raise ValueError("training budget mismatch")
        if arm not in ("raw_equal", "raw_long", "raw_finetune", "v4_init_finetune"):
            if stage["rows"] != manifest["budgets"][s]:
                raise ValueError("node budget mismatch")
        grid.append([r[key]["all"]["cumulative_f1"] for r in fits])
        seen_grid.append([r[key]["seen"]["cumulative_f1"] for r in fits])
        metrics.append([r[key]["all"] for r in fits])
        if s == 2 and grid[-1] != seen_grid[-1]:
            raise ValueError("final seen/all mismatch")
        if arm.endswith("finetune") and s:
            previous = sorted(stages[s-1]["per_seed"], key=lambda r: r["seed"])
            if any(a["start_state_sha256"] != b["end_state_sha256"] for a, b in zip(fits, previous)):
                raise ValueError("fine-tune state chain mismatch")
    # Historical-task maxima exclude stages before that task arrived.
    forgetting = []
    for seed_index in range(6):
        values = [max(metrics[s][seed_index]["per_task_f1"][task] for s in range(task, 3))
                  - metrics[-1][seed_index]["per_task_f1"][task] for task in range(2)]
        forgetting.append(float(np.mean(values)))
    dependency = 0.
    if arm in ("raw_finetune", "v4_init_finetune"):
        source = "raw_equal" if arm == "raw_finetune" else "v4"
        dependency = audits[source][0]["build_seconds"]
    if not cost["fresh_chain_measurement"]:
        raise ValueError("cost chain not measured fresh; do not report cached IO as construction")
    if cost["checkpoint_rechecks"] != 18 or cost["student_fits"] != 18:
        raise ValueError("checkpoint audit incomplete")
    total_build = cost["build_chain_seconds_with_io"] + dependency
    selector_seconds = cost["selector_seconds"]
    selector_runs, selector_steps = cost["selector_runs"], cost["selector_steps"]
    if arm == "v4_init_finetune":
        selector_seconds += audits["v4"][0]["selector_seconds"]
        selector_runs += audits["v4"][0]["selector_runs"]
        selector_steps += audits["v4"][0]["selector_steps"]
    return {"dataset": dataset, "order": order, "arm": arm,
        "rows": [s["rows"] for s in stages], "budgets": manifest["budgets"],
        "stage_classes": manifest["stage_classes"],
        "final_f1": float(np.mean(grid[-1])), "final_seed_std": float(np.std(grid[-1])),
        "final_per_seed": grid[-1], "stage_f1": np.mean(grid, axis=1).tolist(),
        "stage_average_f1": float(np.mean(grid)), "stage_average_seen_f1": float(np.mean(seen_grid)),
        "old_f1": float(np.mean([r["old_f1"] for r in metrics[-1]])),
        "new_f1": float(np.mean([r["new_f1"] for r in metrics[-1]])),
        "forgetting_pp": float(np.mean(forgetting)),
        "build_seconds": total_build, "initializer_build_seconds": dependency,
        "response_seconds": selector_seconds, "selector_runs": selector_runs,
        "selector_steps": selector_steps, "student_seconds_six_seeds": cost["student_seconds"],
        "student_seconds_one_seed_trajectory": cost["student_seconds"] / 6,
        "student_steps_one_seed": cost["student_steps"] // 6,
        "total_seconds_one_seed": total_build + cost["student_wall_seconds"] / 6,
        "cost_scope": "one construction + mean one student trajectory including report/IO; shared preprocessing separate",
        "peak_allocated_mb": max(cost["build_peak_allocated_mb"], cost["student_peak_allocated_mb"]),
        "payload_bytes": cost["memory_payload"][-1]["logical_payload_bytes"],
        "edges": cost["memory_payload"][-1]["edge_entries"]}


def aggregate_orders(rows):
    result = []
    for dataset in ("cora", "citeseer"):
        local = [r for r in rows if r["dataset"] == dataset]
        ref = {r["order"]: r for r in local if r["arm"] == "raw_long"}
        for row in local:
            other = ref[row["order"]]
            row["retention_percent"] = 100 * row["final_f1"] / other["final_f1"]
            row["stage_retention_percent"] = 100 * row["stage_average_f1"] / other["stage_average_f1"]
        for arm in dict.fromkeys(r["arm"] for r in local):
            subset = [r for r in local if r["arm"] == arm]
            if len(subset) != 3:
                raise ValueError("all three orders required")
            record = {"dataset": dataset, "arm": arm, "orders": [r["order"] for r in subset]}
            for field in ("final_f1", "stage_average_f1", "stage_average_seen_f1", "old_f1", "new_f1",
                          "forgetting_pp", "build_seconds", "response_seconds", "selector_runs",
                          "selector_steps", "student_seconds_one_seed_trajectory", "total_seconds_one_seed",
                          "payload_bytes", "edges"):
                record[field] = float(np.mean([r[field] for r in subset]))
            record["final_order_means"] = [r["final_f1"] for r in subset]
            record["order_std"] = float(np.std(record["final_order_means"]))
            record["order_min"] = min(record["final_order_means"])
            record["order_max"] = max(record["final_order_means"])
            record["retention_percent"] = 100 * record["final_f1"] / np.mean([r["final_f1"] for r in ref.values()])
            record["stage_retention_percent"] = 100 * record["stage_average_f1"] / np.mean([r["stage_average_f1"] for r in ref.values()])
            result.append(record)
    return result


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
                     + ["| " + " | ".join(map(str, row)) + " |" for row in rows])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--external-summary", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sources, rows, plans, preprocessing = {}, [], {}, {}
    independent = read(args.root / "independent_acceptance.json", sources)
    if independent["errors"] or len(independent["datasets"]) != 2:
        raise ValueError("independent artifact audit incomplete")
    for audited in independent["datasets"]:
        if (audited["errors"] or audited["report_fit_digests_checked"] != 702
                or audited["memory_artifacts_checked"] != 99
                or audited["synthetic_artifacts_rebuilt"] != 18):
            raise ValueError("independent artifact coverage incomplete")
    for dataset in ("cora", "citeseer"):
        base = args.root / dataset
        plan = read(base / "plan.json", sources)
        accepted = read(base / "acceptance.json", sources)
        if not accepted["complete"] or accepted["reloaded_fits"] != 702:
            raise ValueError("dataset not fully audited")
        plans[dataset] = plan
        preprocessing[dataset] = read(base / "preprocessing.json", sources)
        for order in plan["orders"]:
            payload = read(base / order / "results.json", sources)
            manifest = read(base / order / "manifest.json", sources)
            if manifest["stage_classes"] != plan["orders"][order]:
                raise ValueError("stage order changed")
            if sorted(set(r["arm"] for r in payload["results"])) != sorted(plan["arms"]):
                raise ValueError("arm coverage mismatch")
            for arm in plan["arms"]:
                rows.append(summarize_arm(dataset, order, arm,
                    [r for r in payload["results"] if r["arm"] == arm],
                    payload["costs"][arm], payload["audits"], payload["costs"], manifest))
    aggregates = aggregate_orders(rows)
    comparisons = []
    for dataset in plans:
        v4 = {r["order"]: r for r in rows if r["dataset"] == dataset and r["arm"] == "v4"}
        for arm in plans[dataset]["arms"]:
            peers = [r for r in rows if r["dataset"] == dataset and r["arm"] == arm]
            gains = [v4[r["order"]]["final_f1"] - r["final_f1"] for r in peers]
            stage_gains = [v4[r["order"]]["stage_average_f1"] - r["stage_average_f1"] for r in peers]
            comparisons.append({"dataset": dataset, "arm": arm,
                "v4_minus_final_by_order": gains, "v4_minus_stage_average_by_order": stage_gains,
                "final_positive_orders": sum(g > 1e-8 for g in gains),
                "final_nonnegative_orders": sum(g >= -1e-8 for g in gains),
                "stage_positive_orders": sum(g > 1e-8 for g in stage_gains)})
    external = read(args.external_summary, sources) if args.external_summary else None
    payload = {"rows": rows, "aggregates": aggregates, "comparisons": comparisons,
        "preprocessing": preprocessing, "plans": plans, "sources": sources,
        "archived_external_single_order": external,
        "independent_acceptance": independent,
        "cost_limitations": "fine-tune stage0 dependency includes algorithm but omits its small save/wrapper overhead; diagnostic replays and audit reloads excluded from one-deployment cost; shared preprocessing separate",
        "audit": {"report_fits": 1404, "orders_per_dataset": 3,
                  "test_labels_evaluated": False, "complete": True},
        "aggregation": "equal order means, each over six students; ratio of aggregated F1 means, not mean ratios; order SD is descriptive n=3",
        "forgetting": "mean historical-task max through final minus final; changing seen support/prompt competition also affects it"}
    (args.output / "summary.json").write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    for name, data in [("per_order", rows), ("aggregate", aggregates), ("paired_comparisons", comparisons)]:
        with (args.output / f"{name}.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    blocks = ["# Generated Order and Cost Tables", "Three prespecified orders; not independent new datasets."]
    for dataset in plans:
        local = [r for r in aggregates if r["dataset"] == dataset]
        blocks += [f"## {dataset}: performance", table(
            ["Arm", "Final F1", "Order SD", "Stage mean", "Retention %", "Stage retention %", "Old", "New"],
            [[r["arm"]] + [f"{r[k]:.2f}" for k in ("final_f1", "order_std", "stage_average_f1",
               "retention_percent", "stage_retention_percent", "old_f1", "new_f1")] for r in local])]
        blocks += [f"## {dataset}: fresh costs", table(
            ["Arm", "Build s", "Response s", "Student 1-seed s", "Total 1-seed s", "Payload KiB"],
            [[r["arm"]] + [f"{r[k]:.2f}" for k in ("build_seconds", "response_seconds",
              "student_seconds_one_seed_trajectory", "total_seconds_one_seed")] + [f"{r['payload_bytes']/1024:.2f}"] for r in local])]
        subset = [r for r in rows if r["dataset"] == dataset]
        blocks += [f"## {dataset}: per order", table(
            ["Order", "Arm", "Final", "Stage mean", "Retention %", "Forgetting"],
            [[r["order"], r["arm"]] + [f"{r[k]:.2f}" for k in ("final_f1", "stage_average_f1", "retention_percent", "forgetting_pp")] for r in subset])]
    (args.output / "tables.md").write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "aggregates": len(aggregates), "report_fits": 1404}))


if __name__ == "__main__":
    main()
