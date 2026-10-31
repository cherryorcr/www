"""Summarize the fixed-budget P1-B guarded response experiments."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, pstdev


DATASETS = ("cora", "citeseer")
ORDERS = ("canonical", "order1701", "order1702")
ARMS = ("dispersion", "response_guarded", "matched_random_transfer")


def read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def metric_value(result: dict, name: str, split: str = "all") -> float | None:
    item = result["metrics"].get(split, {}).get(name)
    return None if item is None else float(item["mean"])


def metric_std(result: dict, name: str, split: str = "all") -> float | None:
    item = result["metrics"].get(split, {}).get(name)
    return None if item is None else float(item["std"])


def task_forgetting(results: list[dict], arm: str) -> float | None:
    """Mean positive drop on already-seen task blocks at the final stage."""
    by_stage = {
        int(item["stage"]): item for item in results if item["arm"] == arm
    }
    final = by_stage.get(max(by_stage))
    if final is None or int(final["stage"]) == 0:
        return None
    previous = [by_stage[s] for s in sorted(by_stage) if s < int(final["stage"])]
    drops = []
    for seed_index, final_fit in enumerate(final["per_seed"]):
        final_tasks = final_fit["metrics"]["all"].get("per_task_f1", [])
        for task_id, final_value in enumerate(final_tasks[:-1]):
            historical = []
            for item in previous:
                fit = item["per_seed"][seed_index]
                tasks = fit["metrics"]["all"].get("per_task_f1", [])
                if task_id < len(tasks):
                    historical.append(float(tasks[task_id]))
            if historical:
                drops.append(max(0.0, max(historical) - float(final_value)))
    return None if not drops else mean(drops)


def stage_rows(payload: dict, arm: str) -> list[dict]:
    rows = []
    for result in payload["results"]:
        if result["arm"] != arm:
            continue
        rows.append({
            "stage": int(result["stage"]),
            "rows": int(result["rows"]),
            "f1": metric_value(result, "cumulative_f1"),
            "f1_std": metric_std(result, "cumulative_f1"),
            "old_f1": metric_value(result, "old_f1"),
            "old_f1_std": metric_std(result, "old_f1"),
            "new_f1": metric_value(result, "new_f1"),
            "new_f1_std": metric_std(result, "new_f1"),
            "seen_f1": metric_value(result, "cumulative_f1", "seen"),
            "seen_f1_std": metric_std(result, "cumulative_f1", "seen"),
        })
    return sorted(rows, key=lambda row: row["stage"])


def transfer_stats(root: Path, dataset: str, order: str, arm: str) -> dict:
    order_dir = root / dataset / order
    transfers = []
    accepted = 0
    selector_runs = 0
    selector_steps = 0
    for stage in range(3):
        audit = read_json(order_dir / f"{arm}_s{stage}_construction.json")
        selection = audit.get("selection", {})
        selector_runs += int(audit.get("selector_runs", 0))
        selector_steps += int(audit.get("selector_steps", 0))
        if arm == "response_guarded":
            decisions = selection.get("decisions", [])
            selected = selection.get("selected_class_rows", {})
            moved = 0
            if decisions:
                proposal = decisions[-1].get("allocation", {})
                # A gated transfer is accepted exactly when the final allocation
                # equals the candidate allocation.  On gate failure the audit
                # still records the rejected candidate, while selected rows
                # remain at the dispersion allocation.
                if selected == proposal:
                    moved = 1
                    accepted += 1
            transfers.append(moved)
        elif arm == "matched_random_transfer":
            transfer = selection.get("transfer", {})
            moved = int(transfer.get("rows", 0))
            accepted += int(moved > 0)
            transfers.append(moved)
        else:
            transfers.append(0)
    return {
        "transfer_rows_by_stage": transfers,
        "transfer_stages": sum(value > 0 for value in transfers),
        "selector_runs": selector_runs,
        "selector_steps": selector_steps,
        "response_acceptance_rate": accepted / 3.0 if arm == "response_guarded" else None,
    }


def summarize_order(root: Path, dataset: str, order: str) -> list[dict]:
    order_dir = root / dataset / order
    payload = read_json(order_dir / "results.json")
    output = []
    for arm in ARMS:
        stages = stage_rows(payload, arm)
        if len(stages) != 3:
            raise ValueError(f"{dataset}/{order}/{arm}: expected 3 stages")
        cost = read_json(order_dir / f"{arm}_cost.json")
        transfer = transfer_stats(root, dataset, order, arm)
        final = stages[-1]
        output.append({
            "dataset": dataset,
            "order": order,
            "arm": arm,
            "rows": [item["rows"] for item in stages],
            "stage_f1": [item["f1"] for item in stages],
            "stage_f1_std": [item["f1_std"] for item in stages],
            "stage_average_f1": mean(item["f1"] for item in stages),
            "final_f1": final["f1"],
            "final_f1_std": final["f1_std"],
            "final_old_f1": final["old_f1"],
            "final_new_f1": final["new_f1"],
            "final_old_f1_std": final["old_f1_std"],
            "final_new_f1_std": final["new_f1_std"],
            "final_seen_f1": final["seen_f1"],
            "task_forgetting_pp": task_forgetting(payload["results"], arm),
            "build_seconds": float(cost["stage_build_seconds_sum"]),
            "selector_seconds": float(cost.get("selector_seconds", 0.0)),
            "build_chain_seconds_with_io": float(cost["build_chain_seconds_with_io"]),
            "student_seconds_six_seeds": float(cost["student_seconds"]),
            "total_seconds_six_seeds": float(cost["build_chain_seconds_with_io"] + cost["student_seconds"]),
            "student_steps_six_seeds": int(cost["student_steps"]),
            "selector_runs": transfer["selector_runs"],
            "selector_steps": transfer["selector_steps"],
            "transfer_rows_by_stage": transfer["transfer_rows_by_stage"],
            "transfer_stages": transfer["transfer_stages"],
            "response_acceptance_rate": transfer["response_acceptance_rate"],
            "payload_bytes": int(cost["memory_payload"][-1]["logical_payload_bytes"]),
        })
    return output


def aggregate(rows: list[dict]) -> list[dict]:
    result = []
    for dataset in DATASETS:
        subset_dataset = [row for row in rows if row["dataset"] == dataset]
        for arm in ARMS:
            subset = [row for row in subset_dataset if row["arm"] == arm]
            if len(subset) != len(ORDERS):
                raise ValueError(f"{dataset}/{arm}: incomplete order set")
            baseline = {
                row["order"]: row for row in subset_dataset
                if row["arm"] == "dispersion"
            }
            deltas = [row["final_f1"] - baseline[row["order"]]["final_f1"] for row in subset]
            stage_deltas = [row["stage_average_f1"] - baseline[row["order"]]["stage_average_f1"] for row in subset]
            result.append({
                "dataset": dataset,
                "arm": arm,
                "final_f1_mean": mean(row["final_f1"] for row in subset),
                "final_f1_order_sd": pstdev(row["final_f1"] for row in subset),
                "stage_average_f1_mean": mean(row["stage_average_f1"] for row in subset),
                "final_old_f1_mean": mean(row["final_old_f1"] for row in subset),
                "final_new_f1_mean": mean(row["final_new_f1"] for row in subset),
                "task_forgetting_pp_mean": mean(row["task_forgetting_pp"] for row in subset),
                "build_seconds_mean": mean(row["build_seconds"] for row in subset),
                "selector_seconds_mean": mean(row["selector_seconds"] for row in subset),
                "total_seconds_six_seeds_mean": mean(row["total_seconds_six_seeds"] for row in subset),
                "selector_steps_mean": mean(row["selector_steps"] for row in subset),
                "transfer_stages_total": sum(row["transfer_stages"] for row in subset),
                "response_acceptance_rate_mean": mean(
                    row["response_acceptance_rate"] for row in subset
                    if row["response_acceptance_rate"] is not None
                ) if arm == "response_guarded" else None,
                "delta_final_vs_dispersion_mean": mean(deltas),
                "delta_final_vs_dispersion_sd": pstdev(deltas),
                "delta_stage_average_vs_dispersion_mean": mean(stage_deltas),
                "wins_final_vs_dispersion": sum(value > 1e-9 for value in deltas),
                "wins_stage_average_vs_dispersion": sum(value > 1e-9 for value in stage_deltas),
            })
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [
        row
        for dataset in DATASETS
        for order in ORDERS
        for row in summarize_order(args.root, dataset, order)
    ]
    aggregates = aggregate(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": "p1b-fixed-rate-guarded-response-v1",
        "datasets": list(DATASETS),
        "orders": list(ORDERS),
        "arms": list(ARMS),
        "metric": "all-prompt report-dev cumulative Macro-F1; test labels not evaluated",
        "rows": rows,
        "aggregates": aggregates,
        "interpretation": {
            "response_guarded": "one-row response transfer from dispersion allocation, accepted only when LCB and horizon gates pass; otherwise dispersion fallback",
            "matched_random_transfer": "one-row random transfer with identical fixed total budget, no response selection",
            "cost_scope": "fresh construction plus six student fits per stage chain; shared preprocessing excluded",
            "task_forgetting_pp": "positive mean drop on previously seen task blocks from their best earlier stage to final stage",
        },
    }
    (args.output / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    row_fields = [key for key in rows[0] if key not in {"rows", "stage_f1", "stage_f1_std", "transfer_rows_by_stage"}]
    with (args.output / "per_order.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=row_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in row_fields})
    with (args.output / "aggregate.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregates[0]))
        writer.writeheader()
        writer.writerows(aggregates)
    print(json.dumps({"rows": len(rows), "aggregates": len(aggregates)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
