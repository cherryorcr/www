"""Summarize the controlled P1-X X-operator experiments."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, pstdev


DATASETS = ("cora", "citeseer")
ORDERS = ("canonical", "order1701", "order1702")
ARMS = ("mean_x", "weighted_x", "moment_x")


def load_order(root: Path, dataset: str, order: str) -> dict:
    path = root / dataset / order / "results.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def stage_row(result: dict) -> dict:
    metrics = result["metrics"]["all"]
    def value(name: str):
        item = metrics.get(name)
        if item is None:
            return None
        return float(item["mean"])
    def spread(name: str):
        item = metrics.get(name)
        if item is None:
            return None
        return float(item["std"])
    return {
        "arm": result["arm"],
        "stage": int(result["stage"]),
        "rows": int(result["rows"]),
        "f1": value("cumulative_f1"),
        "f1_std": spread("cumulative_f1"),
        "old_f1": value("old_f1"),
        "old_f1_std": spread("old_f1"),
        "new_f1": value("new_f1"),
        "new_f1_std": spread("new_f1"),
    }


def summarize_order(root: Path, dataset: str, order: str) -> list[dict]:
    payload = load_order(root, dataset, order)
    by_arm = {arm: [] for arm in ARMS}
    for result in payload["results"]:
        row = stage_row(result)
        if row["arm"] in by_arm:
            by_arm[row["arm"]].append(row)
    out = []
    for arm in ARMS:
        stages = sorted(by_arm[arm], key=lambda row: row["stage"])
        if len(stages) != 3:
            raise ValueError(f"{dataset}/{order}/{arm}: expected 3 stages")
        costs = payload["costs"][arm]
        final = stages[-1]
        out.append({
            "dataset": dataset,
            "order": order,
            "arm": arm,
            "rows": [row["rows"] for row in stages],
            "stage_f1": [row["f1"] for row in stages],
            "stage_f1_std": [row["f1_std"] for row in stages],
            "stage_average_f1": mean(row["f1"] for row in stages),
            "final_f1": final["f1"],
            "final_std": final["f1_std"],
            "old_f1": final["old_f1"],
            "new_f1": final["new_f1"],
            "old_f1_std": final["old_f1_std"],
            "new_f1_std": final["new_f1_std"],
            "build_seconds": float(costs["stage_build_seconds_sum"]),
            "build_chain_seconds_with_io": float(costs["build_chain_seconds_with_io"]),
            "student_seconds_six_seeds": float(costs["student_seconds"]),
            "student_steps_six_seeds": int(costs["student_steps"]),
            "payload_bytes": int(costs["memory_payload"][-1]["logical_payload_bytes"]),
            "x_operator": costs.get("x_operator"),
        })
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for dataset in DATASETS:
        for order in ORDERS:
            rows.extend(summarize_order(args.root, dataset, order))

    by_key = {(row["dataset"], row["order"], row["arm"]): row
              for row in rows}
    for row in rows:
        baseline = by_key[(row["dataset"], row["order"], "mean_x")]
        row["delta_final_vs_mean"] = row["final_f1"] - baseline["final_f1"]
        row["delta_stage_average_vs_mean"] = (
            row["stage_average_f1"] - baseline["stage_average_f1"])
        row["delta_old_vs_mean"] = (
            None if row["old_f1"] is None or baseline["old_f1"] is None
            else row["old_f1"] - baseline["old_f1"])
        row["delta_new_vs_mean"] = (
            None if row["new_f1"] is None or baseline["new_f1"] is None
            else row["new_f1"] - baseline["new_f1"])

    aggregates = []
    for dataset in DATASETS:
        for arm in ARMS:
            subset = [row for row in rows
                      if row["dataset"] == dataset and row["arm"] == arm]
            aggregate = {
                "dataset": dataset,
                "arm": arm,
                "orders": [row["order"] for row in subset],
                "final_f1_mean": mean(row["final_f1"] for row in subset),
                "final_f1_order_sd": pstdev(row["final_f1"] for row in subset),
                "stage_average_f1_mean": mean(row["stage_average_f1"] for row in subset),
                "old_f1_mean": mean(row["old_f1"] for row in subset
                                     if row["old_f1"] is not None),
                "new_f1_mean": mean(row["new_f1"] for row in subset
                                     if row["new_f1"] is not None),
                "build_seconds_mean": mean(row["build_seconds"] for row in subset),
                "student_seconds_six_seeds_mean": mean(
                    row["student_seconds_six_seeds"] for row in subset),
                "payload_bytes_mean": mean(row["payload_bytes"] for row in subset),
                "wins_final_vs_mean": sum(row["delta_final_vs_mean"] > 1e-9
                                           for row in subset),
                "wins_stage_average_vs_mean": sum(
                    row["delta_stage_average_vs_mean"] > 1e-9 for row in subset),
                "delta_final_vs_mean": mean(row["delta_final_vs_mean"] for row in subset),
                "delta_final_order_sd_vs_mean": pstdev(
                    row["delta_final_vs_mean"] for row in subset),
                "delta_stage_average_vs_mean": mean(
                    row["delta_stage_average_vs_mean"] for row in subset),
                "delta_old_vs_mean": mean(row["delta_old_vs_mean"] for row in subset
                                           if row["delta_old_vs_mean"] is not None),
                "delta_new_vs_mean": mean(row["delta_new_vs_mean"] for row in subset
                                           if row["delta_new_vs_mean"] is not None),
            }
            aggregates.append(aggregate)

    args.output.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": "p1x-fixed-rows-text-topology-v1",
        "datasets": list(DATASETS),
        "orders": list(ORDERS),
        "arms": list(ARMS),
        "rows": rows,
        "aggregates": aggregates,
        "interpretation": {
            "final_f1": "all-prompt report-dev Macro-F1 at the final stage",
            "stage_average_f1": "equal mean of the three stage cumulative Macro-F1 values",
            "cost_scope": "construction plus six report student fits; shared preprocessing excluded",
            "kcenter": "reported from the frozen small-order comparison, not rerun in P1-X",
        },
    }
    (args.output / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    fields = [key for key in rows[0] if key not in {"rows", "stage_f1", "stage_f1_std"}]
    with (args.output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fields})
    with (args.output / "aggregate.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregates[0]))
        writer.writeheader()
        writer.writerows(aggregates)
    print(json.dumps({"rows": len(rows), "aggregates": len(aggregates)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
