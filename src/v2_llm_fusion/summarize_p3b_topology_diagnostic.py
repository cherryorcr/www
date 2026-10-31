"""Summarize P3-B topology diagnostic JSON artifacts."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def mean(values):
    return sum(values) / len(values) if values else float("nan")


def sd(values):
    m = mean(values)
    return math.sqrt(mean([(v - m) ** 2 for v in values])) if values else float("nan")


def load_results(root: Path):
    rows, topology, costs = [], [], []
    for dataset_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for order_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
            result = json.loads((order_dir / "results.json").read_text())
            for arm in sorted({r["arm"] for r in result["results"]}):
                stages = sorted((r for r in result["results"] if r["arm"] == arm),
                                key=lambda r: r["stage"])
                final = stages[-1]["metrics"]["final_f1"]["mean"]
                rows.append({
                    "dataset": dataset_dir.name,
                    "order": order_dir.name,
                    "arm": arm,
                    "final_f1": final,
                    "stage_average_f1": mean([r["metrics"]["final_f1"]["mean"]
                                               for r in stages]),
                    "old_f1": stages[-1]["metrics"]["old_f1"]["mean"],
                    "new_f1": stages[-1]["metrics"]["new_f1"]["mean"],
                    "old_embedding_drift": stages[-1]["metrics"][
                        "old_embedding_drift"]["mean"],
                })
                topo = json.loads((order_dir / f"{arm}_topology.json").read_text())
                for item in topo:
                    if item["old_rows"]:
                        topology.append({
                            "dataset": dataset_dir.name,
                            "order": order_dir.name,
                            "arm": arm,
                            "old_norm_drift": item["old_old_normalized_relative_drift"],
                            "old_prop_drift": item["old_propagation_relative_drift"],
                            "cross_x_message": item["cross_x_message_relative"],
                            "total_prop_drift": item[
                                "old_total_x_propagation_relative_drift"],
                            "cross_undirected": item["edge_counts"]["cross_undirected"],
                        })
                cost = result["costs"][arm]
                costs.append({"dataset": dataset_dir.name, "order": order_dir.name,
                              "arm": arm, "topology_seconds": cost["topology_seconds"],
                              "student_seconds": cost["student_seconds"],
                              "student_fits": cost["student_fits"],
                              "student_steps": cost["student_steps"]})
    return rows, topology, costs


def aggregate(rows, keys):
    output = []
    groups = sorted({tuple(row[k] for k in keys) for row in rows})
    for group in groups:
        selected = [row for row in rows if tuple(row[k] for k in keys) == group]
        item = {k: v for k, v in zip(keys, group)}
        for key in ("final_f1", "stage_average_f1", "old_f1", "new_f1",
                    "old_embedding_drift", "old_norm_drift", "old_prop_drift",
                    "cross_x_message", "total_prop_drift", "cross_undirected",
                    "topology_seconds", "student_seconds", "student_fits",
                    "student_steps"):
            values = [row[key] for row in selected if key in row and row[key] is not None]
            if values:
                item[f"{key}_mean"] = mean(values)
                item[f"{key}_sd"] = sd(values)
        output.append(item)
    return output


def write_csv(path: Path, rows):
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    rows, topology, costs = load_results(args.root)
    summary = args.root / "summary"
    summary.mkdir(exist_ok=True)
    write_csv(summary / "per_order.csv", rows)
    write_csv(summary / "final_stage_aggregate.csv", aggregate(rows, ["dataset", "arm"]))
    write_csv(summary / "topology_aggregate.csv", aggregate(topology, ["dataset", "arm"]))
    write_csv(summary / "cost_aggregate.csv", aggregate(costs, ["dataset", "arm"]))
    (summary / "summary.json").write_text(json.dumps({
        "per_order": rows,
        "final_stage_aggregate": aggregate(rows, ["dataset", "arm"]),
        "topology_aggregate": aggregate(topology, ["dataset", "arm"]),
        "cost_aggregate": aggregate(costs, ["dataset", "arm"]),
    }, indent=2), encoding="utf-8")
    print(f"wrote {summary}")


if __name__ == "__main__":
    main()
