"""Summarize Unified-Q comparison outputs into CSV/Markdown tables."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, pstdev


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def collect(root: Path):
    rows = []
    for result in root.glob("*/order*/results.json"):
        value = read(result)
        for row in value["rows"]:
            aggregate = row["aggregate"]
            rows.append({
                "dataset": row["dataset"],
                "order": row["order"],
                "arm": row["arm"],
                "stage": int(row["stage"]),
                "macro_f1": float(aggregate["mean_f1"]),
                "accuracy": float(aggregate["mean_accuracy"]),
                "old_f1": float(aggregate.get("old_f1", 0.0) or 0.0),
                "new_f1": float(aggregate.get("new_f1", 0.0) or 0.0),
                "construction_seconds": float(row.get("construction_seconds", 0.0) or 0.0),
            })
    return rows


def grouped(rows, keys):
    groups = {}
    for row in rows:
        key = tuple(row[key] for key in keys)
        groups.setdefault(key, []).append(row)
    return groups


def write(root: Path, rows):
    root.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["dataset", "order", "arm"]
    with (root / "stage_rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    final = []
    for (dataset, order, arm), values in grouped(rows, ["dataset", "order", "arm"]).items():
        last = max(values, key=lambda item: item["stage"])
        final.append({"dataset": dataset, "order": order, "arm": arm,
                      "final_macro_f1": last["macro_f1"],
                      "final_accuracy": last["accuracy"],
                      "final_old_f1": last["old_f1"],
                      "final_new_f1": last["new_f1"],
                      "stage_average_f1": mean(item["macro_f1"] for item in values),
                      "stage_average_accuracy": mean(item["accuracy"] for item in values),
                      "construction_seconds": sum(item["construction_seconds"] for item in values)})
    final_fields = list(final[0]) if final else ["dataset", "order", "arm"]
    with (root / "final_rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=final_fields)
        writer.writeheader()
        writer.writerows(final)

    pooled = []
    for (dataset, arm), values in grouped(final, ["dataset", "arm"]).items():
        pooled.append({"dataset": dataset, "arm": arm,
                       "final_macro_f1_mean": mean(item["final_macro_f1"] for item in values),
                       "final_macro_f1_std": pstdev(item["final_macro_f1"] for item in values) if len(values) > 1 else 0.0,
                       "final_accuracy_mean": mean(item["final_accuracy"] for item in values),
                       "final_accuracy_std": pstdev(item["final_accuracy"] for item in values) if len(values) > 1 else 0.0,
                       "stage_average_f1_mean": mean(item["stage_average_f1"] for item in values),
                       "stage_average_accuracy_mean": mean(item["stage_average_accuracy"] for item in values),
                       "construction_seconds_mean": mean(item["construction_seconds"] for item in values)})
    pooled_fields = list(pooled[0]) if pooled else ["dataset", "arm"]
    with (root / "pooled_rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=pooled_fields)
        writer.writeheader()
        writer.writerows(pooled)

    lines = ["# Unified-Q comparison", "", "## Pooled final results", "",
             "| Dataset | Method | Final F1 | Final Acc. | Stage-average F1 | Stage-average Acc. |", 
             "|---|---|---:|---:|---:|---:|"]
    for item in pooled:
        lines.append("| {dataset} | {arm} | {final_macro_f1_mean:.3f} ± {final_macro_f1_std:.3f} | {final_accuracy_mean:.3f} ± {final_accuracy_std:.3f} | {stage_average_f1_mean:.3f} | {stage_average_accuracy_mean:.3f} |".format(**item))
    lines.extend(["", "## Per-order final results", "",
                  "| Dataset | Order | Method | Final F1 | Final Acc. | Old F1 | New F1 |", 
                  "|---|---|---|---:|---:|---:|---:|"])
    for item in sorted(final, key=lambda value: (value["dataset"], value["order"], value["arm"])):
        lines.append("| {dataset} | {order} | {arm} | {final_macro_f1:.3f} | {final_accuracy:.3f} | {final_old_f1:.3f} | {final_new_f1:.3f} |".format(**item))
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write(args.output, collect(args.root))


if __name__ == "__main__":
    main()

