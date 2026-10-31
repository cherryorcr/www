"""Summarize the matched CaT/GECC construction-seed confirmation."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev


METRICS = ("final_f1", "accuracy", "old_f1", "new_f1")


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def rows_from(root: Path) -> list[dict]:
    rows = []
    for path in sorted(root.glob("*_o*.json")):
        payload = read(path)
        if payload.get("status") != "complete":
            continue
        for item in payload.get("rows", []):
            final = item["final"]
            stage = item.get("stage_scores", [])
            rows.append({
                "dataset": item["dataset"],
                "order": item["order_seed"],
                "construction_seed": item["construction_seed"],
                "method": item["arm"],
                "final_f1": final.get("final_f1"),
                "accuracy": final.get("accuracy"),
                "old_f1": final.get("old_f1"),
                "new_f1": final.get("new_f1"),
                "stage_average_f1": mean(v["final_f1"] for v in stage) if stage else None,
                "stage_average_accuracy": mean(v["accuracy"] for v in stage) if stage else None,
                "memory_rows": item.get("memory_rows"),
                "memory_hash": item.get("memory_hash"),
            })
    return rows


def aggregate(rows: list[dict]) -> list[dict]:
    keys = ("dataset", "method")
    out = []
    for dataset in sorted({r["dataset"] for r in rows}):
        for method in sorted({r["method"] for r in rows if r["dataset"] == dataset}):
            selected = [r for r in rows if r["dataset"] == dataset and r["method"] == method]
            item = {"dataset": dataset, "method": method, "n": len(selected),
                    "orders": sorted({int(r["order"]) for r in selected}),
                    "construction_seeds": sorted({int(r["construction_seed"]) for r in selected})}
            for metric in METRICS + ("stage_average_f1", "stage_average_accuracy"):
                values = [float(r[metric]) for r in selected if r.get(metric) is not None]
                item[f"{metric}_mean"] = mean(values) if values else None
                item[f"{metric}_sd"] = stdev(values) if len(values) > 1 else 0.0
                item[f"{metric}_min"] = min(values) if values else None
            out.append(item)
    return out


def write_csv(path: Path, values: list[dict]) -> None:
    if not values:
        return
    fields = []
    for row in values:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in values:
            writer.writerow({key: json.dumps(value, ensure_ascii=True)
                             if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = rows_from(args.input)
    if len(rows) != 54:
        raise ValueError(f"expected 54 method/order/seed rows, found {len(rows)}")
    summary = aggregate(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "cells.csv", rows)
    write_csv(args.output / "summary.csv", summary)
    (args.output / "summary.json").write_text(
        json.dumps({"status": "complete", "rows": rows, "summary": summary},
                   ensure_ascii=True, indent=2), encoding="utf-8")
    lines = ["# CaT/GECC construction-seed confirmation", "",
             "Same current-delta-only edge-free protocol, 3 orders × 3 construction seeds, Prompt GCN Student seeds 150/151/152.", "",
             "| Dataset | Method | Cells | Final F1 mean±SD | Accuracy mean±SD | Stage F1 mean±SD | Worst F1 |", "|---|---|---:|---:|---:|---:|---:|"]
    for row in summary:
        lines.append(f"| {row['dataset']} | {row['method']} | {row['n']} | {row['final_f1_mean']:.2f}±{row['final_f1_sd']:.2f} | {row['accuracy_mean']:.2f}±{row['accuracy_sd']:.2f} | {row['stage_average_f1_mean']:.2f}±{row['stage_average_f1_sd']:.2f} | {row['final_f1_min']:.2f} |")
    lines += ["", "The table is an adapter-stability confirmation, not an official reproduction of CaT or GECC."]
    (args.output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "rows": len(rows), "groups": len(summary)}, ensure_ascii=True))


if __name__ == "__main__":
    main()
