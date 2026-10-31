"""Summarize current edge-free additional-dataset runs and matched baselines.

This report keeps the new ArXiv/Cora runs separate from the sealed primary
three-dataset table. Missing or incomplete cells are reported as pending; no
method is selected by a favorable cell.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize_extended(root: Path) -> tuple[list[dict], list[dict]]:
    cells: list[dict] = []
    summary: list[dict] = []
    for path in sorted(root.glob("*_o*.json")):
        payload = load(path)
        if payload.get("status") != "complete":
            continue
        protocol = payload["protocol"]
        dataset = protocol["dataset"]
        order = path.stem.rsplit("_o", 1)[-1]
        for row in payload.get("rows", []):
            final = row["final"]
            cells.append({
                "dataset": dataset,
                "order": order,
                "method": row["arm"],
                "final_f1": final["final_f1"],
                "accuracy": final["accuracy"],
                "old_f1": final.get("old_f1"),
                "new_f1": final.get("new_f1"),
                "construction_seconds": row.get("construction_seconds"),
                "memory_rows": row.get("memory_rows"),
                "protocol": protocol["name"],
            })
    for dataset in sorted({row["dataset"] for row in cells}):
        for method in sorted({row["method"] for row in cells if row["dataset"] == dataset}):
            selected = [row for row in cells
                        if row["dataset"] == dataset and row["method"] == method]
            item = {"dataset": dataset, "method": method, "n_orders": len(selected),
                    "orders": ",".join(row["order"] for row in selected)}
            for metric in ("final_f1", "accuracy", "old_f1", "new_f1",
                           "construction_seconds"):
                values = [float(row[metric]) for row in selected
                          if row.get(metric) is not None]
                if values:
                    item[f"{metric}_mean"] = mean(values)
                    item[f"{metric}_sd"] = stdev(values) if len(values) > 1 else 0.0
                    item[f"{metric}_min"] = min(values)
            summary.append(item)
    return cells, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results/extended_evidence_20261011"))
    args = parser.parse_args()
    cells, summary = summarize_extended(args.root)
    write_csv(args.root / "additional_dataset_cells.csv", cells)
    write_csv(args.root / "additional_dataset_summary.csv", summary)
    status = {
        "status": "complete" if cells else "pending",
        "datasets": sorted({row["dataset"] for row in cells}),
        "complete_cells": len(cells),
        "summary_rows": len(summary),
    }
    (args.root / "additional_dataset_status.json").write_text(
        json.dumps(status, indent=2, ensure_ascii=True), encoding="utf-8")

    lines = [
        "# Additional current-protocol datasets",
        "",
        "All rows use current-delta-only construction, edge-free memory, the Prompt GraphText Student, and report-dev evaluation. These rows are supplementary and are not used to retune the primary method.",
        "",
        "| Dataset | Method | Orders | Final F1 mean±SD | Accuracy mean±SD | Old F1 | New F1 | Build s |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        fmt = lambda metric: (f"{row[metric + '_mean']:.2f}±{row[metric + '_sd']:.2f}"
                              if metric + "_mean" in row else "NA")
        lines.append(
            f"| {row['dataset']} | {row['method']} | {row['n_orders']} | "
            f"{fmt('final_f1')} | {fmt('accuracy')} | "
            f"{row.get('old_f1_mean', float('nan')):.2f} | "
            f"{row.get('new_f1_mean', float('nan')):.2f} | "
            f"{row.get('construction_seconds_mean', float('nan')):.1f} |")
    lines += [
        "",
        "ArXiv is an additional large-scale transfer check; Cora is an additional standard small-graph check. The Cora stream uses the three predeclared order manifests extracted from the existing current Prompt Student protocol, but the memory is rebuilt by the current edge-free runner.",
        "",
        "No link-prediction result is merged here: the primary Unified-Q memory is edge-free. The existing link-prediction artifact uses a separate topology-enabled protocol and remains a supplementary structural audit, not evidence for the primary method.",
        "",
        f"Status: `{status['status']}`; complete cells: `{len(cells)}`.",
    ]
    (args.root / "additional_dataset_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(status, ensure_ascii=True))


if __name__ == "__main__":
    main()
