"""Summarize the exact-5% construction-time text feedback diagnostic."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean, stdev


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/uq_exact5_no_node_text_20261015"
DATASETS = ("arxiv", "sportsfit", "bookhis")
SEEDS = (271006, 271108, 271109)


def ci(values: list[float]) -> tuple[float, float]:
    m = mean(values)
    if len(values) < 2:
        return m, m
    # n=3 paired t interval; this is descriptive, not a model-selection test.
    critical = 4.303
    half = critical * stdev(values) / (len(values) ** .5)
    return m - half, m + half


def read_cell(dataset: str, seed: int) -> dict:
    path = OUT / f"{dataset}_o9101_c{seed}" / "results.json"
    return json.loads(path.read_text(encoding="utf-8"))


def primary(dataset: str, seed: int) -> dict:
    path = ROOT / "results/uq_final_evidence_20261012/confirmation" / f"{dataset}_o9101_c{seed}" / "results.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    rows = sorted((row for row in value["rows"] if row["arm"] == "uq_dm_edgefree"), key=lambda row: row["stage"])
    return rows[-1]["scores"]["sealed_test"]


def main() -> None:
    rows = []
    for dataset in DATASETS:
        for seed in SEEDS:
            value = read_cell(dataset, seed)
            assert value["status"] == "complete", (dataset, seed, value["status"])
            no_text = value["rows"][-1]["scores"]["sealed_test"]
            full = primary(dataset, seed)
            for arm, score in (("primary", full), ("feature_only_q_posthoc_text", no_text)):
                rows.append({"dataset": dataset, "construction_seed": seed, "arm": arm,
                             "f1": score["final_f1"], "accuracy": score["accuracy"],
                             "old_f1": score["old_f1"], "new_f1": score["new_f1"]})
    out_csv = OUT / "summary_cells.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Exact-5% construction-time text feedback diagnostic", "",
             "The control removes node-text initialization cost and payload pair-loss feedback during Q optimization, but keeps learned Q, DM, propagation, residual, hard real-text materialization, frozen prefix, and the common Student. Results are sealed-test diagnostics and were not used to tune the main method.", ""]
    summary = []
    for dataset in DATASETS:
        p = [row for row in rows if row["dataset"] == dataset and row["arm"] == "primary"]
        c = [row for row in rows if row["dataset"] == dataset and row["arm"] != "primary"]
        deltas = [b["f1"] - a["f1"] for a, b in zip(p, c)]
        acc_deltas = [b["accuracy"] - a["accuracy"] for a, b in zip(p, c)]
        low, high = ci(deltas)
        summary.append({"dataset": dataset, "primary_f1": mean(r["f1"] for r in p),
                        "control_f1": mean(r["f1"] for r in c), "delta_f1": mean(deltas),
                        "delta_f1_ci_low": low, "delta_f1_ci_high": high,
                        "positive_cells": sum(v > 0 for v in deltas), "cells": len(deltas),
                        "primary_acc": mean(r["accuracy"] for r in p),
                        "control_acc": mean(r["accuracy"] for r in c),
                        "delta_acc": mean(acc_deltas),
                        "worst_f1_delta": min(deltas)})
        lines += [f"## {dataset}", "", "| Arm | F1 | Accuracy |", "|---|---:|---:|"]
        lines += [f"| Frozen primary | {mean(r['f1'] for r in p):.2f} +/- {stdev(r['f1'] for r in p):.2f} | {mean(r['accuracy'] for r in p):.2f} +/- {stdev(r['accuracy'] for r in p):.2f} |",
                  f"| Feature-only Q + post-hoc text | {mean(r['f1'] for r in c):.2f} +/- {stdev(r['f1'] for r in c):.2f} | {mean(r['accuracy'] for r in c):.2f} +/- {stdev(r['accuracy'] for r in c):.2f} |", ""]
        lines += [f"- Paired F1 delta (control - primary): {mean(deltas):+.2f} pp; 95% n=3 paired t CI [{low:+.2f}, {high:+.2f}]; positive cells {sum(v > 0 for v in deltas)}/{len(deltas)}; worst {min(deltas):+.2f} pp.",
                   f"- Paired Accuracy delta: {mean(acc_deltas):+.2f} pp.", ""]
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT / "summary.json").write_text(json.dumps({"cells": rows, "summary": summary}, indent=2), encoding="utf-8")
    print(json.dumps({"status": "complete", "cells": len(rows), "output": str(OUT / "summary.md")}))


if __name__ == "__main__":
    main()
