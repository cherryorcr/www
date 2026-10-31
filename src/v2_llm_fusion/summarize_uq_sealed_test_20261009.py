"""Aggregate sealed-test readouts from one or more evaluator shards."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ARMS = ("uq_qmedoid", "uq_dm_edgefree", "cat_cctp", "gecc_hard")
ARM_LABELS = {
    "uq_qmedoid": "Q-medoid",
    "uq_dm_edgefree": "In-loop CaT-Q",
    "cat_cctp": "CaT + nearest5",
    "gecc_hard": "GECC-hard + TAG",
}


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def merge(paths: list[Path]) -> dict:
    payloads = [read(path) for path in paths]
    if not payloads:
        raise ValueError("no sealed result files")
    config = dict(payloads[0]["config"])
    comparable = {k: v for k, v in config.items()
                  if k not in {"source_roots", "cell_count", "completed_rows"}}
    rows, cells = [], []
    row_keys, cell_keys = set(), set()
    for payload in payloads:
        other = {k: v for k, v in payload["config"].items()
                 if k not in {"source_roots", "cell_count", "completed_rows"}}
        if other != comparable:
            raise ValueError("sealed shards have different evaluator configs")
        for cell in payload.get("cells", []):
            key = (cell["dataset"], cell["order_seed"], cell["construction_seed"])
            if key not in cell_keys:
                cell_keys.add(key)
                cells.append(cell)
        for row in payload.get("rows", []):
            key = (row["dataset"], row["order_seed"], row["construction_seed"],
                   row["arm"], row["student"])
            if key in row_keys:
                raise ValueError(f"duplicate sealed row {key}")
            row_keys.add(key)
            rows.append(row)
    rows.sort(key=lambda row: (row["dataset"], row["order_seed"],
                               row["construction_seed"], ARMS.index(row["arm"]),
                               row["student"]))
    cells.sort(key=lambda cell: (cell["dataset"], cell["order_seed"],
                                 cell["construction_seed"]))
    config["source_roots"] = sorted({str(v) for payload in payloads
                                      for v in payload["config"].get("source_roots", [])})
    config["cell_count"] = len(cells)
    return {"config": config, "cells": cells, "rows": rows,
            "status": "complete" if len(rows) == len(cells) * len(ARMS) * 3 else "incomplete"}


def aggregate(rows: list[dict], by_dataset: bool = True) -> list[dict]:
    buckets = defaultdict(list)
    for row in rows:
        score = row["score"]
        key = ((row["dataset"] if by_dataset else "all_datasets"),
               row["student"], row["arm"])
        buckets[key].append(score)
    output = []
    for (dataset, student, arm), scores in sorted(buckets.items()):
        output.append({
            "dataset": dataset, "student": student, "arm": arm,
            "cells": len(scores),
            "final_f1": float(np.mean([v["final_f1"] for v in scores])),
            "final_f1_std": float(np.std([v["final_f1"] for v in scores], ddof=1)),
            "accuracy": float(np.mean([v["accuracy"] for v in scores])),
            "accuracy_std": float(np.std([v["accuracy"] for v in scores], ddof=1)),
            "old_f1": float(np.mean([v["old_f1"] for v in scores])),
            "new_f1": float(np.mean([v["new_f1"] for v in scores])),
        })
    return output


def paired(rows: list[dict], baseline: str) -> list[dict]:
    lookup = {(r["dataset"], r["order_seed"], r["construction_seed"],
               r["student"], r["arm"]): r["score"] for r in rows}
    groups = defaultdict(list)
    for row in rows:
        if row["arm"] == baseline:
            continue
        key = (row["dataset"], row["order_seed"], row["construction_seed"],
               row["student"])
        current = row["score"]
        ref = lookup.get((*key, baseline))
        if ref is not None:
            groups[(row["dataset"], row["student"], row["arm"])].append(
                (current["final_f1"] - ref["final_f1"],
                 current["accuracy"] - ref["accuracy"]))
    output = []
    for (dataset, student, arm), values in sorted(groups.items()):
        f1 = np.asarray([v[0] for v in values])
        acc = np.asarray([v[1] for v in values])
        output.append({
            "dataset": dataset, "student": student, "arm": arm,
            "baseline": baseline, "cells": len(values),
            "delta_f1": float(f1.mean()), "delta_f1_std": float(f1.std(ddof=1)),
            "delta_accuracy": float(acc.mean()),
            "delta_accuracy_std": float(acc.std(ddof=1)),
            "positive_f1_cells": int((f1 > 0).sum()),
            "positive_accuracy_cells": int((acc > 0).sum()),
        })
    return output


def fmt(mean: float, std: float) -> str:
    return f"{mean:.2f} ± {std:.2f}"


def build_markdown(merged: dict, summary: list[dict], diffs: list[dict]) -> str:
    lines = [
        "# Unified-Q Sealed-Test Results",
        "",
        "本表只统计 `original_id % 10 == 2` 的 sealed test；构造阶段未读取 test，memory 直接复用已冻结 archive。每个 cell 使用三个 Student seeds，并按 27 个 construction cells 汇总。",
        "",
        f"- Cells: `{len(merged['cells'])}`",
        f"- Readout rows: `{len(merged['rows'])}`",
        "- Arms: `uq_qmedoid`, `uq_dm_edgefree`, `cat_cctp`, `gecc_hard`",
        "- Students: `prompt_gcn`, `prompt_sgc`, `prompt_sage`",
        "",
        "## Final Macro-F1 / Accuracy",
        "",
    ]
    for dataset in sorted({r["dataset"] for r in summary}):
        lines += [f"### {dataset}", "",
                  "| Student | Method | Final Macro-F1 | Accuracy | Old F1 | New F1 | Cells |",
                  "|---|---|---:|---:|---:|---:|---:|"]
        for row in [r for r in summary if r["dataset"] == dataset]:
            lines.append("| " + " | ".join([
                row["student"], ARM_LABELS[row["arm"]],
                fmt(row["final_f1"], row["final_f1_std"]),
                fmt(row["accuracy"], row["accuracy_std"]),
                f"{row['old_f1']:.2f}", f"{row['new_f1']:.2f}", str(row["cells"]),
            ]) + " |")
        lines.append("")
    pooled = merged["pooled_summary"]
    lines += ["## All-dataset pooled summary", "",
              "| Student | Method | Final Macro-F1 | Accuracy | Old F1 | New F1 | Cells |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for row in pooled:
        lines.append("| " + " | ".join([
            row["student"], ARM_LABELS[row["arm"]],
            fmt(row["final_f1"], row["final_f1_std"]),
            fmt(row["accuracy"], row["accuracy_std"]),
            f"{row['old_f1']:.2f}", f"{row['new_f1']:.2f}", str(row["cells"]),
        ]) + " |")
    lines.append("")
    lines += ["## Paired Differences", "",
              "正值表示当前 arm 高于同一 dataset/order/seed/Student 下的 baseline。",
              "",
              "| Dataset | Student | Method | Baseline | Δ F1 | Δ Acc | Positive F1 cells | Positive Acc cells |",
              "|---|---|---|---|---:|---:|---:|---:|"]
    for row in diffs:
        lines.append("| " + " | ".join([
            row["dataset"], row["student"], ARM_LABELS[row["arm"]],
            ARM_LABELS[row["baseline"]],
            f"{row['delta_f1']:.2f} ± {row['delta_f1_std']:.2f}",
            f"{row['delta_accuracy']:.2f} ± {row['delta_accuracy_std']:.2f}",
            f"{row['positive_f1_cells']}/{row['cells']}",
            f"{row['positive_accuracy_cells']}/{row['cells']}",
        ]) + " |")
    lines += ["", "## Interpretation Boundary", "",
              "sealed test 只回答冻结方法在未参与构造/选择的数据上的泛化；它不把 CaT/GECC 适配结果变成官方 native reproduction，也不替代外部 baseline fidelity audit。",
              ""]
    return "\n".join(lines)


def plot_metric(summary: list[dict], metric: str, path: Path) -> None:
    datasets = sorted({r["dataset"] for r in summary})
    students = sorted({r["student"] for r in summary})
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.2 * len(datasets), 4.5), squeeze=False)
    for index, dataset in enumerate(datasets):
        ax = axes[0, index]
        x = np.arange(len(students))
        width = 0.18
        for arm_index, arm in enumerate(ARMS):
            values = [next(r[metric] for r in summary
                           if r["dataset"] == dataset and r["student"] == student
                           and r["arm"] == arm) for student in students]
            ax.bar(x + (arm_index - 1.5) * width, values, width,
                   label=ARM_LABELS[arm])
        ax.set_title(dataset)
        ax.set_xticks(x, students, rotation=20)
        ax.set_ylabel(metric.replace("_", " "))
        ax.grid(axis="y", alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    merged = merge(args.input)
    summary = aggregate(merged["rows"])
    pooled = aggregate(merged["rows"], by_dataset=False)
    diffs = paired(merged["rows"], "cat_cctp") + paired(merged["rows"], "gecc_hard")
    args.output.mkdir(parents=True, exist_ok=True)
    merged["summary"] = summary
    merged["pooled_summary"] = pooled
    merged["paired_differences"] = diffs
    (args.output / "summary.json").write_text(
        json.dumps(merged, indent=2, ensure_ascii=True), encoding="utf-8")
    (args.output / "summary.md").write_text(
        build_markdown(merged, summary, diffs), encoding="utf-8")
    audit = {
        "status": merged["status"],
        "expected_cells": 27,
        "completed_cells": len(merged["cells"]),
        "expected_readout_rows": 324,
        "completed_readout_rows": len(merged["rows"]),
        "unique_cell_keys": len({(r["dataset"], r["order_seed"], r["construction_seed"])
                                  for r in merged["cells"]}),
        "unique_readout_keys": len({(r["dataset"], r["order_seed"],
                                      r["construction_seed"], r["arm"], r["student"])
                                     for r in merged["rows"]}),
        "arms": list(ARMS),
        "students": ["prompt_gcn", "prompt_sgc", "prompt_sage"],
        "split": "sealed test: original_id % 10 == 2",
        "construction_reused": True,
        "test_used_for_construction": False,
        "duplicates": False,
    }
    (args.output / "audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=True), encoding="utf-8")
    plot_metric(summary, "final_f1", args.output / "final_f1.png")
    plot_metric(summary, "accuracy", args.output / "accuracy.png")
    print(json.dumps({"status": merged["status"], "cells": len(merged["cells"]),
                      "rows": len(merged["rows"]), "output": str(args.output.resolve())}))


if __name__ == "__main__":
    main()
