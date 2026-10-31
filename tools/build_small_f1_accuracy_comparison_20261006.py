"""Summarize the three-order small-graph F1/Accuracy confirmation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("cora", "citeseer")
ORDERS = ("order19611", "order19612", "order19613")
ARMS = ("cat_nearest", "formal_prompt_margin", "x_conditioned_routing_coherent")
LABELS = {
    "cat_nearest": "CCTP/CaT nearest payload",
    "formal_prompt_margin": "Formal prompt-margin assignment",
    "x_conditioned_routing_coherent": "X-conditioned coherent routing",
}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_one(dataset: str, order: str) -> dict:
    path = ROOT / f"results/x_coherent_routing_sixseed_20261006_{dataset}/{dataset}/{order}/results.json"
    payload = load(path)
    protocol = payload["protocol"]
    if protocol.get("stage_protocol") != "three-stage class-incremental":
        raise ValueError(f"{dataset}/{order}: protocol mismatch")
    if abs(float(protocol.get("memory_rate", -1)) - 0.05) > 1e-9:
        raise ValueError(f"{dataset}/{order}: memory rate mismatch")
    if not protocol.get("current_delta_only") or int(protocol.get("historical_raw_reads", 1)) != 0:
        raise ValueError(f"{dataset}/{order}: continual access mismatch")
    if protocol.get("student_seeds") != [150, 151, 152, 153, 154, 155]:
        raise ValueError(f"{dataset}/{order}: Student seed mismatch")
    result = {"dataset": dataset, "order": order}
    for arm in ARMS:
        rows = [row for row in payload["rows"] if row["arm"] == arm]
        final = max(rows, key=lambda row: int(row["stage"]))
        aggregate = final["aggregate"]
        result[arm] = {
            "final_f1": float(aggregate["mean_f1"]),
            "final_f1_std": float(aggregate["std_f1"]),
            "accuracy": float(aggregate["mean_accuracy"]),
            "accuracy_std": float(aggregate["std_accuracy"]),
            "old_f1": float(aggregate["old_f1"]),
            "new_f1": float(aggregate["new_f1"]),
            "stage_avg_f1": float(aggregate["stage_avg_f1"]),
            "stage_avg_accuracy": float(aggregate["stage_avg_accuracy"]),
        }
    return result


def fmt(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def main() -> None:
    cells = [read_one(dataset, order) for dataset in DATASETS for order in ORDERS]
    output = ROOT / "results/small_f1_accuracy_comparison_20261006"
    output.mkdir(parents=True, exist_ok=True)

    csv_rows = []
    for cell in cells:
        for arm in ARMS:
            csv_rows.append({"dataset": cell["dataset"], "order": cell["order"], "arm": arm, **cell[arm]})
    with (output / "order_cells.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)

    pooled = []
    for dataset in DATASETS:
        for arm in ARMS:
            rows = [cell[arm] for cell in cells if cell["dataset"] == dataset]
            pooled.append({
                "dataset": dataset,
                "arm": arm,
                "method": LABELS[arm],
                "final_f1": mean(row["final_f1"] for row in rows),
                "accuracy": mean(row["accuracy"] for row in rows),
                "old_f1": mean(row["old_f1"] for row in rows),
                "new_f1": mean(row["new_f1"] for row in rows),
                "stage_avg_f1": mean(row["stage_avg_f1"] for row in rows),
                "stage_avg_accuracy": mean(row["stage_avg_accuracy"] for row in rows),
            })
    with (output / "dataset_pooled.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pooled[0]))
        writer.writeheader()
        writer.writerows(pooled)

    lines = [
        "# Cora/Citeseer 小图多顺序 F1 与 Accuracy 比较",
        "",
        "本表单独采用小图三阶段协议：5% memory、`order19611/19612/19613`、6 个 Student seeds、current-delta-only、edge-free memory + Student self-loop。它不与大图 `base+4` 结果做 pooled 混合。",
        "",
        "## 1. Per-order final results",
        "",
        "| Dataset | Order | Method | Final Macro-F1 | Accuracy | Old F1 | New F1 | Stage avg F1 | Stage avg Accuracy |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for cell in cells:
        for arm in ARMS:
            row = cell[arm]
            lines.append(
                f"| {cell['dataset']} | {cell['order']} | {LABELS[arm]} | "
                f"{fmt(row['final_f1'])} +/- {fmt(row['final_f1_std'])} | "
                f"{fmt(row['accuracy'])} +/- {fmt(row['accuracy_std'])} | "
                f"{fmt(row['old_f1'])} | {fmt(row['new_f1'])} | "
                f"{fmt(row['stage_avg_f1'])} | {fmt(row['stage_avg_accuracy'])} |"
            )

    lines += [
        "",
        "## 2. Equal-weight mean over the three orders",
        "",
        "| Dataset | Method | Final Macro-F1 | Accuracy | Old F1 | New F1 | Stage avg F1 | Stage avg Accuracy |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in pooled:
        lines.append(
            f"| {row['dataset']} | {row['method']} | {fmt(row['final_f1'])} | {fmt(row['accuracy'])} | "
            f"{fmt(row['old_f1'])} | {fmt(row['new_f1'])} | {fmt(row['stage_avg_f1'])} | {fmt(row['stage_avg_accuracy'])} |"
        )
    lines += [
        "",
        "## 3. Interpretation",
        "",
        "- 这张表用于确认小图上的顺序稳定性和 F1/Accuracy 是否一致，不把单个 order 的最好值当作结论。",
        "- `X-conditioned coherent routing` 是小图固定 slot 版本；大图主表中的 `Adaptive coherent` 使用自适应 slot，名称和协议应区分。",
        "- Cora/Citeseer 的公开 CaT、GECC、coreset 适配结果另有 baseline card 和 external adapter 结果；本表先聚焦当前 routing 线的三种构造对照。",
    ]
    (ROOT / "docs/current/SMALL_DATASET_F1_ACCURACY_COMPARISON_20261006.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    (output / "summary.json").write_text(json.dumps({"cells": cells, "pooled": pooled}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
