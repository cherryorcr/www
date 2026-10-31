"""Build a four-dataset F1/Accuracy comparison for the large-graph protocol."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]

DATASETS = {
    "arxiv": {
        "external": ROOT / "results/public_baselines_large_corrected_20261003_remote/arxiv/results.json",
        "adaptive": ROOT / "results/x_coherent_adaptive_slots_20261006_arxiv/arxiv_base4/results.json",
    },
    "sportsfit": {
        "external": ROOT / "results/public_baselines_large_corrected_20261003_remote/sportsfit/results.json",
        "adaptive": ROOT / "results/x_coherent_adaptive_slots_20261006_sportsfit/sportsfit_base4/results.json",
    },
    "wikics": {
        "external": ROOT / "results/external_baseline_runs_20261006/wikics/results.json",
        "adaptive": ROOT / "results/adaptive_baseline_runs_20261006/wikics/results.json",
    },
    "bookhis": {
        "external": ROOT / "results/external_baseline_runs_20261006/bookhis/results.json",
        "adaptive": ROOT / "results/adaptive_baseline_runs_20261006/bookhis/results.json",
    },
}

ORDER = (
    "random_real",
    "k_center",
    "medoid",
    "gecc_hard_tag",
    "cat_mmd_tag_cctp_top5",
    "cat_mmd_tag_x_conditioned_routing_coherent_adaptive",
    "cat_mmd_tag_x_conditioned_routing_coherent_adaptive_anchor",
)

LABELS = {
    "random_real": "Random real coreset",
    "k_center": "K-center/FPS",
    "medoid": "Medoid coreset",
    "gecc_hard_tag": "GECC-hard-TAG",
    "cat_mmd_tag_cctp_top5": "CaT-MMD-TAG (CCTP top-5)",
    "cat_mmd_tag_x_conditioned_routing_coherent_adaptive": "Adaptive coherent",
    "cat_mmd_tag_x_conditioned_routing_coherent_adaptive_anchor": "Adaptive coherent + anchor",
}

CATEGORIES = {
    **{arm: "strict adapted baseline" for arm in ORDER[:5]},
    ORDER[5]: "candidate method",
    ORDER[6]: "internal control",
}


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def value(payload: dict, key: str, fallback: str | None = None) -> float:
    result = payload.get(key)
    if result is None and fallback is not None:
        result = payload.get(fallback)
    if result is None:
        raise KeyError(f"Missing metric {key}")
    return float(result)


def validate_pair(external: dict, adaptive: dict, dataset: str) -> None:
    for payload in (external, adaptive):
        protocol = payload["protocol"]
        if protocol.get("dataset") != dataset:
            raise ValueError(f"{dataset}: dataset mismatch")
        if protocol.get("stage_protocol") != "base+4":
            raise ValueError(f"{dataset}: not base+4")
        if abs(float(protocol.get("memory_rate", -1)) - 0.05) > 1e-9:
            raise ValueError(f"{dataset}: not 5% memory")
        if protocol.get("student_seeds") != [100, 101, 102, 103, 104, 105]:
            raise ValueError(f"{dataset}: Student seed mismatch")
        if int(protocol.get("historical_raw_reads", 1)) != 0:
            raise ValueError(f"{dataset}: historical raw reads are nonzero")
        if protocol.get("report_test_labels_used_for_construction", True):
            raise ValueError(f"{dataset}: construction used report/test labels")
    for key in ("groups", "budgets"):
        if external["protocol"].get(key) != adaptive["protocol"].get(key):
            raise ValueError(f"{dataset}: external/adaptive protocol mismatch: {key}")


def rows_from(payload: dict, dataset: str, arms: tuple[str, ...]) -> list[dict]:
    overall = payload.get("overall", {})
    missing = [arm for arm in arms if arm not in overall]
    if missing:
        raise ValueError(f"{dataset}: missing arms {missing}")
    rows = []
    for arm in arms:
        metrics = overall[arm]
        rows.append({
            "dataset": dataset,
            "arm": arm,
            "method": LABELS[arm],
            "category": CATEGORIES[arm],
            "final_f1": value(metrics, "final_f1"),
            "final_f1_std": value(metrics, "final_f1_std"),
            "accuracy": value(metrics, "final_accuracy", "accuracy"),
            "accuracy_std": value(metrics, "final_accuracy_std", "accuracy_std"),
            "stage_avg_f1": value(metrics, "stage_avg_f1"),
            "stage_avg_accuracy": value(metrics, "stage_avg_accuracy"),
            "old_f1": value(metrics, "old_f1"),
            "new_f1": value(metrics, "new_f1"),
            "worst_stage_f1": value(metrics, "worst_stage_f1"),
            "construction_seconds": value(metrics, "construction_seconds"),
            "student_seconds": value(metrics, "student_seconds"),
        })
    return rows


def fmt(x: float, digits: int = 2) -> str:
    return f"{x:.{digits}f}"


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "results/multidataset_f1_accuracy_comparison_20261006")
    args = parser.parse_args()

    cells: list[dict] = []
    protocols: dict[str, dict] = {}
    for dataset, paths in DATASETS.items():
        external = load(paths["external"])
        adaptive = load(paths["adaptive"])
        validate_pair(external, adaptive, dataset)
        protocols[dataset] = external["protocol"]
        cells.extend(rows_from(external, dataset, ORDER[:5]))
        cells.extend(rows_from(adaptive, dataset, ORDER[5:]))

    pooled = []
    for arm in ORDER:
        selected = [row for row in cells if row["arm"] == arm]
        pooled.append({
            "arm": arm,
            "method": LABELS[arm],
            "category": CATEGORIES[arm],
            "final_f1": mean(row["final_f1"] for row in selected),
            "accuracy": mean(row["accuracy"] for row in selected),
            "stage_avg_f1": mean(row["stage_avg_f1"] for row in selected),
            "stage_avg_accuracy": mean(row["stage_avg_accuracy"] for row in selected),
            "old_f1": mean(row["old_f1"] for row in selected),
            "new_f1": mean(row["new_f1"] for row in selected),
            "worst_stage_f1": mean(row["worst_stage_f1"] for row in selected),
            "construction_seconds": mean(row["construction_seconds"] for row in selected),
            "student_seconds": mean(row["student_seconds"] for row in selected),
        })

    index = {(row["dataset"], row["arm"]): row for row in cells}
    target = ORDER[5]
    deltas = []
    for dataset in DATASETS:
        external = [index[(dataset, arm)] for arm in ORDER[:5]]
        best_f1 = max(external, key=lambda row: row["final_f1"])
        best_acc = max(external, key=lambda row: row["accuracy"])
        candidate = index[(dataset, target)]
        deltas.append({
            "dataset": dataset,
            "candidate_f1": candidate["final_f1"],
            "candidate_accuracy": candidate["accuracy"],
            "best_external_f1_method": best_f1["method"],
            "best_external_f1": best_f1["final_f1"],
            "delta_f1": candidate["final_f1"] - best_f1["final_f1"],
            "best_external_accuracy_method": best_acc["method"],
            "best_external_accuracy": best_acc["accuracy"],
            "delta_accuracy": candidate["accuracy"] - best_acc["accuracy"],
        })

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "dataset_method_cells.csv", cells)
    write_csv(output / "pooled_methods.csv", pooled)
    write_csv(output / "candidate_vs_best_external.csv", deltas)
    payload = {"protocols": protocols, "cells": cells, "pooled": pooled, "candidate_vs_best_external": deltas}
    (output / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# 四个大图上的 F1 与 Accuracy 跨数据集比较",
        "",
        "四个数据集均使用 `base+4`、5% cumulative memory、6 个 Student seeds、同一 edge-aware 2-layer GCN + text projection + bidirectional InfoNCE Student。每个数据集的类别顺序和预算按其冻结 manifest 执行，因此跨数据集比较用于观察泛化，不把不同数据集的绝对难度混为同一任务。",
        "",
        "## 1. Dataset-level comparison",
        "",
        "| Dataset | Method | Category | Final Macro-F1 | Accuracy | Stage avg F1 | Old F1 | New F1 | Worst stage | Build s | Student s/seed |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in cells:
        lines.append(
            f"| {row['dataset']} | {row['method']} | {row['category']} | "
            f"{fmt(row['final_f1'])} +/- {fmt(row['final_f1_std'])} | "
            f"{fmt(row['accuracy'])} +/- {fmt(row['accuracy_std'])} | "
            f"{fmt(row['stage_avg_f1'])} | {fmt(row['old_f1'])} | {fmt(row['new_f1'])} | "
            f"{fmt(row['worst_stage_f1'])} | {fmt(row['construction_seconds'], 1)} | {fmt(row['student_seconds'], 1)} |"
        )

    lines += [
        "",
        "## 2. Equal-weight pooled comparison",
        "",
        "| Method | Category | Final Macro-F1 | Accuracy | Stage avg F1 | Old F1 | New F1 | Worst stage | Build s | Student s/seed |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in pooled:
        lines.append(
            f"| {row['method']} | {row['category']} | {fmt(row['final_f1'])} | {fmt(row['accuracy'])} | "
            f"{fmt(row['stage_avg_f1'])} | {fmt(row['old_f1'])} | {fmt(row['new_f1'])} | "
            f"{fmt(row['worst_stage_f1'])} | {fmt(row['construction_seconds'], 1)} | {fmt(row['student_seconds'], 1)} |"
        )

    lines += [
        "",
        "## 3. Adaptive coherent 相对最强外部基线",
        "",
        "| Dataset | Adaptive F1 | Best external F1 method | Best external F1 | Delta F1 | Adaptive Accuracy | Best external Accuracy method | Best external Accuracy | Delta Accuracy |",
        "|---|---:|---|---:|---:|---:|---|---:|---:|",
    ]
    for row in deltas:
        lines.append(
            f"| {row['dataset']} | {fmt(row['candidate_f1'])} | {row['best_external_f1_method']} | "
            f"{fmt(row['best_external_f1'])} | {row['delta_f1']:+.2f} | {fmt(row['candidate_accuracy'])} | "
            f"{row['best_external_accuracy_method']} | {fmt(row['best_external_accuracy'])} | {row['delta_accuracy']:+.2f} |"
        )

    lines += [
        "",
        "## 4. Interpretation",
        "",
        "- WikiCS、Arxiv、SportsFit 上 Adaptive coherent 的 Final Macro-F1 明显高于本轮外部基线，且 Accuracy 同步提升。",
        "- BookHis 上 Adaptive coherent 的 F1 仍低于 K-center，虽然 Accuracy 很高，但 New F1 很低，说明长尾新类存在类别偏置；不能只看 pooled 平均值。",
        "- Anchor 版本作为内部控制单独报告，不把它当作外部基线；它用于判断 payload anchor 是否真正改善泛化。",
        "- Cora/Citeseer 使用三阶段小图协议，另行报告，不与这四个大图的 base+4 数值做 pooled 混合；详见 `SMALL_DATASET_F1_ACCURACY_COMPARISON_20261006.md`。",
        "",
        "## 5. Protocol audit",
        "",
        "| Dataset | Stage protocol | Memory rate | Student seeds | Historical raw reads | Test labels used for construction |",
        "|---|---|---:|---|---:|---:|",
    ]
    for dataset, protocol in protocols.items():
        lines.append(
            f"| {dataset} | {protocol['stage_protocol']} | {protocol['memory_rate']:.2%} | "
            f"{protocol['student_seeds']} | {protocol['historical_raw_reads']} | "
            f"{protocol['report_test_labels_used_for_construction']} |"
        )

    (ROOT / "docs/current/MULTI_DATASET_F1_ACCURACY_COMPARISON_20261006.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
