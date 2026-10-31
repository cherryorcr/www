"""Build the external-baseline comparison for WikiCS and BookHis.

The numeric table only combines runs that use the same base+4 continual
protocol and the same frozen Student.  Methods that are only covered on the
small graphs are reported as coverage notes rather than silently mixed into
the large-graph table.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("wikics", "bookhis")

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
    "random_real": "strict adapted baseline",
    "k_center": "strict adapted baseline",
    "medoid": "strict adapted baseline",
    "gecc_hard_tag": "strict adapted baseline",
    "cat_mmd_tag_cctp_top5": "strict adapted baseline",
    "cat_mmd_tag_x_conditioned_routing_coherent_adaptive": "candidate method",
    "cat_mmd_tag_x_conditioned_routing_coherent_adaptive_anchor": "internal control",
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


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def metric(row: dict, key: str, fallback: str | None = None) -> float:
    value = row.get(key)
    if value is None and fallback is not None:
        value = row.get(fallback)
    if value is None:
        raise KeyError(f"missing metric {key} in {row}")
    return float(value)


def validate_protocol(payload: dict, dataset: str) -> None:
    protocol = payload.get("protocol", {})
    if payload.get("dataset") != dataset:
        raise ValueError(f"{dataset}: result dataset mismatch")
    if protocol.get("stage_protocol") != "base+4":
        raise ValueError(f"{dataset}: expected base+4")
    if abs(float(protocol.get("memory_rate", -1.0)) - 0.05) > 1e-9:
        raise ValueError(f"{dataset}: expected 5% cumulative memory")
    if protocol.get("student_seeds") != [100, 101, 102, 103, 104, 105]:
        raise ValueError(f"{dataset}: Student seeds differ from frozen protocol")
    if int(protocol.get("historical_raw_reads", 1)) != 0:
        raise ValueError(f"{dataset}: historical raw reads are not zero")
    if bool(protocol.get("report_test_labels_used_for_construction", True)):
        raise ValueError(f"{dataset}: report/test labels were used for construction")


def compare_protocols(left: dict, right: dict, dataset: str) -> None:
    """Reject a numeric merge when the stage order or budget differs."""
    for key in ("groups", "budgets", "stage_train_counts"):
        if left.get(key) != right.get(key):
            raise ValueError(f"{dataset}: protocol mismatch in {key}")


def rows_for(payload: dict, dataset: str, arms: tuple[str, ...]) -> list[dict]:
    validate_protocol(payload, dataset)
    overall = payload.get("overall", {})
    missing = [arm for arm in arms if arm not in overall]
    if missing:
        raise ValueError(f"{dataset}: missing arms {missing}")
    rows = []
    for arm in arms:
        values = overall[arm]
        rows.append({
            "dataset": dataset,
            "arm": arm,
            "method": LABELS[arm],
            "category": CATEGORIES[arm],
            "final_f1": metric(values, "final_f1"),
            "final_f1_std": metric(values, "final_f1_std"),
            "accuracy": metric(values, "final_accuracy", "accuracy"),
            "accuracy_std": metric(values, "final_accuracy_std", "accuracy_std"),
            "stage_avg_f1": metric(values, "stage_avg_f1"),
            "stage_avg_accuracy": metric(values, "stage_avg_accuracy"),
            "old_f1": metric(values, "old_f1"),
            "new_f1": metric(values, "new_f1"),
            "worst_stage_f1": metric(values, "worst_stage_f1"),
            "construction_seconds": metric(values, "construction_seconds"),
            "student_seconds": metric(values, "student_seconds"),
        })
    return rows


def fmt(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--external-root", type=Path, required=True)
    parser.add_argument("--adaptive-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    all_rows: list[dict] = []
    protocols: dict[str, dict] = {}
    for dataset in DATASETS:
        external = load(args.external_root / dataset / "results.json")
        adaptive = load(args.adaptive_root / dataset / "results.json")
        validate_protocol(external, dataset)
        validate_protocol(adaptive, dataset)
        compare_protocols(external["protocol"], adaptive["protocol"], dataset)
        protocols[dataset] = external["protocol"]
        all_rows.extend(rows_for(external, dataset, ORDER[:5]))
        all_rows.extend(rows_for(adaptive, dataset, ORDER[5:]))

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "dataset_cells.csv", all_rows)

    pooled = []
    for arm in ORDER:
        cells = [row for row in all_rows if row["arm"] == arm]
        pooled.append({
            "arm": arm,
            "method": LABELS[arm],
            "category": CATEGORIES[arm],
            "final_f1": mean(row["final_f1"] for row in cells),
            "accuracy": mean(row["accuracy"] for row in cells),
            "stage_avg_f1": mean(row["stage_avg_f1"] for row in cells),
            "stage_avg_accuracy": mean(row["stage_avg_accuracy"] for row in cells),
            "old_f1": mean(row["old_f1"] for row in cells),
            "new_f1": mean(row["new_f1"] for row in cells),
            "worst_stage_f1": mean(row["worst_stage_f1"] for row in cells),
            "construction_seconds": mean(row["construction_seconds"] for row in cells),
            "student_seconds": mean(row["student_seconds"] for row in cells),
        })
    write_csv(output / "pooled_methods.csv", pooled)

    by_key = {(row["dataset"], row["arm"]): row for row in all_rows}
    target = "cat_mmd_tag_x_conditioned_routing_coherent_adaptive"
    comparison = []
    for baseline in ORDER[:5]:
        deltas = {
            dataset: by_key[(dataset, target)]["final_f1"]
            - by_key[(dataset, baseline)]["final_f1"]
            for dataset in DATASETS
        }
        comparison.append({
            "baseline": LABELS[baseline],
            "mean_delta_f1": mean(deltas.values()),
            "wikics_delta_f1": deltas["wikics"],
            "bookhis_delta_f1": deltas["bookhis"],
            "positive_datasets": sum(value > 0 for value in deltas.values()),
        })

    payload = {
        "protocols": protocols,
        "dataset_cells": all_rows,
        "pooled_methods": pooled,
        "adaptive_coherent_final_f1_deltas": comparison,
        "coverage_notes": [
            "PUMA-inspired adaptation is covered on Cora/Citeseer only; its runner is not a WikiCS/BookHis implementation.",
            "TAGSAM, GCond, SFGC, GCondenser and STAD are not placed in this strict table because their available runners use different static/full-graph or Student contracts.",
            "CaT-MMD-TAG and GECC-hard-TAG here are mechanism adaptations under the frozen TAG continual contract, not official end-to-end reproductions.",
        ],
    }
    (output / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# WikiCS 与 BookHis 外部基线对比",
        "",
        "本表只合并通过协议审计的结果：`base+4`、5% cumulative memory、6 个相同 Student seeds、current-delta construction、frozen prefix、不能读取历史 raw graph。CaT/GECC 是机制适配版，不冒充官方端到端复现。",
        "",
        "## 1. Dataset-level results",
        "",
        "| Dataset | Method | Category | Final Macro-F1 | Accuracy | Stage avg F1 | Old F1 | New F1 | Worst stage | Build s | Student s/seed |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in all_rows:
        lines.append(
            f"| {row['dataset']} | {row['method']} | {row['category']} | "
            f"{fmt(row['final_f1'])} +/- {fmt(row['final_f1_std'])} | "
            f"{fmt(row['accuracy'])} +/- {fmt(row['accuracy_std'])} | "
            f"{fmt(row['stage_avg_f1'])} | {fmt(row['old_f1'])} | {fmt(row['new_f1'])} | "
            f"{fmt(row['worst_stage_f1'])} | {fmt(row['construction_seconds'], 1)} | {fmt(row['student_seconds'], 1)} |"
        )

    lines += [
        "",
        "## 2. Equal-weight pooled results",
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
        "## 3. Adaptive coherent 相对外部基线",
        "",
        "| Baseline | Mean Final-F1 delta | WikiCS | BookHis | Positive datasets |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in comparison:
        lines.append(
            f"| {row['baseline']} | {row['mean_delta_f1']:+.2f} | "
            f"{row['wikics_delta_f1']:+.2f} | {row['bookhis_delta_f1']:+.2f} | "
            f"{row['positive_datasets']}/{len(DATASETS)} |"
        )

    lines += [
        "",
        "## 4. Coverage and paper-use notes",
        "",
        "- PUMA-inspired baseline 当前已有 Cora/Citeseer 的严格适配结果，但 runner 只覆盖这两张小图，因此不虚构 WikiCS/BookHis 数值。",
        "- TAGSAM、GCond、SFGC、GCondenser、STAD 的可用实现与当前 continual TAG Student/访问约束不完全一致，单独列为 adapted/reference-only，不放进这张严格主表。",
        "- 这张表回答的是：在同一协议下，Adaptive coherent 是否优于合理的外部机制适配基线；不是宣称官方论文方法被原样复现。",
        "",
        "## 5. 结果解释",
        "",
        "- WikiCS：Adaptive coherent 的 Final Macro-F1 为 73.87，超过本表最强外部基线 Medoid 约 18.90 pp、CaT 约 19.57 pp；anchor 版本反而低 4.11 pp，说明收益来自无 anchor 的 routing 行为，而不是简单的 CCTP 约束。",
        "- BookHis：Adaptive coherent 的 Final Macro-F1 为 17.67，仅比本轮 CaT 高 0.32 pp，并低于 K-center 2.36 pp；Accuracy 虽为 60.18，但 New F1 只有 1.23、stage-average F1 只有 13.21，说明长尾新类上存在明显的预测塌缩或类别偏置，不能据此宣称跨数据集稳定 SOTA。",
        "- 因此 pooled 平均值 45.77 不能替代 dataset-level 结论。下一步应优先审计 BookHis 的新类 payload/预测分布，再决定是否把 Adaptive coherent 冻结为主方法。",
        "",
        "## 6. Protocol audit",
        "",
        "| Dataset | Stage protocol | Memory rate | Student seeds | Historical raw reads | Test labels for construction |",
        "|---|---|---:|---|---:|---:|",
    ]
    for dataset in DATASETS:
        protocol = protocols[dataset]
        lines.append(
            f"| {dataset} | {protocol['stage_protocol']} | {protocol['memory_rate']:.2%} | "
            f"{protocol['student_seeds']} | {protocol['historical_raw_reads']} | "
            f"{protocol['report_test_labels_used_for_construction']} |"
        )
    (ROOT / "docs/current/EXTERNAL_BASELINE_COMPARISON_WIKICS_BOOKHIS_20261006.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
