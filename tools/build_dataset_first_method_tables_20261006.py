"""Write per-dataset comparison tables with the main method first."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/current/MULTI_DATASET_BY_DATASET_TABLES_20261006.md"

LARGE_ORDER = (
    "cat_mmd_tag_x_conditioned_routing_coherent_adaptive",
    "cat_mmd_tag_x_conditioned_routing_coherent_adaptive_anchor",
    "cat_mmd_tag_cctp_top5",
    "gecc_hard_tag",
    "k_center",
    "medoid",
    "random_real",
)
LARGE_LABELS = {
    "cat_mmd_tag_x_conditioned_routing_coherent_adaptive": "Adaptive coherent (主方法)",
    "cat_mmd_tag_x_conditioned_routing_coherent_adaptive_anchor": "Adaptive coherent + anchor",
    "cat_mmd_tag_cctp_top5": "CaT-MMD-TAG (CCTP top-5)",
    "gecc_hard_tag": "GECC-hard-TAG",
    "k_center": "K-center/FPS",
    "medoid": "Medoid coreset",
    "random_real": "Random real coreset",
}

SMALL_ORDER = ("x_conditioned_routing_coherent", "formal_prompt_margin", "cat_nearest")
SMALL_LABELS = {
    "x_conditioned_routing_coherent": "X-conditioned coherent routing (小图主线)",
    "formal_prompt_margin": "Formal prompt-margin assignment",
    "cat_nearest": "CCTP/CaT nearest payload",
}


def f(value: float, digits: int = 2) -> str:
    return f"{float(value):.{digits}f}"


def load_large() -> dict[str, list[dict]]:
    path = ROOT / "results/multidataset_f1_accuracy_comparison_20261006/dataset_method_cells.csv"
    data: dict[str, list[dict]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            data.setdefault(row["dataset"], []).append(row)
    return data


def load_small() -> dict[str, list[dict]]:
    path = ROOT / "results/small_f1_accuracy_comparison_20261006/dataset_pooled.csv"
    data: dict[str, list[dict]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            data.setdefault(row["dataset"], []).append(row)
    return data


def large_table(lines: list[str], dataset: str, rows: list[dict]) -> None:
    by_arm = {row["arm"]: row for row in rows}
    lines.extend([
        f"## {dataset}",
        "",
        "主方法固定放第一行。数值为该数据集 base+4 协议、6 个 Student seeds 的 Final Macro-F1 与 Accuracy 均值；其余指标用于观察旧类/新类权衡。",
        "",
        "| Method | Final Macro-F1 | Accuracy | Old F1 | New F1 | Stage avg F1 | Worst stage F1 | Build s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for arm in LARGE_ORDER:
        row = by_arm[arm]
        lines.append(
            f"| {LARGE_LABELS[arm]} | {f(row['final_f1'])} +/- {f(row['final_f1_std'])} | "
            f"{f(row['accuracy'])} +/- {f(row['accuracy_std'])} | {f(row['old_f1'])} | "
            f"{f(row['new_f1'])} | {f(row['stage_avg_f1'])} | {f(row['worst_stage_f1'])} | "
            f"{f(row['construction_seconds'], 1)} |"
        )
    lines.append("")


def small_table(lines: list[str], dataset: str, rows: list[dict]) -> None:
    by_arm = {row["arm"]: row for row in rows}
    lines.extend([
        f"## {dataset}",
        "",
        "小图数值为 `order19611/19612/19613` 三个顺序的等权均值，采用三阶段协议；主线版本固定放第一行。",
        "",
        "| Method | Final Macro-F1 | Accuracy | Old F1 | New F1 | Stage avg F1 | Stage avg Accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for arm in SMALL_ORDER:
        row = by_arm[arm]
        lines.append(
            f"| {SMALL_LABELS[arm]} | {f(row['final_f1'])} | {f(row['accuracy'])} | "
            f"{f(row['old_f1'])} | {f(row['new_f1'])} | {f(row['stage_avg_f1'])} | "
            f"{f(row['stage_avg_accuracy'])} |"
        )
    lines.append("")


def main() -> None:
    large = load_large()
    small = load_small()
    lines = [
        "# 按数据集整理的 F1 与 Accuracy 对比表",
        "",
        "每个数据集单独列出全部方法，第一行固定为当前主方法。大图使用 `base+4`、5% memory、6 个 Student seeds；Cora/Citeseer 使用独立的三阶段小图协议，不做跨协议 pooled。",
        "",
    ]
    for dataset in ("arxiv", "sportsfit", "wikics", "bookhis"):
        large_table(lines, dataset, large[dataset])
    for dataset in ("cora", "citeseer"):
        small_table(lines, dataset, small[dataset])
    lines.extend([
        "## 阅读说明",
        "",
        "- 大图主方法为 Adaptive coherent；Cora/Citeseer 的第一行是固定 slot 的 X-conditioned coherent routing，不能把两者名称和协议混写。",
        "- BookHis 的主方法 Accuracy 较高但 New F1 很低，需结合 New F1 和 Macro-F1 判断，不能只按 Accuracy 排名。",
        "- CaT/GECC/coreset 均为当前 continual TAG 访问约束下的机制适配基线，不等同于官方端到端复现。",
    ])
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
