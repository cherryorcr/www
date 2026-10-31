"""Pool the four-order Bounded X+P small-graph evidence.

The first three orders are stored in the existing three-order supplement;
order19614 is stored in the blind-order paired comparison.  This script only
normalizes and aggregates those frozen artifacts.  It does not rerun training
or select any configuration using the reported scores.
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "results" / "bounded_xp_final_evidence_20261001"
OUT_JSON = EVIDENCE / "four_order_pooled_summary.json"
OUT_CELLS = EVIDENCE / "four_order_cells.csv"
OUT_POOLED = EVIDENCE / "four_order_pooled.csv"
OUT_DELTAS = EVIDENCE / "four_order_pairwise_deltas.csv"
OUT_DELTA_SUMMARY = EVIDENCE / "four_order_delta_summary.csv"
DOC = ROOT / "docs" / "current" / "BOUNDED_XP_FOUR_ORDER_POOLED_SUMMARY_ZH_2026-10-01.md"

DATASETS = ("cora", "citeseer")
ORDERS = ("order19611", "order19612", "order19613", "order19614")
METHODS = (
    "Bounded X+P",
    "CaT-MMD-TAG adapted",
    "GECC-hard-TAG adapted",
    "k-center/FPS",
    "medoid",
)
BASELINES = METHODS[1:]


def read_json(path: Path) -> dict:
    # Python's json decoder accepts the NaN values present in some audit fields.
    return json.loads(path.read_text(encoding="utf-8"))


def finite(values: list[float | None]) -> list[float]:
    return [float(v) for v in values if v is not None and math.isfinite(float(v))]


def avg(values: list[float | None]) -> float | None:
    values = finite(values)
    return mean(values) if values else None


def sd(values: list[float | None]) -> float | None:
    values = finite(values)
    if not values:
        return None
    return stdev(values) if len(values) > 1 else 0.0


def normalize_three_order() -> list[dict]:
    payload = read_json(EVIDENCE / "three_order_supplement_and_cost.json")
    mapping = {
        "Bounded X+P": "Bounded X+P",
        "CaT-MMD-TAG adapted": "CaT-MMD-TAG adapted",
        "GECC-hard-TAG adapted": "GECC-hard-TAG adapted",
        "k-center": "k-center/FPS",
        "medoid": "medoid",
    }
    rows = []
    for row in payload["three_order_rows"]:
        method = mapping[row["method"]]
        rows.append({
            "dataset": row["dataset"],
            "order": row["order"],
            "method": method,
            "final_f1": float(row["final_f1"]),
            "final_accuracy": float(row["final_accuracy"]),
            "stage_average_f1": float(row["stage_average_f1"]),
            "old_f1": None if row.get("old_f1") is None else float(row["old_f1"]),
            "new_f1": None if row.get("new_f1") is None else float(row["new_f1"]),
            "source": "three_order_supplement_and_cost.json",
        })
    return rows


def normalize_order19614() -> list[dict]:
    payload = read_json(EVIDENCE / "order19614_xp_baseline_comparison.json")
    mapping = {
        "bounded_xp": "Bounded X+P",
        "cat_mmd_tag": "CaT-MMD-TAG adapted",
        "gecc_hard_tag": "GECC-hard-TAG adapted",
        # The blind-order report contains both FPS and native aliases.  The
        # frozen pooled table uses the FPS row and the non-native medoid row,
        # matching the values used in the report's headline comparison.
        "k_center_fps": "k-center/FPS",
        "medoid": "medoid",
    }
    rows = []
    for row in payload["rows"]:
        if row["method"] not in mapping:
            continue
        rows.append({
            "dataset": row["dataset"],
            "order": "order19614",
            "method": mapping[row["method"]],
            "final_f1": float(row["final_f1"]),
            "final_accuracy": float(row["final_accuracy"]),
            "stage_average_f1": float(row["stage_average_f1"]),
            "old_f1": None if row.get("old_f1") is None else float(row["old_f1"]),
            "new_f1": None if row.get("new_f1") is None else float(row["new_f1"]),
            "source": "order19614_xp_baseline_comparison.json",
        })
    return rows


def validate_cells(rows: list[dict]) -> None:
    expected = {(dataset, order, method)
                for dataset in DATASETS
                for order in ORDERS
                for method in METHODS}
    actual = {(row["dataset"], row["order"], row["method"]) for row in rows}
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise ValueError(f"cell mismatch; missing={missing}, extra={extra}")
    if len(rows) != len(expected):
        raise ValueError(f"expected {len(expected)} rows, got {len(rows)}")


def pooled_rows(rows: list[dict], group_keys: tuple[str, ...]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in group_keys)].append(row)
    output = []
    for key, values in sorted(groups.items()):
        item = dict(zip(group_keys, key))
        item.update({
            "cells": len(values),
            "final_f1": avg([v["final_f1"] for v in values]),
            "final_f1_std": sd([v["final_f1"] for v in values]),
            "final_accuracy": avg([v["final_accuracy"] for v in values]),
            "final_accuracy_std": sd([v["final_accuracy"] for v in values]),
            "stage_average_f1": avg([v["stage_average_f1"] for v in values]),
            "old_f1": avg([v["old_f1"] for v in values]),
            "new_f1": avg([v["new_f1"] for v in values]),
        })
        output.append(item)
    return output


def delta_rows(rows: list[dict]) -> list[dict]:
    indexed = {(r["dataset"], r["order"], r["method"]): r for r in rows}
    output = []
    for dataset in DATASETS:
        for order in ORDERS:
            xp = indexed[(dataset, order, "Bounded X+P")]
            for baseline in BASELINES:
                ref = indexed[(dataset, order, baseline)]
                output.append({
                    "dataset": dataset,
                    "order": order,
                    "baseline": baseline,
                    "xp_minus_baseline_final_f1": xp["final_f1"] - ref["final_f1"],
                    "xp_minus_baseline_accuracy": xp["final_accuracy"] - ref["final_accuracy"],
                    "xp_minus_baseline_stage_average_f1": xp["stage_average_f1"] - ref["stage_average_f1"],
                })
    return output


def delta_summary(deltas: list[dict]) -> list[dict]:
    output = []
    for dataset in DATASETS:
        for baseline in BASELINES:
            values = [r for r in deltas if r["dataset"] == dataset and r["baseline"] == baseline]
            f1 = [r["xp_minus_baseline_final_f1"] for r in values]
            acc = [r["xp_minus_baseline_accuracy"] for r in values]
            output.append({
                "dataset": dataset,
                "baseline": baseline,
                "cells": len(values),
                "positive_final_f1_cells": sum(v > 0 for v in f1),
                "positive_final_f1_ratio": sum(v > 0 for v in f1) / len(f1),
                "mean_final_f1_delta": avg(f1),
                "std_final_f1_delta": sd(f1),
                "worst_final_f1_delta": min(f1),
                "best_final_f1_delta": max(f1),
                "mean_accuracy_delta": avg(acc),
                "positive_accuracy_cells": sum(v > 0 for v in acc),
            })
    for baseline in BASELINES:
        values = [r for r in deltas if r["baseline"] == baseline]
        f1 = [r["xp_minus_baseline_final_f1"] for r in values]
        acc = [r["xp_minus_baseline_accuracy"] for r in values]
        output.append({
            "dataset": "all",
            "baseline": baseline,
            "cells": len(values),
            "positive_final_f1_cells": sum(v > 0 for v in f1),
            "positive_final_f1_ratio": sum(v > 0 for v in f1) / len(f1),
            "mean_final_f1_delta": avg(f1),
            "std_final_f1_delta": sd(f1),
            "worst_final_f1_delta": min(f1),
            "best_final_f1_delta": max(f1),
            "mean_accuracy_delta": avg(acc),
            "positive_accuracy_cells": sum(v > 0 for v in acc),
        })
    return output


def fmt(value: float | None, digits: int = 2) -> str:
    if value is None or not math.isfinite(float(value)):
        return "N/A"
    return f"{float(value):.{digits}f}"


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_doc(payload: dict) -> str:
    pooled = payload["dataset_pooled"]
    overall = payload["overall_pooled"]
    summaries = payload["delta_summary"]
    cells = payload["cells"]
    lines = [
        "# Bounded X+P：四个独立类别顺序 pooled 汇总",
        "",
        "> 本报告只汇总已经完成的冻结结果，不重新训练、不根据新结果调参。order19611–19613 来自三顺序补充证据，order19614 是额外的盲顺序比较。",
        "",
        "## 1. 统一协议",
        "",
        "- 数据集：Cora、Citeseer。",
        "- 类别顺序：`order19611`、`order19612`、`order19613`、`order19614`；每个顺序三阶段。",
        "- memory：同一 5% cumulative budget。",
        "- Student：原始两层 GCN、可训练文本投影、双向图文 InfoNCE；每个 cell 使用相同 report seeds。",
        "- X+P：`plan_seed=9320`、inner steps=8、outer steps=4、outer lr=0.02，沿用 C0 edge-aware topology；不启用 A 拓扑 revision。",
        "- CaT/GECC：本项目 continual Student 接口下的机制适配，不是官方端到端复现；k-center/FPS 与 medoid 是选择型控制方法。",
        "- SR、X+P+A、C0 不放入这张公开基线 pooled 主表。C0 仍作为内部 anchor 单独记录。",
        "",
        "## 2. 分数据集、四顺序汇总",
        "",
        "数值为四个顺序的均值 ± 跨顺序样本标准差；`Final F1` 和 `Accuracy` 是最终阶段，`Stage-average F1` 是三阶段等权平均。",
        "",
        "| 数据集 | 方法 | Final F1 | Accuracy | Stage-average F1 | Old F1 | New F1 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        for row in [r for r in pooled if r["dataset"] == dataset]:
            lines.append(
                f"| {dataset} | {row['method']} | {fmt(row['final_f1'])} ± {fmt(row['final_f1_std'])} | "
                f"{fmt(row['final_accuracy'])} ± {fmt(row['final_accuracy_std'])} | "
                f"{fmt(row['stage_average_f1'])} | {fmt(row['old_f1'])} | {fmt(row['new_f1'])} |"
            )
    lines += [
        "",
        "## 3. 每个顺序的 Final F1",
        "",
        "| 数据集 | 顺序 | Bounded X+P | CaT-MMD-TAG | GECC-hard-TAG | k-center/FPS | medoid |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    by_cell = {(r["dataset"], r["order"], r["method"]): r for r in cells}
    for dataset in DATASETS:
        for order in ORDERS:
            vals = [fmt(by_cell[(dataset, order, method)]["final_f1"]) for method in METHODS]
            lines.append(f"| {dataset} | {order} | " + " | ".join(vals) + " |")
    lines += [
        "",
        "## 4. X+P 相对基线的配对差值",
        "",
        "正值表示 X+P 的最终阶段 F1 更高。`positive cells` 是同一数据集的四个顺序中 X+P 胜出的数量。",
        "",
        "| 数据集 | 基线 | 平均 ΔFinal F1 | 正向 cell | 最差 Δ | 最好 Δ | 平均 ΔAccuracy |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        if row["dataset"] == "all":
            continue
        lines.append(
            f"| {row['dataset']} | {row['baseline']} | {fmt(row['mean_final_f1_delta']):>6} | "
            f"{row['positive_final_f1_cells']}/{row['cells']} | {fmt(row['worst_final_f1_delta']):>6} | "
            f"{fmt(row['best_final_f1_delta']):>6} | {fmt(row['mean_accuracy_delta']):>6} |"
        )
    lines += [
        "",
        "### 全部数据集与顺序合并",
        "",
        "| 基线 | 平均 ΔFinal F1 | 正向 cell | 最差 Δ | 最好 Δ | 平均 ΔAccuracy |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        if row["dataset"] != "all":
            continue
        lines.append(
            f"| {row['baseline']} | {fmt(row['mean_final_f1_delta'])} | "
            f"{row['positive_final_f1_cells']}/{row['cells']} | {fmt(row['worst_final_f1_delta'])} | "
            f"{fmt(row['best_final_f1_delta'])} | {fmt(row['mean_accuracy_delta'])} |"
        )
    lines += [
        "",
        "## 5. C0 anchor（单独记录）",
        "",
        "order19614 上的 C0 只用于检查 outer optimization 的增量，不能代替四顺序公开基线 pooled：",
        "",
        "| 数据集 | X+P Final F1 | C0 Final F1 | X+P - C0 |",
        "|---|---:|---:|---:|",
    ]
    for row in payload["order19614_c0_anchor"]:
        lines.append(f"| {row['dataset']} | {fmt(row['xp_final_f1'])} | {fmt(row['c0_final_f1'])} | {fmt(row['delta'])} |")
    lines += [
        "",
        "## 6. 结果解释",
        "",
        "1. 四顺序 pooled 后，X+P 的表现应被表述为具有方向性收益和方法互补性，而不是普遍超过所有强基线。",
        "2. CaT、k-center/FPS、medoid 在部分图或顺序上更强；这说明 X+P 的 Student-compatible outer optimization 还受到传播结构、类别顺序和 support plan 的影响。",
        "3. order19614 是没有参与当前参数开发的额外顺序，应作为泛化压力测试；order19613 曾出现在早期拓扑实验中，不能称为完全盲测。",
        "4. construction seed 9321 的独立小实验已经显示 X+P 存在可见的 construction variance，因此 pooled 汇总不能只报告 canonical order 或单个 seed。",
        "5. 该汇总冻结的是证据口径，不自动把 X+P 晋升为普遍 SOTA；论文应同时报告大图正向信号、小图负结果和适配基线边界。",
        "",
        "## 7. 文件与限制",
        "",
        f"- 每 cell 原始汇总：`{OUT_CELLS.as_posix()}`",
        f"- 分数据集 pooled：`{OUT_POOLED.as_posix()}`",
        f"- 配对逐 cell 差值：`{OUT_DELTAS.as_posix()}`",
        f"- 配对 pooled 差值：`{OUT_DELTA_SUMMARY.as_posix()}`",
        "- 本报告没有重新训练，也没有使用最终 F1 选择配置。",
        "- CaT/GECC 仍是机制适配，不能与官方论文数字直接比较。",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    rows = normalize_three_order() + normalize_order19614()
    validate_cells(rows)
    cells = sorted(rows, key=lambda r: (r["dataset"], r["order"], METHODS.index(r["method"])))
    dataset_pooled = pooled_rows(cells, ("dataset", "method"))
    overall_pooled = pooled_rows(cells, ("method",))
    deltas = delta_rows(cells)
    summaries = delta_summary(deltas)

    order14 = read_json(EVIDENCE / "order19614_xp_baseline_comparison.json")["rows"]
    c0_anchor = []
    for dataset in DATASETS:
        xp = next(r for r in order14 if r["dataset"] == dataset and r["method"] == "bounded_xp")
        c0 = next(r for r in order14 if r["dataset"] == dataset and r["method"] == "c0")
        c0_anchor.append({
            "dataset": dataset,
            "xp_final_f1": float(xp["final_f1"]),
            "c0_final_f1": float(c0["final_f1"]),
            "delta": float(xp["final_f1"]) - float(c0["final_f1"]),
        })

    payload = {
        "protocol": {
            "name": "bounded-xp-four-order-pooled-summary-20261001",
            "datasets": list(DATASETS),
            "orders": list(ORDERS),
            "methods": list(METHODS),
            "source_artifacts": [
                "three_order_supplement_and_cost.json",
                "order19614_xp_baseline_comparison.json",
            ],
            "no_rerun": True,
            "order19614_blind_to_current_xp_tuning": True,
        },
        "cells": cells,
        "dataset_pooled": dataset_pooled,
        "overall_pooled": overall_pooled,
        "pairwise_deltas": deltas,
        "delta_summary": summaries,
        "order19614_c0_anchor": c0_anchor,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    write_csv(OUT_CELLS, cells)
    write_csv(OUT_POOLED, dataset_pooled + overall_pooled)
    write_csv(OUT_DELTAS, deltas)
    write_csv(OUT_DELTA_SUMMARY, summaries)
    DOC.write_text(build_doc(payload), encoding="utf-8")
    print(json.dumps({
        "json": str(OUT_JSON),
        "document": str(DOC),
        "cells": len(cells),
        "pooled_rows": len(dataset_pooled),
        "delta_rows": len(deltas),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
