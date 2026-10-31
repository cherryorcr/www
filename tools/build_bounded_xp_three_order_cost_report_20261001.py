"""Build the three-order supplement and cost ledger for Bounded X+P."""
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "results" / "bounded_xp_final_evidence_20261001" / "evidence_matrix.json"
ORDER = ROOT / "results" / "bounded_xp_final_evidence_20261001" / "order19613_xp_baseline_comparison.json"
OUT_JSON = ROOT / "results" / "bounded_xp_final_evidence_20261001" / "three_order_supplement_and_cost.json"
OUT_MD = ROOT / "docs" / "current" / "BOUNDED_XP_THREE_ORDER_SUPPLEMENT_AND_COST_ZH_2026-10-01.md"


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def mean(values):
    return sum(values) / len(values)


def sample_std(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0


def normalize_order_row(row: dict) -> dict:
    return {
        "dataset": row["dataset"],
        "order": row.get("order", "order19613"),
        "method": row["method"],
        "final_f1": float(row["final_f1"]),
        "final_accuracy": float(row["final_accuracy"]),
        "stage_average_f1": float(row["stage_average_f1"]),
        "old_f1": row.get("old_f1"),
        "new_f1": row.get("new_f1"),
    }


def three_order_rows() -> list[dict]:
    evidence = read(EVIDENCE)
    rows = [normalize_order_row(row) for row in evidence["small_xp_by_order"]]
    rows.extend(normalize_order_row(row) for row in evidence["small_external_adapted_by_order"])
    additional = read(ORDER)
    for row in additional["rows"]:
        if row["method"] == "c0":
            continue
        method = {
            "bounded_xp": "Bounded X+P",
            "cat_mmd_tag": "CaT-MMD-TAG adapted",
            "gecc_hard_tag": "GECC-hard-TAG adapted",
            "k_center_native": "k-center",
            "medoid_native": "medoid",
        }[row["method"]]
        rows.append(normalize_order_row({**row, "method": method}))
    expected_methods = {"Bounded X+P", "CaT-MMD-TAG adapted", "GECC-hard-TAG adapted", "k-center", "medoid"}
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["method"])].append(row)
    for key, values in grouped.items():
        if len(values) != 3:
            raise ValueError(f"{key}: expected three orders, got {len(values)}")
    if {key[1] for key in grouped} != expected_methods:
        raise ValueError("method coverage mismatch")
    return rows


def pooled(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["method"])].append(row)
    result = []
    for (dataset, method), values in sorted(grouped.items()):
        result.append({
            "dataset": dataset,
            "method": method,
            "orders": [row["order"] for row in values],
            "order_count": len(values),
            "final_f1": mean([row["final_f1"] for row in values]),
            "final_f1_std_across_orders": sample_std([row["final_f1"] for row in values]),
            "final_accuracy": mean([row["final_accuracy"] for row in values]),
            "stage_average_f1": mean([row["stage_average_f1"] for row in values]),
            "old_f1": mean([row["old_f1"] for row in values if row["old_f1"] is not None]) if any(row["old_f1"] is not None for row in values) else None,
            "new_f1": mean([row["new_f1"] for row in values if row["new_f1"] is not None]) if any(row["new_f1"] is not None for row in values) else None,
        })
    return result


def deltas(pooled_rows: list[dict]) -> list[dict]:
    result = []
    grouped = {(row["dataset"], row["method"]): row for row in pooled_rows}
    per_order = defaultdict(dict)
    for row in three_order_rows():
        per_order[(row["dataset"], row["order"])][row["method"]] = row
    for dataset in ("cora", "citeseer"):
        for method in ("CaT-MMD-TAG adapted", "GECC-hard-TAG adapted", "k-center", "medoid"):
            values = []
            for order in ("order19611", "order19612", "order19613"):
                values.append(per_order[(dataset, order)]["Bounded X+P"]["final_f1"] - per_order[(dataset, order)][method]["final_f1"])
            result.append({
                "dataset": dataset,
                "baseline": method,
                "xp_minus_baseline_final_f1": mean(values),
                "positive_cells": sum(value > 0 for value in values),
                "cell_count": len(values),
                "worst_cell": min(values),
                "cell_deltas": values,
            })
    return result


def xpa_cost(dataset: str) -> dict:
    root = ROOT / "results" / f"xp_joint_independent_20261001_{dataset}" / f"{dataset}_order19613"
    construction = 0.0
    student = 0.0
    for result_path in sorted(root.glob("stage*/audit.json")):
        audit = read(result_path)
        construction += float(audit.get("construction_seconds_wall", 0.0))
    result = read(root / "results.json")
    for row in result["rows"]:
        student += float(row.get("student_wall_seconds", 0.0))
    return {"dataset": dataset, "order": "order19613", "method": "Bounded X+P",
            "construction_seconds": construction, "student_seconds": student,
            "cost_scope": "X+P outer construction and all six Student seeds across three stages"}


def external_cost(dataset: str) -> list[dict]:
    root = ROOT / "results" / f"external_continual_order19613_20261001_{dataset}" / dataset / "order19613"
    result = read(root / "results.json")
    grouped = defaultdict(lambda: {"construction": 0.0, "student": 0.0})
    for row in result["rows"]:
        arm = row["arm"]
        grouped[arm]["construction"] += float(row.get("construction_seconds") or 0.0)
        grouped[arm]["student"] += float(row.get("student_wall_seconds") or 0.0)
    return [{"dataset": dataset, "order": "order19613", "method": arm,
             "construction_seconds": values["construction"],
             "student_seconds": values["student"],
             "cost_scope": "adapter construction and all six Student seeds across three stages"}
            for arm, values in sorted(grouped.items())]


def large_costs() -> list[dict]:
    evidence = read(EVIDENCE)
    rows = evidence["large_strict_rows"]
    grouped = defaultdict(list)
    for row in rows:
        if row["method"] in ("Bounded X+P", "Bounded X+P+A", "k-center", "medoid", "CaT-MMD-TAG adapted", "GECC-hard-TAG adapted"):
            grouped[row["method"]].append(row)
    result = []
    for method, values in sorted(grouped.items()):
        build = [row["actual_construction_seconds"] for row in values if row.get("actual_construction_seconds") is not None]
        student = [row.get("student_seconds_total") for row in values if row.get("student_seconds_total") is not None]
        result.append({"scope": "arXiv/BookChild/SportsFit base+4 pooled", "method": method,
                       "datasets": [row["dataset"] for row in values],
                       "mean_construction_seconds": mean(build) if build else None,
                       "mean_student_seconds_total": mean(student) if student else None,
                       "construction_values": build, "student_values": student})
    return result


def main() -> None:
    rows = three_order_rows()
    pooled_rows = pooled(rows)
    comparison = deltas(pooled_rows)
    small_cost = []
    for dataset in ("cora", "citeseer"):
        small_cost.append(xpa_cost(dataset))
        small_cost.extend(external_cost(dataset))
    payload = {
        "protocol": "three-order supplement plus cost ledger 2026-10-01",
        "three_order_rows": rows,
        "three_order_pooled": pooled_rows,
        "three_order_deltas": comparison,
        "small_order19613_cost": small_cost,
        "large_pooled_cost": large_costs(),
        "limits": [
            "order19613 is an additional validation order but was used in earlier topology experiments; it is not a fully blind order.",
            "CaT/GECC are mechanism adaptations under the project Student/continual interface.",
            "The three-order table excludes X+P+A and C0 because those arms are not available across all three orders in the same external comparison matrix.",
        ],
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Bounded X+P：三顺序补充证据与成本账",
        "",
        "> 本文档把 `order19611/19612/19613` 合并为补充稳定性证据。它不替换原两顺序主表，也不把 X+P+A 混入三顺序公开基线比较。",
        "",
        "## 1. 三顺序汇总",
        "",
        "| 数据集 | 方法 | Final F1 | 跨顺序标准差 | Accuracy | 阶段平均 F1 | Old F1 | New F1 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in pooled_rows:
        old = "N/A" if row["old_f1"] is None else f"{row['old_f1']:.2f}"
        new = "N/A" if row["new_f1"] is None else f"{row['new_f1']:.2f}"
        lines.append(f"| {row['dataset']} | {row['method']} | {row['final_f1']:.2f} | {row['final_f1_std_across_orders']:.2f} | {row['final_accuracy']:.2f} | {row['stage_average_f1']:.2f} | {old} | {new} |")
    lines += [
        "",
        "## 2. X+P 相对公开适配基线",
        "",
        "| 数据集 | 基线 | X+P - 基线 Final F1 | 正向 cell | 最差 cell |",
        "|---|---|---:|---:|---:|",
    ]
    for row in comparison:
        lines.append(f"| {row['dataset']} | {row['baseline']} | {row['xp_minus_baseline_final_f1']:+.2f} | {row['positive_cells']}/{row['cell_count']} | {row['worst_cell']:+.2f} |")
    lines += [
        "",
        "## 3. 小图 order19613 成本",
        "",
        "| 数据集 | 方法 | 构造秒 | Student 秒 |",
        "|---|---|---:|---:|",
    ]
    for row in small_cost:
        name = {"cat_mmd_tag": "CaT-MMD-TAG adapted", "gecc_hard_tag": "GECC-hard-TAG adapted", "k_center_native": "k-center native", "medoid_native": "medoid native"}.get(row["method"], row["method"])
        lines.append(f"| {row['dataset']} | {name} | {row['construction_seconds']:.2f} | {row['student_seconds']:.2f} |")
    lines += [
        "",
        "## 4. 大图成本汇总",
        "",
        "| 范围 | 方法 | 平均构造秒 | 平均 Student 秒 |",
        "|---|---|---:|---:|",
    ]
    for row in large_costs():
        build = "N/A" if row["mean_construction_seconds"] is None else f"{row['mean_construction_seconds']:.2f}"
        student = "N/A" if row["mean_student_seconds_total"] is None else f"{row['mean_student_seconds_total']:.2f}"
        lines.append(f"| {row['scope']} | {row['method']} | {build} | {student} |")
    lines += [
        "",
        "## 5. 论文解释",
        "",
        "三顺序结果不支持把 X+P 写成普遍 SOTA：Cora 上它与 GECC 基本持平但低于 CaT、k-center 和 medoid，Citeseer 上高于 GECC/medoid、接近 k-center 但低于 CaT。更准确的表述是：X+P 在固定 continual 访问约束下提供了有 Student 训练反馈的内容与 payload 优化，并在部分数据集/顺序上带来收益，但该收益依赖传播结构和图规模。",
        "成本上，X+P 的额外构造时间明显高于简单 coreset，但小图 order19613 的成本仍是秒级；大图上应把 outer construction 和 Student retraining 分开报告，不能只比较最终 F1。",
        "",
        f"JSON：`{OUT_JSON.as_posix()}`",
    ]
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"json": str(OUT_JSON), "document": str(OUT_MD), "pooled_rows": len(pooled_rows), "comparisons": len(comparison), "small_cost_rows": len(small_cost)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
