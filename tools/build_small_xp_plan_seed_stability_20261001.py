"""Compare strict small X+P runs with plan seeds 9320 and 9321."""
from __future__ import annotations

import json
from pathlib import Path

from build_bounded_xp_final_evidence_20261001 import safe_mean, stage_metrics


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "bounded_xp_final_evidence_20261001"
DOC = ROOT / "docs" / "current" / "BOUNDED_XP_SMALL_CONSTRUCTION_SEED_STABILITY_ZH_2026-10-01.md"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def records(root: Path, dataset: str, plan_seed: int) -> list[dict]:
    payload = load(root / f"xp_joint_{'full' if plan_seed == 9320 else 'plan9321'}_20261001_{dataset}" / "summary.json")
    output = []
    for block in payload["results"]:
        values = [stage_metrics(row) for row in sorted(block["rows"], key=lambda r: int(r["stage"]))]
        final = values[-1]
        output.append({
            "dataset": dataset,
            "order": block["protocol"]["order"],
            "plan_seed": plan_seed,
            "final_f1": final["f1"],
            "final_accuracy": final["accuracy"],
            "stage_average_f1": safe_mean([v["f1"] for v in values]),
            "old_f1": final["old_f1"],
            "new_f1": final["new_f1"],
            "stage_f1": [v["f1"] for v in values],
        })
    return output


def main() -> None:
    baseline, repeat = [], []
    for dataset in ("cora", "citeseer"):
        baseline.extend(records(ROOT / "results", dataset, 9320))
        repeat.extend(records(ROOT / "results", dataset, 9321))
    by_key = {(r["dataset"], r["order"]): r for r in repeat}
    rows = []
    for row in baseline:
        other = by_key[(row["dataset"], row["order"])]
        rows.append({
            "dataset": row["dataset"],
            "order": row["order"],
            "final_f1_seed9320": row["final_f1"],
            "final_f1_seed9321": other["final_f1"],
            "delta_final_f1": other["final_f1"] - row["final_f1"],
            "accuracy_seed9320": row["final_accuracy"],
            "accuracy_seed9321": other["final_accuracy"],
            "delta_accuracy": other["final_accuracy"] - row["final_accuracy"],
            "old_f1_seed9320": row["old_f1"],
            "old_f1_seed9321": other["old_f1"],
            "delta_old_f1": other["old_f1"] - row["old_f1"],
            "new_f1_seed9320": row["new_f1"],
            "new_f1_seed9321": other["new_f1"],
            "delta_new_f1": other["new_f1"] - row["new_f1"],
            "stage_average_f1_seed9320": row["stage_average_f1"],
            "stage_average_f1_seed9321": other["stage_average_f1"],
            "delta_stage_average_f1": other["stage_average_f1"] - row["stage_average_f1"],
        })
    result = {
        "protocol": "strict small Bounded X+P plan-seed stability 2026-10-01",
        "fixed": ["dataset", "order", "budget", "X/P config", "Student seeds", "report rule"],
        "changed": "plan_seed: 9320 -> 9321",
        "rows": rows,
        "mean_delta_final_f1": sum(r["delta_final_f1"] for r in rows) / len(rows),
        "positive_final_cells": sum(r["delta_final_f1"] > 0 for r in rows),
        "cell_count": len(rows),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "small_xp_plan_seed_stability.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=True), encoding="utf-8")
    lines = [
        "# 小图严格 X+P：construction plan seed 稳定性",
        "",
        "固定类别顺序、memory budget、X/P 配置、Student seeds 和 report 规则，只将 `plan_seed` 从 9320 改为 9321。该实验衡量 support batch plan 的敏感性，不是新的方法或参数调优。",
        "",
        "| 数据集 | 顺序 | seed 9320 Final F1 | seed 9321 Final F1 | 差值 | seed 9320 Accuracy | seed 9321 Accuracy | 新类 F1 差值 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(f"| {r['dataset']} | {r['order']} | {r['final_f1_seed9320']:.3f} | {r['final_f1_seed9321']:.3f} | {r['delta_final_f1']:+.3f} | {r['accuracy_seed9320']:.3f} | {r['accuracy_seed9321']:.3f} | {r['delta_new_f1']:+.3f} |")
    lines += [
        "",
        f"四个 cell 的平均 Final F1 变化为 `{result['mean_delta_final_f1']:+.3f} pp`，正向 cell 为 `{result['positive_final_cells']}/{result['cell_count']}`。如果该变化量接近 Student 数值方差，后续应报告 construction seed 方差，不能只展示单个 plan seed。",
        "",
        "JSON 产物：`results/bounded_xp_final_evidence_20261001/small_xp_plan_seed_stability.json`。",
    ]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
