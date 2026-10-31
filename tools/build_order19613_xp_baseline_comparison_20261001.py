"""Build the paired order19613 comparison for Bounded X+P.

This is an additional-order validation report.  It deliberately keeps the
external mechanisms labelled as adapters and does not merge this order into
the main two-order matrix automatically.
"""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "bounded_xp_final_evidence_20261001"
DOC = ROOT / "docs" / "current" / "BOUNDED_XP_ORDER19613_PAIRED_BASELINE_COMPARISON_ZH_2026-10-01.md"

DATASETS = ("cora", "citeseer")
BASELINE_NAMES = {
    "c0": "C0 append-edge anchor",
    "cat_mmd_tag": "CaT-MMD-TAG adapted",
    "gecc_hard_tag": "GECC-hard-TAG adapted",
    "k_center_native": "k-center native append",
    "medoid_native": "medoid native append",
    "bounded_xp": "Bounded X+P",
}


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def all_rows(path: Path) -> list[dict]:
    return sorted(read(path)["rows"], key=lambda row: int(row["stage"]))


def rows_for(dataset: str) -> list[dict]:
    xp = ROOT / "results" / f"xp_joint_independent_20261001_{dataset}" / f"{dataset}_order19613" / "results.json"
    c0 = ROOT / "results" / f"c0_order19613_20261001_{dataset}" / "results.json"
    external = ROOT / "results" / f"external_continual_order19613_20261001_{dataset}" / dataset / "order19613" / "results.json"
    result = [{"dataset": dataset, "method": "bounded_xp", "source": str(xp), "trajectory": all_rows(xp)}]
    result.append({"dataset": dataset, "method": "c0", "source": str(c0), "trajectory": all_rows(c0)})
    ext_rows = read(external)["rows"]
    for arm in ("cat_mmd_tag", "gecc_hard_tag", "k_center_native", "medoid_native"):
        candidates = [row for row in ext_rows if row["arm"] == arm]
        if len(candidates) != 3:
            raise ValueError(f"{dataset}/{arm}: expected 3 stages, got {len(candidates)}")
        result.append({"dataset": dataset, "method": arm, "source": str(external),
                       "trajectory": sorted(candidates, key=lambda row: int(row["stage"]))})
    return result


def metrics(row: dict) -> dict:
    trajectory_rows = row["trajectory"]
    final = trajectory_rows[-1]
    aggregate = final["aggregate"]
    trajectory = [float(item["aggregate"]["mean_f1"]) for item in trajectory_rows]
    accuracy_trajectory = [float(item["aggregate"]["mean_accuracy"]) for item in trajectory_rows]
    return {
        "final_f1": float(aggregate["mean_f1"]),
        "final_f1_std": float(aggregate.get("std_f1", 0.0)),
        "final_accuracy": float(aggregate["mean_accuracy"]),
        "final_accuracy_std": float(aggregate.get("std_accuracy", 0.0)),
        "old_f1": aggregate.get("old_f1"),
        "new_f1": aggregate.get("new_f1"),
        "stage_average_f1": float(sum(trajectory) / len(trajectory)) if trajectory else None,
        "stage_average_accuracy": float(sum(accuracy_trajectory) / len(accuracy_trajectory)) if accuracy_trajectory else None,
        "memory_rows": int(final["memory_rows"]),
    }


def paired_records(all_rows: list[dict]) -> list[dict]:
    records = []
    for dataset in DATASETS:
        group = {row["method"]: row for row in all_rows if row["dataset"] == dataset}
        xp = group["bounded_xp"]
        for method, values in group.items():
            records.append({
                "dataset": dataset,
                "method": method,
                "xp_minus_method_final_f1": xp["final_f1"] - values["final_f1"],
                "xp_minus_method_accuracy": xp["final_accuracy"] - values["final_accuracy"],
                "xp_minus_method_stage_average_f1": xp["stage_average_f1"] - values["stage_average_f1"],
            })
    return records


def fmt(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.2f}"


def main() -> None:
    all_rows = []
    for dataset in DATASETS:
        for row in rows_for(dataset):
            values = metrics(row)
            all_rows.append({"dataset": dataset, **values, "method": row["method"], "source": row["source"]})
    comparisons = paired_records(all_rows)
    payload = {
        "protocol": {
            "name": "additional-order-19613-bounded-xp-comparison-20261001",
            "datasets": list(DATASETS),
            "order": "order19613",
            "stages": 3,
            "memory": "same 5% cumulative budget as the existing strict small protocol",
            "student": "original two-layer GCN + trainable text projection + bidirectional InfoNCE",
            "report_seeds": list(range(150, 156)),
            "x_p": "plan_seed=9320, inner_steps=8, outer_steps=4, outer_lr=0.02, C0 edge-aware topology",
            "baseline_scope": "CaT/GECC/k-center/medoid are mechanism or native-selection adapters, not official end-to-end reproductions",
            "sr": "excluded",
        },
        "rows": all_rows,
        "paired_comparisons": comparisons,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "order19613_xp_baseline_comparison.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Bounded X+P：order19613 独立顺序与基线比较",
        "",
        "> 这是额外类别顺序验证，不替换两顺序主表。所有方法使用相同 Cora/Citeseer、`order19613`、三阶段、5% cumulative memory、原始 Student 和 6 个 report seeds。",
        "",
        "## 1. Final 与阶段平均结果",
        "",
        "| 数据集 | 方法 | Final F1 | Accuracy | 阶段平均 F1 | 阶段平均 Accuracy | memory rows |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in all_rows:
        lines.append(
            f"| {row['dataset']} | {BASELINE_NAMES[row['method']]} | "
            f"{row['final_f1']:.2f} ± {row['final_f1_std']:.2f} | "
            f"{row['final_accuracy']:.2f} ± {row['final_accuracy_std']:.2f} | "
            f"{fmt(row['stage_average_f1'])} | {fmt(row['stage_average_accuracy'])} | {row['memory_rows']} |"
        )
    lines += [
        "",
        "## 2. X+P 相对同顺序方法的差值",
        "",
        "正值表示 X+P 更高；该表是每个数据集内的配对描述，不是跨数据集显著性检验。",
        "",
        "| 数据集 | 对比方法 | X+P - 对比 Final F1 | X+P - 对比 Accuracy | X+P - 对比阶段平均 F1 |",
        "|---|---|---:|---:|---:|",
    ]
    for row in comparisons:
        if row["method"] == "bounded_xp":
            continue
        lines.append(
            f"| {row['dataset']} | {BASELINE_NAMES[row['method']]} | "
            f"{row['xp_minus_method_final_f1']:+.2f} | "
            f"{row['xp_minus_method_accuracy']:+.2f} | "
            f"{row['xp_minus_method_stage_average_f1']:+.2f} |"
        )
    lines += [
        "",
        "## 3. 解释",
        "",
        "- Cora/order19613 上，X+P 的 Final F1 为 47.42；它高于 GECC-hard-TAG 与 k-center，但低于 CaT-MMD-TAG 和 medoid。Accuracy 为 55.78，高于 CaT、GECC 和 medoid，低于 k-center。",
        "- Citeseer/order19613 上，X+P 的 Final F1 为 53.16；它低于 CaT、k-center 和 medoid，高于 GECC。Accuracy 为 58.46；它低于 CaT 和 k-center，高于 GECC 与 medoid。",
        "- 因而该顺序支持“X+P 有方向性收益、但不是普遍 SOTA”，也说明 F1 与 Accuracy 对图文 memory 的偏好并不完全一致。",
        "- C0 只作为内部 append-edge anchor，用来衡量 outer optimization 的增量；公开比较仍应以适配后的 CaT、GECC、k-center 和 medoid 为主，并在论文中明确适配边界。",
        "",
        "## 4. 证据限制",
        "",
        "order19613 曾在此前拓扑实验中出现，因此这是独立于当前 X+P 参数选择的额外顺序验证，但不能称为完全未使用过的盲测顺序。CaT/GECC 仍是本项目 Student/continual 接口下的机制适配，不是官方端到端复现。",
        "",
        f"JSON：`{(OUT / 'order19613_xp_baseline_comparison.json').as_posix()}`",
    ]
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"json": str(OUT / "order19613_xp_baseline_comparison.json"), "document": str(DOC), "rows": len(all_rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
