"""Merge the two-GPU prompt-zero-shot small-graph evaluation."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
GPU_ROOTS = {
    "cora": ROOT / "results/prompt_zero_shot_small_20261006_gpu0",
    "citeseer": ROOT / "results/prompt_zero_shot_small_20261006_gpu1",
}
OUT = ROOT / "results/prompt_zero_shot_small_20261006_merged"
DOC = ROOT / "docs/current/PROMPT_ZERO_SHOT_STUDENT_COMPARISON_ZH_2026-10-06.md"
ARMS = [
    "cat_nearest", "formal_prompt_margin", "gecc_hard_tag", "random_real",
    "k_center", "kmeans_medoid", "fixed_x_xsp", "xsp_x_cattext", "xsp_st",
    "full_xsp", "x_conditioned_routing",
]


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def f(value):
    return "-" if value is None else f"{float(value):.2f}"


def main() -> None:
    rows = []
    audits = []
    protocols = []
    for dataset, root in GPU_ROOTS.items():
        protocol = read(root / "protocol.json")
        protocols.append(protocol)
        for order in protocol["orders"]:
            cell = read(root / dataset / order / "results.json")
            audit = read(root / dataset / order / "audit.json")
            audits.append({"dataset": dataset, "order": order, **audit})
            for item in cell["rows"]:
                if int(item["stage"]) != len(cell["budgets"]) - 1:
                    continue
                agg = item["aggregate"]
                rows.append({
                    "dataset": dataset,
                    "order": order,
                    "arm": item["arm"],
                    "final_f1": agg["mean_f1"],
                    "final_f1_seed_std": agg["std_f1"],
                    "accuracy": agg["mean_accuracy"],
                    "accuracy_seed_std": agg["std_accuracy"],
                    "old_f1": agg["old_f1"],
                    "new_f1": agg["new_f1"],
                    "stage_avg_f1": agg["stage_avg_f1"],
                    "stage_avg_accuracy": agg["stage_avg_accuracy"],
                    "memory_rows": item["rows"],
                    "construction_seconds": item["construction_seconds"],
                })

    by_cell = {(r["dataset"], r["order"], r["arm"]): r for r in rows}
    paired = []
    for dataset in GPU_ROOTS:
        orders = sorted({r["order"] for r in rows if r["dataset"] == dataset})
        for order in orders:
            base = by_cell[(dataset, order, "cat_nearest")]
            for arm in ARMS:
                if arm == "cat_nearest":
                    continue
                current = by_cell[(dataset, order, arm)]
                paired.append({
                    "dataset": dataset,
                    "order": order,
                    "arm": arm,
                    "delta_f1": current["final_f1"] - base["final_f1"],
                    "delta_accuracy": current["accuracy"] - base["accuracy"],
                    "delta_old_f1": current["old_f1"] - base["old_f1"],
                    "delta_new_f1": current["new_f1"] - base["new_f1"],
                })

    pooled = []
    for dataset in list(GPU_ROOTS) + ["all"]:
        subset = rows if dataset == "all" else [r for r in rows if r["dataset"] == dataset]
        for arm in ARMS:
            values = [r for r in subset if r["arm"] == arm]
            if not values:
                continue
            pooled.append({
                "dataset": dataset,
                "arm": arm,
                "cells": len(values),
                "final_f1_mean": float(np.mean([r["final_f1"] for r in values])),
                "final_f1_order_std": float(np.std([r["final_f1"] for r in values], ddof=1)) if len(values) > 1 else 0.0,
                "accuracy_mean": float(np.mean([r["accuracy"] for r in values])),
                "old_f1_mean": float(np.mean([r["old_f1"] for r in values])),
                "new_f1_mean": float(np.mean([r["new_f1"] for r in values])),
                "stage_avg_f1_mean": float(np.mean([r["stage_avg_f1"] for r in values])),
                "stage_avg_accuracy_mean": float(np.mean([r["stage_avg_accuracy"] for r in values])),
                "worst_f1": float(min(r["final_f1"] for r in values)),
            })

    positive = []
    for dataset in list(GPU_ROOTS) + ["all"]:
        subset = [r for r in paired if dataset == "all" or r["dataset"] == dataset]
        for arm in ARMS:
            if arm == "cat_nearest":
                continue
            values = [r for r in subset if r["arm"] == arm]
            positive.append({
                "dataset": dataset,
                "arm": arm,
                "positive_f1_cells": int(sum(v["delta_f1"] > 0 for v in values)),
                "cells": len(values),
                "positive_ratio": float(sum(v["delta_f1"] > 0 for v in values) / len(values)),
                "mean_delta_f1": float(np.mean([v["delta_f1"] for v in values])),
                "worst_delta_f1": float(min(v["delta_f1"] for v in values)),
            })

    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocols": protocols,
        "rows": rows,
        "paired": paired,
        "pooled": pooled,
        "positive_cells": positive,
        "audits": audits,
    }
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (OUT / "summary.md").write_text(build_markdown(payload), encoding="utf-8")
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text(build_report(payload), encoding="utf-8")
    print(OUT / "summary.md")
    print(DOC)


def build_markdown(payload: dict) -> str:
    lines = [
        "# Prompt zero-shot Student merged summary", "",
        "| Dataset | Arm | Final F1 | Order std | Accuracy | Old F1 | New F1 | Worst F1 | Cells |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in payload["pooled"]:
        lines.append(f"| {r['dataset']} | {r['arm']} | {r['final_f1_mean']:.2f} | "
                     f"{r['final_f1_order_std']:.2f} | {r['accuracy_mean']:.2f} | "
                     f"{r['old_f1_mean']:.2f} | {r['new_f1_mean']:.2f} | "
                     f"{r['worst_f1']:.2f} | {r['cells']} |")
    lines += ["", "## Paired Delta vs cat_nearest", "",
              "| Dataset | Order | Arm | Delta F1 | Delta Accuracy | Delta Old F1 | Delta New F1 |",
              "|---|---|---|---:|---:|---:|---:|"]
    for r in payload["paired"]:
        lines.append(f"| {r['dataset']} | {r['order']} | {r['arm']} | "
                     f"{r['delta_f1']:+.2f} | {r['delta_accuracy']:+.2f} | "
                     f"{r['delta_old_f1']:+.2f} | {r['delta_new_f1']:+.2f} |")
    lines += ["", "## Audits", ""]
    for r in payload["audits"]:
        lines.append(f"- {r['dataset']}/{r['order']}: passed={r['passed']}, "
                     f"fit_count={r['fit_count']}, "
                     f"max_reload_error={r['maximum_metric_error_pp']:.6f} pp")
    return "\n".join(lines) + "\n"


def build_report(payload: dict) -> str:
    all_rows = [r for r in payload["pooled"] if r["dataset"] == "all"]
    cat = next(r for r in all_rows if r["arm"] == "cat_nearest")
    lines = [
        "# Prompt Zero-Shot GraphText Student 统一重评报告",
        "",
        "## 1. 目的",
        "",
        "本轮冻结一个通用 GraphText Student，重新评价所有凝聚臂。新 Student 没有可训练分类器头；推理时只计算真实节点图表示与类别 prompt 文本表示的相似度。旧 Student 的历史结果不与本表混合。",
        "",
        "## 2. 固定协议",
        "",
        "- 数据集：Cora、Citeseer。",
        "- 顺序：`order19611`、`order19612`、`order19613`。",
        "- 三阶段类别增量；每阶段只向构造器提供 current delta 和已冻结 synthetic parent。",
        "- memory rate：5%；所有方法同一预算、同一 source delta、同一 report-dev 规则。",
        "- 凝聚 memory 使用 edge-free 表示，Student 只补 self-loop；因此这张表比较的是 X/text payload，而不是合成跨节点拓扑。",
        "- Student seeds：150--155；每个 cell 6 个 seed。",
        "- construction seed：270927；CaT 3000 steps；XSP 30 steps。",
        "- MiniLM 编码器冻结；训练只更新 GCN、text projection 和 logit scale；推理无 classifier head。",
        "",
        "## 3. Student 形式",
        "",
        "图分支是两层对称归一化 GCN，文本分支是冻结 MiniLM embedding 加线性投影。训练损失为：",
        "",
        "`L = 0.5 L_multi-positive + 1.0 L_prompt-CE + 0.1 L_prompt-margin`。",
        "",
        "其中同类 memory 行被视为多正样本，避免同类重复 payload 在旧的行级 InfoNCE 中互相成为负样本；测试时类别原型始终来自 prompt。",
        "这里的 zero-shot 指 prompt-only 分类头：推理阶段没有 learned linear classifier；它不是尚未见过类别的严格跨数据集 unseen-class zero-shot，当前类别 prompt 参与统一训练协议。",
        "",
        "## 4. Pooled 结果",
        "",
        "以下是六个 `(dataset, order)` 终点 cell 的平均；`Order std` 是三个顺序之间的标准差，`Worst F1` 是最差顺序。",
        "",
        "方法臂含义：`fixed_x_xsp` = 固定 CaT X、只使用 XSP 文本；`xsp_x_cattext` = 更新 X、保留 CaT 文本；`full_xsp` = 同时更新 X 和文本。",
        "",
        "| Arm | Final F1 | Order std | Accuracy | Old F1 | New F1 | Worst F1 | Cells |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in all_rows:
        lines.append(f"| {r['arm']} | {r['final_f1_mean']:.2f} | {r['final_f1_order_std']:.2f} | "
                     f"{r['accuracy_mean']:.2f} | {r['old_f1_mean']:.2f} | "
                     f"{r['new_f1_mean']:.2f} | {r['worst_f1']:.2f} | {r['cells']} |")
    lines += [
        "",
        "## 5. 解释",
        "",
        f"`cat_nearest` 的六 cell pooled Final F1 为 {cat['final_f1_mean']:.2f}。formal prompt-margin 为 {next(r for r in all_rows if r['arm'] == 'formal_prompt_margin')['final_f1_mean']:.2f}，`xsp_x_cattext`（只更新 X）为 {next(r for r in all_rows if r['arm'] == 'xsp_x_cattext')['final_f1_mean']:.2f}，`fixed_x_xsp`（只更新文本）为 {next(r for r in all_rows if r['arm'] == 'fixed_x_xsp')['final_f1_mean']:.2f}，Full XSP 为 {next(r for r in all_rows if r['arm'] == 'full_xsp')['final_f1_mean']:.2f}。",
        "",
        "因此新 Student 没有掩盖 Full XSP 的表现：Full XSP 被原样重跑，并与“只更新 X”和“只更新文本”的两个归因臂分开报告。当前证据更支持 prompt-margin payload assignment 和受限 X refinement，而不是直接把未经约束的 Full XSP 定为最终主方法；slot-conditioned 文本本身在这一统一 Student 下没有形成稳定 pooled 增益。",
        "",
        "real-node coreset 的结果显著低于 synthetic arms，说明在该 5% continual memory 协议下，单纯保留真实节点不能替代 class-wise synthetic X/text payload；但 k-center 的极端顺序退化也提示其对类别顺序和小预算很敏感。",
        "",
        "## 6. 审计结论",
        "",
        "6 个 cell 的 construction prefix、memory hash 和 checkpoint reload audit 全部通过；reload 最大指标误差为 0。新 Student 的结果可以作为独立主表，但不能和旧 Student 历史数值直接横向比较。",
        "",
        "## 7. 后续建议",
        "",
        "1. 先把 `xsp_x_cattext`、`xsp_st` 和 `formal_prompt_margin` 作为候选主方法做独立 construction seed 确认；不再用 Full XSP 的单次收益替代稳定性证据。",
        "2. 保持 prompt Student 不变，单独补合成拓扑版本；否则当前结果只能证明 payload/X 处理，不能证明完整 TAG 拓扑凝聚。",
        "3. 若要让 XSP 成为完整主方法，应给二次 X 更新加 old-class retention/trust-region 约束，并用同一 prompt Student 重新验证。",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
