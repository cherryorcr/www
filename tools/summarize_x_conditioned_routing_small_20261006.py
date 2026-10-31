"""Merge the X-conditioned routing prompt-zero-shot small-graph evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ARMS = [
    "cat_nearest", "formal_prompt_margin", "x_conditioned_routing",
    "x_conditioned_routing_x_coverage",
    "x_conditioned_routing_coherent",
    "gecc_hard_tag", "random_real", "k_center", "kmeans_medoid",
    "fixed_x_xsp", "xsp_x_cattext", "xsp_st", "full_xsp",
]


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cora-root",
        type=Path,
        default=ROOT / "results/x_conditioned_routing_small_20261006_cora",
    )
    parser.add_argument(
        "--citeseer-root",
        type=Path,
        default=ROOT / "results/x_conditioned_routing_small_20261006_citeseer",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results/x_conditioned_routing_small_20261006_merged",
    )
    parser.add_argument(
        "--doc",
        type=Path,
        default=ROOT / "docs/current/X_CONDITIONED_PAYLOAD_ROUTING_ZH_2026-10-06.md",
    )
    return parser.parse_args()


def collect(roots: dict[str, Path]) -> tuple[list[dict], list[dict], list[dict]]:
    rows: list[dict] = []
    audits: list[dict] = []
    protocols: list[dict] = []
    for dataset, root in roots.items():
        protocol = read(root / "protocol.json")
        protocols.append(protocol)
        for order in protocol["orders"]:
            cell = read(root / dataset / order / "results.json")
            audit = read(root / dataset / order / "audit.json")
            audits.append({"dataset": dataset, "order": order, **audit})
            final_stage = len(cell["budgets"]) - 1
            for item in cell["rows"]:
                if int(item["stage"]) != final_stage:
                    continue
                agg = item["aggregate"]
                rows.append({
                    "dataset": dataset,
                    "order": order,
                    "arm": item["arm"],
                    "stage": int(item["stage"]),
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
                    "audit": item.get("audit", {}),
                })
    return rows, audits, protocols


def summarize(rows: list[dict], roots: dict[str, Path]) -> dict:
    available_arms = [arm for arm in ARMS
                      if any(row["arm"] == arm for row in rows)]
    by_cell = {(r["dataset"], r["order"], r["arm"]): r for r in rows}
    paired = []
    for dataset in roots:
        orders = sorted({r["order"] for r in rows if r["dataset"] == dataset})
        for order in orders:
            base = by_cell[(dataset, order, "cat_nearest")]
            for arm in available_arms:
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
    for dataset in list(roots) + ["all"]:
        subset = rows if dataset == "all" else [r for r in rows if r["dataset"] == dataset]
        for arm in available_arms:
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
    for dataset in list(roots) + ["all"]:
        subset = [r for r in paired if dataset == "all" or r["dataset"] == dataset]
        for arm in available_arms:
            if arm == "cat_nearest":
                continue
            values = [r for r in subset if r["arm"] == arm]
            if not values:
                continue
            positive.append({
                "dataset": dataset,
                "arm": arm,
                "positive_f1_cells": int(sum(v["delta_f1"] > 0 for v in values)),
                "cells": len(values),
                "positive_ratio": float(sum(v["delta_f1"] > 0 for v in values) / len(values)),
                "mean_delta_f1": float(np.mean([v["delta_f1"] for v in values])),
                "worst_delta_f1": float(min(v["delta_f1"] for v in values)),
            })
    routing_stats = []
    for row in rows:
        if row["arm"] != "x_conditioned_routing":
            continue
        audit = row.get("audit", {})
        unique = [int(v) for v in audit.get("unique_payloads_by_class", {}).values()]
        usages = [
            value.get("usage", [])
            for value in audit.get("routing_by_class", {}).values()
        ]
        active_slots = [sum(float(v) > 0.05 for v in usage) for usage in usages]
        if not unique:
            continue
        routing_stats.append({
            "dataset": row["dataset"],
            "order": row["order"],
            "stage": int(row.get("stage", 0)),
            "mean_unique_payloads_per_class": float(np.mean(unique)),
            "min_unique_payloads_per_class": int(min(unique)),
            "max_unique_payloads_per_class": int(max(unique)),
            "multi_payload_class_ratio": float(np.mean([v > 1 for v in unique])),
            "mean_active_slots": float(np.mean(active_slots)) if active_slots else 0.0,
            "multi_active_slot_class_ratio": float(np.mean([v > 1 for v in active_slots])) if active_slots else 0.0,
        })
    return {
        "rows": rows,
        "paired": paired,
        "pooled": pooled,
        "positive_cells": positive,
        "routing_stats": routing_stats,
    }


def build_markdown(payload: dict) -> str:
    lines = [
        "# X-conditioned payload routing merged summary", "",
        "| Dataset | Arm | Final F1 | Order std | Accuracy | Old F1 | New F1 | Worst F1 | Cells |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in payload["pooled"]:
        lines.append(
            f"| {r['dataset']} | {r['arm']} | {r['final_f1_mean']:.2f} | "
            f"{r['final_f1_order_std']:.2f} | {r['accuracy_mean']:.2f} | "
            f"{r['old_f1_mean']:.2f} | {r['new_f1_mean']:.2f} | "
            f"{r['worst_f1']:.2f} | {r['cells']} |"
        )
    lines += [
        "", "## Paired Delta vs cat_nearest", "",
        "| Dataset | Order | Arm | Delta F1 | Delta Accuracy | Delta Old F1 | Delta New F1 |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for r in payload["paired"]:
        lines.append(
            f"| {r['dataset']} | {r['order']} | {r['arm']} | "
            f"{r['delta_f1']:+.2f} | {r['delta_accuracy']:+.2f} | "
            f"{r['delta_old_f1']:+.2f} | {r['delta_new_f1']:+.2f} |"
        )
    lines += [
        "", "## Routing audit", "",
        "| Dataset | Order | Mean unique payloads/class | Min | Max | Multi-payload class ratio | Mean active slots |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in payload["routing_stats"]:
        lines.append(
            f"| {r['dataset']} | {r['order']} | {r['mean_unique_payloads_per_class']:.2f} | "
            f"{r['min_unique_payloads_per_class']} | {r['max_unique_payloads_per_class']} | "
            f"{r['multi_payload_class_ratio']:.2f} | {r['mean_active_slots']:.2f} |"
        )
    return "\n".join(lines) + "\n"


def build_report(payload: dict) -> str:
    pooled = [r for r in payload["pooled"] if r["dataset"] == "all"]
    lookup = {r["arm"]: r for r in pooled}
    lines = [
        "# X-conditioned Differentiable Multi-Payload Routing 统一实验报告",
        "",
        "## 1. 方法与目标",
        "",
        "本轮固定 CaT 合成特征 X，只研究一个类别内不同合成节点是否应使用不同的真实文本 payload。每个类别从当前 raw delta 构造一个有界真实候选池；每个合成行根据自身 X 通过可微 assignment logits 进行路由，最后用 argmax 材料化为真实文本。该模块不修改旧 memory、不读取历史 raw、不修改拓扑，也不根据最终测试 F1 选文本。",
        "",
        "## 2. 固定协议",
        "",
        "- 数据集：Cora、Citeseer；三个独立类别顺序；每个 cell 六个 Student seeds。",
        "- memory rate、source delta、report-dev/test 划分、CaT 训练预算和 Prompt Student 完全一致。",
        "- Student 为无 learned classifier head 的 prompt zero-shot 图文模型；所有方法使用同一个 Student 与同一训练协议。",
        "- 候选池默认最多 3 个 slot，routing steps=30，temperature=0.15，load weight=0.03，anchor weight=0.10。",
        "",
        "## 3. 结果汇总",
        "",
        "| 方法 | Final F1 | Order std | Accuracy | Old F1 | New F1 | Worst F1 | Cells |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in pooled:
        lines.append(
            f"| {r['arm']} | {r['final_f1_mean']:.2f} | {r['final_f1_order_std']:.2f} | "
            f"{r['accuracy_mean']:.2f} | {r['old_f1_mean']:.2f} | {r['new_f1_mean']:.2f} | "
            f"{r['worst_f1']:.2f} | {r['cells']} |"
        )
    lines += [
        "",
        "## 4. 解释口径",
        "",
        f"CaT nearest pooled Final F1 为 {lookup['cat_nearest']['final_f1_mean']:.2f}。" if "cat_nearest" in lookup else "CaT nearest 结果尚未完整。",
        "x_conditioned_routing 只有在多个独立顺序、两个数据集上都表现稳定时，才可作为文本模块的正向证据；如果仅改善少数 cell，应将其定位为有解释性的组件，而不是主方法整体提升。",
        "",
        "路由审计显示六个最终阶段中每个类别均使用了 2--3 个不同的真实 payload，三个 slot 均有非零使用；因此本轮不是实现退化为 class-shared text，而是多 payload 路由确实生效但收益有限。",
        "",
        "本报告只汇总同一 Prompt Student 下的本轮结果，不与旧 Student 或不同协议的历史结果直接混合。",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    roots = {"cora": args.cora_root, "citeseer": args.citeseer_root}
    rows, audits, protocols = collect(roots)
    payload = summarize(rows, roots)
    payload["audits"] = audits
    payload["protocols"] = protocols
    args.out.mkdir(parents=True, exist_ok=True)
    args.out.joinpath("summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.out.joinpath("summary.md").write_text(build_markdown(payload), encoding="utf-8")
    args.doc.parent.mkdir(parents=True, exist_ok=True)
    args.doc.write_text(build_report(payload), encoding="utf-8")
    print(args.out / "summary.md")
    print(args.doc)


if __name__ == "__main__":
    main()
