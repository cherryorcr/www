"""Build the unified PUMA/static-TAG comparison and visualizations.

The input strict table is the already audited frozen large-graph comparison.
This script only normalizes newly completed PUMA and static full-history
results into the same schema; it does not recompute metrics or silently mix
the two access protocols.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
INPUT_STRICT = ROOT / "results" / "expanded_baseline_comparison_20261006" / "all_cells.csv"
INPUT_NEW = ROOT / "tmp_remote_results" / "puma_static_20261006"
OUT = ROOT / "results" / "puma_static_unified_comparison_20261006"

DATASETS = ["arxiv", "sportsfit", "wikics", "bookhis"]

STRICT_METHODS = {
    "Adaptive coherent",
    "CaT-MMD-TAG + CCTP top-5",
    "GECC-hard-TAG adapted",
    "K-center/FPS",
    "Medoid coreset",
    "Random real coreset",
}

PLOT_ORDER = [
    "Adaptive coherent",
    "PUMA-inspired TAG",
    "CaT-MMD-TAG + CCTP top-5",
    "GECC-hard-TAG adapted",
    "K-center/FPS",
    "Medoid coreset",
    "Random real coreset",
    "Static-TAG full-history",
]

SHORT = {
    "Adaptive coherent": "Adaptive coherent",
    "PUMA-inspired TAG": "PUMA-inspired",
    "CaT-MMD-TAG + CCTP top-5": "CaT + CCTP",
    "GECC-hard-TAG adapted": "GECC-hard",
    "K-center/FPS": "K-center",
    "Medoid coreset": "Medoid",
    "Random real coreset": "Random",
    "Static-TAG full-history": "Static TAG\n(full-history)",
}


def f(value, default=""):
    if value is None or value == "":
        return default
    return float(value)


def load_strict() -> list[dict]:
    rows = []
    with INPUT_STRICT.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row.get("method") not in STRICT_METHODS:
                continue
            rows.append({
                "dataset": row["dataset"],
                "method": row["method"],
                "category": row.get("category", "strict adapted baseline"),
                "protocol_group": "strict_continual",
                "protocol_status": row.get("protocol_status", "strict"),
                "historical_raw_reads": "0",
                "old_memory_updated": "0",
                "topology": "method-specific strict adapter",
                "final_f1": f(row.get("final_f1")),
                "final_f1_std": f(row.get("final_f1_std")),
                "accuracy": f(row.get("accuracy")),
                "accuracy_std": f(row.get("accuracy_std")),
                "old_f1": f(row.get("old_f1")),
                "new_f1": f(row.get("new_f1")),
                "stage_avg_f1": f(row.get("stage_avg_f1")),
                "worst_stage_f1": f(row.get("worst_stage_f1")),
                "construction_seconds": f(row.get("construction_seconds")),
                "student_seconds": f(row.get("student_seconds")),
                "source": row.get("source", "expanded strict table"),
            })
    return rows


def load_new() -> list[dict]:
    rows = []
    mapping = {
        "puma_arxiv_results.json": ("arxiv", "PUMA-inspired TAG", "strict_continual", "strict adapted external mechanism"),
        "puma_sportsfit_results.json": ("sportsfit", "PUMA-inspired TAG", "strict_continual", "strict adapted external mechanism"),
        "puma_wikics_results.json": ("wikics", "PUMA-inspired TAG", "strict_continual", "strict adapted external mechanism"),
        "puma_bookhis_results.json": ("bookhis", "PUMA-inspired TAG", "strict_continual", "strict adapted external mechanism"),
        "static_arxiv_results.json": ("arxiv", "Static-TAG full-history", "offline_reference", "offline/reference"),
        "static_sportsfit_results.json": ("sportsfit", "Static-TAG full-history", "offline_reference", "offline/reference"),
        "static_wikics_results.json": ("wikics", "Static-TAG full-history", "offline_reference", "offline/reference"),
        "static_bookhis_results.json": ("bookhis", "Static-TAG full-history", "offline_reference", "offline/reference"),
    }
    for filename, (dataset, method, protocol_group, category) in mapping.items():
        payload = json.loads((INPUT_NEW / filename).read_text(encoding="utf-8"))
        arm = next(iter(payload["overall"]))
        value = payload["overall"][arm]
        static = protocol_group == "offline_reference"
        rows.append({
            "dataset": dataset,
            "method": method,
            "category": category,
            "protocol_group": protocol_group,
            "protocol_status": "offline full-history reference" if static else "strict current-delta/frozen-prefix",
            "historical_raw_reads": "full seen raw at every stage" if static else "0",
            "old_memory_updated": "1" if static else "0",
            "topology": "text-kNN topology after joint rebuild" if static else "edge-free; Student self-loops",
            "final_f1": float(value["final_f1"]),
            "final_f1_std": float(value["final_f1_std"]),
            "accuracy": float(value["final_accuracy"]),
            "accuracy_std": float(value["final_accuracy_std"]),
            "old_f1": float(value["old_f1"]),
            "new_f1": float(value["new_f1"]),
            "stage_avg_f1": float(value["stage_avg_f1"]),
            "worst_stage_f1": float(value["worst_stage_f1"]),
            "construction_seconds": float(value["construction_seconds"]),
            "student_seconds": float(value["student_seconds"]),
            "source": str(INPUT_NEW / filename),
        })
    return rows


def mean_rows(rows: list[dict], method: str) -> dict:
    selected = [row for row in rows if row["method"] == method]
    out = {"method": method, "datasets": ",".join(sorted({r["dataset"] for r in selected}))}
    for key in ("final_f1", "accuracy", "old_f1", "new_f1", "stage_avg_f1",
                "worst_stage_f1", "construction_seconds", "student_seconds"):
        values = [r[key] for r in selected if isinstance(r[key], (int, float)) and math.isfinite(r[key])]
        out[key] = float(np.mean(values)) if values else ""
    return out


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(v, digits=2):
    if v in (None, ""):
        return "-"
    return f"{float(v):.{digits}f}"


def write_markdown(rows: list[dict], pooled: list[dict]) -> None:
    lines = [
        "# PUMA、静态 TAG 与严格持续基线统一对比（2026-10-06）", "",
        "本表使用同一 base+4 类别阶段、5% 累积 memory budget、Prompt zero-shot Student、Student seeds `[100..105]` 和相同 report/test 划分。严格持续组只访问当前 raw delta 与已有 synthetic prefix；静态 TAG 组允许每个阶段读取全部已见 raw 并重建旧 memory，因此只作为 offline/reference comparator。", "",
        "## 方法分组与 TAG 文本附着适配", "",
        "| 方法 | 原始机制/角色 | TAG 文本附着适配 | 历史 raw | 旧 memory | 拓扑 | 比较层级 |",
        "|---|---|---|---:|---:|---|---|",
        "| Adaptive coherent | CaT class-wise MMD + X-conditioned coherent routing | 当前 delta 文本池；按合成 X 自适应 1/2/3 slots，按图相似度路由并 hard materialize | 0 | frozen prefix | Student self-loop | 严格持续候选 |",
        "| CaT-MMD-TAG + CCTP top-5 | CaT 分布匹配 | 每个 synthetic row 从当前 delta 合法文本中取 graph-nearest top-5，320 字符内重编码 | 0 | frozen prefix | Student self-loop | 严格持续强基线 |",
        "| GECC-hard-TAG adapted | GECC hard-hop/Lloyd 代表点 | 在 current delta 上选代表点，文本随代表点保留/附着 | 0 | frozen prefix | Student self-loop | 严格持续强基线 |",
        "| K-center/FPS | 真实节点覆盖 coreset | 在 current delta 联合图文空间 FPS，文本随真实节点保留 | 0 | frozen prefix | Student self-loop | 严格持续强基线 |",
        "| Medoid coreset | 真实节点 medoid | 在 current delta 联合图文空间选 medoid，文本随节点保留 | 0 | frozen prefix | Student self-loop | 严格持续强基线 |",
        "| Random real coreset | 真实节点随机控制 | 从 current delta 随机选真实节点及对应文本 | 0 | frozen prefix | Student self-loop | 严格持续强基线 |",
        "| PUMA-inspired TAG | one-hop propagation + confidence pseudo-label teacher + CaT matching | teacher/pseudo-label 后，在合法 current delta 文本池中按 pseudo/teacher label 取 top-5 并重新编码；edge-free memory | 0 | frozen prefix | Student self-loop | 严格持续外部机制适配 |",
        "| Static-TAG full-history | full-history joint graph-text condensation | 每阶段对全部已见 raw 重建 joint graph-text clusters，融合 cluster 文本并材料化 text-kNN | full seen raw | rewrite | text-kNN | offline reference |",
        "",
        "## 严格持续主表：Final Macro-F1 / Accuracy",
        "",
    ]
    for dataset in DATASETS:
        lines += [f"### {dataset}", "", "| 方法 | Final Macro-F1 | Accuracy | Old F1 | New F1 | Stage avg F1 | Build s |", "|---|---:|---:|---:|---:|---:|---:|"]
        selected = [r for r in rows if r["dataset"] == dataset and r["protocol_group"] == "strict_continual"]
        order = ["Adaptive coherent", "PUMA-inspired TAG", "CaT-MMD-TAG + CCTP top-5", "GECC-hard-TAG adapted", "K-center/FPS", "Medoid coreset", "Random real coreset"]
        for method in order:
            for row in selected:
                if row["method"] == method:
                    lines.append(f"| {method} | {fmt(row['final_f1'])} +/- {fmt(row['final_f1_std'])} | {fmt(row['accuracy'])} +/- {fmt(row['accuracy_std'])} | {fmt(row['old_f1'])} | {fmt(row['new_f1'])} | {fmt(row['stage_avg_f1'])} | {fmt(row['construction_seconds'], 1)} |")
        lines.append("")
    lines += [
        "## 静态 TAG offline/reference 表",
        "",
        "| 数据集 | Final Macro-F1 | Accuracy | Old F1 | New F1 | Stage avg F1 | Build s | 访问协议 |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        if row["protocol_group"] == "offline_reference":
            lines.append(f"| {row['dataset']} | {fmt(row['final_f1'])} +/- {fmt(row['final_f1_std'])} | {fmt(row['accuracy'])} +/- {fmt(row['accuracy_std'])} | {fmt(row['old_f1'])} | {fmt(row['new_f1'])} | {fmt(row['stage_avg_f1'])} | {fmt(row['construction_seconds'], 1)} | full seen raw + old-memory rewrite |")
    lines += ["", "## 等权跨数据集平均", "", "| 方法 | 覆盖数据集 | Protocol | Final F1 | Accuracy | Old F1 | New F1 | Stage avg F1 | Build s |", "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in pooled:
        lines.append(f"| {row['method']} | {row['datasets']} | {row['protocol_group']} | {fmt(row['final_f1'])} | {fmt(row['accuracy'])} | {fmt(row['old_f1'])} | {fmt(row['new_f1'])} | {fmt(row['stage_avg_f1'])} | {fmt(row['construction_seconds'], 1)} |")
    lines += [
        "",
        "## 解读边界",
        "",
        "1. PUMA-inspired TAG 是公开 PUMA 机制在本项目 TAG 持续协议中的适配，不是官方 PUMA 原生复现；teacher 使用当前 delta 的一次传播特征和冻结文本，伪标签只用于当前 block，旧 memory 不更新。",
        "2. Static-TAG full-history 是完整已见 raw 的离线参考，不能与严格持续方法直接宣称公平优于/劣于；它主要回答“如果允许重读历史 raw 和重凝聚，统一 Student 能达到什么水平”。",
        "3. 所有数字同时报告 Macro-F1 与 Accuracy。BookHis 的 Accuracy 受类别分布影响明显，必须结合 New F1 和 Macro-F1 解读。",
        "4. 构造时间只统计 synthetic memory construction，不把统一 Student 训练时间混入 build time；Student time 单独保留在 CSV。",
        "",
        "图表见 `figures/`：严格持续 Final F1、Accuracy、Old/New F1、PUMA/static 协议分组和构造成本。",
    ]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_metric(rows: list[dict], metric: str, ylabel: str, filename: str,
                strict_only: bool = False) -> None:
    methods = PLOT_ORDER
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(18, 5.2), sharey=True)
    if len(DATASETS) == 1:
        axes = [axes]
    colors = {
        "Adaptive coherent": "#1f77b4",
        "PUMA-inspired TAG": "#2ca02c",
        "CaT-MMD-TAG + CCTP top-5": "#ff7f0e",
        "GECC-hard-TAG adapted": "#9467bd",
        "K-center/FPS": "#8c564b",
        "Medoid coreset": "#e377c2",
        "Random real coreset": "#7f7f7f",
        "Static-TAG full-history": "#d62728",
    }
    all_heights = []
    for dataset in DATASETS:
        for method in methods:
            if strict_only and method == "Static-TAG full-history":
                continue
            candidates = [r for r in rows if r["dataset"] == dataset and r["method"] == method]
            if candidates:
                row = candidates[0]
                all_heights.append(float(row[metric]) + float(row.get(metric + "_std", 0.0) if metric in {"final_f1", "accuracy"} else 0.0))
    global_max = max(all_heights) if all_heights else 1.0
    for ax, dataset in zip(axes, DATASETS):
        values, errors, labels, bars = [], [], [], []
        for method in methods:
            candidates = [r for r in rows if r["dataset"] == dataset and r["method"] == method]
            if strict_only and method == "Static-TAG full-history":
                continue
            if not candidates:
                continue
            row = candidates[0]
            values.append(row[metric])
            errors.append(row.get(metric + "_std", 0.0) if metric in {"final_f1", "accuracy"} else 0.0)
            labels.append(SHORT[method])
            bars.append(method)
        x = np.arange(len(values))
        ax.bar(x, values, yerr=errors, capsize=3, color=[colors[m] for m in bars], alpha=0.88)
        ax.set_title(dataset)
        ax.set_xticks(x, labels, rotation=55, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.25)
        ax.set_ylim(0, global_max * 1.18)
    axes[0].set_ylabel(ylabel)
    fig.suptitle(ylabel + (" (strict continual only)" if strict_only else ""))
    fig.tight_layout()
    fig.savefig(OUT / "figures" / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_old_new(rows: list[dict]) -> None:
    methods = PLOT_ORDER
    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    for method in methods:
        selected = [r for r in rows if r["method"] == method]
        if not selected:
            continue
        old = np.mean([r["old_f1"] for r in selected])
        new = np.mean([r["new_f1"] for r in selected])
        marker = "s" if method == "Static-TAG full-history" else "o"
        ax.scatter(old, new, s=95, marker=marker, label=SHORT[method].replace("\n", " "))
    ax.set_xlabel("Mean Old-class F1")
    ax.set_ylabel("Mean New-class F1")
    ax.set_title("Old/New F1 trade-off across four large graphs")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "old_new_tradeoff.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_protocol(rows: list[dict]) -> None:
    order = ["Adaptive coherent", "PUMA-inspired TAG", "CaT-MMD-TAG + CCTP top-5", "GECC-hard-TAG adapted", "K-center/FPS", "Medoid coreset", "Random real coreset", "Static-TAG full-history"]
    selected = [r for r in rows if r["method"] in order]
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    xpos = np.arange(len(order))
    vals = []
    for method in order:
        rs = [r for r in selected if r["method"] == method]
        vals.append(np.mean([r["construction_seconds"] for r in rs]) if rs else np.nan)
    bars = ax.bar(xpos, vals, color=["#d62728" if m == "Static-TAG full-history" else "#4c78a8" for m in order])
    ax.set_xticks(xpos, [SHORT[m] for m in order], rotation=45, ha="right")
    ax.set_ylabel("Mean construction time (s)")
    ax.set_title("Construction cost; static TAG is an offline full-history reference")
    ax.grid(axis="y", alpha=0.25)
    for bar, val in zip(bars, vals):
        if np.isfinite(val):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.0f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "construction_cost.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)
    rows = load_strict() + load_new()
    fields = ["dataset", "method", "category", "protocol_group", "protocol_status",
              "historical_raw_reads", "old_memory_updated", "topology", "final_f1",
              "final_f1_std", "accuracy", "accuracy_std", "old_f1", "new_f1",
              "stage_avg_f1", "worst_stage_f1", "construction_seconds", "student_seconds", "source"]
    write_csv(OUT / "unified_cells.csv", rows, fields)

    strict_rows = [r for r in rows if r["protocol_group"] == "strict_continual"]
    offline_rows = [r for r in rows if r["protocol_group"] == "offline_reference"]
    write_csv(OUT / "strict_continual_cells.csv", strict_rows, fields)
    write_csv(OUT / "offline_reference_cells.csv", offline_rows, fields)
    pooled = []
    for method in PLOT_ORDER:
        method_rows = [r for r in rows if r["method"] == method]
        if not method_rows:
            continue
        item = mean_rows(rows, method)
        item["protocol_group"] = method_rows[0]["protocol_group"]
        pooled.append(item)
    pooled_fields = ["method", "datasets", "protocol_group", "final_f1", "accuracy", "old_f1", "new_f1", "stage_avg_f1", "worst_stage_f1", "construction_seconds", "student_seconds"]
    write_csv(OUT / "pooled_methods.csv", pooled, pooled_fields)
    write_markdown(rows, pooled)
    plot_metric(rows, "final_f1", "Final Macro-F1", "final_f1_by_dataset.png", strict_only=True)
    plot_metric(rows, "accuracy", "Accuracy", "accuracy_by_dataset.png", strict_only=True)
    plot_metric(rows, "final_f1", "Final Macro-F1", "final_f1_with_static_reference.png", strict_only=False)
    plot_old_new(rows)
    plot_protocol(rows)
    print(json.dumps({"rows": len(rows), "output": str(OUT), "pooled": len(pooled)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
