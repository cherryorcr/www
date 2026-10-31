"""Summarize the formal Unified-Q large-graph transfer runs."""
from __future__ import annotations

import csv
import json
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    result_root = root / "results" / "unified_q_large_full_20261007"
    out = root / "results" / "unified_q_large_full_20261007"
    rows = []
    final = []
    for path in sorted(result_root.glob("*_unified_q_smoke.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        dataset = payload["dataset"]
        for item in payload["rows"]:
            score = item["score"]
            rows.append({
                "dataset": dataset,
                "arm": item["arm"],
                "stage": item["stage"],
                "raw_rows": item["raw_rows"],
                "new_rows": item["new_rows"],
                "memory_rows": item["memory_rows"],
                "construction_seconds": item["construction_seconds"],
                "final_f1": score["final_f1"],
                "final_f1_std": score["final_f1_std"],
                "accuracy": score["accuracy"],
                "accuracy_std": score["accuracy_std"],
                "old_f1": score["old_f1"],
                "new_f1": score["new_f1"],
                "student_steps": score["student_steps"],
                "student_seeds": ",".join(map(str, score["student_seeds"])),
            })
        last = max((item for item in rows if item["dataset"] == dataset),
                   key=lambda item: item["stage"])
        for arm in {item["arm"] for item in rows if item["dataset"] == dataset}:
            value = max((item for item in rows
                         if item["dataset"] == dataset and item["arm"] == arm),
                        key=lambda item: item["stage"])
            final.append(dict(value))

    stage_fields = list(rows[0])
    with (out / "stage_rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=stage_fields)
        writer.writeheader()
        writer.writerows(rows)
    final_fields = list(final[0])
    with (out / "final_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=final_fields)
        writer.writeheader()
        writer.writerows(final)

    cat = {(row["dataset"]): row["final_f1"] for row in final if row["arm"] == "cat_x"}
    lines = [
        "# Unified-Q 大图迁移实验",
        "",
        "本报告使用 Unified-Q 正式配置：5-stage base+4、5% cumulative memory、Q 优化 30 步、Student 2048 步、3 个 Student seeds。",
        "WikiCS 与 BookHis 使用同一 runner、同一 prompt-only GraphText Student 和同一数据访问约束。",
        "这里的结果用于判断大图迁移可行性，不等同于多顺序最终论文主表。",
        "",
        "## 最终阶段",
        "",
        "| 数据集 | 方法 | Final Macro-F1 | Accuracy | Old F1 | New F1 | 构造时间 | 相对 CaT F1 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(final, key=lambda item: (item["dataset"], item["arm"])):
        delta = float(row["final_f1"]) - float(cat[row["dataset"]])
        lines.append(
            f"| {row['dataset']} | {row['arm']} | {float(row['final_f1']):.3f} ± {float(row['final_f1_std']):.3f} | "
            f"{float(row['accuracy']):.3f} ± {float(row['accuracy_std']):.3f} | "
            f"{float(row['old_f1']) if row['old_f1'] is not None else 0.0:.3f} | "
            f"{float(row['new_f1']):.3f} | {float(row['construction_seconds']):.2f} | {delta:+.3f} |"
        )
    lines.extend([
        "",
        "## 阶段曲线",
        "",
        "| 数据集 | 阶段 | CaT-X F1 | Unified-Q F1 | CaT-X Accuracy | Unified-Q Accuracy |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    datasets = sorted({row["dataset"] for row in rows})
    for dataset in datasets:
        stages = sorted({row["stage"] for row in rows if row["dataset"] == dataset})
        for stage in stages:
            lookup = {(r["arm"]): r for r in rows
                      if r["dataset"] == dataset and r["stage"] == stage}
            lines.append(
                f"| {dataset} | {stage} | {float(lookup['cat_x']['final_f1']):.3f} | "
                f"{float(lookup['unified_q']['final_f1']):.3f} | "
                f"{float(lookup['cat_x']['accuracy']):.3f} | {float(lookup['unified_q']['accuracy']):.3f} |"
            )
    lines.extend([
        "",
        "## 解释",
        "",
        "- WikiCS 上 Unified-Q 在第一个阶段已经低于 CaT-X，后续阶段继续扩大差距，说明当前 Q/文本 materialization 在 WikiCS 的图文分布上没有保留足够的 CaT 特征质量。",
        "- BookHis 上 Unified-Q 在所有阶段都高于 CaT-X，尤其在长尾阶段表现更稳定；但这仍是一个固定顺序和有限 Student seeds，不能直接称为普遍优势。",
        "- 两个数据集的差异说明大图优化重点应先放在 Q 的类别预算、文本候选多样性和 Student proxy 与最终 Student 的一致性，而不是继续加入 boundary 或更多拓扑模块。",
        "- 当前 Unified-Q 可以继续作为大图主线候选，但必须补 WikiCS 的独立顺序和构造 seed，才能判断退化是否稳定。",
        "",
        "## 协议边界",
        "",
        "本轮与 CaT-X 使用同一大图 runner 和同一 Student；没有把旧的 edge-aware Student 基线数字直接混入。",
        "已有公开/adapted baseline 结果应单独标记为不同 Student 或不同 seed 数，不能与本表直接作严格显著性结论。",
    ])
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    for dataset in datasets:
        subset = [row for row in rows if row["dataset"] == dataset]
        stages = sorted({row["stage"] for row in subset})
        figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        for arm, label in (("cat_x", "CaT-X"), ("unified_q", "Unified-Q")):
            values = [next(row for row in subset if row["arm"] == arm and row["stage"] == stage)["final_f1"] for stage in stages]
            acc = [next(row for row in subset if row["arm"] == arm and row["stage"] == stage)["accuracy"] for stage in stages]
            axes[0].plot(stages, values, marker="o", label=label)
            axes[1].plot(stages, acc, marker="o", label=label)
        axes[0].set_title(f"{dataset}: Macro-F1")
        axes[1].set_title(f"{dataset}: Accuracy")
        for axis in axes:
            axis.set_xlabel("Stage")
            axis.grid(alpha=0.25)
            axis.legend()
        figure.savefig(out / f"{dataset}_stage_curves.png", dpi=180)
        plt.close(figure)


if __name__ == "__main__":
    main()
