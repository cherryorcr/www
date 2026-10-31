"""Build a compact report and plots for the hard-consistent Unified-Q study."""
from __future__ import annotations

import csv
from pathlib import Path


METHODS = {
    "cat_x": ("CaT X", "CaT feature condensation + current-delta nearest text; no saved topology"),
    "cat_gm_decoupled": ("CaT + decoupled GM", "CaT warm start + separate multimodal gradient matching"),
    "unified_q_edge_free": ("Unified-Q edge-free", "One class-masked Q derives X/text; topology removed"),
    "unified_q": ("Unified-Q", "One class-masked Q derives X, text, and induced topology"),
    "unified_q_cat_anchor": ("Unified-Q + CaT anchor", "Unified-Q with an extra bounded CaT feature anchor"),
    "unified_q_boundary": ("Unified-Q + boundary", "Unified-Q with current-to-historical boundary sketch"),
}


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def f(row, key):
    return float(row[key])


def main():
    root = Path(__file__).resolve().parents[1]
    input_dirs = {
        "cora": root / "results" / "unified_q_hard_compare_20261007_cora" / "summary",
        "citeseer": root / "results" / "unified_q_hard_compare_20261007_citeseer" / "summary",
    }
    out = root / "results" / "unified_q_hard_compare_20261007_summary"
    out.mkdir(parents=True, exist_ok=True)

    pooled = []
    for dataset, directory in input_dirs.items():
        for row in read_csv(directory / "pooled_rows.csv"):
            row = dict(row)
            row["dataset"] = dataset
            row["method_name"] = METHODS[row["arm"]][0]
            row["method_definition"] = METHODS[row["arm"]][1]
            pooled.append(row)

    cat_by_dataset = {
        dataset: next(row for row in pooled if row["dataset"] == dataset and row["arm"] == "cat_x")
        for dataset in input_dirs
    }
    for row in pooled:
        base = cat_by_dataset[row["dataset"]]
        row["delta_f1_vs_cat"] = f(row, "final_macro_f1_mean") - f(base, "final_macro_f1_mean")
        row["delta_accuracy_vs_cat"] = f(row, "final_accuracy_mean") - f(base, "final_accuracy_mean")

    fields = [
        "dataset", "arm", "method_name", "method_definition",
        "final_macro_f1_mean", "final_macro_f1_std",
        "final_accuracy_mean", "final_accuracy_std",
        "stage_average_f1_mean", "stage_average_accuracy_mean",
        "construction_seconds_mean", "delta_f1_vs_cat", "delta_accuracy_vs_cat",
    ]
    with (out / "pooled_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(pooled)

    lines = [
        "# Unified-Q 硬一致实验总表",
        "",
        "本表汇总 Cora 和 Citeseer、三个类别顺序、三个 Student seeds 的最终阶段结果。",
        "所有方法使用同一 prompt-only GraphText Student、5% cumulative memory、current-delta-only 访问协议。",
        "`delta` 均相对于同数据集的 `CaT X`。",
        "",
        "## 方法定义",
        "",
        "| 方法 | 实际做法 |",
        "|---|---|",
    ]
    for arm, (name, definition) in METHODS.items():
        lines.append(f"| {name} (`{arm}`) | {definition} |")
    lines.extend([
        "",
        "## Pooled results",
        "",
        "| 数据集 | 方法 | Final Macro-F1 | Accuracy | Stage-average F1 | 构造秒数 | ΔF1 vs CaT | ΔAcc vs CaT |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    order = {arm: i for i, arm in enumerate(METHODS)}
    for row in sorted(pooled, key=lambda item: (item["dataset"], order[item["arm"]])):
        lines.append(
            "| {dataset} | {method_name} | {f1:.3f} ± {f1s:.3f} | {acc:.3f} ± {accs:.3f} | "
            "{stage:.3f} | {time:.3f} | {df1:+.3f} | {dacc:+.3f} |".format(
                dataset=row["dataset"], method_name=row["method_name"],
                f1=f(row, "final_macro_f1_mean"), f1s=f(row, "final_macro_f1_std"),
                acc=f(row, "final_accuracy_mean"), accs=f(row, "final_accuracy_std"),
                stage=f(row, "stage_average_f1_mean"), time=f(row, "construction_seconds_mean"),
                df1=f(row, "delta_f1_vs_cat"), dacc=f(row, "delta_accuracy_vs_cat"),
            )
        )
    lines.extend([
        "",
        "## 结论",
        "",
        "- Unified-Q 在两个数据集上均可运行，且与 CaT 相差不大：Cora 约低 3.00 F1，Citeseer 约低 0.90 F1。",
        "- `unified_q_edge_free` 是拓扑消融；它不能证明统一拓扑有效，只能说明 Q 的 X/text 部分本身的影响。",
        "- CaT anchor 没有稳定收益，不进入主方法必需组件。",
        "- boundary sketch 在 Citeseer 明显不稳定，保留为扩展/负结果，不作为主方法结论。",
        "- 因此当前最合理的候选是 `Unified-Q`：它实现了一个 Q 同时决定 X、文本和诱导拓扑，但目前不能宣称超过 CaT 或 SOTA。",
        "",
        "## 公平性边界",
        "",
        "本实验是统一方法内部和 CaT warm-start 方法的严格小图可行性比较，不等同于已经完成所有公开 TAG baseline 的同协议重跑。",
        "外部基线应在论文主表中单独标记为 adapted protocol，不能把历史目录结果和本实验直接混成一个结论。",
    ])
    (out / "comparison_report_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib/numpy unavailable; tables were still written")
        return

    for metric, label, filename in [
        ("final_macro_f1_mean", "Final Macro-F1", "final_f1_by_dataset.png"),
        ("final_accuracy_mean", "Final Accuracy", "final_accuracy_by_dataset.png"),
    ]:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
        for ax, dataset in zip(axes, input_dirs):
            rows = [row for row in pooled if row["dataset"] == dataset]
            x = np.arange(len(rows))
            values = [f(row, metric) for row in rows]
            errors = [f(row, metric.replace("_mean", "_std")) for row in rows]
            colors = ["#2f6f9f" if row["arm"] == "cat_x" else "#6c8ebf" if row["arm"] == "unified_q" else "#b7c9e2" for row in rows]
            ax.bar(x, values, yerr=errors, capsize=3, color=colors, edgecolor="#243447", linewidth=0.5)
            ax.set_title(dataset.capitalize())
            ax.set_ylabel(label)
            ax.set_xticks(x)
            ax.set_xticklabels([row["method_name"].replace(" + ", " +\n") for row in rows], rotation=35, ha="right", fontsize=8)
            ax.grid(axis="y", alpha=0.25)
        fig.suptitle(f"Unified-Q hard-consistent comparison: {label}")
        fig.savefig(out / filename, dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    for dataset in input_dirs:
        rows = [row for row in pooled if row["dataset"] == dataset]
        x = np.arange(len(rows))
        deltas = [f(row, "delta_f1_vs_cat") for row in rows]
        ax.plot(x, deltas, marker="o", linewidth=2, label=dataset.capitalize())
    ax.axhline(0.0, color="#333333", linewidth=1)
    ax.set_xticks(np.arange(len(METHODS)))
    ax.set_xticklabels([name for name, _ in METHODS.values()], rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Final Macro-F1 difference vs CaT")
    ax.set_title("Relative performance of unified-Q variants")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.savefig(out / "delta_f1_vs_cat.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
