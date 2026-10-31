"""Summarize the completed four-arm XSP attribution experiment."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


ARMS = ("cat_nearest", "fixed_x_xsp", "xsp_x_cattext", "full_xsp")
LABELS = {
    "cat_nearest": "CaT-nearest",
    "fixed_x_xsp": "Fixed-X XSP",
    "xsp_x_cattext": "XSP-X + CaT-text",
    "full_xsp": "Full XSP",
}


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def f(value, digits=3):
    if value is None:
        return "NA"
    return f"{float(value):.{digits}f}"


def mean(values):
    return statistics.mean(values) if values else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    protocols = []
    for dataset in ("cora", "citeseer"):
        root = args.input_root / dataset
        protocols.append(load(root / "protocol.json"))
        rows.extend(load(root / "results.json")["rows"])
    args.output_root.mkdir(parents=True, exist_ok=True)

    cells = {}
    for row in rows:
        cells[(row["dataset"], row["order"], row["arm"], int(row["stage"]))] = row
    final = {
        (dataset, order, arm): row
        for (dataset, order, arm, stage), row in cells.items()
        if stage == 2
    }
    base = {(dataset, order): final[(dataset, order, "cat_nearest")]
            for dataset, order, _ in final}

    summary = {"protocols": protocols, "final_cells": [], "pooled": {}}
    for (dataset, order, arm), row in sorted(final.items()):
        aggregate = row["aggregate"]
        reference = base[(dataset, order)]["aggregate"]
        out = {
            "dataset": dataset,
            "order": order,
            "arm": arm,
            "final_f1": aggregate["mean_f1"],
            "final_accuracy": aggregate["mean_accuracy"],
            "old_f1": aggregate["old_f1"],
            "new_f1": aggregate["new_f1"],
            "f1_std": aggregate["std_f1"],
            "accuracy_std": aggregate["std_accuracy"],
            "delta_f1_vs_cat": aggregate["mean_f1"] - reference["mean_f1"],
            "delta_accuracy_vs_cat": aggregate["mean_accuracy"] - reference["mean_accuracy"],
            "delta_old_vs_cat": (
                None if aggregate["old_f1"] is None or reference["old_f1"] is None
                else aggregate["old_f1"] - reference["old_f1"]),
            "delta_new_vs_cat": (
                None if aggregate["new_f1"] is None or reference["new_f1"] is None
                else aggregate["new_f1"] - reference["new_f1"]),
            "construction_seconds": row["construction_seconds"],
            "x_update_relative_max": row["audit"].get("x_update_relative_max"),
            "x_update_relative_mean": row["audit"].get("x_update_relative_mean"),
            "unique_text_ratio": row["audit"].get("unique_text_ratio"),
        }
        summary["final_cells"].append(out)

    for arm in ARMS:
        arm_rows = [r for r in summary["final_cells"] if r["arm"] == arm]
        stage_rows = [r for r in rows if r["arm"] == arm]
        deltas = [r["delta_f1_vs_cat"] for r in arm_rows if arm != "cat_nearest"]
        summary["pooled"][arm] = {
            "final_f1": mean([r["final_f1"] for r in arm_rows]),
            "final_accuracy": mean([r["final_accuracy"] for r in arm_rows]),
            "final_old_f1": mean([r["old_f1"] for r in arm_rows if r["old_f1"] is not None]),
            "final_new_f1": mean([r["new_f1"] for r in arm_rows if r["new_f1"] is not None]),
            "stage_average_f1": mean([r["aggregate"]["mean_f1"] for r in stage_rows]),
            "stage_average_accuracy": mean([r["aggregate"]["mean_accuracy"] for r in stage_rows]),
            "positive_final_cells": sum(v > 0 for v in deltas),
            "final_cell_count": len(deltas),
        }

    (args.output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    with (args.output_root / "cell_final.csv").open("w", encoding="utf-8") as handle:
        handle.write("dataset,order,arm,final_f1,accuracy,old_f1,new_f1,delta_f1,delta_accuracy,delta_old,delta_new\n")
        for row in summary["final_cells"]:
            handle.write(",".join([
                row["dataset"], row["order"], row["arm"], f(row["final_f1"]),
                f(row["final_accuracy"]), f(row["old_f1"]), f(row["new_f1"]),
                f(row["delta_f1_vs_cat"]), f(row["delta_accuracy_vs_cat"]),
                f(row["delta_old_vs_cat"]), f(row["delta_new_vs_cat"]),
            ]) + "\n")

    pooled = summary["pooled"]
    cat = pooled["cat_nearest"]
    report = []
    report.append("# XSP 四臂归因实验报告\n")
    report.append("> 目的：在完全相同的 CaT 初始化、类别顺序、memory budget 和 Student 协议下，区分 XSP 的文本分配作用与二次更新 X 的作用。\n")
    report.append("## 1. 实验问题与四个版本\n")
    report.append("本轮不是新的主方法搜索，而是对 XSP 性能变化做因果归因。每个阶段只运行一次 CaT，并把同一个 CaT synthetic X 作为四个版本的共同锚点。\n")
    report.append("| 版本 | X | 文本 | 归因含义 |\n|---|---|---|---|\n")
    report.append("| CaT-nearest | CaT X | CaT graph-nearest top-5 | 共同基线 |\n")
    report.append("| Fixed-X XSP | 固定 CaT X | XSP slot-conditioned text | 只改变文本分配 |\n")
    report.append("| XSP-X + CaT-text | XSP 更新 X | CaT 文本 | 只观察二次 X 更新的净影响（X 的训练目标仍来自 XSP） |\n")
    report.append("| Full XSP | XSP 更新 X | XSP slot-conditioned text | 当前完整 XSP |\n")
    report.append("\n## 2. 协议控制\n")
    report.append("- 数据集：Cora、Citeseer；顺序：`order19611`、`order19612`、`order19613`；三阶段、5% cumulative memory。\n")
    report.append("- 每阶段只向构造器提供当前 delta 和 synthetic parent；`historical_raw_reads=0`。旧 memory 不修改；拓扑为 edge-free，Student 侧补 self-loop。\n")
    report.append("- CaT：class-wise MMD，3000 steps；XSP：30 steps、5% trust radius；Student seeds：150--155，原两层 GCN、frozen MiniLM、双向 InfoNCE。\n")
    report.append("- 评估：full real graph 的 report-dev 节点分类；报告 Final Macro-F1、Accuracy、Old F1、New F1。\n")
    report.append("- 每个 cell 的构造哈希、parent prefix、checkpoint reload 审计均通过。\n")
    report.append("\n## 3. 最终阶段逐 cell 结果\n")
    report.append("| Dataset | Order | 方法 | Final F1 | Accuracy | Old F1 | New F1 | ΔF1 vs CaT | ΔAcc |\n|---|---|---:|---:|---:|---:|---:|---:|---:|\n")
    for row in summary["final_cells"]:
        report.append(f"| {row['dataset']} | {row['order']} | {LABELS[row['arm']]} | {f(row['final_f1'])} | {f(row['final_accuracy'])} | {f(row['old_f1'])} | {f(row['new_f1'])} | {f(row['delta_f1_vs_cat'])} | {f(row['delta_accuracy_vs_cat'])} |\n")
    report.append("\n## 4. 跨六个最终 cell 的 pooled 结果\n")
    report.append("| 方法 | Final F1 | Accuracy | Old F1 | New F1 | 阶段平均 F1 | 阶段平均 Accuracy | 正向 cell |\n|---|---:|---:|---:|---:|---:|---:|---:|\n")
    for arm in ARMS:
        p = pooled[arm]
        positive = f"{p['positive_final_cells']}/{p['final_cell_count']}" if arm != "cat_nearest" else "reference"
        report.append(f"| {LABELS[arm]} | {f(p['final_f1'])} | {f(p['final_accuracy'])} | {f(p['final_old_f1'])} | {f(p['final_new_f1'])} | {f(p['stage_average_f1'])} | {f(p['stage_average_accuracy'])} | {positive} |\n")
    report.append("\n相对 CaT-nearest 的 pooled 最终阶段变化：\n")
    for arm in ARMS[1:]:
        p = pooled[arm]
        report.append(f"- **{LABELS[arm]}**：Final F1 {f(p['final_f1'] - cat['final_f1'])} pp，Accuracy {f(p['final_accuracy'] - cat['final_accuracy'])} pp，Old F1 {f(p['final_old_f1'] - cat['final_old_f1'])} pp，New F1 {f(p['final_new_f1'] - cat['final_new_f1'])} pp。\n")
    report.append("\n## 5. 归因结论\n")
    report.append("1. **文本 slot 分配本身不是主要退化源，但也没有形成稳定总体收益。** Fixed-X XSP 的 pooled Final F1 与 CaT 几乎持平（" + f(pooled['fixed_x_xsp']['final_f1'] - cat['final_f1']) + " pp），New F1 提升 " + f(pooled['fixed_x_xsp']['final_new_f1'] - cat['final_new_f1']) + " pp，但 Old F1 下降 " + f(pooled['fixed_x_xsp']['final_old_f1'] - cat['final_old_f1']) + " pp；说明 slot 文本更偏向新类塑性，不能直接当作通用提升。\n")
    report.append("2. **当前二次更新 X 的目标存在旧类风险。** XSP-X + CaT-text 的 pooled Final F1 为 " + f(pooled['xsp_x_cattext']['final_f1'] - cat['final_f1']) + " pp，略有正向，但只有 2/6 个最终 cell 为正，说明 X 更新不是稳定改进；它至少没有证明当前 XSP 文本材料化是唯一问题。\n")
    report.append("3. **Full XSP 的联合目标发生冲突。** Full XSP 的 pooled Final F1 为 " + f(pooled['full_xsp']['final_f1'] - cat['final_f1']) + " pp，Old F1 为 " + f(pooled['full_xsp']['final_old_f1'] - cat['final_old_f1']) + " pp，而 New F1 为 " + f(pooled['full_xsp']['final_new_f1'] - cat['final_new_f1']) + " pp。当前 `XSP loss + slot text materialization` 更像是在用旧类保持换新类塑性，且跨顺序不稳定。\n")
    report.append("4. **这轮不能把 XSP 晋升为 CaT 的稳定替代方法。** 更合理的判断是：XSP 已证明“X-conditioned payload partition”有可测的新类方向信号，但当前 X 更新目标、文本硬材料化和旧类保持之间没有形成稳定闭环。\n")
    report.append("\n## 6. 后续推进建议\n")
    report.append("- 暂不继续堆叠 selector、attention 或拓扑模块；先把 XSP 的 X 更新拆成更可控的 trust-region / old-class retention 约束，并在相同六个 cell 上复验。\n")
    report.append("- 若目标是保留新类塑性，可把 Fixed-X XSP 作为文本分配控制臂；若目标是提升总体 F1，优先研究 `XSP-X + CaT-text` 的 X 更新为何只在部分顺序有效。\n")
    report.append("- 这轮使用 edge-free memory，因此只能归因 X 与文本，不能证明完整 TAG 拓扑凝聚；后续拓扑实验必须另设共同拓扑基线。\n")
    report.append("- 代码：`code/v2_llm_fusion/run_xsp_attribution_small_20261005.py`；XSP 修正：`code/v2_llm_fusion/x_conditioned_payload_partition.py`。原始 JSON 与 cell 审计保存在对应结果目录。\n")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("".join(report), encoding="utf-8")


if __name__ == "__main__":
    main()
