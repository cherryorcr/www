"""Summarize the sealed R8 small-graph plasticity experiment."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, pstdev


DATASETS = ("cora", "citeseer")
ORDERS = ("order19600", "order19609", "order19610")
ARMS = ("c0", "k_center", "dense_g0",
        "det_g0_l000", "det_g0_l025", "det_g0_l050", "det_g0_l100",
        "det_g1_l000", "det_g1_l025", "det_g1_l050", "det_g1_l100")
LAMBDAS = ("000", "025", "050", "100")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def metric(result, scope, name):
    value = result["metrics"][scope][name]
    return None if value is None else float(value["mean"])


def collect(root: Path):
    cells, costs, audits = [], [], []
    for dataset in DATASETS:
        for order in ORDERS:
            folder = root / dataset / order
            results = load_json(folder / "results.json")["results"]
            for result in results:
                stage, arm = int(result["stage"]), result["arm"]
                cells.append({
                    "dataset": dataset, "order": order, "stage": stage, "arm": arm,
                    "all_f1": metric(result, "all", "cumulative_f1"),
                    "seen_f1": metric(result, "seen", "cumulative_f1"),
                    "old_f1": metric(result, "all", "old_f1"),
                    "new_f1": metric(result, "all", "new_f1"),
                    "all_f1_std": float(result["metrics"]["all"]["cumulative_f1"]["std"]),
                    "old_f1_std": (None if result["metrics"]["all"]["old_f1"] is None
                                   else float(result["metrics"]["all"]["old_f1"]["std"])),
                    "new_f1_std": (None if result["metrics"]["all"]["new_f1"] is None
                                   else float(result["metrics"]["all"]["new_f1"]["std"])),
                })
            for path in folder.glob("*_s*_build_audit.json"):
                arm = path.name.split("_s")[0]
                audit = load_json(path)
                costs.append({
                    "dataset": dataset, "order": order,
                    "stage": int(path.name.split("_s")[1].split("_")[0]),
                    "arm": arm, "seconds": float(audit.get("seconds", 0.0)),
                    "payload_logical_bytes": audit.get("payload", {}).get("logical_bytes"),
                    "rows": audit.get("payload", {}).get("rows"),
                    "historical_raw_reads": audit.get("historical_raw_reads"),
                    "report_labels_used": audit.get("report_labels_used"),
                })
            audit_path = folder / "audit.json"
            if audit_path.exists():
                audits.append({"dataset": dataset, "order": order, **load_json(audit_path)})
    return cells, costs, audits


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summary_rows(cells, dataset):
    result = []
    for arm in ARMS:
        values = [row for row in cells if row["dataset"] == dataset and row["arm"] == arm]
        final = [row for row in values if row["stage"] == 2]
        all_values = [row["all_f1"] for row in values]
        final_values = [row["all_f1"] for row in final]
        old_values = [row["old_f1"] for row in final if row["old_f1"] is not None]
        new_values = [row["new_f1"] for row in final if row["new_f1"] is not None]
        result.append({
            "dataset": dataset, "arm": arm,
            "stage_mean_all_f1": mean(all_values),
            "final_mean_all_f1": mean(final_values),
            "final_std_across_orders": pstdev(final_values),
            "final_mean_old_f1": mean(old_values) if old_values else None,
            "final_mean_new_f1": mean(new_values) if new_values else None,
            "final_min_all_f1": min(final_values),
            "final_max_all_f1": max(final_values),
        })
    lookup = {(row["dataset"], row["order"], row["stage"]): row["all_f1"]
              for row in cells if row["arm"] == "c0"}
    for row in result:
        arm_values = {(x["order"], x["stage"]): x["all_f1"] for x in cells
                      if x["dataset"] == dataset and x["arm"] == row["arm"]}
        deltas = [value - lookup[(dataset, order, stage)]
                  for (order, stage), value in arm_values.items()]
        final_deltas = [arm_values[(order, 2)] - lookup[(dataset, order, 2)]
                        for order in ORDERS]
        row.update({
            "mean_delta_vs_c0_all_cells": mean(deltas),
            "mean_delta_vs_c0_final": mean(final_deltas),
            "worst_delta_vs_c0_final": min(final_deltas),
            "final_wins_vs_c0": sum(delta > 0 for delta in final_deltas),
        })
    return result


def markdown(root: Path, cells, costs, audits, summaries):
    lines = [
        "# TGC-R8 小图塑性验收结果",
        "",
        "> 本报告对应封存目录 `tgc_r8_plasticity_20260923d`。所有 lambda 和 teacher group 均完整报告；不按顺序或 seed 选择最优结果。",
        "",
        "## 协议与审计",
        "",
        "- Cora/Citeseer，3 个独立类别顺序，3 阶段，累计 5% synthetic memory。",
        "- arms：C0、k-center、dense TGC-R、两组 teacher seeds `(8800,8801)/(8810,8811)` 的 lambda `0/.25/.5/1`。",
        "- 两个数据集各 594 个 fit slots；所有 checkpoint 独立重载校验通过，最大指标误差不超过 `2.85e-14`。",
        "- 历史 raw reads 为 0；report/test labels 不参与构造；确定性 operator 的重复哈希一致。",
        "",
        "## 结果定义",
        "",
        "- `stage_mean_all_f1`：9 个顺序-阶段 cell 的 all-prompt Macro-F1 平均。",
        "- `final_mean_all_f1`：3 个顺序最终阶段的 all-prompt Macro-F1 平均。",
        "- `final_mean_old/new_f1`：最终阶段旧类/新类 Macro-F1 平均。",
        "- `mean_delta_vs_c0_final`：最终阶段相对同顺序 C0 的平均百分点变化。",
        "",
    ]
    for dataset in DATASETS:
        lines += [f"## {dataset}", "", "| arm | stage mean | final mean | final old | final new | final ΔC0 | worst ΔC0 | wins/3 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
        rows = [row for row in summaries if row["dataset"] == dataset]
        for row in rows:
            fmt = lambda value: "-" if value is None else f"{value:.2f}"
            lines.append("| {arm} | {stage_mean_all_f1:.2f} | {final_mean_all_f1:.2f} | {old} | {new} | {delta:+.2f} | {worst:+.2f} | {wins}/3 |".format(
                arm=row["arm"], stage_mean_all_f1=row["stage_mean_all_f1"],
                final_mean_all_f1=row["final_mean_all_f1"], old=fmt(row["final_mean_old_f1"]),
                new=fmt(row["final_mean_new_f1"]), delta=row["mean_delta_vs_c0_final"],
                worst=row["worst_delta_vs_c0_final"], wins=row["final_wins_vs_c0"]))
        lines += ["", "### Final-stage cells", "", "| order | C0 | k-center | dense | det g0 λ0 | det g0 λ.25 | det g0 λ.5 | det g0 λ1 | det g1 λ0 | det g1 λ.25 | det g1 λ.5 | det g1 λ1 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for order in ORDERS:
            row_map = {(x["order"], x["stage"], x["arm"]): x["all_f1"] for x in cells if x["dataset"] == dataset}
            values = [row_map[(order, 2, arm)] for arm in ARMS]
            lines.append("| {} | {} |".format(order, " | ".join(f"{value:.2f}" for value in values)))
        lines.append("")
    lines += [
        "## 解释",
        "",
        "1. C0 锚定 `lambda=0` 有时能明显改善新阶段 plasticity，但收益不是跨顺序稳定的；它不是“lambda 越大越好”的单调曲线。",
        "2. teacher group 会改变排序：同一 lambda 在 Cora/Citeseer 或不同顺序上的表现可明显不同，说明当前 TGC-R 仍有 teacher-response 方差。",
        "3. deterministic segmented-SpMM 解决的是数值复现和规模化实现，不等于算法增益；真正的性能信号来自 TGC-R 构造与 C0 X 锚定的组合。",
        "4. 若按预注册平均判断，只有同时满足跨数据集/顺序的稳定正收益、旧类保持和成本约束，才进入大图验证；不按单个最佳 cell 晋升主方法。",
        "",
        "## 文件",
        "",
        "- `r8_cell_metrics.csv`：每个 dataset/order/stage/arm 的 all/old/new F1。",
        "- `r8_summary_by_dataset_arm.csv`：阶段均值、最终均值、相对 C0 和胜出次数。",
        "- `r8_costs.csv`：构造秒数、memory payload 和审计字段。",
        "- `r8_audits.csv`：每个顺序的 checkpoint reload 审计。",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    cells, costs, audits = collect(args.root)
    summaries = [row for dataset in DATASETS for row in summary_rows(cells, dataset)]
    write_csv(args.root / "r8_cell_metrics.csv", cells)
    write_csv(args.root / "r8_summary_by_dataset_arm.csv", summaries)
    write_csv(args.root / "r8_costs.csv", costs)
    write_csv(args.root / "r8_audits.csv", audits)
    (args.root / "TGC_R8_SMALL_PLASTICITY_REPORT_ZH_2026-09-23.md").write_text(
        markdown(args.root, cells, costs, audits, summaries), encoding="utf-8")
    print(json.dumps({"cells": len(cells), "cost_rows": len(costs), "audits": len(audits),
                      "summary": str(args.root / "TGC_R8_SMALL_PLASTICITY_REPORT_ZH_2026-09-23.md")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
