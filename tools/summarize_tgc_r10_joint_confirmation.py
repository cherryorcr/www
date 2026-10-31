"""Summarize the sealed R10 joint-confirmation factorial."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


CORE_ARMS = ("c0", "content", "topology", "joint")


def load_records(root: Path):
    records = []
    for path in sorted(root.glob("*/order*/results.json")):
        dataset = path.parent.parent.name
        order = path.parent.name
        payload = json.loads(path.read_text(encoding="utf-8"))
        for item in payload["results"]:
            if item["arm"] not in CORE_ARMS:
                continue
            metric = item["metrics"]["all"]
            records.append({
                "dataset": dataset,
                "order": order,
                "stage": int(item["stage"]),
                "arm": item["arm"],
                "f1": float(metric["cumulative_f1"]["mean"]),
                "f1_std": float(metric["cumulative_f1"]["std"]),
                "old_f1": None if metric["old_f1"] is None else float(metric["old_f1"]["mean"]),
                "new_f1": None if metric["new_f1"] is None else float(metric["new_f1"]["mean"]),
            })
    if not records:
        raise SystemExit(f"no core-arm results under {root}")
    return records


def mean(values):
    values = [float(v) for v in values if v is not None]
    return sum(values) / len(values) if values else None


def keyed(records):
    return {(r["dataset"], r["order"], r["stage"], r["arm"]): r for r in records}


def delta_table(records, stage_filter):
    by = keyed(records)
    keys = sorted({(r["dataset"], r["order"], r["stage"])
                   for r in records if r["stage"] in stage_filter})
    rows = []
    for dataset, order, stage in keys:
        c0 = by[(dataset, order, stage, "c0")]
        row = {"dataset": dataset, "order": order, "stage": stage}
        row["c0_f1"] = c0["f1"]
        row["c0_old_f1"] = c0["old_f1"]
        row["c0_new_f1"] = c0["new_f1"]
        for arm in ("content", "topology", "joint"):
            current = by[(dataset, order, stage, arm)]
            row[f"{arm}_f1"] = current["f1"]
            row[f"{arm}_delta"] = current["f1"] - c0["f1"]
            row[f"{arm}_old_delta"] = (None if current["old_f1"] is None or c0["old_f1"] is None
                                        else current["old_f1"] - c0["old_f1"])
            row[f"{arm}_new_delta"] = (None if current["new_f1"] is None or c0["new_f1"] is None
                                        else current["new_f1"] - c0["new_f1"])
        row["interaction"] = (row["joint_delta"] - row["content_delta"]
                               - row["topology_delta"])
        rows.append(row)
    return rows


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value):
    return "NA" if value is None else f"{value:.2f}"


def markdown(summary, final_rows, stage_rows):
    lines = [
        "# TGC-R10 联合 content-topology 独立确认报告",
        "",
        "本报告只读取已 seal 的 Cora/Citeseer、order19601--19603 结果。主分析臂为 C0、content-only、topology-only 和 joint；recursive/maintenance/capacity 不混入本次四臂联合结论。",
        "",
        "## 1. 最终阶段主结果",
        "",
        "| 数据集 | 顺序 | C0 | content | topology | joint | joint-C0 | 交互项 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in final_rows:
        lines.append("| {dataset} | {order} | {c0_f1:.2f} | {content_f1:.2f} | {topology_f1:.2f} | {joint_f1:.2f} | {joint_delta:.2f} | {interaction:.2f} |".format(**row))
    lines += [
        "",
        "## 2. 汇总",
        "",
        "| 范围 | content-C0 | topology-C0 | joint-C0 | 联合交互项 |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, values in summary["aggregates"].items():
        lines.append(f"| {label} | {fmt(values['content_delta'])} | {fmt(values['topology_delta'])} | {fmt(values['joint_delta'])} | {fmt(values['interaction'])} |")
    final = summary["final_checks"]
    lines += [
        "",
        "## 3. 预注册推进条件",
        "",
        f"- 最终阶段 joint-C0 平均：`{final['mean_final_delta']:.2f} pp`，要求 `>= +0.50 pp`。",
        f"- stage 1/2 joint-C0 平均：`{final['mean_stage_delta']:.2f} pp`，要求 `>= 0 pp`。",
        f"- 最差最终 joint-C0：`{final['worst_final_delta']:.2f} pp`，要求 `>= -2 pp`。",
        f"- 最差最终 old-F1 差：`{final['worst_old_delta']:.2f} pp`，要求 `>= -2 pp`。",
        f"- 最终六个 dataset/order cell 中 joint-C0 >= -1 pp：`{final['cells_ge_minus1']}/6`，要求至少 `4/6`。",
        f"- 最终新类 F1 平均差：`{final['mean_new_delta']:.2f} pp`，要求 `>= 0 pp`。",
        f"- 推进条件整体通过：`{final['promotion_pass']}`。",
        "",
        "## 4. 解释",
        "",
        "本轮若出现单个或多个 joint 优于 C0 的 cell，只能说明联合机制存在条件性收益；是否能晋升为主方法取决于最终平均、最差条件、旧类保持和新类塑性是否同时满足。交互项为正时，说明 content 与 topology 的组合效果超过两个单组件差值的加和，但交互项本身不等于 joint 相对 C0 的绝对提升。",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    records = load_records(args.root)
    final_rows = delta_table(records, {2})
    stage_rows = delta_table(records, {1, 2})

    def aggregate(rows):
        result = {}
        for scope, subset in (
            ("all_final_cells", final_rows),
            ("cora_final", [r for r in final_rows if r["dataset"] == "cora"]),
            ("citeseer_final", [r for r in final_rows if r["dataset"] == "citeseer"]),
            ("all_incremental_stages", stage_rows),
        ):
            result[scope] = {
                "content_delta": mean(r["content_delta"] for r in subset),
                "topology_delta": mean(r["topology_delta"] for r in subset),
                "joint_delta": mean(r["joint_delta"] for r in subset),
                "interaction": mean(r["interaction"] for r in subset),
            }
        return result

    final = {
        "mean_final_delta": mean(r["joint_delta"] for r in final_rows),
        "mean_stage_delta": mean(r["joint_delta"] for r in stage_rows),
        "worst_final_delta": min(r["joint_delta"] for r in final_rows),
        "worst_old_delta": min(r["joint_old_delta"] for r in final_rows),
        "cells_ge_minus1": sum(r["joint_delta"] >= -1.0 for r in final_rows),
        "mean_new_delta": mean(r["joint_new_delta"] for r in final_rows),
    }
    final["promotion_pass"] = bool(
        final["mean_final_delta"] >= .5
        and final["mean_stage_delta"] >= 0
        and final["worst_final_delta"] >= -2
        and final["worst_old_delta"] >= -2
        and final["cells_ge_minus1"] >= 4
        and final["mean_new_delta"] >= 0
    )
    summary = {"records": len(records), "aggregates": aggregate(records), "final_checks": final}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(args.out / "final_cells.csv", final_rows)
    write_csv(args.out / "stage_cells.csv", stage_rows)
    (args.out / "report.md").write_text(markdown(summary, final_rows, stage_rows), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
