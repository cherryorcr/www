"""Summarize the four-order CaT-TAG/CCTP baseline matrix.

The matrix is a protocol-level adapter comparison, not an official end-to-end
reproduction of CaT or GECC.  The script keeps dataset/order cells explicit so
that a pooled mean cannot hide order sensitivity.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats


ARMS = [
    "cat_mmd_tag",
    "cat_mmd_tag_text_center",
    "gecc_hard_tag",
    "k_center_native",
    "medoid_native",
]
LABELS = {
    "cat_mmd_tag": "CaT-TAG nearest",
    "cat_mmd_tag_text_center": "CaT-TAG CCTP (text-center)",
    "gecc_hard_tag": "GECC-hard-TAG adapter",
    "k_center_native": "k-center native append",
    "medoid_native": "medoid native append",
}


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def ci(values):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    if len(values) < 2:
        return {"mean": mean, "low": None, "high": None, "n": len(values)}
    half = float(stats.t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))
    return {"mean": mean, "low": mean - half, "high": mean + half, "n": len(values)}


def finite_mean(values):
    values = [float(v) for v in values if v is not None]
    return None if not values else float(np.mean(values))


def load_dataset(path: Path, expected_dataset: str):
    payload = read(path)
    protocol = payload["protocol"]
    if protocol["datasets"] != [expected_dataset]:
        raise ValueError(f"unexpected dataset in {path}: {protocol['datasets']}")
    if protocol["arms"] != ARMS:
        raise ValueError(f"unexpected arms in {path}: {protocol['arms']}")
    if protocol["orders"] != ["order19611", "order19612", "order19613", "order19614"]:
        raise ValueError(f"unexpected orders in {path}: {protocol['orders']}")
    if protocol["report_seeds"] != [150, 151, 152, 153, 154, 155]:
        raise ValueError("report seed mismatch")
    by_key = {}
    for row in payload["rows"]:
        key = (row["dataset"], row["order"], row["arm"], int(row["stage"]))
        if key in by_key:
            raise ValueError(f"duplicate row {key}")
        if row["arm"] not in ARMS:
            raise ValueError(f"unexpected arm {row['arm']}")
        seeds = [s["seed"] for s in row["per_student"]]
        if seeds != protocol["report_seeds"]:
            raise ValueError(f"seed mismatch in {key}: {seeds}")
        aggregate = row["aggregate"]
        for source, saved in (("cumulative_f1", aggregate["mean_f1"]),
                              ("accuracy", aggregate["mean_accuracy"])):
            values = np.asarray([s[source] for s in row["per_student"]], dtype=float)
            if not np.isfinite(values).all() or abs(float(values.mean()) - float(saved)) > 1e-8:
                raise ValueError(f"aggregate mismatch in {key}")
        by_key[key] = row
    expected = {(expected_dataset, order, arm, stage)
                for order in protocol["orders"] for arm in ARMS for stage in range(3)}
    if set(by_key) != expected:
        missing = sorted(expected - set(by_key))
        extra = sorted(set(by_key) - expected)
        raise ValueError(f"coverage mismatch missing={missing} extra={extra}")
    return protocol, by_key


def summarize_cell(dataset, order, arm, by_key, report_seeds):
    trajectory = [by_key[(dataset, order, arm, stage)] for stage in range(3)]
    final = trajectory[-1]
    scores = final["per_student"]
    f1 = np.asarray([s["cumulative_f1"] for s in scores], dtype=float)
    acc = np.asarray([s["accuracy"] for s in scores], dtype=float)
    old = np.asarray([s["old_f1"] for s in scores], dtype=float)
    new = np.asarray([s["new_f1"] for s in scores], dtype=float)
    return {
        "dataset": dataset,
        "order": order,
        "arm": arm,
        "method": LABELS[arm],
        "final_f1": float(f1.mean()),
        "final_f1_std": float(f1.std(ddof=1)),
        "final_accuracy": float(acc.mean()),
        "final_accuracy_std": float(acc.std(ddof=1)),
        "stage_avg_f1": float(np.mean([r["aggregate"]["mean_f1"] for r in trajectory])),
        "stage_avg_accuracy": float(np.mean([r["aggregate"]["mean_accuracy"] for r in trajectory])),
        "old_f1": float(old.mean()),
        "new_f1": float(new.mean()),
        "worst_stage_f1": float(min(r["aggregate"]["mean_f1"] for r in trajectory)),
        "memory_rows": int(final["memory_rows"]),
        "construction_seconds": float(sum(r["construction_seconds"] for r in trajectory
                                            if r["construction_seconds"] is not None))
        if all(r["construction_seconds"] is not None for r in trajectory) else None,
        "student_seconds_per_seed": float(sum(r["student_wall_seconds"] for r in trajectory)
                                           / len(report_seeds)),
        "text_tokens": int(final["resources"]["text_tokens"]),
        "edge_entries": int(final["resources"]["edge_entries"]),
        "per_seed_f1": f1.tolist(),
        "per_seed_accuracy": acc.tolist(),
    }


def aggregate(rows, group_keys):
    groups = {}
    for row in rows:
        key = tuple(row[k] for k in group_keys)
        groups.setdefault(key, []).append(row)
    result = []
    for key, items in sorted(groups.items()):
        def mean(field):
            return float(np.mean([x[field] for x in items]))

        def std(field):
            values = [x[field] for x in items]
            return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0

        out = {k: v for k, v in zip(group_keys, key)}
        out.update({
            "method": items[0]["method"],
            "cells": len(items),
            "final_f1": mean("final_f1"),
            "final_f1_order_std": std("final_f1"),
            "final_accuracy": mean("final_accuracy"),
            "stage_avg_f1": mean("stage_avg_f1"),
            "stage_avg_accuracy": mean("stage_avg_accuracy"),
            "old_f1": mean("old_f1"),
            "new_f1": mean("new_f1"),
            "worst_stage_f1": min(x["worst_stage_f1"] for x in items),
            "memory_rows": mean("memory_rows"),
            "construction_seconds": finite_mean([x["construction_seconds"] for x in items]),
            "student_seconds_per_seed": mean("student_seconds_per_seed"),
            "text_tokens": mean("text_tokens"),
        })
        result.append(out)
    return result


def comparisons(cell_rows, group_key):
    result = []
    groups = sorted(set(r[group_key] for r in cell_rows))
    for group in groups:
        local = [r for r in cell_rows if r[group_key] == group]
        for baseline in ARMS:
            if baseline == "cat_mmd_tag_text_center":
                continue
            cctp = {r["order"]: r for r in local if r["arm"] == "cat_mmd_tag_text_center"}
            base = {r["order"]: r for r in local if r["arm"] == baseline}
            orders = sorted(set(cctp) & set(base))
            f1_delta = [cctp[o]["final_f1"] - base[o]["final_f1"] for o in orders]
            acc_delta = [cctp[o]["final_accuracy"] - base[o]["final_accuracy"] for o in orders]
            stage_delta = [cctp[o]["stage_avg_f1"] - base[o]["stage_avg_f1"] for o in orders]
            result.append({
                group_key: group,
                "baseline_arm": baseline,
                "baseline": LABELS[baseline],
                "cells": len(orders),
                "final_f1_delta": ci(f1_delta),
                "final_accuracy_delta": ci(acc_delta),
                "stage_avg_f1_delta": ci(stage_delta),
                "positive_f1_cells": int(sum(x > 0 for x in f1_delta)),
                "positive_accuracy_cells": int(sum(x > 0 for x in acc_delta)),
                "per_order_f1_delta": dict(zip(orders, f1_delta)),
            })
    return result


def fmt(value, digits=2):
    return "N/A" if value is None else f"{value:.{digits}f}"


def fmt_ci(value):
    if value["low"] is None:
        return fmt(value["mean"])
    return f"{value['mean']:+.2f} [{value['low']:+.2f}, {value['high']:+.2f}]"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cora-results", type=Path, required=True)
    parser.add_argument("--citeseer-results", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    protocols = []
    cells = []
    for dataset, path in (("cora", args.cora_results), ("citeseer", args.citeseer_results)):
        protocol, by_key = load_dataset(path, dataset)
        protocols.append(protocol)
        for order in protocol["orders"]:
            audit_path = args.audit_dir / f"{dataset}_{order}_audit.json"
            audit = read(audit_path)
            if not audit.get("passed") or audit.get("maximum_metric_error_pp") != 0.0:
                raise ValueError(f"audit failed or reload drifted: {audit_path}")
            for arm in ARMS:
                cells.append(summarize_cell(dataset, order, arm, by_key,
                                            protocol["report_seeds"]))

    pooled_by_dataset = aggregate(cells, ["dataset", "arm"])
    pooled_all = aggregate(cells, ["arm"])
    by_dataset_comparisons = comparisons(cells, "dataset")
    for row in cells:
        row["__all__"] = "all"
    all_comparisons = comparisons(cells, "__all__")
    for row in cells:
        row.pop("__all__", None)

    protocol_note = {
        "status": protocols[0]["status"],
        "datasets": ["cora", "citeseer"],
        "orders": protocols[0]["orders"],
        "report_seeds": protocols[0]["report_seeds"],
        "budget": protocols[0]["budget"],
        "student": protocols[0]["student"],
        "construction": protocols[0]["construction"],
        "test_labels_evaluated": protocols[0]["test_labels_evaluated"],
        "topology": protocols[0]["topology"],
        "cctp_definition": "CaT class-balanced MMD X plus current-delta class text-center selector, fixed top-5 MMR and 320-character re-encoding",
        "comparison_scope": "four category orders, six paired Student seeds, one construction seed",
    }
    payload = {
        "protocol": protocol_note,
        "cells": cells,
        "pooled_by_dataset": pooled_by_dataset,
        "pooled_all": pooled_all,
        "cctp_vs_baselines_by_dataset": by_dataset_comparisons,
        "cctp_vs_baselines_all": all_comparisons,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    with (args.output / "cells.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = [k for k in cells[0] if not k.startswith("per_seed_")]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: row[k] for k in fields} for row in cells)
    with (args.output / "pooled_by_dataset.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = list(pooled_by_dataset[0])
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(pooled_by_dataset)

    lines = [
        "# CaT-TAG CCTP 四顺序公开方法适配基线矩阵",
        "",
        "本表使用统一持续 TAG 协议：5% cumulative memory、current-delta-only、历史 synthetic prefix 不修改、原有两层 GCN + text projection + 双向 InfoNCE Student、六个 paired Student seeds。CaT、GECC 是源代码机制适配，不是其官方端到端论文结果。",
        "",
        "## 1. 数据集/顺序均值",
        "",
        "| 数据集 | 方法 | Final Macro-F1 | Accuracy | Stage-average F1 | Old F1 | New F1 | 跨顺序标准差 | 构造秒 | Student 秒/seed |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in pooled_by_dataset:
        lines.append("|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|".format(
            r["dataset"], r["method"], fmt(r["final_f1"]), fmt(r["final_accuracy"]),
            fmt(r["stage_avg_f1"]), fmt(r["old_f1"]), fmt(r["new_f1"]),
            fmt(r["final_f1_order_std"]), fmt(r["construction_seconds"]),
            fmt(r["student_seconds_per_seed"])))
    lines += [
        "",
        "## 2. 全部八个 dataset-order cell 等权均值",
        "",
        "| 方法 | Final Macro-F1 | Accuracy | Stage-average F1 | Old F1 | New F1 | 最差 stage F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in pooled_all:
        lines.append("|{}|{}|{}|{}|{}|{}|{}|".format(
            r["method"], fmt(r["final_f1"]), fmt(r["final_accuracy"]),
            fmt(r["stage_avg_f1"]), fmt(r["old_f1"]), fmt(r["new_f1"]),
            fmt(r["worst_stage_f1"])))
    lines += [
        "",
        "## 3. CCTP 相对其它方法的 Final F1 差值",
        "",
        "正数表示 CCTP 更高。区间是按四个类别顺序的 cell 均值计算的描述性 95% t 区间，不把不同顺序当作完全独立数据集。",
        "",
        "| 数据集 | 对照方法 | Δ Final F1 | 正向 cell | Δ Accuracy | Δ Stage-average F1 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in by_dataset_comparisons:
        f = r["final_f1_delta"]
        a = r["final_accuracy_delta"]
        s = r["stage_avg_f1_delta"]
        lines.append("|{}|{}|{}|{}/{}|{}|{}|".format(
            r["dataset"], r["baseline"], fmt_ci(f), r["positive_f1_cells"], r["cells"],
            fmt_ci(a), fmt_ci(s)))
    lines += [
        "",
        "## 4. 结论",
        "",
        "- CCTP 与 CaT-nearest 使用同一 CaT MMD X 构造，主要用于测量 TAG 文本原型的增益；两者差异不能解释为 X 凝聚算法差异。",
        "- GECC、k-center、medoid 的结果是统一 Student/预算下的适配参照，不能直接写成官方方法论文结果。",
        "- 只有在同一数据集和同一顺序内比较，才可把差异归因于方法；跨数据集 pooled 均值只用于描述总体趋势。",
        "- 若 CCTP 在四顺序上只改善部分数据集，应把贡献表述为 TAG 文本原型对 CaT 骨架的可验证增益，而不是普遍 SOTA。",
        "",
        "## 5. 审计",
        "",
        "八个 dataset-order cell 的 audit 均通过；checkpoint reload 最大指标误差为 0.0 pp；test label 未用于选择。",
        "",
    ]
    (args.output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"cells": len(cells), "pooled": pooled_all,
                      "comparisons": by_dataset_comparisons}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
