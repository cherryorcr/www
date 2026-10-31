"""Build the strict public-baseline matrix for the CaT-TAG payload study.

The current class-shared payload run is paired with the already audited
four-order baseline matrix.  The report deliberately separates strict
current-delta-only comparisons from older canonical/reference results.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


DATASETS = ("cora", "citeseer")
ORDERS = ("order19611", "order19612", "order19613", "order19614")
SEEDS = (150, 151, 152, 153, 154, 155)

OLD_NAMES = {
    "cat_mmd_tag": "CaT-TAG nearest",
    "gecc_hard_tag": "GECC-hard-TAG adapter",
    "k_center_native": "k-center native append",
    "medoid_native": "medoid native append",
}

DISPLAY = {
    "cat_payload_cctp": "CaT-TAG CCTP",
    "cat_payload_class_shared_tagsam": "CaT-TAG class-shared real payload",
    **OLD_NAMES,
}


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def mean_std(values):
    values = np.asarray(values, dtype=float)
    return float(values.mean()), float(values.std(ddof=1))


def current_cell_rows(root: Path):
    result = read(root / "results.json")
    rows = result["rows"]
    grouped = {}
    for row in rows:
        key = (row["dataset"], row["order"], row["arm"])
        grouped.setdefault(key, []).append(row)

    cells = []
    for (dataset, order, arm), trajectory in sorted(grouped.items()):
        if dataset not in DATASETS or order not in ORDERS:
            raise ValueError(f"unexpected current cell: {dataset}/{order}")
        trajectory = sorted(trajectory, key=lambda r: int(r["stage"]))
        if [int(r["stage"]) for r in trajectory] != [0, 1, 2]:
            raise ValueError(f"incomplete stages for {dataset}/{order}/{arm}")
        for row in trajectory:
            seeds = [int(s["seed"]) for s in row["per_student"]]
            if tuple(seeds) != SEEDS:
                raise ValueError(f"seed mismatch for {dataset}/{order}/{arm}")
        final = trajectory[-1]
        f1_mean, f1_std = mean_std(
            [s["cumulative_f1"] for s in final["per_student"]])
        acc_mean, acc_std = mean_std(
            [s["accuracy"] for s in final["per_student"]])
        cells.append({
            "dataset": dataset,
            "order": order,
            "arm": arm,
            "method": DISPLAY.get(arm, arm),
            "final_f1": f1_mean,
            "final_f1_std": f1_std,
            "final_accuracy": acc_mean,
            "final_accuracy_std": acc_std,
            "stage_avg_f1": float(np.mean([
                r["aggregate"]["mean_f1"] for r in trajectory])),
            "stage_avg_accuracy": float(np.mean([
                r["aggregate"]["mean_accuracy"] for r in trajectory])),
            "old_f1": float(np.mean([
                s["old_f1"] for s in final["per_student"]])),
            "new_f1": float(np.mean([
                s["new_f1"] for s in final["per_student"]])),
            "worst_stage_f1": float(min(
                r["aggregate"]["mean_f1"] for r in trajectory)),
            "memory_rows": int(final["memory_rows"]),
            "construction_seconds": float(sum(
                r["construction_seconds"] for r in trajectory)),
            "student_seconds_per_seed": float(sum(
                r["student_wall_seconds"] for r in trajectory) / len(SEEDS)),
        })
    return cells, result["protocol"]


def old_cells(path: Path):
    rows = list(csv.DictReader(path.open(encoding="utf-8", newline="")))
    selected = []
    for row in rows:
        arm = row["arm"]
        if arm not in OLD_NAMES:
            continue
        item = {k: row[k] for k in ("dataset", "order")}
        item.update({
            "arm": arm,
            "method": OLD_NAMES[arm],
            "final_f1": float(row["final_f1"]),
            "final_f1_std": float(row["final_f1_std"]),
            "final_accuracy": float(row["final_accuracy"]),
            "final_accuracy_std": float(row["final_accuracy_std"]),
            "stage_avg_f1": float(row["stage_avg_f1"]),
            "stage_avg_accuracy": float(row["stage_avg_accuracy"]),
            "old_f1": float(row["old_f1"]),
            "new_f1": float(row["new_f1"]),
            "worst_stage_f1": float(row["worst_stage_f1"]),
            "memory_rows": int(float(row["memory_rows"])),
            "construction_seconds": float(row["construction_seconds"]),
            "student_seconds_per_seed": float(row["student_seconds_per_seed"]),
        })
        selected.append(item)
    expected = {(d, o, a) for d in DATASETS for o in ORDERS
                for a in OLD_NAMES}
    actual = {(r["dataset"], r["order"], r["arm"]) for r in selected}
    if actual != expected:
        raise ValueError(f"old strict matrix coverage mismatch: {sorted(expected - actual)}")
    return selected


def aggregate(cells):
    result = []
    arms = sorted({r["arm"] for r in cells})
    for arm in arms:
        items = [r for r in cells if r["arm"] == arm]
        result.append({
            "arm": arm,
            "method": DISPLAY.get(arm, items[0]["method"]),
            "cells": len(items),
            "final_f1": float(np.mean([r["final_f1"] for r in items])),
            "final_accuracy": float(np.mean([r["final_accuracy"] for r in items])),
            "stage_avg_f1": float(np.mean([r["stage_avg_f1"] for r in items])),
            "stage_avg_accuracy": float(np.mean([r["stage_avg_accuracy"] for r in items])),
            "old_f1": float(np.mean([r["old_f1"] for r in items])),
            "new_f1": float(np.mean([r["new_f1"] for r in items])),
            "worst_stage_f1": float(min(r["worst_stage_f1"] for r in items)),
            "construction_seconds": float(np.mean([
                r["construction_seconds"] for r in items])),
            "student_seconds_per_seed": float(np.mean([
                r["student_seconds_per_seed"] for r in items])),
        })
    return result


def paired(cells, candidate):
    index = {(r["dataset"], r["order"], r["arm"]): r for r in cells}
    candidate_rows = [r for r in cells if r["arm"] == candidate]
    arms = sorted({r["arm"] for r in cells if r["arm"] != candidate})
    out = []
    for arm in arms:
        deltas = []
        acc_deltas = []
        for r in candidate_rows:
            b = index[r["dataset"], r["order"], arm]
            deltas.append(r["final_f1"] - b["final_f1"])
            acc_deltas.append(r["final_accuracy"] - b["final_accuracy"])
        out.append({
            "baseline_arm": arm,
            "baseline": DISPLAY.get(arm, arm),
            "delta_final_f1_mean": float(np.mean(deltas)),
            "delta_final_accuracy_mean": float(np.mean(acc_deltas)),
            "positive_f1_cells": int(sum(d > 0 for d in deltas)),
            "positive_accuracy_cells": int(sum(d > 0 for d in acc_deltas)),
            "worst_f1_delta": float(min(deltas)),
            "best_f1_delta": float(max(deltas)),
            "deltas": [float(d) for d in deltas],
        })
    return out


def fmt(value):
    return "N/A" if value is None else f"{value:.2f}"


def write_report(out: Path, payload):
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    with (out / "strict_cells.csv").open("w", newline="", encoding="utf-8") as f:
        fields = list(payload["strict_cells"][0])
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(payload["strict_cells"])

    lines = [
        "# CaT-TAG 统一公开基线比较（四顺序严格矩阵）", "",
        "本报告把最新类共享真实 payload 与已审计四顺序基线对齐。主表只包含：",
        "5% cumulative memory、current-delta-only、历史 synthetic prefix 不改写、",
        "同一两层 GCN + 冻结文本编码器 + 双向 InfoNCE Student、Student seeds 150--155、",
        "相同三阶段和四个类别顺序。CaT/GECC 是机制适配，不是官方端到端结果。", "",
        "## 1. 四顺序等权主表", "",
        "|方法|Final F1|Accuracy|Stage-average F1|Old F1|New F1|最差 stage|构造秒|",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["strict_overall"]:
        lines.append("|{}|{}|{}|{}|{}|{}|{}|{}|".format(
            row["method"], fmt(row["final_f1"]), fmt(row["final_accuracy"]),
            fmt(row["stage_avg_f1"]), fmt(row["old_f1"]), fmt(row["new_f1"]),
            fmt(row["worst_stage_f1"]), fmt(row["construction_seconds"])))

    lines += ["", "## 2. 最新方法相对各基线", "",
              "正数表示类共享真实 payload 更高；四个 cell 为 Cora/Citeseer × 四顺序。", "",
              "|对照|Δ Final F1|Δ Accuracy|正向 F1 cell|最差 Δ|", "|---|---:|---:|---:|---:|"]
    for row in payload["paired"]:
        lines.append("|{}|{:+.2f}|{:+.2f}|{}/8|{:+.2f}|".format(
            row["baseline"], row["delta_final_f1_mean"],
            row["delta_final_accuracy_mean"], row["positive_f1_cells"],
            row["worst_f1_delta"]))

    lines += ["", "## 3. 每个 dataset/order 的 Final F1", "",
              "|数据集|顺序|" + "|".join(
                  DISPLAY.get(a, a) for a in payload["arms"]) + "|",
              "|---|---|" + "|".join("---:" for _ in payload["arms"]) + "|"]
    by_key = {(r["dataset"], r["order"], r["arm"]): r
              for r in payload["strict_cells"]}
    for d in DATASETS:
        for o in ORDERS:
            values = [by_key[d, o, a]["final_f1"] for a in payload["arms"]]
            lines.append("|{}|{}|{}|".format(
                d, o, "|".join(f"{v:.2f}" for v in values)))

    lines += ["", "## 4. 协议分层和未混入主表的结果", "",
              "- `random_real`、full-history re-condensation、raw reference 和 direct Student continual 的旧结果已有记录，但它们来自 canonical/旧三阶段或不同任务状态，不能直接拼进本表。",
              "- 它们应作为 reference/control appendix 单列；本报告不把它们伪装成四顺序严格基线。",
              "- 当前主表中的 `CaT-TAG nearest`、GECC、k-center、medoid 来自已审计的四顺序矩阵；最新类共享版本由本轮重新跑出的同协议矩阵提供。", "",
              "## 5. 结论", "",
              "类共享真实 payload 是否进入主方法，应看它相对 CCTP 和最强 coreset 的 paired cell 稳定性，而不是只看 pooled 均值。该表用于完成第一阶段基线核对，不宣称普遍 SOTA。"]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--current-root", type=Path, required=True)
    ap.add_argument("--old-cells", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    current, protocol = current_cell_rows(args.current_root)
    old = old_cells(args.old_cells)
    current_cctp = {(r["dataset"], r["order"]): r for r in current
                    if r["arm"] == "cat_payload_cctp"}
    current_candidate = {(r["dataset"], r["order"]): r for r in current
                         if r["arm"] == "cat_payload_class_shared_tagsam"}
    if len(current_cctp) != 8 or len(current_candidate) != 8:
        raise ValueError("current matrix must contain 8 CCTP and 8 candidate cells")

    strict = []
    for key, row in sorted(current_cctp.items()):
        strict.append(row)
    for key, row in sorted(current_candidate.items()):
        strict.append(row)
    for row in old:
        if row["arm"] == "cat_mmd_tag":
            strict.append(row)
        elif row["arm"] in ("gecc_hard_tag", "k_center_native", "medoid_native"):
            strict.append(row)

    arms = ["cat_payload_cctp", "cat_payload_class_shared_tagsam",
            "cat_mmd_tag", "gecc_hard_tag", "k_center_native", "medoid_native"]
    if {(r["dataset"], r["order"], r["arm"]) for r in strict} != {
        (d, o, a) for d in DATASETS for o in ORDERS for a in arms
    }:
        raise ValueError("strict matrix coverage mismatch")
    strict = sorted(strict, key=lambda r: (r["dataset"], r["order"], arms.index(r["arm"])))
    candidate = "cat_payload_class_shared_tagsam"
    payload = {
        "protocol": protocol,
        "arms": arms,
        "strict_cells": strict,
        "strict_overall": aggregate(strict),
        "paired": paired(strict, candidate),
        "limits": [
            "four category orders, two datasets, one construction seed",
            "CaT and GECC are mechanism-adapted interfaces, not official end-to-end results",
            "construction time is comparable only within the same runner; sealed or reused artifacts are not zero-cost constructors",
        ],
    }
    write_report(args.output, payload)
    print(json.dumps({
        "output": str(args.output),
        "strict_cells": len(strict),
        "strict_overall": payload["strict_overall"],
        "paired": payload["paired"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
