"""Summarize fixed-X CCTP top-k and soft-prototype controls."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats


ARMS = [
    "cat_mmd_tag_cctp_top1",
    "cat_mmd_tag_cctp_top3",
    "cat_mmd_tag_cctp_top5",
    "cat_mmd_tag_cctp_soft",
]
LABELS = {
    "cat_mmd_tag_cctp_top1": "CCTP top-1",
    "cat_mmd_tag_cctp_top3": "CCTP top-3",
    "cat_mmd_tag_cctp_top5": "CCTP top-5",
    "cat_mmd_tag_cctp_soft": "CCTP soft prototype",
}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def ci(values):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    if len(values) < 2:
        return {"mean": mean, "low": None, "high": None, "n": len(values)}
    half = float(stats.t.ppf(.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))
    return {"mean": mean, "low": mean - half, "high": mean + half, "n": len(values)}


def load(path, dataset, audit_dir):
    payload = read(path)
    protocol = payload["protocol"]
    if protocol.get("fixed_x") is not True:
        raise ValueError("control matrix is not fixed-X")
    if protocol["datasets"] != [dataset] or protocol["arms"] != ARMS:
        raise ValueError(f"protocol mismatch for {dataset}")
    if protocol["orders"] != ["order19611", "order19612", "order19613", "order19614"]:
        raise ValueError("order coverage mismatch")
    rows = {(r["order"], r["arm"], int(r["stage"])): r for r in payload["rows"]}
    expected = {(o, a, s) for o in protocol["orders"] for a in ARMS for s in range(3)}
    if set(rows) != expected:
        raise ValueError(f"row coverage mismatch for {dataset}")
    for order in protocol["orders"]:
        audit = read(audit_dir / f"{dataset}_{order}_audit.json")
        if not audit.get("passed") or audit.get("maximum_metric_error_pp") != 0.0:
            raise ValueError(f"audit failed for {dataset}/{order}")
    return protocol, rows


def cell(dataset, order, arm, rows):
    trajectory = [rows[(order, arm, s)] for s in range(3)]
    final = trajectory[-1]
    scores = final["per_student"]
    f1 = np.asarray([s["cumulative_f1"] for s in scores], dtype=float)
    acc = np.asarray([s["accuracy"] for s in scores], dtype=float)
    return {
        "dataset": dataset, "order": order, "arm": arm, "method": LABELS[arm],
        "final_f1": float(f1.mean()), "final_f1_std": float(f1.std(ddof=1)),
        "final_accuracy": float(acc.mean()),
        "stage_avg_f1": float(np.mean([r["aggregate"]["mean_f1"] for r in trajectory])),
        "stage_avg_accuracy": float(np.mean([r["aggregate"]["mean_accuracy"] for r in trajectory])),
        "old_f1": float(np.mean([s["old_f1"] for s in scores])),
        "new_f1": float(np.mean([s["new_f1"] for s in scores])),
        "worst_stage_f1": float(min(r["aggregate"]["mean_f1"] for r in trajectory)),
        "construction_seconds": float(sum(r["construction_seconds"] for r in trajectory)),
        "student_seconds_per_seed": float(sum(r["student_wall_seconds"] for r in trajectory) / len(scores)),
    }


def pooled(rows, keys):
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)
    out = []
    for key, items in sorted(groups.items()):
        item = {k: v for k, v in zip(keys, key)}
        item.update({
            "method": items[0]["method"], "cells": len(items),
            "final_f1": float(np.mean([r["final_f1"] for r in items])),
            "final_f1_order_std": float(np.std([r["final_f1"] for r in items], ddof=1)),
            "final_accuracy": float(np.mean([r["final_accuracy"] for r in items])),
            "stage_avg_f1": float(np.mean([r["stage_avg_f1"] for r in items])),
            "stage_avg_accuracy": float(np.mean([r["stage_avg_accuracy"] for r in items])),
            "old_f1": float(np.mean([r["old_f1"] for r in items])),
            "new_f1": float(np.mean([r["new_f1"] for r in items])),
            "worst_stage_f1": min(r["worst_stage_f1"] for r in items),
            "construction_seconds": float(np.mean([r["construction_seconds"] for r in items])),
            "student_seconds_per_seed": float(np.mean([r["student_seconds_per_seed"] for r in items])),
        })
        out.append(item)
    return out


def fmt_ci(value):
    if value["low"] is None:
        return f"{value['mean']:+.2f}"
    return f"{value['mean']:+.2f} [{value['low']:+.2f}, {value['high']:+.2f}]"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cora-results", type=Path, required=True)
    parser.add_argument("--citeseer-results", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cells = []
    protocols = []
    for dataset, path in (("cora", args.cora_results), ("citeseer", args.citeseer_results)):
        protocol, rows = load(path, dataset, args.audit_dir)
        protocols.append(protocol)
        cells.extend(cell(dataset, order, arm, rows)
                     for order in protocol["orders"] for arm in ARMS)
    by_dataset = pooled(cells, ["dataset", "arm"])
    overall = pooled(cells, ["arm"])
    comparisons = []
    for dataset in ("cora", "citeseer", "all"):
        local = [r for r in cells if dataset == "all" or r["dataset"] == dataset]
        reference = {r["order"] + ("/" + r["dataset"] if dataset == "all" else ""): r
                     for r in local if r["arm"] == "cat_mmd_tag_cctp_top5"}
        for arm in ARMS:
            if arm == "cat_mmd_tag_cctp_top5":
                continue
            target = {r["order"] + ("/" + r["dataset"] if dataset == "all" else ""): r
                      for r in local if r["arm"] == arm}
            keys = sorted(set(reference) & set(target))
            deltas = [target[k]["final_f1"] - reference[k]["final_f1"] for k in keys]
            comparisons.append({"scope": dataset, "arm": arm, "method": LABELS[arm],
                                "delta_vs_top5": ci(deltas),
                                "positive_cells": int(sum(v > 0 for v in deltas)),
                                "cells": len(deltas)})
    payload = {"protocol": {"fixed_x": True, "orders": protocols[0]["orders"],
                            "report_seeds": protocols[0]["report_seeds"],
                            "soft_temperature": 0.07,
                            "note": "all text arms reuse one CaT X per dataset/order/stage"},
               "cells": cells, "pooled_by_dataset": by_dataset,
               "pooled_all": overall, "comparisons_vs_top5": comparisons}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    with (args.output / "cells.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = list(cells[0])
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(cells)
    lines = ["# CCTP top-k 与 soft prototype 消融", "",
             "固定同一份 CaT X；只改变类别中心文本的离散来源数或 Student 使用的文本 embedding。四个顺序、六个 Student seeds，八个 cell 的 reload audit 全部通过。", "",
             "|数据集|方法|Final F1|Accuracy|Stage-average F1|Old F1|New F1|跨顺序标准差|", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in by_dataset:
        lines.append(f"|{r['dataset']}|{r['method']}|{r['final_f1']:.2f}|{r['final_accuracy']:.2f}|{r['stage_avg_f1']:.2f}|{r['old_f1']:.2f}|{r['new_f1']:.2f}|{r['final_f1_order_std']:.2f}|")
    lines += ["", "|方法|全部八个 cell Final F1|Accuracy|Stage-average F1|Old F1|New F1|", "|---|---:|---:|---:|---:|---:|"]
    for r in overall:
        lines.append(f"|{r['method']}|{r['final_f1']:.2f}|{r['final_accuracy']:.2f}|{r['stage_avg_f1']:.2f}|{r['old_f1']:.2f}|{r['new_f1']:.2f}|")
    lines += ["", "|范围|方法|相对 top-5 的 ΔF1|正向 cell|", "|---|---|---:|---:|"]
    for r in comparisons:
        lines.append(f"|{r['scope']}|{r['method']}|{fmt_ci(r['delta_vs_top5'])}|{r['positive_cells']}/{r['cells']}|")
    (args.output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"cells": len(cells), "overall": overall, "comparisons": comparisons}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
