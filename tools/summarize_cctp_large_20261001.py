"""Summarize large-graph CCTP top-3/top-5 runs against sealed controls."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats


ARMS = ["append_edge", "x_plus_p", "cat_mmd_tag_cctp_top3", "cat_mmd_tag_cctp_top5"]
LABELS = {
    "append_edge": "append-edge",
    "x_plus_p": "X+P",
    "cat_mmd_tag_cctp_top3": "CCTP top-3",
    "cat_mmd_tag_cctp_top5": "CCTP top-5",
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


def load(path, dataset):
    payload = read(path)
    protocol = payload["protocol"]
    if protocol["dataset"] != dataset or protocol["stage_protocol"] != "base+4":
        raise ValueError(f"protocol mismatch in {path}")
    if protocol["arms"] != ARMS:
        raise ValueError(f"arms mismatch in {path}: {protocol['arms']}")
    if protocol["student_seeds"] != [100, 101, 102, 103, 104, 105]:
        raise ValueError("student seed mismatch")
    overall = payload["overall"]
    if set(overall) != set(ARMS):
        raise ValueError("overall arm coverage mismatch")
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arxiv", type=Path, required=True)
    parser.add_argument("--bookchild", type=Path, required=True)
    parser.add_argument("--sportsfit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    datasets = [("arxiv", args.arxiv), ("bookchild", args.bookchild), ("sportsfit", args.sportsfit)]
    payloads = [load(path, name) for name, path in datasets]
    rows = []
    for (dataset, _), payload in zip(datasets, payloads):
        for arm in ARMS:
            row = dict(payload["overall"][arm])
            row.update({"dataset": dataset, "arm": arm, "method": LABELS[arm]})
            rows.append(row)
    pooled = []
    for arm in ARMS:
        local = [r for r in rows if r["arm"] == arm]
        pooled.append({
            "arm": arm, "method": LABELS[arm], "cells": len(local),
            "final_f1": float(np.mean([r["final_f1"] for r in local])),
            "final_accuracy": float(np.mean([r["final_accuracy"] for r in local])),
            "stage_avg_f1": float(np.mean([r["stage_avg_f1"] for r in local])),
            "stage_avg_accuracy": float(np.mean([r["stage_avg_accuracy"] for r in local])),
            "old_f1": float(np.mean([r["old_f1"] for r in local])),
            "new_f1": float(np.mean([r["new_f1"] for r in local])),
            "construction_seconds": float(np.mean([r["construction_seconds"] for r in local])),
            "student_seconds": float(np.mean([r["student_seconds"] for r in local])),
        })
    comparisons = []
    for baseline in ("append_edge", "x_plus_p"):
        for arm in ("cat_mmd_tag_cctp_top3", "cat_mmd_tag_cctp_top5"):
            f1 = [a["final_f1"] - b["final_f1"] for a, b in zip(
                [r for r in rows if r["arm"] == arm],
                [r for r in rows if r["arm"] == baseline])]
            acc = [a["final_accuracy"] - b["final_accuracy"] for a, b in zip(
                [r for r in rows if r["arm"] == arm],
                [r for r in rows if r["arm"] == baseline])]
            comparisons.append({
                "method": LABELS[arm], "baseline": LABELS[baseline],
                "final_f1_delta": ci(f1), "accuracy_delta": ci(acc),
                "positive_f1_cells": int(sum(v > 0 for v in f1)), "cells": len(f1),
                "per_dataset_f1_delta": dict(zip([d for d, _ in datasets], f1)),
            })
    result = {"protocol": {"datasets": [d for d, _ in datasets],
                            "stage_protocol": "base+4", "arms": ARMS,
                            "student_seeds": [100, 101, 102, 103, 104, 105],
                            "status": "CCTP top-k mechanism adaptation; append/X+P are sealed controls"},
              "rows": rows, "pooled": pooled, "comparisons": comparisons}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    with (args.output / "rows.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    lines = ["# 大图 CCTP top-k base+4 结果", "",
             "协议：arXiv、BookChild、SportsFit；base+4；5% cumulative memory；历史 raw 不回读；相同 edge-aware Student 与六个 seeds。CaT/CCTP 是机制适配，不是官方端到端结果。", "",
             "|数据集|方法|Final F1|Accuracy|Stage-average F1|Old F1|New F1|构造秒|Student 秒/seed|", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"|{r['dataset']}|{r['method']}|{r['final_f1']:.2f}|{r['final_accuracy']:.2f}|{r['stage_avg_f1']:.2f}|{r['old_f1']:.2f}|{r['new_f1']:.2f}|{r['construction_seconds']:.1f}|{r['student_seconds']:.1f}|")
    lines += ["", "|方法|三图平均 Final F1|Accuracy|Stage-average F1|Old F1|New F1|", "|---|---:|---:|---:|---:|---:|"]
    for r in pooled:
        lines.append(f"|{r['method']}|{r['final_f1']:.2f}|{r['final_accuracy']:.2f}|{r['stage_avg_f1']:.2f}|{r['old_f1']:.2f}|{r['new_f1']:.2f}|")
    lines += ["", "|方法|对照|Δ Final F1|正向图数|", "|---|---|---:|---:|"]
    for r in comparisons:
        d = r["final_f1_delta"]
        text = f"{d['mean']:+.2f} [{d['low']:+.2f}, {d['high']:+.2f}]"
        lines.append(f"|{r['method']}|{r['baseline']}|{text}|{r['positive_f1_cells']}/{r['cells']}|")
    (args.output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"pooled": pooled, "comparisons": comparisons}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
