"""Summarize the corrected large-graph CCTP/class-shared pair matrix."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


DATASETS = ("arxiv", "bookchild", "sportsfit")
ARMS = ("cat_mmd_tag_cctp_top5", "cat_mmd_tag_class_shared_tagsam")
LABELS = {
    ARMS[0]: "CaT-TAG CCTP top-5",
    ARMS[1]: "CaT-TAG class-shared real payload",
}
SEEDS = [100, 101, 102, 103, 104, 105]


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def mean(values) -> float:
    return float(np.mean(np.asarray(list(values), dtype=float)))


def std(values) -> float:
    values = np.asarray(list(values), dtype=float)
    return float(values.std(ddof=1)) if len(values) > 1 else 0.0


def cell_rows(payload: dict) -> tuple[list[dict], dict]:
    protocol = payload["protocol"]
    if protocol["stage_protocol"] != "base+4":
        raise ValueError("not base+4")
    if protocol["arms"] != list(ARMS):
        raise ValueError(f"unexpected arms: {protocol['arms']}")
    if protocol["student_seeds"] != SEEDS:
        raise ValueError("student seed mismatch")
    if protocol["memory_rate"] != 0.05:
        raise ValueError("memory rate mismatch")

    rows = payload["rows"]
    if len(rows) != 10:
        raise ValueError(f"expected 10 trajectory rows, got {len(rows)}")
    by_arm = {arm: [] for arm in ARMS}
    for row in rows:
        if row["arm"] not in by_arm or int(row["stage"]) not in range(5):
            raise ValueError("invalid trajectory row")
        by_arm[row["arm"]].append(row)

    cells = []
    for arm in ARMS:
        trajectory = sorted(by_arm[arm], key=lambda r: int(r["stage"]))
        if [int(r["stage"]) for r in trajectory] != list(range(5)):
            raise ValueError(f"incomplete stages for {arm}")
        final = trajectory[-1]["score"]
        overall = payload["overall"][arm]
        expected = {
            "final_f1": final["final_f1"],
            "final_accuracy": final["accuracy"],
            "stage_avg_f1": final["stage_avg_f1"],
            "stage_avg_accuracy": final["stage_avg_accuracy"],
            "old_f1": final["old_f1"],
            "new_f1": final["new_f1"],
        }
        for key, value in expected.items():
            if not np.isclose(float(overall[key]), float(value), atol=1e-6):
                raise ValueError(f"overall mismatch {arm}/{key}")
        cells.append({
            "dataset": payload["dataset"],
            "arm": arm,
            "method": LABELS[arm],
            "final_f1": float(overall["final_f1"]),
            "final_f1_std": float(overall["final_f1_std"]),
            "final_accuracy": float(overall["final_accuracy"]),
            "final_accuracy_std": float(overall["final_accuracy_std"]),
            "stage_avg_f1": float(overall["stage_avg_f1"]),
            "stage_avg_accuracy": float(overall["stage_avg_accuracy"]),
            "old_f1": float(overall["old_f1"]),
            "new_f1": float(overall["new_f1"]),
            "worst_stage_f1": float(overall["worst_stage_f1"]),
            "construction_seconds": float(overall["construction_seconds"]),
            "student_seconds_per_seed": float(overall["student_seconds"]),
        })
    return cells, protocol


def stage_rows(payload: dict) -> list[dict]:
    rows = []
    for row in payload["rows"]:
        score = row["score"]
        audit = row["construction_audit"]
        rows.append({
            "dataset": payload["dataset"],
            "stage": int(row["stage"]),
            "arm": row["arm"],
            "method": LABELS[row["arm"]],
            "memory_rows": int(row["memory_rows"]),
            "final_f1": float(score["final_f1"]),
            "accuracy": float(score["accuracy"]),
            "stage_avg_f1": float(score["stage_avg_f1"]),
            "stage_avg_accuracy": float(score["stage_avg_accuracy"]),
            "old_f1": None if score["old_f1"] is None else float(score["old_f1"]),
            "new_f1": float(score["new_f1"]),
            "construction_seconds": float(row["construction_seconds"]),
            "historical_raw_reads": int(audit["historical_raw_reads"]),
            "report_test_labels_used_for_construction": bool(
                audit["report_test_labels_used_for_construction"]),
            "current_delta_only": bool(audit.get("current_delta_only", False)),
            "old_memory_updated": bool(audit.get("old_memory_updated", False)),
            "topology_updated": bool(audit.get("topology_updated", False)),
            "representative_only_construction": bool(
                audit.get("representative_only_construction", False)),
            "class_shared": bool(audit.get("class_shared", False)),
            "text_sources": int(audit.get("text_sources", 0)),
        })
    return sorted(rows, key=lambda r: (r["dataset"], r["stage"], r["arm"]))


def aggregate(cells: list[dict]) -> list[dict]:
    out = []
    for arm in ARMS:
        local = [r for r in cells if r["arm"] == arm]
        out.append({
            "arm": arm,
            "method": LABELS[arm],
            "cells": len(local),
            "final_f1": mean(r["final_f1"] for r in local),
            "final_accuracy": mean(r["final_accuracy"] for r in local),
            "stage_avg_f1": mean(r["stage_avg_f1"] for r in local),
            "stage_avg_accuracy": mean(r["stage_avg_accuracy"] for r in local),
            "old_f1": mean(r["old_f1"] for r in local),
            "new_f1": mean(r["new_f1"] for r in local),
            "worst_stage_f1": min(r["worst_stage_f1"] for r in local),
            "construction_seconds": mean(r["construction_seconds"] for r in local),
            "student_seconds_per_seed": mean(r["student_seconds_per_seed"] for r in local),
        })
    return out


def paired(cells: list[dict], key: str) -> dict:
    index = {(r["dataset"], r["arm"]): r for r in cells}
    deltas = [
        index[(dataset, ARMS[1])][key] - index[(dataset, ARMS[0])][key]
        for dataset in DATASETS
    ]
    return {
        "metric": key,
        "mean_delta": mean(deltas),
        "std_delta": std(deltas),
        "positive_dataset_count": int(sum(v > 0 for v in deltas)),
        "dataset_count": len(deltas),
        "deltas": {dataset: float(delta) for dataset, delta in zip(DATASETS, deltas)},
    }


def write_report(out: Path, payload: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    with (out / "dataset_cells.csv").open("w", newline="", encoding="utf-8") as f:
        fields = list(payload["dataset_cells"][0])
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(payload["dataset_cells"])
    with (out / "stage_rows.csv").open("w", newline="", encoding="utf-8") as f:
        fields = list(payload["stage_rows"][0])
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(payload["stage_rows"])

    def fmt(v):
        return "N/A" if v is None else f"{v:.2f}"

    lines = [
        "# Corrected 大图 class-shared payload pair matrix", "",
        "协议：arXiv、BookChild、SportsFit；base+4；5% cumulative memory；",
        "历史 raw 不回读；current-delta-only；相同 edge-aware 2-layer GCN Student；",
        "冻结 MiniLM；Student seeds 100--105；评估使用不重复累计节点口径。", "",
        "## 1. 三图最终结果", "",
        "|数据集|方法|Final Macro-F1|Accuracy|Stage-average F1|Old F1|New F1|最差 stage|构造秒|Student 秒/seed|",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["dataset_cells"]:
        lines.append("|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|".format(
            row["dataset"], row["method"], fmt(row["final_f1"]),
            fmt(row["final_accuracy"]), fmt(row["stage_avg_f1"]),
            fmt(row["old_f1"]), fmt(row["new_f1"]),
            fmt(row["worst_stage_f1"]), fmt(row["construction_seconds"]),
            fmt(row["student_seconds_per_seed"])))
    lines += ["", "## 2. 三图等权平均", "",
              "|方法|Final F1|Accuracy|Stage-average F1|Old F1|New F1|最差 stage|构造秒|Student 秒/seed|",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in payload["pooled"]:
        lines.append("|{}|{}|{}|{}|{}|{}|{}|{}|{}|".format(
            row["method"], fmt(row["final_f1"]), fmt(row["final_accuracy"]),
            fmt(row["stage_avg_f1"]), fmt(row["old_f1"]), fmt(row["new_f1"]),
            fmt(row["worst_stage_f1"]), fmt(row["construction_seconds"]),
            fmt(row["student_seconds_per_seed"])))
    lines += ["", "## 3. class-shared 相对 CCTP 的差值", "",
              "正数表示 class-shared real payload 更高。", "",
              "|指标|平均差值|正向数据集|各数据集差值|", "|---|---:|---:|---|"]
    for row in payload["paired"]:
        detail = ", ".join(f"{k} {v:+.2f}" for k, v in row["deltas"].items())
        lines.append("|{}|{:+.2f}|{}/{}|{}|".format(
            row["metric"], row["mean_delta"],
            row["positive_dataset_count"], row["dataset_count"], detail))
    lines += ["", "## 4. 逐阶段 Final F1", "",
              "|数据集|stage|memory rows|CCTP F1|class-shared F1|差值|CCTP Acc|class-shared Acc|",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    stage_index = {(r["dataset"], r["stage"], r["arm"]): r
                   for r in payload["stage_rows"]}
    for dataset in DATASETS:
        for stage in range(5):
            a = stage_index[dataset, stage, ARMS[0]]
            b = stage_index[dataset, stage, ARMS[1]]
            lines.append("|{}|{}|{}|{:.2f}|{:.2f}|{:+.2f}|{:.2f}|{:.2f}|".format(
                dataset, stage, a["memory_rows"], a["final_f1"], b["final_f1"],
                b["final_f1"] - a["final_f1"], a["accuracy"], b["accuracy"]))
    lines += ["", "## 5. 审计结论", "",
              f"- 完整阶段 cell：{payload['audit']['stage_rows']}；",
              f"- historical raw reads 非零：{payload['audit']['nonzero_historical_raw_reads']}；",
              f"- 使用 report/test 标签的 cell：{payload['audit']['report_test_label_cells']}；",
              f"- class-shared current-delta-only 失败 cell：{payload['audit']['class_shared_current_delta_failures']}；",
              f"- class-shared 修改旧 memory 的 cell：{payload['audit']['class_shared_old_memory_update_cells']}；",
              f"- class-shared 修改拓扑的 cell：{payload['audit']['class_shared_topology_update_cells']}；",
              f"- class-shared 代表行构造失败 cell：{payload['audit']['class_shared_representative_failures']}。", "",
              "本报告只确认 class-shared payload 的三图迁移与严格评估结果；不把它自动晋升为主方法，也不宣称静态 TAG SOTA。"]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arxiv", type=Path, required=True)
    ap.add_argument("--bookchild", type=Path, required=True)
    ap.add_argument("--sportsfit", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    paths = {
        "arxiv": args.arxiv,
        "bookchild": args.bookchild,
        "sportsfit": args.sportsfit,
    }
    dataset_cells = []
    stage = []
    protocols = []
    for dataset in DATASETS:
        payload = read(paths[dataset])
        if payload["dataset"] != dataset:
            raise ValueError(f"dataset mismatch in {paths[dataset]}")
        cells, protocol = cell_rows(payload)
        dataset_cells.extend(cells)
        stage.extend(stage_rows(payload))
        protocols.append(protocol)

    class_shared_stage = [r for r in stage if r["arm"] == ARMS[1]]
    audit = {
        "stage_rows": len(stage),
        "nonzero_historical_raw_reads": sum(
            r["historical_raw_reads"] != 0 for r in stage),
        "report_test_label_cells": sum(
            r["report_test_labels_used_for_construction"] for r in stage),
        "class_shared_current_delta_failures": sum(
            not r["current_delta_only"] for r in class_shared_stage),
        "class_shared_old_memory_update_cells": sum(
            r["old_memory_updated"] for r in class_shared_stage),
        "class_shared_topology_update_cells": sum(
            r["topology_updated"] for r in class_shared_stage),
        "class_shared_representative_failures": sum(
            not r["representative_only_construction"] for r in class_shared_stage),
    }
    payload = {
        "protocols": protocols,
        "arms": list(ARMS),
        "dataset_cells": dataset_cells,
        "stage_rows": stage,
        "pooled": aggregate(dataset_cells),
        "paired": [paired(dataset_cells, key) for key in (
            "final_f1", "final_accuracy", "stage_avg_f1", "old_f1", "new_f1")],
        "audit": audit,
    }
    write_report(args.output, payload)
    print(json.dumps({"output": str(args.output), "pooled": payload["pooled"],
                      "paired": payload["paired"], "audit": audit},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
