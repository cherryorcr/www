"""Build the strict unified large-graph baseline report.

The input files are the per-dataset ``results.json`` files produced by
``run_public_continual_baselines_large_20261001.py``.  This report keeps the
three method categories explicit:

* sealed internal controls (append-edge and X+P);
* adapted baselines (real-node coreset, CaT-TAG and GECC-TAG);
* the class-shared real-payload candidate.

It does not call any model or use labels.  It only validates the recorded
protocol/audit fields and aggregates the already completed Student scores.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


DATASETS = ("arxiv", "bookchild", "sportsfit")
ARMS = (
    "append_edge",
    "x_plus_p",
    "random_real",
    "k_center",
    "medoid",
    "cat_mmd_tag_cctp_top5",
    "gecc_hard_tag",
    "cat_mmd_tag_class_shared_tagsam",
)
LABELS = {
    "append_edge": "Append-edge (internal control)",
    "x_plus_p": "X+P (internal sealed control)",
    "random_real": "Random real-node coreset (adapted)",
    "k_center": "K-center real-node coreset (adapted)",
    "medoid": "Medoid real-node coreset (adapted)",
    "cat_mmd_tag_cctp_top5": "CaT-TAG CCTP top-5 (adapted)",
    "gecc_hard_tag": "GECC-TAG hard (adapted)",
    "cat_mmd_tag_class_shared_tagsam": "CaT-TAG class-shared real payload (candidate)",
}
CATEGORIES = {
    "append_edge": "internal_control",
    "x_plus_p": "internal_control",
    "random_real": "adapted_coreset",
    "k_center": "adapted_coreset",
    "medoid": "adapted_coreset",
    "cat_mmd_tag_cctp_top5": "adapted_tag",
    "gecc_hard_tag": "adapted_tag",
    "cat_mmd_tag_class_shared_tagsam": "candidate",
}
SEEDS = [100, 101, 102, 103, 104, 105]
TARGET = "cat_mmd_tag_class_shared_tagsam"
METRICS = (
    "final_f1",
    "final_accuracy",
    "stage_avg_f1",
    "old_f1",
    "new_f1",
    "worst_stage_f1",
)


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def mean(values) -> float:
    values = list(values)
    return float(np.mean(np.asarray(values, dtype=float))) if values else float("nan")


def validate_protocol(payload: dict, dataset: str) -> None:
    if payload.get("dataset") != dataset:
        raise ValueError(f"dataset mismatch: expected {dataset}, got {payload.get('dataset')}")
    protocol = payload["protocol"]
    if protocol.get("stage_protocol") != "base+4":
        raise ValueError(f"{dataset}: stage protocol is not base+4")
    if float(protocol.get("memory_rate")) != 0.05:
        raise ValueError(f"{dataset}: memory rate is not 0.05")
    if protocol.get("student_seeds") != SEEDS:
        raise ValueError(f"{dataset}: Student seed mismatch")
    if protocol.get("arms") != list(ARMS):
        raise ValueError(f"{dataset}: method order mismatch: {protocol.get('arms')}")
    if protocol.get("historical_raw_reads") != 0:
        raise ValueError(f"{dataset}: protocol permits historical raw reads")
    if protocol.get("report_test_labels_used_for_construction") is not False:
        raise ValueError(f"{dataset}: protocol permits report/test labels")


def dataset_rows(payload: dict, dataset: str) -> tuple[list[dict], list[dict]]:
    validate_protocol(payload, dataset)
    rows = payload["rows"]
    if len(rows) != 5 * len(ARMS):
        raise ValueError(f"{dataset}: expected {5 * len(ARMS)} stage rows, got {len(rows)}")

    cells = []
    for arm in ARMS:
        trajectory = sorted(
            [r for r in rows if r["arm"] == arm], key=lambda r: int(r["stage"])
        )
        if [int(r["stage"]) for r in trajectory] != list(range(5)):
            raise ValueError(f"{dataset}/{arm}: incomplete stage trajectory")
        final = trajectory[-1]["score"]
        overall = payload["overall"][arm]
        checks = {
            "final_f1": final["final_f1"],
            "final_accuracy": final["accuracy"],
            "stage_avg_f1": final["stage_avg_f1"],
            "stage_avg_accuracy": final["stage_avg_accuracy"],
            "old_f1": final["old_f1"],
            "new_f1": final["new_f1"],
        }
        for key, value in checks.items():
            if not np.isclose(float(overall[key]), float(value), atol=1e-6):
                raise ValueError(f"{dataset}/{arm}: overall/{key} mismatch")
        cells.append({
            "dataset": dataset,
            "arm": arm,
            "method": LABELS[arm],
            "category": CATEGORIES[arm],
            "final_f1": float(overall["final_f1"]),
            "final_f1_std": float(overall["final_f1_std"]),
            "final_accuracy": float(overall["final_accuracy"]),
            "final_accuracy_std": float(overall["final_accuracy_std"]),
            "stage_avg_f1": float(overall["stage_avg_f1"]),
            "stage_avg_accuracy": float(overall["stage_avg_accuracy"]),
            "old_f1": None if overall["old_f1"] is None else float(overall["old_f1"]),
            "new_f1": float(overall["new_f1"]),
            "worst_stage_f1": float(overall["worst_stage_f1"]),
            "construction_seconds": float(overall["construction_seconds"]),
            "student_seconds_per_seed": float(overall["student_seconds"]),
        })

    stage_rows = []
    for row in rows:
        score = row["score"]
        audit = row["construction_audit"]
        stage_rows.append({
            "dataset": dataset,
            "stage": int(row["stage"]),
            "arm": row["arm"],
            "method": LABELS[row["arm"]],
            "memory_rows": int(row["memory_rows"]),
            "final_f1": float(score["final_f1"]),
            "accuracy": float(score["accuracy"]),
            "construction_seconds": float(row["construction_seconds"]),
            "historical_raw_reads": int(audit.get("historical_raw_reads", 0)),
            "report_test_labels_used_for_construction": bool(
                audit.get("report_test_labels_used_for_construction", False)
            ),
            "current_delta_only": audit.get("current_delta_only"),
            "old_memory_updated": audit.get("old_memory_updated"),
            "topology_updated": audit.get("topology_updated"),
            "representative_only_construction": audit.get("representative_only_construction"),
            "class_shared": bool(audit.get("class_shared", False)),
            "text_sources": audit.get("text_sources"),
        })
    return cells, stage_rows


def pooled(cells: list[dict]) -> list[dict]:
    output = []
    for arm in ARMS:
        local = [r for r in cells if r["arm"] == arm]
        row = {"arm": arm, "method": LABELS[arm], "category": CATEGORIES[arm], "datasets": len(local)}
        for key in METRICS + ("construction_seconds", "student_seconds_per_seed"):
            values = [r[key] for r in local if r[key] is not None]
            row[key] = mean(values)
        output.append(row)
    return output


def paired(cells: list[dict]) -> list[dict]:
    index = {(r["dataset"], r["arm"]): r for r in cells}
    output = []
    for baseline in ARMS:
        if baseline == TARGET:
            continue
        for metric in METRICS:
            deltas = [
                index[(dataset, TARGET)][metric] - index[(dataset, baseline)][metric]
                for dataset in DATASETS
            ]
            output.append({
                "baseline_arm": baseline,
                "baseline": LABELS[baseline],
                "metric": metric,
                "mean_delta": mean(deltas),
                "positive_datasets": int(sum(v > 0 for v in deltas)),
                "dataset_count": len(deltas),
                "deltas": {dataset: float(delta) for dataset, delta in zip(DATASETS, deltas)},
            })
    return output


def audit(stage_rows: list[dict]) -> dict:
    target_rows = [r for r in stage_rows if r["arm"] == TARGET]
    return {
        "stage_rows": len(stage_rows),
        "datasets": len(set(r["dataset"] for r in stage_rows)),
        "nonzero_historical_raw_reads": sum(r["historical_raw_reads"] != 0 for r in stage_rows),
        "report_test_label_cells": sum(r["report_test_labels_used_for_construction"] for r in stage_rows),
        "target_cells": len(target_rows),
        "target_current_delta_only_failures": sum(r["current_delta_only"] is not True for r in target_rows),
        "target_old_memory_update_cells": sum(r["old_memory_updated"] is not False for r in target_rows),
        "target_topology_update_cells": sum(r["topology_updated"] is not False for r in target_rows),
        "target_representative_failures": sum(
            r["representative_only_construction"] is not True for r in target_rows
        ),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value) -> str:
    return "N/A" if value is None else f"{float(value):.2f}"


def write_report(out: Path, protocols: list[dict], cells: list[dict], stages: list[dict]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    pooled_rows = pooled(cells)
    pair_rows = paired(cells)
    audit_row = audit(stages)
    write_csv(out / "dataset_cells.csv", cells)
    write_csv(out / "stage_rows.csv", stages)
    write_csv(out / "pooled_methods.csv", pooled_rows)
    pair_csv = []
    for row in pair_rows:
        pair_csv.append({
            "baseline_arm": row["baseline_arm"],
            "baseline": row["baseline"],
            "metric": row["metric"],
            "mean_delta": row["mean_delta"],
            "positive_datasets": row["positive_datasets"],
            "dataset_count": row["dataset_count"],
            **{f"delta_{d}": row["deltas"][d] for d in DATASETS},
        })
    write_csv(out / "class_shared_vs_all.csv", pair_csv)

    report = {
        "protocols": protocols,
        "arms": list(ARMS),
        "labels": LABELS,
        "categories": CATEGORIES,
        "dataset_cells": cells,
        "stage_rows": stages,
        "pooled": pooled_rows,
        "class_shared_vs_all": pair_rows,
        "audit": audit_row,
    }
    (out / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        "# Corrected unified large-graph baseline comparison",
        "",
        "Protocol: base+4, cumulative 5% memory, Student seeds 100--105, same edge-aware 2-layer GCN and bidirectional InfoNCE, frozen MiniLM, current-delta-only construction, no historical raw reads, and no report/test labels in construction.",
        "",
        "The two sealed arms are internal controls. Real-node, CaT-TAG and GECC-TAG rows are mechanism-adapted controls rather than official end-to-end reproductions. The candidate is the CaT X construction with class-shared real text payloads.",
        "",
        "## 1. Dataset-level final results",
        "",
        "|Dataset|Method|Category|Final F1|Accuracy|Stage avg F1|Old F1|New F1|Worst stage|Build s|Student s/seed|",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in cells:
        lines.append(
            f"|{row['dataset']}|{row['method']}|{row['category']}|{fmt(row['final_f1'])}|{fmt(row['final_accuracy'])}|{fmt(row['stage_avg_f1'])}|{fmt(row['old_f1'])}|{fmt(row['new_f1'])}|{fmt(row['worst_stage_f1'])}|{fmt(row['construction_seconds'])}|{fmt(row['student_seconds_per_seed'])}|"
        )

    lines += [
        "",
        "## 2. Equal-weight pooled results",
        "",
        "|Method|Category|Final F1|Accuracy|Stage avg F1|Old F1|New F1|Worst stage|Build s|Student s/seed|",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in pooled_rows:
        lines.append(
            f"|{row['method']}|{row['category']}|{fmt(row['final_f1'])}|{fmt(row['final_accuracy'])}|{fmt(row['stage_avg_f1'])}|{fmt(row['old_f1'])}|{fmt(row['new_f1'])}|{fmt(row['worst_stage_f1'])}|{fmt(row['construction_seconds'])}|{fmt(row['student_seconds_per_seed'])}|"
        )

    lines += [
        "",
        "## 3. Class-shared candidate versus every baseline",
        "",
        "Positive values mean the class-shared candidate is higher. The comparison is paired by dataset and uses the same frozen protocol.",
        "",
        "|Baseline|Metric|Mean delta|Positive datasets|arXiv|BookChild|SportsFit|",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in pair_rows:
        lines.append(
            f"|{row['baseline']}|{row['metric']}|{row['mean_delta']:+.2f}|{row['positive_datasets']}/{row['dataset_count']}|{row['deltas']['arxiv']:+.2f}|{row['deltas']['bookchild']:+.2f}|{row['deltas']['sportsfit']:+.2f}|"
        )

    lines += [
        "",
        "## 4. Audit",
        "",
        f"- Stage rows: {audit_row['stage_rows']} ({audit_row['datasets']} datasets).",
        f"- Non-zero historical raw reads: {audit_row['nonzero_historical_raw_reads']}.",
        f"- Cells using report/test labels for construction: {audit_row['report_test_label_cells']}.",
        f"- Candidate cells: {audit_row['target_cells']}.",
        f"- Candidate current-delta-only failures: {audit_row['target_current_delta_only_failures']}.",
        f"- Candidate old-memory update cells: {audit_row['target_old_memory_update_cells']}.",
        f"- Candidate topology-update cells: {audit_row['target_topology_update_cells']}.",
        f"- Candidate representative-construction failures: {audit_row['target_representative_failures']}.",
        "",
        "The audit counts are computed from every recorded candidate stage, not inferred from the final score.",
    ]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cells, stages, protocols = [], [], []
    for dataset in DATASETS:
        payload = read(args.root / dataset / "results.json")
        local_cells, local_stages = dataset_rows(payload, dataset)
        cells.extend(local_cells)
        stages.extend(local_stages)
        protocols.append(payload["protocol"])
    write_report(args.output, protocols, cells, stages)
    print(json.dumps({"output": str(args.output), "audit": audit(stages)}, indent=2))


if __name__ == "__main__":
    main()
