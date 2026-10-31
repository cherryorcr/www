"""Summarize CCTP, heuristic class-shared and formal class-assigned payloads."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np


DATASETS = ("arxiv", "bookchild", "sportsfit")
BASE_ARMS = (
    "cat_mmd_tag_cctp_top5",
    "cat_mmd_tag_class_shared_tagsam",
    "cat_mmd_tag_class_assigned",
)
DUAL_ARM = "cat_mmd_tag_class_assigned_two_anchor"
TWO_PAYLOAD_ARM = "cat_mmd_tag_class_assigned_two_payload"
PROMPT_MARGIN_ARM = "cat_mmd_tag_class_assigned_prompt_margin"
ARMS = BASE_ARMS + (DUAL_ARM,)
LABELS = {
    BASE_ARMS[0]: "CCTP top-5",
    BASE_ARMS[1]: "Heuristic class-shared",
    BASE_ARMS[2]: "Formal class-assigned",
    DUAL_ARM: "Formal two-anchor class-assigned",
    TWO_PAYLOAD_ARM: "Formal high-dispersion two-payload",
    PROMPT_MARGIN_ARM: "Formal prompt-margin assignment",
}
SEEDS = [100, 101, 102, 103, 104, 105]


def mean(values) -> float:
    values = list(values)
    return float(np.mean(np.asarray(values, dtype=float))) if values else float("nan")


def read(path: Path, dataset: str, expected_arms: tuple[str, ...]) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["dataset"] != dataset:
        raise ValueError(f"dataset mismatch in {path}")
    protocol = payload["protocol"]
    if protocol["stage_protocol"] != "base+4" or protocol["memory_rate"] != 0.05:
        raise ValueError(f"protocol mismatch in {path}")
    if protocol["student_seeds"] != SEEDS or protocol["arms"] != list(expected_arms):
        raise ValueError(f"seed/arm mismatch in {path}: {protocol}")
    return payload


def result_path(root: Path, dataset: str) -> Path:
    """Accept both the fetched flat layout and the runner's *_base4 layout."""
    candidates = (root / dataset / "results.json",
                  root / f"{dataset}_base4" / "results.json")
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"no results.json for {dataset} under {root}; checked {candidates}")


def extract(payload: dict, dataset: str,
            expected_arms: tuple[str, ...]) -> tuple[list[dict], list[dict]]:
    cells, stages = [], []
    for arm in expected_arms:
        overall = payload["overall"][arm]
        cells.append({
            "dataset": dataset,
            "arm": arm,
            "method": LABELS[arm],
            "final_f1": float(overall["final_f1"]),
            "final_accuracy": float(overall["final_accuracy"]),
            "stage_avg_f1": float(overall["stage_avg_f1"]),
            "old_f1": float(overall["old_f1"]),
            "new_f1": float(overall["new_f1"]),
            "worst_stage_f1": float(overall["worst_stage_f1"]),
            "construction_seconds": float(overall["construction_seconds"]),
            "student_seconds": float(overall["student_seconds"]),
        })
    for row in payload["rows"]:
        score = row["score"]
        audit = row["construction_audit"]
        stages.append({
            "dataset": dataset,
            "stage": int(row["stage"]),
            "arm": row["arm"],
            "final_f1": float(score["final_f1"]),
            "accuracy": float(score["accuracy"]),
            "historical_raw_reads": int(audit.get("historical_raw_reads", 0)),
            "report_test_labels_used_for_construction": bool(
                audit.get("report_test_labels_used_for_construction", False)
            ),
            "current_delta_only": audit.get("current_delta_only"),
            "old_memory_updated": audit.get("old_memory_updated"),
            "topology_updated": audit.get("topology_updated"),
            "selected_modes": {
                cls: value.get("selected_mode")
                for cls, value in audit.get("assignment_by_class", {}).items()
            },
        })
    return cells, stages


def pooled(cells: list[dict], arms: tuple[str, ...]) -> list[dict]:
    metrics = ("final_f1", "final_accuracy", "stage_avg_f1", "old_f1",
               "new_f1", "worst_stage_f1", "construction_seconds",
               "student_seconds")
    rows = []
    for arm in arms:
        local = [row for row in cells if row["arm"] == arm]
        rows.append({
            "arm": arm,
            "method": LABELS[arm],
            **{metric: mean(row[metric] for row in local) for metric in metrics},
        })
    return rows


def pairwise(cells: list[dict], candidate: str,
             baselines: tuple[str, ...]) -> list[dict]:
    index = {(row["dataset"], row["arm"]): row for row in cells}
    metrics = ("final_f1", "final_accuracy", "stage_avg_f1", "old_f1", "new_f1")
    output = []
    for baseline in baselines:
        for metric in metrics:
            deltas = [
                index[(dataset, candidate)][metric]
                - index[(dataset, baseline)][metric]
                for dataset in DATASETS
            ]
            output.append({
                "candidate": LABELS[candidate],
                "baseline": LABELS[baseline],
                "baseline_arm": baseline,
                "metric": metric,
                "mean_delta": mean(deltas),
                "positive": sum(value > 0 for value in deltas),
                "deltas": {dataset: float(value)
                           for dataset, value in zip(DATASETS, deltas)},
            })
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True,
                        help="root containing CCTP, heuristic and single-anchor results")
    parser.add_argument("--dual-root", type=Path, default=None,
                        help="optional root containing the two-anchor arm results")
    parser.add_argument("--two-payload-root", type=Path, default=None,
                        help="optional root containing the high-dispersion two-payload arm")
    parser.add_argument("--prompt-margin-root", type=Path, default=None,
                        help="optional root containing the prompt-margin arm")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    cells, stages, protocols = [], [], []
    for dataset in DATASETS:
        payload = read(result_path(args.root, dataset), dataset, BASE_ARMS)
        local_cells, local_stages = extract(payload, dataset, BASE_ARMS)
        cells.extend(local_cells)
        stages.extend(local_stages)
        protocols.append(payload["protocol"])
        if args.dual_root is not None:
            dual_payload = read(result_path(args.dual_root, dataset),
                                dataset, (DUAL_ARM,))
            dual_cells, dual_stages = extract(dual_payload, dataset, (DUAL_ARM,))
            cells.extend(dual_cells)
            stages.extend(dual_stages)
            protocols.append(dual_payload["protocol"])
        if args.two_payload_root is not None:
            two_payload = read(result_path(args.two_payload_root, dataset),
                                dataset, (TWO_PAYLOAD_ARM,))
            two_cells, two_stages = extract(two_payload, dataset,
                                            (TWO_PAYLOAD_ARM,))
            cells.extend(two_cells)
            stages.extend(two_stages)
            protocols.append(two_payload["protocol"])
        if args.prompt_margin_root is not None:
            prompt_payload = read(result_path(args.prompt_margin_root, dataset),
                                   dataset, (PROMPT_MARGIN_ARM,))
            prompt_cells, prompt_stages = extract(
                prompt_payload, dataset, (PROMPT_MARGIN_ARM,))
            cells.extend(prompt_cells)
            stages.extend(prompt_stages)
            protocols.append(prompt_payload["protocol"])

    candidate = (PROMPT_MARGIN_ARM if args.prompt_margin_root is not None
                 else TWO_PAYLOAD_ARM if args.two_payload_root is not None
                 else DUAL_ARM if args.dual_root is not None
                 else BASE_ARMS[2])
    active_arms = BASE_ARMS
    if args.dual_root is not None:
        active_arms += (DUAL_ARM,)
    if args.two_payload_root is not None:
        active_arms += (TWO_PAYLOAD_ARM,)
    if args.prompt_margin_root is not None:
        active_arms += (PROMPT_MARGIN_ARM,)
    baselines = tuple(arm for arm in active_arms if arm != candidate)
    pooled_rows = pooled(cells, active_arms)
    pairs = pairwise(cells, candidate, baselines)
    mode_counter = Counter()
    for row in stages:
        mode_counter.update(row["selected_modes"].values())
    target_stages = [row for row in stages if row["arm"] == candidate]
    audit = {
        "stage_rows": len(stages),
        "candidate_arm": candidate,
        "candidate_stage_rows": len(target_stages),
        "nonzero_historical_raw_reads": sum(
            row["historical_raw_reads"] != 0 for row in stages),
        "report_test_label_cells": sum(
            row["report_test_labels_used_for_construction"] for row in stages),
        "candidate_current_delta_only_failures": sum(
            row["current_delta_only"] is not True for row in target_stages),
        "candidate_old_memory_update_cells": sum(
            row["old_memory_updated"] is not False for row in target_stages),
        "candidate_topology_update_cells": sum(
            row["topology_updated"] is not False for row in target_stages),
        "selected_mode_counts": dict(mode_counter),
    }
    report = {
        "protocols": protocols,
        "cells": cells,
        "pooled": pooled_rows,
        "pairwise": pairs,
        "audit": audit,
    }
    (args.output / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(args.output / "cells.csv", cells)
    write_csv(args.output / "pooled.csv", pooled_rows)
    write_csv(args.output / "formal_vs_baselines.csv", pairs)
    write_csv(args.output / "stage_rows.csv", stages)

    lines = [
        "# Formal class-level payload assignment comparison",
        "",
        "Protocol: base+4, cumulative 5% memory, same edge-aware 2-layer GCN Student, frozen MiniLM, Student seeds 100--105, current-delta-only construction.",
        "",
        "|Dataset|Method|Final F1|Accuracy|Stage avg F1|Old F1|New F1|Build s|Student s/seed|",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in cells:
        lines.append(
            f"|{row['dataset']}|{row['method']}|{row['final_f1']:.2f}|{row['final_accuracy']:.2f}|{row['stage_avg_f1']:.2f}|{row['old_f1']:.2f}|{row['new_f1']:.2f}|{row['construction_seconds']:.1f}|{row['student_seconds']:.1f}|"
        )
    lines += [
        "",
        "## Equal-weight pooled results",
        "",
        "|Method|Final F1|Accuracy|Stage avg F1|Old F1|New F1|Worst stage|Build s|",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in pooled_rows:
        lines.append(
            f"|{row['method']}|{row['final_f1']:.2f}|{row['final_accuracy']:.2f}|{row['stage_avg_f1']:.2f}|{row['old_f1']:.2f}|{row['new_f1']:.2f}|{row['worst_stage_f1']:.2f}|{row['construction_seconds']:.1f}|"
        )
    lines += [
        "",
        "## Formal candidate minus baseline",
        "",
        "|Candidate|Baseline|Metric|Mean delta|Positive datasets|arXiv|BookChild|SportsFit|",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in pairs:
        lines.append(
            f"|{row['candidate']}|{row['baseline']}|{row['metric']}|{row['mean_delta']:+.2f}|{row['positive']}/3|{row['deltas']['arxiv']:+.2f}|{row['deltas']['bookchild']:+.2f}|{row['deltas']['sportsfit']:+.2f}|"
        )
    lines += [
        "",
        "## Selection audit",
        "",
        f"- Candidate stage rows: {audit['candidate_stage_rows']}",
        f"- Non-zero historical raw reads: {audit['nonzero_historical_raw_reads']}",
        f"- Report/test labels used: {audit['report_test_label_cells']}",
        f"- Candidate current-delta-only failures: {audit['candidate_current_delta_only_failures']}",
        f"- Candidate old-memory updates: {audit['candidate_old_memory_update_cells']}",
        f"- Candidate topology updates: {audit['candidate_topology_update_cells']}",
        f"- Selected modes across all recorded class assignments: `{dict(mode_counter)}`",
    ]
    (args.output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "pooled": pooled_rows,
                      "audit": audit}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
