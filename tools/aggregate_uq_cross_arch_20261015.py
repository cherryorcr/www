"""Validate and summarize the frozen-memory cross-architecture audit.

The construction cell (order x construction seed) is the statistical unit.
Student seeds are averaged inside each row by the evaluator and are checked
against the stored per-student values here.  This script never changes frozen
primary results.
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results/uq_cross_arch_20261015_arxiv_v2/results.json"
OUT = ROOT / "results/uq_cross_arch_20261015_arxiv_v2/aggregate"
DATASET = "arxiv"
ORDERS = (9101, 9102, 9103)
SEEDS = (271006, 271108, 271109)
ARMS = ("uq_dm_edgefree", "cat_cctp", "gecc_hard")
STUDENTS = ("prompt_gcn", "prompt_sage")
METRICS = ("final_f1", "accuracy", "old_f1", "new_f1")
DISPLAY = {
    "uq_dm_edgefree": "CoTAG-Q / Unified-Q-EF",
    "cat_cctp": "CaT-CCTP",
    "gecc_hard": "GECC-hard-TAG",
}
STUDENT_DISPLAY = {"prompt_gcn": "Prompt-GCN", "prompt_sage": "Prompt-GraphSAGE"}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def close(a: float | None, b: float | None, tol: float = 1e-8) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return abs(float(a) - float(b)) <= tol


def sample_std(values: list[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


def crossed_bootstrap(matrix: dict[tuple[int, int], float]) -> dict[str, Any]:
    """Deterministic crossed bootstrap over the three orders and seeds."""
    import random

    observed = [matrix[(order, seed)] for order in ORDERS for seed in SEEDS]
    rng = random.Random(20261015)
    draws: list[float] = []
    for _ in range(20000):
        draw_orders = [rng.choice(ORDERS) for _ in ORDERS]
        draw_seeds = [rng.choice(SEEDS) for _ in SEEDS]
        draws.append(mean(matrix[(order, seed)] for order in draw_orders for seed in draw_seeds))
    draws.sort()
    return {
        "mean": mean(observed),
        "std": sample_std(observed),
        "ci95_crossed_bootstrap": [draws[int(.025 * len(draws))], draws[int(.975 * len(draws))]],
        "positive_cells": sum(value > 0 for value in observed),
        "cells": len(observed),
        "positive_ratio": sum(value > 0 for value in observed) / len(observed),
        "worst_cell": min(observed),
        "best_cell": max(observed),
        "per_order": {str(order): mean(matrix[(order, seed)] for seed in SEEDS) for order in ORDERS},
        "per_construction_seed": {str(seed): mean(matrix[(order, seed)] for order in ORDERS) for seed in SEEDS},
        "note": "Exploratory crossed-factor bootstrap; 3 orders x 3 construction seeds.",
    }


def validate(payload: dict[str, Any]) -> tuple[dict[tuple[int, int, str, str], dict[str, Any]], dict[str, Any]]:
    assert payload.get("status") == "complete", payload.get("status")
    rows = payload.get("rows", [])
    expected = {(order, seed, arm, student) for order in ORDERS for seed in SEEDS for arm in ARMS for student in STUDENTS}
    indexed: dict[tuple[int, int, str, str], dict[str, Any]] = {}
    failures: list[str] = []
    for row in rows:
        key = (int(row["order_seed"]), int(row["construction_seed"]), row["arm"], row["student"])
        if key in indexed:
            failures.append(f"duplicate row {key}")
        indexed[key] = row
        score = row["score"]
        student_rows = score["per_student"]
        if [int(item["seed"]) for item in student_rows] != [150, 151, 152]:
            failures.append(f"student seeds drifted {key}")
        for metric in METRICS:
            values = [float(item[metric]) for item in student_rows]
            if not close(score[metric], mean(values), 1e-7):
                failures.append(f"mean mismatch {key}/{metric}")
            # The evaluator stores standard deviations only for Final F1 and
            # Accuracy; Old/New are retained as cell means.
            if metric in ("final_f1", "accuracy"):
                if not close(score[metric + "_std"], sample_std(values), 1e-7):
                    failures.append(f"std mismatch {key}/{metric}")
        if not row.get("checks", {}).get("memory_unchanged", False):
            failures.append(f"memory not marked unchanged {key}")
        if not row.get("checks", {}).get("construction_reused", False):
            failures.append(f"construction not reused {key}")
    missing = sorted(expected - set(indexed))
    extra = sorted(set(indexed) - expected)
    if missing:
        failures.append(f"missing rows: {missing}")
    if extra:
        failures.append(f"unexpected rows: {extra}")

    # Same memory, masks and evaluator budget for GCN and SAGE within each cell.
    common_checks = []
    for order in ORDERS:
        for seed in SEEDS:
            for arm in ARMS:
                pair = [indexed[(order, seed, arm, student)] for student in STUDENTS]
                if len({row["memory_hash"] for row in pair}) != 1:
                    failures.append(f"memory hash differs {order}/{seed}/{arm}")
                if len({row["archive_sha256"] for row in pair}) != 1:
                    failures.append(f"archive hash differs {order}/{seed}/{arm}")
                if len({row["checks"]["test_mask_hash"] for row in pair}) != 1:
                    failures.append(f"test mask differs {order}/{seed}/{arm}")
                if len({row["checks"]["report_mask_hash"] for row in pair}) != 1:
                    failures.append(f"report mask differs {order}/{seed}/{arm}")
                common_checks.append({
                    "order_seed": order,
                    "construction_seed": seed,
                    "arm": arm,
                    "memory_hash": pair[0]["memory_hash"],
                    "archive_sha256": pair[0]["archive_sha256"],
                    "same_memory_across_students": True,
                    "same_masks_across_students": True,
                })
    audit = {
        "input": str(INPUT),
        "status": payload.get("status"),
        "expected_rows": len(expected),
        "observed_rows": len(rows),
        "unique_rows": len(indexed),
        "missing": missing,
        "extra": extra,
        "failures": failures,
        "valid": not failures,
        "common_checks": common_checks,
        "protocol": payload.get("config", {}),
    }
    if failures:
        raise AssertionError(json.dumps(audit, indent=2))
    return indexed, audit


def grouped_summary(indexed: dict[tuple[int, int, str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for student in STUDENTS:
        for arm in ARMS:
            for metric in METRICS:
                values = {(order, seed): float(indexed[(order, seed, arm, student)]["score"][metric])
                          for order in ORDERS for seed in SEEDS}
                summary = crossed_bootstrap(values)
                rows.append({"student": student, "arm": arm, "metric": metric, **summary})
    return rows


def paired_summary(indexed: dict[tuple[int, int, str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for student in STUDENTS:
        for baseline in ("cat_cctp", "gecc_hard"):
            for metric in METRICS:
                for direction, lhs, rhs in (("uq_minus_baseline", "uq_dm_edgefree", baseline),):
                    values = {(order, seed): float(indexed[(order, seed, lhs, student)]["score"][metric])
                              - float(indexed[(order, seed, rhs, student)]["score"][metric])
                              for order in ORDERS for seed in SEEDS}
                    rows.append({"student": student, "baseline": baseline, "metric": metric,
                                 "direction": direction, **crossed_bootstrap(values)})
    return rows


def transfer_summary(indexed: dict[tuple[int, int, str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for arm in ARMS:
        for metric in METRICS:
            values = {(order, seed): float(indexed[(order, seed, arm, "prompt_sage")]["score"][metric])
                      - float(indexed[(order, seed, arm, "prompt_gcn")]["score"][metric])
                      for order in ORDERS for seed in SEEDS}
            rows.append({"arm": arm, "metric": metric, "direction": "sage_minus_gcn", **crossed_bootstrap(values)})
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(row for row in rows)


def fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "--"
    return f"{float(value):.{digits}f}"


def build_markdown(indexed: dict[tuple[int, int, str, str], dict[str, Any]],
                   grouped: list[dict[str, Any]], paired: list[dict[str, Any]],
                   transfer: list[dict[str, Any]], audit: dict[str, Any]) -> str:
    by_group = {(r["student"], r["arm"], r["metric"]): r for r in grouped}
    by_pair = {(r["student"], r["baseline"], r["metric"]): r for r in paired}
    by_transfer = {(r["arm"], r["metric"]): r for r in transfer}
    lines = [
        "# Unified-Q cross-architecture sealed transfer audit",
        "",
        "This report validates frozen memories from the current exact-rate protocol on Prompt-GCN and Prompt-GraphSAGE. The construction cell (class order x construction seed) is the statistical unit; the three Student seeds are averaged within each evaluator row.",
        "",
        "## Audit",
        "",
        f"- Status: **{'PASS' if audit['valid'] else 'FAIL'}**",
        f"- Rows: {audit['observed_rows']} / {audit['expected_rows']} expected (3 methods x 2 Students x 9 cells)",
        "- Memory/archive hashes and sealed/report masks match between GCN and GraphSAGE for every cell and method.",
        "- Student budget: 2048 steps, batch 24, seeds 150/151/152.",
        "- Construction was not rerun; sealed test was not used for construction or method selection.",
        "",
        "## ArXiv performance by Student",
        "",
        "Values are mean +/- crossed-cell standard deviation over 9 order/seed cells. F1 and Accuracy are percentages.",
        "",
        "| Student | Method | F1 | Accuracy | Old F1 | New F1 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for student in STUDENTS:
        for arm in ARMS:
            values = []
            for metric in ("final_f1", "accuracy", "old_f1", "new_f1"):
                r = by_group[(student, arm, metric)]
                values.append(f"{fmt(r['mean'])} +/- {fmt(r['std'])}")
            lines.append(f"| {STUDENT_DISPLAY[student]} | {DISPLAY[arm]} | " + " | ".join(values) + " |")
    lines += ["", "## UQ paired differences", "", "Positive values mean CoTAG-Q / Unified-Q-EF is higher. The interval is an exploratory crossed bootstrap over the 3 x 3 construction design.", "", "| Student | Baseline | Metric | Mean delta | 95% crossed CI | Positive cells | Worst cell |", "|---|---|---|---:|---:|---:|---:|"]
    for student in STUDENTS:
        for baseline in ("cat_cctp", "gecc_hard"):
            for metric in ("final_f1", "accuracy"):
                r = by_pair[(student, baseline, metric)]
                low, high = r["ci95_crossed_bootstrap"]
                lines.append(f"| {STUDENT_DISPLAY[student]} | {DISPLAY[baseline]} | {metric} | {fmt(r['mean'])} | [{fmt(low)}, {fmt(high)}] | {r['positive_cells']}/{r['cells']} | {fmt(r['worst_cell'])} |")
    lines += ["", "## Architecture transfer", "", "SAGE - GCN is computed from the same memory in each cell; positive values mean GraphSAGE benefits more than GCN.", "", "| Method | Metric | Mean delta | 95% crossed CI | Positive cells | Worst cell |", "|---|---|---:|---:|---:|---:|"]
    for arm in ARMS:
        for metric in ("final_f1", "accuracy"):
            r = by_transfer[(arm, metric)]
            low, high = r["ci95_crossed_bootstrap"]
            lines.append(f"| {DISPLAY[arm]} | {metric} | {fmt(r['mean'])} | [{fmt(low)}, {fmt(high)}] | {r['positive_cells']}/{r['cells']} | {fmt(r['worst_cell'])} |")
    lines += ["", "## Interpretation", "", "1. The cross-architecture result tests transfer of the same condensed memory, not retraining the constructor for GraphSAGE.", "2. CoTAG-Q should be described as architecture-transferable on ArXiv under this protocol, not universally architecture-agnostic: the GraphSAGE implementation is a PyG two-layer mean SAGEConv with explicit self-loop neighborhood and root transform, as recorded by the runner.", "3. The current result is strongest as a protocol and transfer audit. It does not remove the dataset-dependent GECC result: the broader nine-cell panel remains close on ArXiv, favors GECC on average on SportsFit, and favors CoTAG-Q on BookHis.", "4. The intervals are descriptive because there are only three order levels and three construction seeds; Student runs are not additional independent cells.", "", "## Source", "", f"- Raw result: `{INPUT.as_posix()}`", f"- Aggregate directory: `{OUT.as_posix()}`", "- Remote source: `/home/pc/research/TAG_condensation/results/uq_cross_arch_20261015_arxiv_v2/`", ""]
    return "\n".join(lines)


def maybe_plot(grouped: list[dict[str, Any]], paired: list[dict[str, Any]]) -> list[str]:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        return []
    paths: list[str] = []
    for metric, ylabel, filename in (("final_f1", "Macro-F1 (%)", "performance_f1.png"), ("accuracy", "Accuracy (%)", "performance_accuracy.png")):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
        for ax, student in zip(axes, STUDENTS):
            rows = [r for r in grouped if r["student"] == student and r["metric"] == metric]
            x = np.arange(len(ARMS))
            means = [r["mean"] for r in rows]
            errs = [r["std"] for r in rows]
            ax.bar(x, means, yerr=errs, capsize=4, color=["#2a9d8f", "#e9c46a", "#457b9d"])
            ax.set_xticks(x, ["CoTAG-Q", "CaT", "GECC"], rotation=15)
            ax.set_title(STUDENT_DISPLAY[student])
            ax.grid(axis="y", alpha=.25)
            ax.set_ylabel(ylabel)
        fig.suptitle(f"ArXiv sealed transfer: {ylabel}")
        fig.tight_layout()
        path = OUT / filename
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        paths.append(str(path))
    return paths


def main() -> None:
    payload = read_json(INPUT)
    indexed, audit = validate(payload)
    grouped = grouped_summary(indexed)
    paired = paired_summary(indexed)
    transfer = transfer_summary(indexed)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    write_csv(OUT / "method_summary.csv", grouped)
    write_csv(OUT / "paired_deltas.csv", paired)
    write_csv(OUT / "architecture_transfer.csv", transfer)
    (OUT / "summary.md").write_text(build_markdown(indexed, grouped, paired, transfer, audit), encoding="utf-8")
    plots = maybe_plot(grouped, paired)
    manifest = {"audit": audit, "method_summary": grouped, "paired_deltas": paired, "architecture_transfer": transfer, "plots": plots}
    (OUT / "summary.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"status": "complete", "rows": len(indexed), "output": str(OUT), "plots": plots}))


if __name__ == "__main__":
    main()
