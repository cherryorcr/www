"""Summarize independently audited C0-M results and apply promotion gates."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean


ARMS = ("c0", "naive_refresh", "c0m")


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metric(entry: dict, scope: str, name: str) -> float | None:
    value = entry["metrics"][scope][name]
    if value is None:
        return None
    return float(value["mean"] if isinstance(value, dict) else value)


def summarize(root: Path) -> dict:
    audit = read(root / "audit_summary.json")
    if audit["status"] != "passed" or audit["fits"] != 324:
        raise ValueError("C0-M audit must pass all 324 fits before summary")
    cells, stages, constructions = [], [], []
    for dataset in ("cora", "citeseer"):
        for order in ("canonical", "order19101", "order19102"):
            cell = root / dataset / order
            result = read(cell / "results.json")
            indexed = {(r["arm"], int(r["stage"])): r for r in result["results"]}
            for stage in range(len(result["manifest"]["stage_classes"])):
                for arm in ARMS:
                    entry = indexed[arm, stage]
                    stages.append({
                        "dataset": dataset, "order": order, "stage": stage,
                        "arm": arm, "rows": int(entry["rows"]),
                        "cumulative_f1": metric(entry, "all", "cumulative_f1"),
                        "old_f1": metric(entry, "all", "old_f1"),
                        "new_f1": metric(entry, "all", "new_f1"),
                        "fit_seconds": sum(float(f["wall_fit_seconds"])
                                            for f in entry["per_seed"]),
                    })
            final_stage = len(result["manifest"]["stage_classes"]) - 1
            final = indexed["c0m", final_stage]
            by_arm = {arm: indexed[arm, final_stage] for arm in ARMS}
            base = by_arm["c0"]
            final_row = {"dataset": dataset, "order": order}
            for arm in ARMS:
                current = by_arm[arm]
                final_row[f"{arm}_final_f1"] = metric(current, "all", "cumulative_f1")
                final_row[f"{arm}_old_f1"] = metric(current, "all", "old_f1")
                final_row[f"{arm}_new_f1"] = metric(current, "all", "new_f1")
                final_row[f"{arm}_stage_avg_f1"] = mean(
                    row["cumulative_f1"] for row in stages
                    if row["dataset"] == dataset and row["order"] == order
                    and row["arm"] == arm)
            final_row["naive_refresh_delta_pp"] = (
                final_row["naive_refresh_final_f1"] - final_row["c0_final_f1"])
            final_row["c0m_delta_pp"] = (
                final_row["c0m_final_f1"] - final_row["c0_final_f1"])
            final_row["c0m_old_delta_pp"] = (
                final_row["c0m_old_f1"] - final_row["c0_old_f1"])
            final_row["c0m_new_delta_pp"] = (
                final_row["c0m_new_f1"] - final_row["c0_new_f1"])
            final_row["c0m_stage_avg_delta_pp"] = (
                final_row["c0m_stage_avg_f1"] - final_row["c0_stage_avg_f1"])
            cells.append(final_row)
            for stage in range(final_stage + 1):
                for arm in ARMS:
                    construction = read(cell / f"{arm}_s{stage}_construction.json")
                    build = construction["build"]
                    maintenance = build.get("maintenance", {})
                    constructions.append({
                        "dataset": dataset, "order": order, "stage": stage,
                        "arm": arm,
                        "build_seconds": float(build.get("build_seconds", 0.0)),
                        "logical_bytes": int(build["payload"]["logical_bytes"]),
                        "rows": int(build["payload"]["rows"]),
                        "updated_rows": int(len(maintenance.get("updated_rows", []))),
                        "max_anchor_relative": float(build.get("max_anchor_relative", 0.0)),
                    })
    c0m_deltas = [row["c0m_delta_pp"] for row in cells]
    old_deltas = [row["c0m_old_delta_pp"] for row in cells]
    new_deltas = [row["c0m_new_delta_pp"] for row in cells]
    positive_cells = sum(value >= -1.0 for value in c0m_deltas)
    c0_build = [row["build_seconds"] for row in constructions if row["arm"] == "c0"]
    c0m_build = [row["build_seconds"] for row in constructions if row["arm"] == "c0m"]
    cost_ratio = sum(c0m_build) / max(sum(c0_build), 1e-12)
    final_memory_ratios = []
    for dataset in ("cora", "citeseer"):
        for order in ("canonical", "order19101", "order19102"):
            c0_final = [row["logical_bytes"] for row in constructions
                        if row["dataset"] == dataset and row["order"] == order
                        and row["arm"] == "c0" and row["stage"] == 2][0]
            c0m_final = [row["logical_bytes"] for row in constructions
                         if row["dataset"] == dataset and row["order"] == order
                         and row["arm"] == "c0m" and row["stage"] == 2][0]
            final_memory_ratios.append(c0m_final / max(c0_final, 1))
    decision = {
        "cell_gate_at_minus_1pp": positive_cells >= 4,
        "mean_final_gain_at_least_1pp": mean(c0m_deltas) >= 1.0,
        "worst_final_at_least_minus_2pp": min(c0m_deltas) >= -2.0,
        "worst_old_at_least_minus_2pp": min(old_deltas) >= -2.0,
        "mean_new_gain_positive": mean(new_deltas) > 0.0,
        "construction_cost_at_most_2x": cost_ratio <= 2.0,
        "all_structural_audits_passed": True,
    }
    summary = {"status": "audited", "cells": len(cells), "fits": audit["fits"],
               "mean_final_delta_pp": mean(c0m_deltas),
               "min_final_delta_pp": min(c0m_deltas),
               "mean_old_delta_pp": mean(old_deltas),
               "mean_new_delta_pp": mean(new_deltas),
               "positive_cell_gate_count": positive_cells,
               "construction_cost_ratio_to_c0": cost_ratio,
               "final_logical_memory_ratio_mean": mean(final_memory_ratios),
               "final_logical_memory_ratio_min": min(final_memory_ratios),
               "final_logical_memory_ratio_max": max(final_memory_ratios),
               "promotion_gate": decision,
               "promoted": all(decision.values()),
               "interpretation": "C0 remains primary unless every gate is true."}
    write_csv(root / "c0m_cell_metrics.csv", cells)
    write_csv(root / "c0m_stage_metrics.csv", stages)
    write_csv(root / "c0m_construction_metrics.csv", constructions)
    (root / "c0m_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# C0-M 小图汇总", "", f"- 晋升：`{summary['promoted']}`",
             f"- final cumulative F1 平均变化：{summary['mean_final_delta_pp']:.2f} pp",
             f"- final cumulative F1 最差变化：{summary['min_final_delta_pp']:.2f} pp",
             f"- old F1 平均变化：{summary['mean_old_delta_pp']:.2f} pp",
             f"- new F1 平均变化：{summary['mean_new_delta_pp']:.2f} pp",
             f"- construction cost / C0：{cost_ratio:.2f}x",
             f"- final logical memory / C0：{summary['final_logical_memory_ratio_mean']:.2f}x (range {summary['final_logical_memory_ratio_min']:.2f}-{summary['final_logical_memory_ratio_max']:.2f})",
             "", "| 数据集 | 顺序 | C0 final | naive refresh | C0-M | C0-M-C0 | old delta | new delta |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in cells:
        lines.append(f"| {row['dataset']} | {row['order']} | {row['c0_final_f1']:.2f} | {row['naive_refresh_final_f1']:.2f} | {row['c0m_final_f1']:.2f} | {row['c0m_delta_pp']:.2f} | {row['c0m_old_delta_pp']:.2f} | {row['c0m_new_delta_pp']:.2f} |")
    (root / "C0M_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    print(json.dumps(summarize(parser.parse_args().root), ensure_ascii=False, indent=2))
