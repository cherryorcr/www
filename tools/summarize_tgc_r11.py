"""Summarize sealed R11 C0-anchor evaluation without selecting by report F1."""

import argparse
import csv
import hashlib
import itertools
import json
import math
import statistics as stats
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean(values):
    values = [float(v) for v in values if v is not None]
    return stats.mean(values) if values else None


def stdev(values):
    values = [float(v) for v in values if v is not None]
    return stats.stdev(values) if len(values) > 1 else 0.0


def paired_interval(values):
    values = [float(v) for v in values]
    if len(values) < 2:
        return [values[0], values[0]] if values else [None, None]
    m = stats.mean(values)
    se = stats.stdev(values) / math.sqrt(len(values))
    return [m - 2.571 * se, m + 2.571 * se]


def write_csv(path, rows):
    if not rows:
        return
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def metric_record(item, base, dataset, order, stage, arm, seed):
    current = item["metrics"]["all"]
    reference = base["metrics"]["all"]
    old_delta = None
    if current.get("old_f1") is not None and reference.get("old_f1") is not None:
        old_delta = current["old_f1"] - reference["old_f1"]
    return {
        "dataset": dataset,
        "order": order,
        "stage": stage,
        "arm": arm,
        "seed": seed,
        "f1": current["cumulative_f1"],
        "old_f1": current.get("old_f1"),
        "new_f1": current.get("new_f1"),
        "delta_f1": current["cumulative_f1"] - reference["cumulative_f1"],
        "delta_old": old_delta,
        "delta_new": current["new_f1"] - reference["new_f1"],
        "seconds": item.get("seconds"),
        "executed_here": bool(item.get("executed_here", False)),
        "steps": item.get("steps"),
        "rows": item.get("rows"),
        "checkpoint_sha256": item.get("checkpoint_sha256"),
    }


def summarize(root):
    spec = read(root / "protocol.json")
    seal = read(root / "selection_seal.json")
    seal_hash = digest(root / "selection_seal.json")
    assert seal["protocol_sha256"] == digest(root / "protocol.json")

    rows = []
    cells = []
    audits = []
    build_rows = []
    results_cache = {}

    for dataset, order in itertools.product(spec["datasets"], spec["orders"]):
        folder = root / dataset / order
        audit = read(folder / "audit.json")
        assert audit["status"] == "passed"
        assert audit["maximum_metric_error_pp"] < 1e-7
        audits.append({"dataset": dataset, "order": order, **audit})
        result = read(folder / "results.json")["results"]
        results_cache[dataset, order] = {(int(item["stage"]), item["arm"]): item for item in result}
        for stage in spec["evaluation_stages"]:
            for arm in spec["arms"]:
                current = results_cache[dataset, order][stage, arm]
                base = results_cache[dataset, order][stage, "c0"]
                seed_values = []
                for seed in spec["report_seeds"]:
                    item = read(folder / f"{arm}_s{stage}_seed{seed}.json")
                    base_item = read(folder / f"c0_s{stage}_seed{seed}.json")
                    assert item["selection_seal_sha256"] == seal_hash
                    assert item["steps"] == base_item["steps"]
                    assert item["rows"] == base_item["rows"]
                    value = metric_record(item, base_item, dataset, order, stage, arm, seed)
                    rows.append(value)
                    seed_values.append(value)
                delta_values = [v["delta_f1"] for v in seed_values]
                old_values = [v["delta_old"] for v in seed_values if v["delta_old"] is not None]
                new_values = [v["delta_new"] for v in seed_values]
                cells.append({
                    "dataset": dataset,
                    "order": order,
                    "stage": stage,
                    "arm": arm,
                    "f1": current["metrics"]["all"]["cumulative_f1"]["mean"],
                    "f1_sd": current["metrics"]["all"]["cumulative_f1"]["std"],
                    "old_f1": current["metrics"]["all"].get("old_f1", {}).get("mean") if current["metrics"]["all"].get("old_f1") else None,
                    "new_f1": current["metrics"]["all"]["new_f1"]["mean"],
                    "delta_f1": mean(delta_values),
                    "delta_old": mean(old_values),
                    "delta_new": mean(new_values),
                    "delta_f1_ci_low": paired_interval(delta_values)[0],
                    "delta_f1_ci_high": paired_interval(delta_values)[1],
                    "seconds": mean([v["seconds"] for v in seed_values]),
                    "executed_fits": sum(v["executed_here"] for v in seed_values),
                    "rows": seed_values[0]["rows"],
                    "steps": seed_values[0]["steps"],
                })

                build_path = folder / f"{arm}_s{stage}_build_audit.json"
                if build_path.exists():
                    build = read(build_path)
                    material = build.get("materialization", {})
                    text = material.get("text", {}) if isinstance(material, dict) else {}
                    build_rows.append({
                        "dataset": dataset,
                        "order": order,
                        "stage": stage,
                        "arm": arm,
                        "alpha": build.get("alpha"),
                        "historical_raw_reads": build.get("historical_raw_reads"),
                        "report_labels_used": build.get("report_labels_used"),
                        "new_x_relative_displacement_mean": build.get("new_x_relative_displacement_mean"),
                        "content_and_topology_share_assignment": build.get("content_and_topology_share_assignment"),
                        "text_char_total": text.get("char_total"),
                        "text_token_total": text.get("token_total"),
                        "memory_sha256": build.get("memory_sha256"),
                    })

    final_stage = max(spec["evaluation_stages"])
    methods = []
    for arm in spec["arms"]:
        final = [v for v in cells if v["stage"] == final_stage and v["arm"] == arm]
        all_cells = [v for v in cells if v["arm"] == arm]
        methods.append({
            "arm": arm,
            "final_f1": mean(v["f1"] for v in final),
            "final_delta_f1": mean(v["delta_f1"] for v in final),
            "final_delta_old": mean(v["delta_old"] for v in final),
            "final_delta_new": mean(v["delta_new"] for v in final),
            "stage_mean_delta_f1": mean(v["delta_f1"] for v in all_cells),
            "worst_final_delta_f1": min(v["delta_f1"] for v in final),
            "worst_final_delta_old": min(v["delta_old"] for v in final if v["delta_old"] is not None),
            "worst_final_delta_new": min(v["delta_new"] for v in final),
            "positive_final_cells": sum(v["delta_f1"] > 0 for v in final),
            "final_cells": len(final),
            "final_descriptive_ci_low": min(v["delta_f1"] for v in final),
            "final_descriptive_ci_high": max(v["delta_f1"] for v in final),
            "mean_student_seconds": mean(v["seconds"] for v in final),
        })

    contrasts = []
    for anchor in ("anchor025", "anchor050", "anchor075"):
        for baseline in ("c0", "joint", "p_initial"):
            for dataset, order in itertools.product(spec["datasets"], spec["orders"]):
                for stage in spec["evaluation_stages"]:
                    left = {(v["seed"]): v for v in rows if v["dataset"] == dataset and v["order"] == order and v["stage"] == stage and v["arm"] == anchor}
                    right = {(v["seed"]): v for v in rows if v["dataset"] == dataset and v["order"] == order and v["stage"] == stage and v["arm"] == baseline}
                    differences = [left[s]["f1"] - right[s]["f1"] for s in spec["report_seeds"]]
                    contrasts.append({
                        "dataset": dataset,
                        "order": order,
                        "stage": stage,
                        "candidate": anchor,
                        "baseline": baseline,
                        "mean_f1_delta": mean(differences),
                        "paired_ci_low": paired_interval(differences)[0],
                        "paired_ci_high": paired_interval(differences)[1],
                    })

    result = {
        "protocol_sha256": digest(root / "protocol.json"),
        "selection_seal_sha256": seal_hash,
        "fit_slots": len(rows),
        "executed_fits": sum(v["executed_here"] for v in rows),
        "audits": audits,
        "methods": methods,
        "contrasts": contrasts,
        "development_only": bool(spec.get("development_only")),
        "independent_confirmation": bool(spec.get("independent_confirmation")),
        "selection_by_report": bool(spec.get("selection_by_report")),
        "student": spec.get("student"),
        "interpretation": "R11 is a sealed development diagnostic; no arm is promoted by this summary.",
    }
    write_csv(root / "per_seed.csv", rows)
    write_csv(root / "cells.csv", cells)
    write_csv(root / "methods.csv", methods)
    write_csv(root / "contrasts.csv", contrasts)
    write_csv(root / "build_diagnostics.csv", build_rows)
    (root / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"methods": methods, "fit_slots": len(rows), "executed_fits": result["executed_fits"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    summarize(parser.parse_args().root)
