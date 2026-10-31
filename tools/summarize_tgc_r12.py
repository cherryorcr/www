"""Summarize sealed R12 class-preserving coverage-floor evaluation."""

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


def paired_interval(values):
    values = [float(v) for v in values]
    if len(values) < 2:
        return [values[0], values[0]] if values else [None, None]
    return [
        stats.mean(values) - 2.571 * stats.stdev(values) / math.sqrt(len(values)),
        stats.mean(values) + 2.571 * stats.stdev(values) / math.sqrt(len(values)),
    ]


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
        "delta_old": (current["old_f1"] - reference["old_f1"]
                       if current.get("old_f1") is not None
                       and reference.get("old_f1") is not None else None),
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

    rows, cells, audits, builds = [], [], [], []
    cache = {}
    for dataset, order in itertools.product(spec["datasets"], spec["orders"]):
        folder = root / dataset / order
        audit = read(folder / "audit.json")
        assert audit["status"] == "passed"
        assert audit["maximum_metric_error_pp"] < 1e-7
        audits.append({"dataset": dataset, "order": order, **audit})
        result = read(folder / "results.json")["results"]
        cache[dataset, order] = {(int(x["stage"]), x["arm"]): x for x in result}
        for stage in spec["evaluation_stages"]:
            for arm in spec["arms"]:
                current = cache[dataset, order][stage, arm]
                seed_rows = []
                for seed in spec["report_seeds"]:
                    item = read(folder / f"{arm}_s{stage}_seed{seed}.json")
                    base_item = read(folder / f"c0_s{stage}_seed{seed}.json")
                    assert item["selection_seal_sha256"] == seal_hash
                    assert item["steps"] == base_item["steps"]
                    assert item["rows"] == base_item["rows"]
                    value = metric_record(item, base_item, dataset, order, stage, arm, seed)
                    rows.append(value)
                    seed_rows.append(value)
                deltas = [x["delta_f1"] for x in seed_rows]
                old = [x["delta_old"] for x in seed_rows if x["delta_old"] is not None]
                new = [x["delta_new"] for x in seed_rows]
                metrics = current["metrics"]["all"]
                cells.append({
                    "dataset": dataset,
                    "order": order,
                    "stage": stage,
                    "arm": arm,
                    "f1": metrics["cumulative_f1"]["mean"],
                    "f1_sd": metrics["cumulative_f1"]["std"],
                    "old_f1": metrics.get("old_f1", {}).get("mean") if metrics.get("old_f1") else None,
                    "new_f1": metrics["new_f1"]["mean"],
                    "delta_f1": mean(deltas),
                    "delta_old": mean(old),
                    "delta_new": mean(new),
                    "delta_f1_ci_low": paired_interval(deltas)[0],
                    "delta_f1_ci_high": paired_interval(deltas)[1],
                    "seconds": mean([x["seconds"] for x in seed_rows]),
                    "executed_fits": sum(x["executed_here"] for x in seed_rows),
                    "rows": seed_rows[0]["rows"],
                    "steps": seed_rows[0]["steps"],
                })
                build_path = folder / f"{arm}_s{stage}_build_audit.json"
                if build_path.exists():
                    build = read(build_path)
                    builds.append({
                        "dataset": dataset,
                        "order": order,
                        "stage": stage,
                        "arm": arm,
                        "alpha": build.get("alpha"),
                        "floor": build.get("floor"),
                        "random": build.get("random"),
                        "random_seed": build.get("random_seed"),
                        "historical_raw_reads": build.get("historical_raw_reads"),
                        "report_labels_used": build.get("report_labels_used"),
                        "class_allowed_nonzero": build.get("class_allowed_nonzero"),
                        "assignment_sha256": build.get("assignment_sha256"),
                        "memory_sha256": build.get("memory_sha256"),
                    })

    final_stage = max(spec["evaluation_stages"])
    methods = []
    for arm in spec["arms"]:
        final = [x for x in cells if x["stage"] == final_stage and x["arm"] == arm]
        all_cells = [x for x in cells if x["arm"] == arm]
        methods.append({
            "arm": arm,
            "final_f1": mean(x["f1"] for x in final),
            "final_delta_f1": mean(x["delta_f1"] for x in final),
            "final_delta_old": mean(x["delta_old"] for x in final),
            "final_delta_new": mean(x["delta_new"] for x in final),
            "stage_mean_delta_f1": mean(x["delta_f1"] for x in all_cells),
            "worst_final_delta_f1": min(x["delta_f1"] for x in final),
            "worst_final_delta_old": min(x["delta_old"] for x in final if x["delta_old"] is not None),
            "worst_final_delta_new": min(x["delta_new"] for x in final),
            "positive_final_cells": sum(x["delta_f1"] > 0 for x in final),
            "final_cells": len(final),
            "mean_student_seconds": mean(x["seconds"] for x in final),
        })

    contrasts = []
    for candidate in spec["arms"]:
        if candidate == "c0":
            continue
        for dataset, order in itertools.product(spec["datasets"], spec["orders"]):
            for stage in spec["evaluation_stages"]:
                left = {(x["seed"]): x for x in rows if x["dataset"] == dataset and x["order"] == order and x["stage"] == stage and x["arm"] == candidate}
                right = {(x["seed"]): x for x in rows if x["dataset"] == dataset and x["order"] == order and x["stage"] == stage and x["arm"] == "c0"}
                differences = [left[s]["f1"] - right[s]["f1"] for s in spec["report_seeds"]]
                contrasts.append({
                    "dataset": dataset,
                    "order": order,
                    "stage": stage,
                    "candidate": candidate,
                    "baseline": "c0",
                    "mean_f1_delta": mean(differences),
                    "paired_ci_low": paired_interval(differences)[0],
                    "paired_ci_high": paired_interval(differences)[1],
                })

    result = {
        "protocol_sha256": digest(root / "protocol.json"),
        "selection_seal_sha256": seal_hash,
        "fit_slots": len(rows),
        "executed_fits": sum(x["executed_here"] for x in rows),
        "audits": audits,
        "methods": methods,
        "contrasts": contrasts,
        "development_only": bool(spec.get("development_only")),
        "independent_confirmation": bool(spec.get("independent_confirmation")),
        "selection_by_report": bool(spec.get("selection_by_report")),
        "student": spec.get("student"),
        "interpretation": "R12 is a sealed development diagnostic; no arm is promoted by this summary.",
    }
    write_csv(root / "per_seed.csv", rows)
    write_csv(root / "cells.csv", cells)
    write_csv(root / "methods.csv", methods)
    write_csv(root / "contrasts.csv", contrasts)
    write_csv(root / "build_diagnostics.csv", builds)
    (root / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"methods": methods, "fit_slots": len(rows), "executed_fits": result["executed_fits"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    summarize(parser.parse_args().root)
