"""Summarize the four Unified-Q evidence tracks without touching the paper."""
from __future__ import annotations

import csv
import glob
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path("/home/pc/research/TAG_condensation")
OUT = ROOT / "results/uq_four_priority_20261017"
DATASETS = ("arxiv", "sportsfit", "bookhis")
ARMS = ("uq_dm_edgefree", "cat_cctp", "gecc_hard")
T95_N3 = 4.303


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def stats(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    n = int(arr.size)
    mean = float(arr.mean()) if n else None
    sd = float(arr.std(ddof=1)) if n > 1 else 0.0 if n else None
    half = float(T95_N3 * sd / math.sqrt(n)) if n > 1 else 0.0
    return {"n": n, "mean": mean, "sd": sd, "ci95_low": None if mean is None else mean - half,
            "ci95_high": None if mean is None else mean + half}


def paired_stats(values: list[float]) -> dict:
    out = stats(values)
    out["positive_ratio"] = (float(sum(v > 0 for v in values) / len(values))
                              if values else None)
    out["worst"] = float(min(values)) if values else None
    out["best"] = float(max(values)) if values else None
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fmt(value) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def main() -> None:
    fresh = []
    fresh_by_cell = {}
    for path in sorted((OUT / "fresh").glob("*/*/results.json")):
        payload = read(path)
        if payload.get("status") != "complete":
            continue
        dataset = payload["protocol"]["dataset"]
        seed = int(payload["protocol"]["construction_seed"])
        fresh_by_cell[(dataset, seed)] = payload
        for arm in ARMS:
            rows = [row for row in payload["rows"] if row["arm"] == arm]
            if not rows:
                continue
            row = max(rows, key=lambda item: int(item["stage"]))
            score = row["scores"]["fresh_sealed"]
            fresh.append({
                "dataset": dataset, "seed": seed, "arm": arm,
                "f1": score["final_f1"], "accuracy": score["accuracy"],
                "old_f1": score.get("old_f1"), "new_f1": score.get("new_f1"),
                "construction_seconds": sum(r["construction_seconds"] for r in rows),
                "construction_peak_mb": max(r["construction_peak_mb"] for r in rows),
                "last_stage": row["stage"], "source": str(path),
            })

    fresh_summary = []
    fresh_pairs = []
    for dataset in DATASETS:
        for arm in ARMS:
            subset = [r for r in fresh if r["dataset"] == dataset and r["arm"] == arm]
            for metric in ("f1", "accuracy", "construction_seconds", "construction_peak_mb"):
                item = {"dataset": dataset, "arm": arm, "metric": metric,
                        **stats([float(r[metric]) for r in subset])}
                fresh_summary.append(item)
        for baseline in ("cat_cctp", "gecc_hard"):
            uq = {int(r["seed"]): r for r in fresh
                  if r["dataset"] == dataset and r["arm"] == "uq_dm_edgefree"}
            base = {int(r["seed"]): r for r in fresh
                    if r["dataset"] == dataset and r["arm"] == baseline}
            for metric in ("f1", "accuracy"):
                diffs = [float(uq[s][metric]) - float(base[s][metric])
                         for s in sorted(set(uq) & set(base))]
                fresh_pairs.append({"dataset": dataset, "baseline": baseline,
                                    "metric": metric, **paired_stats(diffs)})

    no_graph = []
    for path in sorted((ROOT / "results/uq_priority_controls_20261014/no_graph").glob("*.json")):
        payload = read(path)
        row = payload["no_graph"]["rows"][-1]["scores"]["sealed_test"]
        dataset = payload["dataset"]
        seed = int(payload["construction_seed"])
        ref_path = (ROOT / "results/uq_final_evidence_20261012/confirmation"
                    / f"{dataset}_o9101_c{seed}/results.json")
        ref = read(ref_path)
        last_stage = len(ref["protocol"]["groups"]) - 1
        primary = [r for r in ref["rows"]
                   if r["arm"] == "uq_dm_edgefree" and int(r["stage"]) == last_stage][0]
        primary_score = primary["scores"]["sealed_test"]
        no_graph.append({
            "dataset": dataset, "seed": seed,
            "no_graph_f1": row["final_f1"], "primary_f1": primary_score["final_f1"],
            "delta_f1": primary_score["final_f1"] - row["final_f1"],
            "no_graph_accuracy": row["accuracy"], "primary_accuracy": primary_score["accuracy"],
            "delta_accuracy": primary_score["accuracy"] - row["accuracy"],
            "source": str(path),
        })

    no_graph_summary = []
    for dataset in DATASETS:
        subset = [r for r in no_graph if r["dataset"] == dataset]
        for metric in ("no_graph_f1", "primary_f1", "delta_f1",
                       "no_graph_accuracy", "primary_accuracy", "delta_accuracy"):
            no_graph_summary.append({"dataset": dataset, "metric": metric,
                                     **stats([float(r[metric]) for r in subset])})
    no_graph_pairs = []
    for dataset in DATASETS:
        subset = [r for r in no_graph if r["dataset"] == dataset]
        for metric in ("delta_f1", "delta_accuracy"):
            no_graph_pairs.append({"dataset": dataset, "metric": metric,
                                   **paired_stats([float(r[metric]) for r in subset])})

    text_rows = []
    for path in sorted((OUT / "text_budget").glob("*/results.json")):
        payload = read(path)
        if payload.get("status") != "complete":
            continue
        for record in payload["records"]:
            for row in record["rows"]:
                score = row["scores"]["sealed_test"]
                text_rows.append({"dataset": row["dataset"],
                                  "seed": int(record["construction_seed"]),
                                  "arm": row["arm"], "f1": score["final_f1"],
                                  "accuracy": score["accuracy"],
                                  "memory_source": row.get("memory_source", "unknown"),
                                  "source": str(path)})
    text_summary = []
    text_pairs = []
    for dataset in DATASETS:
        for arm in ("primary", "nearest1", "nearest5", "random1", "random5"):
            subset = [r for r in text_rows if r["dataset"] == dataset and r["arm"] == arm]
            text_summary.append({"dataset": dataset, "arm": arm, "metric": "f1",
                                 **stats([float(r["f1"]) for r in subset])})
        primary = {int(r["seed"]): r for r in text_rows
                   if r["dataset"] == dataset and r["arm"] == "primary"}
        for control in ("nearest1", "nearest5", "random1", "random5"):
            other = {int(r["seed"]): r for r in text_rows
                     if r["dataset"] == dataset and r["arm"] == control}
            diffs = [float(primary[s]["f1"]) - float(other[s]["f1"])
                     for s in sorted(set(primary) & set(other))]
            text_pairs.append({"dataset": dataset, "control": control,
                               "metric": "f1", **paired_stats(diffs)})

    resource_rows = []
    resource_sources = {}
    for path in sorted((OUT / "resource").glob("*/results.json")):
        payload = read(path)
        resource_sources[payload["dataset"]] = sorted({arm["memory_source"] for arm in payload["arms"]})
        for arm in payload["arms"]:
            student = arm["student"]
            resource_rows.append({
                "dataset": payload["dataset"], "arm": arm["arm"],
                "memory_rows": arm["memory_rows"],
                "memory_payload_mb": arm["cpu_tensor_payload_bytes"] / 2**20,
                "archive_mb": arm["archive_bytes"] / 2**20,
                "memory_load_seconds": arm["memory_load_seconds"],
                "construction_seconds": arm["construction_seconds_sum"],
                "construction_peak_mb": arm["construction_peak_mb_max"],
                "student_train_seconds": student["student_train_seconds_mean"],
                "student_train_peak_mb": student["student_train_peak_mb_max"],
                "inference_seconds": student["full_graph_inference_seconds_mean"],
                "inference_peak_mb": student["full_graph_inference_peak_mb_max"],
                "memory_source": arm["memory_source"],
            })

    write_csv(OUT / "analysis_fresh_cells.csv", fresh)
    write_csv(OUT / "analysis_fresh_summary.csv", fresh_summary)
    write_csv(OUT / "analysis_fresh_pairs.csv", fresh_pairs)
    write_csv(OUT / "analysis_no_graph_cells.csv", no_graph)
    write_csv(OUT / "analysis_no_graph_summary.csv", no_graph_summary)
    write_csv(OUT / "analysis_no_graph_pairs.csv", no_graph_pairs)
    write_csv(OUT / "analysis_text_cells.csv", text_rows)
    write_csv(OUT / "analysis_text_summary.csv", text_summary)
    write_csv(OUT / "analysis_text_pairs.csv", text_pairs)
    write_csv(OUT / "analysis_resource.csv", resource_rows)

    lines = [
        "# Unified-Q four-priority evidence analysis",
        "",
        "This report is generated after all four tracks completed. It does not modify the paper or frozen primary tables.",
        "Fresh sealed uses original_id % 10 == 4, with the original report-dev and selector masks unchanged.",
        "All paired comparisons use the same construction seeds within a dataset.",
        "The 95% intervals use a t critical value of 4.303 for n=3; they are descriptive, not a claim of asymptotic significance.",
        "",
        "## 1. Fresh sealed final-stage results",
        "",
        "| Dataset | Arm | F1 mean +/- sd | Accuracy mean +/- sd | Construction seconds | Peak GiB |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        for arm in ARMS:
            get = lambda metric: next(x for x in fresh_summary
                                      if x["dataset"] == dataset and x["arm"] == arm and x["metric"] == metric)
            f1, acc = get("f1"), get("accuracy")
            sec, mem = get("construction_seconds"), get("construction_peak_mb")
            lines.append(f"| {dataset} | {arm} | {fmt(f1['mean'])} +/- {fmt(f1['sd'])} | "
                         f"{fmt(acc['mean'])} +/- {fmt(acc['sd'])} | {fmt(sec['mean'])} | {fmt(mem['mean']/1024)} |")
    lines += ["", "### Fresh paired deltas (Unified-Q-EF minus baseline)", "",
              "| Dataset | Baseline | Metric | Mean delta | 95% CI | Positive ratio | Worst |",
              "|---|---|---|---:|---:|---:|---:|"]
    for row in fresh_pairs:
        lines.append(f"| {row['dataset']} | {row['baseline']} | {row['metric']} | {fmt(row['mean'])} | "
                     f"[{fmt(row['ci95_low'])}, {fmt(row['ci95_high'])}] | {fmt(row['positive_ratio'])} | {fmt(row['worst'])} |")

    lines += ["", "## 2. No-graph construction control", "",
              "The downstream Student still uses the same full raw graph; only current-delta graph messages are removed during construction.",
              "", "| Dataset | Primary F1 | No-graph F1 | Primary - no-graph | Positive ratio |",
              "|---|---:|---:|---:|---:|"]
    for dataset in DATASETS:
        p = next(x for x in no_graph_summary if x["dataset"] == dataset and x["metric"] == "primary_f1")
        n = next(x for x in no_graph_summary if x["dataset"] == dataset and x["metric"] == "no_graph_f1")
        d = next(x for x in no_graph_pairs if x["dataset"] == dataset and x["metric"] == "delta_f1")
        lines.append(f"| {dataset} | {fmt(p['mean'])} +/- {fmt(p['sd'])} | {fmt(n['mean'])} +/- {fmt(n['sd'])} | "
                     f"{fmt(d['mean'])} | {fmt(d['positive_ratio'])} |")

    lines += ["", "## 3. Fixed text-budget controls", "",
              "All controls keep X, labels, mass, stage, edge-free memory shape, and a 320-character cap fixed; only stored text sources are changed.",
              "", "| Dataset | Primary | Nearest-1 | Nearest-5 | Random-1 | Random-5 |",
              "|---|---:|---:|---:|---:|---:|"]
    for dataset in DATASETS:
        vals = {}
        for arm in ("primary", "nearest1", "nearest5", "random1", "random5"):
            r = next(x for x in text_summary if x["dataset"] == dataset and x["arm"] == arm)
            vals[arm] = f"{fmt(r['mean'])} +/- {fmt(r['sd'])}"
        lines.append(f"| {dataset} | {vals['primary']} | {vals['nearest1']} | {vals['nearest5']} | {vals['random1']} | {vals['random5']} |")
    lines += ["", "| Dataset | Control | Primary - control | 95% CI | Positive ratio | Worst |",
              "|---|---|---:|---:|---:|---:|"]
    for row in text_pairs:
        lines.append(f"| {row['dataset']} | {row['control']} | {fmt(row['mean'])} | "
                     f"[{fmt(row['ci95_low'])}, {fmt(row['ci95_high'])}] | {fmt(row['positive_ratio'])} | {fmt(row['worst'])} |")

    lines += ["", "## 4. Resource decomposition", "",
              "Student and full-graph inference are measured after memory construction. Missing historical archives were rebuilt in RAM under the frozen protocol and labeled as reconstruction.",
              "", "| Dataset | Arm | Construction peak GiB | Construction sec | Student peak MiB | Inference peak MiB | Memory source |",
              "|---|---|---:|---:|---:|---:|---|"]
    for row in resource_rows:
        lines.append(f"| {row['dataset']} | {row['arm']} | {fmt(row['construction_peak_mb']/1024)} | "
                     f"{fmt(row['construction_seconds'])} | {fmt(row['student_train_peak_mb'])} | "
                     f"{fmt(row['inference_peak_mb'])} | {row['memory_source']} |")

    lines += ["", "## 5. Evidence-level interpretation", "",
              "- Fresh sealed performance is the strongest new validity check. Unified-Q-EF is consistently above GECC-hard on ArXiv and BookHis, while SportsFit is approximately tied and should not be described as a universal win.",
              "- The no-graph control is not uniformly worse. It is therefore evidence that graph-aware construction can matter in some cells, but not proof that every Q gain comes from graph messages.",
              "- Fixed-budget text controls show that primary text materialization is not a stable accuracy winner over nearest/random controls. Text routing should be framed as a unified, auditable construction mechanism rather than an independently established source of accuracy gains.",
              "- The main resource cost is construction, especially dense-Q optimization on larger datasets. Student training and full-graph inference remain comparatively small and are not the cause of the construction OOM risk.",
              "- The missing-archive reconstruction is a provenance limitation for resource measurement only; it is not used to overwrite or relabel the original sealed performance tables.",
    ]
    (OUT / "four_priority_analysis_20261017.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary = {
        "fresh_cells": len(fresh), "fresh_complete_cells": len(fresh_by_cell),
        "text_rows": len(text_rows), "resource_rows": len(resource_rows),
        "fresh_summary": fresh_summary, "fresh_pairs": fresh_pairs,
        "no_graph_summary": no_graph_summary, "no_graph_pairs": no_graph_pairs,
        "text_summary": text_summary, "text_pairs": text_pairs,
        "resource": resource_rows, "resource_sources": resource_sources,
    }
    (OUT / "four_priority_analysis_20261017.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")


if __name__ == "__main__":
    main()
