"""Reconcile C0 artifacts with archived dispersion; never rewrite raw results."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, pstdev

import torch


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def close(a, b):
    return a is None and b is None or (
        a is not None and b is not None and
        math.isclose(float(a), float(b), rel_tol=0, abs_tol=1e-10))


def pooled_checks(metric, groups, stage):
    """Recompute aggregates from recorded per-class scores, without scorer reuse."""
    active = [[metric["per_class_f1"][str(c)] for c in g]
              for g in groups[:stage + 1]]
    old = [v for group in active[:-1] for v in group]
    expected = {"cumulative_f1": mean([v for g in active for v in g]),
                "old_f1": mean(old) if old else None,
                "new_f1": mean(active[-1])}
    return all(close(metric[k], v) for k, v in expected.items()) and all(
        close(a, mean(b)) for a, b in zip(metric["per_task_f1"], active)
    ) and len(metric["per_task_f1"]) == len(active)


def audit(root):
    checks, sources, comparisons, retention = [], {}, [], []

    def check(label, condition):
        checks.append({"check": label, "passed": bool(condition)})

    def record(path):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        sources[path.relative_to(root).as_posix()] = digest.hexdigest()

    for source in ("code/frozen_v1_2026_09_09/run_gate1_continual.py",
                   "code/v2_llm_fusion/small_comparison_metrics.py"):
        record(root / source)

    for dataset in ("cora", "citeseer"):
        cdir = root / "results/p5_rate_curve_small_20260919" / f"{dataset}_r005"
        ldir = root / "results/small_comparison_audited_r2_2026_09_16" / dataset
        cp, lp = cdir / "results.json", ldir / "results.json"
        c0, legacy = read(cp), read(lp)
        record(cp)
        record(lp)
        manifest = legacy["manifest"]
        check(f"{dataset}: source", c0["source_hash_sha256"] == manifest["data_sha256"])
        for key, other in (("stage_classes", "stage_classes"), ("budgets", "budgets"),
                           ("student_seeds", "seeds"), ("student_batch", "batch")):
            check(f"{dataset}: {key}", c0[key] == manifest[other])
        check(f"{dataset}: C0 response off", not c0["response_marginal"]
              and not c0["response_gain_extra"] and c0["allocation_mode"] == "dispersion")
        check(f"{dataset}: test sealed", c0["test_accessed"] is False
              and manifest["test_labels_evaluated"] is False)
        stages = sorted(c0["results"], key=lambda r: r["stage"])
        check(f"{dataset}: stages", [s["stage"] for s in stages] == [0, 1, 2])
        for row in stages:
            stage = row["stage"]
            prior = next(r for r in legacy["results"]
                         if r["stage"] == stage and r["arm"] == "dispersion")
            paths = (cdir / f"append_fixed_s{stage}_memory.pt",
                     ldir / f"dispersion_s{stage}_memory.pt")
            left, right = [torch.load(p, map_location="cpu", weights_only=False)
                           for p in paths]
            for path in paths:
                record(path)
            for key in ("x", "text_emb", "edge_index", "mass", "depth", "stage"):
                check(f"{dataset}/{stage}: memory {key}",
                      left[key].dtype == right[key].dtype and
                      torch.equal(left[key], right[key]))
            check(f"{dataset}/{stage}: texts", left["texts"] == right["texts"])
            check(f"{dataset}/{stage}: seeds", row["student_seeds"] ==
                  [r["seed"] for r in prior["per_seed"]])
            check(f"{dataset}/{stage}: seed count", len(row["per_student"]) == 6
                  and len(prior["per_seed"]) == 6)
            for seed, current, old in zip(row["student_seeds"], row["per_student"], prior["per_seed"]):
                check(f"{dataset}/{stage}/{seed}: steps", row["student_steps"] == old["steps"])
                check(f"{dataset}/{stage}/{seed}: cumulative", close(
                    current["cumulative_macro_f1"], old["metrics"]["all"]["cumulative_f1"]))
                check(f"{dataset}/{stage}/{seed}: accuracy", close(
                    current["cumulative_accuracy"], old["metrics"]["all"]["accuracy"]))
            check(f"{dataset}/{stage}: C0 mean", close(row["cum_dev_f1_mean"],
                  mean(r["cumulative_macro_f1"] for r in row["per_student"])))
            check(f"{dataset}/{stage}: C0 std", close(row["cum_dev_f1_std"],
                  pstdev(r["cumulative_macro_f1"] for r in row["per_student"])))
        payloads = [legacy]
        for family in ("static", "supervised"):
            path = root / "results/five_level_extension_2026_09_17" / family / dataset / "results.json"
            record(path)
            payloads.append(read(path))
        for payload in payloads:
            for row in payload["results"]:
                for score in row["per_seed"]:
                    for field in ("metrics", "selected_metrics"):
                        if field not in score:
                            continue
                        for scope, metric in score[field].items():
                            check(f"{dataset}/{row['arm']}/{row['stage']}/{score['seed']}/{field}/{scope}: pooled",
                                  pooled_checks(metric, c0["stage_classes"], row["stage"]))
        final = stages[-1]
        for arm in ("dispersion", "v4", "raw_long"):
            row = next(r for r in legacy["results"] if r["arm"] == arm and r["stage"] == 2)
            metrics = [s["selected_metrics" if arm == "raw_long" else "metrics"]["all"]
                       for s in row["per_seed"]]
            comparisons.append({"dataset": dataset, "arm": arm,
                                "pooled_final": mean(m["cumulative_f1"] for m in metrics),
                                "pooled_old": mean(m["old_f1"] for m in metrics),
                                "pooled_new": mean(m["new_f1"] for m in metrics),
                                "c0_final": final["cum_dev_f1_mean"],
                                "c0_task_slice_old": final["old_mean_f1_mean"],
                                "c0_task_slice_new": final["current_f1_mean"]})
        raw = sorted((r for r in legacy["results"] if r["arm"] == "raw_long"),
                     key=lambda r: r["stage"])
        raw_stage = [mean(s["selected_metrics"]["all"]["cumulative_f1"]
                          for s in r["per_seed"]) for r in raw]
        c0_stage = [r["cum_dev_f1_mean"] for r in stages]
        archived_dispersion = [r for r in legacy["results"] if r["arm"] == "dispersion"]
        retention.append({
            "dataset": dataset,
            "c0_final": c0_stage[-1], "raw_selected_final": raw_stage[-1],
            "final_retention_percent": 100 * c0_stage[-1] / raw_stage[-1],
            "c0_stage_average": mean(c0_stage),
            "raw_selected_stage_average": mean(raw_stage),
            "stage_average_retention_percent": 100 * mean(c0_stage) / mean(raw_stage),
            "archived_dispersion_seen_stage_average": mean(
                mean(s["metrics"]["seen"]["cumulative_f1"] for s in r["per_seed"])
                for r in archived_dispersion),
        })
    return {"checks": checks, "error_count": sum(not c["passed"] for c in checks),
            "sources": sources, "comparison": comparisons, "retention": retention,
            "scope": "local artifact equality and aggregate reconciliation, not prediction replay",
            "not_checked": ["original C0 predictions and student states were not saved",
                            "large-graph pooled old/new cannot be recovered from scalar summaries"],
            "training_runs": 0, "test_labels_evaluated": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root.resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    result["auditor_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (args.output / "audit.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (args.output / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result["comparison"][0]))
        writer.writeheader()
        writer.writerows(result["comparison"])
    print(json.dumps({"checks": len(result["checks"]), "errors": result["error_count"],
                      "comparison": result["comparison"]}))
    if result["error_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
