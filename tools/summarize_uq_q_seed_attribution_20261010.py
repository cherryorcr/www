"""Summarize the registered shared-Q attribution matrix."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t


ARMS = ("uq_shared_q", "uq_split_q", "uq_no_residual")
NAMES = {
    "uq_shared_q": "Shared Q / full",
    "uq_split_q": "Independent Qx + Qt",
    "uq_no_residual": "Shared Q / no residual",
}
METRICS = ("final_f1", "accuracy", "old_f1", "new_f1")
DATASETS = ("wikics", "sportsfit", "bookhis")
ORDERS = (9101, 9102, 9103)
SEEDS = (271006, 271108, 271109)


def registered_tags():
    return {f"{d}_o{o}_c{s}": (d, o, s)
            for d in DATASETS for o in ORDERS for s in SEEDS}


def validate_cell(item, identity):
    p = item["protocol"]
    if (p["dataset"], p["order_seed"], p["construction_seed"]) != identity:
        raise ValueError("directory/protocol identity mismatch")
    required = {"name": "uq-q-seed-attribution-20261010",
                "student_seeds": [150, 151, 152], "student_steps": 2048,
                "student_batch": 24, "q_steps": 30,
                "sinkhorn_max_steps": 2048, "budget_mode": "fixed_schedule",
                "topology": "edge-free", "sealed_primary_unchanged": True,
                "evaluation_split": "report-dev; not sealed test",
                "no_parameter_search": True, "arms": list(ARMS)}
    for key, value in required.items():
        if p.get(key) != value:
            raise ValueError(f"unregistered protocol {key}: {p.get(key)}")
    if len(p["stage_id_hashes"]) != 5 or len(p["budgets"]) != 5:
        raise ValueError("expected five stages")
    expected = {(a, stage) for a in ARMS for stage in range(5)}
    keys = [(r["arm"], r["stage"]) for r in item["rows"]]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("missing or duplicate arm-stage records")
    for r in item["rows"]:
        if r["memory_rows"] != p["budgets"][r["stage"]]:
            raise ValueError("memory budget mismatch")
        for key in ("budget_ok", "prefix_ok", "finite_ok", "sinkhorn_feasible"):
            if r["checks"].get(key) is not True:
                raise ValueError(f"failed or missing check: {key}")
        a = r["audit"]
        if (a.get("historical_raw_reads") != 0 or a.get("old_update_count") != 0
                or a.get("uses_q_topology") is not False):
            raise ValueError("access/prefix/topology audit mismatch")
        error = a.get("text_hard_forward_max_error")
        if error is None or not np.isfinite(error) or error > 1e-4:
            raise ValueError("text materialization mismatch")
        if r["arm"] == "uq_no_residual" and a.get("x_residual_norm") != 0:
            raise ValueError("no-residual arm has nonzero residual")
        score = r["score"]
        for key in ("student_seeds", "student_steps", "student_batch"):
            if score.get(key) != p[key]:
                raise ValueError(f"row Student protocol mismatch: {key}")
        students = score["per_student"]
        if sorted(v["seed"] for v in students) != p["student_seeds"]:
            raise ValueError("missing/duplicate Student seeds")
        for metric in METRICS:
            if metric == "old_f1" and r["stage"] == 0:
                if score[metric] is not None:
                    raise ValueError("base old-class metric should be absent")
                continue
            values = [v[metric] for v in students]
            if not all(v is not None and np.isfinite(v) for v in values):
                raise ValueError(f"nonfinite metric: {metric}")
            if not np.isclose(score[metric], np.mean(values), rtol=0, atol=1e-8):
                raise ValueError(f"incorrect Student aggregation: {metric}")


def validate_matrix(items):
    if not items:
        return
    protocols = [c["protocol"] for c in items]
    for key in ("source_hashes", "runner_sha256", "student"):
        if any(p[key] != protocols[0][key] for p in protocols):
            raise ValueError(f"mixed implementation/Student: {key}")
    configs = [{k: v for k, v in p["q_config"].items() if k != "seed"}
               for p in protocols]
    if any(c != configs[0] for c in configs):
        raise ValueError("mixed constructor configuration")
    for p in protocols:
        if p["q_config"]["seed"] != p["construction_seed"]:
            raise ValueError("constructor seed mismatch")
    for dataset in DATASETS:
        group = [p for p in protocols if p["dataset"] == dataset]
        if not group:
            continue
        for key in ("budgets", "pool_mask_hash", "report_mask_hash", "reference_protocol_hash"):
            if any(p[key] != group[0][key] for p in group):
                raise ValueError(f"inconsistent dataset protocol: {dataset}/{key}")
        for order in ORDERS:
            same = [p for p in group if p["order_seed"] == order]
            for key in ("groups", "stage_id_hashes"):
                if any(p[key] != same[0][key] for p in same):
                    raise ValueError(f"inconsistent stage allocation: {dataset}/{order}/{key}")


def cells(root: Path):
    out = []
    for tag, identity in registered_tags().items():
        path = root / tag / "results.json"
        if not path.exists():
            continue
        item = json.loads(path.read_text(encoding="utf-8"))
        if item.get("status") != "complete":
            continue
        try:
            validate_cell(item, identity)
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"invalid completed cell {path}: {exc}") from exc
        out.append(item)
    validate_matrix(out)
    return out


def score(row, metric):
    value = row["score"][metric]
    return None if value is None else float(value)


def mean_score(rows, metric):
    values = [score(r, metric) for r in rows if score(r, metric) is not None]
    return float(np.mean(values)) if values else None


def paired_order_interval(current, candidate, metric):
    # Student seeds are averaged within a cell, then construction seeds within
    # an order. Only complete order blocks enter the paired t interval.
    means = {}
    for order in ORDERS:
        block = [c for c in current if c["protocol"]["order_seed"] == order]
        if {c["protocol"]["construction_seed"] for c in block} != set(SEEDS):
            continue
        values = []
        for c in block:
            rows = {r["arm"]: r for r in c["rows"] if r["stage"] == 4}
            values.append(score(rows[ARMS[0]], metric) - score(rows[candidate], metric))
        means[str(order)] = float(np.mean(values))
    values = list(means.values())
    n = len(values)
    center = float(np.mean(values)) if n else None
    ci = None
    if n >= 2:
        half = float(student_t.ppf(.975, n - 1) * np.std(values, ddof=1) / np.sqrt(n))
        ci = [center - half, center + half]
    return {"n_orders": n, "mean": center, "ci95": ci, "per_order": means,
            "scope": "paired t over complete order means; conditional on registered construction seeds; only 3 orders"}


def summarize(root: Path):
    all_cells = cells(root)
    if not all_cells:
        raise ValueError("no complete cells")
    actual = {f"{c['protocol']['dataset']}_o{c['protocol']['order_seed']}_c{c['protocol']['construction_seed']}"
              for c in all_cells}
    missing = sorted(set(registered_tags()) - actual)
    by_dataset = {}
    for dataset in sorted({c["protocol"]["dataset"] for c in all_cells}):
        current = [c for c in all_cells if c["protocol"]["dataset"] == dataset]
        by_dataset[dataset] = {}
        for arm in ARMS:
            final = [r for c in current for r in c["rows"]
                     if r["arm"] == arm and r["stage"] == len(c["protocol"]["stage_id_hashes"]) - 1]
            stage = [r for c in current for r in c["rows"] if r["arm"] == arm]
            by_dataset[dataset][arm] = {
                "cells": len(final),
                "final": {m: {"mean": float(np.mean([score(r, m) for r in final])),
                               "std": float(np.std([score(r, m) for r in final], ddof=1))
                               if len(final) > 1 else None} for m in METRICS},
                "stage_average": {m: mean_score(stage, m) for m in METRICS},
                "worst_stage_f1": float(min(score(r, "final_f1") for r in stage)),
                "worst_final_f1": float(min(score(r, "final_f1") for r in final)),
                "memory_rows": sorted({r["memory_rows"] for r in final}),
                "build_seconds": float(np.mean([r["construction_seconds"] for r in stage])),
                "student_seconds": float(np.mean([r["student_seconds"] for r in final])),
            }

    paired = {}
    for dataset in by_dataset:
        current = [c for c in all_cells if c["protocol"]["dataset"] == dataset]
        for candidate in ARMS[1:]:
            per_metric = {}
            diffs = {m: [] for m in METRICS}
            for c in current:
                last = len(c["protocol"]["stage_id_hashes"]) - 1
                rows = {r["arm"]: r for r in c["rows"]
                        if r["stage"] == last and r["arm"] in ARMS}
                for m in METRICS:
                    diffs[m].append(score(rows["uq_shared_q"], m) - score(rows[candidate], m))
            for m, values in diffs.items():
                per_metric[m] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                    "positive_cells": int(sum(v > 0 for v in values)),
                    "cells": len(values),
                    "worst_cell": float(min(values)),
                    "best_cell": float(max(values)),
                    "order_interval": paired_order_interval(current, candidate, m),
                    "per_cell": [{"order": c["protocol"]["order_seed"],
                                  "seed": c["protocol"]["construction_seed"], "delta": v}
                                 for c, v in zip(current, values)],
                }
            paired[f"{dataset}/shared_vs_{candidate}"] = per_metric

    summary = {"completed_cells": len(all_cells), "expected_cells": 27,
               "missing_cells": missing, "datasets": by_dataset,
               "source_hashes": all_cells[0]["protocol"]["source_hashes"],
               "audit_rows": sum(len(c["rows"]) for c in all_cells),
               "audit_note": "saved access counters checked; not independent filesystem tracing",
               "paired": paired, "registered_arms": list(ARMS),
               "warning": "report-dev attribution; not sealed test; no method selected by score"}
    (root / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    lines = ["# Shared-Q 多 construction seed 归因", "",
             f"完成 cells：{len(all_cells)}。所有 cell 使用同一 fixed-schedule、Student seeds `[150,151,152]` 和 2048 steps；结果为 report-dev，不是 sealed test。",
             "", "## Final Macro-F1 / Accuracy / Old F1 / New F1", ""]
    for dataset, arms in by_dataset.items():
        lines += [f"### {dataset}", "", "| 方法 | cells | Final F1 | Accuracy | Old F1 | New F1 | Stage F1 | 构造秒 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for arm in ARMS:
            row = arms[arm]
            vals = []
            for m in METRICS:
                v = row["final"][m]
                vals.append(f"{v['mean']:.2f} ± {v['std']:.2f}" if v["std"] is not None else f"{v['mean']:.2f}")
            lines.append(f"| {NAMES[arm]} | {row['cells']} | {vals[0]} | {vals[1]} | {vals[2]} | {vals[3]} | {row['stage_average']['final_f1']:.2f} | {row['build_seconds']:.1f} |")
        lines.append("")
    lines += ["## Shared-Q 相对对照", "", "| 数据集 | 对照 | Δ Final F1 | 正收益 cells | 最差 cell | Δ Accuracy | Δ Old F1 | Δ New F1 |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for key, metrics in paired.items():
        dataset, name = key.split("/shared_vs_")
        f, a, o, n = (metrics[m] for m in METRICS)
        lines.append(f"| {dataset} | {NAMES[name]} | {f['mean']:+.2f} | {f['positive_cells']}/{f['cells']} | {f['worst_cell']:+.2f} | {a['mean']:+.2f} | {o['mean']:+.2f} | {n['mean']:+.2f} |")
    lines += ["", "解释：Shared-Q 只在 Q 参数共享意义上与 Independent Qx+Qt 对照；学习 Q 相对 fixed-Q 的收益不由本表单独证明。No-residual 是对当前 full 的单因素消融。", ""]
    (root / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results/uq_q_seed_attribution_20261010"))
    args = parser.parse_args()
    summary = summarize(args.root)
    print(json.dumps({"completed_cells": summary["completed_cells"], "datasets": list(summary["datasets"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
