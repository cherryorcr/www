"""Complete paired summaries and prospective decision, without selecting a winner."""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np

ARMS = ("c0", "t_degree", "xt_degree")
ORDERS = (49011, 49012)
SEEDS = tuple(range(210, 216))
SPLITS = (49101, 49102)


def key(r):
    return (r["task"], r["order"], r.get("split"), r["stage"], r["arm"], r["seed"])


def validate(records):
    expected = set()
    for order, stage, arm, seed in itertools.product(ORDERS, range(5), ARMS, SEEDS):
        expected.add(("nc", order, None, stage, arm, seed))
        for split in SPLITS:
            expected.add(("lp", order, split, stage, arm, seed))
    actual = [key(r) for r in records]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError(f"incomplete or duplicate matrix: {len(actual)} vs {len(expected)}")
    memories = {}
    for r in records:
        k = (r["order"], r["stage"], r["arm"])
        memories.setdefault(k, r["memory_sha256"])
        if memories[k] != r["memory_sha256"]:
            raise ValueError("different memory used across tasks or seeds")


def stat(values):
    a = np.asarray(values, dtype=float)
    return {"mean": float(a.mean()), "sd": float(a.std(ddof=1)) if len(a) > 1 else 0.,
            "n": len(a), "positive_fraction": float((a > 0).mean()),
            "min": float(a.min()), "max": float(a.max())}


def summarize(records):
    validate(records)
    lookup = {key(r): r for r in records}
    rows = []
    for task in ("nc", "lp"):
        scopes = ("all", "seen") if task == "nc" else ("all", "old", "new", "cross", "within_new")
        metrics = ("cumulative_f1", "old_f1", "new_f1") if task == "nc" else ("auc", "ap")
        splits = (None,) if task == "nc" else SPLITS
        for order, split, scope, metric, arm in itertools.product(ORDERS, splits, scopes, metrics, ARMS):
            for period in ("final", "stage_mean"):
                values, gaps = [], []
                stages = (4,) if period == "final" else range(5)
                for seed in SEEDS:
                    v, d = [], []
                    for stage in stages:
                        a = lookup[(task, order, split, stage, arm, seed)]["metrics"][scope][metric]
                        b = lookup[(task, order, split, stage, "c0", seed)]["metrics"][scope][metric]
                        if a is not None and b is not None:
                            factor = 100 if task == "lp" else 1
                            v.append(a * factor)
                            d.append((a - b) * factor)
                    if v:
                        values.append(np.mean(v))
                        gaps.append(np.mean(d))
                if values:
                    rows.append(dict(task=task, order=order, split=split, scope=scope, metric=metric,
                                     arm=arm, period=period, value=stat(values), delta_pp=stat(gaps)))
    return rows


def decision(dataset, rows):
    def gap(task, order, metric, period):
        vals = [r["delta_pp"]["mean"] for r in rows if r["task"] == task and r["order"] == order
                and r["scope"] == "all" and r["metric"] == metric and r["period"] == period
                and r["arm"] == "t_degree"]
        assert len(vals) == (2 if task == "lp" else 1)
        return float(np.mean(vals))
    checks = []
    for order, metric, period in itertools.product(ORDERS, ("auc", "ap"), ("final", "stage_mean")):
        delta = gap("lp", order, metric, period)
        passed = delta > 0 if dataset == "bookchild" else delta >= -.5
        checks.append(dict(task="lp", order=order, metric=metric, period=period, delta_pp=delta, passed=passed))
    for period in ("final", "stage_mean"):
        values = [gap("nc", order, "cumulative_f1", period) for order in ORDERS]
        checks.append(dict(task="nc", order="dataset_mean", metric="cumulative_f1", period=period,
                           delta_pp=float(np.mean(values)), passed=bool(np.mean(values) >= -1)))
        for order, delta in zip(ORDERS, values):
            checks.append(dict(task="nc", order=order, metric="cumulative_f1", period=period,
                               delta_pp=delta, passed=delta >= -2))
    for order in ORDERS:
        delta = gap("nc", order, "old_f1", "final")
        checks.append(dict(task="nc", order=order, metric="old_f1", period="final", delta_pp=delta, passed=delta >= -2))
    return {"candidate": "t_degree", "checks": checks,
            "lp_passed": all(c["passed"] for c in checks if c["task"] == "lp"),
            "nc_passed": all(c["passed"] for c in checks if c["task"] == "nc")}


def main(root):
    records = json.loads((root / "results.json").read_text())
    rows = summarize(records)
    dataset = json.loads((root / "manifest.json").read_text())["dataset"]
    audit = json.loads((root / "audit.json").read_text()) if (root / "audit.json").exists() else None
    verdict = decision(dataset, rows)
    verdict["audit_passed"] = bool(audit and audit["passed"] and audit["strict_metric_reload_passed"])
    costs = []
    for task, arm in itertools.product(("nc", "lp"), ARMS):
        selected = [r for r in records if r["task"] == task and r["arm"] == arm]
        costs.append(dict(task=task, arm=arm, fits=len(selected),
                          train_seconds=sum(r["cost"]["train_seconds"] for r in selected),
                          peak_bytes=max(r["cost"]["peak_allocated_bytes"] for r in selected)))
    result = dict(dataset=dataset, fits=len(records), decision=verdict, rows=rows, costs=costs,
                  note="Paired seed statistics are conditional on shared memories; not independent dataset evidence.")
    (root / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(dict(dataset=dataset, decision=verdict)), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    main(p.parse_args().root)
