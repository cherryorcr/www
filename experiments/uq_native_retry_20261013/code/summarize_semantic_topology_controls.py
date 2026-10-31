"""Summarize four semantic topology controls against the frozen C0/T arms."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from collections import defaultdict
from statistics import mean, stdev

import summarize_topology_causal_ablation as S

NEW_ARMS = ("knn_nearest", "knn_capacity", "cycles_nearest", "cycles_capacity")
BASE_ARMS = ("c0", "internal_only", "t_degree")
ALL_ARMS = BASE_ARMS + NEW_ARMS


def rows(root):
    out = []
    for path in sorted(root.rglob("*_seed*.json")):
        r = json.loads(path.read_text())
        if r.get("arm") in NEW_ARMS:
            out.append(r)
    return out


def record_key(r):
    return tuple(r.get(k) for k in ("task", "order", "stage", "split", "arm", "seed"))


def validate(records, manifest):
    expected = {(task, int(order), stage, split, arm, seed)
                for task in ("nc", "lp") for order, groups in manifest["orders"].items()
                for stage in range(len(groups))
                for split in ([None] if task == "nc" else manifest["splits"])
                for arm in NEW_ARMS for seed in manifest["seeds"]}
    keys = [record_key(r) for r in records]
    if set(keys) != expected or len(keys) != len(set(keys)):
        raise ValueError(f"fit matrix mismatch: expected={len(expected)} actual={len(keys)}")
    return len(expected)


def metric(r, name):
    scope, metric_name = name.rsplit("_", 1)
    return r["metrics"][scope][metric_name]


def fit_trajectory(records, manifest):
    groups = defaultdict(list)
    for r in records:
        factor = 100 if r["task"] == "lp" else 1
        for scope in (("all", "seen") if r["task"] == "nc" else
                      ("all", "old", "new", "cross", "within_new")):
            for name in (("cumulative_f1", "old_f1", "new_f1", "accuracy") if r["task"] == "nc" else ("auc", "ap")):
                if name in r["metrics"].get(scope, {}) and r["metrics"][scope][name] is not None:
                    groups[(r["task"], r["order"], r.get("split"), r["seed"], r["arm"],
                           scope + "_" + name)].append((r["stage"], factor * r["metrics"][scope][name]))
    out = []
    for (task, order, split, seed, arm, name), values in groups.items():
        final = max(values)[0]
        out.append(dict(task=task, order=order, split=split, seed=seed, arm=arm,
                        metric=name, horizon="final", value=next(v for s, v in values if s == final)))
        out.append(dict(task=task, order=order, split=split, seed=seed, arm=arm,
                        metric=name, horizon="stage_mean", value=mean(v for _, v in values)))
    return out


def write(path, data):
    fields = sorted({k for row in data for k in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(data)


def main(args):
    root = Path(args.root)
    manifest = json.loads((root / "manifest.json").read_text())
    records = rows(root)
    expected = validate(records, manifest)
    traj = fit_trajectory(records, manifest)
    summary = []
    groups = defaultdict(list)
    for r in traj:
        groups[(r["task"], r["order"], r["horizon"], r["arm"], r["metric"])].append(r["value"])
    for (task, order, horizon, arm, metric_name), vals in sorted(groups.items(), key=str):
        summary.append(dict(task=task, order=order, horizon=horizon, arm=arm,
                            metric=metric_name, mean=mean(vals),
                            sd=stdev(vals) if len(vals) > 1 else None, n=len(vals)))
    write(root / "trajectory_metrics_percent.csv", traj)
    write(root / "endpoint_summary_percent.csv", summary)
    (root / "summary_status.json").write_text(json.dumps({
        "fit_files": len(records), "expected_fit_files": expected, "complete": True,
        "scope": "new arms only; compare with the prior formal six-arm report by shared keys",
    }, indent=2), encoding="utf-8")
    print(json.dumps({"fit_files": len(records), "expected": expected, "complete": True}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--root", required=True); main(p.parse_args())
