"""Summarize the CTAG topology-causal ablation artifacts.

The runner writes one JSON file per student fit and a construction manifest
per order.  This script keeps the aggregation deterministic and reports
paired differences against C0, including NC, LP, topology and cost fields.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev
from collections import defaultdict


ARMS = ("c0", "cross_only", "internal_only", "t_degree", "random_ring", "degree_shuffle")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def avg(values):
    values = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not values:
        return None
    return mean(values)


def sd(values):
    values = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return stdev(values) if len(values) > 1 else None


def record_key(record):
    return tuple(record.get(k) for k in
                 ("task", "order", "stage", "split", "arm", "seed"))


def validate(records, manifest):
    """Fail closed on missing, repeated, foreign or unexpected fit cells."""
    expected = {(task, int(order), stage, split, arm, seed)
                for task in ("nc", "lp") for order, groups in manifest["orders"].items()
                for stage in range(len(groups))
                for split in ([None] if task == "nc" else manifest["splits"])
                for arm in manifest["arms"] for seed in manifest["seeds"]}
    keys = [record_key(r) for r in records]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate fit keys")
    if set(keys) != expected:
        raise ValueError(f"fit matrix mismatch: missing={len(expected-set(keys))}, "
                         f"extra={len(set(keys)-expected)}")
    for record in records:
        if record["dataset"] != manifest["dataset"]:
            raise ValueError("foreign dataset")
        required = (("cumulative_f1",) if record["task"] == "nc" else ("auc", "ap"))
        for name in required:
            value = metric(record, "all", name)
            if value is None or not math.isfinite(float(value)):
                raise ValueError("missing/non-finite primary metric")
        for value in flatten_metric(record, record["task"]).values():
            if not math.isfinite(value):
                raise ValueError("non-finite subgroup metric")
    return len(expected)


def metric(record, scope, name):
    return record.get("metrics", {}).get(scope, {}).get(name)


def flatten_metric(record, task):
    if task == "nc":
        scopes = ("all", "seen")
        names = ("cumulative_f1", "old_f1", "new_f1", "accuracy")
    else:
        scopes = ("all", "old", "new", "cross", "within_new")
        names = ("auc", "ap")
    out = {}
    for scope in scopes:
        for name in names:
            value = metric(record, scope, name)
            if value is not None:
                out[f"{scope}_{name}"] = float(value)
    return out


def fit_rows(root: Path):
    rows = []
    for path in sorted(root.rglob("*_seed*.json")):
        if path.name in {"manifest.json", "construction.json", "results.json", "complete.json"}:
            continue
        record = read_json(path)
        context = {k: record.get(k) for k in
                   ("task", "dataset", "order", "stage", "arm", "seed", "split")}
        context.update(flatten_metric(record, record.get("task")))
        cost = record.get("cost", {})
        context["train_seconds"] = cost.get("train_seconds")
        context["peak_allocated_bytes"] = cost.get("peak_allocated_bytes")
        rows.append(context)
    return rows


def read_records(root):
    return [read_json(p) for p in sorted(root.rglob("*_seed*.json"))]


def report_metrics(record):
    factor = 100 if record["task"] == "lp" else 1
    return {k: factor * v for k, v in flatten_metric(record, record["task"]).items()}


def trajectories(records, manifest):
    """Average stages within a matched trajectory before pooling repeats."""
    groups = defaultdict(list)
    for r in records:
        for name, value in report_metrics(r).items():
            groups[(r["task"], r["order"], r.get("split"), r["seed"], r["arm"], name)].append((r["stage"], value))
    rows = []
    for (task, order, split, seed, arm, name), values in groups.items():
        last = len(manifest["orders"][str(order)]) - 1
        for horizon in ("final", "stage_mean"):
            selected = [v for s, v in values if horizon == "stage_mean" or s == last]
            if selected:
                rows.append(dict(task=task, order=order, split=split, seed=seed, arm=arm,
                                 metric=name, horizon=horizon, value=mean(selected),
                                 stages=len(selected)))
    return rows


def estimate(values):
    return dict(mean=avg(values), sd=sd(values), n=len(values))


def summarize_trajectories(rows):
    groups = defaultdict(list)
    for row in rows:
        for order in (row["order"], "all"):
            groups[(row["task"], order, row["horizon"], row["arm"], row["metric"])].append(row["value"])
    return [dict(task=t, order=o, horizon=h, arm=a, metric=m, unit="percent", **estimate(v))
            for (t, o, h, a, m), v in groups.items()]


CONTRASTS = {
    **{a + "-c0": {a: 1, "c0": -1} for a in ARMS if a != "c0"},
    "t_degree-internal_only": {"t_degree": 1, "internal_only": -1},
    "t_degree-random_ring": {"t_degree": 1, "random_ring": -1},
    "t_degree-degree_shuffle": {"t_degree": 1, "degree_shuffle": -1},
    "interaction": {"t_degree": 1, "cross_only": -1, "internal_only": -1, "c0": 1},
}


def paired_contrasts(rows):
    matched, groups = defaultdict(dict), defaultdict(list)
    for row in rows:
        key = tuple(row[k] for k in ("task", "order", "split", "seed", "horizon", "metric"))
        if row["arm"] in matched[key]:
            raise ValueError("duplicate trajectory arm")
        matched[key][row["arm"]] = row["value"]
    for (task, order, split, seed, horizon, metric_name), values in matched.items():
        for name, weights in CONTRASTS.items():
            if not weights.keys() <= values.keys():
                raise ValueError("unpaired contrast: " + name)
            delta = sum(values[a] * w for a, w in weights.items())
            for scope in (order, "all"):
                groups[(task, scope, horizon, metric_name, name)].append(delta)
    return [dict(task=t, order=o, horizon=h, metric=m, contrast=c, unit="pp",
                 **estimate(v), positive=sum(x > 1e-12 for x in v),
                 negative=sum(x < -1e-12 for x in v), ties=sum(abs(x) <= 1e-12 for x in v))
            for (t, o, h, m, c), v in groups.items()]


def cosine_rows(root):
    rows = []
    for path in sorted(root.glob("order_*/lp_split_*/cosine_s*.json")):
        for arm, metrics in read_json(path).items():
            row = dict(task="lp", order=int(path.parents[1].name.split("_")[-1]),
                       split=int(path.parent.name.split("_")[-1]),
                       stage=int(path.stem.split("_s")[-1]), seed=None, arm=arm,
                       metrics=metrics)
            rows.append(row)
    return rows


def construction_rows(root: Path):
    rows = []
    for path in sorted(root.glob("order_*/construction.json")):
        order = int(path.parent.name.split("_")[-1])
        for stage_record in read_json(path):
            stage = stage_record["stage"]
            for arm, info in stage_record["arms"].items():
                row = {"order": order, "stage": stage, "arm": arm}
                row.update(info.get("topology", {}))
                rows.append(row)
    return rows


def group_key(row):
    # LP has a split dimension; NC uses None.
    return (row.get("task"), row.get("order"), row.get("stage"),
            row.get("split"), row.get("arm"))


def summarize_fit(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(group_key(row), []).append(row)
    output = []
    for key, values in sorted(grouped.items(), key=lambda item: str(item[0])):
        task, order, stage, split, arm = key
        names = sorted({name for row in values for name in row
                        if name not in {"task", "dataset", "order", "stage", "arm", "seed", "split"}})
        row = {"task": task, "order": order, "stage": stage,
               "split": split, "arm": arm, "n": len(values)}
        for name in names:
            vals = [value[name] for value in values if value.get(name) is not None]
            row[name] = avg(vals)
            row[name + "_sd"] = sd(vals)
        output.append(row)
    return output


def add_delta(summary):
    controls = {}
    for row in summary:
        if row["arm"] == "c0":
            controls[(row["task"], row["order"], row["stage"], row["split"])] = row
    for row in summary:
        control = controls.get((row["task"], row["order"], row["stage"], row["split"]))
        if not control or row["arm"] == "c0":
            continue
        for name, value in list(row.items()):
            if name.endswith("_sd") or name in {"task", "order", "stage", "split", "arm", "n"}:
                continue
            base = control.get(name)
            if isinstance(value, (int, float)) and isinstance(base, (int, float)):
                row["delta_" + name] = value - base
    return summary


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, dataset: str, summary, topology):
    lines = [f"# 拓扑因果拆分结果：{dataset}", "", "该文件由 `summarize_topology_causal_ablation.py` 自动生成。", ""]
    lines += ["## Student 汇总", "", "| task | order | stage | split | arm | n | key metrics |", "|---|---:|---:|---:|---|---:|---|"]
    for row in summary:
        metrics = []
        for name in ("all_cumulative_f1", "seen_cumulative_f1", "all_auc", "all_ap",
                     "delta_all_cumulative_f1", "delta_all_auc", "delta_all_ap",
                     "train_seconds"):
            if row.get(name) is not None:
                metrics.append(f"{name}={row[name]:.4f}")
        lines.append(f"| {row['task']} | {row['order']} | {row['stage']} | {row['split']} | {row['arm']} | {row['n']} | {'; '.join(metrics)} |")
    lines += ["", "## Topology and propagation", "", "| order | stage | arm | rows | pairs | isolated | max degree | degree Gini | old drift | full old drift | cross norm mass | max relative load |", "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in topology:
        lines.append("| {order} | {stage} | {arm} | {rows} | {pairs} | {isolated} | {max_degree} | {degree_gini:.6f} | {old_block_row_l1:.6f} | {full_old_row_l1:.6f} | {cross_norm_mass:.6f} | {max_relative_load:.6f} |".format(**row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(args):
    root = Path(args.root)
    manifest = read_json(root / "manifest.json")
    if args.dataset != manifest["dataset"]:
        raise ValueError("dataset argument does not match manifest")
    records = read_records(root)
    expected = validate(records, manifest)
    combined = read_json(root / "results.json")
    validate(combined, manifest)
    if {record_key(r): r for r in records} != {record_key(r): r for r in combined}:
        raise ValueError("per-fit JSON does not match results.json")
    fit = fit_rows(root)
    summary = add_delta(summarize_fit(fit))
    topology = construction_rows(root)
    expected_topology = {(int(o), s, a) for o, g in manifest["orders"].items()
                         for s in range(len(g)) for a in manifest["arms"]}
    topology_keys = [(r["order"], r["stage"], r["arm"]) for r in topology]
    if len(topology_keys) != len(set(topology_keys)) or set(topology_keys) != expected_topology:
        raise ValueError("construction matrix mismatch")
    traj = trajectories(records, manifest)
    write_csv(root / "trajectory_metrics_percent.csv", traj)
    write_csv(root / "endpoint_summary_percent.csv", summarize_trajectories(traj))
    write_csv(root / "paired_contrasts_pp.csv", paired_contrasts(traj))
    write_csv(root / "cosine_summary_percent.csv",
              summarize_trajectories(trajectories(cosine_rows(root), manifest)))
    write_csv(root / "fit_summary.csv", summary)
    write_csv(root / "topology_summary.csv", topology)
    write_markdown(root / "TOPOLOGY_CAUSAL_ABLATION_RESULTS.md", args.dataset, summary, topology)
    complete = {
        "dataset": args.dataset,
        "fit_files": len(fit),
        "expected_fit_files": expected,
        "construction_rows": len(topology),
        "complete": True,
        "scope": "exact fit/construction matrices and per-fit/combined JSON agreement; not a checkpoint audit",
        "uncertainty": "SD across repeats; seeds/splits sharing a memory are not independent datasets",
        "units": "original fit_summary uses native LP [0,1]; *_percent and contrasts_pp use percentage units",
    }
    (root / "summary_status.json").write_text(json.dumps(complete, indent=2), encoding="utf-8")
    print(json.dumps(complete, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--dataset", required=True)
    main(parser.parse_args())
