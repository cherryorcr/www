"""Generate inspectable comparison tables and independently reconcile cells."""
import csv
import itertools
import json
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/degree_confirmation_20260920"
ARMS = ("c0", "t_degree", "xt_degree")
ORDERS = (49011, 49012)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def mean_metric(records, task, arm, metric, period, scope="all", order=None, split=None):
    vals = [r["metrics"][scope][metric] for r in records if r["task"] == task and r["arm"] == arm
            and (period != "final" or r["stage"] == 4) and (order is None or r["order"] == order)
            and (split is None or r.get("split") == split)]
    return mean(v for v in vals if v is not None) * (100 if task == "lp" else 1)


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
                     + ["| " + " | ".join(map(str, row)) + " |" for row in rows])


def paired(records, task, arm, metric, period, order, split, scope="all"):
    lookup = {(r["stage"], r["arm"], r["seed"]): r["metrics"][scope][metric] for r in records
              if r["task"] == task and r["order"] == order and r.get("split") == split}
    stages = (4,) if period == "final" else range(5)
    gaps = []
    for seed in range(210, 216):
        v = [(lookup[s, arm, seed] - lookup[s, "c0", seed]) * (100 if task == "lp" else 1)
             for s in stages if lookup[s, arm, seed] is not None]
        gaps.append(mean(v))
    return mean(gaps), stdev(gaps), sum(v > 0 for v in gaps)


def main():
    report = ["# Independent Cross-Task Confirmation: Generated Tables",
              "All scores are percentages; differences are percentage points. Orders have equal weight.",
              "Stage mean gives each of five stages equal weight. Paired SD/wins are conditional on one memory/split, not independent datasets."]
    flat, overview = [], {}
    for dataset in ("arxiv", "bookchild"):
        root = OUT / dataset
        assert read(root / "pipeline_complete.json")["completed"]
        summary = read(root / "summary.json")
        records = read(root / "results.json")
        assert len(records) == 540
        for row in summary["rows"]:
            actual = mean_metric(records, row["task"], row["arm"], row["metric"], row["period"],
                                 row["scope"], row["order"], row["split"])
            gap, sd, wins = paired(records, row["task"], row["arm"], row["metric"], row["period"],
                                   row["order"], row["split"], row["scope"])
            assert abs(actual - row["value"]["mean"]) < 1e-9
            assert abs(gap - row["delta_pp"]["mean"]) < 1e-9
            assert abs(sd - row["delta_pp"]["sd"]) < 1e-9
            flat.append(dict(dataset=dataset, **{k: row[k] for k in
                             ("task", "order", "split", "scope", "metric", "arm", "period")},
                             score=actual, delta_pp=gap, paired_sd=sd, seed_wins=wins, seed_count=6))
        report.append("\n## " + dataset)
        metrics = [("nc", "cumulative_f1", "final"), ("nc", "cumulative_f1", "stage_mean"),
                   ("nc", "old_f1", "final"), ("nc", "new_f1", "final"),
                   ("lp", "auc", "final"), ("lp", "ap", "final"),
                   ("lp", "auc", "stage_mean"), ("lp", "ap", "stage_mean")]
        values = {arm: [mean_metric(records, *[task, arm, metric, period])
                        for task, metric, period in metrics] for arm in ARMS}
        overview[dataset] = dict(values=values, columns=metrics, decision=summary["decision"])
        report.append(table(["method", "NC final", "NC stage mean", "NC old", "NC new",
                             "LP final AUC", "LP final AP", "LP stage AUC", "LP stage AP"],
                            [[arm] + [f"{v:.3f}" for v in values[arm]] for arm in ARMS]))
        report.append("\n### T+degree minus C0, per order")
        report.append(table(["order", "NC final", "NC stage mean", "NC old", "NC new",
                             "LP final AUC", "LP final AP", "LP stage AUC", "LP stage AP"],
                            [[order] + [f"{mean_metric(records, task, 't_degree', metric, period, order=order) - mean_metric(records, task, 'c0', metric, period, order=order):+.3f}"
                             for task, metric, period in metrics] for order in ORDERS]))
        report.append("\n### Seen-class classification (not the primary NC decision)")
        report.append(table(["method", "final F1", "stage mean F1"], [[arm] +
                            [f"{mean_metric(records, 'nc', arm, 'cumulative_f1', period, scope='seen'):.3f}"
                             for period in ("final", "stage_mean")] for arm in ARMS]))
        controls = []
        for name in ("x", "text", "xt"):
            vals = []
            for period, metric in itertools.product(("final", "stage_mean"), ("auc", "ap")):
                items = [read(root / f"order_{order}/lp_split_{split}/cosine_s{stage}.json")[name]["all"][metric]
                         for order, split, stage in itertools.product(ORDERS, (49101, 49102),
                                                                     (4,) if period == "final" else range(5))]
                vals.append(f"{100 * mean(items):.3f}")
            controls.append([name] + vals)
        report.append("\n### Zero-training cosine controls")
        report.append(table(["input", "final AUC", "final AP", "stage AUC", "stage AP"], controls))
        topology, construction_costs = [], []
        for order in ORDERS:
            stages = read(root / f"order_{order}/construction.json")
            report.append(f"\nOrder {order} source counts: " + ", ".join(str(s["source_count"]) for s in stages))
            report.append(f"\nOrder {order} memory rows: " + ", ".join(str(s["budget"]) for s in stages))
            for arm in ARMS:
                final = stages[-1]["arms"][arm]
                t = final["topology"]
                topology.append([order, arm, t["rows"], t["pairs"], t["isolated"], t["max_degree"],
                                 f"{final['bytes'] / 2**20:.3f}"])
                common = sum(s["shared_block_seconds"] for s in stages)
                edges = sum(s["arms"][arm]["topology_seconds"] for s in stages)
                construction_costs.append([order, arm, f"{common:.3f}", f"{edges:.3f}", f"{common + edges:.3f}"])
        report.append("\n### Final memory")
        report.append(table(["order", "arm", "rows", "pairs", "isolated", "max degree", "file MiB"], topology))
        report.append("\n### Construction cost (seconds)")
        report.append(table(["order", "arm", "shared X/text block", "topology", "total per method"], construction_costs))
        report.append("\nShared block cost is charged once to each standalone method, not zero. Text cache encoding is reused and unmeasured here.")
        report.append("\n### Training cost")
        report.append(table(["task", "arm", "fits", "total seconds", "seconds per fit", "process peak GiB"],
                            [[c["task"], c["arm"], c["fits"], f"{c['train_seconds']:.3f}",
                              f"{c['train_seconds'] / c['fits']:.3f}", f"{c['peak_bytes'] / 2**30:.3f}"]
                             for c in summary["costs"]]))
        report.append("\nProcess peak includes encoder and inference graph buffers; not isolated synthetic-training memory. Training timers exclude final inference/audit.")
        report.append("\n### Prospective checks")
        report.append(table(["task", "order", "metric", "period", "delta pp", "passed"],
                            [[c["task"], c["order"], c["metric"], c["period"], f"{c['delta_pp']:+.4f}", c["passed"]]
                             for c in summary["decision"]["checks"]]))
    with (OUT / "comparison_rows.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    (OUT / "comparison_tables.md").write_text("\n\n".join(report) + "\n", encoding="utf-8")
    (OUT / "overview.json").write_text(json.dumps(overview, indent=2), encoding="utf-8")
    print(json.dumps(dict(reconciled_cells=len(flat), output=str(OUT)), indent=2))


if __name__ == "__main__":
    main()
