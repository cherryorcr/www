"""Fail-closed P2-A aggregation and preregistered screening, no model search."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ARMS = ("center_concat", "mean_pool", "text_attention", "text_selection",
        "attribute_selection", "graph_selection", "graph_pool", "shuffled_graph")
CONTROLS = ("center_concat", "text_selection", "attribute_selection", "shuffled_graph")


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def mean(values):
    return float(np.mean(values))


def seed_summary(stages, seed, scope="all"):
    metrics = [next(r for r in s["per_seed"] if r["seed"] == seed)["metrics"][scope]
               for s in stages]
    final = metrics[-1]
    forgetting = mean([max(metrics[s]["per_task_f1"][t] for s in range(t, len(metrics)-1))
                       - final["per_task_f1"][t] for t in range(len(metrics)-1)])
    return dict(final=final["cumulative_f1"], stage_mean=mean([r["cumulative_f1"] for r in metrics]),
                old=final["old_f1"], new=final["new_f1"], forgetting=forgetting)


def screen(dataset_rows, paired):
    failures = []
    for dataset in ("cora", "citeseer"):
        for control in CONTROLS:
            item = next(p for p in paired if p["dataset"] == dataset
                        and p["arm"] == "graph_selection" and p["control"] == control)
            for key, threshold in (("final", .5), ("stage_mean", 0.), ("old", -2.), ("new", -2.)):
                if item[key] < threshold:
                    failures.append(dict(dataset=dataset, control=control, metric=key,
                                         gain=item[key], required_min=threshold))
            for order, gain in item["per_order_final"].items():
                if gain < -2:
                    failures.append(dict(dataset=dataset, control=control, metric="order_final",
                                         order=order, gain=gain, required_min=-2.))
    return dict(pass_screen=not failures, failures=failures,
                independent_confirmation="eligible, not yet executed" if not failures else "not triggered",
                promotion=False)


def summarize(root, out):
    out.mkdir(parents=True, exist_ok=True)
    rows, paired, by_order, mechanism, costs, sources, evidence = [], [], [], [], [], {}, {}
    fits, violations, controls_checked = 0, 0, 0
    for dataset in ("cora", "citeseer"):
        droot = root / dataset
        plan, acceptance = read(droot / "plan.json"), read(droot / "acceptance.json")
        if not acceptance["complete"] or plan["steps_override"] is not None:
            raise AssertionError("incomplete or smoke run")
        if plan["arms"] != list(ARMS) or plan["seeds"] != list(range(100, 106)):
            raise AssertionError("unexpected diagnostic matrix")
        if list(plan["orders"]) != ["canonical", "order1701", "order1702"]:
            raise AssertionError("unexpected diagnostic orders")
        evidence[dataset] = {}
        for order in plan["orders"]:
            oroot = droot / order
            path = oroot / "results.json"
            result = read(path)
            sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            if len(result["results"]) != 3*len(ARMS):
                raise AssertionError("missing arm/stage cell")
            supports = {}
            evidence[dataset][order] = {}
            for arm in ARMS:
                stages = sorted([s for s in result["results"] if s["arm"] == arm], key=lambda s: s["stage"])
                if [s["stage"] for s in stages] != [0, 1, 2]:
                    raise AssertionError("duplicate or missing stage")
                for stage in stages:
                    if sorted(r["seed"] for r in stage["per_seed"]) != plan["seeds"]:
                        raise AssertionError("duplicate or missing seed")
                    audit = read(oroot / f"{arm}_s{stage['stage']}_text.json")
                    supports[(arm, stage["stage"])] = audit["control_sha256"]
                    violations += audit["aggregate"]["budget_violations"]
                    for r in stage["per_seed"]:
                        if r["support_sha256"] != audit["support_sha256"]:
                            raise AssertionError("fit/text audit support mismatch")
                        if r["steps"] != max(50, min(2048, 16*stage["rows"])):
                            raise AssertionError("student steps mismatch")
                        fits += 1
                seed_rows = {seed: seed_summary(stages, seed) for seed in plan["seeds"]}
                evidence[dataset][order][arm] = seed_rows
                line = dict(dataset=dataset, order=order, arm=arm,
                            **{k: mean([s[k] for s in seed_rows.values()]) for k in next(iter(seed_rows.values()))})
                by_order.append(line)
                arm_costs = [c for c in result["costs"] if c["arm"] == arm]
                costs.append(dict(dataset=dataset, order=order, arm=arm,
                    text_seconds=sum(c["text_seconds"] for c in arm_costs),
                    encoding_seconds=sum(c["encoding_seconds"] for c in arm_costs),
                    shared_cluster_legacy_seconds=sum(c["shared"]["cluster_legacy_topology_seconds"] for c in arm_costs),
                    shared_bank_query_seconds=sum(c["shared"]["bank_query_seconds"] for c in arm_costs),
                    shared_bank_encoding_seconds=sum(c["shared"]["bank_encoding_seconds"] for c in arm_costs),
                    source_candidate_tokens=sum(c["shared"]["source_content_tokens"] for c in arm_costs),
                    student_seconds=sum(c["student_seconds"] for c in arm_costs),
                    student_steps=sum(c["student_steps"] for c in arm_costs),
                    final_payload_bytes=arm_costs[-1]["payload"]["logical_payload_bytes"]))
            for stage in range(3):
                if len({supports[(a, stage)] for a in ARMS}) != 1:
                    raise AssertionError("different X/rows/topology across arms")
                controls_checked += 1
                shared = read(oroot / f"shared_s{stage}.json")
                texts = {a: read(oroot / f"{a}_s{stage}_text.json")["rows"] for a in ARMS}
                clusters = shared["clusters_global_ids"]
                if sorted(sum(clusters, [])) != sorted(shared["source_ids"]):
                    raise AssertionError("cluster source partition mismatch")
                neighbor_counts = [q["neighbor_count"] for q in shared["graph_query_stats"]]
                item = dict(dataset=dataset, order=order, stage=stage, clusters=len(clusters),
                            isolated_clusters=sum(n == 0 for n in neighbor_counts),
                            mean_neighbor_count=mean(neighbor_counts),
                            retained_tokens={a: mean([r["retained_tokens"] for r in texts[a]]) for a in ARMS})
                graph = texts["graph_selection"]
                for control in ("attribute_selection", "text_selection", "shuffled_graph"):
                    pairs = list(zip(graph, texts[control]))
                    item[f"same_text_vs_{control}"] = mean([a["input_sha256"] == b["input_sha256"] for a,b in pairs])
                    item[f"snippet_jaccard_vs_{control}"] = mean([
                        len(set(a["snippet_ids"]) & set(b["snippet_ids"])) /
                        max(1, len(set(a["snippet_ids"]) | set(b["snippet_ids"]))) for a,b in pairs])
                mechanism.append(item)
        for arm in ARMS:
            flat = [evidence[dataset][o][arm][seed] for o in plan["orders"] for seed in plan["seeds"]]
            line = dict(dataset=dataset, arm=arm, n_order_seed_cells=len(flat))
            for key in flat[0]:
                line[key] = mean([r[key] for r in flat])
                line[key+"_descriptive_std"] = float(np.std([r[key] for r in flat]))
            rows.append(line)
            for control in CONTROLS:
                differences = [{k: evidence[dataset][o][arm][seed][k] - evidence[dataset][o][control][seed][k]
                                for k in flat[0]} for o in plan["orders"] for seed in plan["seeds"]]
                paired.append(dict(dataset=dataset, arm=arm, control=control,
                    **{k: mean([d[k] for d in differences]) for k in flat[0]},
                    positive_final_cells=sum(d["final"] > 0 for d in differences),
                    per_order_final={o: mean([evidence[dataset][o][arm][seed]["final"] -
                                      evidence[dataset][o][control][seed]["final"] for seed in plan["seeds"]])
                                     for o in plan["orders"]}))
    if fits != 864 or violations:
        raise AssertionError("incomplete fits or budget violations")
    screening = screen(rows, paired)
    summary = dict(rows=rows, per_order=by_order, paired=paired, mechanism=mechanism, costs=costs,
                   screening=screening, acceptance=dict(fits=fits, budget_violations=violations,
                      controlled_stage_cells=controls_checked), source_sha256=sources,
                   uncertainty="descriptive order/seed dispersion, not an independent-sample confidence interval")
    save(out / "summary.json", summary)
    for name, table in (("dataset_metrics", rows), ("per_order_metrics", by_order), ("costs", costs)):
        with (out / f"{name}.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    lines = ["# P2-A Generated Results", "", "All-prompt report-dev Macro-F1; equal-weight means over 3 orders x 6 student seeds.", "",
             "| Dataset | Arm | Final | Stage mean | Old | New | Forgetting |", "|---|---|---:|---:|---:|---:|---:|"]
    lines += [f"| {r['dataset']} | {r['arm']} | {r['final']:.2f} | {r['stage_mean']:.2f} | {r['old']:.2f} | {r['new']:.2f} | {r['forgetting']:.2f} |" for r in rows]
    lines += ["", f"Screen passed: {screening['pass_screen']}. Independent confirmation: {screening['independent_confirmation']}.",
              f"Verified {fits} fits, {controls_checked} shared-control stage cells, {violations} token violations.",
              "No test labels; known orders are diagnostics, not independent confirmation."]
    (out / "metrics.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps(dict(acceptance=summary["acceptance"], screening=screening, rows=rows), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.root, args.output)
