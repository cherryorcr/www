"""Strict P2-B aggregation: method seeds averaged, never cherry-picked."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics as S
from pathlib import Path

ARMS = ("center_concat", "fixed_anchor", "semantic_700", "semantic_701", "student_700", "student_701")
FAMILIES = dict(center_concat=["center_concat"], fixed_anchor=["fixed_anchor"],
                semantic=["semantic_700", "semantic_701"], student=["student_700", "student_701"])
KEYS = ("final", "stage_mean", "old", "new", "forgetting")


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def seed_summary(stages, seed, scope="all"):
    metrics = [next(r for r in s["per_seed"] if r["seed"] == seed)["metrics"][scope] for s in stages]
    final = metrics[-1]
    forgetting = S.mean(max(metrics[s]["per_task_f1"][t] for s in range(t, len(metrics)-1))
                        - final["per_task_f1"][t] for t in range(len(metrics)-1))
    return dict(final=final["cumulative_f1"], stage_mean=S.mean(r["cumulative_f1"] for r in metrics),
                old=final["old_f1"], new=final["new_f1"], forgetting=forgetting)


def check_matrix(plan, acceptance):
    if not acceptance["complete"] or plan["classification"] != "known-order screen":
        raise AssertionError("not a complete formal result")
    if plan["arms"] != list(ARMS) or plan["seeds"] != list(range(100, 106)):
        raise AssertionError("wrong arm/seed matrix")
    if list(plan["orders"]) != ["canonical", "order1701", "order1702"]:
        raise AssertionError("wrong orders")
    if (plan["outer_steps"], plan["inner_steps"], plan["validation_steps"], plan["stages"]) != (40, 16, 32, 3):
        raise AssertionError("training budget changed")
    if plan["report_steps_override"] is not None:
        raise AssertionError("smoke report training")


def screen(paired, method_seed_gains):
    expected = {(d, c) for d in ("cora", "citeseer")
                for c in ("center_concat", "fixed_anchor", "semantic")}
    if len(paired) != 6 or {(r["dataset"], r["control"]) for r in paired} != expected:
        raise AssertionError("missing/duplicate screening control")
    expected_seeds = {(d, s) for d in ("cora", "citeseer") for s in (700, 701)}
    if len(method_seed_gains) != 4 or {(r["dataset"], r["method_seed"]) for r in method_seed_gains} != expected_seeds:
        raise AssertionError("missing/duplicate construction seed")
    failures = []
    for row in paired:
        for key, threshold in (("final", .5), ("stage_mean", 0.), ("old", -2.), ("new", -2.)):
            if row[key] < threshold:
                failures.append(dict(dataset=row["dataset"], control=row["control"],
                                     metric=key, gain=row[key], minimum=threshold))
        for order, value in row["per_order_final"].items():
            if value < -2:
                failures.append(dict(dataset=row["dataset"], control=row["control"],
                                     metric="order_final", order=order, gain=value, minimum=-2))
    for row in method_seed_gains:
        if row["gain"] <= 0:
            failures.append(dict(metric="method_seed_final_vs_concat", **row))
    return dict(pass_screen=not failures, failures=failures, promotion=False,
                independent_confirmation="eligible; not yet run" if not failures else "not triggered")


def summarize(root, out):
    out.mkdir(parents=True, exist_ok=True)
    rows, order_rows, paired, costs, mechanism, gains, hashes = [], [], [], [], [], [], {}
    total_fits, total_memories = 0, 0
    for dataset in ("cora", "citeseer"):
        droot = root / dataset
        plan, accept = read(droot / "plan.json"), read(droot / "acceptance.json")
        check_matrix(plan, accept)
        audit = read(droot / "artifact_audit.json")
        if not audit["pass_audit"] or audit["fits"] != 324 or audit["memories"] != 54:
            raise AssertionError("artifact audit missing or wrong size")
        total_memories += audit["memories"]
        ev = {}
        for order in plan["orders"]:
            directory = droot / order
            path = directory / "results.json"
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            result = read(path)["results"]
            if len(result) != len(ARMS)*3:
                raise AssertionError("missing result cells")
            for arm in ARMS:
                stages = sorted([r for r in result if r["arm"] == arm], key=lambda r: r["stage"])
                if [r["stage"] for r in stages] != [0, 1, 2]:
                    raise AssertionError("missing stage")
                meta_seconds = shared_seconds = text_seconds = 0.
                for stage in stages:
                    if sorted(r["seed"] for r in stage["per_seed"]) != plan["seeds"]:
                        raise AssertionError("missing seed")
                    construction = read(directory / f"{arm}_s{stage['stage']}_construction.json")
                    shared = read(directory / f"shared_s{stage['stage']}.json")
                    if shared["evidence"]["budget_violations"]:
                        raise AssertionError("invalid token budget")
                    for fit in stage["per_seed"]:
                        if fit != read(directory / f"{arm}_s{stage['stage']}_seed{fit['seed']}.json"):
                            raise AssertionError("aggregate differs from raw fit")
                        if fit["support_sha256"] != construction["support_sha256"] or fit["steps"] != max(50, min(2048, 16*stage["rows"])):
                            raise AssertionError("wrong memory/steps")
                        total_fits += 1
                    meta = construction["meta"]
                    meta_seconds += meta["seconds"]
                    shared_seconds += shared["cluster_topology_seconds"]
                    text_seconds += shared["evidence"]["concat_seconds"] if arm == "center_concat" else shared["evidence"]["evidence_seconds"]
                    if arm.startswith(("semantic_", "student_")):
                        mechanism.append(dict(dataset=dataset, order=order, arm=arm, stage=stage["stage"],
                            selected_step=meta["selected_step"], initial_loss=meta["initial_validation"],
                            selected_loss=meta["selected_validation"], loss_gain=meta["initial_validation"]-meta["selected_validation"],
                            max_cosine_drift=meta["max_target_cosine_drift"], nonzero_gradient_steps=meta["nonzero_gradient_steps"],
                            train_probe_steps=meta["train_probe_steps"], validation_probe_steps=meta["validation_probe_steps"],
                            seconds=meta["seconds"], peak_allocated_mb=meta["peak_allocated_mb"]))
                for seed in plan["seeds"]:
                    ev[(order, arm, seed)] = seed_summary(stages, seed)
                line = dict(dataset=dataset, order=order, arm=arm,
                    **{k: S.mean(ev[(order, arm, seed)][k] for seed in plan["seeds"]) for k in KEYS})
                order_rows.append(line)
                student_seconds = sum(f["seconds"] for s in stages for f in s["per_seed"])
                construction_seconds = shared_seconds + text_seconds + meta_seconds
                costs.append(dict(dataset=dataset, order=order, arm=arm,
                    common_cluster_topology_seconds=shared_seconds, text_query_seconds=text_seconds,
                    optimization_seconds=meta_seconds, construction_seconds=construction_seconds,
                    report_six_seed_training_seconds=student_seconds,
                    one_seed_chain_training_seconds=student_seconds / len(plan["seeds"]),
                    single_use_warm_cache_seconds=construction_seconds + student_seconds / len(plan["seeds"])))
            for family, members in FAMILIES.items():
                for seed in plan["seeds"]:
                    ev[(order, family, seed)] = {k: S.mean(ev[(order, a, seed)][k] for a in members) for k in KEYS}
        for arm in (*ARMS, "semantic", "student"):
            cells = [ev[(o, arm, seed)] for o in plan["orders"] for seed in plan["seeds"]]
            rows.append(dict(dataset=dataset, arm=arm,
                **{k: S.mean(c[k] for c in cells) for k in KEYS},
                final_descriptive_std=S.pstdev(c["final"] for c in cells)))
        for control in ("center_concat", "fixed_anchor", "semantic"):
            diffs = [{k: ev[(o, "student", seed)][k] - ev[(o, control, seed)][k] for k in KEYS}
                     for o in plan["orders"] for seed in plan["seeds"]]
            paired.append(dict(dataset=dataset, control=control,
                **{k: S.mean(d[k] for d in diffs) for k in KEYS},
                positive_final_cells=sum(d["final"] > 0 for d in diffs),
                per_order_final={o: S.mean(ev[(o, "student", seed)]["final"] - ev[(o, control, seed)]["final"]
                                         for seed in plan["seeds"]) for o in plan["orders"]}))
        for method_seed in (700, 701):
            gains.append(dict(dataset=dataset, method_seed=method_seed,
                gain=S.mean(ev[(o, f"student_{method_seed}", seed)]["final"]-ev[(o, "center_concat", seed)]["final"]
                            for o in plan["orders"] for seed in plan["seeds"])))
    if total_fits != 648 or total_memories != 108:
        raise AssertionError("matrix not complete")
    screening = screen(paired, gains)
    summary = dict(rows=rows, per_order=order_rows, paired=paired, costs=costs, mechanism=mechanism,
        method_seed_gains=gains, screening=screening, source_sha256=hashes,
        acceptance=dict(fits=total_fits, memories=total_memories, artifact_audit_passed=True),
        uncertainty="descriptive dispersion, not independent-sample CI; known-order diagnosis only")
    save(out / "summary.json", summary)
    for name, table in (("dataset_metrics", rows), ("per_order_metrics", order_rows), ("costs", costs), ("mechanism", mechanism)):
        with (out / f"{name}.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    lines = ["# P2-B Results", "", "All-prompt report-dev Macro-F1; 3 known orders, 6 student seeds.", "",
             "|Dataset|Arm|Final|Stage mean|Old|New|Forgetting|", "|---|---|---:|---:|---:|---:|---:|"]
    lines += [f"|{r['dataset']}|{r['arm']}|{r['final']:.2f}|{r['stage_mean']:.2f}|{r['old']:.2f}|{r['new']:.2f}|{r['forgetting']:.2f}|" for r in rows]
    lines += ["", f"Screen passed: {screening['pass_screen']}; independent confirmation: {screening['independent_confirmation']}.",
              "Method seeds are averaged, not selected. Source-validation is only outer-loss-held-out."]
    (out / "metrics.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps(dict(acceptance=summary["acceptance"], screening=screening, rows=rows), indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    summarize(args.root, args.output)
