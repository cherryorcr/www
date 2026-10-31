"""Protocol-checked validation tables and crossed paired confidence intervals."""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev
import numpy as np


NAMES = {
    "uq_dm_edgefree": "In-loop CaT-Q",
    "uq_dm_edgefree_no_gm": "In-loop CaT-Q, no GM",
    "cat_cctp": "CaT-CCTP adapter",
    "uq_dm_edgefree_fixed_q": "In-loop CaT-Q, fixed Q",
    "random_coreset": "Stratified random coreset",
    "kcenter_coreset": "Stratified graph-text k-center",
    "gecc_hard": "GECC-hard TAG adapter",
    "puma_tag": "PUMA-inspired TAG adapter",
    "unchanged": "Same Q/X/text, self loops",
    "text_blank": "Same X, blank text",
    "text_cctp": "Same X, CCTP top-5 text",
    "text_permuted": "Same X, within-class text permutation",
    "topology_hard": "Same Q/X/text, hard graph",
    "topology_weighted": "Same Q/X/text, weighted graph",
    "topology_permuted": "Same Q/X/text, edge-budget-matched shuffled graph",
}
PRIMARY = ("uq_dm_edgefree", "uq_dm_edgefree_no_gm", "cat_cctp")
DIAGNOSTIC = ("unchanged", "text_blank", "text_cctp", "text_permuted",
              "topology_hard", "topology_weighted", "topology_permuted")
METRICS = ("final_f1", "accuracy", "old_f1", "new_f1", "stage_f1", "stage_acc", "worst_stage")
MATCH = ("dataset", "groups", "budgets", "order_seed", "construction_seed", "student",
         "student_steps", "student_batch", "student_seeds", "report_dev_mod",
         "report_mask_hash", "pool_mask_hash", "stage_id_hashes")


def pair_protocol(left, right):
    for key in MATCH:
        if key not in left or key not in right or left[key] != right[key]:
            raise ValueError(f"protocol mismatch: {key}")
    for name in ("graphtext_prompt_student.py", "run_unified_q_large_smoke_20261007.py"):
        if left["source_hashes"].get(name) != right["source_hashes"].get(name):
            raise ValueError(f"evaluator implementation mismatch: {name}")


def crossed_interval(matrix, samples=10000):
    """Average Student seeds first; resample order and construction axes paired."""
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or min(values.shape) < 2 or not np.isfinite(values).all():
        raise ValueError("complete >=2 by >=2 factorial required")
    rng = np.random.default_rng(20261007)
    orders = rng.integers(values.shape[0], size=(samples, values.shape[0]))
    seeds = rng.integers(values.shape[1], size=(samples, values.shape[1]))
    boot = values[orders[:, :, None], seeds[:, None, :]].mean(axis=(1,2))
    return np.quantile(boot, [0.025,0.975]).tolist()


def check_fixed_content(data):
    """Pair diagnostic arms with their own construction, never a different run."""
    if not any(arm in DIAGNOSTIC for arm in data["protocol"]["arms"]):
        return 0
    references = {r["stage"]: r for r in data["rows"] if r["arm"] == "unchanged"}
    checked = 0
    for row in data["rows"]:
        if row["arm"] not in DIAGNOSTIC:
            raise ValueError("fixed-content diagnostics mixed with candidate constructions")
        reference = references.get(row["stage"])
        if reference is None:
            raise ValueError("missing same-run fixed-content reference")
        if not row.get("content_hash") or row["content_hash"] != reference.get("content_hash"):
            raise ValueError("diagnostic changed X content")
        if row["memory_rows"] != reference["memory_rows"]:
            raise ValueError("diagnostic changed memory budget")
        if row["arm"].startswith("topology"):
            if not row.get("text_hash") or row["text_hash"] != reference.get("text_hash"):
                raise ValueError("topology diagnostic changed text")
        elif row.get("operator_hash") != reference.get("operator_hash"):
            raise ValueError("text diagnostic changed propagation operator")
        checked += 1
    return checked


def collect(roots):
    cells, pending = {}, []
    recomputed = 0
    diagnostic_rows_checked = 0
    for root in roots:
        for path in sorted(root.rglob("results.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            protocol = data["protocol"]
            diagnostic_rows_checked += check_fixed_content(data)
            for arm in protocol["arms"]:
                if arm not in NAMES:
                    continue
                rows = sorted([r for r in data["rows"] if r["arm"] == arm], key=lambda r:r["stage"])
                if [r["stage"] for r in rows] != list(range(len(protocol["groups"]))):
                    pending.append(dict(source=str(path), arm=arm, completed_stages=len(rows)))
                    continue
                for r in rows:
                    if r["memory_rows"] != protocol["budgets"][r["stage"]]:
                        raise ValueError("budget differs")
                    if not all(r["checks"].get(k) for k in ("budget_ok", "prefix_ok", "finite_ok")):
                        raise ValueError("invariant missing or false")
                    scores = r["score"]["per_student"]
                    if [s["seed"] for s in scores] != protocol["student_seeds"]:
                        raise ValueError("seed mismatch")
                    for metric in ("final_f1", "accuracy", "old_f1", "new_f1"):
                        values = [s[metric] for s in scores]
                        if metric == "old_f1" and r["stage"] == 0:
                            if any(v is not None for v in values):
                                raise ValueError("unexpected base old metric")
                            continue
                        if not all(math.isfinite(v) for v in values) or abs(mean(values)-r["score"][metric]) > 1e-8:
                            raise ValueError("metric aggregation discrepancy")
                        recomputed += 1
                    if arm.startswith("uq_") and not r["checks"].get("sinkhorn_feasible"):
                        raise ValueError("Sinkhorn check failed")
                key = (protocol["dataset"], protocol["order_seed"], protocol["construction_seed"], arm)
                if key in cells:
                    raise ValueError(f"duplicate cell (cannot select better repeat): {key}")
                final = rows[-1]["score"]
                cell = dict(dataset=key[0], order_seed=key[1], construction_seed=key[2], arm=arm,
                    source=str(path.resolve()), protocol=protocol, final_f1=final["final_f1"],
                    accuracy=final["accuracy"], old_f1=final["old_f1"], new_f1=final["new_f1"],
                    student_f1_sd=final["final_f1_std"], student_acc_sd=final["accuracy_std"],
                    stage_f1=mean(r["score"]["final_f1"] for r in rows),
                    stage_acc=mean(r["score"]["accuracy"] for r in rows),
                    worst_stage=min(r["score"]["final_f1"] for r in rows),
                    construction_seconds=sum(r.get("construction_seconds",0) for r in rows),
                    student_seconds=sum(r["student_seconds"] for r in rows),
                    peak_mb=max(r.get("construction_peak_mb",0) or 0 for r in rows),
                    memory_rows=rows[-1]["memory_rows"], memory_hash=rows[-1]["memory_hash"],
                    per_student=final["per_student"])
                cells[key] = cell
    pairs = []
    for key, cell in cells.items():
        if cell["arm"] not in PRIMARY:
            continue
        control = cells.get((*key[:3], "cat_cctp"))
        if control is not None:
            pair_protocol(cell["protocol"], control["protocol"])
    aggregates=[]
    for dataset in ("wikics", "sportsfit", "bookhis"):
        for arm in PRIMARY:
            items=[cells[k] for k in cells if k[0]==dataset and k[3]==arm]
            if len(items) != 9:
                continue
            if {(r["order_seed"],r["construction_seed"]) for r in items} != {(o,c) for o in (9101,9102,9103) for c in (271006,271007,271008)}:
                raise ValueError("unexpected factorial")
            aggregates.append(dict(dataset=dataset,arm=arm,n_cells=9,
                **{m:mean(r[m] for r in items) for m in METRICS},
                f1_cell_sd=stdev(r["final_f1"] for r in items), acc_cell_sd=stdev(r["accuracy"] for r in items),
                min_cell_f1=min(r["final_f1"] for r in items),
                mean_construction_seconds=mean(r["construction_seconds"] for r in items),
                mean_student_seconds=mean(r["student_seconds"] for r in items), memory_rows=items[0]["memory_rows"]))
        for candidate,control in ((PRIMARY[0],PRIMARY[2]),(PRIMARY[1],PRIMARY[2]),(PRIMARY[1],PRIMARY[0])):
            keys=[(dataset,o,c) for o in (9101,9102,9103) for c in (271006,271007,271008)]
            if not all((*k,a) in cells for k in keys for a in (candidate,control)):
                continue
            diffs=[]
            for k in keys:
                a,b=cells[(*k,candidate)],cells[(*k,control)]
                pair_protocol(a["protocol"],b["protocol"])
                diffs.append({m:a[m]-b[m] for m in METRICS})
            pair=dict(dataset=dataset,candidate=candidate,control=control,n_cells=9,
                positive_cell_ratio=mean(d["final_f1"]>0 for d in diffs),
                worst_delta_f1=min(d["final_f1"] for d in diffs),
                **{f"delta_{m}":mean(d[m] for d in diffs) for m in METRICS})
            for m in ("final_f1","accuracy"):
                pair[f"ci95_{m}"]=crossed_interval(np.array([d[m] for d in diffs]).reshape(3,3))
            pairs.append(pair)
    # Inputs/evaluators must match; a newly constructed diagnostic reference is
    # NOT the original pilot memory. Its causal contrasts stay within its run.
    diagnostic_replays = []
    for key,cell in cells.items():
        main=cells.get((*key[:3],PRIMARY[0]))
        if main is not None and cell["arm"] not in PRIMARY:
            pair_protocol(cell["protocol"],main["protocol"])
        if cell["arm"] == "unchanged" and main is not None:
            diagnostic_replays.append(dict(dataset=cell["dataset"],
                exact_pilot_replay=cell["memory_hash"] == main["memory_hash"],
                f1_delta_from_pilot=cell["final_f1"]-main["final_f1"],
                accuracy_delta_from_pilot=cell["accuracy"]-main["accuracy"],
                source=cell["source"], reference_source=main["source"],
                inference_scope="within-run fixed-content contrasts only; not pooled as another primary cell"))
    expected={(d,o,c,a) for d in ("wikics","sportsfit","bookhis")
              for o in (9101,9102,9103) for c in (271006,271007,271008) for a in PRIMARY}
    missing=sorted(expected-set(cells))
    return dict(cells=list(cells.values()),aggregates=aggregates,paired=pairs,pending=pending,
        diagnostic_replays=diagnostic_replays,
        missing_confirmation_cells=missing,
        audit=dict(complete_method_cells=len(cells),recomputed_metric_means=recomputed,
            expected_confirmation_cells=len(expected), complete_confirmation_cells=len(expected)-len(missing),
            diagnostic_fixed_content_rows_checked=diagnostic_rows_checked,
            diagnostic_scope="same-construction reference, separate from confirmation and external baseline table",
            student_seeds_not_independent_constructions=True,
            paired_ci="crossed order x construction bootstrap; average 3 matched Student seeds first; only 3 orders, exploratory",
            protocol_note="pilot fixed-rate reused only where actual groups, budgets, masks, evaluator hashes match fixed schedule",
            sealed_test=False))


def render(result):
    lines=["# In-loop CaT-Q validation", "", "All metrics are report-dev percentages. No final-test tuning.",
        "Primary remains In-loop CaT-Q; no-GM is a preregistered ablation. No per-dataset winner selection.", ""]
    lines += ["## Independent order/construction confirmation", "",
        "SD below is across 9 order/construction cell means, not 27 independent replicates.","",
        "| Dataset | Method | F1 +/- cell SD | Accuracy +/- cell SD | Stage F1 | Stage Acc | Old F1 | New F1 | Worst stage mean |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in result["aggregates"]:
        lines.append(f"| {r['dataset']} | {NAMES[r['arm']]} | {r['final_f1']:.2f} +/- {r['f1_cell_sd']:.2f} | {r['accuracy']:.2f} +/- {r['acc_cell_sd']:.2f} | {r['stage_f1']:.2f} | {r['stage_acc']:.2f} | {r['old_f1']:.2f} | {r['new_f1']:.2f} | {r['worst_stage']:.2f} |")
    lines += ["", "## Paired confirmation", "", "Crossed bootstrap; three orders is a limited sample.", "",
        "| Dataset | Candidate - control | F1 delta [95% CI] | Acc delta [95% CI] | Positive cells | Worst delta |",
        "|---|---|---:|---:|---:|---:|"]
    for r in result["paired"]:
        lo,hi=r["ci95_final_f1"]; la,ha=r["ci95_accuracy"]
        lines.append(f"| {r['dataset']} | {r['candidate']} - {r['control']} | {r['delta_final_f1']:+.2f} [{lo:+.2f}, {hi:+.2f}] | {r['delta_accuracy']:+.2f} [{la:+.2f}, {ha:+.2f}] | {r['positive_cell_ratio']:.0%} | {r['worst_delta_f1']:+.2f} |")
    for dataset in ("wikics","sportsfit","bookhis"):
        lines += ["",f"## {dataset}: reference order / construction seed", "", "SD here is across 3 Student seeds only.", "",
                  "| Method | Final F1 | Accuracy | Stage F1 | Old F1 | New F1 | Rows | Build s | Student s |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        rows=[r for r in result["cells"] if r["dataset"]==dataset and r["order_seed"]==9101
              and r["construction_seed"]==271006 and r["arm"] not in DIAGNOSTIC]
        for r in sorted(rows,key=lambda r:list(NAMES).index(r["arm"])):
            lines.append(f"| {NAMES[r['arm']]} | {r['final_f1']:.2f} +/- {r['student_f1_sd']:.2f} | {r['accuracy']:.2f} +/- {r['student_acc_sd']:.2f} | {r['stage_f1']:.2f} | {r['old_f1']:.2f} | {r['new_f1']:.2f} | {r['memory_rows']} | {r['construction_seconds']:.1f} | {r['student_seconds']:.1f} |")
    lines += ["", "## Fixed-content attribution (separate construction)", "",
        "Each contrast below uses exactly the SAME X within a diagnostic run. Topology arms also share exactly the SAME text.",
        "The reconstructed reference is NOT a bitwise replay of the earlier pilot. Do not compare these arms against the earlier pilot as a causal pair or pool the replay into primary means.", "",
        "| Dataset | Diagnostic | F1 | Accuracy | F1 delta vs same-run self-loop | Acc delta |",
        "|---|---|---:|---:|---:|---:|"]
    for dataset in ("wikics", "sportsfit", "bookhis"):
        rows={r["arm"]:r for r in result["cells"] if r["dataset"]==dataset and r["arm"] in DIAGNOSTIC}
        if "unchanged" not in rows:
            continue
        ref=rows["unchanged"]
        for arm in DIAGNOSTIC:
            if arm not in rows:
                continue
            r=rows[arm]
            lines.append(f"| {dataset} | {NAMES[arm]} | {r['final_f1']:.2f} | {r['accuracy']:.2f} | {r['final_f1']-ref['final_f1']:+.2f} | {r['accuracy']-ref['accuracy']:+.2f} |")
    lines += ["", "### Replay audit", "",
        "| Dataset | Exact pilot memory replay | F1 difference | Accuracy difference |",
        "|---|---|---:|---:|"]
    for r in result["diagnostic_replays"]:
        lines.append(f"| {r['dataset']} | {r['exact_pilot_replay']} | {r['f1_delta_from_pilot']:+.2f} | {r['accuracy_delta_from_pilot']:+.2f} |")
    lines += ["", "## Limitations", "", "- GECC/PUMA/CaT are explicit mechanism adaptations, not official reproductions.",
        "- Fixed-content topology variants are attribution, not new jointly optimized methods.",
        "- Fixed-content reference replay drift is disclosed above; its source is not established by RNG preservation alone.",
        "- Weighted topology uses weighted symmetric normalization during Student training, full raw graph at inference.",
        "- No boundary sketch; no changes to old synthetic nodes; no unseen-class or cross-dataset claim.",
        "- Timing may include shared resources; initial exclusive pilot remains the efficiency reference.",
        "- Missing cells are not included in independent-confirmation means.","",f"Started incomplete method cells: {len(result['pending'])}. Missing confirmation cells: {len(result['missing_confirmation_cells'])}.","",
        "## Audit", "", "```json",json.dumps(result["audit"],indent=2),"```",""]
    return "\n".join(lines)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--roots",nargs="+",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    result=collect(args.roots)
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/"summary.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    (args.output/"summary.md").write_text(render(result),encoding="utf-8")
    fields=["dataset","order_seed","construction_seed","arm",*METRICS,"student_f1_sd","student_acc_sd","memory_rows","construction_seconds","student_seconds","peak_mb","source"]
    with (args.output/"cells.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore")
        writer.writeheader();writer.writerows(result["cells"])
    print(json.dumps(result["audit"],indent=2))


if __name__ == "__main__":
    main()
