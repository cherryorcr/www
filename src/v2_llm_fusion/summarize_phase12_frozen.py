"""Summarize all preregistered cells; never promote from an incomplete matrix."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from build_phase1_table import summarize, write_csv
import run_frozen_baseline_joint_kernel as E


def evaluate_gates(cells, costs, canonical_tag_delta, complete):
    deltas = np.array([v["joint_minus_c0_pp"] for v in cells])
    old = np.array([v["joint_minus_c0_old_pp"] for v in cells])
    new = np.array([v["joint_minus_c0_new_pp"] for v in cells])
    core = np.array([v["joint_minus_kcenter_pp"] for v in cells])
    gates = {"complete_36_cells_and_audited": bool(complete),
        "positive_cells_at_least_75pct": bool(len(deltas) and np.mean(deltas > 0) >= .75),
        "mean_final_gain_at_least_1pp": bool(len(deltas) and deltas.mean() >= 1),
        "worst_final_not_below_minus2pp": bool(len(deltas) and deltas.min() >= -2),
        "worst_old_not_below_minus2pp": bool(len(old) and old.min() >= -2),
        "mean_new_positive": bool(len(new) and new.mean() > 0),
        "half_new_cells_positive": bool(len(new) and np.mean(new > 0) >= .5),
        "per_dataset_cost_at_most_2x": bool(costs and all(v["build_plus_one_student_ratio"] <= 2 for v in costs)),
        "mean_not_below_kcenter": bool(len(core) and core.mean() >= 0),
        "canonical_mean_not_below_tagsam_selfloop": canonical_tag_delta >= 0}
    return {"gates": gates, "promote": all(gates.values()),
            "positive_final_cells": int((deltas > 0).sum()), "total_cells": len(cells),
            "mean_final_delta_pp": float(deltas.mean()) if len(deltas) else None,
            "worst_final_delta_pp": float(deltas.min()) if len(deltas) else None,
            "worst_old_delta_pp": float(old.min()) if len(old) else None,
            "mean_new_delta_pp": float(new.mean()) if len(new) else None,
            "positive_new_cells": int((new > 0).sum()),
            "mean_vs_kcenter_pp": float(core.mean()) if len(core) else None,
            "canonical_mean_vs_tagsam_selfloop_pp": canonical_tag_delta}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--partial", action="store_true")
    args = ap.parse_args()
    output = args.root / "summary"
    output.mkdir(exist_ok=True)
    arms, cells, stage_rows, sources = [], [], [], {}
    expected = {f"{d}/s{n}_{o}_r{r:02d}" for d in ("cora", "citeseer")
                for n in (3,5) for o in ("canonical", "order19101", "order19102") for r in (1,5,10)}
    found = set()
    for dataset in ("cora", "citeseer"):
        for path in sorted((args.root / "matrix" / dataset).glob("s*_*_r*/results.json")):
            content = json.loads(path.read_text())
            cfg = content["manifest"]
            n = len(cfg["stage_classes"])
            order = path.parent.name.split("_")[1]
            name = f"{dataset}/{path.parent.name}"
            found.add(name)
            sources[str(path)] = E.C.sha(path)
            current = {}
            for arm in E.MATRIX_ARMS:
                stages = sorted((s for s in content["results"] if s["arm"] == arm), key=lambda s: s["stage"])
                if len(stages) != n or any(len(s["per_seed"]) != 6 for s in stages):
                    raise ValueError(f"incomplete cell {name}/{arm}")
                summary = summarize(stages)
                audits = content["audits"][arm]
                summary.update(dataset=dataset, cell=name, order=order, stage_count=n, rate=cfg["rate"], arm=arm,
                    build_seconds_chain=sum(a["build_seconds"] for a in audits),
                    final_memory_bytes=audits[-1]["payload"]["logical_bytes"],
                    final_edge_entries=audits[-1]["payload"]["edge_entries"])
                summary["build_plus_one_student_seconds"] = summary["build_seconds_chain"] + summary["student_seconds_one_seed_chain"]
                current[arm] = summary
                arms.append(summary)
                for stage, audit in zip(stages, audits):
                    stage_rows.append({"dataset": dataset, "cell": name, "order": order, "stage_count": n,
                        "rate": cfg["rate"], "arm": arm, "stage": stage["stage"], "rows": stage["rows"],
                        "f1": stage["metrics"]["all"]["cumulative_f1"]["mean"],
                        "old_f1": (stage["metrics"]["all"]["old_f1"] or {}).get("mean"),
                        "new_f1": stage["metrics"]["all"]["new_f1"]["mean"],
                        "build_seconds": audit["build_seconds"],
                        "student_seconds_mean": float(np.mean([r["seconds"] for r in stage["per_seed"]])),
                        "memory_bytes": audit["payload"]["logical_bytes"],
                        "edge_entries": audit["payload"]["edge_entries"],
                        "normalized_old_block_relative_drift": audit.get("normalized_old_block_relative_drift")})
            control, candidate, coreset = current["c0"], current["joint_kernel"], current["k_center"]
            delta = np.array(candidate["final_per_seed"]) - np.array(control["final_per_seed"])
            half = 2.5705818356 * delta.std(ddof=1) / np.sqrt(len(delta))
            cells.append({"dataset": dataset, "cell": name, "order": order, "stage_count": n, "rate": cfg["rate"],
                **{a + "_final_f1": current[a]["final_f1"] for a in E.MATRIX_ARMS},
                "joint_minus_c0_pp": candidate["final_f1"] - control["final_f1"],
                "joint_minus_c0_stage_avg_pp": candidate["stage_average_f1"] - control["stage_average_f1"],
                "joint_minus_c0_old_pp": candidate["final_old_f1"] - control["final_old_f1"],
                "joint_minus_c0_new_pp": candidate["final_new_f1"] - control["final_new_f1"],
                "joint_minus_kcenter_pp": candidate["final_f1"] - coreset["final_f1"],
                "joint_minus_recluster_pp": candidate["final_f1"] - current["recluster"]["final_f1"],
                "paired_student_seed_delta": delta.tolist(),
                "paired_seed_ci95": [float(delta.mean()-half), float(delta.mean()+half)],
                "build_plus_one_student_ratio": candidate["build_plus_one_student_seconds"] / control["build_plus_one_student_seconds"],
                "logical_memory_ratio": candidate["final_memory_bytes"] / control["final_memory_bytes"]})
    if not args.partial and found != expected:
        raise ValueError(f"matrix missing {sorted(expected-found)}; extra {sorted(found-expected)}")
    cost_rows, overview = [], []
    for dataset in ("cora", "citeseer"):
        subset = [r for r in arms if r["dataset"] == dataset]
        values = {a: sum(r["build_plus_one_student_seconds"] for r in subset if r["arm"] == a) for a in E.MATRIX_ARMS}
        if values["c0"]:
            cost_rows.append({"dataset": dataset, "build_plus_one_student_ratio": values["joint_kernel"] / values["c0"],
                              **{a + "_seconds": v for a,v in values.items()}})
        for arm in E.MATRIX_ARMS:
            selected = [r for r in subset if r["arm"] == arm]
            if not selected:
                continue
            overview.append({"dataset": dataset, "arm": arm, "cells": len(selected),
                **{k: float(np.mean([r[k] for r in selected])) for k in (
                    "final_f1", "stage_average_f1", "final_old_f1", "final_new_f1", "final_memory_bytes")},
                **{k: float(sum(r[k] for r in selected)) for k in (
                    "build_seconds_chain", "student_seconds_six_seeds", "student_steps_six_seeds")}})
    phase1 = json.loads((output / "phase1_summary.json").read_text())
    candidate_canonical = [r["final_f1"] for r in arms if r["arm"] == "joint_kernel"
                           and r["rate"] == .05 and r["order"] == "canonical" and r["stage_count"] == 3]
    tag = [r["final_f1"] for r in phase1["rows"] if r["arm"] == "tagsam_static_selfloop"]
    tag_delta = float(np.mean(candidate_canonical) - np.mean(tag))
    audit_paths = [args.root / f"audit_{d}.json" for d in ("cora", "citeseer")]
    audits = [json.loads(p.read_text()) for p in audit_paths if p.exists()]
    complete = found == expected and len(audits) == 2 and all(
        not a["partial"] and a["independent_model_reevaluation"] and a["error_count"] == 0
        and a["checked_fits"] == 1746 for a in audits)
    decision = evaluate_gates(cells, cost_rows, tag_delta, complete)
    suffix = "_partial" if args.partial else ""
    for name, rows in (("matrix_cells", cells), ("matrix_arms", arms), ("matrix_stages", stage_rows),
                       ("matrix_overview", overview), ("matrix_costs", cost_rows)):
        if rows:
            write_csv(output / f"{name}{suffix}.csv", rows)
    E.C.save_json(output / f"phase2_summary{suffix}.json", {"decision": decision, "cells": cells,
        "overview": overview, "costs": cost_rows, "sources": sources, "summarizer_sha256": E.C.sha(__file__),
        "uncertainty_scope": "paired seed CIs condition on one memory and class order; cells share data and are not independent datasets"})
    print(json.dumps(decision), flush=True)
    if not args.partial and not complete:
        raise SystemExit("full matrix audit required before completion")


if __name__ == "__main__":
    main()
