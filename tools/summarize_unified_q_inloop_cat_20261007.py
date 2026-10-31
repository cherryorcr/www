"""Aggregate complete, protocol-matched in-loop CaT cells; never select winners."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from statistics import mean, stdev


NAMES = {
    "uq_dm_edgefree": "In-loop CaT-Q (edge-free)",
    "uq_dm_joint": "In-loop CaT-Q (Q topology)",
    "cat_cctp": "CaT-CCTP",
    "uq_v3": "Unified-Q v3",
    "uq_edgefree384": "Unified-Q edge-free, matched 384 proxy",
    "uq_dm_edgefree_no_gm": "In-loop CaT-Q, no GM",
    "uq_dm_edgefree_fixed_q": "In-loop CaT-Q, fixed Q",
}
PAIRINGS = (("uq_dm_joint", "uq_v3"),
            ("uq_dm_edgefree", "uq_edgefree384"),
            ("uq_dm_edgefree_no_gm", "uq_dm_edgefree"),
            ("uq_dm_edgefree_fixed_q", "uq_dm_edgefree"),
            ("uq_dm_edgefree", "cat_cctp"))
PROTOCOL_KEYS = ("dataset", "groups", "budgets", "budget_mode", "order_seed",
    "construction_seed", "student", "student_steps", "student_batch", "student_seeds",
    "report_dev_mod", "evaluation_split", "report_mask_hash", "pool_mask_hash",
    "stage_id_hashes", "access", "boundary_enabled", "raw_eval_graph", "max_stages")


def require_pair_protocol(left, right):
    for key in PROTOCOL_KEYS:
        if key not in left or key not in right or left[key] != right[key]:
            raise ValueError(f"paired protocol mismatch: {key}")
    for name in ("graphtext_prompt_student.py", "external_continual_mechanisms.py",
                 "run_unified_q_large_smoke_20261007.py"):
        if left["source_hashes"].get(name) != right["source_hashes"].get(name):
            raise ValueError(f"paired implementation mismatch: {name}")


def collect(root):
    cells, pending, recomputed = {}, [], 0
    for path in sorted(root.rglob("results.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        protocol = data["protocol"]
        grouped = defaultdict(list)
        for row in data["rows"]:
            grouped[row["arm"]].append(row)
        for arm in protocol["arms"]:
            rows = sorted(grouped[arm], key=lambda r: r["stage"])
            stages = len(protocol["groups"])
            if [r["stage"] for r in rows] != list(range(stages)):
                pending.append({"path": str(path), "arm": arm, "stages": len(rows)})
                continue
            for row in rows:
                if row["memory_rows"] != protocol["budgets"][row["stage"]]:
                    raise ValueError("row budget mismatch")
                if not row["checks"].get("prefix_ok"):
                    raise ValueError("frozen-prefix check missing or failed")
                score = row["score"]
                if [s["seed"] for s in score["per_student"]] != protocol["student_seeds"]:
                    raise ValueError("Student seed mismatch")
                for key in ("final_f1", "accuracy", "old_f1", "new_f1"):
                    values = [s[key] for s in score["per_student"]]
                    if key == "old_f1" and row["stage"] == 0:
                        if any(v is not None for v in values) or score[key] is not None:
                            raise ValueError("base stage has old-class results")
                        continue
                    if not all(math.isfinite(v) for v in values):
                        raise ValueError("nonfinite scores")
                    if abs(mean(values) - score[key]) > 1e-8:
                        raise ValueError(f"incorrect aggregated score: {key}")
                    recomputed += 1
                if arm != "cat_cctp":
                    if not row["checks"].get("sinkhorn_feasible"):
                        raise ValueError("Sinkhorn infeasible")
                    if row["audit"]["text_hard_forward_max_error"] > 1e-4:
                        raise ValueError("text hard-forward mismatch")
                    if row["audit"]["cat_anchor_used"]:
                        raise ValueError("in-loop study unexpectedly uses CaT warm-start")
            key = (protocol["dataset"], protocol["order_seed"], protocol["construction_seed"], arm)
            if key in cells:
                raise ValueError(f"duplicate cell, do not choose a better repeat: {key}")
            final = rows[-1]["score"]
            stats = [r["audit"].get("distribution_matching", {}) for r in rows]
            dm = [s for s in stats if s.get("enabled")]
            cells[key] = {"dataset": key[0], "order_seed": key[1], "construction_seed": key[2],
                "arm": arm, "protocol": protocol, "source": str(path.resolve()),
                "final_f1": final["final_f1"], "final_f1_std": final["final_f1_std"],
                "accuracy": final["accuracy"], "accuracy_std": final["accuracy_std"],
                "old_f1": final["old_f1"], "new_f1": final["new_f1"],
                "stage_f1": mean(r["score"]["final_f1"] for r in rows),
                "stage_acc": mean(r["score"]["accuracy"] for r in rows),
                "worst_stage": min(r["score"]["final_f1"] for r in rows),
                "memory_rows": rows[-1]["memory_rows"],
                "construction_seconds": sum(r["construction_seconds"] for r in rows),
                "student_seconds": sum(r["student_seconds"] for r in rows),
                "construction_peak_mb": max(r["construction_peak_mb"] or 0 for r in rows),
                "distribution": {k: mean(s[k] for s in dm) for k in (
                    "initial_train", "final_train", "initial_holdout_bank", "final_holdout_bank")} if dm else {},
                "unique_text_ratio": mean(r["audit"].get("unique_text_ratio", 1.) for r in rows)
                    if arm != "cat_cctp" else None,
                "student_scores": final["per_student"]}
    pairs = []
    for key, current in cells.items():
        for candidate, control in PAIRINGS:
            if key[3] != candidate or (*key[:3], control) not in cells:
                continue
            reference = cells[(*key[:3], control)]
            require_pair_protocol(current["protocol"], reference["protocol"])
            delta = [a["final_f1"] - b["final_f1"] for a, b in
                     zip(current["student_scores"], reference["student_scores"])]
            pairs.append({"dataset": key[0], "order_seed": key[1], "construction_seed": key[2],
                "candidate": candidate, "control": control, "delta_f1": mean(delta),
                "delta_f1_student_sd": stdev(delta) if len(delta) > 1 else 0,
                "delta_accuracy": current["accuracy"] - reference["accuracy"],
                "delta_old_f1": current["old_f1"] - reference["old_f1"],
                "delta_new_f1": current["new_f1"] - reference["new_f1"]})
    return {"cells": list(cells.values()), "paired": pairs, "pending": pending,
            "audit": {"recomputed_metric_means": recomputed, "complete_cells": len(cells),
                      "method_stages": sum(len(c["protocol"]["groups"]) for c in cells.values()),
                      "limitation": "one reference order and construction seed; report-dev only"}}


def render(result):
    lines = ["# Unified-Q In-loop CaT Results", "",
        "Numbers are percentages. SD is across three Student seeds, not independent construction/order repeats.",
        "All arms use the same Prompt Student and report-dev evaluation. No final-test selection.", ""]
    for dataset in ("wikics", "sportsfit", "bookhis"):
        rows = [r for r in result["cells"] if r["dataset"] == dataset]
        if not rows:
            continue
        lines += [f"## {dataset}", "",
            "| Method | Final F1 | Accuracy | Stage F1 | Old F1 | New F1 | Build s | Peak MiB |",
            "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for r in sorted(rows, key=lambda r: list(NAMES).index(r["arm"])):
            lines.append(f"| {NAMES[r['arm']]} | {r['final_f1']:.2f} +/- {r['final_f1_std']:.2f} | "
                f"{r['accuracy']:.2f} +/- {r['accuracy_std']:.2f} | {r['stage_f1']:.2f} | "
                f"{r['old_f1']:.2f} | {r['new_f1']:.2f} | {r['construction_seconds']:.1f} | {r['construction_peak_mb']:.0f} |")
        lines += ["", f"Memory rows: {rows[0]['memory_rows']}.", ""]
    lines += ["## Matched Differences", "", "| Dataset | Candidate - Control | F1 pp | Acc pp | Old pp | New pp |",
              "|---|---|---:|---:|---:|---:|"]
    for p in result["paired"]:
        lines.append(f"| {p['dataset']} | {p['candidate']} - {p['control']} | {p['delta_f1']:+.2f} | "
            f"{p['delta_accuracy']:+.2f} | {p['delta_old_f1']:+.2f} | {p['delta_new_f1']:+.2f} |")
    lines += ["", "## Distribution Diagnostics", "", "Arithmetic means across stages, not performance metrics.", "",
        "| Dataset | Arm | Train before | Train after | Holdout bank before | Holdout bank after |",
        "|---|---|---:|---:|---:|---:|"]
    for row in result["cells"]:
        d = row["distribution"]
        if d:
            lines.append(f"| {row['dataset']} | {row['arm']} | {d['initial_train']:.5f} | {d['final_train']:.5f} | "
                         f"{d['initial_holdout_bank']:.5f} | {d['final_holdout_bank']:.5f} |")
    lines += ["", "## Audit", "", json.dumps(result["audit"], indent=2), "",
              f"Incomplete arms: {len(result['pending'])}. Incomplete scores are not included.", ""]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("results/unified_q_inloop_cat_20261007"))
    args = p.parse_args()
    result = collect(args.root)
    args.root.mkdir(parents=True, exist_ok=True)
    (args.root / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (args.root / "summary.md").write_text(render(result), encoding="utf-8")
    print(json.dumps(result["audit"], indent=2))
    print("pending:", len(result["pending"]))


if __name__ == "__main__":
    main()
