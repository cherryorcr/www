"""Summarize complete revision cells without mixing budget policies."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


NAMES = {
    "cat_cctp": "CaT-MMD + CCTP top-5", "uq_consistent": "UQ consistency fixes",
    "uq_balanced": "UQ + group/class balance", "uq_semantic": "UQ + semantic initialization",
    "uq_text_frozen": "UQ text frozen", "uq_topology_frozen": "UQ topology frozen",
    "uq_edge_free": "UQ edge-free", "uq_alternating": "UQ alternating Q/R",
}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(root: Path) -> tuple[list[dict], list[dict]]:
    finals, stages = [], []
    for path in sorted(root.rglob("results.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("status") != "complete":
            continue
        protocol = result["protocol"]
        cell = {"dataset": protocol["dataset"], "budget_mode": protocol["budget_mode"],
                "order_seed": protocol["order_seed"],
                "construction_seed": protocol["construction_seed"],
                "source": str(path.relative_to(root))}
        for arm in sorted({r["arm"] for r in result["rows"]}):
            rows = sorted([r for r in result["rows"] if r["arm"] == arm], key=lambda r: r["stage"])
            if len(rows) != len(protocol["groups"]):
                continue
            for row in rows:
                a, s = row["audit"], row["score"]
                stages.append(dict(cell, arm=arm, stage=row["stage"], raw_rows=row["raw_rows"],
                    new_rows=row["new_rows"], memory_rows=row["memory_rows"],
                    macro_f1=s["final_f1"], accuracy=s["accuracy"], old_f1=s["old_f1"], new_f1=s["new_f1"],
                    construction_seconds=row["construction_seconds"], student_seconds=row["student_seconds"],
                    construction_peak_mb=row["construction_peak_mb"],
                    q_row_error=a.get("q_row_sum_max_error"), text_unique_ratio=a.get("unique_text_ratio"),
                    real_new_loss_mass=a.get("real_new_loss_mass"),
                    synthetic_new_loss_mass=a.get("synthetic_new_loss_mass"),
                    propagation_max_error=row["checks"].get("propagation_max_error"),
                    text_forward_max_error=a.get("text_hard_forward_max_error")))
            last, score = rows[-1], rows[-1]["score"]
            violations = 0
            for row in rows:
                checks = row["checks"]
                violations += int(not all(checks.get(k) is True for k in ("budget_ok", "prefix_ok", "finite_ok")))
                violations += int(checks.get("propagation_max_error", 0) > 1e-6)
                violations += int(row["audit"].get("text_hard_forward_max_error", 0) > 1e-4)
            finals.append(dict(cell, arm=arm, final_f1=score["final_f1"], f1_student_std=score["final_f1_std"],
                accuracy=score["accuracy"], accuracy_student_std=score["accuracy_std"],
                stage_avg_f1=float(np.mean([r["score"]["final_f1"] for r in rows])),
                stage_avg_accuracy=float(np.mean([r["score"]["accuracy"] for r in rows])),
                old_f1=score["old_f1"], new_f1=score["new_f1"],
                worst_stage=min(r["score"]["final_f1"] for r in rows), memory_rows=last["memory_rows"],
                construction_seconds=sum(r["construction_seconds"] for r in rows),
                student_seconds=sum(r["student_seconds"] for r in rows),
                construction_peak_mb=max(r["construction_peak_mb"] or 0 for r in rows),
                violations=violations))
    return finals, stages


def paired_aggregate(final: list[dict]) -> list[dict]:
    """Pair cells first; never treat Student replicas as independent orders."""
    aggregate = []
    for dataset, mode in sorted({(r["dataset"], r["budget_mode"]) for r in final}):
        pool = [r for r in final if (r["dataset"], r["budget_mode"]) == (dataset, mode)]
        keys = [(r["arm"], r["order_seed"], r["construction_seed"]) for r in pool]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate method cell would double-count evidence")
        cats = {(r["order_seed"], r["construction_seed"]): r for r in pool if r["arm"] == "cat_cctp"}
        for arm in NAMES:
            rows = [r for r in pool if r["arm"] == arm and (r["order_seed"], r["construction_seed"]) in cats]
            if not rows:
                continue
            orders = sorted({r["order_seed"] for r in rows})
            same_seeds = all(
                {r["construction_seed"] for r in rows if r["order_seed"] == o} ==
                {r["construction_seed"] for r in rows if r["order_seed"] == orders[0]} for o in orders)
            record = {"dataset": dataset, "budget_mode": mode, "arm": arm,
                      "cells": len(rows), "orders": len(orders)}
            for metric, short in (("final_f1", "f1"), ("accuracy", "accuracy")):
                values = [r[metric] for r in rows]
                deltas = [r[metric] - cats[(r["order_seed"], r["construction_seed"])][metric] for r in rows]
                order_delta = [float(np.mean([d for r, d in zip(rows, deltas) if r["order_seed"] == o]))
                               for o in orders]
                low = high = None
                if len(order_delta) > 1 and same_seeds:
                    from scipy.stats import t
                    half = t.ppf(.975, len(order_delta) - 1) * np.std(order_delta, ddof=1) / np.sqrt(len(order_delta))
                    low, high = float(np.mean(order_delta) - half), float(np.mean(order_delta) + half)
                record.update({f"mean_{short}": float(np.mean(values)),
                    f"cell_std_{short}": float(np.std(values, ddof=1)) if len(values) > 1 else 0,
                    f"mean_delta_{short}": float(np.mean(deltas)),
                    f"positive_cell_ratio_{short}": float(np.mean(np.asarray(deltas) > 0)),
                    f"worst_cell_delta_{short}": min(deltas),
                    f"order_cluster_t95_low_{short}": low,
                    f"order_cluster_t95_high_{short}": high})
            for metric in ("stage_avg_f1", "stage_avg_accuracy", "old_f1", "new_f1",
                           "worst_stage", "construction_seconds", "student_seconds"):
                record[f"mean_{metric}"] = float(np.mean([r[metric] for r in rows]))
            record["min_worst_stage"] = min(r["worst_stage"] for r in rows)
            record["memory_rows_min"] = min(r["memory_rows"] for r in rows)
            record["memory_rows_max"] = max(r["memory_rows"] for r in rows)
            record["same_construction_seeds_each_order"] = same_seeds
            aggregate.append(record)
    return aggregate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args()
    final, stages = summarize(args.root)
    write_csv(args.root / "final_cells.csv", final)
    write_csv(args.root / "stage_cells.csv", stages)
    lines = ["# Unified-Q revision comparison", "",
             "All metrics are report-dev percentages, not final test. Student is unchanged.",
             "Per-cell uncertainty is over Student seeds only. Different budget policies are separate.",
             "Construction and Student times sum all stages; peak memory includes resident data/model.", ""]
    for key in sorted({(r["dataset"], r["budget_mode"], r["order_seed"], r["construction_seed"]) for r in final}):
        rows = [r for r in final if (r["dataset"], r["budget_mode"], r["order_seed"], r["construction_seed"]) == key]
        baseline = next((r for r in rows if r["arm"] == "cat_cctp"), None)
        lines.extend([f"## {key[0]} / {key[1]} / order {key[2]} / construction {key[3]}", "",
            "| Method | Final F1 | Accuracy | Stage-average F1 | Old F1 | New F1 | Delta vs CaT | Construct s | Peak MiB |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
        for row in sorted(rows, key=lambda r: list(NAMES).index(r["arm"])):
            delta = row["final_f1"] - baseline["final_f1"] if baseline else float("nan")
            lines.append(f"| {NAMES[row['arm']]} | {row['final_f1']:.3f} +/- {row['f1_student_std']:.3f} | "
                         f"{row['accuracy']:.3f} +/- {row['accuracy_student_std']:.3f} | "
                         f"{row['stage_avg_f1']:.3f} | {row['old_f1']:.3f} | {row['new_f1']:.3f} | "
                         f"{delta:+.3f} | {row['construction_seconds']:.2f} | {row['construction_peak_mb']:.1f} |")
        lines.append("")
    lines.extend(["## Audit", "", f"Complete method cells: {len(final)}.",
                  f"Runtime invariant violations: {sum(r['violations'] for r in final)}.",
                  "Full stage metrics and audit values are in stage_cells.csv.", ""])
    aggregate = paired_aggregate(final)
    write_csv(args.root / "aggregate_cells.csv", aggregate)
    lines.extend(["## Paired confirmation", "",
        "Fixed-schedule primary comparison only. Each cell averages the same three Student seeds.",
        "Cell standard deviations describe order/construction variation; they are not per-Student deviations.",
        "Incomplete grids are provisional. A variant evaluated in only one cell must not be ranked against a nine-cell mean.", "",
        "| Dataset | Method | Cells / orders | Final F1 | Accuracy | Delta F1 | Paired order CI | Positive cells | Old F1 | New F1 |",
        "|---|---|---:|---:|---:|---:|---|---:|---:|---:|"])
    for row in aggregate:
        if row["budget_mode"] != "fixed_schedule" or row["arm"] not in ("cat_cctp", "uq_semantic"):
            continue
        lo, hi = row["order_cluster_t95_low_f1"], row["order_cluster_t95_high_f1"]
        ci = "not estimable" if lo is None else f"[{lo:+.3f}, {hi:+.3f}]"
        lines.append(f"| {row['dataset']} | {NAMES[row['arm']]} | {row['cells']} / {row['orders']} | "
            f"{row['mean_f1']:.3f} +/- {row['cell_std_f1']:.3f} | "
            f"{row['mean_accuracy']:.3f} +/- {row['cell_std_accuracy']:.3f} | "
            f"{row['mean_delta_f1']:+.3f} | {ci} | {row['positive_cell_ratio_f1']:.1%} | "
            f"{row['mean_old_f1']:.3f} | {row['mean_new_f1']:.3f} |")
    lines.append("")
    audit_path = args.root / "audit_report.json"
    if audit_path.exists():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        warnings = audit["numerical_warnings"]
        lines.extend(["## Numerical feasibility", "",
            f"Independent integrity audit passed: {audit['integrity_passed']}.",
            f"Sinkhorn row residual above diagnostic tolerance 1e-3: {len(warnings)} recorded stages.",
            "The fixed 30-iteration solver is approximate; zero runtime artifact violations does not imply exact transport feasibility.", ""])
    lines.extend(["Paired confidence intervals in aggregate_cells.csv average construction/Student seeds within order",
                  "before applying a t interval across orders. Single-order runs have no generalization CI.",
                  "With only three orders the interval is descriptive and imprecise; Student replicates are not independent orders.", ""])
    (args.root / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    if args.plots and final:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for dataset in sorted({r["dataset"] for r in final}):
            rows = [r for r in final if r["dataset"] == dataset and r["budget_mode"] == "fixed_schedule"
                    and r["order_seed"] == 9101 and r["construction_seed"] == 271006]
            rows.sort(key=lambda r: list(NAMES).index(r["arm"]))
            if not rows:
                continue
            fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
            for ax, metric, std, title in zip(axes, ("final_f1", "accuracy"),
                                             ("f1_student_std", "accuracy_student_std"), ("Macro-F1", "Accuracy")):
                positions = np.arange(len(rows))
                ax.barh(positions, [r[metric] for r in rows], xerr=[r[std] for r in rows],
                        color=["#257B83" if r["arm"] == "uq_semantic" else "#87949F" for r in rows], capsize=3)
                ax.set_yticks(positions, [NAMES[r["arm"]] for r in rows])
                ax.set_ylim(len(rows) - .5, -.5)
                ax.set_xlim(0, 100)
                ax.set_xlabel(f"{title} (%)")
                ax.grid(axis="x", alpha=.2)
                for index, row in enumerate(rows):
                    ax.text(row[metric] + row[std] + .8, index, f"{row[metric]:.2f}",
                            va="center", fontsize=8)
            fig.suptitle(f"{dataset}: fixed schedule, order 9101, three Student seeds")
            fig.tight_layout()
            fig.savefig(args.root / f"{dataset}_ablation.png", dpi=160)
            plt.close(fig)
            paired = [r for r in final if r["dataset"] == dataset and r["budget_mode"] == "fixed_schedule"]
            baseline = {(r["order_seed"], r["construction_seed"]): r for r in paired if r["arm"] == "cat_cctp"}
            candidates = [r for r in paired if r["arm"] == "uq_semantic" and
                          (r["order_seed"], r["construction_seed"]) in baseline]
            if not candidates:
                continue
            orders = sorted({r["order_seed"] for r in candidates})
            seeds = sorted({r["construction_seed"] for r in candidates})
            fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
            for ax, metric, title in zip(axes, ("final_f1", "accuracy"), ("Macro-F1", "Accuracy")):
                grid = np.full((len(orders), len(seeds)), np.nan)
                for row in candidates:
                    grid[orders.index(row["order_seed"]), seeds.index(row["construction_seed"])] = (
                        row[metric] - baseline[(row["order_seed"], row["construction_seed"])][metric])
                span = max(1, float(np.nanmax(np.abs(grid))))
                plotted = ax.imshow(grid, vmin=-span, vmax=span, cmap="RdBu", aspect="auto")
                ax.set_xticks(range(len(seeds)), seeds)
                ax.set_yticks(range(len(orders)), orders)
                ax.set_xlabel("Construction seed")
                ax.set_ylabel("Class order")
                ax.set_title(f"Delta {title} (pp)")
                for i in range(len(orders)):
                    for j in range(len(seeds)):
                        if np.isfinite(grid[i, j]):
                            ax.text(j, i, f"{grid[i,j]:+.2f}", ha="center", va="center",
                                    color="white" if abs(grid[i,j]) > span*.6 else "black")
                fig.colorbar(plotted, ax=ax, shrink=.8)
            fig.suptitle(f"{dataset}: preregistered Unified-Q minus CaT-CCTP")
            fig.tight_layout()
            fig.savefig(args.root / f"{dataset}_paired_confirmation.png", dpi=160)
            plt.close(fig)
    print(json.dumps({"complete_cells": len(final), "stages": len(stages)}, indent=2))


if __name__ == "__main__":
    main()
