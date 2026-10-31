"""Create compact tables and plots from the frozen Unified-Q diagnostics."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "unified_q_diagnostics_20261007"
OUT = RESULTS / "assets"
ORDERS = (9101, 9102, 9103)
SEEDS = (271006, 271007, 271008)
METRICS = ("final_f1", "accuracy", "old_f1", "new_f1", "stage_avg_f1",
           "stage_avg_accuracy", "worst_stage", "construction_seconds", "student_seconds")


def read_csv(name: str):
    with (RESULTS / name).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def number(row, key):
    return float(row[key])


def avg(values):
    return mean(values) if values else float("nan")


def sd(values):
    return stdev(values) if len(values) > 1 else 0.0


def completed_reference(rows):
    return [
        row for row in rows
        if row["run"].startswith(("diag_", "retry_diag_", "transfer_", "memretry_transfer_"))
        and row["arm"] in {"cat_cctp", "uq_30", "uq_64", "uq_128", "uq_proxy384", "uq_v3"}
    ]


def paired_cells(rows):
    grouped = {}
    for row in rows:
        if (row["dataset"] not in ("wikics", "bookhis")
                or row["arm"] not in ("cat_cctp", "uq_v3")
                or row["budget_mode"] != "fixed_rate"
                or row["protocol_family"] != "frozen-prefix"
                or int(row["order_seed"]) not in ORDERS
                or int(row["construction_seed"]) not in SEEDS):
            continue
        key = (row["dataset"], row["order_seed"], row["construction_seed"])
        arms = grouped.setdefault(key, {})
        if row["arm"] in arms:
            raise ValueError(f"Duplicate completed cell: {key} {row['arm']}")
        arms[row["arm"]] = row
    output = []
    for (dataset, order, construction), arms in sorted(grouped.items()):
        if "cat_cctp" not in arms or "uq_v3" not in arms:
            continue
        if arms["cat_cctp"]["memory_rows"] != arms["uq_v3"]["memory_rows"]:
            raise ValueError(f"Different paired budgets: {dataset} {order} {construction}")
        delta_f1 = number(arms["uq_v3"], "final_f1") - number(arms["cat_cctp"], "final_f1")
        delta_acc = number(arms["uq_v3"], "accuracy") - number(arms["cat_cctp"], "accuracy")
        item = {
            "dataset": dataset,
            "order_seed": int(order),
            "construction_seed": int(construction),
            "cat_f1": number(arms["cat_cctp"], "final_f1"),
            "uq_f1": number(arms["uq_v3"], "final_f1"),
            "delta_f1": delta_f1,
            "cat_accuracy": number(arms["cat_cctp"], "accuracy"),
            "uq_accuracy": number(arms["uq_v3"], "accuracy"),
            "delta_accuracy": delta_acc,
            "cat_run": arms["cat_cctp"]["run"],
            "uq_run": arms["uq_v3"]["run"],
            "memory_rows": int(arms["cat_cctp"]["memory_rows"]),
        }
        for metric in METRICS:
            item[f"cat_{metric}"] = number(arms["cat_cctp"], metric)
            item[f"uq_{metric}"] = number(arms["uq_v3"], metric)
            item[f"delta_{metric}"] = item[f"uq_{metric}"] - item[f"cat_{metric}"]
        output.append(item)
    return output


def aggregate_pairs(pairs):
    methods, uncertainty, orders = [], [], []
    for dataset in sorted({r["dataset"] for r in pairs}):
        selected = [r for r in pairs if r["dataset"] == dataset]
        for arm, prefix in (("cat_cctp", "cat"), ("uq_v3", "uq")):
            item = {"dataset": dataset, "arm": arm, "cells": len(selected),
                    "memory_rows_min": min(r["memory_rows"] for r in selected),
                    "memory_rows_max": max(r["memory_rows"] for r in selected)}
            for metric in METRICS:
                values = [r[f"{prefix}_{metric}"] for r in selected]
                item[metric] = mean(values)
                item[metric + "_cell_std"] = sd(values)
            methods.append(item)
        for order in ORDERS:
            rows = [r for r in selected if r["order_seed"] == order]
            if rows:
                orders.append({"dataset": dataset, "order_seed": order, "cells": len(rows),
                    "reference_order": order == 9101,
                    **{f"delta_{m}": mean(r[f"delta_{m}"] for r in rows) for m in METRICS}})
        for metric in ("final_f1", "accuracy", "old_f1", "new_f1"):
            by_order = [r[f"delta_{metric}"] for r in orders if r["dataset"] == dataset]
            values = [r[f"delta_{metric}"] for r in selected]
            independent = [r[f"delta_{metric}"] for r in selected if r["order_seed"] != 9101]
            # Three orders, not 27 Student runs, are the inferential units.
            half = 4.3026527299 * sd(by_order) / math.sqrt(3) if len(selected) == 9 else None
            uncertainty.append({"dataset": dataset, "metric": metric, "cells": len(values),
                "orders": len(by_order), "mean_delta": mean(values),
                "cell_std": sd(values), "positive_cells": sum(v > 0 for v in values),
                "positive_ratio": sum(v > 0 for v in values) / len(values), "worst_delta": min(values),
                "order_t_ci_low": mean(by_order) - half if half is not None else None,
                "order_t_ci_high": mean(by_order) + half if half is not None else None,
                "independent_cells": len(independent),
                "independent_mean_delta": mean(independent) if independent else None})
    return methods, uncertainty, orders


def protocol_audit(pairs):
    failures, checked_stages, recomputed_metrics, paired_protocols = [], 0, 0, 0
    protocol_keys = ("dataset", "groups", "budgets", "budget_mode", "order_seed",
        "construction_seed", "student", "student_steps", "student_batch", "student_seeds",
        "report_dev_mod", "evaluation_split", "report_mask_hash", "pool_mask_hash",
        "stage_id_hashes", "boundary_enabled", "raw_eval_graph", "access")
    core_hashes, solver_ceilings = {}, set()
    for pair in pairs:
        sources = [json.loads((RESULTS / pair[f"{p}_run"] / "results.json").read_text()) for p in ("cat", "uq")]
        for key in protocol_keys:
            if sources[0]["protocol"].get(key) != sources[1]["protocol"].get(key):
                failures.append({"run": pair["uq_run"], "key": key, "reason": "paired protocol differs"})
        paired_protocols += 1
        for data, arm in zip(sources, ("cat_cctp", "uq_v3")):
            p = data["protocol"]
            for name, digest in p["source_hashes"].items():
                if name != "run_unified_q_diagnostics_20261007.py":
                    core_hashes.setdefault(name, set()).add(digest)
            rows = [r for r in data["rows"] if r["arm"] == arm]
            if data["status"] != "complete" or [r["stage"] for r in rows] != list(range(len(p["groups"]))):
                failures.append({"arm": arm, "reason": "missing or duplicated stages"})
            for row in rows:
                checked_stages += 1
                check = row["checks"]
                for key in ("budget_ok", "prefix_ok", "finite_ok"):
                    if check.get(key) is not True:
                        failures.append({"arm": arm, "stage": row["stage"], "key": key})
                if row["memory_rows"] != p["budgets"][row["stage"]]:
                    failures.append({"arm": arm, "reason": "incorrect memory budget"})
                score = row["score"]
                if sorted(r["seed"] for r in score["per_student"]) != sorted(p["student_seeds"]):
                    failures.append({"arm": arm, "reason": "missing Student seeds"})
                for key in ("final_f1", "accuracy", "old_f1", "new_f1"):
                    values = [r[key] for r in score["per_student"] if r.get(key) is not None]
                    if score.get(key) is None:
                        if values:
                            failures.append({"arm": arm, "stage": row["stage"], "key": key, "reason": "missing aggregate"})
                        continue
                    actual = mean(values)
                    if not math.isfinite(actual) or abs(actual - score[key]) > 1e-8:
                        failures.append({"arm": arm, "stage": row["stage"], "key": key, "reason": "mean mismatch"})
                    recomputed_metrics += 1
                if arm == "uq_v3":
                    a = row["audit"]
                    solver_ceilings.add(a["config"]["sinkhorn_max_steps"])
                    if not check.get("sinkhorn_feasible") or check.get("propagation_max_error", 1) > 1e-6:
                        failures.append({"arm": arm, "stage": row["stage"], "reason": "transport or propagation failed"})
                    if a["topology"].get("degree_preserved") is not True:
                        failures.append({"arm": arm, "stage": row["stage"], "reason": "degree invariant failed"})
                    if a["topology"].get("new_old_edges") != 0:
                        failures.append({"arm": arm, "stage": row["stage"], "reason": "unexpected boundary edges"})
    for name, digests in core_hashes.items():
        if len(digests) != 1:
            failures.append({"module": name, "reason": "core code differs across confirmation"})
    return {"paired_protocols": paired_protocols, "checked_stages": checked_stages,
        "recomputed_means": recomputed_metrics, "failures": failures,
        "core_code_hash_counts": {k: len(v) for k, v in core_hashes.items()},
        "solver_ceilings": sorted(solver_ceilings),
        "access_claim_scope": "constructor API and recorded checks; no OS-level historical-read trace",
        "degree_reference": "initial Q-induced top-k graph, not raw graph",
        "all_18_pairs_complete": len(pairs) == 18}


def dump_csv(path, rows):
    if rows:
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def completion_audit(root):
    tasks = {}
    for name in ("diagnostics_plan.json", "followup_finish_plan.json",
                 "followup_confirmation_plan.json", "followup_transfer_plan.json"):
        plan = root / name
        if plan.exists():
            tasks.update({r["tag"]: r for r in json.loads(plan.read_text())["tasks"]})
    for dataset in ("wikics", "bookhis"):
        for budget in ("fixed_rate", "fixed_schedule"):
            tag = f"ab_{dataset}_{budget}"
            tasks[tag] = {"kind": "ab", "tag": tag}
    status = []
    for tag, task in sorted(tasks.items()):
        filename = {"ab": "student_ab.json", "operator": "operator_ab.json"}.get(task["kind"], "results.json")
        completed = None
        for run in (tag, "retry_" + tag, "memretry_" + tag):
            path = root / run / filename
            if not path.exists():
                continue
            record = json.loads(path.read_text())
            # A/B has no status field; its full grid is validated by collect().
            if (task["kind"] == "ab" and len(record.get("rows", [])) == 60) or record.get("status") == "complete":
                completed = run
                break
        status.append({"registered_task": tag, "completed_run": completed,
                       "status": "complete" if completed else "incomplete"})
    repairs = []
    for arm in ("uq_30", "uq_v3"):
        old = root / f"transfer_sportsfit_{arm}/protocol.json"
        new = root / f"memretry_transfer_sportsfit_{arm}/protocol.json"
        if not old.exists() or not new.exists():
            continue
        old_p, new_p = (json.loads(p.read_text()) for p in (old, new))
        differences = [k for k in set(old_p) | set(new_p) if k != "source_hashes" and old_p.get(k) != new_p.get(k)]
        changed_modules = [k for k in old_p["source_hashes"] if old_p["source_hashes"][k] != new_p["source_hashes"].get(k)]
        repairs.append({"arm": arm, "protocol_differences": differences, "changed_modules": changed_modules})
        if differences or changed_modules != ["unified_q_multimodal.py"]:
            raise ValueError("Memory retry changed more than checkpoint implementation")
    return {"registered_tasks": status, "all_complete": all(r["completed_run"] for r in status),
            "memory_retry_protocol_checks": repairs}


def write_summary(reference, pairs, student, numerical):
    lines = [
        "# Unified-Q diagnostics assets",
        "",
        "This folder is generated from the frozen report-dev artifacts. It does not select a method from final-test scores.",
        "",
        f"Completed reference/diagnostic cells: {len(reference)}",
        f"Completed paired confirmation cells: {len(pairs)}",
        f"Student attribution rows: {len(student)}",
        f"Numerical audit rows: {len(numerical)}",
        "",
        "## Paired confirmation",
        "",
        "| Dataset | Order | Construction | CaT F1 | Unified-Q F1 | Delta F1 | CaT Acc | Unified-Q Acc | Delta Acc |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in pairs:
        lines.append(
            f"| {row['dataset']} | {row['order_seed']} | {row['construction_seed']} | "
            f"{row['cat_f1']:.3f} | {row['uq_f1']:.3f} | {row['delta_f1']:+.3f} | "
            f"{row['cat_accuracy']:.3f} | {row['uq_accuracy']:.3f} | {row['delta_accuracy']:+.3f} |"
        )
    lines += ["", "## Aggregate deltas", "", "| Dataset | Cells | Mean Delta F1 | Std Delta F1 | Positive F1 Cells | Mean Delta Accuracy |", "|---|---:|---:|---:|---:|---:|"]
    for dataset in sorted({row["dataset"] for row in pairs}):
        selected = [row for row in pairs if row["dataset"] == dataset]
        f1 = [row["delta_f1"] for row in selected]
        acc = [row["delta_accuracy"] for row in selected]
        lines.append(f"| {dataset} | {len(selected)} | {avg(f1):+.3f} | {sd(f1):.3f} | {sum(v > 0 for v in f1)} | {avg(acc):+.3f} |")
    lines += ["", "## Interpretation guardrails", "", "- Prompt and legacy Student results are attribution controls, not interchangeable benchmark rows.", "- Fixed-rate and fixed-schedule protocols remain separate.", "- A positive BookHis delta and a negative WikiCS delta must be reported as dataset-dependent behavior, not pooled SOTA.", "- Numerical residual plots show solver feasibility; they do not imply a performance gain.", "- SportsFit was completed after an activation-checkpoint memory fix that preserves the same mathematical objective and gradients."]
    methods, uncertainty, orders = aggregate_pairs(pairs)
    lines += ["", "## Confirmation method means", "",
        "SD is across the 9 order x construction cell means, not 27 independent observations.", "",
        "|Dataset|Method|Cells|Final F1|Accuracy|Stage F1|Stage Acc|Old F1|New F1|",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in methods:
        lines.append(f"|{r['dataset']}|{r['arm']}|{r['cells']}|{r['final_f1']:.2f} +/- {r['final_f1_cell_std']:.2f}|"
            f"{r['accuracy']:.2f} +/- {r['accuracy_cell_std']:.2f}|{r['stage_avg_f1']:.2f}|"
            f"{r['stage_avg_accuracy']:.2f}|{r['old_f1']:.2f}|{r['new_f1']:.2f}|")
    lines += ["", "## Order-level uncertainty", "",
        "Descriptive 95% t intervals over only 3 order means, including the development/reference order.",
        "Student runs within a cell are averaged first. New F1 is final-stage new classes, not an unseen-class test.", "",
        "|Dataset|Metric|Delta|95% order t interval|Positive cells|Independent-order delta|",
        "|---|---|---:|---|---:|---:|"]
    for r in uncertainty:
        interval = "pending" if r["order_t_ci_low"] is None else f"[{r['order_t_ci_low']:+.2f}, {r['order_t_ci_high']:+.2f}]"
        independent = "NA" if r["independent_mean_delta"] is None else f"{r['independent_mean_delta']:+.2f}"
        lines.append(f"|{r['dataset']}|{r['metric']}|{r['mean_delta']:+.2f}|{interval}|{r['positive_cells']}/{r['cells']}|{independent}|")
    lines += ["", "## Third-dataset transfer", "", "One order/one construction seed, separate from the 3x3 confirmation.", "",
              "|Method|Final F1|Accuracy|Old F1|New F1|", "|---|---:|---:|---:|---:|"]
    for r in reference:
        if r["dataset"] == "sportsfit":
            lines.append(f"|{r['arm']}|{float(r['final_f1']):.2f}|{float(r['accuracy']):.2f}|{float(r['old_f1']):.2f}|{float(r['new_f1']):.2f}|")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_plots(reference, pairs, student, numerical):
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    plt.rcParams.update({"font.size": 9, "axes.titlesize": 11, "axes.labelsize": 9})

    methods = ["cat_cctp", "uq_30", "uq_64", "uq_128", "uq_proxy384", "uq_v3"]
    labels = {"cat_cctp": "CaT-CCTP", "uq_30": "UQ-30", "uq_64": "UQ-64", "uq_128": "UQ-128", "uq_proxy384": "UQ-384", "uq_v3": "UQ-v3"}
    for metric, ylabel, name in (("final_f1", "Final Macro-F1", "reference_f1.png"), ("accuracy", "Final Accuracy", "reference_accuracy.png")):
        datasets = sorted({row["dataset"] for row in reference})
        fig, axes = plt.subplots(1, max(1, len(datasets)), figsize=(6.8 * max(1, len(datasets)), 4.8), squeeze=False, constrained_layout=True)
        for axis, dataset in zip(axes[0], datasets):
            values, errors, names = [], [], []
            for method in methods:
                rows = [row for row in reference if row["dataset"] == dataset and row["arm"] == method]
                if not rows:
                    continue
                values.append(avg([number(row, metric) for row in rows]))
                errors.append(number(rows[0], "final_f1_std" if metric == "final_f1" else "accuracy_std"))
                names.append(labels[method])
            x = np.arange(len(values))
            axis.bar(x, values, yerr=errors, capsize=3, color=["#356a95" if n == "CaT-CCTP" else "#c17c2f" if n == "UQ-v3" else "#91a8bd" for n in names], edgecolor="#273746")
            axis.set_title(dataset)
            axis.set_ylabel(ylabel)
            axis.set_xticks(x)
            axis.set_xticklabels(names, rotation=35, ha="right")
            axis.grid(axis="y", alpha=0.25)
        fig.suptitle("Reference order/seed; error bars: Student-seed SD; UQ-30/64/128 are Sinkhorn iterations")
        fig.savefig(OUT / name, dpi=180)
        plt.close(fig)

    if pairs:
        means, _, orders = aggregate_pairs(pairs)
        datasets = sorted({r["dataset"] for r in pairs})
        fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.4), constrained_layout=True)
        for col, dataset in enumerate(datasets):
            rows = [next(r for r in means if r["dataset"] == dataset and r["arm"] == a)
                    for a in ("uq_v3", "cat_cctp")]
            for axis, metric, label in ((axes[0, col], "final_f1", "Macro-F1 (%)"), (axes[1, col], "accuracy", "Accuracy (%)")):
                values = [r[metric] for r in rows]
                bars = axis.bar([0, 1], values, yerr=[r[metric + "_cell_std"] for r in rows],
                    capsize=4, color=["#287f78", "#607fa2"])
                axis.bar_label(bars, labels=[f"{v:.2f}" for v in values], padding=8)
                axis.set_xticks([0, 1], ["Unified-Q v3", "CaT-CCTP"])
                axis.set_ylim(0, 100)
                axis.set_title(dataset)
                axis.set_ylabel(label)
                axis.grid(axis="y", alpha=0.2)
        fig.suptitle("Fixed-rate confirmation: 3 orders x 3 construction seeds; cell-mean SD")
        fig.savefig(OUT / "confirmation_main.png", dpi=180)
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
        for axis, dataset in zip(axes, datasets):
            rows = [r for r in orders if r["dataset"] == dataset]
            x = np.arange(len(rows))
            for j, (key, label, color) in enumerate((("final_f1", "Overall", "#607fa2"),
                    ("old_f1", "Old classes", "#287f78"), ("new_f1", "New classes", "#d49546"))):
                axis.bar(x + (j - 1) * .24, [r[f"delta_{key}"] for r in rows], .24, label=label, color=color)
            axis.axhline(0, color="#333333", linewidth=1)
            axis.set_xticks(x, [str(r["order_seed"]) + ("\nreference" if r["reference_order"] else "\nindependent") for r in rows])
            axis.set_ylabel("F1 delta vs CaT (percentage points)")
            axis.set_title(dataset)
            axis.grid(axis="y", alpha=.2)
        axes[0].legend()
        fig.suptitle("Final-stage old/new trade-off: three construction seeds per order")
        fig.savefig(OUT / "order_old_new.png", dpi=180)
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
        for axis, key, ylabel in ((axes[0], "delta_f1", "Unified-Q v3 - CaT F1"), (axes[1], "delta_accuracy", "Unified-Q v3 - CaT Accuracy")):
            datasets = sorted({row["dataset"] for row in pairs})
            for idx, dataset in enumerate(datasets):
                vals = [row[key] for row in pairs if row["dataset"] == dataset]
                x = np.full(len(vals), idx, dtype=float) + np.linspace(-0.08, 0.08, max(1, len(vals)))
                axis.scatter(x, vals, s=34, alpha=0.8, label=dataset)
                if vals:
                    axis.errorbar(idx, avg(vals), yerr=sd(vals), fmt="_", color="black", linewidth=2, capsize=4)
            axis.axhline(0.0, color="#333333", linewidth=1)
            axis.set_xticks(range(len(datasets)))
            axis.set_xticklabels(datasets)
            axis.set_ylabel(ylabel)
            axis.grid(axis="y", alpha=0.25)
        axes[0].legend()
        fig.suptitle("Paired order x construction cells; black bars: cell SD, not confidence intervals")
        fig.savefig(OUT / "paired_delta_confirmation.png", dpi=180)
        plt.close(fig)

    if student:
        selected = [row for row in student if row["arm"] in {"cat_cctp", "uq_semantic"} and row["operator"] == "common"]
        if selected:
            fig, axis = plt.subplots(figsize=(8.4, 4.6), constrained_layout=True)
            groups = sorted({(row["dataset"], row["arm"]) for row in selected})
            x = np.arange(len(groups))
            legacy = [number(next(row for row in selected if (row["dataset"], row["arm"], row["student"]) == (*group, "legacy")), "final_f1") for group in groups]
            prompt = [number(next(row for row in selected if (row["dataset"], row["arm"], row["student"]) == (*group, "prompt")), "final_f1") for group in groups]
            width = 0.36
            axis.bar(x - width / 2, legacy, width, label="Legacy Student", color="#9aa6b2")
            axis.bar(x + width / 2, prompt, width, label="Prompt Student", color="#2f7d9b")
            axis.set_xticks(x)
            axis.set_xticklabels([f"{d}\n{a}" for d, a in groups])
            axis.set_ylabel("Final Macro-F1")
            axis.set_title("Same-memory Student attribution")
            axis.legend()
            axis.grid(axis="y", alpha=0.25)
            fig.savefig(OUT / "student_attribution.png", dpi=180)
            plt.close(fig)

    if numerical:
        max_error = [max(number(row, "max_row_error"), number(row, "max_column_relative_error")) for row in numerical]
        tolerance = [number(row, "tolerance") for row in numerical]
        fig, axis = plt.subplots(figsize=(8.5, 4.5), constrained_layout=True)
        axis.scatter(np.arange(len(max_error)), max_error, s=12, alpha=0.55, label="max residual")
        axis.axhline(1e-4, color="#c0392b", linewidth=1.5, label="strict tolerance 1e-4")
        axis.set_yscale("log")
        axis.set_xlabel("Audited stage")
        axis.set_ylabel("Sinkhorn residual")
        axis.set_title("Transport feasibility audit (fixed-iteration controls may exceed tolerance)")
        axis.legend()
        axis.grid(True, which="both", alpha=0.2)
        fig.savefig(OUT / "sinkhorn_residuals.png", dpi=180)
        plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cells = read_csv("method_cells.csv")
    student = read_csv("operator_attribution.csv")
    numerical = read_csv("numerical_audit.csv")
    reference = completed_reference(cells)
    pairs = paired_cells(cells)
    methods, uncertainty, orders = aggregate_pairs(pairs)
    for name, rows in (("paired_cells", pairs), ("confirmation_methods", methods),
                       ("confirmation_uncertainty", uncertainty), ("confirmation_orders", orders)):
        dump_csv(OUT / f"{name}.csv", rows)
    audit = protocol_audit(pairs)
    (OUT / "confirmation_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    if audit["failures"]:
        raise ValueError(f"Confirmation audit failed: {audit['failures']}")
    completion = completion_audit(RESULTS)
    (OUT / "completion_audit.json").write_text(json.dumps(completion, indent=2) + "\n", encoding="utf-8")
    write_summary(reference, pairs, student, numerical)
    make_plots(reference, pairs, student, numerical)
    print(json.dumps({"reference_cells": len(reference), "paired_cells": len(pairs), "student_rows": len(student), "numerical_rows": len(numerical), "output": str(OUT)}))


if __name__ == "__main__":
    main()
