"""Build one auditable pre-submission evidence bundle from existing JSON/CSV runs.

The script deliberately keeps incompatible protocols in separate tables.  It
does not choose a method by score and it does not touch sealed-test files.
"""
from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "pre_submission_evidence_20261010"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True), encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def safe_sd(values: list[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


def aggregate(rows: list[dict], group_keys: tuple[str, ...], metrics: tuple[str, ...]) -> list[dict]:
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        buckets[tuple(row.get(key) for key in group_keys)].append(row)
    output = []
    for key, values in sorted(buckets.items(), key=lambda item: tuple(str(v) for v in item[0])):
        item = dict(zip(group_keys, key))
        item["n"] = len(values)
        for metric in metrics:
            numbers = [float(v[metric]) for v in values if v.get(metric) is not None]
            if numbers:
                item[f"{metric}_mean"] = mean(numbers)
                item[f"{metric}_sd"] = safe_sd(numbers)
                item[f"{metric}_min"] = min(numbers)
        output.append(item)
    return output


def load_matched() -> list[dict]:
    payload = read_json(ROOT / "results/student_method_factorial_20261010/final_matched_operator/summary.json")
    rows = []
    for record in payload["records"]:
        rows.append({
            "dataset": record["dataset"],
            "method": record["method"],
            "student": record["student_name"],
            "source": "matched_operator_report_dev",
            "n_orders": record["order_count"],
            "final_f1": record["final_f1_mean"],
            "accuracy": record["accuracy_mean"],
            "old_f1": record["old_f1_mean"],
            "new_f1": record["new_f1_mean"],
            "order_sd_f1": record["final_f1_sd_order"],
        })
    return rows


def load_sage() -> list[dict]:
    root = ROOT / "results/uq_sage_extended_backup_20261009"
    rows = []
    pattern = re.compile(r"(?P<dataset>[^_]+)_o(?P<order>\d+)_c(?P<seed>\d+)$")
    for path in sorted(root.glob("*_o*_c*/results.json")):
        match = pattern.match(path.parent.name)
        if not match:
            continue
        payload = read_json(path)
        for row in payload.get("rows", []):
            score = row.get("score", {})
            rows.append({
                "dataset": match.group("dataset"),
                "order": int(match.group("order")),
                "construction_seed": int(match.group("seed")),
                "method": row["arm"],
                "student": "Prompt SAGE",
                "source": "frozen_memory_sage_transfer_report_dev",
                "final_f1": score.get("final_f1"),
                "accuracy": score.get("accuracy"),
                "old_f1": score.get("old_f1"),
                "new_f1": score.get("new_f1"),
                "student_peak_mb": row.get("student_peak_mb"),
                "student_seconds": row.get("student_seconds"),
            })
    return rows


def load_seed_attribution() -> list[dict]:
    root = ROOT / "results/uq_q_seed_attribution_20261010"
    rows = []
    paths = [path for path in root.glob("*_o*_c*/results.json")
             if "_recovery" not in path.parent.name]
    # The one numerical-recovery cell is kept in a suffixed directory so the
    # original failed run remains immutable and auditable.
    paths += list(root.glob("*_o*_c*_recovery/results.json"))
    for path in sorted(paths):
        payload = read_json(path)
        if payload.get("status") != "complete":
            continue
        protocol = payload["protocol"]
        by_arm: dict[str, list[dict]] = defaultdict(list)
        for row in payload.get("rows", []):
            by_arm[row["arm"]].append(row)
        for arm, arm_rows in by_arm.items():
            final = max(arm_rows, key=lambda row: int(row["stage"]))
            score = final["score"]
            rows.append({
                "dataset": protocol["dataset"],
                "order": protocol["order_seed"],
                "construction_seed": protocol["construction_seed"],
                "method": arm,
                "student": "Prompt GCN",
                "source": "construction_seed_attribution_report_dev",
                "final_f1": score["final_f1"],
                "accuracy": score["accuracy"],
                "old_f1": score.get("old_f1"),
                "new_f1": score.get("new_f1"),
                "recovered": "_recovery" in path.parent.name,
            })
    # A prior interrupted run is kept with an `.incomplete_*` suffix.  Do not
    # count it beside the separately completed recovery cell.
    unique = {}
    for row in rows:
        key = (row["dataset"], row["order"], row["construction_seed"], row["method"])
        if key not in unique or row.get("recovered", False):
            unique[key] = row
    return list(unique.values())


def load_small_supplement() -> list[dict]:
    rows = []
    for dataset, directory in (("cora", "unified_q_compare_20261006_cora_r3"),
                               ("citeseer", "unified_q_compare_20261006_citeseer_r4")):
        path = ROOT / "results" / directory / "summary/final_rows.csv"
        with path.open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                row["source"] = "small_graph_old_topology_enabled_protocol"
                row["dataset"] = dataset
                rows.append(row)
    return rows


def load_ablations() -> list[dict]:
    payload = read_json(ROOT / "results/unified_q_matrix_20261008/summary.json")
    rows = []
    values = payload.get("ablations", [])
    if isinstance(values, dict):
        values = [dict(value, dataset=key) for key, value in values.items()]
    for value in values:
        item = dict(value)
        item["source"] = "single_cell_component_ablation_report_dev"
        rows.append(item)
    return rows


def load_primary_costs() -> list[dict]:
    payload = read_json(ROOT / "results/unified_q_matrix_20261008/summary.json")
    rows = []
    for value in payload.get("external", []):
        rows.append({
            "dataset": value["dataset"],
            "method": value["arm"],
            "orders": value.get("n_orders"),
            "construction_seconds": value.get("construction_seconds"),
            "student_seconds": value.get("student_seconds"),
            "memory_rows": value.get("memory_rows"),
            "final_f1": value.get("final_f1"),
            "accuracy": value.get("accuracy"),
            "source": "unified_q_matrix_external_report_dev",
        })
    return rows


def maybe_plot(matched: list[dict], sage: list[dict], small: list[dict]) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        return
    OUT.mkdir(parents=True, exist_ok=True)

    methods = ["Unified-Q edge-free", "GECC-hard TAG adapter", "CaT-CCTP adapter"]
    labels = {"Unified-Q edge-free": "Unified-Q", "GECC-hard TAG adapter": "GECC",
              "CaT-CCTP adapter": "CaT-CCTP"}
    datasets = ["wikics", "sportsfit", "bookhis"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    for axis, dataset in zip(axes, datasets):
        values = []
        errors = []
        for method in methods:
            found = [row for row in matched if row["dataset"] == dataset and
                     row["student"] == "Prompt Student" and row["method"] == method]
            values.append(found[0]["final_f1"] if found else float("nan"))
            errors.append(found[0]["order_sd_f1"] if found else 0.0)
        axis.bar(np.arange(len(methods)), values, yerr=errors, capsize=3,
                 color=["#2f6f9f", "#8a8f98", "#c58c3a"])
        axis.set_title(dataset)
        axis.set_xticks(np.arange(len(methods)), [labels[m] for m in methods], rotation=25)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Final Macro-F1 (%)")
    fig.suptitle("Matched Prompt Student: primary methods")
    fig.tight_layout()
    fig.savefig(OUT / "primary_prompt_gcn_comparison.png", dpi=180)
    plt.close(fig)

    grouped = aggregate(sage, ("dataset", "method"), ("final_f1", "accuracy"))
    fig, axis = plt.subplots(figsize=(9, 4.5))
    names = sorted({row["dataset"] for row in grouped})
    arms = sorted({row["method"] for row in grouped})
    x = np.arange(len(names))
    width = 0.8 / max(1, len(arms))
    for index, arm in enumerate(arms):
        values = [next((row["final_f1_mean"] for row in grouped
                        if row["dataset"] == dataset and row["method"] == arm), float("nan"))
                  for dataset in names]
        axis.bar(x + index * width, values, width, label=arm)
    axis.set_xticks(x + width * (len(arms) - 1) / 2, names)
    axis.set_ylabel("Final Macro-F1 (%)")
    axis.set_title("Prompt SAGE frozen-memory transfer")
    axis.legend(fontsize=8)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "sage_transfer_comparison.png", dpi=180)
    plt.close(fig)

    if small:
        rows = [row for row in small if row["arm"] in {"cat_x", "unified_q"}]
        fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
        for axis, dataset in zip(axes, ("cora", "citeseer")):
            selected = [row for row in rows if row["dataset"] == dataset]
            arms = ["cat_x", "unified_q"]
            vals = [mean(float(r["final_macro_f1"]) for r in selected if r["arm"] == arm)
                    for arm in arms]
            axis.bar([0, 1], vals, color=["#c58c3a", "#2f6f9f"])
            axis.set_title(dataset)
            axis.set_xticks([0, 1], ["CaT-X", "Unified-Q"])
            axis.grid(axis="y", alpha=0.25)
        axes[0].set_ylabel("Final Macro-F1 (%)")
        fig.suptitle("Supplementary small-graph protocol (old topology-enabled run)")
        fig.tight_layout()
        fig.savefig(OUT / "small_graph_supplement.png", dpi=180)
        plt.close(fig)


def markdown(matched, sage, seed, small, ablations, costs) -> str:
    lines = [
        "# Pre-submission evidence bundle",
        "",
        "This bundle keeps primary matched-operator evidence, frozen-memory SAGE transfer, construction-seed attribution, and old-protocol small-graph evidence separate.",
        "",
        "## Primary matched Prompt/Legacy GCN",
        "",
        "| Dataset | Method | Student | Final F1 | Accuracy | Old F1 | New F1 | Order SD | Orders |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in matched:
        lines.append(f"| {row['dataset']} | {row['method']} | {row['student']} | {row['final_f1']:.2f} | {row['accuracy']:.2f} | {row['old_f1']:.2f} | {row['new_f1']:.2f} | {row['order_sd_f1']:.2f} | {row['n_orders']} |")
    lines += ["", "## Prompt SAGE transfer", "",
              "SAGE uses frozen memories from the registered constructor matrix; it does not re-optimize Unified-Q for SAGE.", "",
              "| Dataset | Arm | Cells | Final F1 mean±SD | Accuracy mean±SD | Worst F1 cell | Peak MB |", "|---|---|---:|---:|---:|---:|---:|"]
    for row in aggregate(sage, ("dataset", "method"), ("final_f1", "accuracy", "student_peak_mb")):
        lines.append(f"| {row['dataset']} | {row['method']} | {row['n']} | {row['final_f1_mean']:.2f}±{row['final_f1_sd']:.2f} | {row['accuracy_mean']:.2f}±{row['accuracy_sd']:.2f} | {row['final_f1_min']:.2f} | {row.get('student_peak_mb_mean', float('nan')):.1f} |")
    lines += ["", "## Construction-seed attribution", "",
              "One cell required a higher Sinkhorn iteration cap because of a numerical tolerance miss. It is included only as a separately marked recovery cell; the original partial output remains preserved.", "",
              "| Dataset | Arm | Complete cells | Final F1 mean±SD | Accuracy mean±SD | Worst cell |", "|---|---|---:|---:|---:|---:|"]
    for row in aggregate(seed, ("dataset", "method"), ("final_f1", "accuracy")):
        lines.append(f"| {row['dataset']} | {row['method']} | {row['n']} | {row['final_f1_mean']:.2f}±{row['final_f1_sd']:.2f} | {row['accuracy_mean']:.2f}±{row['accuracy_sd']:.2f} | {row['final_f1_min']:.2f} |")
    lines += ["", "## Supplementary Cora/Citeseer", "",
              "These results are retained as supplementary evidence only. They use the earlier topology-enabled small-graph protocol and are not pooled with the current edge-free primary table.", "",
              "| Dataset | Arm | Final F1 mean | Accuracy mean | Stage F1 mean | Orders |", "|---|---|---:|---:|---:|---:|"]
    for row in aggregate(small, ("dataset", "arm"), ("final_macro_f1", "final_accuracy", "stage_average_f1")):
        lines.append(f"| {row['dataset']} | {row['arm']} | {row['final_macro_f1_mean']:.2f} | {row['final_accuracy_mean']:.2f} | {row['stage_average_f1_mean']:.2f} | {row['n']} |")
    lines += ["", "## Primary construction cost", "",
              "| Dataset | Arm | Build s | Student s | Memory rows |", "|---|---|---:|---:|---:|"]
    for row in costs:
        lines.append(f"| {row['dataset']} | {row['method']} | {row['construction_seconds']:.1f} | {row['student_seconds']:.1f} | {row['memory_rows']} |")
    lines += ["", "## Component ablation source", "",
              "The full component matrix is preserved in `component_ablations.json` and the detailed source report. It is intentionally not collapsed into the external-baseline ranking.", "",
              "[Detailed component matrix](../unified_q_matrix_20261008/summary.md)", ""]
    lines += ["", "## Interpretation", "",
              "- The primary table supports a dataset- and Student-dependent Unified-Q benefit, not universal SOTA.",
              "- The SAGE table tests constructor transfer without changing the memory; it is a generalization check, not a new tuned result.",
              "- The seed-attribution table is the correct place to judge shared-Q stability. The one numerical recovery cell is marked separately and is not silently merged with the original failed run.",
              "- The Cora/Citeseer rows show that the earlier topology-enabled version is not a clean confirmation of the current edge-free method; they should be reported as supplementary protocol evidence.",
              "- No link-prediction claim is made here because the primary Unified-Q memory is edge-free.",
              ""]
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    matched = load_matched()
    sage = load_sage()
    seed = load_seed_attribution()
    small = load_small_supplement()
    ablations = load_ablations()
    costs = load_primary_costs()
    write_csv(OUT / "primary_matched.csv", matched)
    write_csv(OUT / "sage_transfer_cells.csv", sage)
    write_csv(OUT / "seed_attribution_cells.csv", seed)
    write_csv(OUT / "small_graph_supplement.csv", small)
    write_json(OUT / "component_ablations.json", ablations)
    write_json(OUT / "primary_costs.json", costs)
    write_json(OUT / "counts.json", {
        "primary_matched_rows": len(matched),
        "sage_transfer_cells": len(sage),
        "seed_attribution_rows": len(seed),
        "seed_attribution_complete_cells": len({(r['dataset'], r['order'], r['construction_seed']) for r in seed}),
        "small_graph_rows": len(small),
        "component_ablation_rows": len(ablations),
        "primary_cost_rows": len(costs),
    })
    (OUT / "summary.md").write_text(markdown(matched, sage, seed, small, ablations, costs), encoding="utf-8")
    maybe_plot(matched, sage, small)
    print(json.dumps({"output": str(OUT), "matched": len(matched), "sage": len(sage),
                      "seed": len(seed), "small": len(small), "ablations": len(ablations)}))


if __name__ == "__main__":
    main()
