"""Report-dev-only, cell-paired summary of the Q text coverage experiment."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from summarize_uq_final_evidence_20261012 import ci, export_csv, fmt, tabulate

LABELS = {
    "uq_dm_edgefree": "Unified-Q-EF (Q-top1)",
    "uq_qmedoid": "Unified-Q + Q-medoid",
    "uq_qcoverage": "Unified-Q + Q-coverage",
    "cat_cctp": "CaT-CCTP (adapted)",
    "adaptive_coherent": "Adaptive coherent (internal)",
    "gecc_hard": "GECC-hard (adapted)",
}
METRICS = ("final_f1", "accuracy", "old_f1", "new_f1")
CELL_KEYS = ("dataset", "order_seed", "construction_seed", "arm")


def aggregate_diagnostics(cells, diagnostics, class_diagnostics):
    completed = {tuple(row[k] for k in CELL_KEYS): row for row in cells}
    stages, classes = defaultdict(list), defaultdict(list)
    for row in diagnostics:
        key = tuple(row[k] for k in CELL_KEYS)
        if key in completed and "soft_hard_cosine_gap_mean" in row:
            stages[key].append(row)
    for row in class_diagnostics:
        classes[tuple(row[k] for k in CELL_KEYS)].append(row)
    output = []
    for key, rows in sorted(stages.items()):
        local_classes = classes[key]
        raw_counts = defaultdict(int)
        for row in local_classes:
            raw_counts[row["stage"]] += row["raw_rows"]
        if not local_classes or any(raw_counts[row["stage"]] <= 0 for row in rows):
            raise ValueError("assignment diagnostics lack raw-row denominators")
        total_slots = sum(row["slots"] for row in rows)
        total_raw = sum(raw_counts[row["stage"]] for row in rows)
        item = dict(zip(CELL_KEYS, key))
        item.update(label=LABELS[key[-1]], stages=len(rows), slots=total_slots,
                    raw_rows=total_raw, classes=len(local_classes),
                    unique_text_ratio=completed[key]["unique_text_ratio"])
        for metric in ("hard_slot_coverage", "positive_mass_slot_coverage", "unique_source_ratio",
                       "slot_source_entropy", "soft_hard_cosine_gap_mean", "q_relative_update"):
            item[metric] = sum(row[metric] * row["slots"] for row in rows) / total_slots
        for metric in ("row_entropy", "normalized_row_entropy"):
            item[metric] = sum(row[metric] * raw_counts[row["stage"]] for row in rows) / total_raw
        item["soft_hard_cosine_gap_max"] = max(row["soft_hard_cosine_gap_max"] for row in rows)
        item["class_coverage_cosine"] = float(np.mean([row["class_coverage_cosine"] for row in local_classes]))
        output.append(item)
    return output


def validate_score(score):
    if score["student_seeds"] != [150, 151, 152] or score["student_steps"] != 2048 or score["student_batch"] != 24:
        raise ValueError("different Student seeds or training budget")
    samples = score["per_student"]
    if [row["seed"] for row in samples] != score["student_seeds"]:
        raise ValueError("missing or duplicate Student runs")
    for metric in METRICS:
        values = [row[metric] for row in samples]
        if any(value is None or not math.isfinite(value) for value in values):
            raise ValueError(f"invalid final-stage score: {metric}")
        if not np.isclose(score[metric], np.mean(values), atol=1e-7, rtol=0):
            raise ValueError(f"Student mean mismatch: {metric}")
    for metric in ("final_f1", "accuracy"):
        if not np.isclose(score[metric + "_std"], np.std([r[metric] for r in samples], ddof=1), atol=1e-7, rtol=0):
            raise ValueError(f"Student SD mismatch: {metric}")


def summarize(root):
    cells, student_rows, diagnostics, class_diagnostics, sources, pending, excluded = [], [], [], [], [], [], []
    seen, signatures = set(), {}
    for path in sorted(root.glob("*/results.json")):
        raw = path.read_bytes()
        result = json.loads(raw)
        protocol = result["protocol"]
        if result.get("status") == "smoke_complete" or protocol.get("max_stages", 0) < len(protocol["budgets"]):
            excluded.append(dict(path=str(path), reason="smoke/sub-stage run; not formal evidence"))
            continue
        if protocol["name"] != "uq-q-coverage-confirmation-20261012-v2":
            raise ValueError(f"unexpected protocol in {path}")
        if protocol["method_selection_from_test"] or "report-dev only" not in protocol["evaluation_split"]:
            raise ValueError("coverage experiments must use report-dev only")
        if not np.isclose(protocol["memory_rate"], .05) or not np.isclose(
                protocol["actual_final_rate"], protocol["budgets"][-1] / sum(protocol["stage_raw_rows"])):
            raise ValueError("budget denominator differs")
        if not protocol["source_hashes"].get("unified_q_payload_projection.py"):
            raise ValueError("hard text projection source hash missing")
        common = {key: protocol[key] for key in ("dataset", "order_seed", "construction_seed")}
        cell_key = tuple(common.values())
        signature = {key: protocol[key] for key in (
            "groups", "budgets", "stage_id_hashes", "pool_mask_hash", "report_mask_hash", "test_mask_hash",
            "student", "student_seeds", "student_steps", "student_batch", "q_configs", "source_hashes")}
        if cell_key in signatures and signatures[cell_key] != signature:
            raise ValueError(f"protocol differs within a paired cell: {cell_key}")
        signatures[cell_key] = signature
        sources.append(dict(path=str(path.resolve()), sha256=hashlib.sha256(raw).hexdigest(),
                            status=result["status"], runtime=result.get("runtime"), protocol=protocol))
        last = len(protocol["budgets"]) - 1
        finished = set()
        for row in result["rows"]:
            arm = row["arm"]
            if row["stage"] != last or row["memory_rows"] != protocol["budgets"][-1]:
                raise ValueError(f"incomplete or wrong-budget final row: {path} {arm}")
            key = (*cell_key, arm)
            if key in seen:
                raise ValueError(f"duplicate cell: {key}")
            seen.add(key)
            stages = [s for s in result["stages"] if s["arm"] == arm]
            if sorted(s["stage"] for s in stages) != list(range(last + 1)):
                raise ValueError("final score is missing construction stages")
            score = row["score"]
            validate_score(score)
            finished.add(arm)
            cells.append(dict(**common, arm=arm, label=LABELS[arm], method_class=row["method_class"],
                **{metric: score[metric] for metric in METRICS},
                final_f1_student_std=score["final_f1_std"], accuracy_student_std=score["accuracy_std"],
                construction_seconds=row["construction_seconds"], construction_peak_mb=row.get("construction_peak_mb"),
                unique_text_ratio=row["unique_texts"] / row["memory_rows"], memory_rows=row["memory_rows"],
                actual_rate=protocol["actual_final_rate"], memory_hash=row["memory_hash"]))
            for student in score["per_student"]:
                student_rows.append(dict(**common, arm=arm, **student))
            for stage in stages:
                diagnostic = stage["audit"].get("coverage_diagnostics", {})
                numeric = {k: v for k, v in diagnostic.items() if isinstance(v, (int, float))}
                diagnostics.append(dict(**common, arm=arm, stage=stage["stage"], slots=stage["new_rows"],
                    construction_seconds=stage["construction_seconds_including_reused"],
                    reused_cat_seconds=stage["reused_cat_seconds"], diagnostic_seconds=stage["diagnostic_seconds"],
                    **numeric))
                class_diagnostics.extend(dict(**common, arm=arm, stage=stage["stage"], **item)
                                         for item in diagnostic.get("per_class", []))
        pending.append(dict(**common, status=result["status"], pending=[a for a in protocol["arms"] if a not in finished]))
    grouped = defaultdict(list)
    for row in cells:
        grouped[row["dataset"], row["arm"]].append(row)
    summary = []
    for (dataset, arm), rows in sorted(grouped.items()):
        item = dict(dataset=dataset, arm=arm, label=LABELS[arm], cells=len(rows), method_class=rows[0]["method_class"])
        for metric in METRICS:
            values = [r[metric] for r in rows]
            item[metric] = float(np.mean(values))
            item[metric + "_cell_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        item["final_f1_student_std_mean"] = float(np.mean([r["final_f1_student_std"] for r in rows]))
        for metric in ("construction_seconds", "construction_peak_mb", "unique_text_ratio"):
            values = [r[metric] for r in rows if r[metric] is not None]
            item[metric] = float(np.mean(values)) if values else None
        summary.append(item)
    pairs = []
    by_key = {tuple(r[k] for k in ("dataset", "order_seed", "construction_seed", "arm")): r for r in cells}
    comparisons = [(arm, "uq_dm_edgefree") for arm in LABELS if arm != "uq_dm_edgefree"]
    comparisons += [("uq_qcoverage", arm) for arm in ("uq_qmedoid", "cat_cctp", "adaptive_coherent", "gecc_hard")]
    for dataset in sorted({r["dataset"] for r in cells}):
        for left, right in comparisons:
            matched = [(r, by_key[dataset, r["order_seed"], r["construction_seed"], right])
                       for r in cells if r["dataset"] == dataset and r["arm"] == left
                       and (dataset, r["order_seed"], r["construction_seed"], right) in by_key]
            if not matched:
                continue
            for metric in METRICS:
                differences = [l[metric] - r[metric] for l, r in matched]
                low, high = ci(differences)
                worst = int(np.argmin(differences))
                pairs.append(dict(dataset=dataset, left=left, right=right, metric=metric, cells=len(matched),
                    delta=float(np.mean(differences)), ci_low=low, ci_high=high,
                    positive_cells=sum(v > 0 for v in differences), worst_delta=differences[worst],
                    worst_order=matched[worst][0]["order_seed"], worst_seed=matched[worst][0]["construction_seed"]))
    diagnostic_cells = aggregate_diagnostics(cells, diagnostics, class_diagnostics)
    return dict(cells=cells, student_rows=student_rows, diagnostics=diagnostics,
                diagnostic_cells=diagnostic_cells,
                class_diagnostics=class_diagnostics, summary=summary, pairs=pairs,
                sources=sources, pending=pending, excluded=excluded)


def plots(output, evidence):
    if not evidence["summary"]:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    datasets = sorted({r["dataset"] for r in evidence["summary"]})
    colors = ["#347c58", "#548caf", "#b64d55", "#81749c", "#cb9850", "#6a777d"]
    labels = ["Q-top1", "Q-medoid", "Q-coverage", "CaT-CCTP", "Coherent", "GECC-hard"]
    fig, axes = plt.subplots(2, len(datasets), figsize=(5 * len(datasets), 7), squeeze=False, layout="constrained")
    for column, dataset in enumerate(datasets):
        for index, metric in enumerate(("final_f1", "accuracy")):
            ax = axes[index, column]
            for i, arm in enumerate(LABELS):
                row = next((r for r in evidence["summary"] if r["dataset"] == dataset and r["arm"] == arm), None)
                if row is None:
                    continue
                student_sd = row.get(metric + "_student_std_mean")
                ax.bar(i, row[metric], yerr=student_sd, capsize=2.5, color=colors[i], width=.7,
                       error_kw={"elinewidth": .8, "ecolor": "#333333"})
                ax.text(i, row[metric] + (student_sd or 0) + 1, f"{row[metric]:.2f}", ha="center", fontsize=8)
            ax.set_xticks(range(len(labels)), labels, rotation=25, ha="right")
            ax.set_ylim(0, 104)
            display = {"wikics": "WikiCS", "sportsfit": "SportsFit", "bookhis": "BookHis"}.get(dataset, dataset)
            ax.set_title(display + (" - Macro-F1" if index == 0 else " - Accuracy"))
            ax.set_ylabel("Percent")
            ax.grid(axis="y", color="#dddddd", linewidth=.5)
            ax.set_axisbelow(True)
    fig.suptitle("Matched report-dev comparison\n"
                 "Error bars: mean within-cell SD of three Student seeds", fontsize=11)
    for extension in ("png", "pdf"):
        fig.savefig(output / f"report_dev_comparison.{extension}", dpi=180)
    plt.close(fig)
    if not evidence.get("diagnostic_cells"):
        return
    diagnostic_datasets = sorted({r["dataset"] for r in evidence["diagnostic_cells"]})
    fig, axes = plt.subplots(2, len(diagnostic_datasets), figsize=(4 * len(diagnostic_datasets), 6),
                             squeeze=False, layout="constrained")
    q_arms = list(LABELS)[:3]
    q_colors = ["#347c58", "#548caf", "#b64d55"]
    for column, dataset in enumerate(diagnostic_datasets):
        display = {"wikics": "WikiCS", "sportsfit": "SportsFit", "bookhis": "BookHis"}.get(dataset, dataset)
        for index, (metric, title) in enumerate((("soft_hard_cosine_gap_mean", "Soft/hard gap"),
                                               ("class_coverage_cosine", "Class semantic coverage"))):
            ax = axes[index, column]
            for i, arm in enumerate(q_arms):
                values = [r[metric] for r in evidence["diagnostic_cells"]
                          if r["dataset"] == dataset and r["arm"] == arm]
                if not values:
                    continue
                value = float(np.mean(values))
                ax.bar(i, value, color=q_colors[i], width=.65)
                ax.text(i, value + .02, f"{value:.3f}", ha="center", fontsize=9)
            ax.set_xticks(range(3), ["Q-top1", "Q-medoid", "Q-coverage"])
            ax.set_ylim(0, 1)
            ax.set_ylabel("Cosine gap" if index == 0 else "Mean max cosine")
            ax.set_title(display + " - " + title, fontsize=10)
            ax.grid(axis="y", color="#dddddd", linewidth=.5)
            ax.set_axisbelow(True)
    fig.suptitle("Text projection diagnostics\nGap weighted by slots; semantic coverage averaged by class", fontsize=11)
    for extension in ("png", "pdf"):
        fig.savefig(output / f"projection_diagnostics.{extension}", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-plots", action="store_true",
                        help="Export validated tables on compute hosts; render figures locally.")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("preserve previous report; use a fresh output")
    evidence = summarize(args.root)
    args.output.mkdir(parents=True)
    for name in ("cells", "student_rows", "diagnostics", "class_diagnostics", "diagnostic_cells", "summary", "pairs"):
        export_csv(args.output / (name + ".csv"), evidence[name])
    (args.output / "provenance.json").write_text(json.dumps(
        {k: evidence[k] for k in ("sources", "pending", "excluded")}, indent=2), encoding="utf-8")
    text = ["# Q-conditioned text coverage confirmation", "",
        "Report-dev only. All arms share class order, construction seed, legal cumulative 5% budget, "
        "MiniLM and Prompt GCN training (2048 steps, batch 24, Student seeds 150/151/152). "
        "Sealed tables are not read or changed. Final-stage scores only; stage-average F1 is not measured.", "",
        "Three Student seeds are averaged inside each order/construction cell. Student SD is replay variation; "
        "cell SD and paired 95% t intervals require multiple independent cells. A single-cell difference is exploratory.", "",
        "Q-medoid/Q-coverage change the training hard forward and final projection together: learned Q/X can "
        "also change. This is a projection-rule intervention, not an identical-X posthoc text experiment. "
        "Adaptive coherent is an internal historical candidate. CaT/GECC are mechanism-adapted baselines, "
        "not official-protocol reproductions. Coherent costs include its reused CaT construction.", ""]
    for dataset in ("wikics", "sportsfit", "bookhis"):
        rows = [r for r in evidence["summary"] if r["dataset"] == dataset]
        text += [f"## {dataset}", ""]
        if not rows:
            text += ["Pending: no completed formal arm in this snapshot.", ""]
            continue
        rows.sort(key=lambda r: list(LABELS).index(r["arm"]))
        text += [tabulate(["Method", "Cells", "F1", "Student SD", "Cell SD", "Acc", "Old F1", "New F1", "Construct s", "Peak GiB", "Unique text ratio"],
            [[r["label"], r["cells"], fmt(r["final_f1"]), fmt(r["final_f1_student_std_mean"]),
              fmt(r["final_f1_cell_std"]), fmt(r["accuracy"]), fmt(r["old_f1"]), fmt(r["new_f1"]),
              fmt(r["construction_seconds"], 1), fmt(r["construction_peak_mb"] / 1024 if r["construction_peak_mb"] is not None else None),
              fmt(r["unique_text_ratio"], 3)] for r in rows]), ""]
        pairs = [r for r in evidence["pairs"] if r["dataset"] == dataset and r["left"] == "uq_qcoverage"
                 and r["metric"] in ("final_f1", "accuracy")]
        text += [tabulate(["Q-coverage minus", "Metric", "Cells", "Delta pp", "95% paired CI", "Positive", "Worst delta"],
            [[LABELS[r["right"]], r["metric"], r["cells"], fmt(r["delta"]),
              f"[{fmt(r['ci_low'])}, {fmt(r['ci_high'])}]" if r["ci_low"] is not None else "not estimable (one cell)",
              f"{r['positive_cells']}/{r['cells']}", fmt(r["worst_delta"])] for r in pairs]), ""]
    text += ["## Diagnostics", "",
        "Stage-level slot coverage, normalized row entropy, column source entropy, source/string diversity, "
        "soft/hard cosine gap and class semantic coverage are in diagnostics.csv and class_diagnostics.csv. "
        "CaT/coherent/GECC have no Q, so assignment diagnostics are not invented for these arms. "
        "Peak memory is live CUDA allocated memory; elapsed time was measured on a shared research server."]
    if evidence["diagnostic_cells"]:
        text += ["", "Completed-cell diagnostics aggregate all stages. Slot coverage, source entropy and soft/hard "
                 "gap use slot counts; assignment entropy uses raw-row counts; semantic coverage gives each "
                 "class equal weight. Text diversity uses the final saved memory. Smaller cosine gap and "
                 "larger coverage are diagnostic properties, not causal evidence of better classification.", "",
            tabulate(["Dataset", "Method", "Order", "Construction", "Slots", "Hard slot coverage",
                      "Normalized entropy", "Unique sources", "Unique texts", "Soft/hard gap", "Class coverage"],
                [[r["dataset"], r["label"], r["order_seed"], r["construction_seed"], r["slots"],
                  fmt(r["hard_slot_coverage"], 3), fmt(r["normalized_row_entropy"], 3),
                  fmt(r["unique_source_ratio"], 3), fmt(r["unique_text_ratio"], 3),
                  fmt(r["soft_hard_cosine_gap_mean"], 3), fmt(r["class_coverage_cosine"], 3)]
                 for r in evidence["diagnostic_cells"]])]
    if not args.no_plots:
        text += ["", "![Report-dev comparison](report_dev_comparison.png)"]
        if evidence["diagnostic_cells"]:
            text += ["", "![Text projection diagnostics](projection_diagnostics.png)"]
    (args.output / "summary.md").write_text("\n".join(text), encoding="utf-8")
    if not args.no_plots:
        plots(args.output, evidence)
    print(json.dumps(dict(report=str(args.output), completed_arm_cells=len(evidence["cells"]), pending=evidence["pending"])))


if __name__ == "__main__":
    main()

