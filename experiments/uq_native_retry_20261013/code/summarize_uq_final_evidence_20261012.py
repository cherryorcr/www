"""Auditable cell-level summaries of the preregistered final evidence batch."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy.stats import t

LABELS = {
    "uq_dm_edgefree": "Unified-Q-EF", "cat_cctp": "CaT-CCTP", "gecc_hard": "GECC-hard",
    "gecc_soft": "GECC-soft (native FCM)", "tagsam_selector_response": "TAGSAM (native selector adaptation)",
    "stad_native_local": "STAD (native core, local Qwen)",
    "control_shared": "Shared Q", "control_independent": "Independent Q",
    "control_shared_no_residual": "Shared Q, no residual",
    "control_independent_no_residual": "Independent Q, no residual",
    "control_shared_equal": "Shared Q, equal parameter budget",
    "control_independent_equal": "Independent Q, equal parameter budget",
    "control_soft_stored": "Shared Q, soft text stored",
}
METRICS = ("final_f1", "accuracy", "old_f1", "new_f1", "stage_average_f1")
PAIRS = {
    "confirmation": (("uq_dm_edgefree", "cat_cctp"), ("uq_dm_edgefree", "gecc_hard")),
    "controls": (("control_shared", "control_independent"),
                 ("control_shared_no_residual", "control_independent_no_residual"),
                 ("control_shared_equal", "control_independent_equal"),
                 ("control_soft_stored", "control_shared")),
}
ADAPTED_BASELINES = ("gecc_soft", "tagsam_selector_response", "stad_native_local")
CONFIRMATION_DATASETS = ("arxiv", "bookhis", "sportsfit")
CONFIRMATION_ORDERS = (9101, 9102, 9103)
CONFIRMATION_SEEDS = (271006, 271108, 271109)
CONFIRMATION_ARMS = ("uq_dm_edgefree", "cat_cctp", "gecc_hard")
EXPECTED_CONFIRMATION_CELLS = {(order, seed) for order in CONFIRMATION_ORDERS for seed in CONFIRMATION_SEEDS}


def average(values):
    values = [float(value) for value in values if value is not None]
    return float(np.mean(values)) if values else None


def deviation(values):
    values = [float(value) for value in values if value is not None]
    return float(np.std(values, ddof=1)) if len(values) > 1 else None


def ci(values):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return None, None
    half = float(t.ppf(.975, len(values) - 1) * np.std(values, ddof=1) / np.sqrt(len(values)))
    return float(values.mean() - half), float(values.mean() + half)


def export_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value, digits=2):
    return "--" if value is None else f"{value:.{digits}f}"


def tabulate(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |",
                      "| " + " | ".join("---" for _ in headers) + " |"]
                     + ["| " + " | ".join(map(str, row)) + " |" for row in rows])


def comparison_protocol(result):
    protocol = dict(result["protocol"])
    if protocol.get("name") in ("stad-native-current-delta-20261012-v1", "stad-native-current-delta-20261013-v2"):
        # This producer omits display metadata, but records the effective schedule
        # and access audits. Only this inspected producer may use the normalization.
        cumulative = np.cumsum(protocol["stage_raw_rows"])
        expected = [round(float(n) * protocol["memory_rate"]) for n in cumulative]
        if protocol.get("class_floor_adjustments") or protocol["budgets"] != expected:
            raise ValueError("STAD budget metadata requires an explicit schedule audit")
        if protocol.get("access") != "current delta only; own frozen synthetic prefix; no historical raw reads":
            raise ValueError("STAD access differs")
        if protocol.get("method_selection_from_test") is not False:
            raise ValueError("STAD test selection is not excluded")
        for row in result["rows"]:
            audit = row.get("audit", {})
            if (row.get("checks", {}).get("prefix_ok") is not True
                    or audit.get("historical_raw_reads") != 0
                    or audit.get("report_dev_used_for_selection") is not False
                    or audit.get("test_used_for_selection") is not False):
                raise ValueError("STAD access/selection audit missing or failed")
            if protocol["name"] == "stad-native-current-delta-20261013-v2":
                split = audit.get("internal_split", {})
                if (protocol.get("internal_split_policy") != "classwise-10pct-sparse-v1"
                        or split.get("policy") != protocol["internal_split_policy"]
                        or split.get("disjoint") is not True
                        or split.get("external_nodes_used") is not False):
                    raise ValueError("STAD sparse-class split audit missing or failed")
        protocol["name"] = "uq-final-frozen-confirmation-20261012-v1"
        protocol["budget_mode"] = "fixed_rate_current_legal_pool"
        protocol["evaluation_split"] = "report_dev and sealed_test stored separately; protocol frozen before either"
        protocol["access"] = "current-delta-only; own frozen synthetic prefix; zero historical raw reads"
    return protocol


def confirmation_completion(cells, sources):
    registered, scored = {}, defaultdict(set)
    for source in sources:
        if source["path"].split("/")[0] != "confirmation":
            continue
        protocol = source["protocol"]
        for arm in protocol.get("arms", [protocol.get("arm")]):
            key = protocol["dataset"], protocol["order_seed"], protocol["construction_seed"], arm
            registered[key] = source["path"]
    for cell in cells:
        if cell["category"] == "confirmation":
            key = cell["dataset"], cell["order_seed"], cell["construction_seed"], cell["arm"]
            scored[key].add(cell["split"])
    rows = []
    for dataset in CONFIRMATION_DATASETS:
        for arm in CONFIRMATION_ARMS:
            for order in CONFIRMATION_ORDERS:
                for seed in CONFIRMATION_SEEDS:
                    key = dataset, order, seed, arm
                    complete = {"report_dev", "sealed_test"} <= scored[key]
                    rows.append(dict(dataset=dataset, arm=arm, label=LABELS[arm], order_seed=order,
                        construction_seed=seed, status="complete" if complete else
                        ("partial" if key in registered else "not_started"),
                        completed_student_runs=3 if complete else 0, source=registered.get(key, "")))
    return rows


def summarize(snapshot):
    cells, student_rows, diagnostics, sources, failures = [], [], [], [], []
    protocols = {}
    for entry in snapshot["entries"]:
        if entry["errors"]:
            raise ValueError(f"invalid evidence: {entry['path']}: {entry['errors']}")
        result, path = entry["result"], entry["path"]
        protocol = result["protocol"]
        category = path.split("/")[0]
        dataset, order, seed = protocol["dataset"], protocol["order_seed"], protocol["construction_seed"]
        common = dict(category=category, dataset=dataset, order_seed=order, construction_seed=seed)
        comparable = comparison_protocol(result)
        signature = {key: comparable.get(key) for key in
                     ("name", "groups", "budgets", "stage_raw_rows", "memory_rate",
                      "actual_final_rate", "budget_mode", "class_floor_adjustments",
                      "student", "student_seeds", "student_steps", "student_batch",
                      "pool_mask_hash", "report_mask_hash", "test_mask_hash", "stage_id_hashes",
                      "evaluation_split", "access")}
        signature["student_source_hash"] = protocol.get("source_hashes", {}).get(
            "graphtext_prompt_student.py")
        cell_key = dataset, order, seed
        if cell_key in protocols and protocols[cell_key] != signature:
            raise ValueError(f"common protocol mismatch: {cell_key}")
        protocols[cell_key] = signature
        sources.append(dict(path=path, sha256=entry["source_sha256"], bytes=entry["bytes"],
                            complete_arms=entry["complete_arms"], status=result["status"],
                            protocol=protocol))
        for row in result["rows"]:
            for key in ("budget_ok", "prefix_ok", "finite_ok", "sinkhorn_feasible"):
                if row["checks"].get(key) is False:
                    raise ValueError(f"failed {key}: {path}")
            base = dict(**common, arm=row["arm"], stage=row["stage"], memory_rows=row["memory_rows"],
                        new_rows=row["new_rows"], source=path, source_sha256=entry["source_sha256"])
            for split, scores in row["scores"].items():
                for student in scores["per_student"]:
                    student_rows.append(dict(**base, split=split, student_seed=student["seed"],
                        **{key: student.get(key) for key in METRICS if key != "stage_average_f1"}))
            audit = row.get("audit", {})
            diag = audit.get("diagnostics", {})
            if "assignments" in diag:
                trace = audit.get("trace", [])
                record = dict(**base, construction_seconds=row["construction_seconds"],
                    peak_gib=row["construction_peak_mb"] / 1024,
                    assignment_parameters=audit.get("assignment_parameter_count"),
                    residual_parameters=audit.get("residual_parameter_count"),
                    legal_assignment_entries=audit.get("legal_assignment_entries"),
                    equal_parameter_budget=audit.get("equal_parameter_budget"),
                    parameterization=audit.get("parameterization"),
                    q_gradient_norm=average([point.get("q_gradient_norm") for point in trace]),
                    residual_gradient_norm=average([point.get("residual_gradient_norm") for point in trace]))
                record.update({key: diag.get(key) for key in
                    ("unique_source_ratio", "unique_text_ratio", "within_memory_text_cosine",
                     "soft_hard_cosine_gap_mean", "soft_hard_cosine_gap_max", "support_top1_agreement",
                     "raw_row_assignment_l1")})
                for role, values in diag["assignments"].items():
                    record.update({role + "_" + key: value for key, value in values.items()})
                diagnostics.append(record)
        for arm in entry["complete_arms"]:
            stages = sorted([row for row in result["rows"] if row["arm"] == arm], key=lambda row: row["stage"])
            last = stages[-1]
            for split, scores in last["scores"].items():
                cell = dict(**common, arm=arm, split=split, source=path, source_sha256=entry["source_sha256"],
                    memory_rows=last["memory_rows"], actual_rate=protocol["actual_final_rate"],
                    construction_seconds=sum(row["construction_seconds"] for row in stages),
                    construction_peak_gib=max(row["construction_peak_mb"] for row in stages) / 1024,
                    stages=len(stages), student_seeds=",".join(map(str, scores["student_seeds"])))
                cell.update({key: scores[key] for key in METRICS if key != "stage_average_f1"})
                cell["stage_average_f1"] = average([row["scores"][split]["final_f1"] for row in stages])
                cell["student_final_f1_std"] = deviation([row["final_f1"] for row in scores["per_student"]])
                cells.append(cell)
        pending = set(protocol.get("arms", [protocol.get("arm")])) - set(entry["complete_arms"])
        if pending:
            failures.append(dict(path=path, pending=sorted(pending)))
    keys = [(row["category"], row["dataset"], row["order_seed"], row["construction_seed"],
             row["arm"], row["split"]) for row in cells]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate cells in evidence snapshot")
    summary = []
    groups = defaultdict(list)
    for cell in cells:
        groups[(cell["category"], cell["dataset"], cell["split"], cell["arm"])].append(cell)
    for (category, dataset, split, arm), items in groups.items():
        record = dict(category=category, dataset=dataset, split=split, arm=arm, label=LABELS.get(arm, arm),
            cells=len(items), orders=len({row["order_seed"] for row in items}),
            construction_seeds=len({row["construction_seed"] for row in items}),
            student_runs=3 * len(items),
            confirmation_complete=(category == "confirmation" and dataset in CONFIRMATION_DATASETS
                                   and arm in CONFIRMATION_ARMS
                                   and {(row["order_seed"], row["construction_seed"]) for row in items}
                                   == EXPECTED_CONFIRMATION_CELLS),
            memory_rows=items[0]["memory_rows"], actual_rate=items[0]["actual_rate"],
            construction_seconds=average([row["construction_seconds"] for row in items]),
            construction_peak_gib=max(row["construction_peak_gib"] for row in items),
            worst_final_f1=min(row["final_f1"] for row in items),
            mean_within_cell_student_std=average([row["student_final_f1_std"] for row in items]))
        for key in METRICS:
            record[key + "_mean"] = average([row[key] for row in items])
            record[key + "_cell_std"] = deviation([row[key] for row in items])
        summary.append(record)
    pairs = []
    for category, comparisons in PAIRS.items():
        for dataset in sorted({row["dataset"] for row in cells if row["category"] == category}):
            for split in ("report_dev", "sealed_test"):
                selected = [row for row in cells if (row["category"], row["dataset"], row["split"]) ==
                            (category, dataset, split)]
                for left, right in comparisons:
                    indexed = {arm: {(row["order_seed"], row["construction_seed"]): row
                                    for row in selected if row["arm"] == arm} for arm in (left, right)}
                    matched = sorted(set(indexed[left]) & set(indexed[right]))
                    if not matched:
                        continue
                    for metric in METRICS:
                        differences = [(key, indexed[left][key][metric] - indexed[right][key][metric])
                                       for key in matched if indexed[left][key][metric] is not None
                                       and indexed[right][key][metric] is not None]
                        if not differences:
                            continue
                        values = [value for _, value in differences]
                        low, high = ci(values)
                        worst_key, worst_value = min(differences, key=lambda item: item[1])
                        pairs.append(dict(category=category, dataset=dataset, split=split, left=left, right=right,
                            metric=metric, cells=len(values), mean_delta=float(np.mean(values)), ci_low=low,
                            ci_high=high, positive_cells=sum(value > 0 for value in values),
                            positive_ratio=sum(value > 0 for value in values) / len(values),
                            worst_delta=worst_value, worst_order=worst_key[0], worst_construction=worst_key[1],
                            scope="9 paired order/construction cells" if category == "confirmation"
                                  and {key for key, _ in differences} == EXPECTED_CONFIRMATION_CELLS
                                  else "provisional; one order or incomplete construction batch"))
    for dataset in sorted({row["dataset"] for row in cells}):
        for split in ("report_dev", "sealed_test"):
            main = {(row["order_seed"], row["construction_seed"]): row for row in cells
                    if row["category"] == "confirmation" and row["dataset"] == dataset
                    and row["split"] == split and row["arm"] == "uq_dm_edgefree"}
            for baseline in ADAPTED_BASELINES:
                other = {(row["order_seed"], row["construction_seed"]): row for row in cells
                         if row["category"] in ("baselines", "stad") and row["dataset"] == dataset
                         and row["split"] == split and row["arm"] == baseline}
                matched = sorted(set(main) & set(other))
                for metric in METRICS:
                    differences = [(key, main[key][metric] - other[key][metric]) for key in matched
                                   if main[key][metric] is not None and other[key][metric] is not None]
                    if not differences:
                        continue
                    values = [value for _, value in differences]
                    low, high = ci(values)
                    worst_key, worst_value = min(differences, key=lambda item: item[1])
                    pairs.append(dict(category="adapted_baseline", dataset=dataset, split=split,
                        left="uq_dm_edgefree", right=baseline, metric=metric, cells=len(values),
                        mean_delta=float(np.mean(values)), ci_low=low, ci_high=high,
                        positive_cells=sum(value > 0 for value in values),
                        positive_ratio=sum(value > 0 for value in values) / len(values),
                        worst_delta=worst_value, worst_order=worst_key[0],
                        worst_construction=worst_key[1],
                        scope="matched current-delta adaptation; single cell is exploratory"
                              if len(values) == 1 else "matched current-delta adaptation"))
    return dict(cells=cells, student_rows=student_rows, diagnostics=diagnostics, summary=summary,
                pairs=pairs, sources=sources, pending=failures,
                confirmation_completion=confirmation_completion(cells, sources))


def plots(output, evidence):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.dpi": 180})
    estimates = [row for row in evidence["pairs"] if row["category"] == "confirmation"
                 and row["split"] == "sealed_test" and row["metric"] == "final_f1"]
    if estimates:
        fig, ax = plt.subplots(figsize=(8, 3.4), layout="constrained")
        for i, row in enumerate(estimates):
            color = "#2166ac" if row["right"] == "cat_cctp" else "#c07023"
            ax.scatter(row["mean_delta"], i, color=color, zorder=3, s=40)
            if row["ci_low"] is not None:
                ax.hlines(i, row["ci_low"], row["ci_high"], color=color, linewidth=2)
            ax.text(row["ci_high"] if row["ci_high"] is not None else row["mean_delta"], i + .12,
                    f"  {row['positive_cells']}/{row['cells']} positive", fontsize=9)
        ax.axvline(0, color="#555555", linestyle="--", linewidth=1)
        ax.set_yticks(range(len(estimates)), [f"{row['dataset'].title()}: EF - {LABELS[row['right']]}" for row in estimates])
        ax.set_xlabel("Final Macro-F1 difference (percentage points); paired 95% t CI")
        count = min(row["cells"] for row in estimates)
        ax.set_title(f"Final confirmation: 5% legal-pool budget, sealed test (at least {count}/9 cells)")
        ax.margins(x=.3, y=.5)
        for suffix in ("png", "pdf"):
            fig.savefig(output / f"confirmation_paired.{suffix}")
        plt.close(fig)
    controls = [row for row in evidence["summary"] if row["category"] == "controls" and row["split"] == "sealed_test"]
    datasets = sorted({row["dataset"] for row in controls})
    if datasets:
        order = [arm for arm in LABELS if arm.startswith("control_")]
        fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 4.5), squeeze=False,
                                 layout="constrained")
        for ax, dataset in zip(axes[0], datasets):
            rows = [next((row for row in controls if row["dataset"] == dataset and row["arm"] == arm), None)
                    for arm in order]
            rows = [row for row in rows if row is not None]
            for i, row in enumerate(rows):
                color = "#2166ac" if "independent" not in row["arm"] else "#c07023"
                ax.scatter(row["final_f1_mean"], i, color=color, s=35,
                           marker="o" if "independent" not in row["arm"] else "s")
                if row["final_f1_cell_std"] is not None:
                    ax.errorbar(row["final_f1_mean"], i, xerr=row["final_f1_cell_std"], color=color, capsize=3)
                ax.text(row["final_f1_mean"] + .08, i + .2,
                        f"{row['final_f1_mean']:.2f} (n={row['cells']})", fontsize=8)
            labels = [row["label"].replace(", ", "\n") for row in rows]
            ax.set_yticks(range(len(rows)), labels)
            ax.invert_yaxis()
            ax.set_title(dataset.title())
            ax.set_xlabel("Final Macro-F1 (%)")
            ax.grid(axis="x", color="#dddddd", linewidth=.5)
            ax.margins(x=.3, y=.08)
        fig.suptitle("Controlled Q attribution: sealed test; error bars are cell SD, not CI")
        for suffix in ("png", "pdf"):
            fig.savefig(output / f"controlled_q.{suffix}")
        plt.close(fig)
    matched = [row for row in evidence["pairs"] if row["category"] == "adapted_baseline"
               and row["split"] == "sealed_test" and row["metric"] == "final_f1"]
    if matched:
        methods = [arm for arm in ADAPTED_BASELINES if any(row["right"] == arm for row in matched)]
        names = sorted({row["dataset"] for row in matched})
        fig, axes = plt.subplots(1, len(names), figsize=(4.6 * len(names), 3.8), squeeze=False,
                                 layout="constrained", sharex=True)
        for ax, dataset in zip(axes[0], names):
            ax.axvline(0, color="#555555", linestyle="--", linewidth=1)
            for i, arm in enumerate(methods):
                row = next((item for item in matched if item["dataset"] == dataset
                            and item["right"] == arm), None)
                if row is None:
                    continue
                ax.scatter(row["mean_delta"], i, s=42, color="#2166ac")
                if row["ci_low"] is not None:
                    ax.hlines(i, row["ci_low"], row["ci_high"], color="#2166ac", linewidth=2)
                ax.annotate(f"n={row['cells']}", (row["mean_delta"], i), xytext=(5, 5),
                            textcoords="offset points", fontsize=8)
            ax.set_yticks(range(len(methods)), [LABELS[arm] for arm in methods])
            ax.set_ylim(len(methods) - .5, -.5)
            ax.set_title(dataset.title())
            ax.grid(axis="x", color="#dddddd", linewidth=.5)
            ax.set_xlabel("EF - adapted baseline (F1 pp)")
        fig.suptitle("Matched sealed-test cells; n=1 points have no uncertainty interval")
        for suffix in ("png", "pdf"):
            fig.savefig(output / f"adapted_baselines.{suffix}")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("use a fresh report directory; preserve prior evidence")
    raw = args.snapshot.read_bytes()
    snapshot = json.loads(raw)
    if snapshot["collection_issues"]:
        raise ValueError(snapshot["collection_issues"])
    evidence = summarize(snapshot)
    args.output.mkdir(parents=True)
    for name in ("cells", "student_rows", "diagnostics", "summary", "pairs", "confirmation_completion"):
        export_csv(args.output / (name + ".csv"), evidence[name])
    manifest = dict(snapshot=str(args.snapshot.resolve()), sha256=hashlib.sha256(raw).hexdigest(),
                    captured_at=snapshot["captured_at"], sources=evidence["sources"],
                    pending=evidence["pending"], audits=snapshot["audits"])
    (args.output / "provenance.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    paragraphs = ["# Unified-Q-EF final evidence", "",
        f"Snapshot captured: {snapshot['captured_at']}. Source SHA-256: `{manifest['sha256']}`.", "",
        "Student seeds 150/151/152 are averaged within each order/construction cell. Cell SD describes "
        "variation of these means; paired 95% t intervals use matched cells, not 27 independent Student runs. "
        "Each dataset confirmation requires all nine cells. Controls have one class order and remain provisional. "
        "No results are used to select methods or parameters.", "",
        "Construction times exclude data loading/shared text-cache preparation. Peak memory is CUDA "
        "allocated memory during construction, not total system RAM. All external methods are explicitly "
        "adapted baselines; none is labeled an original-protocol official reproduction.", ""]
    completion_rows = []
    for dataset in CONFIRMATION_DATASETS:
        for arm in CONFIRMATION_ARMS:
            items = [row for row in evidence["confirmation_completion"]
                     if (row["dataset"], row["arm"]) == (dataset, arm)]
            counts = {status: sum(row["status"] == status for row in items)
                      for status in ("complete", "partial", "not_started")}
            completion_rows.append([dataset, LABELS[arm], counts["complete"], counts["partial"],
                                    counts["not_started"], sum(row["completed_student_runs"] for row in items)])
    paragraphs += ["## Confirmation completion", "",
        "Each dataset/method requires nine order/construction cells with three Student seeds per cell. "
        "The grid includes planned cells without a result file; partial-file counts alone cannot measure "
        "completion. A cell is complete only when both report-dev and sealed-test trajectories are present. "
        "Detailed identities are in confirmation_completion.csv.", "",
        tabulate(["Dataset", "Method", "Complete / 9", "Partial", "Not started", "Completed Student runs / 27"],
                 completion_rows), ""]
    for split in ("report_dev", "sealed_test"):
        paragraphs += [f"## {split}", ""]
        rows = sorted([row for row in evidence["summary"] if row["split"] == split],
                      key=lambda row: (row["category"], row["dataset"], list(LABELS).index(row["arm"]) if row["arm"] in LABELS else 999))
        paragraphs += [tabulate(["Batch", "Dataset", "Method", "Cells", "Final F1", "Cell SD", "Acc",
                                "Old F1", "New F1", "Stage F1", "Construct s", "Peak GiB"],
            [[row["category"], row["dataset"], row["label"], row["cells"], fmt(row["final_f1_mean"]),
              fmt(row["final_f1_cell_std"]), fmt(row["accuracy_mean"]), fmt(row["old_f1_mean"]),
              fmt(row["new_f1_mean"]), fmt(row["stage_average_f1_mean"]),
              fmt(row["construction_seconds"], 1), fmt(row["construction_peak_gib"])] for row in rows]), ""]
        pairs = [row for row in evidence["pairs"] if row["split"] == split and row["metric"] == "final_f1"]
        paragraphs += [tabulate(["Dataset", "Comparison (left - right)", "Cells", "Delta F1", "95% paired CI",
                                "Positive", "Worst delta", "Worst cell"],
            [[row["dataset"], LABELS[row["left"]] + " - " + LABELS[row["right"]], row["cells"],
              fmt(row["mean_delta"]), f"[{fmt(row['ci_low'])}, {fmt(row['ci_high'])}]",
              f"{row['positive_cells']}/{row['cells']}", fmt(row["worst_delta"]),
              f"o{row['worst_order']}/c{row['worst_construction']}"] for row in pairs]), ""]
        baseline_pairs = [row for row in evidence["pairs"] if row["category"] == "adapted_baseline"
                          and row["split"] == split and row["metric"] in ("final_f1", "accuracy")]
        if baseline_pairs:
            paragraphs += ["### Matched adapted baselines", "",
                "These are mechanism-adapted runs under the common Student, not official-protocol "
                "reproductions. One matched order/construction cell does not establish statistical "
                "superiority; no confidence interval is reported for it.", "",
                tabulate(["Dataset", "EF - adapted baseline", "Metric", "Cells", "Delta (pp)",
                          "95% paired CI", "Positive", "Worst delta"],
                    [[row["dataset"], LABELS[row["right"]], row["metric"], row["cells"],
                      fmt(row["mean_delta"]), f"[{fmt(row['ci_low'])}, {fmt(row['ci_high'])}]",
                      f"{row['positive_cells']}/{row['cells']}", fmt(row["worst_delta"])]
                     for row in baseline_pairs]), ""]
    if any(row["category"] == "adapted_baseline" and row["metric"] == "final_f1"
           for row in evidence["pairs"]):
        paragraphs += ["![Matched adapted baselines](adapted_baselines.png)", ""]
    paragraphs += ["## Diagnostics and limits", "",
        "`diagnostics.csv` contains stage-level slot coverage, normalized assignment entropy, source/text "
        "diversity, soft/hard cosine gap, feature/text support agreement, actual parameter counts, gradients "
        "and construction costs. Shared-Q support agreement is one by construction; this is a consistency "
        "property, not evidence of an accuracy gain. Across-class text cosine uses only the first 512 slots. "
        "Equal-budget controls change the update subspace and therefore cannot isolate parameter count "
        "perfectly. Soft-stored embeddings cannot be reconstructed exactly from the reference strings.", "",
        f"Pending trajectories in this snapshot: {sum(len(row['pending']) for row in evidence['pending'])}.", "",
        "![Final confirmation paired comparison](confirmation_paired.png)", "",
        "![Controlled Q comparison](controlled_q.png)", ""]
    (args.output / "summary.md").write_text("\n".join(paragraphs), encoding="utf-8")
    plots(args.output, evidence)
    print(json.dumps(dict(report=str(args.output), complete_cells=len(evidence["cells"]) // 2,
                         student_stage_rows=len(evidence["student_rows"]), pending=manifest["pending"])))


if __name__ == "__main__":
    main()
