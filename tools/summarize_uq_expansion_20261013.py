"""Separate matched-cell expansion from nine-cell confirmation and references."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import t

from collect_uq_expansion_20261013 import canonical_sha


DATASETS = ("arxiv", "sportsfit", "bookhis", "cora", "citeseer")
DISPLAY = dict(arxiv="ArXiv", sportsfit="SportsFit", bookhis="BookHis",
               cora="Cora", citeseer="Citeseer")
METHODS = {
    "uq_dm_edgefree": ("Unified-Q-EF", "proposed", "Shared Q, in-loop DM/GM, hard real text; edge-free"),
    "cat_cctp": ("CaT-CCTP", "mechanism adaptation", "CaT-style DM on current delta; common nearest-5 text"),
    "gecc_hard": ("GECC-hard", "mechanism adaptation", "Classwise hard clustering of propagated features; common nearest-5 text"),
    "random_coreset": ("Random", "selection control", "Class-stratified real rows with their own text"),
    "kcenter_coreset": ("K-center", "selection control", "Exact greedy FPS in normalized X/text joint space"),
    "cluster_medoid": ("Cluster-medoid", "selection control", "MiniBatchKMeans followed by nearest real cluster member; not exact k-medoids"),
    "herding_coreset": ("Herding", "selection control", "Greedy class-mean herding in normalized X/text joint space"),
    "puma_tag": ("PUMA-inspired", "inspired reference", "Pseudo-teacher adapter; NOT native PUMA reproduction"),
    "gcond_x_core": ("GCond-x (core)", "native-core adaptation", "Pinned SGC/gradient distance; 1000 outer steps, no inner training; nearest-5 text"),
    "sfgc_core": ("SFGC (bounded core)", "native-core adaptation", "Pinned GCN/trajectory matching; 5 experts x 800 epochs, 200 x 200 unroll; nearest-5 text"),
    "gecc_soft": ("GECC-soft (native FCM)", "native-core adaptation", "Native fuzzy C-means branch; common feature-nearest text attachment"),
    "tagsam_selector_response": ("TAGSAM (bounded core)", "budget-limited reference", "Native selector/response mechanism; 100 x 10 instead of 5000 x 15; 320-char text"),
    "stad_native_local": ("STAD (local Qwen)", "different-supervision reference", "10% labels/10% internal-val/80% pseudo; local Qwen2.5-1.5B; native topology"),
}
CORE = ("uq_dm_edgefree", "cat_cctp", "gecc_hard")
SCOPE = tuple(METHODS)[:10]
QUALIFIED_REFERENCES = ("tagsam_selector_response", "stad_native_local")
REFERENCE_REASON = {
    "tagsam_selector_response": "Reduced native distillation budget: 100 x 10 rather than 5000 x 15",
    "stad_native_local": "Different label access, local Qwen2.5-1.5B, and stored synthetic topology",
}
COMMON_SOURCES = ("graphtext_prompt_student.py", "run_unified_q_large_smoke_20261007.py",
                  "external_continual_mechanisms.py", "run_unified_q_revision_20261007.py")
METRICS = ("final_f1", "accuracy", "old_f1", "new_f1", "stage_average_f1")
SIGNATURE = ("groups", "budgets", "stage_raw_rows", "memory_rate", "actual_final_rate",
             "student", "student_seeds", "student_steps", "student_batch", "pool_mask_hash",
             "report_mask_hash", "test_mask_hash", "stage_id_hashes")


def number(value):
    return "--" if value is None else f"{value:.2f}"


def mean(values):
    valid = [value for value in values if value is not None]
    return float(np.mean(valid)) if valid else None


def sd(values):
    valid = [value for value in values if value is not None]
    return float(np.std(valid, ddof=1)) if len(valid) > 1 else None


def markdown(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |"] +
        ["| " + " | ".join(map(str, row)) + " |" for row in rows])


def export_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)


def validate_scores(score, seeds):
    students = score["per_student"]
    if sorted(row["seed"] for row in students) != sorted(seeds):
        raise ValueError("Student seed coverage differs")
    for key in METRICS[:-1]:
        values = [row[key] for row in students if row.get(key) is not None]
        if values and (not np.isfinite(values).all() or min(values) < 0 or max(values) > 100):
            raise ValueError(f"invalid percentage: {key}")
        if values and not np.isclose(np.mean(values), score[key], atol=1e-8):
            raise ValueError(f"stored aggregate differs from Student scores: {key}")
        if len(values) > 1 and not np.isclose(np.std(values, ddof=1), score[key+"_std"], atol=1e-8):
            raise ValueError(f"stored std differs: {key}")


def native_stage_diagnostic(row, protocol, path, digest, result_status):
    audit = row.get("audit", {})
    trace = audit.get("trace", [])
    gradients = [item["feature_gradient_norm"] for item in trace]
    objective_key = ("gradient_distance" if row["arm"] == "gcond_x_core"
                     else "normalized_parameter_distance")
    objectives = [item[objective_key] for item in trace]
    expert_losses = [item["training_loss"] for item in audit.get("expert_audit", [])]
    update = audit.get("feature_update_norm")
    values = gradients + objectives + expert_losses + ([] if update is None else [update])
    if not np.isfinite(values).all():
        raise ValueError(f"nonfinite native diagnostic: {path}, stage {row['stage']}")
    zero_feedback = bool(gradients and max(gradients) == 0 and update == 0)
    return dict(dataset=protocol["dataset"], arm=row["arm"], stage=row["stage"],
        stage_class_count=len(protocol["groups"][row["stage"]]),
        result_status=result_status, feature_update_norm=update,
        gradient_samples=len(gradients), zero_gradient_samples=sum(v == 0 for v in gradients),
        gradient_min=min(gradients) if gradients else None,
        gradient_max=max(gradients) if gradients else None,
        objective_first=objectives[0] if objectives else None,
        objective_last=objectives[-1] if objectives else None,
        expert_loss_min=min(expert_losses) if expert_losses else None,
        expert_loss_max=max(expert_losses) if expert_losses else None,
        zero_feature_feedback=zero_feedback, source=path, source_sha256=digest)


def summarize(snapshot):
    if snapshot.get("errors"):
        raise ValueError("snapshot has unreadable evidence")
    signatures, cells, stages, sources, pending = {}, [], [], [], []
    native_diagnostics = []
    implementation_hashes = defaultdict(set)
    for entry in snapshot["entries"]:
        result, path = entry["result"], entry["path"]
        if canonical_sha(result) != entry["canonical_sha256"]:
            raise ValueError(f"snapshot integrity failed: {path}")
        protocol = result["protocol"]
        dataset = protocol["dataset"]
        if dataset not in DATASETS:
            continue
        order, seed = protocol["order_seed"], protocol["construction_seed"]
        if protocol["method_selection_from_test"] is not False:
            raise ValueError(f"test selection not excluded: {path}")
        if protocol["student_steps"] != 2048 or protocol["student_batch"] != 24:
            raise ValueError("Student budget differs")
        if protocol["student_seeds"] != [150, 151, 152]:
            raise ValueError("Student seeds differ")
        if not np.isclose(protocol["memory_rate"], .05):
            raise ValueError("memory ratio differs")
        expected = [round(float(n)*.05) for n in np.cumsum(protocol["stage_raw_rows"])]
        if not protocol.get("class_floor_adjustments") and protocol["budgets"] != expected:
            raise ValueError("fixed-rate schedule differs")
        if not np.isclose(protocol["actual_final_rate"], protocol["budgets"][-1]/sum(protocol["stage_raw_rows"]), atol=1e-12):
            raise ValueError("actual memory ratio differs")
        signature = {key: protocol[key] for key in SIGNATURE}
        key = dataset, order, seed
        if key in signatures and signature != signatures[key]:
            raise ValueError(f"cross-method protocol differs: {path}")
        signatures[key] = signature
        category = "confirmation" if "/confirmation/" in path else "expansion"
        if "/baselines/" in path or "/stad/" in path:
            category = "native_reference"
        arms = protocol.get("arms", [protocol.get("arm")])
        for name in COMMON_SOURCES:
            digest = protocol.get("source_hashes", {}).get(name)
            if digest:
                implementation_hashes[name].add(digest)
        manifest = entry.get("runtime_manifest")
        if manifest:
            for field in ("dataset", "order_seed", "construction_seed"):
                if manifest[field] != protocol[field]:
                    raise ValueError(f"runtime manifest identity differs: {path}")
            if manifest.get("method_selection_from_test") is not False:
                raise ValueError("runtime manifest permits test selection")
            if manifest.get("common_student_steps") != 2048 or manifest.get("common_student_batch") != 24:
                raise ValueError("runtime Student budget differs")
            student_files = [digest for name, digest in manifest["source_files"].items()
                             if Path(name).name == "graphtext_prompt_student.py"]
            if student_files != [protocol["source_hashes"]["graphtext_prompt_student.py"]]:
                raise ValueError("runtime Student source differs from declared source")
        sources.append(dict(path=path, source_sha256=entry["source_sha256"],
            canonical_sha256=entry["canonical_sha256"], status=result["status"],
            bytes=entry["bytes"], category=category, dataset=dataset,
            order_seed=order, construction_seed=seed, runner_hash=protocol["runner_hash"]))
        for arm in arms:
            if arm not in METHODS:
                continue
            rows = sorted((row for row in result["rows"] if row["arm"] == arm), key=lambda row: row["stage"])
            if arm in ("gcond_x_core", "sfgc_core"):
                native_diagnostics.extend(native_stage_diagnostic(row, protocol, path,
                    entry["source_sha256"], result["status"]) for row in rows)
            complete = (result["status"] == "complete" and
                        [row["stage"] for row in rows] == list(range(len(protocol["groups"]))))
            if result["status"] == "complete" and not complete:
                raise ValueError("complete result is missing a stage")
            if not complete:
                pending.append(dict(dataset=dataset, arm=arm, source=path,
                                    completed_stages=len(rows), status=result["status"]))
                continue
            student_hash = protocol.get("source_hashes", {}).get("graphtext_prompt_student.py")
            if category != "native_reference" and not student_hash:
                raise ValueError("common Student source hash missing")
            common = dict(dataset=dataset, arm=arm, label=METHODS[arm][0], identity=METHODS[arm][1],
                category=category, order_seed=order, construction_seed=seed, source=path,
                source_sha256=entry["source_sha256"], student_source_hash=student_hash,
                memory_rows=protocol["budgets"][-1], actual_rate=protocol["actual_final_rate"])
            for row in rows:
                if row["memory_rows"] != protocol["budgets"][row["stage"]]:
                    raise ValueError("row memory budget differs")
                for check in ("budget_ok", "prefix_ok", "finite_ok"):
                    if row["checks"].get(check) is not True:
                        raise ValueError(f"memory check failed: {check}")
                if row["checks"].get("sinkhorn_feasible") is False:
                    raise ValueError("infeasible assignment")
                for split in ("report_dev", "sealed_test"):
                    score = row["scores"][split]
                    validate_scores(score, protocol["student_seeds"])
                    stages.append(dict(**common, split=split, stage=row["stage"],
                        final_f1=score["final_f1"], accuracy=score["accuracy"],
                        old_f1=score["old_f1"], new_f1=score["new_f1"]))
            for split in ("report_dev", "sealed_test"):
                score = rows[-1]["scores"][split]
                cells.append(dict(**common, split=split, **{key: score[key] for key in METRICS[:-1]},
                    f1_student_std=score["final_f1_std"], acc_student_std=score["accuracy_std"],
                    stage_average_f1=mean([row["scores"][split]["final_f1"] for row in rows]),
                    construction_seconds=sum(row["construction_seconds"] for row in rows),
                    construction_peak_gib=max(row["construction_peak_mb"] for row in rows)/1024,
                    student_seconds=sum(row["student_seconds"] for row in rows),
                    eval_rows=score["eval_rows"], per_student=score["per_student"]))
    hashes = {row["student_source_hash"] for row in cells if row["student_source_hash"]}
    if len(hashes) != 1:
        raise ValueError("Student implementation hashes differ")
    if any(len(values) != 1 for values in implementation_hashes.values()):
        raise ValueError("common construction/data/evaluation sources differ")
    keys = [(row["dataset"], row["order_seed"], row["construction_seed"], row["arm"], row["split"]) for row in cells]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate completed experimental cells; do not select by score")
    matched = [row for row in cells if row["order_seed"] == 9101 and row["construction_seed"] == 271006]
    stability, paired = [], []
    for dataset in DATASETS[:3]:
        for split in ("report_dev", "sealed_test"):
            indexed = {}
            for arm in CORE:
                selected = [row for row in cells if row["dataset"] == dataset and row["arm"] == arm and row["split"] == split and row["category"] == "confirmation"]
                indexed[arm] = {(row["order_seed"], row["construction_seed"]): row for row in selected}
                expected = {(order, seed) for order in (9101, 9102, 9103) for seed in (271006, 271108, 271109)}
                if set(indexed[arm]) != expected:
                    raise ValueError(f"nine-cell confirmation incomplete: {dataset}/{arm}")
                stability.append(dict(dataset=dataset, split=split, arm=arm, label=METHODS[arm][0],
                    cells=len(selected), student_runs=3*len(selected),
                    **{key+"_mean": mean([row[key] for row in selected]) for key in METRICS},
                    **{key+"_cell_std": sd([row[key] for row in selected]) for key in METRICS},
                    construction_seconds=mean([row["construction_seconds"] for row in selected]),
                    construction_peak_gib=max(row["construction_peak_gib"] for row in selected),
                    worst_f1=min(row["final_f1"] for row in selected)))
            for comparator in CORE[1:]:
                for metric in METRICS:
                    differences = [(key, indexed[CORE[0]][key][metric]-indexed[comparator][key][metric]) for key in sorted(indexed[CORE[0]])]
                    values = np.array([value for _, value in differences])
                    half = float(t.ppf(.975, len(values)-1)*values.std(ddof=1)/np.sqrt(len(values)))
                    worst = min(differences, key=lambda item: item[1])
                    paired.append(dict(dataset=dataset, split=split, comparator=comparator, metric=metric,
                        cells=len(values), mean_delta=float(values.mean()),
                        ci_low=float(values.mean()-half), ci_high=float(values.mean()+half),
                        positive_cells=int((values > 0).sum()), positive_ratio=float((values > 0).mean()),
                        worst_delta=worst[1], worst_order=worst[0][0], worst_construction=worst[0][1]))
    coverage = []
    process_text = snapshot["processes"]["stdout"]
    statuses = "\n".join(snapshot.get("statuses", {}).values())
    for dataset in DATASETS:
        for arm in METHODS:
            if dataset in DATASETS[3:] and arm not in SCOPE:
                continue
            completed = next((row for row in matched if row["dataset"] == dataset and row["arm"] == arm and row["split"] == "sealed_test"), None)
            candidates = [row for row in pending if row["dataset"] == dataset and row["arm"] == arm]
            active = any("--dataset " + dataset in line and ("--arms " + arm in line or "--arm " + arm in line) for line in process_text.splitlines())
            failed = any(Path(row["source"]).parent.name + " exit=" in statuses and
                Path(row["source"]).parent.name + " exit=0" not in statuses for row in candidates)
            status = "complete" if completed else "running" if active else "failed / queued retry" if failed else "queued"
            coverage.append(dict(dataset=dataset, arm=arm, label=METHODS[arm][0], identity=METHODS[arm][1],
                status=status, source=completed["source"] if completed else "",
                completed_stages=max([row["completed_stages"] for row in candidates], default=0)))
    return dict(collected_at=snapshot["collected_at"], matched=matched, stability=stability,
        paired=paired, coverage=coverage, stages=stages, sources=sources,
        native_diagnostics=native_diagnostics,
        checks=dict(snapshot_sources=len(sources), completed_unique_cells=len(cells)//2,
                    matched_complete_cells=len(matched)//2, protocol_groups=len(signatures),
                    student_source_hash=next(iter(hashes)), no_duplicate_cells=True,
                    stored_student_means_and_stds_verified=True,
                    common_source_hashes={name: next(iter(values))
                                          for name, values in implementation_hashes.items()},
                    qualified_references_separated=True,
                    native_diagnostic_stages=len(native_diagnostics),
                    native_zero_feature_feedback_stages=sum(
                        row["zero_feature_feedback"] for row in native_diagnostics)))


def write_report(bundle, output):
    lines = ["# Unified-Q-EF: Five-dataset Baseline Expansion", "",
        "Snapshot: " + bundle["collected_at"], "",
        "## Reading the Tables", "",
        "The expanded panel uses order 9101, construction seed 271006, and Student seeds 150/151/152. "
        "All completed stages are required. Values are percentages; +/- denotes Student-seed SD, NOT a construction-seed CI.", "",
        "The protocol uses cumulative 5% of the legal construction pool (integer rounding), current-delta-only access, "
        "frozen synthetic prefixes, MiniLM, and the same prompt GCN (2048 steps, batch 24). "
        "Cora/Citeseer use the project's canonical split, not Planetoid's 20-label-per-class benchmark. "
        "Inference uses the shared raw transductive graph; stored memories are edge-free except STAD's separate reference.", "",
        "These are adaptations, not official paper-score reproductions. Construction budgets differ by method. "
        "SFGC uses 5 experts/200 outer steps rather than the official example's 200/2000. "
        "TAGSAM uses a reduced 100 x 10 distillation budget. STAD changes label access and its LLM. "
        "The latter two belong to a separately qualified reference panel.", ""]
    for dataset in DATASETS:
        lines.extend(["## " + DISPLAY[dataset], ""])
        rows = []
        for arm in METHODS:
            if arm in QUALIFIED_REFERENCES:
                continue
            cell = next((row for row in bundle["matched"] if row["dataset"] == dataset and row["arm"] == arm and row["split"] == "sealed_test"), None)
            if not cell:
                continue
            rows.append([cell["label"], cell["identity"], f"{cell['final_f1']:.2f} +/- {cell['f1_student_std']:.2f}",
                f"{cell['accuracy']:.2f} +/- {cell['acc_student_std']:.2f}", number(cell["old_f1"]), number(cell["new_f1"]),
                number(cell["stage_average_f1"]), number(cell["construction_seconds"]), number(cell["construction_peak_gib"])])
        lines.extend([markdown(["Method", "Identity", "Final F1", "Accuracy", "Old F1", "New F1", "Stage F1", "Build s", "Peak GiB"], rows), ""])
    references = [row for row in bundle["matched"]
                  if row["arm"] in QUALIFIED_REFERENCES and row["split"] == "sealed_test"]
    if references:
        lines.extend(["## Qualified Native TAG References (Separate Panel)", "",
            "These results use the common evaluation masks and Student but differ in construction budget "
            "or supervision/topology. They are not a matched-budget ranking against the main table.", ""])
        lines.append(markdown(["Dataset", "Method", "F1 +/- Student SD", "Acc +/- Student SD",
                               "Qualification", "Build s", "Peak GiB"], [
            [DISPLAY[row["dataset"]], row["label"],
             f"{row['final_f1']:.2f} +/- {row['f1_student_std']:.2f}",
             f"{row['accuracy']:.2f} +/- {row['acc_student_std']:.2f}",
             REFERENCE_REASON[row["arm"]], number(row["construction_seconds"]),
             number(row["construction_peak_gib"])] for row in references]))
        lines.append("")
    lines.extend(["## Nine-cell Confirmation (Separate Population)", "",
        "Each cell averages the same three Student seeds; SD below is across 3 orders x 3 construction seeds. "
        "The paired t intervals summarize these nine order/seed cells and are not cross-dataset guarantees.", ""])
    lines.append(markdown(["Dataset", "Method", "F1 +/- cell SD", "Acc +/- cell SD", "Worst F1"], [
        [DISPLAY[row["dataset"]], row["label"], f"{row['final_f1_mean']:.2f} +/- {row['final_f1_cell_std']:.2f}",
         f"{row['accuracy_mean']:.2f} +/- {row['accuracy_cell_std']:.2f}", number(row["worst_f1"])]
        for row in bundle["stability"] if row["split"] == "sealed_test"]))
    lines.extend(["", markdown(["Dataset", "EF minus", "F1 delta", "95% paired t CI", "Positive cells", "Worst delta"], [
        [DISPLAY[row["dataset"]], METHODS[row["comparator"]][0], f"{row['mean_delta']:+.2f}",
         f"[{row['ci_low']:+.2f}, {row['ci_high']:+.2f}]", f"{row['positive_cells']}/{row['cells']}", f"{row['worst_delta']:+.2f}"]
        for row in bundle["paired"] if row["split"] == "sealed_test" and row["metric"] == "final_f1"]), "",
        "## Coverage", ""])
    lines.append(markdown(["Method"] + [DISPLAY[d] for d in DATASETS], [
        [METHODS[arm][0]] + [next((row["status"] for row in bundle["coverage"] if row["dataset"] == dataset and row["arm"] == arm), "not registered") for dataset in DATASETS]
        for arm in METHODS]))
    lines.extend(["", "## Adaptation Map", "", markdown(["Method", "Identity", "Implementation"], list(METHODS.values())), "",
        "## Native-core Feedback Diagnostics", "",
        "Native classification proxies are restricted to classes in the current delta. A singleton-class "
        "softmax has zero discriminative cross-entropy gradient. The following stages have both a zero "
        "final feature-update norm and zero gradients in every logged sample. These are adaptation "
        "limitations, not evidence that the original official method is ineffective. All registered "
        "scores are retained, and all per-stage diagnostics are exported to native_diagnostics.csv.", ""])
    zero_rows = [row for row in bundle["native_diagnostics"] if row["zero_feature_feedback"]]
    lines.extend([markdown(["Dataset", "Method", "Stage (0-based)", "Classes", "Feature update",
                            "Zero / logged gradients", "Objective first / last"], [
        [DISPLAY[row["dataset"]], METHODS[row["arm"]][0], row["stage"], row["stage_class_count"],
         row["feature_update_norm"], f"{row['zero_gradient_samples']}/{row['gradient_samples']}",
         f"{row['objective_first']:.4g} / {row['objective_last']:.4g}"] for row in zero_rows]), "",
        "## Scope and Limitations", "",
        "- The broader panel is a single order/construction cell, not a five-dataset stability claim.",
        "- Keep selection controls, inspired references, native-core adaptations, and component ablations distinct.",
        "- Do not rank historical 5.71% ArXiv or another Student against the current actual-5% panel.",
        "- Full-history retention requires matched schedule/splits; the old report-dev reference is not imported here.",
        "- Cost is observed wall time and PyTorch allocated peak, includes common data tensors, and may reflect concurrent jobs. It is not an isolated throughput benchmark.",
        "- Queued/failed results are missing, never zero. Historical sealed tables were not modified.", ""])
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def make_figures(bundle, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none", "pdf.fonttype": 42})
    chart_arms = SCOPE + ("gecc_soft",)
    for metric, title, deviation in (("final_f1", "Macro-F1 (%)", "f1_student_std"),
                                      ("accuracy", "Accuracy (%)", "acc_student_std")):
        fig, axes = plt.subplots(1, 5, figsize=(16.5, 7.5), sharey=True)
        for ax, dataset in zip(axes, DATASETS):
            for y, arm in enumerate(chart_arms):
                cell = next((row for row in bundle["matched"] if row["dataset"] == dataset and row["arm"] == arm and row["split"] == "sealed_test"), None)
                if cell:
                    color = "#167A84" if arm == CORE[0] else "#525B65"
                    marker = "D" if arm == CORE[0] else "s" if arm == "puma_tag" else "o"
                    ax.errorbar(cell[metric], y, xerr=cell[deviation], fmt=marker, color=color,
                                markersize=6, capsize=3, linewidth=1.1)
                    ax.annotate(f"{cell[metric]:.1f}", (cell[metric], y), xytext=(0, -14),
                                textcoords="offset points", ha="center", fontsize=8)
                else:
                    status = next((row["status"] for row in bundle["coverage"]
                                   if row["dataset"] == dataset and row["arm"] == arm), "not registered")
                    ax.text(50, y, status.replace(" / ", "\n"), ha="center", va="center",
                            color="#888888", fontsize=8)
            ax.set(title=DISPLAY[dataset], xlim=(0, 100), xlabel=title)
            ax.set_xticks([0, 25, 50, 75, 100])
            ax.grid(axis="x", color="#E3E6E8", linewidth=.6)
            ax.set_axisbelow(True)
        axes[0].set_yticks(range(len(chart_arms)), [METHODS[arm][0] for arm in chart_arms])
        axes[0].invert_yaxis()
        axes[0].set_ylim(len(chart_arms)-.3, -.7)
        fig.suptitle("Matched-cell baseline expansion", fontsize=15, y=.98)
        fig.text(.5, .925, "5% legal-pool ratio | order 9101 | construction 271006 | 3 Student seeds | error bars: Student SD", ha="center")
        fig.text(.02, .025, "Adaptations, not official reproductions. PUMA is inspired; SFGC is compute-bounded. Missing cells show their status.\nTAGSAM/STAD references with budget or supervision differences are listed separately in the tables.", fontsize=9)
        fig.tight_layout(rect=(0, .095, 1, .905))
        for suffix in ("png", "pdf"):
            fig.savefig(output / f"matched_{metric}.{suffix}", dpi=180)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(9.5, 4.3))
    plot_rows = [row for row in bundle["paired"] if row["split"] == "sealed_test" and row["metric"] == "final_f1" and row["comparator"] == "gecc_hard"]
    for y, row in enumerate(plot_rows):
        ax.errorbar(row["mean_delta"], y, xerr=[[row["mean_delta"]-row["ci_low"]], [row["ci_high"]-row["mean_delta"]]],
                    fmt="D", color="#167A84", capsize=5, markersize=7)
        ax.annotate(f"{row['mean_delta']:+.2f} [{row['ci_low']:+.2f}, {row['ci_high']:+.2f}]   {row['positive_cells']}/9 positive",
                    (row["ci_high"], y), xytext=(12, 0), textcoords="offset points", va="center", fontsize=9)
    ax.set_yticks(range(len(plot_rows)), [DISPLAY[row["dataset"]] for row in plot_rows])
    ax.set(xlim=(-2.4, 9.3), ylim=(-.65, 2.65), xlabel="Unified-Q-EF minus GECC-hard: Macro-F1 (percentage points)",
           title="Nine-cell paired confirmation")
    ax.axvline(0, color="#777777", linewidth=1, linestyle="--")
    ax.grid(axis="x", color="#E3E6E8", linewidth=.6)
    fig.text(.11, .025, "95% paired t intervals across 3 orders x 3 construction seeds; each cell averages 3 Student seeds.", fontsize=9)
    fig.tight_layout(rect=(0, .065, 1, 1))
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"nine_cell_paired.{suffix}", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bundle = summarize(json.loads(args.snapshot.read_text(encoding="utf-8")))
    args.output.mkdir(parents=True, exist_ok=True)
    for key in ("matched", "stability", "paired", "coverage", "stages", "sources", "native_diagnostics"):
        export_csv(args.output / (key + ".csv"), [{k: v for k, v in row.items() if k != "per_student"} for row in bundle[key]])
    (args.output / "bundle.json").write_text(json.dumps(bundle, indent=2, allow_nan=False), encoding="utf-8")
    (args.output / "verification.json").write_text(json.dumps(bundle["checks"], indent=2), encoding="utf-8")
    write_report(bundle, args.output)
    make_figures(bundle, args.output)
    print(json.dumps(bundle["checks"]))


if __name__ == "__main__":
    main()
