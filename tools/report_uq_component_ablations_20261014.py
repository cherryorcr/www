"""Read-only, protocol-checked reports for the registered component experiments."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

from scipy.stats import t


DATASETS = ("arxiv", "sportsfit", "bookhis")
SEEDS = (271006, 271108, 271109)
ORDER = 9101
METRICS = ("final_f1", "accuracy", "old_f1", "new_f1", "stage_average_f1")
SPLITS = ("sealed_test", "report_dev")
SIGNATURE = ("groups", "budgets", "stage_raw_rows", "memory_rate", "actual_final_rate",
             "student", "student_seeds", "student_steps", "student_batch", "pool_mask_hash",
             "report_mask_hash", "test_mask_hash", "stage_id_hashes", "budget_mode")
COMMON_SOURCES = ("unified_q_multimodal.py", "graphtext_prompt_student.py",
                  "external_continual_mechanisms.py", "run_unified_q_revision_20261007.py",
                  "run_unified_q_large_smoke_20261007.py")
LABELS = {
    "uq_dm_edgefree": "Unified-Q-EF (frozen primary)",
    "uq_no_gm": "No GM", "uq_no_dm": "No DM", "uq_fixed_q": "Fixed structured Q (primary)",
    "control_shared": "Shared Q (controlled)", "control_independent": "Independent Q",
    "control_shared_no_residual": "Shared Q, no residual",
    "control_independent_no_residual": "Independent Q, no residual",
    "control_shared_equal": "Shared Q, equal-count subspace",
    "control_independent_equal": "Independent Q, equal-count subspace",
    "control_soft_stored": "Soft forward and soft storage",
    "uq_random_init_learned": "Permuted initial costs, learned Q",
    "uq_random_init_fixed": "Permuted initial costs, fixed Q",
    "uq_structured_fixed_control": "Fixed structured Q (controlled)",
    "uq_class_shared_payload": "Class-shared text, fixed X",
    "uq_random5_payload": "Random-5 text, fixed X",
}
FAMILY = {arm: ("original_loss" if arm in ("uq_dm_edgefree", "uq_no_gm", "uq_no_dm", "uq_fixed_q")
                else "storage_text" if arm.endswith("payload") else "controlled_assignment")
          for arm in LABELS}
PAIRS = (
    ("original_loss", "uq_dm_edgefree", "uq_no_gm"),
    ("original_loss", "uq_dm_edgefree", "uq_no_dm"),
    ("original_loss", "uq_dm_edgefree", "uq_fixed_q"),
    ("controlled_assignment", "control_shared", "control_independent"),
    ("controlled_assignment", "control_shared_no_residual", "control_independent_no_residual"),
    ("controlled_assignment", "control_shared", "control_shared_no_residual"),
    ("controlled_assignment", "control_independent", "control_independent_no_residual"),
    ("subspace_sensitivity", "control_shared_equal", "control_independent_equal"),
    ("controlled_assignment", "control_shared", "control_soft_stored"),
    ("controlled_assignment", "control_shared", "uq_random_init_learned"),
    ("controlled_assignment", "uq_structured_fixed_control", "uq_random_init_fixed"),
    ("controlled_assignment", "uq_random_init_learned", "uq_random_init_fixed"),
    ("storage_text", "uq_dm_edgefree", "uq_class_shared_payload"),
    ("storage_text", "uq_dm_edgefree", "uq_random5_payload"),
    ("implementation_drift", "uq_dm_edgefree", "control_shared"),
)
AUDIT_KEYS = ("constructor", "config", "shared", "diagnostics", "trace", "assignments",
              "historical_raw_reads", "old_update_count", "old_x_relative_change_max",
              "assignment_parameter_count", "residual_parameter_count", "legal_assignment_entries",
              "equal_parameter_budget", "parameter_budget_caveat", "text_storage",
              "forward_and_storage_match", "stored_vs_reference_string_max_error",
              "text_hard_forward_max_error", "primary_feature_hash", "intervention",
              "text_unique_ratio", "training_on_stored_payload",
              "construction_forward_uses_altered_payload", "randomized_initial",
              "distribution_matching", "q_row_sum_max_error", "q_column_relative_error")


def mean(values):
    values = [float(v) for v in values if v is not None]
    return statistics.mean(values) if values else None


def sd(values):
    values = [float(v) for v in values if v is not None]
    return statistics.stdev(values) if len(values) > 1 else None


def interval(values):
    if len(values) < 2:
        return None, None
    half = float(t.ppf(.975, len(values) - 1)) * statistics.stdev(values) / math.sqrt(len(values))
    center = statistics.mean(values)
    return center - half, center + half


def collected_entries(root):
    locations = (
        ("primary", root / "results/uq_final_evidence_20261012/confirmation"),
        ("controls", root / "results/uq_final_evidence_20261012/controls"),
        ("new_components", root / "results/uq_component_ablations_20261014/cells"),
        ("supplementary", root / "results/uq_component_ablations_20261014/supplementary"),
    )
    entries = []
    for category, directory in locations:
        for path in sorted(directory.glob("*/results.json")):
            raw = path.read_bytes()
            result = json.loads(raw)
            p = result["protocol"]
            if p["dataset"] not in DATASETS or p["order_seed"] != ORDER or p["construction_seed"] not in SEEDS:
                continue
            selected = []
            for row in result["rows"]:
                if row["arm"] not in LABELS:
                    continue
                row = dict(row)
                audit = row.get("audit", {})
                row["audit"] = {k: audit[k] for k in AUDIT_KEYS if k in audit}
                row.pop("score", None)
                selected.append(row)
            entries.append(dict(category=category, path=str(path.relative_to(root)),
                sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw),
                result=dict(protocol=p, rows=selected, status=result["status"])))
    queue = root / "results/uq_component_ablations_20261014/queue.status"
    supplementary = root / "results/uq_component_ablations_20261014/supplementary.status"
    return dict(captured_at=datetime.now(timezone.utc).isoformat(), root=str(root), entries=entries,
        queue_status=queue.read_text().strip() if queue.exists() else "not_started",
        supplementary_status=supplementary.read_text().strip() if supplementary.exists() else "not_started")


def check_protocol(primary, other):
    different = [k for k in SIGNATURE if primary.get(k) != other.get(k)]
    different += [k for k in COMMON_SOURCES if primary["source_hashes"].get(k) != other["source_hashes"].get(k)]
    if different:
        raise ValueError(f"incompatible ablation protocol: {different}")


def verify_score(score):
    if (score["student_seeds"] != [150, 151, 152] or score["student_steps"] != 2048
            or score["student_batch"] != 24):
        raise ValueError("Student seeds or training budget differ")
    students = score["per_student"]
    if sorted(s["seed"] for s in students) != [150, 151, 152]:
        raise ValueError("missing or duplicate Student evaluation")
    for metric in METRICS[:-1]:
        observed = [s[metric] for s in students if s.get(metric) is not None]
        if any(not math.isfinite(x) for x in observed):
            raise ValueError("nonfinite Student score")
        expected = math.fsum(observed) / len(observed) if observed else None
        aggregate = score.get(metric)
        if (expected is None) != (aggregate is None) or (expected is not None and abs(expected - aggregate) > 1e-8):
            raise ValueError(f"Student aggregation mismatch: {metric}")


def build(snapshot):
    primary = {}
    for entry in snapshot["entries"]:
        if entry["category"] == "primary":
            p = entry["result"]["protocol"]
            primary[(p["dataset"], p["construction_seed"])] = entry
    cells, stages, diagnostics, sources = [], [], [], []
    evaluated = 0
    for entry in snapshot["entries"]:
        result, path = entry["result"], entry["path"]
        p = result["protocol"]
        key = p["dataset"], p["construction_seed"]
        if key not in primary:
            raise ValueError(f"primary reference absent: {key}")
        reference = primary[key]["result"]
        check_protocol(reference["protocol"], p)
        if p.get("method_selection_from_test") is not False:
            raise ValueError("test-based selection not excluded")
        sources.append(dict(path=path, sha256=entry["sha256"], category=entry["category"], protocol=p))
        groups = defaultdict(list)
        for row in result["rows"]:
            arm = row["arm"]
            if entry["category"] != "primary" and arm == "uq_dm_edgefree":
                source_row = next(r for r in reference["rows"] if r["arm"] == arm and r["stage"] == row["stage"])
                if (row["memory_hash"] != source_row["memory_hash"] or row["scores"] != source_row["scores"]
                        or not row.get("reused_evaluation")):
                    raise ValueError("payload replay differs from frozen primary")
                # Replay diagnostics are useful; do not duplicate the primary's scores or costs.
            else:
                groups[arm].append(row)
            if row["memory_rows"] != p["budgets"][row["stage"]]:
                raise ValueError(f"memory budget differs: {path}")
            for flag in ("budget_ok", "prefix_ok", "finite_ok", "sinkhorn_feasible"):
                if row["checks"].get(flag) is False:
                    raise ValueError(f"failed {flag}: {path}")
            audit = row.get("audit", {})
            if audit.get("historical_raw_reads") != 0:
                raise ValueError(f"historical raw access not excluded: {path}")
            base = dict(dataset=p["dataset"], order_seed=p["order_seed"], construction_seed=p["construction_seed"],
                        arm=arm, family=FAMILY[arm], stage=row["stage"], memory_rows=row["memory_rows"],
                        source=path, source_sha256=entry["sha256"], replay=row.get("reused_evaluation", False))
            for split in SPLITS:
                score = row["scores"][split]
                verify_score(score)
                evaluated += 3
                stages.append(dict(**base, split=split, **{m: score.get(m) for m in METRICS[:-1]},
                    per_student=score["per_student"]))
            d = audit.get("diagnostics", {})
            trace = audit.get("trace", [])
            record = dict(**base, construction_seconds=row.get("construction_seconds"),
                construction_peak_gib=row.get("construction_peak_mb", 0) / 1024,
                assignment_parameters=audit.get("assignment_parameter_count"),
                residual_parameters=audit.get("residual_parameter_count"),
                legal_assignment_entries=audit.get("legal_assignment_entries"),
                q_gradient_norm=mean([s.get("q_gradient_norm") for s in trace]),
                residual_gradient_norm=mean([s.get("residual_gradient_norm") for s in trace]))
            for k, v in d.items():
                if isinstance(v, (int, float, str)):
                    record[k] = v
            for role, values in d.get("assignments", {}).items():
                record.update({role + "_" + k: v for k, v in values.items()})
            record["stored_text_unique_ratio"] = audit.get("text_unique_ratio", d.get("unique_text_ratio"))
            record["text_storage"] = audit.get("text_storage")
            diagnostics.append(record)
        for arm, rows in groups.items():
            stage_ids = sorted(r["stage"] for r in rows)
            if len(stage_ids) != len(set(stage_ids)):
                raise ValueError(f"duplicate stages: {path} {arm}")
            if stage_ids != list(range(len(p["budgets"]))):
                continue
            rows.sort(key=lambda r: r["stage"])
            last = rows[-1]
            for split in SPLITS:
                score = last["scores"][split]
                cells.append(dict(dataset=p["dataset"], order_seed=p["order_seed"], construction_seed=p["construction_seed"],
                    arm=arm, family=FAMILY[arm], split=split, source=path, source_sha256=entry["sha256"],
                    memory_hash=last["memory_hash"], memory_rows=last["memory_rows"],
                    actual_rate=p["actual_final_rate"], stages=len(rows), student_runs=3,
                    **{m: score.get(m) for m in METRICS[:-1]},
                    stage_average_f1=mean([r["scores"][split]["final_f1"] for r in rows]),
                    student_f1_sd=sd([s["final_f1"] for s in score["per_student"]]),
                    student_acc_sd=sd([s["accuracy"] for s in score["per_student"]]),
                    construction_seconds=sum(r["construction_seconds"] for r in rows),
                    construction_peak_gib=max(r["construction_peak_mb"] for r in rows) / 1024))
    keys = [(r["dataset"], r["construction_seed"], r["arm"], r["split"]) for r in cells]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate complete cells; keep one authoritative trajectory")
    grouped = defaultdict(list)
    for row in cells:
        grouped[(row["dataset"], row["arm"], row["split"])].append(row)
    summary = []
    for (dataset, arm, split), rows in grouped.items():
        item = dict(dataset=dataset, arm=arm, family=FAMILY[arm], label=LABELS[arm], split=split,
            cells=len(rows), construction_seeds=",".join(str(r["construction_seed"]) for r in sorted(rows, key=lambda r:r["construction_seed"])),
            orders=1, student_runs=len(rows) * 3, mean_within_cell_student_f1_sd=mean([r["student_f1_sd"] for r in rows]))
        for metric in METRICS + ("construction_seconds", "construction_peak_gib"):
            item[metric + "_mean"] = mean([r[metric] for r in rows])
            item[metric + "_cell_sd"] = sd([r[metric] for r in rows])
        summary.append(item)
    pairs = []
    for dataset in DATASETS:
        for family, left, right in PAIRS:
            for split in SPLITS:
                index = {arm: {r["construction_seed"]: r for r in cells
                    if r["dataset"] == dataset and r["arm"] == arm and r["split"] == split} for arm in (left, right)}
                matches = sorted(set(index[left]) & set(index[right]))
                for metric in METRICS:
                    values = [(seed, index[left][seed][metric] - index[right][seed][metric]) for seed in matches
                        if index[left][seed][metric] is not None and index[right][seed][metric] is not None]
                    if not values:
                        continue
                    diffs = [v for _, v in values]
                    low, high = interval(diffs)
                    worst_seed, worst = min(values, key=lambda v:v[1])
                    pairs.append(dict(dataset=dataset, family=family, left=left, right=right, split=split,
                        metric=metric, cells=len(values), mean_delta=mean(diffs), delta_sd=sd(diffs),
                        ci_low=low, ci_high=high, positive_cells=sum(v > 0 for v in diffs),
                        positive_ratio=sum(v > 0 for v in diffs) / len(diffs), worst_seed=worst_seed,
                        worst_delta=worst, scope="one fixed order; construction-cell paired t CI"))
    coverage = []
    for dataset in DATASETS:
        for arm in LABELS:
            expected = (SEEDS[:1] if arm.endswith("_equal") else SEEDS)
            found = {r["construction_seed"] for r in cells if r["dataset"] == dataset and r["arm"] == arm and r["split"] == "sealed_test"}
            coverage.append(dict(dataset=dataset, arm=arm, family=FAMILY[arm], complete=len(found),
                expected=len(expected), missing=sorted(set(expected) - found)))
    return dict(captured_at=snapshot["captured_at"], queue_status=snapshot["queue_status"],
        supplementary_status=snapshot["supplementary_status"], cells=cells, stages=stages,
        summary=summary, pairs=pairs, diagnostics=diagnostics, coverage=coverage, sources=sources,
        checks=dict(protocols_checked=len(sources), score_records_checked=evaluated,
                    primary_replays_not_double_counted=True, complete_cells=len(keys)//2))


def export_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row if k not in ("protocol", "per_student")))
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows({k:v for k,v in row.items() if k in fields} for row in rows)


def fmt(x, digits=2):
    return "--" if x is None else f"{x:.{digits}f}"


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
        + ["| " + " | ".join(str(x) for x in row) + " |" for row in rows])


def report(bundle, output):
    lines = ["# Unified-Q-EF component ablations", "", f"Captured: {bundle['captured_at']}", "",
        "External baselines are in the separate baseline expansion report. This report contains only component interventions.", "",
        "Exact cumulative 5% legal-pool schedule; order 9101; construction seeds 271006/271108/271109; Student seeds 150/151/152, 2048 steps, MiniLM. Report-dev and sealed test are stored separately. These are post-hoc diagnostics, with no parameter selection from these results.", "",
        "Mean +/- SD below is across construction-cell means (each averages 3 Student seeds). With n=1, construction SD and paired CI are unavailable. Paired 95% t intervals use construction cells, not Student runs, and describe one fixed order only.", "",
        "Controlled shared Q is a separate implementation. Compare independent, random-initialized and soft-storage controls against its matched shared version. The primary-versus-controlled drift is explicitly reported and is not component attribution.", "",
        "Class-shared and random-5 are fixed-X, storage-only interventions. Random-5 has one 320-character total budget across sources. Equal-count subspaces alter update geometry and are sensitivity tests.", "",
        f"Main queue: `{bundle['queue_status']}`; supplementary queue: `{bundle['supplementary_status']}`.", ""]
    titles = {"original_loss": "Original-constructor loss controls", "controlled_assignment": "Matched assignment controls", "storage_text": "Storage-only text controls"}
    for dataset in DATASETS:
        lines.extend([f"## {dataset.title()}", ""])
        selected = [r for r in bundle["summary"] if r["dataset"] == dataset and r["split"] == "sealed_test"]
        for family, title in titles.items():
            rows = [r for arm in LABELS for r in selected if r["arm"] == arm and r["family"] == family]
            if not rows:
                continue
            lines.extend([f"### {title}", "", table(
                ["Component", "n", "F1 mean +/- SD", "Acc mean +/- SD", "Old F1", "New F1", "Stage avg F1", "Build sec", "Peak GiB"],
                [[r["label"], r["cells"], fmt(r["final_f1_mean"])+" +/- "+fmt(r["final_f1_cell_sd"]),
                  fmt(r["accuracy_mean"])+" +/- "+fmt(r["accuracy_cell_sd"]), fmt(r["old_f1_mean"]),
                  fmt(r["new_f1_mean"]), fmt(r["stage_average_f1_mean"]), fmt(r["construction_seconds_mean"],1),
                  fmt(r["construction_peak_gib_mean"])] for r in rows]), ""])
        comparisons = [r for r in bundle["pairs"] if r["dataset"] == dataset and r["split"] == "sealed_test" and r["metric"] == "final_f1"]
        lines.extend(["### Paired F1 deltas", "", table(["Left minus right", "Scope", "n", "Delta pp", "95% CI", "Positive cells", "Worst pp"],
            [[LABELS[r["left"]]+" - "+LABELS[r["right"]], r["family"], r["cells"], fmt(r["mean_delta"]),
              "--" if r["ci_low"] is None else "["+fmt(r["ci_low"])+", "+fmt(r["ci_high"])+"]",
              str(r["positive_cells"])+"/"+str(r["cells"]), fmt(r["worst_delta"])] for r in comparisons]), ""])
    lines.extend(["## Coverage", "", table(["Dataset", "Component", "Complete/expected", "Missing construction seeds"],
        [[r["dataset"], LABELS[r["arm"]], str(r["complete"])+"/"+str(r["expected"]), str(r["missing"])] for r in bundle["coverage"]]), "",
        "## Diagnostics and audit limits", "",
        "`diagnostics.csv` records per-stage hard slot coverage, positive mass coverage, row entropy, column support entropy/effective support size where available, text uniqueness/diversity, soft/hard cosine gap, and Q/residual gradient norms. Positive mass coverage near 1 follows the capacity constraint and is not evidence of efficacy. Gradient norms have different parameter scales. Diagnostics from replay are not duplicated in performance statistics.", "",
        "Blank diagnostics were not collected by the historical producer; they are missing values, not zeros. Storage-only controls report stored-text uniqueness; they do not have a new construction assignment. Timing is measured wall time, and replay timing is a diagnostic cost rather than the frozen primary's historical cost.", "",
        "All source paths, hashes and protocols are in `sources.json`; all stages/Student scores are in `bundle.json`. Incomplete trajectories are not included in final-score tables.", ""])
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def figures(bundle, output):
    try:
        import matplotlib
    except ImportError:
        (output / "figures_skipped.txt").write_text(
            "matplotlib is unavailable in the reporting environment; "
            "the data tables and verification bundle were still exported.\n",
            encoding="utf-8")
        return
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":10, "axes.spines.top":False, "axes.spines.right":False})
    for family, filename in (("original_loss", "loss_paired"), ("controlled_assignment", "assignment_paired"), ("storage_text", "text_paired")):
        selected = [r for r in bundle["pairs"] if r["family"] == family and r["split"] == "sealed_test" and r["metric"] == "final_f1"]
        comparisons = [(l,r) for f,l,r in PAIRS if f == family and any(x["left"] == l and x["right"] == r for x in selected)]
        if not selected:
            continue
        fig, axes = plt.subplots(1,3,figsize=(15,max(3.2, .72*len(comparisons)+1.2)), sharex=True, layout="constrained")
        values = [v for r in selected for v in (r["ci_low"],r["ci_high"],r["mean_delta"]) if v is not None]
        extent = max(max(map(abs, values)), .5) * 1.22
        for ax,dataset in zip(axes,DATASETS):
            ax.axvline(0,color="#555555",linestyle="--",linewidth=1)
            for i,(left,right) in enumerate(comparisons):
                row = next((r for r in selected if r["dataset"]==dataset and r["left"]==left and r["right"]==right),None)
                if row is None:
                    ax.text(0,i,"pending",ha="center",color="#777777")
                    continue
                ax.scatter(row["mean_delta"],i,s=38,c="#2166ac",marker="o" if row["cells"]==3 else "x",zorder=3)
                if row["ci_low"] is not None:
                    ax.hlines(i,row["ci_low"],row["ci_high"],color="#2166ac",linewidth=1.5)
                ax.annotate(f"{row['mean_delta']:+.2f}; n={row['cells']}",(row["mean_delta"],i),xytext=(0,8),textcoords="offset points",fontsize=9,ha="center")
            ax.set_yticks(range(len(comparisons)),[LABELS[l]+"\nminus "+LABELS[r] for l,r in comparisons] if ax is axes[0] else [""]*len(comparisons))
            ax.set_ylim(len(comparisons)-.5,-.65)
            ax.set_xlim(-extent,extent)
            ax.set_title(dataset.title())
            ax.set_xlabel("Macro-F1 delta (pp)")
            ax.grid(axis="x",color="#dddddd",linewidth=.5)
        fig.suptitle(f"{family.replace('_',' ').title()}: order 9101; paired 95% t CI across construction cells\nn=1: no interval; each cell averages 3 Student seeds")
        for ext in ("png","pdf"):
            fig.savefig(output / f"{filename}.{ext}",dpi=180)
        plt.close(fig)


def publish(root, destination):
    snapshot = collected_entries(root)
    bundle = build(snapshot)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output = destination / ("report_utc_" + stamp)
    output.mkdir(parents=True, exist_ok=False)
    for name,value in (("snapshot",snapshot),("bundle",bundle),("sources",bundle["sources"]),("verification",bundle["checks"])):
        (output / (name+".json")).write_text(json.dumps(value,indent=2,allow_nan=False),encoding="utf-8")
    for key in ("cells","stages","summary","pairs","diagnostics","coverage"):
        export_csv(output / (key+".csv"),bundle[key])
    report(bundle,output)
    figures(bundle,output)
    receipt = dict(report=str(output.relative_to(root)), captured_at=bundle["captured_at"],
        completed=sum(r["complete"] for r in bundle["coverage"]), expected=sum(r["expected"] for r in bundle["coverage"]),
        all_registered_complete=all(not r["missing"] for r in bundle["coverage"]), checks=bundle["checks"])
    temporary = destination / "latest_report_receipt.tmp"
    temporary.write_text(json.dumps(receipt,indent=2),encoding="utf-8")
    temporary.replace(destination / "latest_report_receipt.json")
    print(json.dumps(receipt),flush=True)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,default=Path("/home/pc/research/TAG_condensation"))
    parser.add_argument("--destination",type=Path)
    args = parser.parse_args()
    publish(args.root,args.destination or args.root / "results/uq_component_ablations_20261014")
