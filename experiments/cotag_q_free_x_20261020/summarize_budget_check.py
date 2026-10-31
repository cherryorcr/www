"""Keep single-construction budget diagnostics separate from primary statistics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import summarize as S


COMPONENTS = ("loss", "gradient_match", "distribution_match", "propagation")


def score_rows(record, arm, updates, source):
    stages = [row for row in record["rows"] if row["arm"] == arm]
    final = stages[-1]
    scores = []
    for architecture in ("GCN", "GraphSAGE"):
        selected = final["scores"] if architecture == "GCN" else record["sage"][arm]["scores"]
        for split in S.SPLITS:
            per_student = selected[split]["per_student"]
            assert [row["seed"] for row in per_student] == [150, 151, 152]
            for metric in ("final_f1", "accuracy", "old_f1", "new_f1"):
                values = [row[metric] for row in per_student]
                scores.append(dict(dataset=record["protocol"]["dataset"], arm=arm,
                    updates=updates, architecture=architecture, split=split, metric=metric,
                    construction_seed=271006, order_seed=9101, n_constructions=1,
                    n_students=3, mean=float(np.mean(values)),
                    student_sd=float(np.std(values, ddof=1)), source=source))
    return scores


def audit_long(record, primary):
    p, reference = record["protocol"], primary["protocol"]
    assert all(p[key] == reference[key] for key in S.MATCH)
    assert p["source_hashes"] == reference["source_hashes"]
    assert p["control_hash"] == reference["control_hash"]
    assert p["primary_file_hash"] == reference["primary_file_hash"]
    assert p["sage_hash"] == reference["sage_hash"]
    assert p["sage_student"] == reference["sage_student"]
    assert p["memory_rate"] == .05 and not p["method_selection_from_test"]
    for row in record["rows"]:
        expected = dict(reference["q_config"], steps=100, sinkhorn_tolerance=2e-4)
        if row["arm"] == S.FREE:
            expected["lambda_x_anchor"] = 0.
        assert row["audit"]["config"] == expected
        assert row["memory_rows"] == p["budgets"][row["stage"]]
        assert all(row["checks"][key] for key in
                   ("budget_ok", "prefix_ok", "finite_ok", "sinkhorn_feasible"))
        assert row["audit"]["historical_raw_reads"] == 0
        assert row["audit"]["old_update_count"] == 0
        assert row["audit"]["text_hard_forward_max_error"] < 1e-4
    for arm, entry in record.get("sage", {}).items():
        stages = [row for row in record["rows"] if row["arm"] == arm]
        assert [row["stage"] for row in stages] == list(range(len(p["groups"])))
        assert entry["memory_hash"] == stages[-1]["memory_hash"]
    return True


def compare(scores, key, left, right):
    grouped = {}
    grouping = ("dataset", "architecture", "split", "metric",
                "arm" if key == "updates" else "updates")
    for row in scores:
        identifier = tuple(row[name] for name in grouping)
        grouped.setdefault(identifier, {})[row[key]] = row
    result = []
    for identifier, group in grouped.items():
        if left not in group or right not in group:
            continue
        result.append(dict(zip(grouping, identifier), contrast=f"{left} - {right}",
            difference=group[left]["mean"] - group[right]["mean"],
            n_constructions=1, paired_construction_ci=None))
    return result


def markdown(scores, changes, contrasts, diagnostics, failures):
    lines = ["# Longer-budget diagnostic", "",
        "One construction (271006), one order (9101), three Student fits per memory.",
        "The 30-update rows below use that same construction, not the three-construction primary mean.",
        "100-update runs use an amended Sinkhorn tolerance of 2e-4 instead of 1e-4.",
        "Therefore this is not a pure step-only contrast. No construction-level CI can be estimated here.",
        "Test-mask observations were inspected before; these are not fresh blind evidence.", ""]
    for split in S.SPLITS:
        lines += [f"## {split}", "", "| Dataset | Student | Arm | Updates | F1 | Accuracy |",
                  "|---|---|---|---:|---:|---:|"]
        for row in scores:
            if row["split"] != split or row["metric"] != "final_f1":
                continue
            accuracy = next(r for r in scores if r["metric"] == "accuracy" and all(
                r[key] == row[key] for key in ("dataset", "architecture", "arm", "updates", "split")))
            lines.append(f"| {S.DS_LABEL[row['dataset']]} | {row['architecture']} | {S.LABEL[row['arm']]} | "
                         f"{row['updates']} | {row['mean']:.2f} | {accuracy['mean']:.2f} |")
        lines += ["", "| Dataset | Student | Arm | F1 change, 100 minus 30 |",
                  "|---|---|---|---:|"]
        for row in changes:
            if row["split"] == split and row["metric"] == "final_f1":
                lines.append(f"| {S.DS_LABEL[row['dataset']]} | {row['architecture']} | "
                             f"{S.LABEL[row['arm']]} | {row['difference']:+.2f} |")
        lines += ["", "| Dataset | Student | Updates | F1 difference, full minus independent X |",
                  "|---|---|---:|---:|"]
        for row in contrasts:
            if row["split"] == split and row["metric"] == "final_f1":
                lines.append(f"| {S.DS_LABEL[row['dataset']]} | {row['architecture']} | "
                             f"{row['updates']} | {row['difference']:+.2f} |")
        lines.append("")
    lines += ["## Loss stability", "",
        "Two adjacent changes between the final three ten-update means must both be below 1%.",
        "Each objective is examined within its own run; their absolute minima are not equivalent.", "",
        "| Dataset | Arm | Stage | Component | Last loss | Window changes (%) | Plateau |",
        "|---|---|---:|---|---:|---|---|"]
    for row in diagnostics:
        changes_pct = ", ".join(f"{v * 100:.2f}" for v in row.get("last_two_relative_changes", []))
        lines.append(f"| {S.DS_LABEL[row['dataset']]} | {S.LABEL[row['arm']]} | {row['stage']} | "
                     f"{row['component']} | {row.get('last_value', float('nan')):.6f} | "
                     f"{changes_pct or 'sparse observations'} | {row.get('plateau_both_below_1pct', 'unavailable')} |")
    lines += ["", "## Attempts and failures", "",
              "Missing or failed outcomes are not zero scores. Failed attempts are not independent repetitions.", ""]
    for row in failures:
        message = row["failure_message"].replace("\n", " ")
        lines.append(f"- {row['path']}: {row['status']}; {row['saved_stages']} saved stage rows; "
                     f"{row['failure_type']}: {message}")
    return "\n".join(lines) + "\n"


def plots(output, traces):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
        "axes.spines.right": False, "pdf.fonttype": 42, "savefig.dpi": 200})
    colors = {S.FULL: "#CB4B44", S.FREE: "#168A80"}
    for dataset in sorted(set(row["dataset"] for row in traces)):
        final_stage = 2 if dataset == "arxiv" else 4
        fig, axes = plt.subplots(2, 2, figsize=(10, 6), layout="constrained")
        for ax, component in zip(axes.flat, COMPONENTS):
            any_data = False
            for arm in (S.FULL, S.FREE):
                points = sorted((row for row in traces if row["dataset"] == dataset
                    and row["arm"] == arm and row["stage"] == final_stage
                    and component in row), key=lambda row: row["step"])
                if not points:
                    continue
                initial = points[0][component]
                if initial == 0:
                    continue
                any_data = True
                ax.plot([row["step"] for row in points],
                    [row[component] / initial for row in points],
                    color=colors[arm], ls="-" if arm == S.FULL else "--",
                    label="CoTAG-Q" if arm == S.FULL else "Independent X", lw=1.6)
            ax.set_title(component.replace("_", " ").title())
            ax.set_xlabel("Construction update (final stage)")
            ax.set_ylabel("Component / its initial value")
            ax.set_xlim(1, 100)
            ax.set_ylim(bottom=0)
            ax.grid(alpha=.2)
            if any_data:
                ax.legend(frameon=False)
            else:
                ax.text(.5, .5, "No completed final stage", transform=ax.transAxes, ha="center")
        fig.suptitle(f"{S.DS_LABEL[dataset]} | one construction | 100-update diagnostic")
        fig.savefig(output / f"components_{dataset}.png")
        fig.savefig(output / f"components_{dataset}.pdf")
        plt.close(fig)


def main(args):
    primary, _, _ = S.collect(args.results)
    scores, audits, failures, traces, diagnostics, costs = [], [], [], [], [], []
    for dataset in S.DATASETS:
        record = primary[(dataset, 271006)]
        for arm in (S.FULL, S.FREE):
            scores += score_rows(record, arm, 30, "primary experiment, construction 271006")
    complete_arms = set()
    folders = sorted(args.results.glob("cotag_q_free_x_budget100*"))
    for folder in folders:
        for path in sorted(folder.glob("*/results.json")):
            record = S.read(path)
            dataset = record["protocol"]["dataset"]
            if record.get("status") != "complete":
                failure = record.get("failure", {})
                stopped = folder.name == "cotag_q_free_x_budget100_20261020" and dataset == "arxiv"
                failures.append(dict(path=str(path.relative_to(args.results)),
                    status="interrupted (process absence verified)" if stopped else record.get("status"),
                    raw_file_status=record.get("status"), saved_stages=len(record.get("rows", [])),
                    failure_type="TerminatedDuringGPUConflict" if stopped else failure.get("type", "interrupted/unresolved"),
                    failure_message="first attempt overlapped the primary queue and was stopped; immutable raw file still says running" if stopped
                        else failure.get("message", "see BUDGET_CHECK.md and process audit")))
            if not record.get("rows") or not record.get("sage"):
                continue
            audit_long(record, primary[(dataset, 271006)])
            for arm in (S.FULL, S.FREE):
                if arm not in record["sage"]:
                    continue
                key = (dataset, arm)
                if key in complete_arms:
                    raise AssertionError(f"duplicate complete 100-update arm: {key}")
                complete_arms.add(key)
                scores += score_rows(record, arm, 100, str(path))
                audits.append(dict(dataset=dataset, arm=arm, path=str(path),
                    result_hash=S.sha(path), checks="passed", tolerance_amendment=True))
                for row in record["rows"]:
                    if row["arm"] != arm:
                        continue
                    trace = S.dense_trace(row)
                    for point in trace:
                        traces.append(dict(dataset=dataset, arm=arm, stage=row["stage"], **point))
                    costs.append(dict(dataset=dataset, arm=arm, stage=row["stage"],
                        construction_seconds=row["construction_seconds"],
                        construction_peak_mib=row["construction_peak_mb"],
                        sinkhorn_max_iterations=row["audit"].get("sinkhorn_max_iterations"),
                        includes_global_evaluation_cache=True))
                    for component in COMPONENTS:
                        diagnostics.append(dict(dataset=dataset, arm=arm, stage=row["stage"],
                            component=component, **S.plateau(trace, component)))
    scores.sort(key=lambda r: (S.DATASETS.index(r["dataset"]), r["architecture"],
                              r["arm"], r["updates"], r["split"], r["metric"]))
    changes = compare(scores, "updates", 100, 30)
    contrasts = compare(scores, "arm", S.FULL, S.FREE)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, rows in (("scores", scores), ("budget_changes", changes),
                       ("arm_contrasts", contrasts), ("plateau_diagnostics", diagnostics),
                       ("trajectories", traces), ("costs", costs), ("attempts", failures)):
        S.write_csv(args.output / f"{name}.csv", rows)
    payload = dict(completed_100_arms=len(complete_arms), expected_arms=6,
                   audits=audits, scores=scores, changes=changes, contrasts=contrasts,
                   diagnostics=diagnostics, costs=costs, attempts=failures)
    (args.output / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (args.output / "summary.md").write_text(markdown(scores, changes, contrasts, diagnostics, failures), encoding="utf-8")
    if args.plot:
        plots(args.output, traces)
    print(json.dumps(dict(completed_100_arms=len(complete_arms), output=str(args.output))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=S.HERE / "results")
    parser.add_argument("--output", type=Path, default=S.HERE / "analysis_budget100")
    parser.add_argument("--plot", action="store_true")
    main(parser.parse_args())
