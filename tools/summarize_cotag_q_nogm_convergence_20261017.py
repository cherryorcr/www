"""Summarize no-GM update sweeps against paired no-GM30 report-dev results.

Partial continual sequences contribute traces, but never final-cell scores.
The baseline may contain sealed scores; this report only reads report_dev.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from statistics import mean, stdev


DATASETS = ("arxiv", "sportsfit", "bookhis")
SEEDS = (271006, 271108, 271109)
STEPS = (30, 60, 100, 200)
METRICS = ("final_f1", "accuracy", "old_f1", "new_f1",
           "stage_average_f1", "stage_average_accuracy",
           "construction_seconds", "construction_peak_gib")
PROTOCOL_KEYS = ("dataset", "groups", "budgets", "stage_raw_rows", "order_seed",
                 "construction_seed", "student_seeds", "student_steps",
                 "student_batch", "student", "memory_rate", "actual_final_rate")


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def execution_failure(path: Path, dataset: str, steps: int, seed: int) -> dict | None:
    """Use the last queue event; failed runners can leave status='running'."""
    logs = path.parents[3] / "logs"
    status = logs / "status"
    if not status.exists():
        return None
    last = None
    for line in status.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if (len(fields) >= 5 and fields[0] in ("start", "finish")
                and fields[2:5] == [dataset, str(steps), str(seed)]):
            last = fields
    if not last or last[0] != "finish":
        return None
    code = next((int(f[3:]) for f in last[5:] if f.startswith("rc=")), 0)
    if not code:
        return None
    records = []
    messages = []
    for log in sorted(logs.glob(f"q*_{dataset}_steps{steps}_c{seed}.log")):
        content = log.read_text(encoding="utf-8", errors="replace")
        records.append(dict(path=str(log), sha256=hashlib.sha256(log.read_bytes()).hexdigest()))
        messages.extend(line.strip() for line in content.splitlines()
                        if line.startswith(("torch.OutOfMemoryError:", "RuntimeError:",
                                            "ValueError:", "AssertionError:")))
    kind = ("oom" if any("OutOfMemoryError" in line for line in messages) else
            "sinkhorn_tolerance" if any("Sinkhorn failed tolerance" in line for line in messages)
            else "process_failure")
    return dict(dataset=dataset, steps=steps, seed=seed, status="failed",
                exit_code=code, finished_at=last[1], failure_kind=kind,
                reason=messages[-1] if messages else f"queue exit code {code}",
                messages=messages, logs=records, result_path=str(path))


def check_protocol(result: dict, baseline: dict) -> dict:
    current, prior = result["protocol"], baseline["protocol"]
    differences = [key for key in PROTOCOL_KEYS if current.get(key) != prior.get(key)]
    if current.get("objective_variant") != "no_gm":
        differences.append("objective_variant")
    if current.get("method_selection_from_test") is not False:
        differences.append("method_selection_from_test")
    a = {k: v for k, v in current["q_config"].items() if k != "steps"}
    b = {k: v for k, v in prior["q_config"].items() if k != "steps"}
    if a != b or a.get("lambda_gradient") != 0.0:
        differences.append("q_config_except_steps")
    if current["q_config"]["steps"] != current["construction_steps"]:
        differences.append("construction_steps")
    for name, digest in prior["source_hashes"].items():
        if current.get("source_hashes", {}).get(name) != digest:
            differences.append("source_hash:" + name)
    if differences:
        raise ValueError("protocol mismatch: " + ", ".join(differences))
    missing = [key for key in ("pool_mask_hash", "report_mask_hash", "stage_id_hashes")
               if key not in current]
    return dict(recorded_fields_match=True, unrecorded_current_receipts=missing,
                mask_identity_independently_verified=False)


def score_of(row: dict) -> dict:
    score = row["scores"]["report_dev"] if "scores" in row else row["score"]
    if score.get("split") != "report_dev":
        raise ValueError("only report-dev scores are allowed")
    if score.get("student_seeds") != [150, 151, 152]:
        raise ValueError("unexpected Student seeds")
    return score


def trace_record(dataset: str, steps: int, seed: int, row: dict, source: str) -> dict:
    audit = row.get("audit", {})
    trace = sorted(audit.get("trace", []), key=lambda t: t["step"])
    losses = {int(t["step"]): float(t["loss"]) for t in trace}
    last = max(losses, default=None)
    window_start = max((s for s in losses if s <= last - 20), default=None) if last else None
    relative = ((losses[window_start] - losses[last]) / max(abs(losses[window_start]), 1e-12)
                if window_start is not None else None)
    # Practical plateau diagnostic: two consecutive 10-update windows with
    # <1% relative loss reduction. This is a stopping heuristic, not a proof
    # that the underlying optimization gradient is zero.
    ordered = sorted(losses)
    window_drops = []
    for end in ordered:
        start = end - 10
        if start in losses and losses[start] != 0:
            window_drops.append(dict(start=start, end=end,
                                     relative_loss_drop=(losses[start] - losses[end]) /
                                     max(abs(losses[start]), 1e-12)))
    last_two = window_drops[-2:]
    plateau = (len(last_two) == 2 and
               all(abs(item["relative_loss_drop"]) < 0.01 for item in last_two))
    return dict(dataset=dataset, steps=steps, construction_seed=seed, stage=row["stage"],
                source=source, final_observed_loss=losses.get(last), last_trace_step=last,
                loss_at_30=losses.get(30), last_window_start=window_start,
                last_window_relative_loss_drop=relative,
                loss_window_10_drops=window_drops,
                plateau_tolerance=0.01,
                plateau_last_two_10_windows=plateau,
                trace=trace, sinkhorn_max_iterations=audit.get("sinkhorn_max_iterations"),
                construction_seconds=row["construction_seconds"],
                construction_peak_gib=row["construction_peak_mb"] / 1024.0,
                numerical_repair=row.get("numerical_sinkhorn_repair"))


def cell_record(result: dict, source: str, steps: int) -> dict:
    protocol = result["protocol"]
    rows = sorted(result["rows"], key=lambda r: r["stage"])
    if result.get("status") != "complete" or [r["stage"] for r in rows] != list(range(len(protocol["groups"]))):
        raise ValueError("incomplete continual sequence")
    scores = [score_of(row) for row in rows]
    traces = [row["audit"].get("trace", []) for row in rows]
    return dict(dataset=protocol["dataset"], steps=steps,
                construction_seed=protocol["construction_seed"], order_seed=protocol["order_seed"],
                source=source, **{k: scores[-1].get(k) for k in METRICS[:4]},
                stage_average_f1=mean(s["final_f1"] for s in scores),
                stage_average_accuracy=mean(s["accuracy"] for s in scores),
                construction_seconds=sum(r["construction_seconds"] for r in rows),
                construction_peak_gib=max(r["construction_peak_mb"] for r in rows) / 1024.0,
                mean_final_observed_loss=mean(t[-1]["loss"] for t in traces) if all(traces) else None,
                per_student=scores[-1]["per_student"],
                has_numerical_repair=any(r.get("numerical_sinkhorn_repair") for r in rows))


def aggregate(cells: list[dict]) -> list[dict]:
    output = []
    for dataset in DATASETS:
        for steps in STEPS:
            group = [c for c in cells if c["dataset"] == dataset and c["steps"] == steps]
            if not group:
                continue
            record = dict(dataset=dataset, steps=steps, n_construction_seeds=len(group),
                          construction_seeds=sorted(c["construction_seed"] for c in group))
            for key in METRICS:
                values = [c[key] for c in group if c.get(key) is not None]
                record[key + "_mean"] = mean(values) if values else None
                record[key + "_std"] = stdev(values) if len(values) > 1 else None
            output.append(record)
    return output


def paired(cells: list[dict]) -> list[dict]:
    output = []
    index = {(c["dataset"], c["steps"], c["construction_seed"]): c for c in cells}
    # Independent units are construction seeds; Student seeds are averaged first.
    t_critical = {2: 12.7062047364321, 3: 4.3026527299113}
    for dataset in DATASETS:
        for steps in STEPS[1:]:
            seeds = [s for s in SEEDS if (dataset, steps, s) in index and (dataset, 30, s) in index]
            if not seeds:
                continue
            for metric in METRICS:
                deltas = [index[dataset, steps, s][metric] - index[dataset, 30, s][metric]
                          for s in seeds if index[dataset, steps, s].get(metric) is not None
                          and index[dataset, 30, s].get(metric) is not None]
                if not deltas:
                    continue
                n, delta = len(deltas), mean(deltas)
                half = t_critical[n] * stdev(deltas) / n**0.5 if n in t_critical else None
                output.append(dict(dataset=dataset, steps=steps, versus_steps=30, metric=metric,
                                   n_construction_seeds=n, mean_delta=delta,
                                   matched_baseline_mean=mean(index[dataset, 30, s][metric] for s in seeds
                                                              if index[dataset, 30, s].get(metric) is not None),
                                   matched_sweep_mean=mean(index[dataset, steps, s][metric] for s in seeds
                                                           if index[dataset, steps, s].get(metric) is not None),
                                   ci95_low=delta - half if half is not None else None,
                                   ci95_high=delta + half if half is not None else None,
                                   positive_seed_ratio=sum(v > 0 for v in deltas) / n,
                                   worst_seed_delta=min(deltas), deltas=deltas, seeds=seeds))
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows({k: json.dumps(v) if isinstance(v, (list, dict)) else v
                         for k, v in row.items()} for row in rows)


def plot(output: Path, summaries: list[dict], traces: list[dict]) -> str | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return "matplotlib unavailable; CSV and JSON retained for local plotting"
    colors = {30: "#6b7280", 60: "#0072b2", 100: "#009e73", 200: "#d55e00"}
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), constrained_layout=True)
    for col, dataset in enumerate(DATASETS):
        group = sorted((r for r in summaries if r["dataset"] == dataset), key=lambda r: r["steps"])
        for row_index, metric in enumerate(("final_f1", "construction_seconds")):
            ax = axes[row_index, col]
            if group:
                ax.errorbar([r["steps"] for r in group], [r[metric + "_mean"] for r in group],
                            yerr=[r[metric + "_std"] or 0.0 for r in group],
                            marker="o", capsize=3, color="#0072b2")
                for r in group:
                    ax.annotate("n=" + str(r["n_construction_seeds"]),
                                (r["steps"], r[metric + "_mean"]),
                                xytext=(4, 5), textcoords="offset points", fontsize=8)
            ax.set_title(dataset)
            ax.set_xlabel("Construction updates")
            ax.set_ylabel("Report-dev Macro-F1 (%)" if row_index == 0 else "Construction time (s)")
            ax.grid(axis="y", alpha=0.2)
    fig.savefig(output / "completed_cells.png", dpi=200)
    plt.close(fig)
    fig, axes = plt.subplots(3, 3, figsize=(11, 8), constrained_layout=True)
    for col, dataset in enumerate(DATASETS):
        for stage in range(3):
            ax = axes[stage, col]
            for steps in STEPS:
                group = [r for r in traces if (r["dataset"], r["stage"], r["steps"]) == (dataset, stage, steps)]
                points = {}
                for record in group:
                    for t in record["trace"]:
                        points.setdefault(t["step"], []).append(t["loss"])
                if points:
                    x = sorted(points)
                    ax.plot(x, [mean(points[s]) for s in x], color=colors[steps],
                            label=f"{steps} updates", linewidth=1.5)
            ax.set_title(f"{dataset}, stage {stage + 1}")
            ax.set_xlabel("Construction update")
            ax.set_ylabel("Observed pre-update objective")
            ax.set_yscale("log")
            ax.grid(axis="y", alpha=0.2)
            if ax.lines:
                ax.legend(frameon=False, fontsize=7)
    fig.savefig(output / "loss_traces.png", dpi=200)
    plt.close(fig)
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/home/pc/research/TAG_condensation"))
    parser.add_argument("--sweep-root", type=Path, action="append")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--render-report", type=Path)
    args = parser.parse_args()
    if args.render_report:
        report = read(args.render_report / "summary.json")
        note = plot(args.render_report, report["aggregate"], report["stage_traces"])
        print(json.dumps(dict(output=str(args.render_report), plot_note=note)))
        return
    roots = args.sweep_root or [args.root / "results/cotag_q_nogm_step_sensitivity_20261017"]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output = args.output or roots[0] / ("report_utc_" + stamp)
    output.mkdir(parents=True, exist_ok=False)
    cells, traces, pending, sources, errors, receipts, failures = [], [], [], [], [], [], []
    for dataset in DATASETS:
        for seed in SEEDS:
            baseline_path = args.root / "results/uq_component_ablations_20261014/cells" / f"{dataset}_no_gm_o9101_c{seed}/results.json"
            try:
                baseline = read(baseline_path)
                if baseline["protocol"]["q_config"]["lambda_gradient"] != 0.0:
                    raise ValueError("baseline must be no-GM")
                if baseline["protocol"]["q_config"]["steps"] != 30:
                    raise ValueError("baseline must use 30 updates")
                cells.append(cell_record(baseline, str(baseline_path), 30))
                sources.append(dict(path=str(baseline_path), sha256=hashlib.sha256(baseline_path.read_bytes()).hexdigest()))
                traces.extend(trace_record(dataset, 30, seed, row, str(baseline_path)) for row in baseline["rows"])
            except (OSError, ValueError, KeyError) as exc:
                errors.append(dict(path=str(baseline_path), error=str(exc)))
                continue
            for steps in STEPS[1:]:
                found = [p for root in roots if (p := root / dataset / f"steps{steps}/c{seed}/results.json").exists()]
                if not found:
                    pending.append(dict(dataset=dataset, steps=steps, seed=seed, status="not_present"))
                    continue
                if len(found) != 1:
                    errors.append(dict(dataset=dataset, steps=steps, seed=seed, error="duplicate result paths; select roots explicitly"))
                    continue
                path = found[0]
                try:
                    result = read(path)
                    receipt = check_protocol(result, baseline)
                    receipts.append(dict(path=str(path), **receipt))
                    sources.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
                    for row in result["rows"]:
                        if score_of(row)["eval_rows"] != score_of(baseline["rows"][row["stage"]])["eval_rows"]:
                            raise ValueError("report-dev evaluation row count changed")
                        traces.append(trace_record(dataset, steps, seed, row, str(path)))
                    if result.get("status") == "complete":
                        cells.append(cell_record(result, str(path), steps))
                    else:
                        failure = execution_failure(path, dataset, steps, seed)
                        if failure:
                            failure["completed_stages"] = len(result["rows"])
                            failure["raw_result_status"] = result.get("status")
                            failures.append(failure)
                        else:
                            pending.append(dict(dataset=dataset, steps=steps, seed=seed,
                                                status=result.get("status"), completed_stages=len(result["rows"])))
                except (OSError, ValueError, KeyError) as exc:
                    errors.append(dict(path=str(path), error=str(exc)))
    summaries, comparisons = aggregate(cells), paired(cells)
    plot_note = plot(output, summaries, traces)
    report = dict(created_utc=stamp, split="report_dev_only", objective_variant="no_gm",
                  baseline="paired no-GM30; no full-GM baseline mixed in", cells=cells,
                  aggregate=summaries, paired_vs_30=comparisons, stage_traces=traces,
                  pending=pending, execution_failures=failures, errors=errors, source_receipts=sources,
                  protocol_checks=receipts, sealed_results_untouched=True, plot_note=plot_note,
                  caveats=["One class order; CI units are construction seeds, not individual Student runs.",
                           "Partial sequences are excluded from final-cell means and paired differences.",
                           "Loss reduction alone does not demonstrate downstream improvement or stationarity.",
                           "Solver iteration counts are adaptive despite a common Sinkhorn configuration.",
                           "no-GM with joint-DM disabled has no explicit text-loss-to-Q feedback.",
                           "Current sweep protocol omits mask hash receipts; row counts and recorded fields are checked."])
    (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for name, rows in (("cells", cells), ("aggregate", summaries), ("paired_vs_30", comparisons),
                       ("stage_traces", traces), ("execution_failures", failures)):
        write_csv(output / (name + ".csv"), rows)
    lines = ["# no-GM construction convergence (report-dev only)", "",
             "Completed sequences only. Values average Student seeds 150/151/152 within each construction seed.", "",
             "| Dataset | Updates | Construction seeds | Final F1 | Accuracy | Time (s) | Peak allocated (GiB) |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for r in summaries:
        lines.append(f"| {r['dataset']} | {r['steps']} | {r['n_construction_seeds']} | {r['final_f1_mean']:.3f} | {r['accuracy_mean']:.3f} | {r['construction_seconds_mean']:.1f} | {r['construction_peak_gib_mean']:.2f} |")
    lines.extend(["", "## Matched construction seeds", "",
                  "| Dataset | Updates | Seeds | Metric | no-GM30 | Longer run | Paired delta |",
                  "|---|---:|---:|---|---:|---:|---:|"])
    for r in comparisons:
        if r["metric"] in ("final_f1", "accuracy"):
            lines.append(f"| {r['dataset']} | {r['steps']} | {r['n_construction_seeds']} | {r['metric']} | {r['matched_baseline_mean']:.3f} | {r['matched_sweep_mean']:.3f} | {r['mean_delta']:+.3f} |")
    plateau_rows = [t for t in traces if t.get("plateau_last_two_10_windows")]
    lines.extend(["", "## Practical loss plateau", "",
                  "A stage is marked plateaued only when the last two consecutive 10-update windows each reduce the observed objective by less than 1%. This is a practical stopping rule, not a mathematical convergence proof.", "",
                  f"Plateaued stage traces: {len(plateau_rows)}/{len(traces)}.",
                  "The detailed 10-update reductions are recorded in `stage_traces.csv`.",
                  "", "Paired deltas and construction-seed CIs: `paired_vs_30.csv`.",
                  "Per-stage objective traces, including finished stages of running cells: `stage_traces.csv`.",
                  f"Pending entries: {len(pending)}. Failed cells: {len(failures)}. Validation errors: {len(errors)}.", ""])
    if failures:
        lines.extend(["## Failed executions", "",
                      "Queue exit records identify failures even when results.json still says running.", "",
                      "| Dataset | Updates | Construction seed | Failure |",
                      "|---|---:|---:|---|"])
        for failure in failures:
            lines.append(f"| {failure['dataset']} | {failure['steps']} | {failure['seed']} | {failure['failure_kind']} |")
        lines.extend(["", "Failure messages and log hashes: `execution_failures.csv`.", ""])
    lines.extend("- " + caveat for caveat in report["caveats"])
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(dict(output=str(output), completed_sweep_cells=sum(c["steps"] != 30 for c in cells),
                          baseline_cells=sum(c["steps"] == 30 for c in cells), pending=len(pending),
                          failed_cells=len(failures), errors=errors, aggregate=summaries), indent=2))


if __name__ == "__main__":
    main()
