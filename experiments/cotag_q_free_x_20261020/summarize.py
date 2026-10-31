"""Audit paired memories and summarize nested Student repetitions."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import t


HERE = Path(__file__).resolve().parent
DATASETS = ("arxiv", "sportsfit", "bookhis")
SEEDS = (271006, 271108, 271109)
FULL, FREE, FIXED = "uq_dm_edgefree", "independent_x", "uq_fixed_q"
LABEL = {FULL: "CoTAG-Q", FREE: "Independent X + learned text Q", FIXED: "Fixed Q (archive)"}
DS_LABEL = dict(arxiv="ArXiv", sportsfit="SportsFit", bookhis="BookHis")
SPLITS = ("report_dev", "sealed_test")
KEYS = ("final_f1", "accuracy", "old_f1", "new_f1", "stage_mean_f1")
MATCH = ("dataset", "order_seed", "construction_seed", "groups", "budgets",
         "stage_raw_rows", "student_seeds", "student_steps", "student_batch",
         "student", "pool_mask_hash", "report_mask_hash", "test_mask_hash", "stage_id_hashes")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def moments(values):
    values = np.asarray(values, float)
    return dict(n=len(values), mean=float(values.mean()),
                sd=float(values.std(ddof=1)) if len(values) > 1 else None)


def paired(left, right):
    diff = np.asarray(left, float) - np.asarray(right, float)
    out = moments(diff)
    half = (float(t.ppf(.975, len(diff) - 1) * diff.std(ddof=1) / np.sqrt(len(diff)))
            if len(diff) > 1 else None)
    return dict(out, ci_low=None if half is None else out["mean"] - half,
                ci_high=None if half is None else out["mean"] + half,
                positive_count=int((diff > 0).sum()), positive_ratio=float((diff > 0).mean()),
                worst_difference=float(diff.min()), paired_differences=diff.tolist())


def dense_trace(row):
    audit = row["audit"]
    trace = audit.get("component_trajectory") or audit.get("trace") or []
    names = {"gm": "gradient_match", "distribution": "distribution_match"}
    return [{names.get(k, k): v for k, v in point.items()} for point in trace]


def plateau(trace, component):
    points = {point["step"]: point[component] for point in trace if component in point}
    if not points:
        return {}
    last = max(points)
    out = dict(first_value=points[min(points)], last_value=points[last], trace_points=len(points))
    if last >= 30 and all(step in points for step in range(last - 29, last + 1)):
        windows = [float(np.mean([points[i] for i in range(start, start + 10)]))
                   for start in (last - 29, last - 19, last - 9)]
        changes = [abs(b - a) / max(abs(a), 1e-12) for a, b in zip(windows, windows[1:])]
        out.update(last_three_window_means=windows, last_two_relative_changes=changes,
                   plateau_both_below_1pct=all(change < .01 for change in changes))
    return out


def collect(root):
    results, audits, fixed_results = {}, [], {}
    hashes = dict(runner_hash=sha(HERE / "run.py"), control_hash=sha(HERE / "independent_x.py"))
    for folder in ("cotag_q_free_x_20261020_v1", "cotag_q_free_x_20261020_v2"):
        for path in sorted((root / folder).glob("*/results.json")):
            record = read(path)
            if record.get("status") != "complete":
                continue
            p = record["protocol"]
            key = (p["dataset"], p["construction_seed"])
            if key in results:
                raise AssertionError(f"duplicate complete pair: {key}")
            assert p["order_seed"] == 9101 and p["student_seeds"] == [150, 151, 152]
            assert p["student_steps"] == 2048 and p["student_batch"] == 24
            assert p["q_config"]["steps"] == 30 and p["memory_rate"] == .05
            assert not p["method_selection_from_test"]
            assert all(p[k] == h for k, h in hashes.items())
            assert all(p["primary_protocol_checks"].values())
            stage_count = 3 if p["dataset"] == "arxiv" else 5
            assert len(p["groups"]) == stage_count
            arm_audit = []
            for arm in (FULL, FREE):
                rows = [r for r in record["rows"] if r["arm"] == arm]
                assert [r["stage"] for r in rows] == list(range(stage_count))
                assert [r["memory_rows"] for r in rows] == p["budgets"]
                for row in rows:
                    check, audit = row["checks"], row["audit"]
                    assert all(check[k] for k in ("budget_ok", "prefix_ok", "finite_ok", "sinkhorn_feasible"))
                    assert audit["historical_raw_reads"] == 0 and audit["old_update_count"] == 0
                    assert audit["text_hard_forward_max_error"] < 1e-4
                    assert not audit["uses_q_topology"] and audit["topology"]["mode"] == "edge_free"
                    if arm == FREE:
                        assert audit["config"]["lambda_x_anchor"] == 0
                        assert audit["free_x_update_count"] == 30
                    for split in SPLITS:
                        score = row["scores"][split]
                        assert [v["seed"] for v in score["per_student"]] == [150, 151, 152]
                        for metric in KEYS[:4]:
                            values = [r[metric] for r in score["per_student"]]
                            if all(v is None for v in values):
                                assert score[metric] is None
                            else:
                                assert abs(score[metric] - np.mean(values)) < 1e-8
                sage = record["sage"][arm]
                assert sage["memory_hash"] == rows[-1]["memory_hash"]
                for split in SPLITS:
                    assert [v["seed"] for v in sage["scores"][split]["per_student"]] == [150, 151, 152]
                arm_audit.append(dict(arm=arm, stages=stage_count,
                    memory_reconstructed=rows[-1]["memory_reconstructed"],
                    reconstruction_hash_match=rows[-1].get("reconstruction_hash_match"),
                    final_memory_hash=rows[-1]["memory_hash"]))
            fpath = root / "uq_component_ablations_20261014/cells" / (
                f"{p['dataset']}_fixed_q_o9101_c{p['construction_seed']}/results.json")
            fixed = read(fpath)
            fp = fixed["protocol"]
            assert fixed["status"] == "complete"
            assert all(fp[k] == p[k] for k in MATCH)
            assert all(sha(HERE / "frozen_code" / name) == value
                       for name, value in fp["source_hashes"].items())
            assert fp["primary_source_hash"] == p["primary_file_hash"]
            assert fp["q_config"] == dict(p["q_config"], train_q=False)
            assert [r["stage"] for r in fixed["rows"]] == list(range(stage_count))
            results[key], fixed_results[key] = record, fixed
            audits.append(dict(dataset=key[0], construction_seed=key[1],
                result_path=str(path), result_hash=sha(path), fixed_path=str(fpath),
                fixed_hash=sha(fpath), checks="passed", arms=arm_audit))
    return results, fixed_results, audits


def observations(results, fixed_results):
    rows, drift, diagnostics, traces = [], [], [], []
    for (ds, seed), result in sorted(results.items()):
        for arm in (FULL, FREE, FIXED):
            record = result if arm != FIXED else fixed_results[(ds, seed)]
            stages = [r for r in record["rows"] if r["arm"] == arm]
            final = stages[-1]
            for split in SPLITS:
                for architecture in ("GCN", "GraphSAGE"):
                    if arm == FIXED and architecture == "GraphSAGE":
                        continue
                    score = (final["scores"][split] if architecture == "GCN" else
                             record["sage"][arm]["scores"][split])
                    for student in score["per_student"]:
                        value = dict(dataset=ds, construction_seed=seed,
                                     student_seed=student["seed"], arm=arm,
                                     architecture=architecture, split=split)
                        value.update({metric: student[metric] for metric in KEYS[:4]})
                        if architecture == "GCN":
                            value["stage_mean_f1"] = float(np.mean([
                                next(v for v in r["scores"][split]["per_student"]
                                     if v["seed"] == student["seed"])["final_f1"] for r in stages]))
                        rows.append(value)
                if arm == FULL and final.get("original_scores"):
                    drift.append(dict(dataset=ds, construction_seed=seed, split=split,
                        hash_match=final["reconstruction_hash_match"],
                        original_f1=final["original_scores"][split]["final_f1"],
                        reconstructed_f1=final["scores"][split]["final_f1"],
                        difference_f1=final["scores"][split]["final_f1"] - final["original_scores"][split]["final_f1"]))
            for row in stages:
                a = row["audit"]
                diagnostic = dict(dataset=ds, construction_seed=seed, arm=arm,
                    stage=row["stage"], new_rows=row["new_rows"], memory_rows=row["memory_rows"],
                    construction_seconds=row["construction_seconds"],
                    construction_peak_mib=row["construction_peak_mb"],
                    cost_provenance="archive" if arm == FIXED or (row.get("primary_metrics_reused")
                        and not row.get("memory_reconstructed")) else "current run",
                    unique_text_ratio=a.get("unique_text_ratio"),
                    assignment_entropy=a.get("q_entropy"))
                diagnostic.update(a.get("diagnostics", {}))
                for key in ("x_displacement_ratio_mean", "x_displacement_ratio_max",
                            "fraction_outside_original_trust_radius", "x_residual_ratio_max"):
                    if key in a:
                        diagnostic[key] = a[key]
                trace = dense_trace(row)
                for component in ("loss", "gradient_match", "distribution_match", "propagation"):
                    for key, value in plateau(trace, component).items():
                        diagnostic[f"{component}_{key}"] = value
                diagnostics.append(diagnostic)
                for point in trace:
                    traces.append(dict(dataset=ds, construction_seed=seed, arm=arm,
                                       stage=row["stage"], **point))
    return rows, drift, diagnostics, traces


def aggregate(rows):
    means, comparisons, construction = [], [], []
    for ds in DATASETS:
        for architecture in ("GCN", "GraphSAGE"):
            for split in SPLITS:
                values = {}
                for arm in (FULL, FREE, FIXED):
                    selected = [r for r in rows if (r["dataset"], r["architecture"], r["split"], r["arm"]) ==
                                (ds, architecture, split, arm)]
                    if not selected:
                        continue
                    values[arm] = {}
                    for seed in sorted(set(r["construction_seed"] for r in selected)):
                        group = [r for r in selected if r["construction_seed"] == seed]
                        record = dict(dataset=ds, architecture=architecture, split=split,
                                      arm=arm, construction_seed=seed, student_repetitions=len(group))
                        for metric in KEYS:
                            if metric in group[0]:
                                record[metric] = float(np.mean([r[metric] for r in group]))
                        values[arm][seed] = record
                        construction.append(record)
                    for metric in KEYS:
                        if metric in next(iter(values[arm].values())):
                            means.append(dict(dataset=ds, architecture=architecture, split=split,
                                arm=arm, metric=metric, **moments([r[metric] for r in values[arm].values()])))
                for control in (FREE, FIXED):
                    if control not in values or FULL not in values:
                        continue
                    seeds = sorted(set(values[FULL]) & set(values[control]))
                    for metric in KEYS:
                        if metric not in values[FULL][seeds[0]]:
                            continue
                        comparison = paired([values[FULL][s][metric] for s in seeds],
                                            [values[control][s][metric] for s in seeds])
                        comparisons.append(dict(dataset=ds, architecture=architecture, split=split,
                            contrast=f"{FULL} - {control}", metric=metric, construction_seeds=seeds,
                            **comparison))
    return means, comparisons, construction


def fmt(value):
    return f"{value['mean']:.2f} +/- {value['sd']:.2f}" if value["sd"] is not None else f"{value['mean']:.2f} (n=1)"


def markdown(means, comparisons, audits):
    text = ["# Independent feature optimization: audited results", "",
        f"Completed paired constructions: {len(audits)}/9. No paper edits.", "",
        "Means and sample SD are over construction seeds after averaging three Student seeds per memory.",
        "ArXiv has 3 stages; SportsFit and BookHis have 5. Order seed is 9101.",
        "Historical test results use a previously inspected mask and are not fresh blind evidence.", ""]
    for split in SPLITS:
        text += [f"## {split}", "", "| Dataset | Student | Arm | Final Macro-F1 | Accuracy |",
                 "|---|---|---|---:|---:|"]
        for ds in DATASETS:
            for architecture in ("GCN", "GraphSAGE"):
                for arm in (FULL, FREE, FIXED):
                    selected = {r["metric"]: r for r in means if (r["dataset"], r["architecture"], r["split"], r["arm"]) ==
                                (ds, architecture, split, arm)}
                    if selected:
                        text.append(f"| {DS_LABEL[ds]} | {architecture} | {LABEL[arm]} | {fmt(selected['final_f1'])} | {fmt(selected['accuracy'])} |")
        text += ["", "Paired differences in percentage points; positive favors CoTAG-Q.", "",
                 "| Dataset | Student | Control | Difference | 95% paired t CI | Positive constructions | Worst difference |",
                 "|---|---|---|---:|---|---:|---:|"]
        for r in comparisons:
            if r["split"] != split or r["metric"] != "final_f1":
                continue
            ci = f"[{r['ci_low']:.2f}, {r['ci_high']:.2f}]" if r["ci_low"] is not None else "unavailable"
            text.append(f"| {DS_LABEL[r['dataset']]} | {r['architecture']} | {LABEL[r['contrast'].split(' - ')[1]]} | {r['mean']:+.2f} | {ci} | {r['positive_count']}/{r['n']} | {r['worst_difference']:+.2f} |")
        text.append("")
    text += ["## Interpretation limits", "",
        "- Independent X removes the moving Q-conditioned feature construction, trust-radius projection and X anchor. It retains learned Q for text and propagation targets.",
        "- This tests that constrained parameterization as a package, not shared Q alone or identical objectives.",
        "- GraphSAGE uses the previously frozen self-loop mean implementation with a separate root mapping.",
        "- Missing original full memories are reconstructed and compared using their actual new scores. Drift is listed separately.",
        "- Three construction seeds in one class order give limited statistical power; nine Student fits are not nine independent memories.",
        "- Per-stage construction peaks include the existing transductive evaluation cache and cannot establish physical streaming-memory savings.",
        "- Training losses are diagnostic. A limited-budget difference is not a proof of asymptotic superiority.", ""]
    return "\n".join(text)


def plots(output, means, comparisons, traces):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "savefig.dpi": 200, "pdf.fonttype": 42})
    colors = {FULL: "#CB4B44", FREE: "#168A80", FIXED: "#979797"}
    for split in SPLITS:
        fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), layout="constrained")
        for i, architecture in enumerate(("GCN", "GraphSAGE")):
            for j, ds in enumerate(DATASETS):
                ax = axes[i, j]
                arms = (FULL, FREE, FIXED) if i == 0 else (FULL, FREE)
                for x, arm in enumerate(arms):
                    match = [r for r in means if (r["dataset"], r["architecture"], r["split"], r["arm"], r["metric"]) ==
                             (ds, architecture, split, arm, "final_f1")]
                    if not match:
                        continue
                    value = match[0]
                    ax.bar(x, value["mean"], color=colors[arm], width=.6, zorder=2)
                    ax.errorbar(x, value["mean"], yerr=value["sd"] or 0, fmt="none", color="#303030", capsize=4, zorder=3)
                    ax.annotate(f"{value['mean']:.2f}", (x, value["mean"] + (value["sd"] or 0)),
                                xytext=(0, 5), textcoords="offset points", ha="center", fontsize=10)
                ax.set_xticks(range(len(arms)), ["CoTAG-Q", "Independent X", "Fixed Q"][:len(arms)])
                ax.set_title(f"{DS_LABEL[ds]} | {architecture}")
                ax.set_ylim(0, 100)
                ax.grid(axis="y", alpha=.18, zorder=0)
                if j == 0:
                    ax.set_ylabel("Final Macro-F1 (%)")
        title = "Report-development mask" if split == "report_dev" else "Previously inspected test mask"
        fig.suptitle(f"{title} | mean and SD over construction seeds", fontsize=13)
        fig.savefig(output / f"final_f1_{split}.png")
        fig.savefig(output / f"final_f1_{split}.pdf")
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout="constrained")
    for ax, ds in zip(axes, DATASETS):
        for arm in (FULL, FREE):
            selected = [r for r in traces if r["dataset"] == ds and r["arm"] == arm
                        and r["stage"] == (2 if ds == "arxiv" else 4)]
            for seed in sorted(set(r["construction_seed"] for r in selected)):
                points = sorted((r for r in selected if r["construction_seed"] == seed), key=lambda r: r["step"])
                first = points[0]["loss"]
                ax.plot([r["step"] for r in points], [r["loss"] / first for r in points],
                        color=colors[arm], alpha=.45, lw=1.1,
                        ls="-" if arm == FULL else "--",
                        marker="o" if len(points) < 10 else None, markersize=3)
            ax.plot([], [], color=colors[arm], ls="-" if arm == FULL else "--",
                    label="CoTAG-Q" if arm == FULL else "Independent X")
        ax.set_title(DS_LABEL[ds])
        ax.set_xlabel("Construction update (final stage)")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
        ax.set_ylim(0, 1.15)
        ax.grid(alpha=.18)
    axes[0].set_ylabel("Loss / initial loss (within each run)")
    axes[-1].legend(frameon=False)
    fig.suptitle("Within-run normalized loss; sparse archive observations marked with dots", fontsize=12)
    fig.savefig(output / "final_stage_loss.png")
    fig.savefig(output / "final_stage_loss.pdf")
    plt.close(fig)


def main(args):
    results, fixed, audits = collect(args.results)
    rows, drift, diagnostic, traces = observations(results, fixed)
    means, differences, construction = aggregate(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, values in (("student_scores", rows), ("construction_scores", construction),
                         ("aggregate_scores", means), ("paired_differences", differences),
                         ("reconstruction_drift", drift), ("stage_diagnostics", diagnostic),
                         ("loss_trajectories", traces)):
        write_csv(args.output / f"{name}.csv", values)
    payload = dict(completed=len(audits), expected=9, audits=audits, aggregates=means,
                   paired_differences=differences, reconstruction_drift=drift)
    (args.output / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (args.output / "summary.md").write_text(markdown(means, differences, audits), encoding="utf-8")
    if args.plot and means:
        plots(args.output, means, differences, traces)
    print(json.dumps(dict(completed=len(audits), expected=9, output=str(args.output))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=HERE / "results")
    parser.add_argument("--output", type=Path, default=HERE / "analysis")
    parser.add_argument("--plot", action="store_true")
    main(parser.parse_args())
