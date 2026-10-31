"""Summarize CoTAG-Q construction-step sensitivity without touching sealed tables."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


DATASETS = ("arxiv", "sportsfit", "bookhis")
SEEDS = (271006, 271108, 271109)
STEPS = (10, 30, 60, 100)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def trace_summary(audit: dict) -> dict:
    trace = audit.get("trace", [])
    if not trace:
        return {"loss_initial": None, "loss_final": None, "loss_step20": None,
                "loss_step30": None, "loss_drop": None}
    losses = {int(item["step"]): float(item["loss"]) for item in trace}
    first = losses[min(losses)]
    last = losses[max(losses)]
    return {"loss_initial": first, "loss_final": last,
            "loss_step20": losses.get(20), "loss_step30": losses.get(30),
            "loss_drop": first - last}


def row_from_primary(root: Path, dataset: str, seed: int) -> dict:
    path = root / "results/uq_final_evidence_20261012/confirmation" / \
        f"{dataset}_o9101_c{seed}/results.json"
    result = read_json(path)
    rows = sorted((row for row in result["rows"]
                   if row["arm"] == "uq_dm_edgefree"), key=lambda x: x["stage"])
    if not rows:
        raise FileNotFoundError(f"no primary rows in {path}")
    final = rows[-1]
    traces = [trace_summary(row.get("audit", {})) for row in rows]
    losses = [v["loss_final"] for v in traces if v["loss_final"] is not None]
    return dict(dataset=dataset, steps=30, construction_seed=seed,
                source="sealed_confirmation_report_dev_reused_after_protocol_hash",
                final_f1=final["scores"]["report_dev"]["final_f1"],
                accuracy=final["scores"]["report_dev"]["accuracy"],
                stage_average_f1=float(np.mean([row["scores"]["report_dev"]["final_f1"]
                                                for row in rows])),
                stage_average_accuracy=float(np.mean([row["scores"]["report_dev"]["accuracy"]
                                                      for row in rows])),
                construction_seconds=float(sum(row.get("construction_seconds", 0.0)
                                               for row in rows)),
                construction_peak_mb=float(max(row.get("construction_peak_mb", 0.0)
                                               for row in rows)),
                loss_initial=float(np.mean([v["loss_initial"] for v in traces
                                            if v["loss_initial"] is not None])),
                loss_final=float(np.mean(losses)) if losses else None,
                loss_step20=float(np.mean([v["loss_step20"] for v in traces
                                           if v["loss_step20"] is not None])) if any(v["loss_step20"] is not None for v in traces) else None,
                loss_step30=float(np.mean([v["loss_step30"] for v in traces
                                           if v["loss_step30"] is not None])) if any(v["loss_step30"] is not None for v in traces) else None,
                loss_drop=float(np.mean([v["loss_drop"] for v in traces
                                         if v["loss_drop"] is not None])) if losses else None)


def row_from_sweep(root: Path, dataset: str, steps: int, seed: int) -> dict:
    path = root / "results/cotag_q_step_sensitivity_20261016" / dataset / \
        f"steps{steps}/c{seed}/results.json"
    result = read_json(path)
    rows = sorted(result["rows"], key=lambda x: x["stage"])
    if not rows or result.get("status") != "complete":
        raise FileNotFoundError(f"incomplete sensitivity result: {path}")
    traces = [trace_summary(row.get("audit", {})) for row in rows]
    losses = [v["loss_final"] for v in traces if v["loss_final"] is not None]
    return dict(dataset=dataset, steps=steps, construction_seed=seed,
                source="construction_step_sensitivity_report_dev",
                final_f1=rows[-1]["score"]["final_f1"],
                accuracy=rows[-1]["score"]["accuracy"],
                stage_average_f1=float(np.mean([row["score"]["final_f1"] for row in rows])),
                stage_average_accuracy=float(np.mean([row["score"]["accuracy"] for row in rows])),
                construction_seconds=float(sum(row["construction_seconds"] for row in rows)),
                construction_peak_mb=float(max(row["construction_peak_mb"] for row in rows)),
                loss_initial=float(np.mean([v["loss_initial"] for v in traces
                                            if v["loss_initial"] is not None])),
                loss_final=float(np.mean(losses)) if losses else None,
                loss_step20=float(np.mean([v["loss_step20"] for v in traces
                                           if v["loss_step20"] is not None])) if any(v["loss_step20"] is not None for v in traces) else None,
                loss_step30=float(np.mean([v["loss_step30"] for v in traces
                                           if v["loss_step30"] is not None])) if any(v["loss_step30"] is not None for v in traces) else None,
                loss_drop=float(np.mean([v["loss_drop"] for v in traces
                                         if v["loss_drop"] is not None])) if losses else None)


def aggregate(rows: list[dict]) -> list[dict]:
    output = []
    keys = ("final_f1", "accuracy", "stage_average_f1", "stage_average_accuracy",
            "construction_seconds", "construction_peak_mb", "loss_final")
    for dataset in DATASETS:
        for steps in STEPS:
            cell = [row for row in rows if row["dataset"] == dataset and row["steps"] == steps]
            if not cell:
                continue
            item = dict(dataset=dataset, steps=steps, n=len(cell))
            for key in keys:
                vals = [row[key] for row in cell if row.get(key) is not None]
                item[key + "_mean"] = float(np.mean(vals)) if vals else None
                item[key + "_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0 if vals else None
            output.append(item)
    return output


def maybe_plot(output: Path, rows: list[dict], summary: list[dict]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        (output / "plot_note.txt").write_text(f"matplotlib unavailable: {exc}\n", encoding="utf-8")
        return
    colors = {"arxiv": "#2f5d8c", "sportsfit": "#c47f2c", "bookhis": "#5b7f55"}
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.5), constrained_layout=True)
    for dataset in DATASETS:
        vals = [v for v in summary if v["dataset"] == dataset]
        if not vals:
            continue
        x = [v["steps"] for v in vals]
        y = [v["final_f1_mean"] for v in vals]
        e = [v["final_f1_std"] for v in vals]
        axes[0].errorbar(x, y, yerr=e, marker="o", linewidth=1.8,
                         color=colors[dataset], label=dataset)
        y = [v["accuracy_mean"] for v in vals]
        e = [v["accuracy_std"] for v in vals]
        axes[1].errorbar(x, y, yerr=e, marker="o", linewidth=1.8,
                         color=colors[dataset], label=dataset)
        y = [v["construction_seconds_mean"] for v in vals]
        e = [v["construction_seconds_std"] for v in vals]
        axes[2].errorbar(x, y, yerr=e, marker="o", linewidth=1.8,
                         color=colors[dataset], label=dataset)
    axes[0].set_ylabel("Report-dev Macro-F1 (%)")
    axes[1].set_ylabel("Report-dev Accuracy (%)")
    axes[2].set_ylabel("Construction time (s)")
    for ax in axes:
        ax.set_xlabel("Outer construction updates")
        ax.set_xticks(STEPS)
        ax.grid(axis="y", alpha=.25)
        ax.set_xscale("symlog", linthresh=10)
    axes[0].legend(frameon=False, fontsize=8)
    fig.savefig(output / "step_sensitivity.png", dpi=220)
    fig.savefig(output / "step_sensitivity.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/home/pc/research/TAG_condensation"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or (args.root / "results/cotag_q_step_sensitivity_20261016")
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    missing = []
    for dataset in DATASETS:
        for seed in SEEDS:
            try:
                rows.append(row_from_primary(args.root, dataset, seed))
            except Exception as exc:
                missing.append(dict(dataset=dataset, steps=30, seed=seed, reason=str(exc)))
            for steps in (10, 60, 100):
                try:
                    rows.append(row_from_sweep(args.root, dataset, steps, seed))
                except Exception as exc:
                    missing.append(dict(dataset=dataset, steps=steps, seed=seed, reason=str(exc)))
    summary = aggregate(rows)
    write_json(output / "summary.json", dict(rows=rows, aggregate=summary, missing=missing,
                                              sealed_results_untouched=True))
    with (output / "rows.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted(rows[0]) if rows else ["dataset"])
        writer.writeheader()
        writer.writerows(rows)
    with (output / "aggregate.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted(summary[0]) if summary else ["dataset"])
        writer.writeheader()
        writer.writerows(summary)
    maybe_plot(output, rows, summary)
    print(json.dumps(dict(output=str(output), complete_rows=len(rows),
                          missing=len(missing), aggregate=summary), indent=2,
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
