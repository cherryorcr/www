"""Build auditable overnight summaries without touching sealed results.

This report combines already completed construction cells.  It keeps the
numerical Sinkhorn-recovery cell visible as a repair and never overwrites the
strict primary tables.
"""
from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "unified_q_overnight_20261012"
DATASETS = ("arxiv", "sportsfit", "bookhis", "wikics")
STABILITY_DATASETS = ("sportsfit", "bookhis", "wikics")
METHODS = ("Unified-Q-EF", "GECC-hard-TAG", "CaT-CCTP")


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


def ci95(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    mean = float(np.mean(values))
    if len(values) < 2:
        return mean, mean
    half = float(student_t.ppf(0.975, len(values) - 1) *
                 np.std(values, ddof=1) / math.sqrt(len(values)))
    return mean - half, mean + half


def final_stage(payload: dict, arm: str) -> dict:
    rows = [row for row in payload.get("rows", []) if row.get("arm") == arm]
    if not rows:
        raise ValueError(f"arm {arm} missing from {payload}")
    return max(rows, key=lambda row: int(row["stage"]))


def load_uq_cells() -> list[dict]:
    root = ROOT / "results" / "uq_q_seed_attribution_20261010"
    found: dict[tuple[str, int, int], dict] = {}
    for path in root.glob("*_o*_c*/results.json"):
        if not re.fullmatch(r"(?:wikics|sportsfit|bookhis)_o\d+_c\d+", path.parent.name):
            continue
        payload = read_json(path)
        if payload.get("status") != "complete":
            continue
        p = payload["protocol"]
        row = final_stage(payload, "uq_shared_q")
        key = (p["dataset"], int(p["order_seed"]), int(p["construction_seed"]))
        found[key] = {
            "dataset": p["dataset"], "order": int(p["order_seed"]),
            "construction_seed": int(p["construction_seed"]),
            "method": "Unified-Q-EF", "final_f1": row["score"]["final_f1"],
            "accuracy": row["score"]["accuracy"], "old_f1": row["score"].get("old_f1"),
            "new_f1": row["score"].get("new_f1"), "source": str(path),
            "recovered": False, "sinkhorn_max_steps": p.get("sinkhorn_max_steps"),
            "student_f1_sd": row["score"].get("final_f1_std"),
        }
    recovery = root / "sportsfit_o9103_c271108_recovery" / "results.json"
    if recovery.exists():
        payload = read_json(recovery)
        p = payload["protocol"]
        row = final_stage(payload, "uq_shared_q")
        key = (p["dataset"], int(p["order_seed"]), int(p["construction_seed"]))
        found[key] = {
            "dataset": p["dataset"], "order": int(p["order_seed"]),
            "construction_seed": int(p["construction_seed"]),
            "method": "Unified-Q-EF", "final_f1": row["score"]["final_f1"],
            "accuracy": row["score"]["accuracy"], "old_f1": row["score"].get("old_f1"),
            "new_f1": row["score"].get("new_f1"), "source": str(recovery),
            "recovered": True, "sinkhorn_max_steps": p.get("sinkhorn_max_steps"),
            "student_f1_sd": row["score"].get("final_f1_std"),
        }
    return list(found.values())


def load_baseline_cells() -> list[dict]:
    root = ROOT / "results" / "baseline_seed_attribution_20261010"
    out = []
    for path in root.glob("*.json"):
        if path.name in {"summary.json"}:
            continue
        payload = read_json(path)
        if payload.get("status") != "complete":
            continue
        p = payload["protocol"]
        for arm, name in (("gecc_hard", "GECC-hard-TAG"), ("cat_cctp", "CaT-CCTP")):
            for row in payload.get("rows", []):
                if row.get("arm") != arm:
                    continue
                out.append({
                    "dataset": p["dataset"], "order": int(row["order_seed"]),
                    "construction_seed": int(row["construction_seed"]), "method": name,
                    "final_f1": row["final"]["final_f1"], "accuracy": row["final"]["accuracy"],
                    "old_f1": row["final"].get("old_f1"), "new_f1": row["final"].get("new_f1"),
                    "source": str(path), "recovered": False, "sinkhorn_max_steps": None,
                    "student_f1_sd": row["final"].get("final_f1_std"),
                })
    return out


def load_sage() -> list[dict]:
    root = ROOT / "results" / "uq_sage_extended_backup_20261009"
    out = []
    for path in root.glob("*_o*_c*/results.json"):
        payload = read_json(path)
        p = payload["protocol"]
        for row in payload.get("rows", []):
            out.append({
                "dataset": p["dataset"], "order": int(p["order_seed"]),
                "construction_seed": int(p["construction_seed"]),
                "method": row["arm"], "final_f1": row["score"]["final_f1"],
                "accuracy": row["score"]["accuracy"],
                "old_f1": row["score"].get("old_f1"),
                "new_f1": row["score"].get("new_f1"),
                "student_f1_sd": row["score"].get("final_f1_std"),
                "student_peak_mb": row.get("student_peak_mb"),
                "student_seconds": row.get("student_seconds"),
                "source": str(path),
            })
    return out


def summarize_cells(cells: list[dict]) -> tuple[list[dict], list[dict]]:
    summary = []
    paired = []
    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in cells:
        buckets[(row["dataset"], row["method"])].append(row)
    for (dataset, method), rows in sorted(buckets.items()):
        for metric in ("final_f1", "accuracy", "old_f1", "new_f1"):
            values = [float(r[metric]) for r in rows if r.get(metric) is not None]
            lo, hi = ci95(values)
            summary.append({
                "dataset": dataset, "method": method, "cells": len(rows),
                "metric": metric, "mean": float(np.mean(values)) if values else None,
                "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "ci95_low": lo, "ci95_high": hi,
                "worst": min(values) if values else None,
                "recovered_cells": sum(bool(r.get("recovered")) for r in rows),
            })
    uq = {(r["dataset"], r["order"], r["construction_seed"]): r
          for r in cells if r["method"] == "Unified-Q-EF"}
    for baseline in ("GECC-hard-TAG", "CaT-CCTP"):
        base = {(r["dataset"], r["order"], r["construction_seed"]): r
                for r in cells if r["method"] == baseline}
        for dataset in STABILITY_DATASETS:
            keys = sorted(k for k in uq.keys() if k[0] == dataset and k in base)
            for metric in ("final_f1", "accuracy"):
                deltas = [float(uq[k][metric]) - float(base[k][metric]) for k in keys]
                lo, hi = ci95(deltas)
                worst_index = int(np.argmin(deltas)) if deltas else 0
                paired.append({
                    "dataset": dataset, "comparison": f"Unified-Q-EF minus {baseline}",
                    "metric": metric, "cells": len(deltas),
                    "mean_delta": float(np.mean(deltas)) if deltas else None,
                    "sd_delta": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0,
                    "ci95_low": lo, "ci95_high": hi,
                    "positive_cells": sum(v > 0 for v in deltas),
                    "positive_ratio": (sum(v > 0 for v in deltas) / len(deltas)) if deltas else None,
                    "worst_delta": min(deltas) if deltas else None,
                    "worst_order": keys[worst_index][1] if deltas else None,
                    "worst_seed": keys[worst_index][2] if deltas else None,
                    "recovered_in_uq": sum(bool(uq[k].get("recovered")) for k in keys),
                })
    return summary, paired


def summarize_sage(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    names = {"uq_dm_edgefree": "Unified-Q-EF", "gecc_hard": "GECC-hard-TAG",
             "cat_cctp": "CaT-CCTP", "uq_qmedoid": "Unified-Q-QMedoid"}
    clean = []
    for row in rows:
        item = dict(row)
        item["method"] = names.get(item["method"], item["method"])
        clean.append(item)
    summary = []
    for dataset, method in sorted({(r["dataset"], r["method"]) for r in clean}):
        vals = [r for r in clean if r["dataset"] == dataset and r["method"] == method]
        for metric in ("final_f1", "accuracy"):
            values = [float(r[metric]) for r in vals]
            lo, hi = ci95(values)
            summary.append({"dataset": dataset, "method": method, "metric": metric,
                            "cells": len(vals), "mean": float(np.mean(values)),
                            "sd": float(np.std(values, ddof=1)), "ci95_low": lo,
                            "ci95_high": hi, "worst": min(values)})
    paired = []
    main = {(r["dataset"], r["order"], r["construction_seed"]): r for r in clean
            if r["method"] == "Unified-Q-EF"}
    for baseline in ("GECC-hard-TAG", "CaT-CCTP"):
        other = {(r["dataset"], r["order"], r["construction_seed"]): r for r in clean
                 if r["method"] == baseline}
        for dataset in STABILITY_DATASETS:
            keys = sorted(k for k in main if k[0] == dataset and k in other)
            for metric in ("final_f1", "accuracy"):
                deltas = [float(main[k][metric]) - float(other[k][metric]) for k in keys]
                lo, hi = ci95(deltas)
                paired.append({"dataset": dataset, "comparison": f"Unified-Q-EF minus {baseline}",
                               "metric": metric, "cells": len(deltas),
                               "mean_delta": float(np.mean(deltas)),
                               "sd_delta": float(np.std(deltas, ddof=1)),
                               "ci95_low": lo, "ci95_high": hi,
                               "positive_cells": sum(v > 0 for v in deltas),
                               "positive_ratio": sum(v > 0 for v in deltas) / len(deltas),
                               "worst_delta": min(deltas)})
    return clean, summary, paired


def plot_cells(cells: list[dict], sage_summary: list[dict]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    OUT.mkdir(parents=True, exist_ok=True)
    colors = {"Unified-Q-EF": "#2f6f9f", "GECC-hard-TAG": "#8a8f98", "CaT-CCTP": "#c58c3a"}
    for metric, ylabel, filename in (("final_f1", "Final Macro-F1 (%)", "construction_f1.png"),
                                     ("accuracy", "Accuracy (%)", "construction_accuracy.png")):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=False)
        for axis, dataset in zip(axes, STABILITY_DATASETS):
            methods = list(METHODS)
            x = np.arange(len(methods))
            means, sds = [], []
            for method in methods:
                vals = [float(r[metric]) for r in cells if r["dataset"] == dataset and r["method"] == method]
                means.append(float(np.mean(vals)) if vals else float("nan"))
                sds.append(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)
            axis.bar(x, means, yerr=sds, capsize=3, color=[colors[m] for m in methods])
            axis.set_title(dataset)
            axis.set_xticks(x, ["CoTAG-Q", "GECC-hard", "CaT-CCTP"], rotation=25)
            axis.grid(axis="y", alpha=0.25)
        axes[0].set_ylabel(ylabel)
        fig.suptitle("Construction/order stability under the matched protocol")
        fig.tight_layout()
        fig.savefig(OUT / filename, dpi=180)
        plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 4.5))
    datasets = list(STABILITY_DATASETS)
    methods = ["Unified-Q-EF", "GECC-hard-TAG", "CaT-CCTP"]
    x = np.arange(len(datasets))
    width = 0.25
    for idx, method in enumerate(methods):
        vals = [next((r["mean"] for r in sage_summary if r["dataset"] == d and
                      r["method"] == method and r["metric"] == "final_f1"), float("nan"))
                for d in datasets]
        axis.bar(x + (idx - 1) * width, vals, width, label=method)
    axis.set_xticks(x, datasets)
    axis.set_ylabel("Prompt GraphSAGE Final Macro-F1 (%)")
    axis.set_title("Frozen-memory cross-architecture transfer")
    axis.legend(fontsize=8)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "sage_transfer_f1.png", dpi=180)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cells = load_uq_cells() + load_baseline_cells()
    stability = [r for r in cells if r["dataset"] in STABILITY_DATASETS]
    summary, paired = summarize_cells(stability)
    sage_rows = load_sage()
    sage_clean, sage_summary, sage_paired = summarize_sage(sage_rows)
    write_csv(OUT / "construction_cells.csv", stability)
    write_csv(OUT / "construction_summary.csv", summary)
    write_csv(OUT / "construction_paired_deltas.csv", paired)
    write_csv(OUT / "sage_cells.csv", sage_clean)
    write_csv(OUT / "sage_summary.csv", sage_summary)
    write_csv(OUT / "sage_paired_deltas.csv", sage_paired)
    plot_cells(stability, sage_summary)

    lines = [
        "# Unified-Q-EF Overnight Evidence Bundle (2026-10-12)", "",
        "This bundle only aggregates completed artifacts. It does not modify sealed results or select a method by report-dev score.",
        "", "## Protocol audit", "",
        "- Construction: current-delta-only, frozen synthetic prefix, cumulative 5% memory ratio.",
        "- Evaluation: common Prompt Student for the construction stability panel; Prompt GraphSAGE only for frozen-memory transfer.",
        "- Student seeds: 150, 151, 152; construction cells are paired by order and construction seed.",
        "- Unified-Q recovery: SportsFit/order 9103/seed 271108 uses Sinkhorn max steps 4096 after the registered 2048 run failed only a numerical tolerance check. It is counted in this audit with an explicit recovery flag and is not merged into the sealed primary table.",
        "", "## Construction stability", "",
        "| Dataset | Comparison | Metric | Cells | Mean delta | 95% CI | Positive cells | Worst delta |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in paired:
        lines.append(f"| {row['dataset']} | {row['comparison']} | {row['metric']} | {row['cells']} | {row['mean_delta']:+.2f} | [{row['ci95_low']:+.2f}, {row['ci95_high']:+.2f}] | {row['positive_cells']}/{row['cells']} | {row['worst_delta']:+.2f} |")
    lines += ["", "The paired cells quantify stability, not a universal SOTA claim. Accuracy and Macro-F1 are kept separate because GECC can have higher accuracy while Unified-Q-EF has higher Macro-F1.", "", "## Cross-architecture transfer", "", "| Dataset | Method | Cells | Final F1 | Accuracy | Worst F1 |", "|---|---|---:|---:|---:|---:|"]
    for row in sage_summary:
        if row["metric"] == "final_f1":
            acc = next(v for v in sage_summary if v["dataset"] == row["dataset"] and v["method"] == row["method"] and v["metric"] == "accuracy")
            lines.append(f"| {row['dataset']} | {row['method']} | {row['cells']} | {row['mean']:.2f} +/- {row['sd']:.2f} | {acc['mean']:.2f} +/- {acc['sd']:.2f} | {row['worst']:.2f} |")
    lines += ["", "Prompt GraphSAGE uses the same frozen memories and does not re-optimize Unified-Q. This is a transfer test, not a tuned result."]
    if (OUT / "random_q_control.md").exists():
        lines += ["", "## Literal random-Q control", "", "A reference-order literal random-Q control is complete on SportsFit, BookHis, and WikiCS. It is an attribution result and is not merged into the 3-order x 3-construction-seed stability panel. See `random_q_control.md`."]
    lines += ["", "## Files", "", "- `construction_cells.csv`, `construction_summary.csv`, `construction_paired_deltas.csv`", "- `sage_cells.csv`, `sage_summary.csv`, `sage_paired_deltas.csv`", "- `construction_f1.png`, `construction_accuracy.png`, `sage_transfer_f1.png`", "- `q_attribution_audit.md`, `q_attribution_matrix.csv`, `q_attribution_status.csv`, `random_q_control.md`", ""]
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(OUT / "manifest.json", {
        "construction_rows": len(stability), "construction_summary_rows": len(summary),
        "construction_paired_rows": len(paired), "sage_rows": len(sage_clean),
        "recovered_uq_cells": sum(bool(r.get("recovered")) for r in stability if r["method"] == "Unified-Q-EF"),
        "sealed_results_modified": False,
        "output": str(OUT),
    })
    print(json.dumps({"output": str(OUT), "construction_rows": len(stability), "sage_rows": len(sage_clean)}, ensure_ascii=True))


if __name__ == "__main__":
    main()
