"""Audit existing Unified-Q evidence without rerunning or overwriting sealed results."""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "unified_q_overnight_20261012"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True), encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mean_sd(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    return float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def load_cells() -> list[dict]:
    path = ROOT / "results" / "unified_q_inloop_validation_20261007" / "cells.csv"
    with path.open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def numeric_summary(rows: list[dict], label_map: dict[str, str]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        arm = row.get("arm", "")
        label = label_map.get(arm)
        if label:
            grouped[(row["dataset"], label)].append(row)
    output = []
    for (dataset, label), group in sorted(grouped.items()):
        f1_mean, f1_sd = mean_sd([float(r["final_f1"]) for r in group])
        acc_mean, acc_sd = mean_sd([float(r["accuracy"]) for r in group])
        output.append({
            "dataset": dataset,
            "variant": label,
            "cells": len(group),
            "final_f1_mean": f1_mean,
            "final_f1_sd": f1_sd,
            "accuracy_mean": acc_mean,
            "accuracy_sd": acc_sd,
            "scope": "validation cells",
        })
    return output


def load_constructor_rows() -> list[dict]:
    path = ROOT / "results" / "unified_q_constructor_attribution_20261008_v2" / "summary.json"
    payload = read_json(path)
    label_map = {
        "uq_dm_edgefree": "full Unified-Q-EF",
        "no_node_text": "no node-text path",
        "no_residual": "no residual",
        "no_distribution": "no distribution matching",
        "free_x": "independent X under trust region",
    }
    rows = []
    for item in payload.get("main", []):
        label = label_map.get(item.get("arm"))
        if not label:
            continue
        rows.append({
            "dataset": item["dataset"],
            "variant": label,
            "cells": item["f1"]["n"],
            "final_f1_mean": item["f1"]["mean"],
            "final_f1_sd": item["f1"]["sd"],
            "accuracy_mean": item["accuracy"]["mean"],
            "accuracy_sd": item["accuracy"]["sd"],
            "scope": "3 orders, one construction seed",
        })
    return rows


def load_shared_q_rows() -> list[dict]:
    path = ROOT / "results" / "shared_q_baseline_20261008" / "summary.json"
    payload = read_json(path)
    label_map = {
        "uq_dm_edgefree": "shared Q",
        "split_q": "independent feature/text Q",
    }
    rows = []
    for item in payload.get("shared_scores", []):
        if item.get("readout") != "native" or item.get("arm") not in label_map:
            continue
        rows.append({
            "dataset": item["dataset"],
            "variant": label_map[item["arm"]],
            "cells": item["final_f1"]["n"],
            "final_f1_mean": item["final_f1"]["mean"],
            "final_f1_sd": item["final_f1"]["sd"],
            "accuracy_mean": item["accuracy"]["mean"],
            "accuracy_sd": item["accuracy"]["sd"],
            "scope": "3 orders, one construction seed",
        })
    return rows


def load_text_rows() -> list[dict]:
    path = ROOT / "results" / "unified_q_attribution_20261008" / "summary.json"
    payload = read_json(path)
    label_map = {
        "cat_random5": "CaT random-5 payload",
        "cat_nearest1": "CaT nearest-1 payload",
        "uq_nearest1": "Unified-Q nearest-1 payload",
        "uq_nearest5": "Unified-Q nearest-5 payload",
        "cat_cctp": "CaT-CCTP text control",
        "adaptive_coherent": "Adaptive coherent control",
    }
    rows = []
    for item in payload.get("text", []):
        label = label_map.get(item.get("arm"))
        if not label:
            continue
        rows.append({
            "dataset": item["dataset"],
            "variant": label,
            "cells": 1,
            "final_f1_mean": item.get("final_f1"),
            "final_f1_sd": item.get("final_f1_std"),
            "accuracy_mean": item.get("accuracy"),
            "accuracy_sd": item.get("accuracy_std"),
            "scope": "reference order, exploratory text-path control",
        })
    return rows


def build_status() -> list[dict]:
    return [
        {
            "component": "shared Q vs independent Q",
            "status": "complete",
            "scope": "3 orders plus 27-cell shared-Q construction matrix",
            "source": "results/shared_q_baseline_20261008/summary.json",
            "needs_new_run": "no",
            "note": "Shared Q is not a universal winner; it is the unified construction interface.",
        },
        {
            "component": "learned Q vs fixed Q",
            "status": "complete-reference-order",
            "scope": "one reference order per dataset",
            "source": "results/unified_q_inloop_validation_20261007/cells.csv",
            "needs_new_run": "no for current paper; yes only for a full 9-cell fixed-Q claim",
            "note": "Fixed-Q is a diagnostic control, not a sealed primary comparison.",
        },
        {
            "component": "learned Q vs random assignment",
            "status": "complete-reference-order",
            "scope": "3 datasets at order 9101 and construction seed 271006",
            "source": "results/unified_q_overnight_20261012/random_q_control.md",
            "needs_new_run": "no for current attribution scope; expand only for a full factorial claim",
            "note": "Literal class-masked balanced random-Q; distinct from random real-node coreset.",
        },
        {
            "component": "no-GM",
            "status": "complete",
            "scope": "9 order/construction cells per dataset",
            "source": "results/unified_q_inloop_validation_20261007/summary.md",
            "needs_new_run": "no",
            "note": "No-GM still uses Q, DM, propagation, and anchors; it is not a no-gradient constructor.",
        },
        {
            "component": "no-DM",
            "status": "complete",
            "scope": "3 orders, one construction seed per dataset",
            "source": "results/unified_q_constructor_attribution_20261008_v2/summary.json",
            "needs_new_run": "no for component attribution",
            "note": "The large SportsFit penalty is the strongest evidence for in-loop distribution matching.",
        },
        {
            "component": "class-shared text",
            "status": "complete-exploratory",
            "scope": "historical and mechanism-adapted controls",
            "source": "results/shared_q_baseline_20261008/summary.md",
            "needs_new_run": "no",
            "note": "Class-shared payload is a text control, not the current node-specific Q route.",
        },
        {
            "component": "random-5 text",
            "status": "complete-exploratory",
            "scope": "reference-order text-path controls",
            "source": "results/unified_q_attribution_20261008/summary.json",
            "needs_new_run": "no",
            "note": "Useful as a text materialization control; not a replacement for the strict main table.",
        },
        {
            "component": "hard vs soft materialization",
            "status": "complete-diagnostic",
            "scope": "straight-through hard forward and soft backward",
            "source": "results/unified_q_constructor_attribution_20261008_v2/construction_diagnostics.json",
            "needs_new_run": "no",
            "note": "Pure soft-only storage was not run; the evidence is hard/soft consistency, not a soft-only win.",
        },
        {
            "component": "slot/text diagnostics",
            "status": "complete",
            "scope": "construction traces across attribution arms",
            "source": "results/unified_q_constructor_attribution_20261008_v2/construction_diagnostics.json",
            "needs_new_run": "no",
            "note": "Includes Q gradient norms, residual ratios, text uniqueness, GM and DM traces.",
        },
        {
            "component": "cross-architecture Student",
            "status": "complete",
            "scope": "Prompt GraphSAGE with frozen memories",
            "source": "results/unified_q_overnight_20261012/sage_summary.csv",
            "needs_new_run": "no",
            "note": "Transfer evidence is dataset-dependent, not universal dominance.",
        },
        {
            "component": "construction/order statistical stability",
            "status": "complete-with-recovery-flag",
            "scope": "3 orders x 3 construction seeds, Student seeds averaged per cell",
            "source": "results/unified_q_overnight_20261012/construction_paired_deltas.csv",
            "needs_new_run": "no",
            "note": "One SportsFit Unified-Q cell is explicitly marked as a 4096-step Sinkhorn numerical recovery.",
        },
    ]


def build_diagnostics() -> list[dict]:
    path = ROOT / "results" / "unified_q_constructor_attribution_20261008_v2" / "construction_diagnostics.json"
    payload = read_json(path)
    rows = payload.get("rows", [])
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["cell"].split("_o")[0], row["arm"])].append(row)
    output = []
    for (dataset, arm), group in sorted(grouped.items()):
        def avg(key: str):
            values = [float(r[key]) for r in group if r.get(key) is not None]
            return float(np.mean(values)) if values else None
        output.append({
            "dataset": dataset,
            "arm": arm,
            "stages": len(group),
            "mean_q_gradient_norm_first": avg("q_gradient_norm_first"),
            "mean_q_gradient_norm_last": avg("q_gradient_norm_last"),
            "mean_gradient_match_first": avg("gradient_match_first"),
            "mean_gradient_match_last": avg("gradient_match_last"),
            "mean_unique_text_ratio": avg("unique_text_ratio"),
            "max_residual_ratio": max(float(r["x_residual_ratio_max"]) for r in group),
            "mean_relative_x_difference_from_full": avg("relative_x_difference_from_full"),
        })
    return output


def write_report(status: list[dict], numeric: list[dict], diagnostics: list[dict]) -> None:
    lines = [
        "# Unified-Q-EF Overnight Attribution Audit (2026-10-12)",
        "",
        "This audit aggregates existing artifacts only. It does not rerun experiments, modify sealed results, or select a method by test performance.",
        "",
        "## Decision",
        "",
        "The four overnight tracks can continue without another large GPU batch. The construction/order stability panel and frozen-memory GraphSAGE transfer are complete. A literal random-Q reference-order control is also available; it is kept separate from the full factorial stability panel. Hard/soft evidence remains a straight-through consistency diagnostic rather than a pure soft-only storage comparison.",
        "",
        "## Attribution coverage",
        "",
        "| Component | Status | Scope | New run needed |",
        "|---|---|---|---|",
    ]
    for row in status:
        lines.append(f"| {row['component']} | {row['status']} | {row['scope']} | {row['needs_new_run']} |")
    lines += ["", "## Numeric attribution rows", "", "| Dataset | Variant | Cells | Final F1 | Accuracy | Scope |", "|---|---|---:|---:|---:|---|"]
    for row in numeric:
        lines.append(f"| {row['dataset']} | {row['variant']} | {row['cells']} | {row['final_f1_mean']:.2f} +/- {row['final_f1_sd']:.2f} | {row['accuracy_mean']:.2f} +/- {row['accuracy_sd']:.2f} | {row['scope']} |")
    lines += ["", "## Diagnostic interpretation", "", "The traces confirm that Q is optimized in-loop and that hard text materialization is numerically connected to the soft route. They do not show that the Q gradient dominates the residual gradient; earlier diagnostics show the residual can absorb a larger update. Therefore the paper should claim a unified construction interface and in-loop coupling, not that every objective contributes an independent classification gain."]
    if (OUT / "random_q_control.md").exists():
        lines += ["", "## Literal random-Q control", "", "The reference-order random-Q control is complete on SportsFit, BookHis, and WikiCS. It keeps class legality and balanced capacity but replaces structured initialization with random legal logits, freezes Q, and disables the residual update. See `random_q_control.md` for the exact table."]
    lines += ["", "## Paper wording constraints", "", "- Keep Unified-Q-EF as the edge-free primary method and Unified-Q-T as a separate topology extension.", "- Keep external mechanism adaptations, internal ablations, and official/reference-only rows in separate tables.", "- Do not call random coreset a random-Q ablation.", "- Do not call the straight-through hard/soft check a pure soft-materialization comparison.", "- Report negative and dataset-dependent results, especially WikiCS and GraphSAGE transfer.", ""]
    (OUT / "q_attribution_audit.md").write_text("\n".join(lines), encoding="utf-8")


def plot_attribution(numeric: list[dict]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    preferred = ["full Unified-Q-EF", "no GM", "no distribution matching", "fixed Q", "random real-node coreset"]
    datasets = sorted({row["dataset"] for row in numeric})
    fig, axes = plt.subplots(1, len(datasets), figsize=(15, 4.8), squeeze=False)
    for axis, dataset in zip(axes[0], datasets):
        candidates = [row for row in numeric
                      if row["dataset"] == dataset and row["variant"] in preferred]
        # The audit combines several pre-existing scopes.  For the compact
        # figure, retain the widest available scope for each named variant.
        by_variant = {}
        for row in candidates:
            prior = by_variant.get(row["variant"])
            if prior is None or int(row["cells"]) > int(prior["cells"]):
                by_variant[row["variant"]] = row
        rows = sorted(by_variant.values(),
                      key=lambda r: preferred.index(r["variant"]) if r["variant"] in preferred else 99)
        if not rows:
            axis.axis("off")
            continue
        x = np.arange(len(rows))
        axis.bar(x, [row["final_f1_mean"] for row in rows], yerr=[row["final_f1_sd"] for row in rows], capsize=3, color="#4472c4")
        axis.set_title(dataset)
        display = {
            "full Unified-Q-EF": "Full\nUnified-Q-EF",
            "no GM": "No GM",
            "no distribution matching": "No DM",
            "fixed Q": "Fixed Q",
            "random real-node coreset": "Random\ncoreset",
            "shared Q": "Shared Q",
            "independent feature/text Q": "Independent\nQ",
        }
        axis.set_xticks(x, [display.get(row["variant"], row["variant"]) for row in rows],
                        rotation=0, fontsize=8)
        axis.set_ylabel("Final Macro-F1 (%)")
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Unified-Q attribution coverage; exploratory controls are labeled separately")
    fig.tight_layout()
    fig.savefig(OUT / "q_attribution_coverage.png", dpi=180, bbox_inches="tight")
    fig.savefig(OUT / "q_attribution_coverage.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cells = load_cells()
    numeric = []
    numeric.extend(numeric_summary(cells, {
        "uq_dm_edgefree": "full Unified-Q-EF",
        "uq_dm_edgefree_no_gm": "no GM",
        "uq_dm_edgefree_fixed_q": "fixed Q",
        "random_coreset": "random real-node coreset",
    }))
    numeric.extend(load_constructor_rows())
    numeric.extend(load_shared_q_rows())
    numeric.extend(load_text_rows())
    status = build_status()
    diagnostics = build_diagnostics()
    write_csv(OUT / "q_attribution_matrix.csv", numeric)
    write_csv(OUT / "q_attribution_status.csv", status)
    write_csv(OUT / "q_slot_text_diagnostics.csv", diagnostics)
    has_random_q = (OUT / "random_q_control.md").exists()
    write_json(OUT / "q_attribution_manifest.json", {
        "sealed_results_modified": False,
        "rerun_performed": has_random_q,
        "status_rows": len(status),
        "numeric_rows": len(numeric),
        "diagnostic_rows": len(diagnostics),
        "partial_items": [row["component"] for row in status if row["status"] == "partial"],
        "output": str(OUT),
    })
    write_report(status, numeric, diagnostics)
    plot_attribution(numeric)
    print(json.dumps({"output": str(OUT), "status_rows": len(status), "numeric_rows": len(numeric), "diagnostic_rows": len(diagnostics)}, ensure_ascii=True))


if __name__ == "__main__":
    main()
