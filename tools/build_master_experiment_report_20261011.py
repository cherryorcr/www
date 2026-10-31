"""Build one auditable index over the final and supplementary experiments."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/pre_submission_evidence_20261011"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    primary = [row for row in read_csv(args.root / "pre_submission_evidence_20261010/primary_matched.csv")
               if row.get("student") == "Prompt Student"]
    sage = read_csv(args.root / "pre_submission_evidence_20261010/sage_transfer_cells.csv")
    seed = read_csv(args.root / "baseline_seed_attribution_20261010/summary/summary.csv")
    extra = read_csv(args.root / "extended_evidence_20261011/additional_dataset_summary.csv")
    lp_path = args.root / "link_prediction_20260920/summary.json"
    lp = json.loads(lp_path.read_text(encoding="utf-8")) if lp_path.exists() else {}

    lines = [
        "# Master experiment report (2026-10-11)",
        "",
        "This index separates the sealed primary comparison, construction-seed baseline confirmation, SAGE transfer, additional datasets, and the older topology-enabled link-prediction audit.",
        "",
        "## Primary matched Prompt Student",
        "",
        "| Dataset | Method | Final F1 | Accuracy | Old F1 | New F1 | Orders |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in primary:
        lines.append(f"| {row['dataset']} | {row['method']} | {float(row['final_f1']):.2f} | {float(row['accuracy']):.2f} | {float(row['old_f1']):.2f} | {float(row['new_f1']):.2f} | {row['n_orders']} |")

    lines += ["", "## Matched external adapter seed confirmation", "",
              "| Dataset | Adapter | Cells | Final F1 mean±SD | Accuracy mean±SD | Worst F1 |",
              "|---|---|---:|---:|---:|---:|"]
    for row in seed:
        lines.append(f"| {row['dataset']} | {row['method']} | {row['n']} | {float(row['final_f1_mean']):.2f}±{float(row['final_f1_sd']):.2f} | {float(row['accuracy_mean']):.2f}±{float(row['accuracy_sd']):.2f} | {float(row['final_f1_min']):.2f} |")
    if not seed:
        lines.append("| pending | pending | pending | pending | pending | pending |")

    lines += ["", "## Prompt SAGE transfer", "",
              "| Dataset | Method | Cells | Final F1 mean±SD | Accuracy mean±SD | Worst F1 |",
              "|---|---|---:|---:|---:|---:|"]
    for dataset in sorted({r["dataset"] for r in sage}):
        for method in sorted({r["method"] for r in sage if r["dataset"] == dataset}):
            rows = [r for r in sage if r["dataset"] == dataset and r["method"] == method]
            f1 = [float(r["final_f1"]) for r in rows]
            acc = [float(r["accuracy"]) for r in rows]
            lines.append(f"| {dataset} | {method} | {len(rows)} | {sum(f1)/len(f1):.2f}±{(sum((v-sum(f1)/len(f1))**2 for v in f1)/max(1,len(f1)-1))**0.5:.2f} | {sum(acc)/len(acc):.2f}±{(sum((v-sum(acc)/len(acc))**2 for v in acc)/max(1,len(acc)-1))**0.5:.2f} | {min(f1):.2f} |")

    lines += ["", "## Additional datasets", "",
              "| Dataset | Method | Orders | Final F1 mean±SD | Accuracy mean±SD |",
              "|---|---|---:|---:|---:|"]
    for row in extra:
        lines.append(f"| {row['dataset']} | {row['method']} | {row['n_orders']} | {float(row['final_f1_mean']):.2f}±{float(row['final_f1_sd']):.2f} | {float(row['accuracy_mean']):.2f}±{float(row['accuracy_sd']):.2f} |")
    if not extra:
        lines.append("| pending | pending | pending | pending | pending |")

    lines += ["", "## Link prediction scope", ""]
    if lp:
        lines += ["The existing link-prediction artifact is the topology-enabled 2026-09-20 protocol and is not merged into the edge-free Unified-Q primary claim.", "",
                  "| Dataset | C0 AUC | Joint-kernel AUC | Degree-balanced AUC | Coverage AUC |",
                  "|---|---:|---:|---:|---:|"]
        for row in lp.get("methods", []):
            if row.get("arm") == "c0":
                dataset = row["dataset"]
                vals = {r["arm"]: r for r in lp["methods"] if r["dataset"] == dataset}
                lines.append(f"| {dataset} | {float(vals['c0']['final_auc']):.2f} | {float(vals['joint_kernel']['final_auc']):.2f} | {float(vals['degree_balanced']['final_auc']):.2f} | {float(vals['coverage']['final_auc']):.2f} |")
    else:
        lines.append("No link-prediction artifact found.")

    lines += ["", "## Interpretation guardrails", "",
              "- External methods are current-delta-only adapters, not official reimplementations unless explicitly marked.",
              "- Prompt Student and Legacy Student numbers are separate; the Student main effect is not a condensation-method gain.",
              "- Additional ArXiv/Cora rows are supplementary transfer checks and are not used to tune the primary method.",
              "- The edge-free primary method does not support a direct link-prediction claim; the topology-enabled link-prediction artifact is reported as a separate scope audit.",
              ""]
    (args.output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    (args.output / "counts.json").write_text(json.dumps({
        "primary_rows": len(primary), "seed_rows": len(seed),
        "sage_cells": len(sage), "additional_rows": len(extra),
        "link_prediction_present": bool(lp),
    }, indent=2), encoding="utf-8")
    print(json.dumps({"primary_rows": len(primary), "seed_rows": len(seed),
                      "additional_rows": len(extra)}, ensure_ascii=True))


if __name__ == "__main__":
    main()
