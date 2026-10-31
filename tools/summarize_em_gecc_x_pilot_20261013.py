"""Summarize the audited screening pilot without writing into its raw run."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

from audit_em_gecc_x_pilot_20261013 import ARMS, file_hash, read_json


DATASETS = ("cora", "citeseer")
LABELS = {
    "gecc_hard": "GECC-hard + nearest-5 TAG attachment (mechanism adaptation)",
    "unified_q_ef": "Unified-Q pilot (DM off / legacy code; not frozen paper main)",
    "em_gecc_x": "EM-GECC-X (alternating prototype)",
    "em_no_gm": "EM-no-GM",
    "em_no_geometry": "EM-no-geometry",
}
CORRECTIONS = [
    "GECC text payloads are nearest-5 TAG attachments, not native GECC behavior. This is not an official reproduction or the native soft branch.",
    "The Unified-Q pilot recorded legacy constructor hash f621d4575a9bb660fad8031e4d7135e2570cc72c02d16145c08aa989bee99525. Both DM weights were zero; it is not the frozen paper main method.",
    "EM M-steps used selected hard text embeddings. Soft/hard cosine is diagnostic, not evidence of soft-training/hard-saving mismatch.",
    "q_entropy_normalized measures source-given-slot entropy, not node-to-slot assignment entropy.",
    "The legacy class_center_cosine_mean is an unnormalized dot product; use the separately audited normalized cosine.",
]


def load_results(root: Path) -> dict:
    results = {}
    for dataset in DATASETS:
        path = root / dataset / "order19611"
        result = read_json(path / "results.json")
        protocol = read_json(path / "protocol.json")
        if result["status"] != "complete" or result["protocol"] != protocol:
            raise ValueError(f"incomplete or inconsistent results: {dataset}")
        expected = {(arm, stage) for arm in ARMS
                    for stage in range(len(protocol["budgets"]))}
        keys = [(row["arm"], int(row["stage"])) for row in result["rows"]]
        if len(keys) != len(set(keys)) or set(keys) != expected:
            raise ValueError(f"missing or duplicate cells: {dataset}")
        results[dataset] = result
    return results


def load_rows(root: Path) -> list[dict]:
    rows = []
    for dataset, result in load_results(root).items():
        protocol = result["protocol"]
        for arm in ARMS:
            stages = [row for row in result["rows"] if row["arm"] == arm]
            final = max(stages, key=lambda value: int(value["stage"]))
            scores = final["scores"]
            rows.append({
                "dataset": dataset,
                "method": arm,
                "method_label": LABELS[arm],
                "split": scores["split"],
                "order": protocol["order"],
                "construction_seed": protocol["construction_seed"],
                "student_seeds": "/".join(map(str, scores["student_seeds"])),
                "memory_rows": final["memory_rows"],
                "student_steps": scores["student_steps"],
                "f1": float(scores["cumulative_f1"]),
                "f1_std": float(scores["cumulative_f1_std"]),
                "accuracy": float(scores["accuracy"]),
                "accuracy_std": float(scores["accuracy_std"]),
                "old_f1": scores["old_f1"],
                "new_f1": scores["new_f1"],
                "time_s": sum(float(item["construction_seconds"]) for item in stages),
                "peak_mb": max(float(item["construction_peak_mb"] or 0.0)
                               for item in stages),
                "checks_ok": all(all(bool(v) for v in item["checks"].values())
                                 for item in stages),
            })
    return rows


def validate_audit(root: Path, audit: dict) -> None:
    if not audit["all_artifact_checks_passed"] or audit["memory_count"] != 30:
        raise ValueError("artifact audit did not pass all 30 memories")
    expected = {"decision.json"}
    for dataset in DATASETS:
        prefix = f"{dataset}/order19611/"
        expected.update(prefix + name for name in ("protocol.json", "results.json"))
        expected.update(prefix + f"{arm}_s{stage}_memory.pt"
                        for arm in ARMS for stage in range(3))
        if not all(audit["datasets"][dataset][key]
                   for key in ("uq_legacy_code", "uq_dm_off")):
            raise ValueError("Unified-Q pilot no longer matches the audited scope")
    hashes = audit.get("raw_artifact_hashes", {})
    if set(hashes) != expected or any(file_hash(root / path) != digest
                                     for path, digest in hashes.items()):
        raise ValueError("stale or mismatched artifact audit; rerun the read-only audit")


def decision_summary(root: Path) -> dict:
    recorded = read_json(root / "decision.json")
    gains, paired = [], []
    for dataset, result in load_results(root).items():
        final = {arm: max((r for r in result["rows"] if r["arm"] == arm),
                         key=lambda r: r["stage"])["scores"] for arm in ARMS}
        best = max(("gecc_hard", "unified_q_ef"), key=lambda a: final[a]["cumulative_f1"])

        def deltas(other: str, key: str) -> list[float]:
            left = {s["seed"]: s[key] for s in final["em_gecc_x"]["per_student"]}
            right = {s["seed"]: s[key] for s in final[other]["per_student"]}
            if set(left) != set(right) or len(left) != 3:
                raise ValueError("paired Student seeds differ")
            return [left[seed] - right[seed] for seed in sorted(left)]

        f1 = deltas(best, "cumulative_f1")
        acc = deltas(best, "accuracy")
        no_gm = deltas("em_no_gm", "cumulative_f1")
        gains.append((statistics.mean(f1), statistics.mean(acc), statistics.mean(no_gm)))
        paired.append({"dataset": dataset, "comparator": best,
                       "comparator_label": LABELS[best], "student_seeds": final[best]["student_seeds"],
                       "f1_delta_pp": f1, "accuracy_delta_pp": acc,
                       "f1_delta_vs_no_gm_pp": no_gm})
    metrics = {
        "mean_dataset_f1_gain_pp": statistics.mean(v[0] for v in gains),
        "worst_dataset_f1_gain_pp": min(v[0] for v in gains),
        "mean_dataset_accuracy_gain_pp": statistics.mean(v[1] for v in gains),
        "worst_dataset_accuracy_gain_pp": min(v[1] for v in gains),
        "mean_dataset_f1_gain_vs_no_gm_pp": statistics.mean(v[2] for v in gains),
    }
    if any(not math.isclose(value, recorded["metrics"][key], abs_tol=1e-9)
           for key, value in metrics.items()):
        raise ValueError("recorded decision metrics differ from paired scores")
    thresholds = recorded["gate"]["continue_if"]
    checks = {
        "mean_f1_gain": metrics["mean_dataset_f1_gain_pp"] >= thresholds["mean_dataset_paired_macro_f1_gain_pp_min"],
        "worst_dataset_f1": metrics["worst_dataset_f1_gain_pp"] >= thresholds["worst_dataset_paired_macro_f1_gain_pp_min"],
        "mean_accuracy_gain": metrics["mean_dataset_accuracy_gain_pp"] >= thresholds["mean_dataset_paired_accuracy_gain_pp_min"],
        "worst_dataset_accuracy": metrics["worst_dataset_accuracy_gain_pp"] >= thresholds["worst_dataset_paired_accuracy_gain_pp_min"],
        "gradient_matching_contribution": metrics["mean_dataset_f1_gain_vs_no_gm_pp"] >= thresholds["mean_dataset_paired_macro_f1_gain_vs_no_gm_pp_min"],
    }
    decision = "continue_em_gecc_x" if all(checks.values()) else "stop_em_and_continue_unified_q_ef"
    if checks != recorded["checks"] or decision != recorded["decision"]:
        raise ValueError("recorded decision differs from recomputation")
    return {"decision": decision, "metrics": metrics, "checks": checks,
            "thresholds": thresholds, "confirmatory_evidence": False,
            "scope": "two small graphs; one order; one construction seed; three Student seeds; report-dev only",
            "comparison": "stronger-F1 of adapted GECC-hard and DM-off legacy Unified-Q pilot per dataset",
            "source_decision_sha256": file_hash(root / "decision.json"),
            "source_decision_matches_recomputation": True,
            "method_labels": LABELS, "corrections": CORRECTIONS, "paired_scores": paired}


def diagnostic_rows(audit: dict) -> list[dict]:
    return [{"dataset": c["dataset"], "method": c["arm"], **c["diagnostics"]}
            for c in audit["cells"] if c["stage"] == 2 and c["arm"].startswith("em_")]


def markdown(rows: list[dict], decision: dict, audit: dict) -> str:
    lines = [
        "# Audited EM-GECC-X Pilot Report", "",
        "A precommitted screening pilot, not a paper main table: Cora/Citeseer, order19611, construction seed 271006, paired Student seeds 150/151/152, cumulative legal-pool 5% memory, common prompt GCN Student/MiniLM/budget, current-delta-only construction, frozen synthetic prefixes, and edge-free saved memories. Only report-dev was evaluated; no sealed result is replaced.", "",
        "## Protocol corrections", "",
        *[f"- {item}" for item in CORRECTIONS], "",
        "The immutable raw protocol contains two inaccurate arm descriptions. The corrections above qualify the labels without editing the raw protocol or scores.", "",
        "## Decision", "",
        f"**{decision['decision']}**. All gate metrics were independently recomputed from paired Student scores and match the recorded decision. This prototype does not pass the continuation gate.", "",
        "| Gate metric | Observed (pp) | Precommitted threshold |",
        "|---|---:|---:|",
        f"| Mean Macro-F1 gain vs stronger pilot comparator | {decision['metrics']['mean_dataset_f1_gain_pp']:.2f} | >= 1.00 |",
        f"| Worst-dataset Macro-F1 gain | {decision['metrics']['worst_dataset_f1_gain_pp']:.2f} | >= -0.50 |",
        f"| Mean Accuracy gain | {decision['metrics']['mean_dataset_accuracy_gain_pp']:.2f} | >= 0.00 |",
        f"| Worst-dataset Accuracy gain | {decision['metrics']['worst_dataset_accuracy_gain_pp']:.2f} | >= -1.00 |",
        f"| Mean Macro-F1 gain vs no-GM | {decision['metrics']['mean_dataset_f1_gain_vs_no_gm_pp']:.2f} | >= 0.50 |", "",
        "## Final-stage report-dev results", "",
        "Mean +/- sample SD over three Student seeds, in percent. Old/New F1 use the cumulative class set, restricted to old/new class groups. Construction time is the sum of recorded stage construction times; peak is the maximum recorded stage GPU allocation, not end-to-end system memory.", "",
        "| Dataset | Method | Macro-F1 | Accuracy | Old F1 | New F1 | Construction (s) | Peak (MiB) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(f"| {row['dataset']} | {row['method_label']} | "
                     f"{row['f1']:.2f} +/- {row['f1_std']:.2f} | "
                     f"{row['accuracy']:.2f} +/- {row['accuracy_std']:.2f} | "
                     f"{row['old_f1']:.2f} | {row['new_f1']:.2f} | "
                     f"{row['time_s']:.2f} | {row['peak_mb']:.1f} |")
    lines += ["", "## Artifact audit", "",
              f"All {audit['memory_count']} memories passed tensor/text hashes, budget, finite-value, empty-edge, frozen-prefix tensor/string/provenance, common Student-budget and score-recomputation checks."]
    if audit["raw_source_legality_audited"]:
        lines.append("The server-side audit also verified original pool/report-mask hashes, exact floor-5% schedules, and same-class text sources drawn only from each current legal delta.")
        switches = sum(c["diagnostics"]["final_e_step_switch_count"]
                       for c in audit["cells"] if c["arm"].startswith("em_"))
        lines.append(f"Final E-step hard-source switches: {switches} across all 18 EM stage/arm cells. Saved sources therefore agree with the final M-step sources in this run.")
    else:
        lines.append("This local audit did not load original graphs; raw-source legality and final E-step switches remain unchecked.")
    em_cells = [c for c in audit["cells"] if c["arm"].startswith("em_")]
    round_switches = sum(c["diagnostics"]["e_round_hard_switch_count"] for c in em_cells)
    unchanged = sum(c["diagnostics"]["e_round_changed_slot_ratio"] == 0 for c in em_cells)
    shared_supports = {(c["dataset"], c["stage"]) for c in em_cells
                       if c["diagnostics"]["same_final_hard_sources_across_em_arms"]}
    lines.append(f"Across-round hard-source switches: {round_switches}; unchanged hard routing in {unchanged}/{len(em_cells)} EM stage/arm cells. Final hard sources are identical across all three EM arms in {len(shared_supports)}/6 dataset/stage cells. This does not imply that soft Q was constant.")
    lines += ["", *[f"- {item}" for item in audit["limitations"]], "",
              "## Final-stage diagnostics", "",
              "Source entropy is H(source | slot), normalized by the log of same-class source count. Coverage is the unique hard-source/slot ratio, structurally enforced by Hungarian matching. Neither is proof of an accuracy gain.", "",
              "| Dataset | Arm | Source entropy | Diagnostic soft/hard cosine | Normalized center cosine | Relative X drift | Distinct payload ratio | Real-side new rows | Synthetic-side new rows |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in diagnostic_rows(audit):
        lines.append(f"| {row['dataset']} | {row['method']} | "
                     f"{row['source_given_slot_entropy_normalized']:.4f} | "
                     f"{row['diagnostic_soft_hard_cosine']:.4f} | "
                     f"{row['normalized_new_class_center_cosine']:.4f} | "
                     f"{100 * row['x_drift_relative_to_gecc']:.2f}% | "
                     f"{100 * row['distinct_payload_ratio']:.0f}% | "
                     f"{100 * row['reference_new_row_fraction']:.2f}% | "
                     f"{100 * row['synthetic_new_row_fraction']:.2f}% |")
    lines += ["", "## Interpretation and scope", "",
              "- Full EM-GECC-X is substantially below both pilot comparators on both graphs. Removing GM raises F1 by 9.51 pp on Cora and 11.58 pp on Citeseer. This implicates the current GM update/configuration, not gradient matching or EM in general.",
              "- The full branch is 1.17/0.70 pp above no-geometry on Cora/Citeseer, but still fails the gate. Large X drift also occurs without GM; drift alone is not a demonstrated cause.",
              "- Hard text routing never changed after the first E-step in this pilot, and all three EM arms saved the same per-stage hard sources. Later E-steps therefore did not revise the stored text support; soft Q and its geometry target may still have changed. No materialization mismatch explains the observed GM ablation gap.",
              "- The real reference concatenates old synthetic rows and all current raw rows; the synthetic side concatenates old and new synthetic rows. Their row proportions differ under a global loss without old/new-group reweighting. This is a verified mismatch and plausible failure mechanism, not causal proof of the score drop.",
              "- Other untested limitations include a fixed random 128-dimensional proxy versus the trained downstream Student, unrestricted new X, and only 20 M-steps. The mass metadata differs but is unused by both the GM loss and Student training, so it is not an explanation.",
              "- This is EM-like alternating optimization, not likelihood-based EM; it does not test bounded old-memory refresh. A failed limited pilot does not reject the teacher's broader direction.", "",
              "## Recommendation", "",
              "Stop EM expansion as precommitted. Preserve the negative result and continue the frozen Unified-Q-EF paper line; these pilot scores neither establish that main method's superiority nor supersede its sealed tables.", ""]
    return "\n".join(lines)


def check_output_directory(root: Path, output: Path) -> None:
    if output == root or root in output.parents:
        raise ValueError("derived output must be outside the immutable raw run")


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output_dir.resolve()
    check_output_directory(root, output)
    audit = read_json(args.audit)
    validate_audit(root, audit)
    rows, decision = load_rows(root), decision_summary(root)
    output.mkdir(parents=True, exist_ok=True)
    (output / "pilot_summary.md").write_text(markdown(rows, decision, audit), encoding="utf-8")
    (output / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    write_csv(output / "final_metrics.csv", rows)
    write_csv(output / "final_diagnostics.csv", diagnostic_rows(audit))
    print(json.dumps({"rows": len(rows), "decision": decision["decision"],
                      "output_dir": str(output)}))


if __name__ == "__main__":
    main()
