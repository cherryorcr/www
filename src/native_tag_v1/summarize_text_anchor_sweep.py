#!/usr/bin/env python3
"""Aggregate the development-only Text-Anchored ASM trajectory sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def collect(root: Path) -> list[dict]:
    rows = []
    for directory in sorted(root.glob("c*_w*")):
        trajectory_path = directory / "trajectory.json"
        alignment_path = directory / "text_alignment.json"
        inputs_path = directory / "diagnostic_inputs.json"
        if not (trajectory_path.exists() and alignment_path.exists() and inputs_path.exists()):
            continue
        trajectory = load_json(trajectory_path)
        alignment = load_json(alignment_path)
        inputs = load_json(inputs_path)
        alignment_by_step = {int(row["outer_step"]): row for row in alignment["rows"]}
        seed = int(inputs["seed"])
        weight = float(trajectory["text_anchor_weight"])
        for item in trajectory["trajectory"]:
            step = int(item["outer_step"])
            metrics = alignment_by_step[step]["metrics"]
            rows.append({
                "condensation_seed": seed,
                "text_anchor_weight": weight,
                "text_anchor_scope": trajectory["text_anchor_scope"],
                "outer_step": step,
                "checkpoint_macro_f1": float(item["selected_student_mean_macro_f1"]),
                "old_stage_mean_macro_f1": float(item["selected_student_old_mean_f1"]),
                "current_stage_macro_f1": float(item["selected_student_current_f1_mean"]),
                "endpoint_macro_f1": float(item["endpoint_student_mean_macro_f1"]),
                "all_symmetric_infonce": float(metrics["all"]["symmetric_infonce"]),
                "current_symmetric_infonce": float(metrics["current_within"]["symmetric_infonce"]),
                "all_positive_cosine": float(metrics["all"]["positive_cosine_mean"]),
                "current_positive_cosine": float(metrics["current_within"]["positive_cosine_mean"]),
                "x_norm_mean": float(item["x_norm_mean"]),
                "artifact": str(directory / item["artifact"]),
                "artifact_sha256": item["artifact_sha256"],
            })
    if not rows:
        raise ValueError(f"no complete sweep directories found under {root}")
    return sorted(rows, key=lambda row: (row["text_anchor_weight"], row["outer_step"],
                                         row["condensation_seed"]))


def aggregate(rows: list[dict]) -> dict:
    by_key = {(row["condensation_seed"], row["text_anchor_weight"], row["outer_step"]): row
              for row in rows}
    seeds = sorted({row["condensation_seed"] for row in rows})
    weights = sorted({row["text_anchor_weight"] for row in rows})
    steps = sorted({row["outer_step"] for row in rows})
    if 0.0 not in weights:
        raise ValueError("a zero-weight matched baseline is required")

    metrics = ("checkpoint_macro_f1", "old_stage_mean_macro_f1", "current_stage_macro_f1",
               "endpoint_macro_f1", "all_symmetric_infonce", "current_symmetric_infonce",
               "all_positive_cosine", "current_positive_cosine", "x_norm_mean")
    cross_seed = []
    paired = []
    eligible = []
    for weight in weights:
        for step in steps:
            group = [by_key[(seed, weight, step)] for seed in seeds]
            summary = {"text_anchor_weight": weight, "outer_step": step,
                       "condensation_seeds": seeds}
            summary.update({name: mean(row[name] for row in group) for name in metrics})
            cross_seed.append(summary)
            if weight == 0.0 or step == 0:
                continue
            deltas = []
            for seed in seeds:
                current = by_key[(seed, weight, step)]
                baseline = by_key[(seed, 0.0, step)]
                delta = {"condensation_seed": seed}
                delta.update({name: current[name] - baseline[name] for name in metrics})
                deltas.append(delta)
            record = {"text_anchor_weight": weight, "outer_step": step,
                      "paired_deltas_vs_zero": deltas,
                      "mean_old_stage_delta": mean(d["old_stage_mean_macro_f1"] for d in deltas)}
            paired.append(record)
            # Post-hoc development screen: same-step gains must be consistent in
            # cumulative and current-stage F1, mean old-stage response may not fall,
            # and label-free correspondence must improve for every condensation seed.
            if (all(d["checkpoint_macro_f1"] > 0 for d in deltas)
                    and all(d["current_stage_macro_f1"] > 0 for d in deltas)
                    and record["mean_old_stage_delta"] >= 0
                    and all(d["all_symmetric_infonce"] < 0 for d in deltas)):
                eligible.append(record)

    selected = None
    if eligible:
        selected = max(eligible, key=lambda item: next(
            row["checkpoint_macro_f1"] for row in cross_seed
            if row["text_anchor_weight"] == item["text_anchor_weight"]
            and row["outer_step"] == item["outer_step"]))
    return {
        "protocol": "native_tag_v1_text_anchor_development_sweep_v1",
        "selection_data": "checkpoint/dev metrics and label-free alignment only",
        "test_or_future_used_for_selection": False,
        "candidate_screen": {
            "status": "post_hoc_development_screen",
            "requirements": [
                "positive same-step checkpoint Macro-F1 delta for every condensation seed",
                "positive same-step current-stage Macro-F1 delta for every condensation seed",
                "non-negative cross-seed mean old-stage Macro-F1 delta",
                "lower label-free all-pair symmetric InfoNCE for every condensation seed",
            ],
        },
        "seeds": seeds,
        "weights": weights,
        "outer_steps": steps,
        "rows": rows,
        "cross_seed_means": cross_seed,
        "paired_comparisons": paired,
        "eligible_candidates": eligible,
        "selected_candidate": selected,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = aggregate(collect(args.root))
    dump_json(args.output, result)
    print(json.dumps(result["selected_candidate"], ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
