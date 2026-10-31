"""Export audited v7-A metadata, checking every final metric against raw fits."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev


SEEDS = tuple(range(9300, 9306))
CANDIDATES = ("c0", "recluster", "multi_prototype")
METRICS = ("cumulative_f1", "old_f1", "new_f1")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_close(actual, expected, context):
    if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-8):
        raise ValueError(f"{context}: {actual} != {expected}")


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(root):
    audit_path = root / "audit_summary_reload.json"
    audit = read(audit_path)
    if audit["status"] != "passed" or len(audit["cells"]) != 8:
        raise ValueError("Expected a passed, eight-cell tensor audit")
    cells, candidates, fits = [], [], []
    input_hashes = {audit_path.name: sha(audit_path)}
    for cell in audit["cells"]:
        folder = root / cell["dataset"] / cell["order"] / f"stage{cell['stage']}"
        selection = read(folder / "selection.json")
        results = read(folder / "results.json")
        if selection["selection"] != cell["selection"] or results["status"] != "complete":
            raise ValueError(f"Selection or completion mismatch: {folder}")
        indexed = {(row["candidate"], row["seed"]): row for row in results["fits"]}
        expected = {(name, seed) for name in CANDIDATES for seed in SEEDS}
        if set(indexed) != expected or len(results["fits"]) != len(expected):
            raise ValueError(f"Duplicate or missing fits: {folder}")
        raw_by_name = {}
        cell_fits = []
        for name in CANDIDATES:
            raw = []
            for seed in SEEDS:
                path = folder / f"{name}_seed{seed}.json"
                fit = read(path)
                if fit != indexed[name, seed]:
                    raise ValueError(f"Per-fit JSON differs from results.json: {path}")
                if fit["steps"] != max(50, min(2048, 16 * cell["rows"])):
                    raise ValueError(f"Unexpected student budget: {path}")
                if len(fit["curve"]) != 4 or fit["curve"][-1]["step"] != fit["steps"]:
                    raise ValueError(f"Incomplete response curve: {path}")
                raw.append(fit["curve"][-1]["report"]["all"])
                cell_fits.append(fit)
                input_hashes[path.relative_to(root).as_posix()] = sha(path)
            raw_by_name[name] = raw
        fits.extend(cell_fits)
        for name in CANDIDATES:
            entry = cell["candidates"][name]
            values = [row["cumulative_f1"] for row in raw_by_name[name]]
            gains = [row["cumulative_f1"] - base["cumulative_f1"]
                     for row, base in zip(raw_by_name[name], raw_by_name["c0"])]
            for metric in METRICS:
                check_close(mean(row[metric] for row in raw_by_name[name]),
                            entry["report"][metric], f"{folder}/{name}/{metric}")
            for actual, saved in zip(gains, entry["gain"]["values"]):
                check_close(actual, saved, f"{folder}/{name}/paired_gain")
            check_close(mean(gains), entry["gain"]["mean"], f"{folder}/{name}/mean_gain")
            check_close(stdev(gains), entry["gain"]["seed_sd"], f"{folder}/{name}/gain_sd")
            candidates.append({
                "dataset": cell["dataset"], "order": cell["order"], "stage": cell["stage"],
                "candidate": name, "rows": cell["rows"], "seed_count": len(SEEDS),
                "f1_mean": mean(values), "f1_seed_sd": stdev(values),
                "old_f1": entry["report"]["old_f1"], "new_f1": entry["report"]["new_f1"],
                "paired_gain_pp": mean(gains), "gain_seed_sd": stdev(gains),
                "gain_descriptive_lower": entry["gain"]["descriptive_95pct_t_interval"][0],
                "gain_descriptive_upper": entry["gain"]["descriptive_95pct_t_interval"][1],
                "positive_seeds": sum(g > 0 for g in gains), "score": entry["score"],
                "coverage_gain": entry["coverage_gain"], "dynamic_distance": entry["dynamic_distance"],
                "payload_bytes": entry["payload_bytes"], "singleton_new_rows": entry["singleton_new_rows"],
                "changed_slots": entry["changed_slots"], "selected": cell["selection"]["selected"] == name,
            })
        for field, raw_field in (("diagnostic_training_seconds", "training_seconds"),
                                 ("diagnostic_evaluation_seconds", "evaluation_seconds"),
                                 ("diagnostic_optimizer_steps", "steps")):
            check_close(sum(f[raw_field] for f in cell_fits), cell[field], f"{folder}/{field}")
        cells.append({
            "dataset": cell["dataset"], "order": cell["order"], "stage": cell["stage"],
            "rows": cell["rows"], "steps_per_fit": cell_fits[0]["steps"],
            **{name + "_f1": cell["candidates"][name]["report"]["cumulative_f1"] for name in CANDIDATES},
            "selected": cell["selection"]["selected"], "selected_gain_pp": cell["gain"]["mean"],
            "spearman": cell["ranking"]["spearman"], "top1_agreement": cell["ranking"]["top1_agreement"],
            "regret_pp": cell["regret"], "geometry_gain_pp": cell["controls"]["geometry_gain"],
            "cost_ratio_to_c0": cell["cost"]["accounted_ratio_to_c0"],
            "accounted_seconds": cell["cost"]["accounted_single_use_seconds"],
            "accounted_c0_seconds": cell["cost"]["accounted_c0_seconds"],
        })
    if len(fits) != audit["fits"]:
        raise ValueError("Total fit count mismatch")
    alternatives = [row for row in candidates if row["candidate"] != "c0"]
    positives = [row for row in alternatives if row["paired_gain_pp"] > 0]
    ratios = [row["cost_ratio_to_c0"] for row in cells]
    precision = read(root / "precision_diagnostic.json")
    summary = {
        "status": "metadata_crosscheck_passed", "fits_crosschecked": len(fits),
        "candidate_rows": len(candidates), "cells": len(cells),
        "training_observation_times": sum(len(f["curve"]) for f in fits),
        "split_metric_observations": sum(2 * len(f["curve"]) for f in fits),
        "student_optimizer_steps": sum(f["steps"] for f in fits),
        "student_training_seconds_sum": sum(f["training_seconds"] for f in fits),
        "student_evaluation_seconds_sum": sum(f["evaluation_seconds"] for f in fits),
        "gradient_runs": sum(c["cost"]["gradient_runs"] for c in audit["cells"]),
        "gradient_updates": sum(c["cost"]["gradient_optimizer_updates"] for c in audit["cells"]),
        "gradient_observations": sum(c["cost"]["gradient_observations"] for c in audit["cells"]),
        "gradient_seconds_sum": sum(c["cost"]["gradient_seconds"] for c in audit["cells"]),
        "score_seconds_sum": sum(c["cost"]["score_seconds"] for c in audit["cells"]),
        "cost_ratio_min_mean_max": [min(ratios), mean(ratios), max(ratios)],
        "positive_alternative_cells": len(positives),
        "positive_with_nonpositive_coverage": sum(row["coverage_gain"] <= 0 for row in positives),
        "positive_cells_by_candidate": {
            name: sum(row["candidate"] == name for row in positives) for name in CANDIDATES[1:]
        },
        "cells_with_positive_headroom": sum(c["oracle_headroom"] > 0 for c in audit["cells"]),
        "top1_agreement": mean(row["top1_agreement"] for row in cells),
        "uniform_all_expected_gain_pp": mean(c["controls"]["uniform_all_gain"] for c in audit["cells"]),
        "uniform_non_c0_expected_gain_pp": mean(c["controls"]["uniform_non_c0_gain"] for c in audit["cells"]),
        "precision_max_gap32": max(c["max_gap32"] for c in precision["cells"]),
        "precision_max_gap64": max(c["max_gap64"] for c in precision["cells"]),
        "precision_selection_changes": sum(c["selected32"] != c["selected64"] for c in precision["cells"]),
        "decision_from_tensor_audit": audit["decision"],
        "scope": "Local JSON crosscheck only; tensor/logit/gradient reload audit was performed remotely. "
                 "Timing sums are measured components, not end-to-end wall time. "
                 "Report-dev reused; oracle and uniform expectations are diagnostics, not trained policies.",
        "input_sha256": input_hashes,
    }
    write_csv(root / "cell_metrics.csv", cells)
    write_csv(root / "candidate_metrics.csv", candidates)
    (root / "metadata_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {key: value for key, value in summary.items() if key != "input_sha256"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    print(json.dumps(summarize(parser.parse_args().root), indent=2))
