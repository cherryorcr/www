"""Recompute key comparisons directly from snapshotted Student scores."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics

from scipy.stats import t


def close(actual, expected, name):
    if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-9):
        raise AssertionError(f"{name}: {actual} differs from {expected}")


def verify(snapshot, bundle):
    raw, confirmation = {}, defaultdict(dict)
    student_scores = 0
    for entry in snapshot["entries"]:
        result = entry["result"]
        if result["status"] != "complete":
            continue
        protocol = result["protocol"]
        dataset = protocol["dataset"]
        grouped = defaultdict(list)
        for row in result["rows"]:
            grouped[row["arm"]].append(row)
        for arm, rows in grouped.items():
            if arm not in {row["arm"] for row in bundle["coverage"]}:
                continue
            rows.sort(key=lambda row: row["stage"])
            if [row["stage"] for row in rows] != list(range(len(protocol["groups"]))):
                raise AssertionError("Incomplete stages in a completed source")
            for split in ("report_dev", "sealed_test"):
                key = dataset, arm, protocol["order_seed"], protocol["construction_seed"], split
                if key in raw:
                    raise AssertionError(f"Duplicate raw experimental cell: {key}")
                final = rows[-1]["scores"][split]
                students = final["per_student"]
                if sorted(row["seed"] for row in students) != [150, 151, 152]:
                    raise AssertionError(f"Student seeds differ: {key}")
                metrics = {}
                for metric in ("final_f1", "accuracy", "old_f1", "new_f1"):
                    values = [row[metric] for row in students if row[metric] is not None]
                    metrics[metric] = statistics.mean(values) if values else None
                    if values:
                        close(final[metric], metrics[metric], str(key) + metric)
                        close(final[metric + "_std"], statistics.stdev(values), str(key) + metric + " SD")
                metrics["stage_average_f1"] = statistics.mean(
                    statistics.mean(student["final_f1"] for student in row["scores"][split]["per_student"])
                    for row in rows)
                raw[key] = metrics
                student_scores += len(students)
                if "/confirmation/" in entry["path"]:
                    confirmation[dataset, arm, split][key[2:4]] = metrics

    for row in bundle["matched"]:
        key = row["dataset"], row["arm"], row["order_seed"], row["construction_seed"], row["split"]
        for metric, expected in raw[key].items():
            if expected is not None:
                close(row[metric], expected, str(key) + metric)

    paired_checks = []
    for row in bundle["paired"]:
        ours = confirmation[row["dataset"], "uq_dm_edgefree", row["split"]]
        baseline = confirmation[row["dataset"], row["comparator"], row["split"]]
        if set(ours) != set(baseline) or len(ours) != 9:
            raise AssertionError("Paired confirmation must contain the same nine cells")
        keys = sorted(ours)
        differences = [ours[key][row["metric"]] - baseline[key][row["metric"]] for key in keys]
        average = statistics.mean(differences)
        half = t.ppf(.975, len(keys)-1) * statistics.stdev(differences) / math.sqrt(len(keys))
        for field, expected in (("mean_delta", average), ("ci_low", average-half),
                                ("ci_high", average+half), ("worst_delta", min(differences))):
            close(row[field], expected, field)
        if row["positive_cells"] != sum(value > 0 for value in differences):
            raise AssertionError("Positive-cell count differs")
        if row["split"] == "sealed_test" and row["metric"] == "final_f1":
            paired_checks.append(dict(dataset=row["dataset"], comparator=row["comparator"],
                mean_delta=average, ci_low=average-half, ci_high=average+half,
                positive_cells=row["positive_cells"], cells=9))

    for row in bundle["stability"]:
        population = confirmation[row["dataset"], row["arm"], row["split"]]
        for metric in ("final_f1", "accuracy", "old_f1", "new_f1", "stage_average_f1"):
            values = [cell[metric] for cell in population.values()]
            close(row[metric + "_mean"], statistics.mean(values), metric + " mean")
            close(row[metric + "_cell_std"], statistics.stdev(values), metric + " cell SD")
        close(row["worst_f1"], min(cell["final_f1"] for cell in population.values()), "worst F1")

    return dict(status="passed", final_student_score_records=student_scores,
        matched_rows_verified=len(bundle["matched"]), paired_rows_verified=len(bundle["paired"]),
        stability_rows_verified=len(bundle["stability"]), sealed_f1_paired=paired_checks,
        scope="Arithmetic from stored Student metrics; predictions/confusion matrices are not available in this snapshot")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Verification receipts are immutable; choose a new output")
    result = verify(json.loads(args.snapshot.read_text(encoding="utf-8")),
                    json.loads(args.bundle.read_text(encoding="utf-8")))
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
