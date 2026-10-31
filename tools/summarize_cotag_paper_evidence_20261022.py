"""Prepare matched paper tables from existing audited results; no model runs."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev

from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/cotag_paper_presentation_20261022"
SOURCES = {}


def source(path):
    path = ROOT / path
    data = path.read_bytes()
    SOURCES[path.relative_to(ROOT).as_posix()] = hashlib.sha256(data).hexdigest()
    return data.decode("utf-8-sig")


def rows(path):
    return list(csv.DictReader(source(path).splitlines()))


def save(name, records):
    with (OUTPUT / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def paired(values):
    average, sd = mean(values), stdev(values)
    width = t.ppf(0.975, len(values) - 1) * sd / math.sqrt(len(values))
    return dict(difference_pp=average, ci_low=average-width, ci_high=average+width,
                positive_constructions=sum(v > 0 for v in values),
                worst_difference_pp=min(values))


def main():
    OUTPUT.mkdir(parents=True, exist_ok=False)
    controls = [r for r in rows("results/uq_final_evidence_report_20261013_132047/cells.csv")
                if r["category"] == "controls" and r["split"] == "sealed_test"
                and r["arm"] in ("control_shared", "control_independent")]
    assert len(controls) == 18
    checked = []
    for row in controls:
        path = "results/cotag_shared_controls_snapshot_20261022/" + row["source"]
        raw = json.loads(source(path))
        assert SOURCES[path] == row["source_sha256"]
        protocol = raw["protocol"]
        assert protocol["dataset"] == row["dataset"]
        assert protocol["order_seed"] == 9101
        assert protocol["construction_seed"] == int(row["construction_seed"])
        assert protocol["student_seeds"] == [150, 151, 152]
        assert protocol["student_steps"] == 2048
        assert protocol["q_config"]["steps"] == 30
        stages = sorted((r for r in raw["rows"] if r["arm"] == row["arm"]),
                        key=lambda r: r["stage"])
        assert len(stages) == len(protocol["groups"])
        scores = stages[-1]["scores"]["sealed_test"]["per_student"]
        assert len(scores) == 3
        values = dict(final_f1=mean(r["final_f1"] for r in scores),
                      accuracy=mean(r["accuracy"] for r in scores),
                      construction_peak_gib=max(r["construction_peak_mb"] for r in stages)/1024,
                      construction_seconds=sum(r["construction_seconds"] for r in stages))
        for key, value in values.items():
            assert math.isclose(value, float(row[key]), abs_tol=1e-8)
        checked.append(dict(dataset=row["dataset"], arm=row["arm"],
                            construction_seed=int(row["construction_seed"]),
                            stages=len(stages), **values))
    summaries, comparisons = [], []
    for dataset in ("arxiv", "sportsfit", "bookhis"):
        arms = {}
        for arm in ("control_shared", "control_independent"):
            selected = sorted((r for r in checked if r["dataset"] == dataset and r["arm"] == arm),
                              key=lambda r: r["construction_seed"])
            assert len(selected) == 3
            assert len({r["stages"] for r in selected}) == 1
            arms[arm] = selected
            summary = dict(dataset=dataset, arm=arm, stages=selected[0]["stages"],
                           class_orders=1, construction_seeds=3, students_per_memory=3)
            for metric in ("final_f1", "accuracy", "construction_peak_gib", "construction_seconds"):
                values = [r[metric] for r in selected]
                summary[metric + "_mean"] = mean(values)
                summary[metric + "_sd"] = stdev(values)
            summaries.append(summary)
        a, b = arms["control_shared"], arms["control_independent"]
        assert [r["construction_seed"] for r in a] == [r["construction_seed"] for r in b]
        comparisons.append(dict(dataset=dataset,
            **paired([x["final_f1"]-y["final_f1"] for x, y in zip(a, b)]),
            peak_reduction_pct=100*(1-mean(r["construction_peak_gib"] for r in a)
                                      /mean(r["construction_peak_gib"] for r in b)),
            time_reduction_pct=100*(1-mean(r["construction_seconds"] for r in a)
                                      /mean(r["construction_seconds"] for r in b))))
    save("shared_q_matched_summary.csv", summaries)
    save("shared_q_matched_pairs.csv", comparisons)
    temporal = rows("results/cotag_three_concerns_20261022_verified/temporal_all.csv")
    temporal = [r for r in temporal if r["stage"] == "4" and r["mode"] == "arrived_snapshot"
                and r["split"] == "test" and r["cohort"] == "cumulative"
                and r["variant"] in ("cat_cctp", "gecc_nearest5", "uq_dm_edgefree")]
    assert len(temporal) == 9
    save("temporal_final_summary.csv", temporal)
    contrasts = rows("results/cotag_three_concerns_20261022_verified/temporal_contrasts.csv")
    contrasts = [r for r in contrasts if r["period"] == "final" and r["mode"] == "arrived_snapshot"
                 and r["cohort"] == "cumulative" and r["left"] == "uq_dm_edgefree"
                 and r["right"] in ("cat_cctp", "gecc_nearest5")]
    assert len(contrasts) == 6
    save("temporal_final_pairs.csv", contrasts)
    (OUTPUT / "verification.json").write_text(json.dumps(dict(
        controlled_runs_rechecked=len(checked), controlled_dataset_pairs=len(comparisons),
        temporal_final_entries=len(temporal), temporal_final_contrasts=len(contrasts),
        statistical_unit="construction mean after averaging three students",
        note="Existing results only. No experiments, paper edits or main-table replacement.",
        source_sha256=SOURCES), indent=2), encoding="utf-8")
    for row in summaries:
        print(f"{row['dataset']:9} {row['arm']:19} F1 {row['final_f1_mean']:.2f} +/- {row['final_f1_sd']:.2f}"
              f" Acc {row['accuracy_mean']:.2f} +/- {row['accuracy_sd']:.2f}"
              f" GPU {row['construction_peak_gib_mean']:.3f} s {row['construction_seconds_mean']:.2f}")
    for row in contrasts:
        print(f"temporal {row['dataset']} minus {row['right']}: {float(row['mean']):.3f}"
              f" [{float(row['ci_low']):.3f}, {float(row['ci_high']):.3f}]")


if __name__ == "__main__":
    main()
