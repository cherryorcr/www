"""Reaggregate completed mechanism, fidelity and chronological evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[2]
SOURCES = {}


def csv(path):
    path = ROOT / path
    SOURCES[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return pd.read_csv(path)


def read(path):
    SOURCES[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text(encoding="utf-8"))


def describe(values):
    a = np.asarray(values, float)
    mean, sd = a.mean(), a.std(ddof=1)
    width = t.ppf(.975, len(a) - 1) * sd / np.sqrt(len(a))
    return dict(mean=float(mean), sd=float(sd), n=len(a), ci_low=float(mean-width), ci_high=float(mean+width),
                positive=int((a > 0).sum()), worst=float(a.min()), values=a.tolist())


def main(output):
    output.mkdir(parents=True, exist_ok=False)
    free = csv("experiments/cotag_q_free_x_20261020/analysis/student_scores.csv")
    free = free[free.split == "sealed_test"]
    means = free.groupby(["dataset", "architecture", "arm", "construction_seed"])["final_f1"].mean()
    direct = []
    for dataset in ("arxiv", "sportsfit", "bookhis"):
        for arch in ("GCN", "GraphSAGE"):
            g = means.loc[(dataset, arch)]
            if "independent_x" not in g.index.get_level_values(0):
                continue
            a, b = g.loc["uq_dm_edgefree"], g.loc["independent_x"]
            direct.append(dict(dataset=dataset, architecture=arch, full=float(a.mean()), control=float(b.mean()),
                               **describe(a-b)))
    pd.DataFrame(direct).to_csv(output / "direct_x.csv", index=False)

    ablations = csv("results/uq_component_ablations_20261014/report_utc_20261013_201811_267213/cells.csv")
    ablations = ablations[(ablations.split == "sealed_test") & (ablations.order_seed == 9101) &
                          ablations.arm.isin(["uq_dm_edgefree", "uq_fixed_q", "uq_no_dm", "uq_no_gm"])]
    ablation_summary, ablation_pairs = [], []
    for dataset, frame in ablations.groupby("dataset"):
        full = frame[frame.arm == "uq_dm_edgefree"].set_index("construction_seed")
        for arm, sub in frame.groupby("arm"):
            if len(sub) != 3:
                continue
            ablation_summary.append(dict(dataset=dataset, arm=arm, f1=float(sub.final_f1.mean()),
                                         accuracy=float(sub.accuracy.mean()), f1_sd=float(sub.final_f1.std(ddof=1))))
            if arm != "uq_dm_edgefree":
                ablation_pairs.append(dict(dataset=dataset, arm=arm,
                                            **describe(full.final_f1 - sub.set_index("construction_seed").final_f1)))
    pd.DataFrame(ablation_summary).to_csv(output / "ablations.csv", index=False)
    pd.DataFrame(ablation_pairs).to_csv(output / "ablation_contrasts.csv", index=False)

    controls = csv("results/uq_final_evidence_report_20261013_132047/cells.csv")
    controls = controls[(controls.category == "controls") & (controls.split == "sealed_test") &
                        controls.arm.isin(["control_shared", "control_independent"])]
    checked_controls = 0
    for row in controls.itertuples():
        path = ROOT / "results/cotag_shared_controls_snapshot_20261022" / row.source
        raw = read(path)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row.source_sha256
        records = [r for r in raw["rows"] if r["arm"] == row.arm]
        last = sorted(records, key=lambda r: r["stage"])[-1]
        f1 = np.mean([r["final_f1"] for r in last["scores"]["sealed_test"]["per_student"]])
        peak = max(r["construction_peak_mb"] for r in records) / 1024
        seconds = sum(r["construction_seconds"] for r in records)
        assert abs(f1-row.final_f1) < 1e-8 and abs(peak-row.construction_peak_gib) < 1e-8
        assert abs(seconds-row.construction_seconds) < 1e-8
        checked_controls += 1
    resources = []
    for dataset, frame in controls.groupby("dataset"):
        a = frame[frame.arm == "control_shared"].set_index("construction_seed")
        b = frame[frame.arm == "control_independent"].set_index("construction_seed")
        resources.append(dict(dataset=dataset,
            shared_peak=float(a.construction_peak_gib.mean()), independent_peak=float(b.construction_peak_gib.mean()),
            shared_seconds=float(a.construction_seconds.mean()), independent_seconds=float(b.construction_seconds.mean()),
            peak_reduction_pct=float(100*(1-a.construction_peak_gib.mean()/b.construction_peak_gib.mean())),
            time_reduction_pct=float(100*(1-a.construction_seconds.mean()/b.construction_seconds.mean())),
            **describe(a.final_f1-b.final_f1)))
    pd.DataFrame(resources).to_csv(output / "shared_assignment.csv", index=False)

    text = csv("results/cotag_q_four_gap_20261015_v1/report_final_20261015T190000/text_students.csv")
    text = text[text.split == "sealed_test"].groupby(["dataset", "arm", "construction_seed"]).final_f1.mean()
    text_rows = []
    for dataset in text.index.get_level_values(0).unique():
        g = text.loc[dataset]
        for control in ("no_text_init", "no_pair_feedback", "both_off"):
            a, b = g.loc["full"], g.loc[control]
            text_rows.append(dict(dataset=dataset, control=control, full=float(a.mean()), control_f1=float(b.mean()), **describe(a-b)))
    pd.DataFrame(text_rows).to_csv(output / "text_paths.csv", index=False)

    temporal_path = "results/cotag_gecc_single_text_temporal_20261022/analysis/v1/"
    temporal = csv(temporal_path + "per_construction.csv")
    source_summary = csv(temporal_path + "summary.csv")
    keys = ["dataset", "variant", "stage", "mode", "split", "cohort"]
    aggregation = temporal.groupby(keys).agg(f1=("f1", "mean"), f1_sd=("f1", "std"),
                                            accuracy=("accuracy", "mean"), accuracy_sd=("accuracy", "std"))
    joined = aggregation.join(source_summary.set_index(keys), lsuffix="_recalc", how="inner")
    assert len(joined) == len(aggregation)
    assert np.allclose(joined.f1_recalc, joined.f1, atol=1e-10)
    assert np.allclose(joined.f1_sd, joined.f1_construction_sd, atol=1e-10)
    assert np.allclose(joined.accuracy_recalc, joined.accuracy, atol=1e-10)
    assert np.allclose(joined.accuracy_sd, joined.accuracy_construction_sd, atol=1e-10)
    aggregation.reset_index().to_csv(output / "temporal_all.csv", index=False)
    test = temporal[(temporal.split == "test") & (temporal.stage != "five_stage_mean")].copy()
    test["stage"] = test.stage.astype(int)
    contrasts = []
    for (dataset, mode, cohort), frame in test.groupby(["dataset", "mode", "cohort"]):
        for period, selected in (("final", frame[frame.stage == 4]), ("stage_mean", frame)):
            wide = selected.groupby(["construction_seed", "variant"]).f1.mean().unstack()
            full_name = next(c for c in wide if c.startswith("cotag") or c.startswith("uq_"))
            gecc_name = next(c for c in wide if c != "gecc_nearest1" and "gecc" in c)
            for left, right in ((full_name, gecc_name), (full_name, "gecc_nearest1"),
                                (full_name, "cat_cctp"), ("gecc_nearest1", gecc_name)):
                contrasts.append(dict(dataset=dataset, mode=mode, cohort=cohort, period=period,
                                      left=left, right=right, **describe(wide[left]-wide[right])))
    pd.DataFrame(contrasts).to_csv(output / "temporal_contrasts.csv", index=False)

    fidelity_root = ROOT / "results/cotag_fidelity_temporal_20261021"
    fidelity_rows = []
    for filename in ("tagsam_budgets.csv", "tagsam_native_epochs.csv"):
        df = csv(f"results/cotag_fidelity_temporal_20261021/analysis/{filename}")
        for row in df[df.split == "sealed_test"].itertuples():
            raw = read(fidelity_root / row.source)
            records = raw["rows"]
            stage_row = next(r for r in records if r["stage"] == row.stage)
            scores = stage_row.get("scores", {}).get("sealed_test")
            if scores is None:
                raise AssertionError(f"unexpected fidelity schema: {row.source}")
            vals = np.array([r["final_f1"] for r in scores["per_student"]])
            assert abs(vals.mean()-row.f1) < 1e-8
            assert abs(vals.std(ddof=1)-row.f1_student_sd) < 1e-8
            fidelity_rows.append(dict(dataset=row.dataset, stage=row.stage, outer_steps=row.outer_steps,
                                      teacher_budget=row.teacher_budget, f1=float(vals.mean()), sd=float(vals.std(ddof=1)),
                                      source=row.source))
    pd.DataFrame(fidelity_rows).to_csv(output / "tagsam_verified.csv", index=False)
    (output / "verification.json").write_text(json.dumps(dict(
        direct_pairs_recomputed=len(direct), controlled_runs_checked=checked_controls,
        text_contrasts_recomputed=len(text_rows), temporal_groups_recomputed=len(aggregation),
        tagsam_stage_rows_checked=len(fidelity_rows),
        scope="aggregation and selected raw-result checks; reuses prior baseline/temporal artifact audits",
    ), indent=2), encoding="utf-8")
    (output / "sources.json").write_text(json.dumps(SOURCES, indent=2), encoding="utf-8")
    print(json.dumps(dict(direct=direct, assignment=resources), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.output)
