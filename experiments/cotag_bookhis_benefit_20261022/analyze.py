"""Reconcile BookHis class contributions and historical stage trajectories."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, t
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

FULL, GECC = "uq_dm_edgefree", "gecc_hard"
SOURCES = {}


def read(path):
    SOURCES[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text(encoding="utf-8"))


def interval(values):
    a = np.asarray(values, float)
    sd = a.std(ddof=1)
    width = t.ppf(.975, len(a) - 1) * sd / np.sqrt(len(a))
    return dict(n=len(a), mean=float(a.mean()), sd=float(sd), ci_low=float(a.mean() - width),
                ci_high=float(a.mean() + width), positive=int((a > 0).sum()),
                worst=float(a.min()), values=a.tolist())


def crossed(frame, metric):
    matrix = frame.pivot(index="order", columns="construction", values=metric).to_numpy()
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise AssertionError("expected complete 3x3 crossed design")
    rng = np.random.default_rng(271022)
    orders = rng.integers(0, 3, (20000, 3))
    constructions = rng.integers(0, 3, (20000, 3))
    means = matrix[orders[:, :, None], constructions[:, None, :]].mean(axis=(1, 2))
    out = interval(matrix.flatten())
    out.update(crossed_low=float(np.quantile(means, .025)), crossed_high=float(np.quantile(means, .975)))
    return out


def historical(directory, out):
    rows = []
    paths = sorted(directory.glob("bookhis_*/results.json"))
    if len(paths) != 9:
        raise AssertionError(f"expected nine historical runs, got {len(paths)}")
    for path in paths:
        data = read(path)
        assert data["status"] == "complete"
        p = data["protocol"]
        for row in data["rows"]:
            if row["arm"] not in (FULL, GECC):
                continue
            for split, score in row["scores"].items():
                for student in score["per_student"]:
                    rows.append(dict(order=p["order_seed"], construction=p["construction_seed"],
                                     arm=row["arm"], stage=row["stage"], split=split, **student))
    df = pd.DataFrame(rows)
    df.to_csv(out / "historical_students.csv", index=False)
    keys = ["order", "construction", "stage", "split"]
    measures = ["final_f1", "accuracy", "old_f1", "new_f1"]
    means = df.groupby(keys + ["arm"])[measures].mean().reset_index()
    means.to_csv(out / "historical_constructions.csv", index=False)
    a, b = (means[means.arm == arm].set_index(keys) for arm in (FULL, GECC))
    diffs = (a[measures] - b[measures]).reset_index()
    diffs.to_csv(out / "historical_paired.csv", index=False)
    summary = []
    for (split, stage), frame in diffs.groupby(["split", "stage"]):
        for metric in measures:
            if stage == 0 and metric == "old_f1":
                continue
            result = crossed(frame, metric)
            summary.append(dict(split=split, stage=stage, metric=metric, **result))
    stage_average = diffs.groupby(["order", "construction", "split"])["final_f1"].mean().reset_index()
    for split, frame in stage_average.groupby("split"):
        summary.append(dict(split=split, stage="mean", metric="final_f1", **crossed(frame, "final_f1")))
    result = pd.DataFrame(summary)
    result.to_csv(out / "historical_summary.csv", index=False)
    return result


def diagnostic(directory, out):
    paths = sorted(directory.glob("bookhis_*/results.json"))
    if len(paths) != 3:
        raise AssertionError("three diagnostic constructions required")
    flat, per_class, drift, metadata_all = [], [], [], []
    verified, prediction_files = 0, 0
    for path in paths:
        data = read(path)
        assert data["status"] == "complete" and len(data["rows"]) == 10
        p = data["protocol"]["base_protocol"]
        seed = p["construction_seed"]
        metadata = read(path.parent / "class_metadata.json")
        metadata_all += [dict(construction=seed, **r) for r in metadata]
        cohort_path = path.parent / "evaluation_cohorts.npz"
        SOURCES[str(cohort_path.resolve())] = hashlib.sha256(cohort_path.read_bytes()).hexdigest()
        cohorts = np.load(cohort_path)
        y = cohorts["labels"]
        assert not (cohorts["pool_mask"] & (cohorts["test_mask"] | cohorts["report_dev_mask"])).any()
        assert not (cohorts["test_mask"] & cohorts["report_dev_mask"]).any()
        for row in data["rows"]:
            stage, arm = row["stage"], row["arm"]
            groups = p["groups"]
            seen = [c for g in groups[:stage + 1] for c in g]
            old, new = [c for g in groups[:stage] for c in g], groups[stage]
            assert all(row["checks"][k] for k in ("prefix_ok", "finite_ok", "budget_ok"))
            if arm == FULL:
                assert row["checks"]["saved_cotag_stage_matches"]
            assert sorted(s["seed"] for s in row["per_student"]) == [150, 151, 152]
            for student in row["per_student"]:
                prediction_files += 1
                pred_path = path.parent / student["prediction_file"]
                h = hashlib.sha256(pred_path.read_bytes()).hexdigest()
                assert h == student["prediction_sha256"]
                SOURCES[str(pred_path.resolve())] = h
                pred = np.load(pred_path)["predictions"]
                for split, key in (("report_dev", "report_dev_mask"), ("sealed_test", "test_mask")):
                    mask = cohorts[key].astype(bool) & np.isin(y, seen)
                    precision, recall, f1, support = precision_recall_fscore_support(
                        y[mask], pred[mask], labels=seen, zero_division=0)
                    expected = student["scores"][split]
                    values = dict(final_f1=100 * f1_score(y[mask], pred[mask], labels=seen,
                                                        average="macro", zero_division=0),
                                  accuracy=100 * accuracy_score(y[mask], pred[mask]),
                                  old_f1=float(np.mean([f1[seen.index(c)] for c in old]) * 100) if old else None,
                                  new_f1=float(np.mean([f1[seen.index(c)] for c in new]) * 100))
                    for key, value in values.items():
                        if value is not None:
                            assert abs(value - expected[key]) < 1e-10
                            verified += 1
                    base = dict(construction=seed, student=student["seed"], arm=arm, stage=stage, split=split)
                    flat.append(dict(**base, **values))
                    for j, c in enumerate(seen):
                        item = dict(class_id=c, precision=100 * precision[j], recall=100 * recall[j],
                                    f1=100 * f1[j], support=int(support[j]), memory_rows=row["quotas"][c])
                        stored = {r["class_id"]: r for r in expected["per_class"]}[c]
                        assert all(abs(item[k] - stored[k]) < 1e-10 for k in ("precision", "recall", "f1", "support"))
                        per_class.append(dict(**base, **item))
                    original = next(s for s in row["primary_scores"][split]["per_student"] if s["seed"] == student["seed"])
                    drift.append(dict(**base, primary_hash_match=row["checks"]["primary_memory_matches"],
                                      f1_drift=values["final_f1"] - original["final_f1"],
                                      accuracy_drift=values["accuracy"] - original["accuracy"]))
    students, classes, meta = pd.DataFrame(flat), pd.DataFrame(per_class), pd.DataFrame(metadata_all)
    students.to_csv(out / "diagnostic_students.csv", index=False)
    classes = classes.merge(meta, on=["construction", "class_id"], validate="many_to_one")
    classes.to_csv(out / "class_students.csv", index=False)
    pd.DataFrame(drift).to_csv(out / "replay_drift.csv", index=False)
    keys = ["construction", "stage", "split", "class_id"]
    means = classes.groupby(keys + ["arm"])[["f1", "recall", "precision", "support", "memory_rows"]].mean()
    paired = means.xs(FULL, level="arm") - means.xs(GECC, level="arm")
    paired = paired[["f1", "recall", "precision"]].rename(columns=lambda c: "delta_" + c).reset_index()
    paired = paired.merge(meta, on=["construction", "class_id"], validate="many_to_one")
    support = means.xs(FULL, level="arm")["support"].rename("support").reset_index()
    paired = paired.merge(support, on=keys, validate="one_to_one")
    paired["f1_contribution"] = paired.delta_f1 / paired.groupby(["construction", "stage", "split"]).class_id.transform("count")
    paired["accuracy_contribution"] = paired.delta_recall * paired.support / paired.groupby(["construction", "stage", "split"]).support.transform("sum")
    paired.to_csv(out / "class_contributions.csv", index=False)
    scalar_means = students.groupby(["construction", "stage", "split", "arm"])[["final_f1", "accuracy", "old_f1", "new_f1"]].mean()
    scalar_delta = scalar_means.xs(FULL, level="arm") - scalar_means.xs(GECC, level="arm")
    contribution = paired.groupby(["construction", "stage", "split"])[["f1_contribution", "accuracy_contribution"]].sum()
    assert np.allclose(contribution.f1_contribution, scalar_delta.final_f1, atol=1e-10)
    assert np.allclose(contribution.accuracy_contribution, scalar_delta.accuracy, atol=1e-10)
    aggregate_rows = []
    for (stage, split), group in scalar_delta.reset_index().groupby(["stage", "split"]):
        for metric in ("final_f1", "accuracy", "old_f1", "new_f1"):
            if stage == 0 and metric == "old_f1":
                continue
            aggregate_rows.append(dict(stage=stage, split=split, metric=metric, **interval(group[metric])))
    pd.DataFrame(aggregate_rows).to_csv(out / "diagnostic_summary.csv", index=False)
    class_summary = []
    for (stage, split, cls), group in paired.groupby(["stage", "split", "class_id"]):
        r = group.iloc[0]
        class_summary.append(dict(stage=stage, split=split, class_id=cls, prompt=r.prompt,
                                  frequency_group=r.frequency_group, train_count=int(r.train_count),
                                  test_support=int(r.support), introduced_stage=int(r.introduced_stage),
                                  f1_contribution=float(group.f1_contribution.mean()),
                                  accuracy_contribution=float(group.accuracy_contribution.mean()),
                                  **interval(group.delta_f1)))
    pd.DataFrame(class_summary).to_csv(out / "class_summary.csv", index=False)
    group_summary = []
    final = paired[paired.stage == 4]
    groups = final.groupby(["construction", "split", "frequency_group"]).agg(
        delta_f1=("delta_f1", "mean"), f1_contribution=("f1_contribution", "sum"),
        accuracy_contribution=("accuracy_contribution", "sum"), test_support=("support", "sum"))
    groups.reset_index().to_csv(out / "frequency_per_construction.csv", index=False)
    for (split, frequency), g in groups.reset_index().groupby(["split", "frequency_group"]):
        group_summary.append(dict(split=split, frequency_group=frequency,
                                  f1_contribution=float(g.f1_contribution.mean()),
                                  accuracy_contribution=float(g.accuracy_contribution.mean()),
                                  test_support=int(g.test_support.iloc[0]), **interval(g.delta_f1)))
    pd.DataFrame(group_summary).to_csv(out / "frequency_summary.csv", index=False)
    associations = []
    for split, frame in final.groupby("split"):
        points = frame.groupby("class_id")[["train_count", "delta_f1"]].mean()
        associations.append(dict(split=split, train_frequency_delta_f1_spearman=float(
            spearmanr(points.train_count, points.delta_f1).statistic), classes=len(points)))
    support_sensitivity = []
    for split, frame in final.groupby("split"):
        measured = frame[frame.support > 0]
        values = measured.groupby("construction").delta_f1.mean()
        support_sensitivity.append(dict(split=split, evaluated_classes=int(measured.class_id.nunique()),
                                        empty_classes=sorted(set(frame.class_id)-set(measured.class_id)),
                                        **interval(values)))
    pd.DataFrame(support_sensitivity).to_csv(out / "nonempty_class_sensitivity.csv", index=False)
    return dict(prediction_files=prediction_files, metric_scalars_verified=verified, per_class_records=len(classes),
                decomposition_checks=2 * len(contribution), associations=associations,
                note="post-hoc one-order, three-construction diagnostic; not a replacement primary table")


def plot(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cls = pd.read_csv(out / "class_summary.csv")
    cls = cls[(cls.stage == 4) & (cls.split == "sealed_test")].sort_values("train_count")
    history = pd.read_csv(out / "historical_constructions.csv")
    history = history[history.split == "sealed_test"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), layout="constrained")
    from matplotlib.patches import Patch
    colors = {"low": "#2f6ba5", "middle": "#bd8c2c", "high": "#888888"}
    positions = np.arange(len(cls))
    axes[0].bar(positions, cls["mean"], color=[colors[k] for k in cls.frequency_group], width=.68)
    points = pd.read_csv(out / "class_contributions.csv")
    points = points[(points.stage == 4) & (points.split == "sealed_test")]
    for i, row in enumerate(cls.itertuples()):
        values = points[points.class_id == row.class_id].sort_values("construction").delta_f1.to_numpy()
        axes[0].scatter(i + np.linspace(-.15, .15, len(values)), values, s=10,
                        color="#222222", zorder=3)
        if row.test_support == 0:
            axes[0].annotate("No test\nqueries", (i, 0), xytext=(i, 2.0), fontsize=7, ha="center")
    axes[0].axhline(0, color="#444444", lw=.8)
    axes[0].set_xticks(positions, [f"{int(r.class_id)}\n{int(r.train_count):,}" for r in cls.itertuples()], fontsize=8)
    axes[0].set_ylabel("Class F1 difference (percentage points)")
    axes[0].set_xlabel("Class ID / training count, from least to most frequent")
    axes[0].set_title("(a) CoTAG-Q minus GECC, one order / three seeds", fontsize=10, loc="left")
    axes[0].legend(handles=[Patch(facecolor=color, label=group) for group, color in colors.items()],
                   title="Training frequency; dots: paired seeds", loc="lower left",
                   ncol=3, frameon=False, fontsize=8, title_fontsize=8)
    for arm, label, color, style, marker in ((FULL, "CoTAG-Q", "#2f6ba5", "-", "o"),
                                           (GECC, "GECC-hard", "#bd8c2c", "--", "s")):
        g = history[history.arm == arm].groupby("stage").final_f1.agg(["mean", "std"])
        axes[1].errorbar(g.index, g["mean"], yerr=g["std"], marker=marker, linestyle=style,
                         capsize=3, lw=1.6, label=label, color=color)
    axes[1].set_xticks(range(5), ["Base", "1", "2", "3", "4"])
    axes[1].set_ylabel("Macro-F1 (%)")
    axes[1].set_xlabel("Increment")
    axes[1].set_title("(b) Historical 3 orders x 3 seeds, mean +/- SD", fontsize=10, loc="left")
    axes[1].legend(frameon=False, fontsize=9)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=.18)
        axis.set_axisbelow(True)
    fig.savefig(out / "bookhis_benefit.png", dpi=180)
    fig.savefig(out / "bookhis_benefit.pdf")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--historical", type=Path)
    parser.add_argument("--diagnostic", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    historical(args.historical or args.root / "results/uq_final_evidence_20261012/confirmation", args.output)
    verification = diagnostic(args.diagnostic, args.output) if args.diagnostic else {}
    if args.diagnostic:
        plot(args.output)
    (args.output / "verification.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")
    (args.output / "sources.json").write_text(json.dumps(SOURCES, indent=2), encoding="utf-8")
    print(json.dumps(verification, indent=2))
