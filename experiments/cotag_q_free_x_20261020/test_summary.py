"""Independent recalculation from per-Student records, without aggregation helpers."""
import json
import math
from pathlib import Path
import statistics

import pytest

HERE = Path(__file__).resolve().parent


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_three_construction_statistics():
    grouped = {}
    for folder in ("cotag_q_free_x_20261020_v1", "cotag_q_free_x_20261020_v2"):
        for path in (HERE / "results" / folder).glob("*/results.json"):
            result = load(path)
            if result.get("status") != "complete":
                continue
            protocol = result["protocol"]
            for arm in ("uq_dm_edgefree", "independent_x"):
                final = [row for row in result["rows"] if row["arm"] == arm][-1]
                for architecture in ("GCN", "GraphSAGE"):
                    source = final["scores"] if architecture == "GCN" else result["sage"][arm]["scores"]
                    for split in ("report_dev", "sealed_test"):
                        for metric in ("final_f1", "accuracy", "old_f1", "new_f1"):
                            key = (protocol["dataset"], architecture, split, metric, arm)
                            grouped.setdefault(key, {})[protocol["construction_seed"]] = statistics.mean(
                                row[metric] for row in source[split]["per_student"])
    summary = load(HERE / "analysis/summary.json")
    assert summary["completed"] == 9
    for row in summary["aggregates"]:
        key = (row["dataset"], row["architecture"], row["split"], row["metric"], row["arm"])
        if key not in grouped:
            continue
        values = list(grouped[key].values())
        assert len(values) == 3
        assert row["mean"] == pytest.approx(statistics.mean(values), abs=1e-10)
        assert row["sd"] == pytest.approx(statistics.stdev(values), abs=1e-10)
    for row in summary["paired_differences"]:
        if row["contrast"] != "uq_dm_edgefree - independent_x" or row["metric"] == "stage_mean_f1":
            continue
        key = (row["dataset"], row["architecture"], row["split"], row["metric"])
        full, free = grouped[(*key, "uq_dm_edgefree")], grouped[(*key, "independent_x")]
        differences = [full[seed] - free[seed] for seed in sorted(full)]
        mean = statistics.mean(differences)
        half = 4.302652729911275 * statistics.stdev(differences) / math.sqrt(3)
        assert row["mean"] == pytest.approx(mean, abs=1e-10)
        assert row["ci_low"] == pytest.approx(mean - half, abs=1e-9)
        assert row["ci_high"] == pytest.approx(mean + half, abs=1e-9)
        assert row["positive_count"] == sum(value > 0 for value in differences)
        assert row["worst_difference"] == pytest.approx(min(differences), abs=1e-10)


def test_plateau_requires_dense_last_thirty():
    import summarize as S
    assert S.plateau([dict(step=i, loss=1.) for i in range(1, 31)], "loss")["plateau_both_below_1pct"]
    assert "plateau_both_below_1pct" not in S.plateau([dict(step=i, loss=1.) for i in (1, 10, 20, 30)], "loss")
    trace = [dict(step=i, loss=1. if i <= 10 else .5) for i in range(1, 31)]
    assert not S.plateau(trace, "loss")["plateau_both_below_1pct"]


def test_single_construction_comparison_has_no_ci():
    from summarize_budget_check import compare
    base = dict(dataset="arxiv", architecture="GCN", split="report_dev", metric="final_f1", arm="independent_x")
    rows = [dict(base, updates=30, mean=50.), dict(base, updates=100, mean=51.)]
    change = compare(rows, "updates", 100, 30)[0]
    assert change["difference"] == 1.
    assert change["n_constructions"] == 1
    assert change["paired_construction_ci"] is None
