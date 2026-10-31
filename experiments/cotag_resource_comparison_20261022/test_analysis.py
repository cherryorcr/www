import numpy as np

from analyze import METRICS, paired_rows, V


def test_independent_scores_keep_absent_classes():
    truth = np.array([0, 0, 1, 1])
    pred = np.array([0, 1, 1, 1])
    f1, accuracy = V.independent_scores(truth, pred, np.array([0, 1, 2]))
    assert np.isclose(f1, 100 * (2 / 3 + 4 / 5) / 3)
    assert accuracy == 75


def test_pairs_use_construction_and_relative_denominator():
    runs = []
    for seed, left, right in ((1, 2., 4.), (2, 4., 8.), (3, 8., 16.)):
        for arm, value in (("incremental_q", left), ("full_history", right)):
            runs.append(dict(dataset="d", arm=arm, construction_seed=seed,
                             **{k: value for k in METRICS}))
    rows = paired_rows("d", "incremental_q", "full_history", runs)
    reduction = next(r for r in rows if r["metric"] == "peak_gib_reduction_pct")
    assert reduction["n"] == 3 and reduction["mean"] == 50
    f1 = next(r for r in rows if r["metric"] == "f1")
    assert f1["positive"] == 0 and f1["worst"] == -8
    assert not paired_rows("missing", "incremental_q", "full_history", runs)
