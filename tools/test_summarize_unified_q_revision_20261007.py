from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_unified_q_revision_20261007 import paired_aggregate


def _row(arm, order, seed, f1, mode="fixed_schedule"):
    row = {"dataset": "toy", "budget_mode": mode, "arm": arm,
           "order_seed": order, "construction_seed": seed, "final_f1": f1,
           "accuracy": f1, "memory_rows": 10}
    for key in ("stage_avg_f1", "stage_avg_accuracy", "old_f1", "new_f1", "worst_stage"):
        row[key] = f1
    row.update(construction_seconds=1, student_seconds=2)
    return row


def test_pairs_identical_order_and_seed_before_averaging():
    rows = [_row("cat_cctp", 1, 1, 90), _row("cat_cctp", 1, 2, 10),
            _row("uq_semantic", 1, 2, 15)]
    candidate = next(r for r in paired_aggregate(rows) if r["arm"] == "uq_semantic")
    assert candidate["mean_delta_f1"] == 5
    assert candidate["cells"] == 1
    assert candidate["order_cluster_t95_low_f1"] is None


def test_orders_not_replicates_set_ci_degrees_of_freedom():
    rows = []
    for order, delta in ((1, 1), (2, 2), (3, 3)):
        for seed in (1, 2, 3):
            rows.extend((_row("cat_cctp", order, seed, 20), _row("uq_semantic", order, seed, 20+delta)))
    candidate = next(r for r in paired_aggregate(rows) if r["arm"] == "uq_semantic")
    assert candidate["orders"] == 3 and candidate["cells"] == 9
    assert candidate["mean_delta_f1"] == 2
    assert candidate["order_cluster_t95_low_f1"] == pytest.approx(-.48414, abs=1e-4)
    assert candidate["positive_cell_ratio_f1"] == 1


def test_budget_protocols_stay_separate():
    rows = [_row("cat_cctp", 1, 1, 20), _row("uq_semantic", 1, 1, 21),
            _row("cat_cctp", 1, 1, 60, "fixed_rate"), _row("uq_semantic", 1, 1, 70, "fixed_rate")]
    result = {r["budget_mode"]: r for r in paired_aggregate(rows) if r["arm"] == "uq_semantic"}
    assert result["fixed_schedule"]["mean_delta_f1"] == 1
    assert result["fixed_rate"]["mean_delta_f1"] == 10


def test_rejects_duplicate_results():
    row = _row("cat_cctp", 1, 1, 20)
    with pytest.raises(ValueError, match="duplicate"):
        paired_aggregate([row, row.copy()])


def test_partial_unbalanced_order_grid_has_no_ci():
    rows = []
    for order, seed in ((1, 1), (1, 2), (2, 1)):
        rows.extend((_row("cat_cctp", order, seed, 20), _row("uq_semantic", order, seed, 25)))
    candidate = next(r for r in paired_aggregate(rows) if r["arm"] == "uq_semantic")
    assert not candidate["same_construction_seeds_each_order"]
    assert candidate["order_cluster_t95_low_f1"] is None
