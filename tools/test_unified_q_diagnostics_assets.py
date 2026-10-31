import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location("uq_assets", Path(__file__).with_name("build_unified_q_diagnostics_assets_20261007.py"))
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)


def cell(arm, order=9101, seed=271006, delta=2.0, **kwargs):
    item = dict(dataset="wikics", arm=arm, order_seed=str(order),
        construction_seed=str(seed), run=f"{arm}_{order}_{seed}",
        budget_mode="fixed_rate", protocol_family="frozen-prefix", memory_rows="410")
    item.update({metric: 50.0 + (delta if arm == "uq_v3" else 0) for metric in A.METRICS})
    item.update(kwargs)
    return item


def test_pairing_excludes_other_protocols_and_unpaired_cells():
    rows = [cell("cat_cctp"), cell("uq_v3"),
            cell("cat_cctp", 9102), cell("uq_v3", 9102, budget_mode="fixed_schedule")]
    result = A.paired_cells(rows)
    assert len(result) == 1
    assert result[0]["delta_f1"] == 2.0
    assert result[0]["delta_accuracy"] == 2.0


def test_duplicate_completed_cells_are_not_silently_overwritten():
    with pytest.raises(ValueError, match="Duplicate completed"):
        A.paired_cells([cell("cat_cctp"), cell("cat_cctp"), cell("uq_v3")])


def test_different_budgets_cannot_pair():
    with pytest.raises(ValueError, match="Different paired budgets"):
        A.paired_cells([cell("cat_cctp"), cell("uq_v3", memory_rows="411")])


def test_confidence_uses_three_order_means():
    rows = [cell(a, o, s, delta=i + 1) for i, o in enumerate(A.ORDERS)
            for s in A.SEEDS for a in ("cat_cctp", "uq_v3")]
    methods, ci, orders = A.aggregate_pairs(A.paired_cells(rows))
    f1 = next(r for r in ci if r["metric"] == "final_f1")
    assert f1["cells"] == 9
    assert f1["independent_cells"] == 6
    assert f1["mean_delta"] == 2.0
    assert f1["independent_mean_delta"] == 2.5
    assert f1["positive_cells"] == 9
    assert f1["order_t_ci_low"] < 0
    assert len(methods) == 2
    assert len(orders) == 3


def test_incomplete_grid_has_no_confirmation_interval():
    _, ci, _ = A.aggregate_pairs(A.paired_cells([cell("cat_cctp"), cell("uq_v3")]))
    assert all(r["order_t_ci_low"] is None for r in ci)
