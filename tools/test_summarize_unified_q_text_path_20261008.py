import numpy as np
import copy
import pytest

from summarize_unified_q_text_path_20261008 import paired_summary, validate_plan


def test_paired_ci_uses_construction_cells_and_handles_constant_effect():
    values = {(o, s): 2.0 for o in (1, 2) for s in (10, 11)}
    result = paired_summary(values, [1, 2], [10, 11])
    assert result["cells"] == result["positive_cells"] == 4
    assert result["ci95_crossed_bootstrap"] == [2.0, 2.0]
    assert result["mean"] == 2 and result["std"] == 0


def test_paired_counts_and_order_seed_marginals():
    values = {(1, 10): -2., (1, 11): 0., (2, 10): 1., (2, 11): 3.}
    result = paired_summary(values, [1, 2], [10, 11])
    assert result["positive_cells"] == 2 and result["worst_cell"] == -2
    assert result["per_order"] == {"1": -1., "2": 2.}
    assert result["per_construction_seed"] == {"10": -.5, "11": 1.5}
    assert result["mean"] == np.mean(list(values.values()))


def _plan_cells():
    plan = dict(tasks=[dict(dataset="d", order=1, seed=s) for s in (10, 11)],
                student_seeds=[150, 151, 152], steps=2048, arms=["candidate", "original"], students=["gcn", "sgc"])
    cells = [dict(protocol=dict(dataset="d", order_seed=1, construction_seed=s,
        student_seeds=plan["student_seeds"], student_steps=2048, arms=plan["arms"],
        students={"gcn": {}, "sgc": {}}, max_stages=5, source_hashes={"a": "hash"}, runner_sha256="runner",
        student_batch=24, groups=[[0], [1]], budgets=[1, 2], pool_mask_hash="pool",
        report_mask_hash="report", stage_id_hashes=["stage0", "stage1"])) for s in (10, 11)]
    return plan, cells


def test_plan_reports_missing_cells_without_treating_student_seeds_as_cells():
    plan, cells = _plan_cells()
    result = validate_plan(cells[:1], plan)
    assert result["completed_cells"] == 1 and result["expected_cells"] == 2
    assert result["missing_cells"] == [dict(dataset="d", order=1, seed=11)]
    assert validate_plan(cells, plan)["missing_cells"] == []


@pytest.mark.parametrize("field,value", [("runner_sha256", "changed"), ("report_mask_hash", "changed"),
    ("student_steps", 100), ("construction_seed", 999)])
def test_plan_rejects_protocol_drift_and_unplanned_cells(field, value):
    plan, cells = _plan_cells()
    cells = copy.deepcopy(cells)
    cells[1]["protocol"][field] = value
    with pytest.raises(AssertionError):
        validate_plan(cells, plan)


def test_plan_rejects_duplicate_cells():
    plan, cells = _plan_cells()
    with pytest.raises(AssertionError, match="duplicate"):
        validate_plan([cells[0], cells[0]], plan)


def test_plan_accepts_permuted_groups_but_not_changed_budgets():
    plan, cells = _plan_cells()
    plan["tasks"][1]["order"] = cells[1]["protocol"]["order_seed"] = 2
    cells[1]["protocol"]["groups"] = [[1], [0]]
    cells[1]["protocol"]["stage_id_hashes"] = ["stage1", "stage0"]
    assert validate_plan(cells, plan)["fixed_budget_and_split_across_orders"]
    cells[1]["protocol"]["budgets"] = [1, 3]
    with pytest.raises(AssertionError, match="fixed budget"):
        validate_plan(cells, plan)
