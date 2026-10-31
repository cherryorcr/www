import copy
import math
from pathlib import Path

import pytest

import summarize_unified_q_matrix_20261008 as S


def fixture():
    scores = [dict(seed=seed, final_f1=50.0, accuracy=60.0, old_f1=None, new_f1=50.0)
              for seed in (150,151,152)]
    row = dict(arm="uq_dm_edgefree", stage=0, memory_rows=4,
               checks=dict(budget_ok=True, prefix_ok=True, finite_ok=True, sinkhorn_feasible=True),
               score=dict(final_f1=50.0, accuracy=60.0, old_f1=None, new_f1=50.0,
                          per_student=scores, final_f1_std=0.0, accuracy_std=0.0),
               construction_seconds=1.0, student_seconds=2.0, construction_peak_mb=10,
               memory_hash="test")
    return dict(protocol=dict(dataset="test", groups=[[0]], budgets=[4], order_seed=9101,
        construction_seed=271006, student_seeds=[150,151,152]), rows=[row])


def test_only_complete_cells_are_aggregated():
    data=fixture()
    assert S.read_cell(data,"uq_dm_edgefree",Path("result.json"))["final_f1"]==50.0
    data["protocol"]["groups"].append([1])
    assert S.read_cell(data,"uq_dm_edgefree",Path("result.json")) is None


@pytest.mark.parametrize("issue", ["mean", "seed", "budget", "prefix", "sinkhorn"])
def test_invalid_results_fail_closed(issue):
    data=fixture(); row=data["rows"][0]
    if issue=="mean": row["score"]["final_f1"]=51
    if issue=="seed": row["score"]["per_student"][0]["seed"]=999
    if issue=="budget": row["memory_rows"]=5
    if issue=="prefix": row["checks"]["prefix_ok"]=False
    if issue=="sinkhorn": row["checks"]["sinkhorn_feasible"]=False
    with pytest.raises(ValueError):
        S.read_cell(data,"uq_dm_edgefree",Path("result.json"))


def test_three_paired_units_not_nine_student_replicates():
    assert S.t_interval([1.,1.,1.])==[1.,1.]
    with pytest.raises(ValueError): S.t_interval([1.]*9)
    lo,hi=S.t_interval([-1.,0.,1.])
    assert lo<0<hi and math.isclose(-lo,hi)


def test_recovery_keeps_failure_history_and_requires_matching_cell():
    attempt = dict(dataset="wikics", order_seed=9101, construction_seed=271006,
                   arm="uq_no_q_anchor", source="failed/results.json")
    completed = dict(attempt, source="recovered/results.json")
    pending, recovered = S.split_incomplete([attempt], {S.key(completed): completed})
    assert pending == []
    assert recovered == [dict(attempt, completed_source="recovered/results.json")]
    other_seed = dict(completed, construction_seed=271007)
    pending, recovered = S.split_incomplete([attempt], {S.key(other_seed): other_seed})
    assert pending == [attempt]
    assert recovered == []
