import pytest

from run_budget_check import assignment_checks


def audit(error):
    return dict(q_row_sum_max_error=error, q_column_relative_error=error,
        sinkhorn_max_row_error=error, sinkhorn_max_column_relative_error=error)


def test_checker_uses_recorded_tolerance_and_rejects_large_errors():
    data = audit(0.00019997358322143555)
    assert assignment_checks(data, 2e-4)["sinkhorn_feasible"]
    assert not assignment_checks(data, 1e-4)["sinkhorn_feasible"]
    assert not assignment_checks(audit(0.000201), 2e-4)["sinkhorn_feasible"]


def test_full_matrix_rounding_cannot_hide_local_solver_failure():
    data = audit(2e-4)
    data["q_row_sum_max_error"] += 4 * 2**-23
    assert assignment_checks(data, 2e-4)["sinkhorn_feasible"]
    data["sinkhorn_max_row_error"] += 2**-23
    assert not assignment_checks(data, 2e-4)["sinkhorn_feasible"]


def test_nonfinite_errors_and_invalid_tolerances_fail():
    for bad in (float("nan"), float("inf"), -1.):
        data = audit(0.)
        data["q_row_sum_max_error"] = bad
        assert not assignment_checks(data, 2e-4)["sinkhorn_feasible"]
    for bad in (0., -1., float("nan"), float("inf")):
        with pytest.raises(ValueError):
            assignment_checks(audit(0.), bad)
