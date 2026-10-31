import run_tgc_joint_student_response_v2 as R


def _grid(current, old):
    return {
        "current_f1": {"mean": current, "std": 0.5},
        "old_mean_f1": None if old is None else {"mean": old, "std": 0.5},
    }


def _response(values, old_values):
    return {
        "grids": [_grid(c, o) for c, o in zip(values, old_values)],
    }


def test_stage_curve_includes_terminal_step():
    assert R.stage_curve(59) == [128, 512, 944]


def test_terminal_weight_dominates_early_gain():
    reference = _response([50.0, 50.0, 50.0], [40.0, 40.0, 40.0])
    candidate = _response([45.0, 52.0, 54.0], [40.0, 40.0, 40.0])
    out = R.response_summary_v2(candidate, reference, rank_penalty_weight=0.0)
    assert out["terminal_gain"] == 4.0
    assert out["weighted_gain"] > 0.0
