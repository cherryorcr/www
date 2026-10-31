import run_tgc_joint_student_response as R


def grid(current, old):
    return {
        "current_f1": {"mean": current},
        "old_mean_f1": None if old is None else {"mean": old},
    }


def response(values, old_values=None, variance=0.0):
    old_values = old_values or [None] * len(values)
    grids = [grid(c, o) for c, o in zip(values, old_values)]
    return {
        "grids": grids,
        "tail_mean": sum(values) / len(values),
        "tail_std": variance,
        "cumulative_tail_std": variance,
        "old_tail_mean": (None if all(v is None for v in old_values)
                           else sum(v for v in old_values if v is not None)
                           / sum(v is not None for v in old_values)),
    }


def test_response_summary_uses_new_old_variance_and_horizon():
    ref = response([50.0, 54.0], [40.0, 42.0])
    cand = response([51.0, 56.0], [39.0, 44.0], variance=1.0)
    out = R.response_summary(cand, ref, old_weight=0.25, variance_weight=0.1)
    assert out["new_gain"] == 1.5
    assert out["old_delta"] == 0.5
    assert out["minimum_horizon_gain"] == 0.75
    assert out["utility"] == 1.5 + 0.25 * 0.5 - 0.1 * 2.0


def test_fallback_can_be_selected_without_report_f1():
    ref = response([50.0, 50.0], [40.0, 40.0])
    assert R.response_summary(ref, ref)["utility"] == 0.0
