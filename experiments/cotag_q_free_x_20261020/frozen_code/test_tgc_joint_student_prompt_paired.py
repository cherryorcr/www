import run_tgc_joint_student_prompt_paired as R


def _row(seed, new_gap, new_margin, old_gap, old_margin,
         delta_loss=1.0, delta_margin=0.0):
    return {
        "seed": seed,
        "delta_prompt_gap": new_gap,
        "delta_prompt_margin": new_margin,
        "old_prompt_gap": old_gap,
        "old_prompt_margin": old_margin,
        "delta_loss": delta_loss,
        "delta_margin": delta_margin,
    }


def _response(rows):
    return {"grids": [{"per_seed": rows}]}


def test_paired_prompt_response_rewards_new_and_old_margin():
    reference = _response([_row(300, 0.10, 0.20, 0.30, 0.20)])
    candidate = _response([_row(300, 0.20, 0.30, 0.40, 0.25,
                                delta_loss=0.9, delta_margin=0.1)])

    out = R.response_summary_prompt_paired(candidate, reference)

    assert out["terminal_gain"] > 0.0
    assert out["terminal_old_delta"] > 0.0
    assert out["utility"] > 0.0


def test_paired_prompt_response_can_reject_new_gain_with_old_loss():
    reference = _response([_row(300, 0.10, 0.20, 0.30, 0.20)])
    candidate = _response([_row(300, 0.50, 0.40, 0.10, 0.05,
                                delta_loss=0.9, delta_margin=0.1)])

    out = R.response_summary_prompt_paired(candidate, reference)

    assert out["terminal_old_delta"] < 0.0
