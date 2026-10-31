import run_tgc_joint_student_response_v3 as R


def _grid(delta_loss, delta_margin, old_loss, old_margin, std=0.0):
    return {
        "delta_loss": {"mean": delta_loss, "std": std},
        "delta_margin": {"mean": delta_margin, "std": std},
        "old_loss": {"mean": old_loss, "std": std},
        "old_margin": {"mean": old_margin, "std": std},
    }


def _response(points):
    return {"grids": [_grid(*point) for point in points]}


def test_label_free_summary_rewards_current_delta_and_old_retention():
    reference = _response([
        (2.0, 0.10, 1.0, 0.10),
        (2.0, 0.10, 1.0, 0.10),
        (2.0, 0.10, 1.0, 0.10),
    ])
    candidate = _response([
        (1.8, 0.20, 0.9, 0.15),
        (1.7, 0.25, 0.9, 0.15),
        (1.6, 0.30, 0.9, 0.15),
    ])

    out = R.response_summary_v3(candidate, reference)

    assert out["terminal_gain"] > 0.0
    assert out["terminal_old_delta"] > 0.0
    assert out["utility"] > 0.0


def test_label_free_summary_penalizes_response_variance():
    reference = _response([
        (2.0, 0.10, 1.0, 0.10),
        (2.0, 0.10, 1.0, 0.10),
        (2.0, 0.10, 1.0, 0.10),
    ])
    stable = _response([
        (1.9, 0.15, 1.0, 0.10, 0.01),
        (1.9, 0.15, 1.0, 0.10, 0.01),
        (1.9, 0.15, 1.0, 0.10, 0.01),
    ])
    noisy = _response([
        (1.9, 0.15, 1.0, 0.10, 0.50),
        (1.9, 0.15, 1.0, 0.10, 0.50),
        (1.9, 0.15, 1.0, 0.10, 0.50),
    ])

    stable_out = R.response_summary_v3(stable, reference)
    noisy_out = R.response_summary_v3(noisy, reference)

    assert noisy_out["variance_penalty"] > stable_out["variance_penalty"]
    assert noisy_out["utility"] < stable_out["utility"]
