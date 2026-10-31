import run_tgc_joint_student_prompt_profile as R


def _grid(seed_rows):
    return {"per_seed": seed_rows}


def _row(seed, new, old, delta_loss=1.0, delta_margin=0.0):
    return {
        "seed": seed,
        "delta_profile": {"0": new[0], "1": new[1]},
        "old_profile": {"2": old[0], "3": old[1]},
        "delta_loss": delta_loss,
        "delta_margin": delta_margin,
    }


def _response(rows):
    old_keys = sorted(rows[0]["old_profile"])
    new_keys = sorted(rows[0]["delta_profile"])
    summary = lambda values: {"mean": sum(values) / len(values),
                              "std": 0.0, "min": min(values),
                              "max": max(values), "n": len(values)}
    return {"grids": [{
        "per_seed": rows,
        "old_profile": {key: summary([row["old_profile"][key] for row in rows])
                        for key in old_keys},
        "delta_profile": {key: summary([row["delta_profile"][key] for row in rows])
                          for key in new_keys},
    }]}


def test_profile_summary_keeps_per_class_lower_quantile():
    reference = _response([_row(300, (0.10, 0.10), (0.20, 0.20))])
    candidate = _response([_row(300, (0.20, 0.30), (0.25, 0.25),
                                delta_loss=0.9, delta_margin=0.1)])

    out = R.response_summary_profile(candidate, reference)

    assert out["terminal_gain"] > 0.0
    assert out["terminal_old_delta"] > 0.0
    assert out["terminal_lcb"] > 0.0


def test_profile_summary_rejects_a_bad_old_class_in_lcb():
    reference = _response([_row(300, (0.10, 0.10), (0.20, 0.20))])
    candidate = _response([_row(300, (0.30, 0.30), (0.40, -0.10),
                                delta_loss=0.9, delta_margin=0.1)])

    out = R.response_summary_profile(candidate, reference)

    assert out["terminal_old_delta"] < 0.0
