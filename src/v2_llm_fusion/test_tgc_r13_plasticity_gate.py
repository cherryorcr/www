import run_tgc_r13_plasticity_gate as M


def details(gains, old=0.0):
    return [{
        "seed": seed,
        "gains": {str(step): {"old": old, "new": gains, "mixed": gains}
                  for step in M.CHECKPOINTS},
        "curve_score": gains,
    } for seed in M.SELECTOR_SEEDS]


def test_new64_requires_majority_and_old_guard():
    d = details(0.1)
    assert M._policy(d, "selected_new64")["accepted"]
    assert not M._policy(details(0.1, old=-0.02), "selected_new64")["accepted"]


def test_curve_can_fail_when_one_checkpoint_is_bad():
    d = details(0.1)
    d[0]["gains"]["128"]["old"] = -0.02
    assert not M._policy(d, "selected_curve")["accepted"]


def test_conservative_requires_all_positive_seeds():
    d = details(0.1)
    d[2]["gains"]["64"]["new"] = -0.001
    assert not M._policy(d, "selected_conservative")["accepted"]
