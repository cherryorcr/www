import math

import pytest

from summarize_tgc_r4_relational import interval, promotion_decision


def test_constant_paired_difference_has_zero_interval_width():
    assert interval([1.] * 6) == [1., 1.]


def test_six_seed_interval_uses_sample_sd():
    lo, hi = interval([-2, -1, 0, 0, 1, 2])
    assert hi == pytest.approx(2.570581836 * math.sqrt(2) / math.sqrt(6))
    assert lo == -hi


def test_new_class_and_old_class_gates_cannot_be_hidden_by_mean():
    limits = {"mean_final_min_pp": .5, "cells_ge_minus1_min": 3,
              "worst_final_min_pp": -2, "worst_old_min_pp": -2,
              "mean_new_min_pp": 0, "mean_vs_random_min_pp": 0,
              "max_build_seconds_per_stage": 600}
    main = {"final_delta_pp": 1., "cells_ge_minus1": 4, "worst_final_delta_pp": 0.,
            "worst_final_old_delta_pp": -3., "final_new_delta_pp": -.1, "final_mean_f1": 60.}
    gates = promotion_decision(main, {"final_mean_f1": 59.},
                               [{"executed_primary_build_with_diagnostics_seconds": 10.}], limits)
    assert gates["mean_final"] and not gates["worst_old"] and not gates["mean_new"]
    assert not all(gates.values())


def test_interval_rejects_unplanned_count():
    with pytest.raises(ValueError):
        interval([1, 2, 3])
