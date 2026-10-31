"""Guard/fallback accounting must not turn repeated fits into an intervention."""
import pytest

from summarize_tgc_r2b_student_revision import policy_source, promotion, values


def test_policy_source_does_not_use_report_metrics():
    assert policy_source("stop_k8", {}) == "c0"
    assert policy_source("guarded_k4", {"audit": {"accepted": False}}) == "c0"
    assert policy_source("guarded_k4", {"audit": {"accepted": True}}) == "student_k4"
    assert policy_source("random_k4", {}) == "random_k4"


def test_promotion_requires_every_frozen_guardrail():
    s1 = {"delta_all_f1_pp": 0.}
    s2 = {"delta_all_f1_pp": .5, "cells_ge_minus1": 3, "delta_seen_new_pp": 0.,
          "delta_vs_random_pp": .1, "delta_vs_teacher_pp": 0.}
    assert promotion(s1, s2)["passed"]
    assert not promotion({"delta_all_f1_pp": -.001}, s2)["passed"]
    assert not promotion(s1, {**s2, "delta_vs_random_pp": 0.})["passed"]
    assert not promotion(s1, {**s2, "delta_all_f1_pp": .499999})["passed"]


def test_missing_seeds_are_not_silently_averaged():
    with pytest.raises(ValueError, match="report seeds"):
        values({"per_seed": []}, "all", "cumulative_f1")
