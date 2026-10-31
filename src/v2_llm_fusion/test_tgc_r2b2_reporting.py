"""Missing cells, shared fit reuse and descriptive gate errors stay explicit."""
import pytest

from summarize_tgc_r2b_response_alignment import (
    ARMS, SLOTS, gate_diagnosis, matrix, temporary_steps,
)


def test_full_matrix_is_1248_slots():
    assert len(ARMS) == 26 and SLOTS == 1248
    rows = [{"stage": s, "arm": a} for s in (1, 2) for a in ARMS]
    assert len(matrix(rows)) == 52
    with pytest.raises(ValueError, match="matrix"):
        matrix(rows[:-1])
    with pytest.raises(ValueError, match="matrix"):
        matrix(rows + [rows[0]])


def test_gate_mean_sign_is_not_significance():
    row = gate_diagnosis(True, -.4, -1.2, .4)
    assert row["accepted_negative_mean"] and not row["accepted_ci_below_zero"]
    row = gate_diagnosis(False, .5, -.1, 1.1)
    assert row["rejected_positive_mean"] and not row["rejected_ci_above_zero"]
    assert gate_diagnosis(False, .5, .1, .9)["rejected_ci_above_zero"]


def test_temporary_suffix_steps_support_legacy_and_new():
    assert temporary_steps({"outer_train_student_steps": 32,
                            "validation_student_steps": 8, "baseline_student_steps": 4}) == 44
    assert temporary_steps({"suffix_student_steps": 176}) == 176
