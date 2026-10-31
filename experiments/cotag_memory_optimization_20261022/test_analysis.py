import numpy as np
import pytest

from analyze import independent_scores, stats


def test_macro_denominator_includes_unobserved_declared_class():
    f1, acc = independent_scores(np.array([0, 0, 1]), np.array([0, 1, 1]), [0, 1, 2])
    assert f1 == pytest.approx(100 * 4 / 9)
    assert acc == pytest.approx(100 * 2 / 3)


def test_paired_confidence_uses_constructions():
    s = stats([-.2, .0, .2])
    assert s["n"] == 3
    assert s["mean"] == pytest.approx(0.)
    assert s["sd"] == pytest.approx(.2)
    assert s["ci_low"] == pytest.approx(-.4968275, abs=1e-6)
    assert s["ci_high"] == pytest.approx(.4968275, abs=1e-6)


def test_single_construction_has_no_ci():
    assert stats([5.])["ci_low"] is None
