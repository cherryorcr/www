import copy

import pytest

import summarize_tgc_selector_confirmation as H


def complete():
    return {"results": [{"stage": s, "arm": a, "per_seed": [{"seed": x} for x in H.B.REPORT_SEEDS]}
                        for s in (0, 1, 2) for a in H.B.ARMS]}


def test_matrix_is_once_per_arm_and_stage():
    value = complete()
    assert len(H.matrix(value)) == 27
    value["results"].append(copy.deepcopy(value["results"][0]))
    with pytest.raises(ValueError, match="duplicated"):
        H.matrix(value)


def test_seed_duplication_is_not_an_independent_observation():
    value = complete()
    value["results"][0]["per_seed"][1]["seed"] = H.B.REPORT_SEEDS[0]
    with pytest.raises(ValueError, match="seeds"):
        H.matrix(value)


def test_pairing_and_record_count():
    assert H.paired([2.] * 6, [1.] * 6) == (1., 1., 1.)
    assert H.SLOTS == 648
    with pytest.raises(ValueError):
        H.paired([2.], [1.])
