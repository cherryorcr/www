import numpy as np
import pytest
from summarize_unified_q_validation_20261007 import crossed_interval, pair_protocol, MATCH, check_fixed_content


def test_ci_constant_and_seed_repeatability():
    assert crossed_interval(np.full((3,3), 2.0)) == [2.0,2.0]
    a=np.arange(9).reshape(3,3)
    assert crossed_interval(a) == crossed_interval(a)
    lo,hi=crossed_interval(a)
    assert lo < a.mean() < hi


def test_ci_rejects_missing_factorial_and_invalid_shape():
    with pytest.raises(ValueError):
        crossed_interval(np.ones((1,3)))
    with pytest.raises(ValueError):
        crossed_interval(np.full((3,3),np.nan))


def test_pair_allows_same_schedule_but_not_different_budget():
    p={k:0 for k in MATCH}
    p["source_hashes"]={"graphtext_prompt_student.py":"abc", "run_unified_q_large_smoke_20261007.py":"def"}
    pair_protocol(dict(p,budget_mode="fixed_rate"),dict(p,budget_mode="fixed_schedule"))
    with pytest.raises(ValueError,match="budgets"):
        pair_protocol(p,dict(p,budgets=[1,2]))
    with pytest.raises(ValueError,match="implementation"):
        pair_protocol(p,dict(p,source_hashes={}))


def diagnostic_rows():
    reference=dict(arm="unchanged",stage=0,content_hash="x",text_hash="text",operator_hash="self",memory_rows=3)
    topology=dict(reference,arm="topology_hard",operator_hash="hard")
    text=dict(reference,arm="text_blank",text_hash="blank")
    return dict(protocol=dict(arms=["unchanged","topology_hard","text_blank"]),rows=[reference,topology,text])


def test_fixed_content_is_paired_with_own_reference():
    assert check_fixed_content(diagnostic_rows()) == 3


@pytest.mark.parametrize("key,value,error", [("content_hash","changed","X content"),
    ("text_hash","changed","changed text"),("memory_rows",4,"budget")])
def test_topology_controls_reject_content_changes(key,value,error):
    data=diagnostic_rows()
    data["rows"][1][key]=value
    with pytest.raises(ValueError,match=error):
        check_fixed_content(data)


def test_text_controls_reject_operator_changes():
    data=diagnostic_rows()
    data["rows"][2]["operator_hash"]="changed"
    with pytest.raises(ValueError,match="operator"):
        check_fixed_content(data)
