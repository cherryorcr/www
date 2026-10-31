from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest


path = Path(__file__).with_name("summarize_unified_q_inloop_cat_20261007.py")
spec = importlib.util.spec_from_file_location("inloop_summary", path)
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)


def protocol():
    p = {key: key for key in S.PROTOCOL_KEYS}
    p["source_hashes"] = {name: name for name in (
        "graphtext_prompt_student.py", "external_continual_mechanisms.py",
        "run_unified_q_large_smoke_20261007.py")}
    return p


def test_matching_protocol_is_accepted():
    p = protocol()
    S.require_pair_protocol(p, deepcopy(p))


@pytest.mark.parametrize("key", ["budgets", "student_steps", "order_seed", "report_mask_hash"])
def test_different_protocol_is_rejected(key):
    a, b = protocol(), protocol()
    b[key] = "changed"
    with pytest.raises(ValueError, match="protocol mismatch"):
        S.require_pair_protocol(a, b)


def test_different_student_implementation_is_rejected():
    a, b = protocol(), protocol()
    b["source_hashes"]["graphtext_prompt_student.py"] = "changed"
    with pytest.raises(ValueError, match="implementation mismatch"):
        S.require_pair_protocol(a, b)


def test_no_artifacts_are_not_reported_as_zero_scores(tmp_path):
    result = S.collect(tmp_path)
    assert result["cells"] == []
    assert result["audit"]["complete_cells"] == 0
    assert result["paired"] == []
