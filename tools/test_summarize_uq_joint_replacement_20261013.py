"""Protect the matched-cell audit from silent protocol and objective drift."""
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import pytest

import summarize_uq_joint_replacement_20261013 as S


ROOT = Path(__file__).resolve().parents[1] / "results/uq_joint_replacement_20261013"


@pytest.mark.parametrize("dataset", S.DATASETS)
def test_existing_matched_cell_is_complete_and_audited(dataset):
    final, stages = S.audit(dataset, ROOT)
    assert {row["split"] for row in final} == {"report_dev", "sealed_test"}
    assert len(stages) == len(S.read(ROOT / f"{dataset}_candidate.json")["protocol"]["budgets"])
    assert all(row["joint_q_gradient"] > 0 and row["hard_forward_error"] < 1e-5
               for row in stages)


@pytest.mark.parametrize("field,value,error", [
    ("pool_mask_hash", "changed", "protocol differs"),
    ("student_seeds", [151], "protocol differs"),
    ("budgets", [1], "protocol differs"),
])
def test_mismatched_protocol_rejected(field, value, error):
    original = S.read

    def changed(path):
        result = deepcopy(original(path))
        if path.name == "bookhis_candidate.json":
            result["protocol"][field] = value
        return result

    with patch.object(S, "read", side_effect=changed), pytest.raises(ValueError, match=error):
        S.audit("bookhis", ROOT)


def test_extra_constructor_change_rejected():
    original = S.read

    def changed(path):
        result = deepcopy(original(path))
        if path.name == "bookhis_candidate.json":
            result["protocol"]["joint_replacement"]["config"]["lr_q"] = 0.2
        return result

    with patch.object(S, "read", side_effect=changed), pytest.raises(ValueError, match="effective constructor"):
        S.audit("bookhis", ROOT)


def test_failed_historical_access_rejected():
    original = S.read

    def changed(path):
        result = deepcopy(original(path))
        if path.name == "bookhis_candidate.json":
            result["rows"][0]["audit"]["historical_raw_reads"] = 1
        return result

    with patch.object(S, "read", side_effect=changed), pytest.raises(ValueError, match="access/topology"):
        S.audit("bookhis", ROOT)
