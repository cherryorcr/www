from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_unified_q_constructor_attribution_20261008 import diagnostic_record


def test_diagnostics_separate_payload_change_from_x_change():
    full = dict(x=torch.tensor([[1., 0.], [0., 1.]]),
        text_source_ids=[[3], [5]], texts=["a", "b"])
    block = dict(full, text_source_ids=[[3], [7]], texts=["a", "a"])
    trace = dict(gradient_match=.5, q_gradient_norm=.2)
    audit = dict(trace=[trace, trace], x_residual_ratio_max=.04,
        distribution_matching=dict(initial_train=2., final_train=1.,
            initial_holdout_bank=3., final_holdout_bank=2.))
    result = diagnostic_record(block, audit, full)
    assert result["relative_x_difference_from_full"] == 0.
    assert result["changed_payload_rows_from_full"] == 1
    assert result["unique_text_ratio"] == .5
    block["x"] = full["x"] * 1.1
    result = diagnostic_record(block, audit, full)
    assert result["relative_x_difference_from_full"] == pytest.approx(.1)
    assert result["dm_holdout_initial"] == 3.


def test_disabled_distribution_remains_missing_not_zero():
    block = dict(x=torch.ones(1, 2), text_source_ids=[[3]], texts=["a"])
    trace = dict(gradient_match=.5, q_gradient_norm=.2)
    audit = dict(trace=[trace], x_residual_ratio_max=0.,
        distribution_matching=dict.fromkeys(("initial_train", "final_train",
            "initial_holdout_bank", "final_holdout_bank")))
    result = diagnostic_record(block, audit, block)
    assert result["dm_train_initial"] is None
    assert result["dm_holdout_final"] is None
