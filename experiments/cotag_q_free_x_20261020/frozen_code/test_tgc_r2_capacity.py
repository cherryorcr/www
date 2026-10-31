"""Regression tests for subset execution and honest capacity accounting."""
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

import run_tgc_revision_gate as R
import tgc_revision_gate as T
from run_tgc_r2_capacity import CONFIGS
from test_tgc_revision_gate import block


def test_arm_subsets_are_c0_first_and_validated():
    assert R.normalize_arms(["tgc_r", "c0"]) == ["c0", "tgc_r"]
    for arms in ([], ["tgc_r"], ["c0", "c0"], ["c0", "unknown"]):
        with pytest.raises(ValueError):
            R.normalize_arms(arms)


def test_registered_quotas_not_clipped_by_protection():
    for n in (25, 67, 94, 110):
        for protection, quota, _ in CONFIGS.values():
            assert n - math.ceil(protection * n) >= math.floor(quota * n)


def test_zero_radius_high_quota_records_effective_capacity():
    cfg = T.RevisionConfig(protected_fraction=0.4, revision_fraction_of_old=0.5,
                          revision_radius=0.0, teacher_steps=0)
    old = block(10)
    revised, audit = T.revise_old_memory(old, block(3, 100, 1), cfg)
    assert audit["revision_quota"] == 5
    assert audit["updated_rows"] == []
    T.assert_revision_invariants(old, revised, audit, cfg)


def test_subset_runner_evaluates_only_registered_arms():
    with TemporaryDirectory() as folder:
        args = SimpleNamespace(output=folder, rate=0.05, seeds=[100],
                               arms=["c0", "tgc_r"], device="cpu", student_steps=4,
                               reference_root=None)
        memory = block(4)
        data = {"dataset": "cora", "pool_mask": torch.ones(4, dtype=torch.bool),
                "report_dev_mask": torch.zeros(4, dtype=torch.bool), "encoder": None}
        calls = []

        def fake_fit(mem, data, stage, seed, arm, *rest):
            calls.append(arm)
            return {"steps": 4, "executed_here": True}

        with (patch.object(R, "stage_groups", return_value=[[0]]),
              patch.object(R.A, "set_stream", return_value=([torch.arange(4)], [4])),
              patch.object(R.C, "sha", return_value="hash"),
              patch.object(R.U, "build_stage_block", return_value=memory),
              patch.object(R.SCA, "append_memory", return_value=memory),
              patch.object(R, "fit", side_effect=fake_fit),
              patch.object(R.C, "aggregate", return_value={"all": {"cumulative_f1": {"mean": 0}}})):
            R.run_cell(data, "canonical", args, T.RevisionConfig(), {"teacher": {}})
        assert calls == ["c0", "tgc_r"]
        completed = R.json.loads((Path(folder) / "cora/canonical/completion.json").read_text())
        assert completed["expected_fits"] == completed["fits"] == 2


def test_radius_invariant_detects_illegal_displacement():
    old = block(4)
    revised = T.clone_memory(old)
    revised["x"][0] += 5
    with pytest.raises(AssertionError, match="radius"):
        T.assert_revision_invariants(old, revised,
            {"updated_rows": [0], "revision_quota": 1}, T.RevisionConfig())
