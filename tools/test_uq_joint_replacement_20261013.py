"""The joint-pair candidate changes only the distribution objective."""
from dataclasses import asdict
from unittest.mock import patch

import torch

import run_uq_final_confirmation_20261012 as F


def test_joint_replacement_config_is_a_two_weight_change():
    base = F.construction_config("uq_dm_edgefree", 271006)
    candidate = F.construction_config(F.JOINT_REPLACEMENT_ARM, 271006)
    changes = {key: (asdict(base)[key], asdict(candidate)[key])
               for key in asdict(base) if asdict(base)[key] != asdict(candidate)[key]}
    assert changes == {"lambda_distribution": (1.0, 0.0),
                       "lambda_joint_distribution": (0.0, 1.0)}
    assert candidate.joint_distribution_mode == "paired"
    assert candidate.old_update_radius == 0


def test_candidate_uses_edge_free_constructor_and_frozen_append():
    cfg = F.construction_config(F.JOINT_REPLACEMENT_ARM, 271006)
    block = {"x": torch.zeros(2, 3), "edge_index": torch.empty((2, 0), dtype=torch.long)}
    parent = {"x": torch.ones(2, 3), "edge_index": torch.empty((2, 0), dtype=torch.long)}
    with patch.object(F.R.UQ, "optimize_stage", return_value=(block, {"ok": True})) as optimize, \
         patch.object(F.R.L, "append_edge_free", return_value={"appended": True}) as append:
        memory, audit = F.construct(F.JOINT_REPLACEMENT_ARM, {}, parent, torch.tensor([0, 1]),
                                    cfg, None, None, 271006, "cpu")
    assert memory == {"appended": True} and audit == {"ok": True}
    assert optimize.call_args.kwargs == {"use_boundary": False, "use_topology": False}
    assert optimize.call_args.args[1] is parent
    append.assert_called_once_with(parent, block)
