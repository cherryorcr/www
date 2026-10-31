from dataclasses import asdict, replace

import pytest
import torch
import torch.nn.functional as F

import unified_q_multimodal as U
import run_unified_q_inloop_cat_20261007 as I
import run_unified_q_matrix_20261008 as R
from test_unified_q_multimodal import DummyEncoder, _delta


def test_full_config_is_exactly_the_frozen_config():
    assert asdict(R.q_config("uq_dm_edgefree", 271006, 30, 2048)) == asdict(
        I.config("uq_dm_edgefree", 271006, 30, 2048))


@pytest.mark.parametrize("arm,changes", [
    ("uq_no_gm", {"lambda_gradient"}),
    ("uq_no_dm", {"lambda_distribution"}),
    ("uq_fixed_q", {"train_q"}),
    ("uq_frozen_text", {"text_assignment"}),
    ("uq_detached_text", {"text_assignment"}),
    ("uq_no_prop", {"lambda_propagation"}),
    ("uq_no_residual", {"train_residual"}),
    ("uq_no_text_init", {"text_weight"}),
    ("uq_no_q_anchor", {"lambda_q_anchor"}),
    ("uq_no_x_anchor", {"lambda_x_anchor"}),
    ("uq_dm_only", {"lambda_gradient", "lambda_propagation", "lambda_q_anchor", "lambda_x_anchor"}),
])
def test_ablation_changes_only_declared_components(arm, changes):
    base = asdict(R.q_config("uq_dm_edgefree", 271006, 30, 2048))
    actual = asdict(R.q_config(arm, 271006, 30, 2048))
    assert {key for key in base if base[key] != actual[key]} == changes


def test_cross_class_logits_are_not_pooling_weights_or_gradient_paths():
    torch.manual_seed(71)
    raw_y = torch.tensor([0, 0, 0, 1, 1, 1])
    slot_y = torch.tensor([0, 0, 1, 1])
    mask = raw_y[:, None] != slot_y[None, :]
    logits = torch.randn(6, 4).masked_fill(mask, -1e4).requires_grad_()
    q = U._masked_sinkhorn(logits, raw_y, slot_y, .35, 40)
    other = logits.detach().masked_fill(mask, 1e4)
    q_other = U._masked_sinkhorn(other, raw_y, slot_y, .35, 40)
    assert torch.equal(q[mask], torch.zeros_like(q[mask]))
    torch.testing.assert_close(q, q_other, rtol=0, atol=0)
    (q * torch.randn_like(q)).sum().backward()
    assert torch.equal(logits.grad[mask], torch.zeros_like(logits.grad[mask]))
    assert logits.grad[~mask].abs().sum() > 0


def test_st_text_has_hard_value_and_soft_derivative():
    torch.manual_seed(13)
    q = torch.rand(6, 3, requires_grad=True)
    texts = F.normalize(torch.randn(6, 8), dim=1)
    soft = F.normalize(q.T @ texts / q.sum(0)[:, None], dim=1)
    hard = texts[q.argmax(0)]
    st = hard + (soft - soft.detach())
    torch.testing.assert_close(st, hard, rtol=0, atol=0)
    probe = torch.randn_like(st)
    actual = torch.autograd.grad((st * probe).sum(), q, retain_graph=True)[0]
    expected = torch.autograd.grad((soft * probe).sum(), q)[0]
    torch.testing.assert_close(actual, expected)
    assert actual.abs().sum() > 0


@pytest.mark.parametrize("mode,requires_grad", [("learned", True), ("detached", False), ("frozen", False)])
def test_constructor_text_branch_and_materialization(monkeypatch, mode, requires_grad):
    torch.manual_seed(5)
    original = U._loss
    observed = []
    def spy(model, memory, *args, **kwargs):
        if memory["x"].requires_grad:
            observed.append(memory["text_emb"].requires_grad)
        return original(model, memory, *args, **kwargs)
    monkeypatch.setattr(U, "_loss", spy)
    cfg = replace(R.q_config("uq_dm_edgefree", 271006, 1, 2048),
                  proxy_hidden=8, proxy_out=8, distribution_hidden=12,
                  text_assignment=mode)
    block, audit = U.optimize_stage(_delta(), None, torch.tensor([0,0,0,1,1,1]),
        DummyEncoder(), torch.randn(2,8), cfg, "cpu", use_boundary=False, use_topology=False)
    assert observed == [requires_grad]
    assert audit["text_hard_forward_max_error"] < 1e-4
    assert block["edge_index"].numel() == 0


def test_partial_arm_is_rejected_before_losing_its_parent():
    with pytest.raises(RuntimeError, match="partial arm"):
        R.completed_arm([dict(arm="x", stage=0)], "x", 5)
    assert R.completed_arm([dict(arm="x", stage=i) for i in range(5)], "x", 5)
    assert not R.completed_arm([], "x", 5)


def test_rounding_warning_does_not_relax_solver_tolerance():
    a = dict(q_row_sum_max_error=1.00136e-4, q_column_relative_error=2e-7,
             sinkhorn_max_row_error=9.99e-5, sinkhorn_max_column_relative_error=2e-7)
    check = R.assignment_checks(a)
    assert check["sinkhorn_feasible"] and check["numerical_warning"]
    a["sinkhorn_max_row_error"] = 1.0001e-4
    assert not R.assignment_checks(a)["sinkhorn_feasible"]
    a["sinkhorn_max_row_error"] = 9.99e-5
    a["q_row_sum_max_error"] = 1.02e-4
    assert not R.assignment_checks(a)["sinkhorn_feasible"]
