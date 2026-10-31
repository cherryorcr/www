from dataclasses import replace
import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import unified_q_multimodal as U
from unified_q_sparsification import refine_degree_preserving
from test_unified_q_multimodal import DummyEncoder, _delta


def test_strict_sinkhorn_marginals_and_second_derivative():
    torch.manual_seed(66)
    scores = (torch.randn(23, 7) * 3).requires_grad_()
    mass = torch.full((7,), 23 / 7)
    audit = []
    q = U._log_sinkhorn(scores, mass, 30, 1e-4, 512, audit)
    assert audit[0]["row_error"] <= 1e-4
    assert audit[0]["column_relative_error"] <= 1e-4
    grad = torch.autograd.grad(q.square().sum(), scores, create_graph=True)[0]
    assert torch.isfinite(torch.autograd.grad(grad.square().sum(), scores)[0]).all()


def test_strict_sinkhorn_does_not_silently_accept_failure():
    scores = torch.tensor([[20., 0.], [20., 0.], [0., 0.]])
    with pytest.raises(RuntimeError, match="Sinkhorn failed"):
        U._log_sinkhorn(scores, torch.tensor([1.5, 1.5]), 1, 1e-8, 1)


def test_fixed_iteration_solver_is_backward_compatible():
    torch.manual_seed(9)
    scores = torch.randn(13, 3)
    mass = torch.full((3,), 13 / 3)
    old = scores.clone()
    for _ in range(30):
        old = old - old.logsumexp(1, keepdim=True)
        old = old + mass.log()[None, :] - old.logsumexp(0, keepdim=True)
    torch.testing.assert_close(U._log_sinkhorn(scores, mass, 30), old.exp(), rtol=0, atol=0)


def test_checkpointed_solver_keeps_first_and_second_derivatives():
    torch.manual_seed(19)
    scores = torch.randn(9, 3, requires_grad=True)
    mass = torch.full((3,), 3.0)
    actual = U._log_sinkhorn(scores, mass, 128)
    expected = U._sinkhorn_chunk(scores, mass.log(), 128).exp()
    a = torch.autograd.grad(actual.square().sum(), scores, create_graph=True)[0]
    b = torch.autograd.grad(expected.square().sum(), scores, create_graph=True)[0]
    torch.testing.assert_close(a, b)
    ah = torch.autograd.grad(a.square().sum(), scores)[0]
    bh = torch.autograd.grad(b.square().sum(), scores)[0]
    torch.testing.assert_close(ah, bh)


def test_large_block_checkpoint_matches_short_unrolled_solver(monkeypatch):
    torch.manual_seed(67)
    scores = torch.randn(13, 7, requires_grad=True)
    mass = torch.full((7,), 13 / 7)
    monkeypatch.setattr(U, "_SINKHORN_LARGE_BLOCK", 1)
    actual = U._log_sinkhorn(scores, mass, 30)
    expected = U._sinkhorn_chunk(scores, mass.log(), 30).exp()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    a = torch.autograd.grad(actual.square().sum(), scores, create_graph=True)[0]
    b = torch.autograd.grad(expected.square().sum(), scores, create_graph=True)[0]
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    ah = torch.autograd.grad(a.square().sum(), scores)[0]
    bh = torch.autograd.grad(b.square().sum(), scores)[0]
    torch.testing.assert_close(ah, bh, rtol=0, atol=0)


def test_cosine_propagation_scale_does_not_depend_on_feature_width():
    x = torch.nn.functional.normalize(torch.tensor([[1., 2.]]), dim=1)
    y = torch.nn.functional.normalize(torch.tensor([[2., 1.]]), dim=1)
    x2 = torch.nn.functional.normalize(x.repeat(1, 2), dim=1)
    y2 = torch.nn.functional.normalize(y.repeat(1, 2), dim=1)
    torch.testing.assert_close((1 - (x * y).sum(1)).mean(), (1 - (x2 * y2).sum(1)).mean())
    torch.testing.assert_close((x - y).square().mean(), 2 * (x2 - y2).square().mean())


def test_one_slot_class_has_no_assignment_degrees_of_freedom():
    logits = torch.tensor([[1.0], [2.0], [3.0]], requires_grad=True)
    q = U._log_sinkhorn(logits, torch.tensor([3.0]), 30, 1e-4)
    torch.testing.assert_close(q, torch.ones_like(q))
    loss = (q[:, 0] * torch.tensor([1.0, 5.0, 9.0])).sum()
    gradient = torch.autograd.grad(loss, logits)[0]
    torch.testing.assert_close(gradient, torch.zeros_like(gradient))


@pytest.mark.parametrize("mode", ["init", "q_fixed", "r_fixed", "old_bounded", "old_unbounded"])
def test_update_counts_and_old_memory_is_not_mutated(mode):
    torch.manual_seed(8)
    labels = torch.tensor([0, 0, 0, 1, 1, 1])
    cfg = U.Config(steps=2, proxy_hidden=8, proxy_out=8, sinkhorn_tolerance=1e-4)
    prompt = torch.randn(2, 8)
    parent, _ = U.optimize_stage(_delta(), None, labels, DummyEncoder(), prompt, cfg, "cpu", use_boundary=False)
    old_x = parent["x"].clone()
    if mode == "init":
        cfg = replace(cfg, steps=0)
    if mode == "q_fixed":
        cfg = replace(cfg, train_q=False)
    if mode == "r_fixed":
        cfg = replace(cfg, train_residual=False)
    if mode.startswith("old"):
        cfg = replace(cfg, old_update_radius=0.01 if mode == "old_bounded" else -1)
    block, audit = U.optimize_stage(_delta(1, 12), parent, labels, DummyEncoder(), prompt,
                                    cfg, "cpu", use_boundary=False)
    memory = U.append_memory(parent, block)
    assert torch.equal(parent["x"], old_x)
    assert audit["q_update_count"] == (0 if mode in ("init", "q_fixed") else 2)
    assert audit["residual_update_count"] == (0 if mode in ("init", "r_fixed") else 2)
    assert memory["texts"][:6] == parent["texts"]
    if mode == "old_bounded":
        assert 0 < audit["old_x_relative_change_max"] <= 0.01001
    elif not mode.startswith("old"):
        torch.testing.assert_close(memory["x"][:6], old_x, rtol=0, atol=0)


def test_swaps_keep_degrees_and_improve_message_target():
    x = torch.tensor([[1., 0.], [0., 1.], [1., 0.], [0., 1.]])
    edges = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]])
    weights = torch.ones(4, 4)
    weights[0, 2] = weights[2, 0] = 3
    new_edges, audit = refine_degree_preserving(edges, weights, x, x, max_swaps=2)
    assert torch.equal(torch.bincount(new_edges[0], minlength=4), torch.ones(4, dtype=torch.long))
    assert audit["accepted_swaps"] > 0
    assert audit["propagation_after"] < audit["propagation_before"]


def test_v3_materialization_and_old_edge_invariants():
    torch.manual_seed(5)
    cfg = U.Config(steps=2, proxy_hidden=8, proxy_out=8, sinkhorn_tolerance=1e-4,
                   propagation_metric="cosine", topology_refine_swaps=2)
    labels, prompt = torch.tensor([0, 0, 0, 1, 1, 1]), torch.randn(2, 8)
    parent, _ = U.optimize_stage(_delta(), None, labels, DummyEncoder(), prompt, cfg, "cpu", use_boundary=False)
    block, audit = U.optimize_stage(_delta(1, 12), parent, labels, DummyEncoder(), prompt, cfg, "cpu", use_boundary=False)
    memory = U.append_memory(parent, block)
    assert torch.equal(memory["edge_index"][:, (memory["edge_index"] < 6).all(0)], parent["edge_index"])
    assert audit["topology"]["degree_preserved"]
    assert audit["text_hard_forward_max_error"] < 1e-5
