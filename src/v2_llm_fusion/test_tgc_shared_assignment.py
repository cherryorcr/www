"""Contracts for the new differentiable coarsening and materialization path."""
from dataclasses import replace

import torch

import tgc_shared_assignment as J
from test_tgc_relational_response import fixture

torch.set_num_threads(2)


class Encoder:
    def encode(self, texts):
        return torch.stack([torch.randn(384, generator=torch.Generator().manual_seed(sum(map(ord, t))))
                            for t in texts])


def inputs():
    old, delta, base, lifted = fixture()
    q = torch.arange(6).repeat_interleave(2)
    labels = torch.arange(2).repeat_interleave(6)
    h = torch.randn(20, 16, generator=torch.Generator().manual_seed(15))
    h = torch.nn.functional.normalize(h, dim=1)
    a = J.raw_adjacency(lifted, "cpu")
    return old, delta, base, q, labels, h, a


def test_objective_directional_gradient():
    old, _, _, q, labels, h, a = inputs()
    p = .9 * torch.nn.functional.one_hot(q, 6).double() + .1 / 6
    p.requires_grad_()
    cfg = J.Config()
    loss, _, _ = J.objective(p, p.detach(), h.double(), a.double(), len(old["x"]), cfg)
    gradient, = torch.autograd.grad(loss, p)
    direction = torch.randn(p.shape, generator=torch.Generator().manual_seed(22), dtype=torch.float64)
    direction *= .001
    eps = 1e-4
    plus = J.objective(p.detach() + eps * direction, p.detach(), h.double(), a.double(), 8, cfg)[0]
    minus = J.objective(p.detach() - eps * direction, p.detach(), h.double(), a.double(), 8, cfg)[0]
    torch.testing.assert_close((plus - minus) / (2 * eps), (gradient * direction).sum(), atol=1e-8, rtol=1e-4)


def test_soft_assignment_loss_and_source_constraints():
    old, delta, base, q, labels, h, a = inputs()
    cfg = replace(J.Config(), steps=32)
    p, coarse, audit = J.learn(q, labels, h, a, 8, cfg)
    assert audit["curve"][-1]["loss"] < audit["curve"][0]["loss"]
    assert audit["min_effective_sources"] > 1
    assert torch.count_nonzero(p[:6, 3:]) == 0 and torch.count_nonzero(p[6:, :3]) == 0
    torch.testing.assert_close(p.sum(1), torch.ones(12))
    candidates, material = J.materialize(base, delta, p, coarse, Encoder(), cfg)
    assert max(map(len, candidates["joint"]["texts"][8:])) <= 320
    assert torch.equal(candidates["content"]["edge_index"], base["edge_index"])
    assert torch.equal(candidates["topology"]["x"], base["x"])
    assert torch.equal(candidates["joint"]["text_emb"], candidates["content"]["text_emb"])
    torch.testing.assert_close(candidates["joint"]["x"][8:], (p.T @ delta["x"]) / p.sum(0)[:, None])
    for memory in candidates.values():
        assert memory["edge_index"].shape == base["edge_index"].shape
        edges = set(map(tuple, memory["edge_index"].T.tolist()))
        assert len(edges) == memory["edge_index"].shape[1]
        assert all((j, i) in edges and i != j for i, j in edges)
        assert torch.equal(memory["x"][:8], base["x"][:8])
    lift = J.lift_soft(old, delta, candidates["joint"], p)
    actual = J.M.adjacency(lift, "cpu").to_dense()[:8, :8]
    expected = J.R.C.training_adjacency(candidates["joint"], "cpu").to_dense()[:8, :8]
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-5)


def test_capacity_fixed_count_and_local_transfer():
    old, delta, base, q, labels, h, a = inputs()
    transferred = J.transfer_initial(q, labels, h[8:], 0, 1, "cpu")
    classes = J.row_classes(transferred, labels, 6)
    assert int((classes == 0).sum()) == 2 and int((classes == 1).sum()) == 4
    cfg = replace(J.Config(), steps=4)
    baseline = J.learn(q, labels, h, a, 8, cfg)
    selected, audit = J.capacity_search(q, labels, h, a, 8, cfg, "cpu", baseline)
    assert sum(audit["selected_allocation"].values()) == 6
    assert audit["rows_transferred"] <= 1
    assert selected[0].shape == baseline[0].shape
    assert len(audit["trials"]) == 3


def test_malformed_assignment_rejected():
    import pytest
    _, _, _, q, labels, _, _ = inputs()
    q[0] = 5
    with pytest.raises(ValueError, match="mixed-class"):
        J.row_classes(q, labels, 6)
