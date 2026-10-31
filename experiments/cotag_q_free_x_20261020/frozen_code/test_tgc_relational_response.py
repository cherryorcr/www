"""Numerical and access-boundary contracts for the independent R4 experiment."""
from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

import tgc_relational_response as M

torch.set_num_threads(2)


def memory(n, seed):
    gen = torch.Generator().manual_seed(seed)
    edges = torch.stack((torch.arange(n), torch.arange(n).roll(1)))
    return {"x": torch.randn(n, 8, generator=gen),
            "text_emb": F.normalize(torch.randn(n, 384, generator=gen), dim=1),
            "texts": [f"text {i}" for i in range(n)],
            "mass": torch.ones(n), "depth": torch.zeros(n, dtype=torch.long),
            "stage": torch.zeros(n, dtype=torch.long),
            "edge_index": torch.cat((edges, edges.flip(0)), dim=1)}


def fixture():
    old, delta, block = memory(8, 1), memory(12, 2), memory(6, 3)
    delta["current_delta_ids"] = torch.arange(12)
    base = M.T._concat_support(old, block)
    base.pop("old_rows")
    base.pop("current_delta_ids")
    assignment = torch.arange(6).repeat_interleave(2)
    lift = M.lifted_support(old, delta, base, assignment)
    return old, delta, base, lift


def small_config():
    return M.Config(horizon=2, outer_steps=2, teacher_steps=2, direct_steps=2,
                    teacher_batch=8, train_seeds=(8100,), validation_seeds=(8110,),
                    teacher_seeds=(8000,), new_train_queries=8, new_validation_queries=2)


def test_lift_degree_and_normalized_old_block():
    old, delta, base, lift = fixture()
    old_n = len(old["x"])
    def dense_edges(m):
        weights = m.get("edge_weight", torch.ones(m["edge_index"].shape[1]))
        return torch.sparse_coo_tensor(m["edge_index"], weights.double(),
                                      (len(m["x"]), len(m["x"]))).to_dense()
    b, u = dense_edges(base), dense_edges(lift)
    torch.testing.assert_close(u, u.T)
    torch.testing.assert_close(u[:old_n].sum(1), b[:old_n].sum(1))
    torch.testing.assert_close(u[old_n:, old_n:], dense_edges(delta))
    bn = M.weighted_adjacency(base["edge_index"], len(b), "cpu", dtype=torch.float64).to_dense()
    un = M.weighted_adjacency(lift["edge_index"], len(u), "cpu", lift["edge_weight"],
                               torch.float64).to_dense()
    torch.testing.assert_close(bn[:old_n, :old_n], un[:old_n, :old_n], rtol=0, atol=1e-14)
    assert torch.count_nonzero(u.diagonal()) == 0


def test_duplicate_edges_have_additive_weight():
    e = torch.tensor([[0, 0, 1, 1], [1, 1, 0, 0]])
    duplicate = M.weighted_adjacency(e, 2, "cpu", torch.full((4,), .5)).to_dense()
    unique = M.weighted_adjacency(torch.tensor([[0, 1], [1, 0]]), 2, "cpu").to_dense()
    torch.testing.assert_close(duplicate, unique)


@pytest.mark.parametrize("mode", ["uniform", "compact"])
@pytest.mark.parametrize("batch", [8, 64, 100])
def test_sampler_unique_and_short_side(mode, batch):
    rows = M.sample_rows(40, 5, batch, torch.Generator().manual_seed(9), mode, .8)
    assert len(rows) == min(batch, 40) == len(rows.unique())
    if mode == "compact":
        assert int((rows < 5).sum()) == 5


def test_probe_split_and_relation_indexing():
    plan = M.probe_plan(20, 40, 1, small_config())
    assert not set(plan["train"].tolist()) & set(plan["validation"].tolist())
    logits = torch.eye(6) * 12.
    mask = torch.tensor([True, True, True, False, False, False])
    scores = M.stats(logits, mask)
    assert scores["old_retrieval"] == scores["new_retrieval"] == 1.
    assert scores["old_to_new_mass"] < .001
    assert M.divergence(logits, M.probabilities(logits), mask) < 1e-6
    assert M.divergence(logits.roll(1, 1), M.probabilities(logits), mask) > .001


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_canonical_unroll_1_4_32(device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    M.numerical_check(memory(12, 4), 8100, device, horizons=(1, 4, 32))


@pytest.fixture(scope="module")
def context():
    old, delta, base, lifted = fixture()
    contexts, audit = M.build_context(base, old, delta, lifted, 1, small_config(), "cpu")
    return base, contexts["lift_compact"], audit


def test_stop_is_noop_and_diagnostic_runs(context):
    base, ctx, audit = context
    candidate, info = M.optimize(base, ctx, [1], 1, "new", "relation", small_config(), "cpu", stop=True)
    assert M.T.memory_hash(candidate) == M.T.memory_hash(base)
    assert info["nonzero_gradient_steps"] == 0
    assert len(audit["supports"]) == 4


@pytest.mark.parametrize("location", ["new", "old", "joint"])
def test_edit_masks_and_immutability(context, location):
    base, ctx, _ = context
    before = M.T.memory_hash(base), M.T.memory_hash(ctx["support"])
    candidate, audit = M.optimize(base, ctx, [1], 1, location, "relation", small_config(), "cpu")
    allowed = ([1] if location == "old" else list(range(8, 14)))
    if location == "joint":
        allowed += [1]
    M.validate_memory(base, candidate, allowed, .1)
    assert audit["nonzero_gradient_steps"] == 2
    assert audit["final"]["train"]["old_kl"] <= .0010001
    assert before == (M.T.memory_hash(base), M.T.memory_hash(ctx["support"]))


def test_zero_constraint_backtracks(context):
    base, ctx, _ = context
    _, audit = M.optimize(base, ctx, [1], 1, "new", "relation",
                          replace(small_config(), old_epsilon=0.), "cpu")
    assert audit["final"]["train"]["old_kl"] <= 1e-7
    assert len(audit["feasibility"]) > 1


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_relation_hypergradient_finite_difference(device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    old, delta, base, support = fixture()
    x, text = base["x"].double().to(device), base["text_emb"].double().to(device)
    adj = M.adjacency(base, device).double()
    px, pt = support["x"].double().to(device), support["text_emb"].double().to(device)
    pa = M.adjacency(support, device).double()
    ids, rows = torch.arange(6, 15, device=device), torch.tensor([8, 9], device=device)
    target = M.probabilities(torch.eye(len(ids), dtype=torch.float64, device=device))
    mask = ids >= 8
    relative = torch.zeros_like(x[rows], requires_grad=True)
    def loss_at(u, differentiable):
        params = M.unroll(M.S.changed_x(x, rows, u), adj, text, 8100, 2, differentiable)
        return M.divergence(M.logits(params, px, pa, pt, ids, .07), target, mask)
    grad, = torch.autograd.grad(loss_at(relative, True), relative)
    direction = grad.detach() / grad.detach().norm()
    analytic = (grad * direction).sum()
    numeric = (loss_at(1e-8 * direction, False) - loss_at(-1e-8 * direction, False)) / 2e-8
    torch.testing.assert_close(analytic, numeric, rtol=.002, atol=1e-6)
