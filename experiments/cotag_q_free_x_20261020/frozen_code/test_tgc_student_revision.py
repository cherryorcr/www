"""Contracts for student-compatible historical-X revision."""
import torch
import torch.nn.functional as F
import pytest

import tgc_student_revision as S
import tgc_revision_gate as T


def memory(n=12, d=8, seed=4):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(n, d, generator=g),
            "text_emb": F.normalize(torch.randn(n, 384, generator=g), dim=1),
            "texts": [f"t{i}" for i in range(n)],
            "mass": torch.ones(n), "depth": torch.zeros(n, dtype=torch.long),
            "stage": torch.zeros(n, dtype=torch.long),
            "edge_index": torch.tensor([list(range(n - 1)), list(range(1, n))]),}


def support(old, new):
    block = {"x": new["x"], "text_emb": new["text_emb"], "texts": new["texts"],
             "mass": torch.ones(len(new["x"])), "depth": torch.zeros(len(new["x"]), dtype=torch.long),
             "stage": torch.ones(len(new["x"]), dtype=torch.long),
             "edge_index": new["edge_index"]}
    return T._concat_support(old, {**block, "current_delta_ids": torch.arange(len(new["x"]))})


def test_project_respects_radius():
    x = torch.randn(3, 4)
    S.project(x, .1)
    assert float(x.norm(dim=1).max()) <= .100001


def test_stop_gradient_is_exact_noop():
    old, new = memory(12), memory(8, seed=5)
    sup = support(old, new)
    cfg = S.Config(outer_steps=4)
    out, audit = S.optimize(old, sup, [1, 2], 1, 1, "cpu", cfg, stop=True)
    assert torch.equal(out["x"], old["x"])
    assert audit["nonzero_gradient_steps"] == 0


def test_student_revision_is_bounded_and_input_immutable():
    old, new = memory(12), memory(8, seed=6)
    sup = support(old, new)
    before = T.memory_hash(old)
    out, audit = S.optimize(old, sup, [1, 2], 1, 1, "cpu", S.Config(outer_steps=4))
    assert T.memory_hash(old) == before
    assert audit["nonzero_gradient_steps"] == 4
    S.invariants(old, out, [1, 2], 12, [], .1)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_unroll_matches_canonical_adam(device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    S.numerical_check(memory(), 7500, device)


@pytest.mark.parametrize("horizon", [1, 4, 8])
def test_hypergradient_matches_directional_finite_difference(horizon):
    m = memory()
    x, text = m["x"].double(), m["text_emb"].double()
    adj = S.R.C.training_adjacency(m, "cpu").double()
    ids = torch.tensor([1, 2])
    probe = torch.arange(4, 12)
    u = torch.zeros_like(x[ids], requires_grad=True)

    def loss_at(relative, differentiable):
        revised = S.changed_x(x, ids, relative)
        params = S.unroll(revised, adj, text, 7500, horizon, differentiable)
        return S.contrastive(params, x, adj, text, probe)

    grad, = torch.autograd.grad(loss_at(u, True), u)
    direction = grad.detach() / grad.detach().norm()
    analytic = (grad * direction).sum()
    assert torch.isfinite(grad).all() and float(grad.norm()) > 0
    # Adam's denominator can vary sharply near zero gradients. Check
    # convergence at two small double-precision perturbations, not one scale.
    for eps in (1e-7, 1e-8):
        numerical = (loss_at(eps * direction, False)
                     - loss_at(-eps * direction, False)) / (2 * eps)
        print({"horizon": horizon, "epsilon": eps,
               "analytic": float(analytic.detach()),
               "finite_difference": float(numerical.detach())})
        torch.testing.assert_close(analytic, numerical, atol=1e-6, rtol=.001)


def test_matched_random_has_equal_per_row_displacement():
    m = memory()
    candidate = T.clone_memory(m)
    candidate["x"][[1, 2]] += .01
    random = S.matched_random(m, candidate, [1, 2], 1)
    torch.testing.assert_close((candidate["x"] - m["x"]).norm(dim=1),
                               (random["x"] - m["x"]).norm(dim=1))
    S.invariants(m, random, [1, 2], 12, [], .1)
