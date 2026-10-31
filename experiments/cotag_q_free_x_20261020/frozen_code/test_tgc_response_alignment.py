"""Fixed-prefix trajectory equivalence and historical-X gradient contracts."""
import pytest
import torch

import tgc_response_alignment as A
from test_tgc_student_revision import memory, support


@pytest.mark.parametrize("warmup", [0, 64, 256])
@pytest.mark.parametrize("horizon", [1, 4])
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_prefix_suffix_matches_uninterrupted_adam(warmup, horizon, device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    A.numerical_check(memory(), 7500, warmup, horizon, device)


@pytest.mark.parametrize("horizon", [1, 4])
def test_zero_prefix_reproduces_previous_unroll(horizon):
    m = memory()
    x, text = m["x"], m["text_emb"]
    adj = A.R.C.training_adjacency(m, "cpu")
    prefix = A.make_prefix(x, adj, text, 7500, 0)
    actual = A.suffix(prefix, x, adj, text, horizon)
    expected = A.S.unroll(x, adj, text, 7500, horizon)
    for key in actual:
        torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)


@pytest.mark.parametrize("warmup", [64, 256])
def test_suffix_directional_finite_difference(warmup):
    m = memory()
    x, text = m["x"].double(), m["text_emb"].double()
    adj = A.R.C.training_adjacency(m, "cpu").double()
    prefix = A.make_prefix(x, adj, text, 7500, warmup)
    rows, probe = torch.tensor([1, 2]), torch.arange(4, 12)
    u = torch.zeros_like(x[rows], requires_grad=True)

    def objective(value, differentiable):
        params = A.suffix(prefix, A.S.changed_x(x, rows, value), adj, text, 4, differentiable)
        return A.S.contrastive(params, x, adj, text, probe)

    grad, = torch.autograd.grad(objective(u, True), u)
    assert torch.isfinite(grad).all() and float(grad.norm()) > 0
    direction = grad.detach() / grad.detach().norm()
    analytical = (grad * direction).sum()
    for epsilon in (1e-6, 1e-7):
        numerical = (objective(direction * epsilon, False)
                     - objective(-direction * epsilon, False)) / (2 * epsilon)
        torch.testing.assert_close(analytical, numerical, atol=1e-6, rtol=.001)


def test_optimization_preserves_prefixes_inputs_and_stop():
    old, new = memory(), memory(8, seed=7)
    probe = support(old, new)
    cfg = A.S.Config(outer_steps=4)
    prefixes, _ = A.prepare_prefixes(old, 64, "cpu", cfg)
    before = {s: A.prefix_hash(p) for s, p in prefixes.items()}
    out, audit = A.optimize(old, probe, [1, 2], 1, 1, prefixes, "cpu", cfg)
    assert audit["nonzero_gradient_steps"] > 0
    A.S.invariants(old, out, [1, 2], 12, [], .1)
    stop, stop_audit = A.optimize(old, probe, [1, 2], 1, 1, prefixes, "cpu", cfg, stop=True)
    assert torch.equal(stop["x"], old["x"])
    assert stop_audit["nonzero_gradient_steps"] == 0
    assert {s: A.prefix_hash(p) for s, p in prefixes.items()} == before
