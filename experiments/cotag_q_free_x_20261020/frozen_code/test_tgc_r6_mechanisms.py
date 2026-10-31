"""R6 numerical, representation and intervention contracts."""
from dataclasses import replace

import torch
import torch.nn.functional as F
import tgc_r6_mechanisms as K
from test_tgc_shared_assignment import inputs, Encoder
from test_tgc_relational_response import small_config

torch.set_num_threads(2)


def test_parent_factorial_changes_exact_fields():
    old, _, _, _, _, _, _ = inputs()
    revised = K.T.clone_memory(old)
    revised["x"] += .1
    revised["text_emb"] += .2
    revised["texts"] = [t + " changed" for t in old["texts"]]
    revised["edge_index"] = old["edge_index"].roll(2, dims=1)
    for bits in ("000", "100", "010", "001", "111"):
        result = K.blend_parent(old, revised, bits)
        assert torch.equal(result["x"], (revised if bits[0] == "1" else old)["x"])
        assert result["texts"] == (revised if bits[1] == "1" else old)["texts"]
        assert torch.equal(result["edge_index"], (revised if bits[2] == "1" else old)["edge_index"])


def test_symmetric_operator_matches_canonical():
    _, _, base, _, _, _, _ = inputs()
    raw = K.J.raw_adjacency(base, "cpu")
    actual = K.symmetric_sparse(raw).to_dense()
    expected = K.R.C.training_adjacency(base, "cpu").to_dense()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(K.symmetric_dense(raw.to_dense()), expected)


def test_symmetric_objective_directional_gradient():
    _, _, _, q, _, h, a = inputs()
    p = (.9 * F.one_hot(q, 6).double() + .1/6).requires_grad_()
    h, a = h.double(), a.double()
    cfg = K.J.Config()
    loss = K.objective(p, p.detach(), h, a, 8, cfg, "symmetric")[0]
    grad, = torch.autograd.grad(loss, p)
    direction = torch.randn(p.shape, dtype=torch.float64, generator=torch.Generator().manual_seed(113)) * .001
    eps = 1e-4
    high = K.objective(p.detach()+eps*direction, p.detach(), h, a, 8, cfg, "symmetric")[0]
    low = K.objective(p.detach()-eps*direction, p.detach(), h, a, 8, cfg, "symmetric")[0]
    torch.testing.assert_close((high-low)/(2*eps), (grad*direction).sum(), atol=1e-8, rtol=1e-4)


def test_step_zero_and_modes_keep_sources():
    _, _, _, q, labels, h, a = inputs()
    ctx = {"h": h, "adjacency": a}
    cfg = replace(K.J.Config(), steps=0)
    p, _, _ = K.learn(q, labels, ctx, 8, cfg)
    uniform = (labels[:, None] == torch.tensor([0, 0, 0, 1, 1, 1])[None]).float()/3
    expected = (1-cfg.floor)*((1-cfg.initial_uniform)*F.one_hot(q, 6)+cfg.initial_uniform*uniform)+cfg.floor*uniform
    torch.testing.assert_close(p, expected)
    for mode in ("rw", "symmetric"):
        p, _, info = K.learn(q, labels, ctx, 8, replace(cfg, steps=8), mode)
        assert info["curve"][-1]["loss"] < info["curve"][0]["loss"]
        assert torch.count_nonzero(p[:6, 3:]) == 0


def test_weighted_same_edges_and_explicit_unit_control():
    _, delta, base, q, labels, h, a = inputs()
    p, coarse, _ = K.learn(q, labels, {"h": h, "adjacency": a}, 8, replace(K.J.Config(), steps=4))
    memory, _ = K.materialize(base, delta, p, coarse, Encoder(), K.J.Config())
    control = K.weighted_variant(memory, torch.ones(len(memory["x"]), len(memory["x"])))
    torch.testing.assert_close(K.R.C.training_adjacency(control, "cpu").to_dense(),
                               K.R.C.training_adjacency(memory, "cpu").to_dense(), rtol=0, atol=0)
    weighted = K.weighted_variant(memory, coarse)
    e = weighted["edge_index"]
    assert torch.equal(e[:, e[0] != e[1]], memory["edge_index"])
    torch.testing.assert_close(K.R.C.training_adjacency(weighted, "cpu").to_dense(),
                               K.R.C.training_adjacency(weighted, "cpu").to_dense().T)
    pooled = K.pooled_text_variant(memory, delta, p)
    assert pooled["text_representation_scope"].startswith("embedding-only")
    assert torch.equal(pooled["text_emb"][:8], memory["text_emb"][:8])


def test_trained_views_reproduce_r5():
    old, delta, base, q, labels, h, a = inputs()
    cfg = small_config()
    ctx = K.context(old, delta, base, q, cfg, "cpu")
    expected, _ = K.J.teacher_views(ctx["support"], base, cfg, "cpu")
    torch.testing.assert_close(ctx["h"], expected, rtol=0, atol=0)
