from dataclasses import replace

import pytest
import torch

import unified_q_multimodal as U
import external_continual_mechanisms as C
from test_unified_q_multimodal import DummyEncoder, _delta
import run_unified_q_inloop_cat_20261007 as R


def config(**kwargs):
    base = U.Config(steps=3, proxy_hidden=8, proxy_out=8, sinkhorn_steps=30,
                    sinkhorn_tolerance=1e-5, sinkhorn_max_steps=256,
                    lambda_distribution=1, distribution_hidden=16,
                    distribution_bank_size=3, distribution_chunk_size=5)
    return replace(base, **kwargs)


def test_distribution_matches_cat_nonlinear_class_mmd_and_preserves_rng():
    torch.manual_seed(12)
    real, synth = torch.randn(9, 4), torch.randn(5, 4, requires_grad=True)
    ry, sy = torch.tensor([2] * 6 + [5] * 3), torch.tensor([2] * 3 + [5] * 2)
    rng = torch.random.get_rng_state().clone()
    target = U._distribution_target(real, ry, config(), 9)
    assert torch.equal(rng, torch.random.get_rng_state())
    expected = torch.stack([C.class_mmd(C.random_features(real, w),
        C.random_features(synth, w), ry, sy) for w in target["weights"]]).mean()
    actual = U._distribution_loss(synth, sy, target)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual, synth, retain_graph=True)[0],
                               torch.autograd.grad(expected, synth)[0])
    assert not target["means"].requires_grad
    torch.testing.assert_close(U._distribution_loss(real, ry, target), torch.tensor(0.), atol=1e-12, rtol=0)


def test_nonlinear_distribution_reaches_q_logits_and_residual():
    torch.manual_seed(25)
    raw, ry, sy = torch.randn(12, 4), torch.tensor([0] * 6 + [1] * 6), torch.tensor([0, 0, 1, 1])
    logits = torch.randn(12, 4, requires_grad=True)
    residual = torch.zeros(4, 4, requires_grad=True)
    q = U._masked_sinkhorn(logits, ry, sy, 1., 100)
    x = q.T @ raw / q.sum(0)[:, None] + residual
    # A linear class mean is already conserved by balanced Q and cannot train it.
    for c in (0, 1):
        torch.testing.assert_close(x[sy == c].mean(0), raw[ry == c].mean(0))
    target = U._distribution_target(raw, ry, config(), 14)
    loss = U._distribution_loss(x, sy, target)
    gq, gr = torch.autograd.grad(loss, (logits, residual), create_graph=True)
    assert gq.abs().sum() > 1e-6 and gr.abs().sum() > 1e-6
    for h in torch.autograd.grad(gq.square().sum() + gr.square().sum(), (logits, residual)):
        assert torch.isfinite(h).all()


def test_target_chunking_is_exact_and_checks_class_support():
    torch.manual_seed(18)
    raw, y = torch.randn(13, 4), torch.tensor([0] * 7 + [1] * 6)
    a = U._distribution_target(raw, y, config(distribution_chunk_size=2), 8)
    b = U._distribution_target(raw, y, config(distribution_chunk_size=100), 8)
    torch.testing.assert_close(a["weights"], b["weights"])
    torch.testing.assert_close(a["means"], b["means"])
    with pytest.raises(ValueError, match="classes"):
        U._distribution_loss(raw[:7], y[:7], a)


@pytest.mark.parametrize("topology", [False, True])
def test_inloop_has_no_cat_warm_start_and_keeps_prefix(topology):
    torch.manual_seed(42)
    delta, prompt, labels = _delta(), torch.randn(2, 8), torch.tensor([0, 0, 0, 1, 1, 1])
    cfg = config()
    parent, _ = U.optimize_stage(delta, None, labels, DummyEncoder(), prompt,
        replace(cfg, steps=0), "cpu", use_boundary=False, use_topology=topology)
    old_x, old_text, old_edges = parent["x"].clone(), parent["text_emb"].clone(), parent["edge_index"].clone()
    block, audit = U.optimize_stage(_delta(1, 12), parent, labels, DummyEncoder(), prompt,
        cfg, "cpu", use_boundary=False, use_topology=topology)
    memory = U.append_memory(parent, block)
    torch.testing.assert_close(memory["x"][:6], old_x, rtol=0, atol=0)
    torch.testing.assert_close(memory["text_emb"][:6], old_text, rtol=0, atol=0)
    torch.testing.assert_close(parent["edge_index"], old_edges)
    assert audit["cat_anchor_used"] is False
    assert audit["q_update_count"] == audit["residual_update_count"] == cfg.steps
    assert audit["text_hard_forward_max_error"] < 1e-5
    assert audit["x_residual_ratio_max"] <= cfg.trust_radius + 1e-5
    stats = audit["distribution_matching"]
    assert stats["enabled"] and not stats["warm_start"]
    assert not stats["holdout_used_for_optimization"]
    assert all(torch.isfinite(torch.tensor(stats[k])) for k in (
        "initial_train", "final_train", "initial_holdout_bank", "final_holdout_bank"))


def test_distribution_only_learns_without_proxy_or_cat(monkeypatch):
    torch.manual_seed(17)
    delta, labels, prompt = _delta(), torch.tensor([0, 0, 0, 1, 1, 1]), torch.randn(2, 8)
    def fail(*args, **kwargs):
        raise AssertionError("must not instantiate a proxy or run CaT")
    monkeypatch.setattr(U.P, "GraphTextPromptStudent", fail)
    monkeypatch.setattr(C, "cat_block", fail)
    cfg = config(steps=20, lambda_gradient=0, lambda_propagation=0,
                 lambda_q_anchor=0, lambda_x_anchor=0)
    _, audit = U.optimize_stage(delta, None, labels, DummyEncoder(), prompt, cfg,
        "cpu", use_boundary=False, use_topology=False)
    stats = audit["distribution_matching"]
    assert stats["final_train"] < stats["initial_train"]
    assert audit["trace"][0]["q_gradient_norm"] > 0
    assert audit["trace"][0]["residual_gradient_norm"] > 0
    assert all(row["gradient_match"] == 0 for row in audit["trace"])


def test_disabled_distribution_does_not_build_bank(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("disabled distribution must preserve legacy execution")
    monkeypatch.setattr(U, "_distribution_target", fail)
    _, audit = U.optimize_stage(_delta(), None, torch.tensor([0, 0, 1, 1]),
        DummyEncoder(), torch.randn(2, 8), config(lambda_distribution=0),
        "cpu", use_boundary=False, use_topology=False)
    assert not audit["distribution_matching"]["enabled"]


def test_pretrained_cat_anchor_rejected_for_inloop():
    with pytest.raises(ValueError, match="pre-trained CaT"):
        U.optimize_stage(_delta(), None, torch.tensor([0, 1]), DummyEncoder(),
            torch.randn(2, 8), config(), "cpu", cat_anchor_x=torch.zeros(2, 4))


def test_arms_are_single_factor_controls():
    from dataclasses import asdict
    for left, right, changed in (("uq_v3", "uq_dm_joint", "lambda_distribution"),
            ("uq_edgefree384", "uq_dm_edgefree", "lambda_distribution"),
            ("uq_dm_edgefree", "uq_dm_edgefree_no_gm", "lambda_gradient"),
            ("uq_dm_edgefree", "uq_dm_edgefree_fixed_q", "train_q")):
        a, b = asdict(R.config(left, 10)), asdict(R.config(right, 10))
        assert {k for k in a if a[k] != b[k]} == {changed}
        assert R.use_topology(left) == R.use_topology(right)
