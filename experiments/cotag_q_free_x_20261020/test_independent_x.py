from dataclasses import replace

import pytest
import torch

import independent_x as C
import unified_q_multimodal as U
from test_unified_q_multimodal import DummyEncoder, _delta


LABELS = torch.tensor([0, 0, 0, 1, 1, 1])
torch.set_num_threads(2)


def config(**kwargs):
    return replace(U.Config(steps=3, proxy_hidden=8, proxy_out=8,
        sinkhorn_steps=30, sinkhorn_tolerance=1e-5, sinkhorn_max_steps=256,
        lambda_distribution=1, distribution_hidden=16, distribution_bank_size=3,
        distribution_chunk_size=5), **kwargs)


def test_initial_state_and_initial_objective_match_full():
    torch.manual_seed(53)
    delta, prompt = _delta(), torch.randn(2, 8)
    base = config(sinkhorn_max_steps=2048)
    a, _ = U.optimize_stage(delta, None, LABELS, DummyEncoder(), prompt,
        replace(base, steps=0), "cpu", use_boundary=False, use_topology=False)
    b, _ = C.optimize(delta, None, LABELS, DummyEncoder(), prompt,
        replace(base, steps=0, lambda_x_anchor=0), "cpu")
    for key in ("x", "text_emb", "labels", "mass"):
        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
    _, aa = U.optimize_stage(delta, None, LABELS, DummyEncoder(), prompt,
        replace(base, steps=1), "cpu", use_boundary=False, use_topology=False)
    _, bb = C.optimize(delta, None, LABELS, DummyEncoder(), prompt,
        replace(base, steps=1, lambda_x_anchor=0), "cpu")
    for key in ("loss", "gradient_match", "distribution_match", "propagation", "q_anchor"):
        assert aa["trace"][0][key] == pytest.approx(bb["trace"][0][key], abs=1e-7)


def test_control_can_leave_original_region_and_keeps_real_text_prefix():
    torch.manual_seed(52)
    delta, prompt = _delta(), torch.randn(2, 8)
    parent, _ = C.optimize(delta, None, LABELS, DummyEncoder(), prompt,
        config(steps=0, lambda_x_anchor=0), "cpu")
    old = {k: v.clone() if torch.is_tensor(v) else list(v) for k, v in parent.items()}
    block, audit = C.optimize(_delta(1, 12), parent, LABELS, DummyEncoder(), prompt,
        config(steps=3, trust_radius=0, lambda_x_anchor=0), "cpu")
    assert audit["x_displacement_ratio_max"] > .01
    assert audit["fraction_outside_original_trust_radius"] > 0
    assert all(r["x_anchor"] == 0 for r in audit["trace"])
    assert audit["trace"][0]["free_x_gradient_norm"] > 0
    assert audit["trace"][0]["q_gradient_norm"] > 0
    for key in ("x", "text_emb", "labels", "mass"):
        assert torch.equal(parent[key], old[key])
    assert parent["texts"] == old["texts"]
    assert audit["text_hard_forward_max_error"] < 1e-6
    assert not block["edge_index"].numel()


def test_rejects_accidentally_active_x_anchor():
    with pytest.raises(ValueError, match="no X anchor"):
        C.optimize({}, None, LABELS, None, None, config(), "cpu")
