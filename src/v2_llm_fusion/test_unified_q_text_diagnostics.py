from dataclasses import replace

import torch

import unified_q_multimodal as U
from unified_q_text_diagnostics import TextPathDiagnostics, q_text_medoid, payload_metrics
from unified_q_payload_projection import q_text_coverage
from test_unified_q_multimodal import DummyEncoder, _delta
from test_unified_q_distribution import config


def test_medoid_minimizes_weighted_distortion_and_is_class_legal():
    torch.manual_seed(44)
    text = torch.nn.functional.normalize(torch.randn(9, 8), dim=1)
    raw_labels = torch.tensor([0] * 5 + [1] * 4)
    labels = torch.tensor([0, 0, 1])
    q = U._masked_sinkhorn(torch.randn(9, 3), raw_labels, labels, 1, 100)
    selected = q_text_medoid(q, text, raw_labels, labels, chunk=1)
    assert torch.equal(raw_labels[selected], labels)
    for j in range(3):
        costs = (q[:, j, None] * (1 - text @ text.T)).sum(0) / q[:, j].sum()
        best = costs[raw_labels == labels[j]].min()
        torch.testing.assert_close(costs[selected[j]], best)
    result = payload_metrics(q, text, raw_labels, labels)
    assert result["macro"]["medoid_distortion"] <= result["macro"]["native_distortion"] + 1e-6


def test_coverage_projection_is_class_legal_and_uses_distinct_supports():
    text = torch.nn.functional.normalize(torch.tensor([
        [1., 0., 0.], [0.98, 0.02, 0.], [0., 1., 0.],
        [0., 0.98, 0.02], [0., 0., 1.], [0., 0., 0.98],
    ]), dim=1)
    raw_labels = torch.tensor([0, 0, 0, 0, 1, 1])
    labels = torch.tensor([0, 0, 1])
    q = U._masked_sinkhorn(torch.tensor([
        [5., 0., -10.], [4.5, .1, -10.], [0., 5., -10.],
        [.1, 4.5, -10.], [-10, -10, 4.], [-10, -10, 4.],
    ]), raw_labels, labels, 1, 100)
    selected = q_text_coverage(q, text, raw_labels, labels)
    assert torch.equal(raw_labels[selected], labels)
    assert len(torch.unique(selected[labels == 0])) == 2


def test_coverage_forward_and_materialization_are_identical():
    torch.manual_seed(48)
    delta, labels, prompt = _delta(), torch.tensor([0, 0, 1, 1]), torch.randn(2, 8)
    cfg = replace(config(steps=3), text_materialization="q_coverage")
    captured = {}
    block, audit = U.optimize_stage(delta, None, labels, DummyEncoder(), prompt, cfg,
        "cpu", use_boundary=False, use_topology=False,
        diagnostic_callback=captured.update)
    chosen = q_text_coverage(captured["q"], captured["delta"]["text_emb"],
        captured["delta"]["y"], labels)
    assert block["text_source_ids"] == [[int(captured["delta"]["ids"][i])] for i in chosen]
    assert audit["text_hard_forward_max_error"] < 1e-5
    assert audit["text_projection"]["rule"] == "q_coverage"


def test_callback_preserves_result_rng_and_does_not_accumulate_grad():
    torch.manual_seed(77)
    delta, labels, prompt = _delta(), torch.tensor([0, 0, 1, 1]), torch.randn(2, 8)
    cfg = config(steps=3)
    diag = TextPathDiagnostics(cfg.steps)
    result = {}
    def observe(state):
        before = None if state["q_logits"].grad is None else state["q_logits"].grad.clone()
        diag(state)
        after = state["q_logits"].grad
        assert (before is None and after is None) or torch.equal(before, after)
        torch.rand(20)
    baseline, _ = U.optimize_stage(delta, None, labels, DummyEncoder(), prompt, cfg,
        "cpu", use_boundary=False, use_topology=False)
    rng = torch.get_rng_state().clone()
    observed, _ = U.optimize_stage(delta, None, labels, DummyEncoder(), prompt, cfg,
        "cpu", use_boundary=False, use_topology=False, optimization_callback=observe,
        diagnostic_callback=lambda state: result.update(diag.final(state)))
    for key in ("x", "text_emb", "labels", "mass", "edge_index"):
        torch.testing.assert_close(baseline[key], observed[key], rtol=0, atol=0)
    assert baseline["text_source_ids"] == observed["text_source_ids"]
    assert torch.equal(rng, torch.get_rng_state())
    assert len(result["snapshots"]) == 2
    assert result["snapshots"][0]["gradients"]["direct_text_q_norm"] > 0


def test_detached_text_has_zero_direct_gradient():
    diag = TextPathDiagnostics(1)
    U.optimize_stage(_delta(), None, torch.tensor([0, 0, 1, 1]), DummyEncoder(),
        torch.randn(2, 8), replace(config(steps=1), text_assignment="detached"),
        "cpu", use_boundary=False, use_topology=False, optimization_callback=diag)
    assert diag.snapshots[0]["gradients"]["direct_text_q_norm"] == 0


def test_medoid_forward_and_materialization_are_identical_with_frozen_prefix():
    torch.manual_seed(47)
    delta, labels, prompt = _delta(), torch.tensor([0, 0, 1, 1]), torch.randn(2, 8)
    cfg = replace(config(steps=3), text_materialization="q_text_medoid")
    parent, _ = U.optimize_stage(delta, None, labels, DummyEncoder(), prompt, cfg,
        "cpu", use_boundary=False, use_topology=False)
    captured = {}
    block, audit = U.optimize_stage(_delta(1, 12), parent, labels, DummyEncoder(), prompt, cfg,
        "cpu", use_boundary=False, use_topology=False, diagnostic_callback=captured.update)
    chosen = q_text_medoid(captured["q"], captured["delta"]["text_emb"],
        captured["delta"]["y"], labels)
    assert block["text_source_ids"] == [[int(captured["delta"]["ids"][i])] for i in chosen]
    assert audit["text_hard_forward_max_error"] < 1e-5
    memory = U.append_memory(parent, block)
    torch.testing.assert_close(memory["x"][:len(labels)], parent["x"], rtol=0, atol=0)
    torch.testing.assert_close(memory["text_emb"][:len(labels)], parent["text_emb"], rtol=0, atol=0)
    assert audit["q_update_count"] == cfg.steps


def test_invalid_materialization_rejected():
    import pytest
    with pytest.raises(ValueError, match="unknown hard text"):
        U.optimize_stage(_delta(), None, torch.tensor([0, 1]), DummyEncoder(),
            torch.randn(2, 8), replace(config(), text_materialization="other"), "cpu")
