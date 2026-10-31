from dataclasses import replace

import pytest
import torch

import unified_q_gap_diagnostics as D
import unified_q_multimodal as U
from test_unified_q_multimodal import DummyEncoder, _delta
from test_unified_q_distribution import config


def test_geometry_distinguishes_scale_from_direction():
    raw = torch.tensor([[1., 0.], [1., 0.], [0., 2.], [0., 2.]])
    labels = torch.tensor([0, 0, 1, 1])
    result = D.class_geometry(raw, 3 * raw, labels, raw, labels, .05)
    for record in result["classes"]:
        assert record["raw_message_cosine"] == pytest.approx(1.)
        assert record["distance_over_bound"] > 30


def test_observer_preserves_original_and_measures_residual():
    torch.manual_seed(42)
    delta, labels, prompt = _delta(), torch.tensor([0, 0, 1, 1]), torch.randn(2, 8)
    cfg = config(steps=3)
    baseline, _ = U.optimize_stage(delta, None, labels, DummyEncoder(), prompt,
        cfg, "cpu", use_boundary=False, use_topology=False)
    records = []
    observed, _ = U.optimize_stage(delta, None, labels, DummyEncoder(), prompt,
        cfg, "cpu", use_boundary=False, use_topology=False,
        diagnostic_callback=lambda state: records.append(D.representation_diagnostic(state)))
    torch.testing.assert_close(baseline["x"], observed["x"], rtol=0, atol=0)
    assert baseline["texts"] == observed["texts"]
    assert records[0]["residual_cap_fraction_max"] <= 1.0001
    assert records[0]["holdout_dm"] >= 0
    assert all(c["pool_class_mean_error"] < 1e-4 for c in records[0]["classes"])


def test_slice_does_not_include_future_rows():
    memory = {k: torch.arange(6) for k in ("x", "text_emb", "labels", "mass", "depth", "stage")}
    memory.update(texts=list("abcdef"), text_source_ids=[[i] for i in range(6)],
                  edge_index=torch.empty(2, 0, dtype=torch.long))
    block = D.edgefree_slice(memory, 2, 4)
    assert block["texts"] == ["c", "d"]
    assert block["text_source_ids"] == [[2], [3]]


def test_batched_nearest_preserves_content_and_matches_policy():
    import external_continual_mechanisms as M
    torch.manual_seed(17)
    delta = _delta()
    labels = torch.tensor([0, 0, 1, 1])
    base = dict(x=torch.randn(4, 4), labels=labels)
    class Encoder:
        def encode(self, texts):
            return torch.tensor(DummyEncoder().encode(texts))
    encoder = Encoder()
    results = D.common_nearest_payloads(delta, delta["x"], base, encoder)
    for k, result in results.items():
        expected, audit = M.attach_text(delta, delta["x"], base["x"], labels, encoder, M.Config(text_sources=k))
        assert result["texts"] == expected["texts"]
        assert result["text_source_ids"] == audit["text_source_ids"]
        assert result["x"] is base["x"]
        for cls, ids in zip(labels, result["text_source_ids"]):
            assert all(delta["y"][i] == cls for i in ids)


@pytest.mark.parametrize("steps", [0, 3])
def test_message_pooling_uses_same_source_at_init_update_projection_and_save(steps):
    torch.manual_seed(9)
    delta, labels, prompt = _delta(), torch.tensor([0, 0, 1, 1]), torch.randn(2, 8)
    observed = []
    cfg = config(steps=steps, feature_pooling="message")
    block, audit = U.optimize_stage(delta, None, labels, DummyEncoder(), prompt, cfg,
        "cpu", use_boundary=False, use_topology=False,
        diagnostic_callback=lambda state: observed.append(state))
    state=observed[0]
    q=state["q"]
    pooled=q.T @ state["target_message"] / q.sum(0)[:,None]
    ratio=(block["x"]-pooled).norm(dim=1)/pooled.norm(dim=1).clamp_min(1e-3)
    assert float(ratio.max()) <= cfg.trust_radius+1e-5
    if steps == 0:
        torch.testing.assert_close(block["x"],pooled,rtol=0,atol=0)
    else:
        assert audit["trace"][0]["q_gradient_norm"] > 0
        assert audit["trace"][0]["residual_gradient_norm"] > 0
    assert audit["text_hard_forward_max_error"] < 1e-5
    assert audit["feature_pooling"] == "message"
    assert not block["edge_index"].numel()


def test_message_pooling_keeps_initial_q_and_frozen_prefix():
    torch.manual_seed(32)
    delta, labels, prompt = _delta(), torch.tensor([0, 0, 1, 1]), torch.randn(2, 8)
    states=[]
    cfg=config(steps=0)
    blocks=[]
    for pooling in ("raw", "message"):
        block,_=U.optimize_stage(delta,None,labels,DummyEncoder(),prompt,
            replace(cfg,feature_pooling=pooling),"cpu",use_boundary=False,use_topology=False,
            diagnostic_callback=lambda s:states.append(s))
        blocks.append(block)
    torch.testing.assert_close(states[0]["q0"],states[1]["q0"],rtol=0,atol=0)
    assert blocks[0]["texts"]==blocks[1]["texts"]
    parent=blocks[1]
    old=parent["x"].clone()
    block,_=U.optimize_stage(_delta(1,12),parent,labels,DummyEncoder(),prompt,
        config(steps=2,feature_pooling="message"),"cpu",use_boundary=False,use_topology=False)
    memory=U.append_memory(parent,block)
    torch.testing.assert_close(memory["x"][:len(old)],old,rtol=0,atol=0)


def test_message_pooling_propagation_is_identity_without_residual():
    torch.manual_seed(2)
    _,audit=U.optimize_stage(_delta(),None,torch.tensor([0,0,1,1]),DummyEncoder(),torch.randn(2,8),
        config(steps=2,feature_pooling="message",train_residual=False,propagation_metric="cosine"),
        "cpu",use_boundary=False,use_topology=False)
    assert max(abs(row["propagation"]) for row in audit["trace"]) < 1e-6


def test_unknown_pooling_and_double_propagation_rejected():
    args=(_delta(),None,torch.tensor([0,1]),DummyEncoder(),torch.randn(2,8))
    with pytest.raises(ValueError,match="pooling"):
        U.optimize_stage(*args,config(feature_pooling="unknown"),"cpu",use_topology=False)
    with pytest.raises(ValueError,match="edge-free"):
        U.optimize_stage(*args,config(feature_pooling="message"),"cpu",use_topology=True)
