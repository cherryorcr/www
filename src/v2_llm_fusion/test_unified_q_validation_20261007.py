from dataclasses import replace
import random
import numpy as np
import torch
import pytest
import unified_q_multimodal as U
import run_unified_q_matched_baselines_20261007 as B
import run_unified_q_fixed_content_20261007 as D
from test_unified_q_multimodal import DummyEncoder, _delta


@pytest.mark.parametrize("arm", ["random_coreset", "kcenter_coreset"])
def test_coreset_budget_unique_legal_and_reencoded(arm):
    torch.manual_seed(13)
    delta = _delta()
    block, audit = B.coreset(delta, 6, arm, DummyEncoder(), 91, "cpu")
    assert len(block["x"]) == 6
    assert len(set(audit["source_ids"])) == 6
    assert set(audit["source_ids"]) <= set(delta["ids"].tolist())
    assert torch.bincount(block["labels"]).tolist() == [3, 3]
    torch.testing.assert_close(block["text_emb"], torch.tensor(DummyEncoder().encode(block["texts"])))
    again, _ = B.coreset(delta, 6, arm, DummyEncoder(), 91, "cpu")
    assert torch.equal(block["x"], again["x"])


def test_callback_leaves_method_identical_and_topology_preserves_counts():
    torch.manual_seed(42)
    delta, prompt, labels = _delta(), torch.randn(2,8), torch.tensor([0,0,0,1,1,1])
    cfg = U.Config(steps=2, proxy_hidden=8, proxy_out=8, lambda_distribution=1,
                   distribution_hidden=12, sinkhorn_steps=32)
    expected, expected_audit = U.optimize_stage(delta, None, labels, DummyEncoder(), prompt,
                                cfg, "cpu", use_boundary=False, use_topology=False)
    captured = {}
    def callback(state):
        assert not state["q"].requires_grad
        captured.update(D.topology_snapshot(state))
    actual, audit = U.optimize_stage(delta, None, labels, DummyEncoder(), prompt,
                  cfg, "cpu", use_boundary=False, use_topology=False, diagnostic_callback=callback)
    for key in ("x", "text_emb", "labels", "edge_index", "mass"):
        assert torch.equal(actual[key], expected[key])
    assert audit == expected_audit
    assert captured["hard"].shape == captured["permuted"].shape
    assert torch.isfinite(captured["weight"]).all()
    assert torch.equal(captured["weight"].diag(), torch.zeros(6))


def test_permutation_same_class_does_not_change_payload_multiset():
    labels = torch.tensor([0,0,0,1,1,2])
    order = D.permutation(labels, 14, True)
    assert torch.equal(labels[order], labels)
    assert sorted(order.tolist()) == list(range(6))


def test_dense_weighted_operator_has_same_binary_limit():
    edge = torch.tensor([[0,1],[1,2]])
    raw = U._dense_adjacency(edge, 3, "cpu")
    torch.testing.assert_close(U._normalize_dense(raw), D.P.normalized_adjacency(edge,3,"cpu").to_dense())


@pytest.mark.parametrize("raises", [False, True])
def test_callback_preserves_rng_streams_even_on_error(raises):
    torch.manual_seed(42)
    delta, prompt, labels = _delta(), torch.randn(2,8), torch.tensor([0,0,0,1,1,1])
    cfg = U.Config(steps=0, proxy_hidden=8, proxy_out=8, sinkhorn_steps=32)
    before = {}
    def callback(state):
        before["python"] = random.getstate()
        before["numpy"] = np.random.get_state()
        before["torch"] = torch.get_rng_state().clone()
        before["cuda"] = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        random.random()
        np.random.random(10)
        torch.rand(10)
        if before["cuda"]:
            for device in range(torch.cuda.device_count()):
                torch.rand(10, device=f"cuda:{device}")
        if raises:
            raise RuntimeError("intentional callback failure")
    def run():
        U.optimize_stage(delta, None, labels, DummyEncoder(), prompt, cfg, "cpu",
            use_boundary=False, use_topology=False, diagnostic_callback=callback)
    if raises:
        with pytest.raises(RuntimeError, match="intentional callback failure"):
            run()
    else:
        run()
    assert random.getstate() == before["python"]
    now = np.random.get_state()
    assert now[0] == before["numpy"][0] and now[2:] == before["numpy"][2:]
    np.testing.assert_array_equal(now[1], before["numpy"][1])
    assert torch.equal(torch.get_rng_state(), before["torch"])
    for actual, expected in zip(torch.cuda.get_rng_state_all() if before["cuda"] else [], before["cuda"]):
        assert torch.equal(actual, expected)
