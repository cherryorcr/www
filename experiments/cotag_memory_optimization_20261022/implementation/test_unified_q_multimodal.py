from __future__ import annotations

import pathlib
import sys
from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import unified_q_multimodal as U
import graphtext_prompt_student as P


class DummyEncoder:
    def encode(self, texts):
        rows = []
        for text in texts:
            seed = sum(ord(char) for char in text) % 10000
            generator = torch.Generator().manual_seed(seed)
            rows.append(torch.randn(8, generator=generator).tolist())
        return rows


def _delta(stage=0, offset=0):
    n = 12
    return {
        "ids": torch.arange(offset, offset + n),
        "stage": stage,
        "x": torch.randn(n, 4),
        "y": torch.tensor([0] * 6 + [1] * 6),
        "text_emb": torch.randn(n, 8),
        "texts": [f"node {offset + i}" for i in range(n)],
        "edge_index": torch.tensor([
            [0, 1, 1, 2, 3, 4, 6, 7, 7, 8, 9, 10],
            [1, 0, 2, 1, 4, 3, 7, 6, 8, 7, 10, 9],
        ]),
        "boundary_edge_index": torch.empty((2, 0), dtype=torch.long),
    }


def test_unified_q_builds_hard_consistent_block():
    torch.manual_seed(7)
    cfg = U.Config(steps=2, proxy_hidden=8, proxy_out=8,
                   sinkhorn_steps=8, materialize_top_k=1)
    block, audit = U.optimize_stage(
        _delta(), None, torch.tensor([0, 0, 0, 1, 1, 1]),
        DummyEncoder(), torch.randn(2, 8), cfg, "cpu",
        use_boundary=False, use_topology=True)
    assert len(block["x"]) == 6
    assert len(block["texts"]) == 6
    assert block["edge_index"].shape[0] == 2
    assert audit["straight_through_text"] is True
    assert torch.isfinite(block["text_emb"]).all()


def test_boundary_mapping_uses_parent_support_ids_only():
    torch.manual_seed(11)
    cfg = U.Config(steps=1, proxy_hidden=8, proxy_out=8,
                   sinkhorn_steps=6, materialize_top_k=1)
    first, _ = U.optimize_stage(
        _delta(), None, torch.tensor([0, 0, 0, 1, 1, 1]),
        DummyEncoder(), torch.randn(2, 8), cfg, "cpu",
        use_boundary=False, use_topology=True)
    parent = U.append_memory(None, first)
    second_delta = _delta(stage=1, offset=12)
    second_delta["boundary_edge_index"] = torch.tensor([[0, 1, 2], [0, 1, 2]])
    second, audit = U.optimize_stage(
        second_delta, parent, torch.tensor([0, 0, 0, 1, 1, 1]),
        DummyEncoder(), torch.randn(2, 8), cfg, "cpu",
        use_boundary=True, use_topology=True)
    memory = U.append_memory(parent, second)
    assert len(memory["x"]) == 12
    assert audit["uses_boundary_sketch"] is True
    assert torch.isfinite(memory["x"]).all()


def test_old_and_new_hard_graph_matches_student_after_append():
    parent = {"x": torch.randn(2, 4),
              "edge_index": torch.tensor([[0, 1, 0], [1, 0, 1]])}
    edges = torch.tensor([[2, 3], [3, 2]])
    expected = P.normalized_adjacency(
        torch.cat((parent["edge_index"], edges), 1), 4, "cpu").to_dense()
    actual = U._hard_adjacency(edges, parent, 4, "cpu")
    torch.testing.assert_close(actual, expected)
    assert actual[0, 1].item() == pytest.approx(0.5)
    assert actual[2, 3].item() == pytest.approx(0.5)


def test_sparse_reference_deduplicates_edges_and_keeps_parent():
    delta = _delta()
    parent = {"x": torch.randn(2, 4), "edge_index": torch.tensor([[0, 1], [1, 0]])}
    actual, x = U._reference_graph(delta, parent, torch.zeros(12, 2), "cpu")
    expected = P.normalized_adjacency(
        torch.cat((parent["edge_index"], delta["edge_index"] + 2), 1), 14, "cpu")
    assert actual.is_sparse
    torch.testing.assert_close(actual.to_dense(), expected.to_dense())
    torch.testing.assert_close(x[:2], parent["x"])


def test_sparse_topology_equals_dense_mathematics_and_has_gradient():
    delta = _delta()
    q = torch.rand(12, 6, requires_grad=True)
    _, actual, _ = U._soft_topology(q, delta, None, torch.zeros(12, 0))
    raw_adj = U._dense_adjacency(delta["edge_index"], 12, "cpu")
    mass = q.sum(0)
    expected = q.T @ raw_adj @ q / (mass[:, None] * mass[None, :]).sqrt()
    expected = expected * (1 - torch.eye(6))
    torch.testing.assert_close(actual, expected)
    gradient = torch.autograd.grad(actual.square().sum(), q, create_graph=True)[0]
    assert gradient.abs().sum() > 0
    assert torch.isfinite(torch.autograd.grad(gradient.square().sum(), q)[0]).all()


def test_group_class_weights_match_mass_across_compression():
    real_y = torch.tensor([0] * 4 + [1] * 2 + [2] * 20 + [3] * 10)
    syn_y = torch.tensor([0] * 4 + [1] * 2 + [2] + [3])
    real_g = torch.tensor([0] * 6 + [1] * 30)
    syn_g = torch.tensor([0] * 6 + [1] * 2)
    rw = U._group_class_weights(real_y, real_g, "group_class", real_y.device, torch.float32)
    sw = U._group_class_weights(syn_y, syn_g, "group_class", syn_y.device, torch.float32)
    for c in range(4):
        assert rw[real_y == c].sum() == pytest.approx(0.25)
        assert sw[syn_y == c].sum() == pytest.approx(0.25)
    assert rw[real_g == 1].sum() == pytest.approx(sw[syn_g == 1].sum())


def test_student_mass_matches_synthetic_group_mass_and_allows_reference_only_group():
    # Group 2/class 4 is present in the reference but has no allocated slot.
    ref_y = torch.tensor([0, 0, 1, 1, 2, 2, 4])
    ref_g = torch.tensor([0, 0, 1, 1, 1, 1, 1])
    syn_y = torch.tensor([0, 1, 2])
    syn_g = torch.tensor([0, 1, 1])
    rw, sw = U._proxy_weights(ref_y, ref_g, syn_y, syn_g,
                              "student_mass", torch.device("cpu"), torch.float64)
    assert rw.sum().item() == pytest.approx(1.0)
    assert sw.sum().item() == pytest.approx(1.0)
    for pair in {(0, 0), (1, 1), (1, 2)}:
        ref_mass = rw[((ref_g == pair[0]) & (ref_y == pair[1]))].sum()
        syn_mass = sw[((syn_g == pair[0]) & (syn_y == pair[1]))].sum()
        assert ref_mass.item() == pytest.approx(syn_mass.item())
    assert rw[((ref_g == 1) & (ref_y == 4))].sum().item() == pytest.approx(0.0)


def test_student_mass_rejects_synthetic_group_missing_from_reference():
    ref_y = torch.tensor([0, 1])
    ref_g = torch.tensor([0, 0])
    syn_y = torch.tensor([0, 2])
    syn_g = torch.tensor([0, 1])
    with pytest.raises(ValueError, match="missing"):
        U._proxy_weights(ref_y, ref_g, syn_y, syn_g,
                         "student_mass", torch.device("cpu"), torch.float32)


@pytest.mark.parametrize("weighted", [False, True])
def test_chunked_contrastive_matches_full_value_and_second_derivative(weighted):
    torch.manual_seed(72)
    graph = torch.randn(7, 5, requires_grad=True)
    text = torch.randn(7, 5, requires_grad=True)
    scale = torch.tensor(1.2, requires_grad=True)
    labels = torch.tensor([0, 0, 1, 1, 1, 2, 2])
    weights = torch.tensor([1, 2, 1, 2, 3, 2, 1.0]) if weighted else None
    full = P.multi_positive_pair_loss(graph, text, labels, scale, weights)
    chunked = P.multi_positive_pair_loss(graph, text, labels, scale, weights, chunk_size=3)
    torch.testing.assert_close(chunked, full)
    fg = torch.autograd.grad(full, (graph, text, scale), create_graph=True)
    cg = torch.autograd.grad(chunked, (graph, text, scale), create_graph=True)
    for a, b in zip(fg, cg):
        torch.testing.assert_close(a, b)
    fh = torch.autograd.grad(sum(g.square().sum() for g in fg), (graph, text, scale))
    ch = torch.autograd.grad(sum(g.square().sum() for g in cg), (graph, text, scale))
    for a, b in zip(fh, ch):
        torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-5)


def test_weighted_contrastive_is_invariant_to_duplicating_one_class():
    torch.manual_seed(3)
    graph, text = torch.randn(3, 5), torch.randn(3, 5)
    labels = torch.tensor([0, 1, 2])
    original = P.multi_positive_pair_loss(graph, text, labels, torch.tensor(1.0))
    rows = torch.tensor([0, 0, 0, 1, 2])
    weights = torch.tensor([1 / 9, 1 / 9, 1 / 9, 1 / 3, 1 / 3])
    repeated = P.multi_positive_pair_loss(graph[rows], text[rows], labels[rows],
                                          torch.tensor(1.0), weights)
    torch.testing.assert_close(original, repeated)


def test_default_student_loss_preserves_historical_formula():
    torch.manual_seed(13)
    g, t = torch.randn(6, 4), torch.randn(6, 4)
    y, scale = torch.tensor([0, 0, 1, 1, 2, 2]), torch.tensor(0.7)
    sim = scale * (g @ t.T)
    pos = y[:, None] == y[None, :]
    expected = -0.5 * ((P._masked_logsumexp(sim, pos, 1) - sim.logsumexp(1)).mean()
                       + (P._masked_logsumexp(sim, pos, 0) - sim.logsumexp(0)).mean())
    torch.testing.assert_close(P.multi_positive_pair_loss(g, t, y, scale), expected)


def test_diverse_initialization_is_deterministic_legal_and_distinct():
    torch.manual_seed(5)
    delta = _delta()
    slots = torch.tensor([0, 0, 0, 1, 1, 1])
    cfg = U.Config()
    first = U._initial_logits(delta, None, slots, cfg, "cpu")
    second = U._initial_logits(delta, None, slots, cfg, "cpu")
    torch.testing.assert_close(first[1], second[1])
    sources = first[2]["prototype_source_ids"]
    assert len(set(sources)) == len(slots)
    assert torch.equal(delta["y"][torch.tensor(sources)], slots)
    torch.testing.assert_close(first[1].sum(1), torch.ones(12), atol=1e-4, rtol=1e-4)
    assert first[1][delta["y"][:, None] != slots[None, :]].count_nonzero() == 0


def test_disabled_boundary_does_not_leak_into_initialization():
    torch.manual_seed(5)
    delta = _delta()
    delta["boundary_edge_index"] = torch.tensor([[0, 1], [100, 101]])
    parent = {"x": torch.randn(2, 4), "support_ids": [[100], [101]]}
    labels = torch.tensor([0, 0, 0, 1, 1, 1])
    _, _, audit = U._initial_logits(delta, parent, labels, U.Config(), "cpu",
                                     boundary=torch.zeros(12, 2))
    assert audit["boundary_nonzero"] == 0


@pytest.mark.parametrize("mode", ["joint", "alternating"])
def test_frozen_payload_and_topology_and_update_schedule(mode):
    torch.manual_seed(6)
    cfg = U.Config(steps=2, proxy_hidden=8, proxy_out=8, sinkhorn_steps=10,
                   text_assignment="frozen", topology_assignment="frozen",
                   optimization=mode, pair_chunk_size=3)
    delta, prompt = _delta(), torch.randn(2, 8)
    labels = torch.tensor([0, 0, 0, 1, 1, 1])
    init_delta = dict(delta)
    init_delta["text_emb"] = U._norm(torch.tensor(DummyEncoder().encode(delta["texts"])))
    _, q0, _ = U._initial_logits(init_delta, None, labels, cfg, "cpu")
    expected_edges, _ = U._materialize_edges(q0, init_delta, None, torch.zeros(12, 0), cfg)
    expected_texts, _, _, _ = U._materialize_text(init_delta, q0, labels, DummyEncoder(), cfg)
    block, audit = U.optimize_stage(delta, None, labels, DummyEncoder(), prompt, cfg,
                                    "cpu", use_boundary=False)
    assert block["texts"] == expected_texts
    assert torch.equal(block["edge_index"], expected_edges)
    assert audit["q_update_count"] == (2 if mode == "joint" else 1)
    assert audit["residual_update_count"] == (2 if mode == "joint" else 1)
    assert audit["x_residual_ratio_max"] <= cfg.trust_radius + 1e-5
    assert audit["text_hard_forward_max_error"] < 1e-5


def test_invalid_edges_and_topk_fail_explicitly():
    with pytest.raises(ValueError):
        U._dense_adjacency(torch.tensor([[0], [10]]), 3, "cpu")
    with pytest.raises(ValueError, match="top-1"):
        U.optimize_stage(_delta(), None, torch.tensor([0, 1]), DummyEncoder(),
                          torch.randn(2, 8), U.Config(materialize_top_k=3), "cpu")


def test_materialized_text_ties_match_forward_argmax():
    delta = _delta()
    labels = torch.tensor([0, 1])
    q = torch.zeros(12, 2)
    q[:6, 0] = 1
    q[6:, 1] = 1
    texts, sources, encoded, _ = U._materialize_text(delta, q, labels, DummyEncoder(), U.Config())
    chosen = q.argmax(0)
    assert texts == [delta["texts"][int(i)] for i in chosen]
    assert sources == [[0], [6]]
    torch.testing.assert_close(encoded, U._norm(torch.tensor(DummyEncoder().encode(texts))))
