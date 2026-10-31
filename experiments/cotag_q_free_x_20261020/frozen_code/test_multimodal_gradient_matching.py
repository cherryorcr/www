from __future__ import annotations

import torch
import torch.nn.functional as F

import multimodal_gradient_matching as GM


class Encoder:
    def encode(self, texts):
        values = []
        for text in texts:
            seed = sum(ord(c) for c in text) % 10007
            g = torch.Generator().manual_seed(seed)
            values.append(torch.randn(8, generator=g))
        return F.normalize(torch.stack(values), dim=1)


def make_delta():
    g = torch.Generator().manual_seed(7)
    x = torch.randn(6, 8, generator=g)
    text = F.normalize(torch.randn(6, 8, generator=g), dim=1)
    return {
        "ids": torch.arange(6),
        "stage": 0,
        "x": x,
        "y": torch.tensor([0, 0, 0, 1, 1, 1]),
        "text_emb": text,
        "texts": [f"class {int(y)} source {i}" for i, y in enumerate([0, 0, 0, 1, 1, 1])],
        "edge_index": torch.tensor([[0, 1, 1, 2, 3, 4],
                                     [1, 0, 2, 1, 4, 3]]),
    }


def test_joint_gradient_matching_materializes_row_specific_text():
    delta = make_delta()
    labels = torch.tensor([0, 0, 1, 1])
    prompt = F.normalize(torch.randn(2, 8, generator=torch.Generator().manual_seed(9)), dim=1)
    cfg = GM.Config(proxy_hidden=16, proxy_out=8, proxy_seeds=(3,), steps=2,
                    candidate_cap=0, materialize_top_k=1)
    block, audit, old = GM.optimize_stage(
        delta, None, delta["x"][:4], labels, Encoder(), prompt, cfg, "cpu")
    assert old is None
    assert block["x"].shape == (4, 8)
    assert block["text_emb"].shape == (4, 8)
    assert len(block["texts"]) == 4
    assert len(block["labels"]) == 4
    assert audit["x_updated_after_cat"]
    assert audit["text_routing_updated"]
    assert audit["historical_raw_reads"] == 0
    assert torch.isfinite(block["x"]).all()


def test_bounded_old_revision_keeps_non_x_parent_fields():
    delta = make_delta()
    parent = {
        "x": delta["x"][:2].clone(),
        "text_emb": delta["text_emb"][:2].clone(),
        "labels": torch.tensor([0, 0]),
        "texts": ["old 0", "old 1"],
        "mass": torch.ones(2),
        "depth": torch.zeros(2, dtype=torch.long),
        "stage": torch.zeros(2, dtype=torch.long),
        "edge_index": torch.empty((2, 0), dtype=torch.long),
    }
    labels = torch.tensor([1, 1])
    prompt = F.normalize(torch.randn(2, 8, generator=torch.Generator().manual_seed(10)), dim=1)
    cfg = GM.Config(proxy_hidden=16, proxy_out=8, proxy_seeds=(4,), steps=1,
                    allow_old_update=True, old_trust_radius=0.02)
    block, audit, old = GM.optimize_stage(
        delta, parent, delta["x"][2:4], labels, Encoder(), prompt, cfg, "cpu")
    memory = GM.append_memory(parent, block, old)
    assert audit["old_memory_updated"]
    assert torch.equal(memory["labels"][2:], labels)
    assert memory["texts"][:2] == parent["texts"]
    shift = memory["x"][:2] - parent["x"]
    limit = cfg.old_trust_radius * parent["x"].norm(dim=1).clamp_min(1e-3) + 1e-5
    assert bool((shift.norm(dim=1) <= limit).all())


def test_decoupled_gradient_blocks_are_differentiable():
    delta = make_delta()
    prompt = F.normalize(torch.randn(2, 8, generator=torch.Generator().manual_seed(12)), dim=1)
    cfg = GM.Config(proxy_hidden=16, proxy_out=8, proxy_seeds=(5,), steps=1,
                    gradient_matching_mode="decoupled")
    block, audit, _ = GM.optimize_stage(
        delta, None, delta["x"][:4], torch.tensor([0, 0, 1, 1]),
        Encoder(), prompt, cfg, "cpu")
    assert block["x"].shape[0] == 4
    assert audit["gradient_matching_mode"] == "decoupled"


def test_fixed_slot_topology_is_balanced_and_append_preserves_edges():
    delta = make_delta()
    labels = torch.tensor([0, 0, 1, 1])
    x0 = delta["x"][:4].clone()
    cfg = GM.Config(topology_mode="fixed_q", topology_sinkhorn_steps=20)
    edges, audit, q = GM.build_fixed_topology(
        delta, x0, labels, delta["text_emb"], delta["text_emb"][:4],
        cfg, "cpu")
    assert edges.shape[0] == 2
    assert edges.numel() == 0 or int(edges.min()) >= 0
    assert edges.numel() == 0 or int(edges.max()) < len(x0)
    assert audit["q_row_sum_max_error"] < 1e-4
    assert audit["raw_undirected_edges"] == 3
    assert audit["target_undirected_edges"] == 2
    assert q.shape == (6, 4)
    block = {
        "x": x0,
        "text_emb": delta["text_emb"][:4],
        "labels": labels,
        "texts": ["a", "b", "c", "d"],
        "mass": torch.ones(4),
        "depth": torch.zeros(4, dtype=torch.long),
        "stage": torch.ones(4, dtype=torch.long),
        "edge_index": edges,
    }
    parent = dict(block)
    parent["x"] = parent["x"][:2]
    parent["text_emb"] = parent["text_emb"][:2]
    parent["labels"] = parent["labels"][:2]
    parent["texts"] = parent["texts"][:2]
    parent["mass"] = parent["mass"][:2]
    parent["depth"] = parent["depth"][:2]
    parent["stage"] = parent["stage"][:2]
    parent["edge_index"] = torch.empty((2, 0), dtype=torch.long)
    memory = GM.append_memory(parent, block, None)
    assert torch.equal(memory["edge_index"][:, :0], parent["edge_index"])
    assert len(memory["x"]) == 6
    if edges.numel():
        assert bool((memory["edge_index"][:, -edges.shape[1]:] >= 2).all())


def test_em_like_topology_reassigns_without_breaking_gradient_path():
    delta = make_delta()
    prompt = F.normalize(torch.randn(2, 8,
                                     generator=torch.Generator().manual_seed(13)),
                         dim=1)
    cfg = GM.Config(proxy_hidden=16, proxy_out=8, proxy_seeds=(6,), steps=2,
                    gradient_matching_mode="decoupled", topology_mode="em_q",
                    topology_refine_steps=2, em_updates=1, em_interval=1)
    block, audit, _ = GM.optimize_stage(
        delta, None, delta["x"][:4], torch.tensor([0, 0, 1, 1]),
        Encoder(), prompt, cfg, "cpu")
    assert block["edge_index"].shape[0] == 2
    assert audit["topology_updated"]
    assert len(audit["em_trace"]) == 1
