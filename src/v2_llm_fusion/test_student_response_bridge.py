import torch
import torch.nn.functional as F

import student_response_bridge as RB


def _delta() -> dict:
    generator = torch.Generator().manual_seed(101)
    return {
        "x": torch.randn(10, 6, generator=generator),
        "text_emb": F.normalize(torch.randn(10, 5, generator=generator), dim=1),
        "y": torch.tensor([0] * 5 + [1] * 5),
        "texts": ["source"] * 10,
        "ids": torch.arange(10),
        "stage": 1,
    }


def _parent() -> dict:
    generator = torch.Generator().manual_seed(102)
    return {
        "x": torch.randn(4, 6, generator=generator),
        "text_emb": F.normalize(torch.randn(4, 5, generator=generator), dim=1),
        "texts": ["old"] * 4,
        "mass": torch.ones(4),
        "depth": torch.zeros(4, dtype=torch.long),
        "stage": torch.zeros(4, dtype=torch.long),
        "edge_index": torch.empty((2, 0), dtype=torch.long),
    }


def test_response_bridge_updates_only_new_prefix_is_immutable():
    delta = _delta()
    parent = _parent()
    x0 = delta["x"][:3].clone()
    t0 = delta["text_emb"][:3].clone()
    labels = torch.tensor([0, 0, 1])
    cfg = RB.ResponseConfig(
        inner_steps=2, outer_steps=2, inner_batch=4,
        response_seeds=(9300,), mmd_hidden=8, mmd_probes=2,
    )
    result, audit = RB.optimize_response_block(
        delta, delta["x"], x0, t0, ["new"] * 3, [[0], [1], [2]],
        labels, parent, cfg, "response_xt", torch.ones(3), "cpu")
    assert torch.equal(parent["x"], _parent()["x"])
    assert result["x"].shape == x0.shape
    assert result["text_emb"].shape == t0.shape
    assert audit["old_prefix_updated"] is False
    assert audit["topology_updated"] is False
    assert len(audit["trace"]) == 2
    assert all(torch.isfinite(value).all() for value in result.values()
               if isinstance(value, torch.Tensor))


def test_response_t_only_keeps_x_exactly():
    delta = _delta()
    x0 = delta["x"][:3].clone()
    t0 = delta["text_emb"][:3].clone()
    labels = torch.tensor([0, 0, 1])
    cfg = RB.ResponseConfig(
        inner_steps=2, outer_steps=1, inner_batch=4,
        response_seeds=(9301,), mmd_hidden=8, mmd_probes=2,
    )
    result, _ = RB.optimize_response_block(
        delta, delta["x"], x0, t0, ["new"] * 3, [[0], [1], [2]],
        labels, None, cfg, "response_t_only", torch.ones(3), "cpu")
    assert torch.equal(result["x"], x0)
    assert torch.isfinite(result["text_emb"]).all()


def test_support_response_uses_current_delta_only():
    delta = _delta()
    x0 = delta["x"][:3].clone()
    t0 = delta["text_emb"][:3].clone()
    labels = torch.tensor([0, 0, 1])
    cfg = RB.ResponseConfig(
        inner_steps=2, outer_steps=1, inner_batch=4,
        response_seeds=(9302,), mmd_hidden=8, mmd_probes=2,
        new_weight=0.0, prompt_weight=0.0,
    )
    prompt = F.normalize(torch.randn(2, 5, generator=torch.Generator().manual_seed(103)), dim=1)
    result, audit = RB.optimize_response_block(
        delta, delta["x"], x0, t0, ["new"] * 3, [[0], [1], [2]],
        labels, None, cfg, "support_xt", torch.ones(3), "cpu",
        prompt_emb=prompt, prompt_ids=torch.tensor([0, 1]))
    assert result["x"].shape == x0.shape
    assert audit["support_response_objective"]
    assert len(audit["final_response"]) == 1


def test_new_margin_response_has_explicit_new_class_signal():
    delta = _delta()
    x0 = delta["x"][:3].clone()
    t0 = delta["text_emb"][:3].clone()
    labels = torch.tensor([0, 0, 1])
    cfg = RB.ResponseConfig(
        inner_steps=2, outer_steps=1, inner_batch=4,
        response_seeds=(9303,), mmd_hidden=8, mmd_probes=2,
        old_weight=0.0, new_weight=0.0, prompt_weight=0.0,
        support_weight=0.0, support_prompt_weight=0.0,
        new_margin_weight=1.0, new_margin=0.2,
    )
    prompt = F.normalize(
        torch.randn(2, 5, generator=torch.Generator().manual_seed(104)),
        dim=1)
    result, audit = RB.optimize_response_block(
        delta, delta["x"], x0, t0, ["new"] * 3, [[0], [1], [2]],
        labels, None, cfg, "new_margin_x_only", torch.ones(3), "cpu",
        prompt_emb=prompt, prompt_ids=torch.tensor([0, 1]))
    assert result["x"].shape == x0.shape
    assert torch.isfinite(result["x"]).all()
    assert audit["mode"] == "new_margin_x_only"
    assert audit["trace"][0]["new_margin_response"] >= 0.0

