"""Toy contracts for the first GC-SR gradient implementation."""
import torch
import torch.nn.functional as F

import gc_sr_gradient as GC


def memory(n=10, d=8, seed=7, stage=0):
    gen = torch.Generator().manual_seed(seed)
    return {
        "x": torch.randn(n, d, generator=gen),
        "text_emb": F.normalize(torch.randn(n, 384, generator=gen), dim=1),
        "texts": [f"text-{i}" for i in range(n)],
        "mass": torch.ones(n),
        "depth": torch.zeros(n, dtype=torch.long),
        "stage": torch.full((n,), stage, dtype=torch.long),
        "edge_index": torch.stack((torch.arange(n - 1), torch.arange(1, n))),
    }


def test_diagnostic_is_finite_and_does_not_mutate_inputs():
    base, support = memory(), memory(14, seed=8, stage=1)
    base_hash, support_hash = GC.memory_hash(base), GC.memory_hash(support)
    out = GC.diagnostic(base, support, "cpu", GC.Config(batch=0))
    assert len(out["rows"]) == 2
    assert torch.isfinite(torch.tensor(out["mean_gradient_cosine"]))
    assert GC.memory_hash(base) == base_hash
    assert GC.memory_hash(support) == support_hash


def test_gradient_revision_changes_only_x_and_respects_old_anchor():
    base, support = memory(), memory(14, seed=8, stage=1)
    cfg = GC.Config(outer_steps=2, eval_every=1, batch=0,
                    old_anchor_weight=0.1, old_radius=0.03, new_radius=0.05)
    out, audit = GC.optimize(base, support, [1, 2, 8, 9], 8, 1, "cpu", cfg)
    assert GC.memory_hash(base) != GC.memory_hash(out)
    assert torch.equal(base["edge_index"], out["edge_index"])
    assert base["texts"] == out["texts"]
    assert audit["nonzero_hypergradient_steps"] == cfg.outer_steps
    assert audit["max_relative_displacement"] <= cfg.new_radius + 1e-5
    changed = torch.where((base["x"] != out["x"]).any(1))[0].tolist()
    assert set(changed).issubset({1, 2, 8, 9})


def test_output_only_arm_is_registered_and_can_backpropagate():
    base, support = memory(), memory(13, seed=9, stage=1)
    cfg = GC.Config(outer_steps=2, eval_every=1, batch=4,
                    gradient_weight=0.0, output_weight=1.0,
                    old_anchor_weight=0.0)
    out, audit = GC.optimize(base, support, [8, 9], 8, 1, "cpu", cfg)
    assert torch.isfinite(out["x"]).all()
    assert audit["nonzero_hypergradient_steps"] == cfg.outer_steps


def test_group_gradient_arm_has_new_group_observation():
    base, support = memory(), memory(13, seed=9, stage=1)
    cfg = GC.Config(outer_steps=1, eval_every=1, batch=4,
                    gradient_weight=0.0, output_weight=0.0,
                    new_gradient_weight=1.0, old_gradient_weight=0.25,
                    old_anchor_weight=0.0)
    _, audit = GC.optimize(base, support, [8, 9], 8, 1, "cpu", cfg)
    groups = audit["curve"][0]["train"]["seeds"][0]["groups"]
    assert "new" in groups and torch.isfinite(torch.tensor(groups["new"]["gradient_cosine"]))


def test_sensitivity_selection_is_partitioned_and_deterministic():
    base = memory(12)
    first, audit = GC.select_editable_rows(base, 8, 2, 2, "cpu", seed=9120)
    second, _ = GC.select_editable_rows(base, 8, 2, 2, "cpu", seed=9120)
    assert first == second == audit["selected_rows"]
    assert len(first) == 4
    assert sum(row < 8 for row in first) == 2
    assert sum(row >= 8 for row in first) == 2
