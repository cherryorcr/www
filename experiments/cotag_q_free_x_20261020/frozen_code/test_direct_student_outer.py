"""Unit checks for the direct Student outer construction."""
from __future__ import annotations

import torch

import direct_student_outer as DSO
from prepare_provenance_x_support import build_provenance_map


def _memory(n: int, dim: int = 8) -> dict:
    generator = torch.Generator().manual_seed(100 + n)
    x = torch.randn(n, dim, generator=generator)
    text = torch.randn(n, 384, generator=generator)
    pairs = []
    for index in range(n - 1):
        pairs.extend(((index, index + 1), (index + 1, index)))
    return {
        "x": x,
        "text_emb": text,
        "texts": [str(index) for index in range(n)],
        "edge_index": torch.tensor(pairs, dtype=torch.long).T.contiguous(),
        "mass": torch.ones(n),
        "depth": torch.zeros(n, dtype=torch.long),
        "stage": torch.zeros(n, dtype=torch.long),
    }


def _config() -> DSO.Config:
    return DSO.Config(
        inner_steps=1, inner_batch=8, outer_steps=1, support_batch=8,
        train_seeds=(1,), validation_seeds=(2,), plan_seed=3, a_pool=3,
    )


def test_x_outer_has_second_order_path_and_preserves_fields():
    base, support = _memory(8), _memory(12)
    output, audit = DSO.optimize(
        base, support, 4, [0, 1, 6, 7], "x", "cpu", _config())
    assert audit["outer_gradient_variables"] == ["x_relative"]
    assert any(
        row["gradient_norms"]["x_relative"] > 0
        for row in audit["curve"]
    )
    assert torch.equal(base["text_emb"], output["text_emb"])
    assert torch.equal(base["edge_index"], output["edge_index"])
    changed = torch.where((base["x"] != output["x"]).any(1))[0].tolist()
    assert set(changed).issubset({0, 1, 6, 7})
    assert audit["historical_raw_reads"] == 0
    assert audit["report_labels_used"] is False


def test_p_and_a_are_materialized_under_the_same_outer_objective():
    base, support = _memory(8), _memory(12)
    cfg = _config()
    p_output, p_audit = DSO.optimize(
        base, support, 4, [], "p", "cpu", cfg)
    assert p_audit["outer_gradient_variables"] == ["p_logits"]
    assert "p_assignment_entropy" in p_audit
    assert torch.equal(base["edge_index"], p_output["edge_index"])
    a_output, a_audit = DSO.optimize(
        base, support, 4, [], "a", "cpu", cfg)
    assert a_audit["outer_gradient_variables"] == ["a_logits"]
    assert a_audit["materialization"]["hard_cross_edges"] > 0
    assert len(a_output["edge_index"]) == 2


def test_support_train_and_validation_are_disjoint():
    plan = DSO.make_support_plan(_memory(12), old_rows=4, batch=8, seed=17)
    for name, expected in (("old", 4), ("new", 8)):
        train = {row for batch in plan["splits"]["train"][name]
                 for row in batch}
        validation = {row for batch in plan["splits"]["validation"][name]
                      for row in batch}
        assert train.isdisjoint(validation)
        assert len(train | validation) == expected


def test_new_loss_scale_changes_only_new_support_weight():
    base, support = _memory(8), _memory(12)
    support_class_id = torch.tensor(
        [-1, -1, -1, -1, 0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
    plan = DSO.make_support_plan(
        support, old_rows=4, batch=8, seed=17,
        support_class_id=support_class_id, stratify_new=True)
    model = DSO._student(11, 8, 384, "cpu")
    params = {name: value for name, value in model.named_parameters()}
    adjacency = DSO.fixed_norm_adj(support["edge_index"], 12, "cpu")
    baseline_cfg = _config()
    scaled_cfg = DSO.replace(baseline_cfg, new_loss_scale=2.5)
    baseline = DSO.support_losses(
        params, support, adjacency, plan, "validation", baseline_cfg)
    scaled = DSO.support_losses(
        params, support, adjacency, plan, "validation", scaled_cfg)
    assert torch.allclose(scaled["old"], baseline["old"])
    assert torch.allclose(scaled["new"], baseline["new"])
    assert torch.allclose(
        scaled["total"] - baseline["total"],
        (scaled_cfg.new_loss_scale - 1.0) * baseline["new"])


def test_old_graph_anchor_fraction_selects_highest_drift_rows():
    base = _memory(8)
    candidate = DSO.clone_memory(base)
    candidate["x"][0] = candidate["x"][0] + 0.8
    candidate["x"][1] = candidate["x"][1] + 0.3
    params = {name: value for name, value in
              DSO._student(21, 8, 384, "cpu").named_parameters()}
    adjacency = DSO.fixed_norm_adj(base["edge_index"], 8, "cpu")
    full = DSO.old_graph_anchor_loss(
        params, candidate, adjacency, base, old_rows=8, fraction=1.0)
    focused = DSO.old_graph_anchor_loss(
        params, candidate, adjacency, base, old_rows=8, fraction=0.25)
    assert torch.isfinite(full) and torch.isfinite(focused)
    assert float(focused.detach()) >= 0.0
    zero = DSO.old_graph_anchor_loss(
        params, candidate, adjacency, base, old_rows=8, fraction=0.0)
    assert float(zero.detach()) == 0.0


def test_a_preserves_zero_cross_degree():
    base = _memory(8)
    # Keep only within-block edges, so no old/new candidate pair exists.
    base["edge_index"] = torch.tensor(
        [(0, 1), (1, 0), (2, 3), (3, 2), (4, 5), (5, 4), (6, 7), (7, 6)],
        dtype=torch.long).T.contiguous()
    support = _memory(12)
    _, audit = DSO.optimize(base, support, 4, [], "a", "cpu", _config())
    assert audit["materialization"]["hard_cross_edges"] == 0


def test_a_soft_mass_matches_materialized_degree():
    base = _memory(8)
    support = _memory(12)
    cfg = _config()
    logits, pools, degrees = DSO._init_a_logits(base, 4, cfg, "cpu")
    weights, _ = DSO._a_weights(
        logits, pools, degrees, cfg.a_temperature, straight_through=False)
    cursor = 0
    for pool, degree in zip(pools, degrees):
        width = len(pool)
        assert torch.isclose(
            weights[cursor:cursor + width].sum(),
            torch.tensor(float(degree)))
        cursor += width
    hard_weights, _ = DSO._a_weights(
        logits, pools, degrees, cfg.a_temperature, straight_through=True)
    cursor = 0
    for pool, degree in zip(pools, degrees):
        width = len(pool)
        assert torch.isclose(
            hard_weights[cursor:cursor + width].sum(),
            torch.tensor(float(degree)))
        cursor += width


def test_a_revision_cap_zero_keeps_original_cross_edges():
    base, support = _memory(8), _memory(12)
    cfg = DSO.Config(
        inner_steps=1, inner_batch=8, outer_steps=1, support_batch=8,
        a_pool=4, a_max_revisions_per_row=0,
        train_seeds=(1,), validation_seeds=(2,), plan_seed=3)
    _, audit = DSO.optimize(base, support, 4, [], "a", "cpu", cfg)
    material = audit["materialization"]
    assert material["hard_revised_cross_edges"] == 0
    assert material["max_hard_revisions_per_row"] == 0
    assert all(value == 0 for value in material["hard_revisions_per_row"])


def test_a_revision_cap_zero_is_a_valid_frozen_topology_control():
    base, support = _memory(8), _memory(12)
    cfg = DSO.Config(
        inner_steps=1, inner_batch=8, outer_steps=1, support_batch=8,
        a_pool=4, a_max_revisions_per_row=0,
        train_seeds=(1,), validation_seeds=(2,), plan_seed=3)
    output, audit = DSO.optimize(base, support, 4, [], "a", "cpu", cfg)
    assert audit["curve"][0]["gradient_norms"]["a_logits"] == 0.0
    assert torch.equal(output["edge_index"], base["edge_index"])


def test_a_revision_cap_one_is_enforced_per_new_row():
    base, support = _memory(8), _memory(12)
    cfg = DSO.Config(
        inner_steps=1, inner_batch=8, outer_steps=1, support_batch=8,
        a_pool=4, a_max_revisions_per_row=1,
        train_seeds=(1,), validation_seeds=(2,), plan_seed=3)
    _, audit = DSO.optimize(base, support, 4, [], "a", "cpu", cfg)
    material = audit["materialization"]
    assert material["max_hard_revisions_per_row"] <= 1
    assert all(value <= 1 for value in material["hard_revisions_per_row"])


def test_marginal_a_uses_paired_student_probe_and_preserves_edge_budget():
    base, support = _memory(8), _memory(12)
    cfg = DSO.Config(
        inner_steps=1, inner_batch=8, outer_steps=1, support_batch=8,
        a_pool=3, a_max_revisions_per_row=1, a_marginal=True,
        a_marginal_inner_steps=1, train_seeds=(1,), validation_seeds=(2,),
        plan_seed=3)
    output, audit = DSO.optimize(
        base, support, 4, [4, 5, 6, 7], "x+p+a", "cpu", cfg)
    material = audit["materialization"]
    assert material["a_marginal"] is True
    assert material["a_marginal_candidate_count"] >= 0
    assert material["a_marginal_edge_count"] == len(
        DSO._undirected_pairs(base["edge_index"], len(base["x"])))
    assert len(output["edge_index"]) % 2 == 0
    assert "a_logits" not in audit["outer_gradient_variables"]


def test_a_revision_margin_requires_a_new_edge_advantage():
    logits = torch.tensor([[1.0, 1.02, 0.0]], requires_grad=True)
    pools = [[0, 1, 2]]
    degrees = [1]
    masks = [[True, False, False]]
    _, conservative = DSO._a_weights(
        logits, pools, degrees, temperature=1.0, straight_through=True,
        existing_masks=masks, max_revisions_per_row=1,
        revision_margin=0.10)
    _, permissive = DSO._a_weights(
        logits, pools, degrees, temperature=1.0, straight_through=True,
        existing_masks=masks, max_revisions_per_row=1,
        revision_margin=0.01)
    assert conservative == [[0]]
    assert permissive == [[1]]


def test_class_constrained_p_has_no_cross_class_transport_and_keeps_gradients():
    logits = torch.randn(
        4, 6, generator=torch.Generator().manual_seed(792), requires_grad=True)
    target_classes = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    source_classes = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
    assignment = DSO.class_constrained_assignment(
        logits, target_classes, source_classes, iterations=8)
    assert torch.allclose(assignment.sum(1), torch.ones(4), atol=1e-5)
    assert torch.equal(assignment[:2, 3:], torch.zeros(2, 3))
    assert torch.equal(assignment[2:, :3], torch.zeros(2, 3))
    assignment.square().sum().backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_bounded_old_context_transport_preserves_new_mass_and_is_capped():
    base, support = _memory(8), _memory(12)
    cfg = DSO.Config(p_old_context_ratio=0.25, p_old_context_cap=0.25,
                     p_temperature=0.2, p_sinkhorn_iters=8)
    logits = DSO._init_p_logits(base, support, 4, cfg, "cpu")
    variables = {"p_logits": logits.clone().requires_grad_(True)}
    candidate, audit = DSO._candidate(
        base, support, 4, variables, "p", torch.arange(4, 8), cfg, "cpu")
    assert torch.isfinite(candidate["x"]).all()
    assert audit["p_old_context_ratio"] == 0.25
    assert torch.allclose(audit["p_assignment"].sum(1), torch.ones(4), atol=1e-5)


def test_response_gate_uses_per_new_row_gain():
    base, support = _memory(8), _memory(12)
    cfg = DSO.Config(p_old_context_ratio=0.25, p_old_context_cap=0.25,
                     p_response_gate=True, p_gate_margin=0.0,
                     p_temperature=0.2, p_sinkhorn_iters=8)
    logits = DSO._init_p_logits(base, support, 4, cfg, "cpu")
    variables = {"p_logits": logits.clone().requires_grad_(True)}
    candidate, audit = DSO._candidate(
        base, support, 4, variables, "p", torch.arange(4, 8), cfg, "cpu")
    assert torch.isfinite(candidate["x"]).all()
    assert 0.0 <= audit["p_response_gate"] <= 1.0


def test_response_gate_pairs_mixed_and_pure_new_and_reuses_validation_decision():
    base, support = _memory(8), _memory(12)
    cfg = DSO.Config(
        inner_steps=1, inner_batch=8, outer_steps=1, support_batch=8,
        p_old_context_ratio=0.25, p_old_context_cap=0.25,
        p_response_gate=True, p_gate_margin=0.0,
        p_gate_old_tolerance=0.0, p_temperature=0.2, p_sinkhorn_iters=8,
        train_seeds=(1,), validation_seeds=(2,), plan_seed=3)
    _, audit = DSO.optimize(
        base, support, 4, [], "p", "cpu", cfg)
    validation_row = audit["curve"][0]["validation"]["seeds"][0]
    for key in (
            "p_response_new_delta", "p_response_old_delta",
            "p_response_mixed_new", "p_response_pure_new"):
        assert key in validation_row
    summary = audit["p_response_gate_summary"]
    assert summary["source"] == "best_validation_checkpoint"
    assert summary["seed_count"] == 1
    assert audit["materialization"]["p_response_gate"] == summary[
        "keep_old_context"]
    assert audit["materialization"]["p_old_context_ratio"] in (0.0, 0.25)


def test_response_gate_can_keep_old_context_per_new_class():
    base, support = _memory(8), _memory(12)
    support_class_id = torch.tensor(
        [-1, -1, -1, -1, 0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
    base_slot_class_id = torch.tensor(
        [-1, -1, -1, -1, 0, 0, 1, 1], dtype=torch.long)
    cfg = DSO.Config(
        inner_steps=1, inner_batch=8, outer_steps=1, support_batch=8,
        p_old_context_ratio=0.25, p_old_context_cap=0.25,
        p_response_gate=True, p_gate_per_class=True,
        train_seeds=(1,), validation_seeds=(2,), plan_seed=3)
    _, audit = DSO.optimize(
        base, support, 4, [4, 5, 6, 7], "x-class+p+a", "cpu", cfg,
        support_class_id=support_class_id,
        base_slot_class_id=base_slot_class_id)
    validation = audit["curve"][0]["validation"]["seeds"][0]
    assert validation["p_response_gate_per_class"] is True
    assert len(validation["p_response_gate_rows"]) == 4
    summary = audit["p_response_gate_summary"]
    assert summary["per_class"] is True
    assert len(summary["keep_old_context_rows"]) == 4
    assert audit["materialization"]["p_response_gate_rows"] == [
        float(value) for value in summary["keep_old_context_rows"]]


def test_message_aware_a_prior_is_bounded_to_candidate_pool():
    base = _memory(8)
    cfg = DSO.Config(a_pool=3, a_message_weight=0.5)
    logits, pools, _ = DSO._init_a_logits(base, 4, cfg, "cpu")
    assert logits.shape[0] == len(pools) == 4
    assert all(len(pool) <= 3 for pool in pools)
    assert torch.isfinite(logits).all()


def test_x_class_p_a_uses_one_joint_outer_objective():
    base, support = _memory(8), _memory(12)
    support_class_id = torch.tensor(
        [-1, -1, -1, -1, 0, 0, 0, 1, 1, 1, 0, 1], dtype=torch.long)
    base_slot_class_id = torch.tensor(
        [-1, -1, -1, -1, 0, 1, 0, 1], dtype=torch.long)
    prompts = torch.randn(
        2, 384, generator=torch.Generator().manual_seed(791))
    cfg = DSO.Config(
        inner_steps=1, inner_batch=8, outer_steps=1, support_batch=8,
        outer_lr=0.01, p_mix=0.25, a_pool=3,
        p_class_constrained=True,
        candidate_prompt_margin_weight=1.0,
        candidate_prompt_margin=0.20,
        old_graph_anchor_weight=0.5,
        train_seeds=(1,), validation_seeds=(2,), plan_seed=3)
    output, audit = DSO.optimize(
        base, support, 4, [4, 5, 6, 7], "x-class+p+a", "cpu", cfg,
        support_class_id=support_class_id,
        prompt_emb=prompts, prompt_ids=torch.tensor([0, 1]),
        base_slot_class_id=base_slot_class_id)
    assert audit["outer_gradient_variables"] == [
        "x_class_relative", "p_logits", "a_logits"]
    assert audit["candidate_prompt_margin_aware"] is True
    row = audit["curve"][0]["validation"]["seeds"][0]
    assert row["candidate_prompt_selected_rows"] == 4
    assert row["candidate_prompt_class_count"] == 2
    assert row["class_representation_classes"] >= 1
    assert torch.isfinite(torch.tensor(row["class_representation_loss"]))
    assert torch.isfinite(torch.tensor(row["old_graph_anchor_loss"]))
    assert torch.equal(base["x"][:4], output["x"][:4])
    assert len(output["x"]) == len(base["x"])
    assert "p_assignment_entropy" in audit
    assert "hard_cross_edges" in audit["materialization"]
    assert any(
        audit["curve"][0]["gradient_norms"][name] > 0
        for name in ("x_class_relative", "p_logits", "a_logits"))


def test_prompt_loss_is_optional_and_uses_new_class_metadata():
    base, support = _memory(8), _memory(12)
    class_id = torch.tensor(
        [-1, -1, -1, -1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.long)
    prompt_emb = torch.randn(2, 384, generator=torch.Generator().manual_seed(77))
    cfg = DSO.Config(
        inner_steps=1, inner_batch=8, outer_steps=1, support_batch=8,
        prompt_weight=0.5, train_seeds=(1,), validation_seeds=(2,),
        plan_seed=3, a_pool=3,
    )
    _, audit = DSO.optimize(
        base, support, 4, [0, 1, 6, 7], "x", "cpu", cfg,
        support_class_id=class_id, prompt_emb=prompt_emb,
        prompt_ids=torch.tensor([0, 1]))
    assert audit["config"]["prompt_weight"] == 0.5
    assert all(
        "prompt_loss" in row["train"]["seeds"][0]
        and "prompt_loss" in row["validation"]["seeds"][0]
        for row in audit["curve"]
    )


def test_prompt_margin_is_recorded_separately():
    base, support = _memory(8), _memory(12)
    class_id = torch.tensor(
        [-1, -1, -1, -1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.long)
    prompt_emb = torch.randn(2, 384, generator=torch.Generator().manual_seed(78))
    cfg = DSO.Config(
        inner_steps=1, inner_batch=8, outer_steps=1, support_batch=8,
        prompt_margin_weight=0.5, prompt_margin=0.2,
        train_seeds=(1,), validation_seeds=(2,), plan_seed=3, a_pool=3,
    )
    _, audit = DSO.optimize(
        base, support, 4, [0, 1, 6, 7], "x", "cpu", cfg,
        support_class_id=class_id, prompt_emb=prompt_emb,
        prompt_ids=torch.tensor([0, 1]))
    assert audit["config"]["prompt_margin_weight"] == 0.5
    assert all(
        "prompt_margin_loss" in row["validation"]["seeds"][0]
        for row in audit["curve"]
    )


def test_geometry_x_updates_only_new_rows_and_records_geometry_terms():
    base, support = _memory(8), _memory(12)
    base_slot_class_id = torch.tensor(
        [-1, -1, -1, -1, 0, 0, 1, 1], dtype=torch.long)
    prompt_emb = torch.randn(
        2, 384, generator=torch.Generator().manual_seed(790))
    cfg = DSO.Config(
        inner_steps=1, inner_batch=8, outer_steps=1, support_batch=8,
        geometry_anchor_weight=0.5, geometry_margin_weight=1.0,
        geometry_smoothness_weight=0.25,
        train_seeds=(1,), validation_seeds=(2,), plan_seed=3, a_pool=3,
    )
    output, audit = DSO.optimize(
        base, support, 4, [4, 5, 6, 7], "x-geometry", "cpu", cfg,
        base_slot_class_id=base_slot_class_id, prompt_emb=prompt_emb,
        prompt_ids=torch.tensor([0, 1]))
    assert audit["outer_gradient_variables"] == ["x_geometry_relative"]
    assert any(
        row["gradient_norms"]["x_geometry_relative"] > 0
        for row in audit["curve"]
    )
    assert all(
        "geometry_margin_loss" in row["validation"]["seeds"][0]
        for row in audit["curve"]
    )
    assert torch.equal(base["x"][:4], output["x"][:4])
    assert torch.equal(base["text_emb"], output["text_emb"])
    assert torch.equal(base["edge_index"], output["edge_index"])
    assert any((base["x"][4:] != output["x"][4:]).any(1))
    assert audit["config"]["geometry_margin_weight"] == 1.0


def test_class_representation_x_uses_current_delta_targets_and_keeps_payload_fixed():
    base, support = _memory(8), _memory(12)
    support_class_id = torch.tensor(
        [-1, -1, -1, -1, 0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
    base_slot_class_id = torch.tensor(
        [-1, -1, -1, -1, 0, 0, 1, 1], dtype=torch.long)
    cfg = DSO.Config(
        inner_steps=1, inner_batch=8, outer_steps=1, support_batch=8,
        geometry_anchor_weight=0.5, geometry_smoothness_weight=0.25,
        class_representation_weight=1.0,
        train_seeds=(1,), validation_seeds=(2,), plan_seed=3, a_pool=3,
    )
    output, audit = DSO.optimize(
        base, support, 4, [4, 5, 6, 7], "x-class-rep", "cpu", cfg,
        support_class_id=support_class_id,
        base_slot_class_id=base_slot_class_id)
    assert audit["class_representation_aware"] is True
    assert audit["outer_gradient_variables"] == ["x_class_rep_relative"]
    assert any(
        row["gradient_norms"]["x_class_rep_relative"] > 0
        for row in audit["curve"]
    )
    assert "class_representation_loss" in audit["curve"][0]["validation"]["seeds"][0]
    assert torch.equal(base["x"][:4], output["x"][:4])
    assert torch.equal(base["text_emb"], output["text_emb"])
    assert torch.equal(base["edge_index"], output["edge_index"])
    assert any((base["x"][4:] != output["x"][4:]).any(1))


def test_class_aware_x_routes_new_class_gradients_and_freezes_old_rows():
    base = _memory(8)
    support = _memory(20)
    support_class_id = torch.tensor(
        [-1, -1, -1, -1] + [0] * 8 + [1] * 8, dtype=torch.long)
    base_slot_class_id = torch.tensor(
        [-1, -1, -1, -1, 0, 0, 1, 1], dtype=torch.long)
    cfg = DSO.Config(
        inner_steps=1, inner_batch=8, outer_steps=1, support_batch=8,
        train_seeds=(1,), validation_seeds=(2,), plan_seed=3, a_pool=3,
    )
    output, audit = DSO.optimize(
        base, support, 4, [4, 5, 6, 7], "x-class", "cpu", cfg,
        support_class_id=support_class_id,
        base_slot_class_id=base_slot_class_id)
    assert audit["class_aware"] is True
    assert audit["outer_gradient_variables"] == ["x_class_relative"]
    assert set(audit["curve"][0]["train"]["classwise"]) == {"0", "1"}
    assert torch.equal(base["x"][:4], output["x"][:4])
    assert torch.equal(base["text_emb"], output["text_emb"])
    assert torch.equal(base["edge_index"], output["edge_index"])
    assert any((base["x"][4:] != output["x"][4:]).any(1))


def test_provenance_map_conserves_mass_and_keeps_old_metadata_fixed():
    base, support = _memory(8), _memory(36)
    old_rows = 4
    base["mass"][old_rows:] = 8
    support_class_id = torch.tensor(
        [-1] * old_rows + [0] * 16 + [1] * 16, dtype=torch.long)
    base_slot_class_id = torch.tensor(
        [-1] * old_rows + [0, 0, 1, 1], dtype=torch.long)
    mapped, audit = build_provenance_map(
        base, support, old_rows, support_class_id, base_slot_class_id)
    assert torch.equal(mapped["support_slot_id"][:old_rows],
                       torch.full((old_rows,), -1, dtype=torch.long))
    assert torch.equal(mapped["base_slot_group_id"][:old_rows],
                       torch.full((old_rows,), -1, dtype=torch.long))
    assert set(mapped["support_slot_id"][old_rows:].tolist()) == {0, 1, 2, 3}
    assert all(row["capacity_conserved"] for row in audit["classes"])
    assert all(row["assigned_counts"] == row["slot_capacities"]
               for row in audit["classes"])


def test_provenance_x_routes_each_slot_group_and_preserves_payload_fields():
    base, support = _memory(8), _memory(36)
    old_rows = 4
    base["mass"][old_rows:] = 8
    support_class_id = torch.tensor(
        [-1] * old_rows + [0] * 16 + [1] * 16, dtype=torch.long)
    base_slot_class_id = torch.tensor(
        [-1] * old_rows + [0, 0, 1, 1], dtype=torch.long)
    mapped, _ = build_provenance_map(
        base, support, old_rows, support_class_id, base_slot_class_id)
    cfg = _config()
    output, audit = DSO.optimize(
        base, support, old_rows, [4, 5, 6, 7], "x-provenance", "cpu", cfg,
        support_slot_id=mapped["support_slot_id"],
        base_slot_group_id=mapped["base_slot_group_id"])
    assert audit["provenance_aware"] is True
    assert audit["outer_gradient_variables"] == ["x_provenance_relative"]
    assert set(audit["curve"][0]["train"]["slotwise"]) == {"0", "1", "2", "3"}
    assert torch.equal(base["x"][:old_rows], output["x"][:old_rows])
    assert torch.equal(base["text_emb"], output["text_emb"])
    assert torch.equal(base["edge_index"], output["edge_index"])
    assert any((base["x"][old_rows:] != output["x"][old_rows:]).any(1))


def test_slot_representation_x_routes_slot_targets_and_preserves_payload_fields():
    base, support = _memory(8), _memory(36)
    old_rows = 4
    base["mass"][old_rows:] = 8
    support_class_id = torch.tensor(
        [-1] * old_rows + [0] * 16 + [1] * 16, dtype=torch.long)
    base_slot_class_id = torch.tensor(
        [-1] * old_rows + [0, 0, 1, 1], dtype=torch.long)
    mapped, _ = build_provenance_map(
        base, support, old_rows, support_class_id, base_slot_class_id)
    cfg = DSO.Config(
        inner_steps=1, inner_batch=8, outer_steps=1, support_batch=8,
        slot_representation_weight=0.25, train_seeds=(1,),
        validation_seeds=(2,), plan_seed=3, a_pool=3,
    )
    output, audit = DSO.optimize(
        base, support, old_rows, [4, 5, 6, 7], "x-slot-rep", "cpu", cfg,
        support_slot_id=mapped["support_slot_id"],
        base_slot_group_id=mapped["base_slot_group_id"])
    assert audit["slot_representation_aware"] is True
    assert audit["outer_gradient_variables"] == ["x_slot_rep_relative"]
    assert any(
        row["gradient_norms"]["x_slot_rep_relative"] > 0
        for row in audit["curve"]
    )
    assert "slot_representation_loss" in (
        audit["curve"][0]["train"]["slotwise"]["0"]["seeds"][0])
    assert torch.equal(base["x"][:old_rows], output["x"][:old_rows])
    assert torch.equal(base["text_emb"], output["text_emb"])
    assert torch.equal(base["edge_index"], output["edge_index"])
    assert any((base["x"][old_rows:] != output["x"][old_rows:]).any(1))


def test_slot_representation_global_validation_skips_train_only_slot():
    base, support = _memory(6), _memory(13)
    old_rows = 4
    # Slot 0 has one raw row and therefore no held-out validation row under
    # the sealed 1/0 split; slot 1 still supplies the global validation loss.
    support_slot_id = torch.tensor(
        [-1] * old_rows + [0] + [1] * 8, dtype=torch.long)
    base_slot_group_id = torch.tensor(
        [-1] * old_rows + [0, 1], dtype=torch.long)
    cfg = DSO.Config(
        inner_steps=1, inner_batch=8, outer_steps=1, support_batch=8,
        slot_representation_weight=0.25, train_seeds=(1,),
        validation_seeds=(2,), plan_seed=3, a_pool=3,
    )
    output, audit = DSO.optimize(
        base, support, old_rows, [4, 5], "x-slot-rep", "cpu", cfg,
        support_slot_id=support_slot_id,
        base_slot_group_id=base_slot_group_id)
    assert audit["slot_representation_aware"] is True
    assert torch.equal(base["x"][:old_rows], output["x"][:old_rows])
    validation = audit["curve"][0]["validation"]["global"]["seeds"][0]
    assert validation["slot_representation_skipped_groups"] == [0]


def test_operator_aligned_provenance_uses_candidate_graph_and_slot_targets():
    base, support = _memory(8), _memory(36)
    old_rows = 4
    base["mass"][old_rows:] = 8
    support_class_id = torch.tensor(
        [-1] * old_rows + [0] * 16 + [1] * 16, dtype=torch.long)
    base_slot_class_id = torch.tensor(
        [-1] * old_rows + [0, 0, 1, 1], dtype=torch.long)
    mapped, _ = build_provenance_map(
        base, support, old_rows, support_class_id, base_slot_class_id)
    cfg = DSO.Config(
        inner_steps=1, inner_batch=8, outer_steps=1, support_batch=8,
        operator_aligned=True, train_seeds=(1,), validation_seeds=(2,),
        plan_seed=3, a_pool=3,
    )
    output, audit = DSO.optimize(
        base, support, old_rows, [4, 5, 6, 7], "x-provenance", "cpu", cfg,
        support_slot_id=mapped["support_slot_id"],
        base_slot_group_id=mapped["base_slot_group_id"])
    assert audit["config"]["operator_aligned"] is True
    assert audit["provenance_aware"] is True
    assert torch.equal(base["text_emb"], output["text_emb"])
    assert torch.equal(base["edge_index"], output["edge_index"])
    assert any((base["x"][old_rows:] != output["x"][old_rows:]).any(1))


def test_operator_aligned_prompt_objective_uses_new_slot_classes():
    base, support = _memory(8), _memory(36)
    old_rows = 4
    base["mass"][old_rows:] = 8
    support_class_id = torch.tensor(
        [-1] * old_rows + [0] * 16 + [1] * 16, dtype=torch.long)
    base_slot_class_id = torch.tensor(
        [-1] * old_rows + [0, 0, 1, 1], dtype=torch.long)
    mapped, _ = build_provenance_map(
        base, support, old_rows, support_class_id, base_slot_class_id)
    cfg = DSO.Config(
        inner_steps=1, inner_batch=8, outer_steps=1, support_batch=8,
        operator_aligned=True, prompt_weight=0.5,
        train_seeds=(1,), validation_seeds=(2,), plan_seed=3, a_pool=3,
    )
    prompts = torch.randn(2, 384, generator=torch.Generator().manual_seed(79))
    _, audit = DSO.optimize(
        base, support, old_rows, [4, 5, 6, 7], "x-provenance", "cpu", cfg,
        support_class_id=support_class_id, prompt_emb=prompts,
        prompt_ids=torch.tensor([0, 1]),
        base_slot_class_id=base_slot_class_id,
        support_slot_id=mapped["support_slot_id"],
        base_slot_group_id=mapped["base_slot_group_id"])
    assert audit["config"]["operator_aligned"] is True
    assert "prompt_loss" in audit["curve"][0]["train"]["slotwise"]["0"]["seeds"][0]


def test_operator_aligned_class_xp_uses_candidate_graph_and_class_targets():
    base, support = _memory(8), _memory(12)
    old_rows = 4
    support_class_id = torch.tensor(
        [-1] * old_rows + [0] * 4 + [1] * 4, dtype=torch.long)
    base_slot_class_id = torch.tensor(
        [-1] * old_rows + [0, 0, 1, 1], dtype=torch.long)
    cfg = DSO.Config(
        inner_steps=1, inner_batch=8, outer_steps=1, support_batch=8,
        operator_aligned=True, p_class_constrained=True,
        train_seeds=(1,), validation_seeds=(2,), plan_seed=3)
    output, audit = DSO.optimize(
        base, support, old_rows, [4, 5, 6, 7], "x-class+p", "cpu", cfg,
        support_class_id=support_class_id,
        base_slot_class_id=base_slot_class_id)
    assert audit["operator_aligned_aware"] is True
    assert audit["operator_aligned_classwise"] is True
    assert audit["p_class_constrained"] is True
    assert audit["materialization"]["p_cross_class_mass_max"] < 1e-5
    assert audit["materialization"]["p_cross_class_mass_total"] < 1e-5
    assert audit["outer_gradient_variables"] == [
        "x_class_relative", "p_logits"]
    assert torch.equal(base["edge_index"], output["edge_index"])
    assert torch.equal(base["x"][:old_rows], output["x"][:old_rows])
    assert "p_assignment_entropy" in audit
    assert any(
        audit["curve"][0]["gradient_norms"][name] > 0
        for name in ("x_class_relative", "p_logits"))


def test_freeze_text_inner_keeps_text_parameters_out_of_adam_trajectory():
    base, support = _memory(8), _memory(12)
    cfg = DSO.Config(
        inner_steps=2, inner_batch=8, outer_steps=1, support_batch=8,
        freeze_text_inner=True, train_seeds=(1,), validation_seeds=(2,),
        plan_seed=3, a_pool=3,
    )
    _, audit = DSO.optimize(
        base, support, 4, [4, 5, 6, 7], "x", "cpu", cfg)
    assert audit["config"]["freeze_text_inner"] is True
    assert audit["outer_gradient_variables"] == ["x_relative"]


def test_provenance_slot_plan_keeps_sealed_train_validation_membership():
    support = _memory(36)
    old_rows = 4
    slot_id = torch.full((len(support["x"]),), -1, dtype=torch.long)
    slot_id[old_rows:] = torch.arange(32) // 8
    plan = DSO.make_support_plan(
        support, old_rows, 8, 17, support_slot_id=slot_id,
        stratify_slot=True)
    filtered = DSO._slot_plan(plan, slot_id, 0, 8)
    train = {row for batch in filtered["splits"]["train"]["new"]
             for row in batch["targets"]}
    validation = {row for batch in filtered["splits"]["validation"]["new"]
                  for row in batch["targets"]}
    assert train.isdisjoint(validation)
    assert train | validation == set(torch.where(slot_id == 0)[0].tolist())


def test_old_response_barrier_records_c0_reference_and_response():
    base, support = _memory(8), _memory(12)
    cfg = DSO.Config(
        inner_steps=1, inner_batch=8, outer_steps=1, support_batch=8,
        old_barrier_weight=1.0, old_barrier_margin=0.02,
        train_seeds=(1,), validation_seeds=(2,), plan_seed=3, a_pool=3,
    )
    _, audit = DSO.optimize(
        base, support, 4, [4, 5, 6, 7], "x", "cpu", cfg)
    assert audit["old_barrier"]["weight"] == 1.0
    assert audit["old_barrier"]["validation_reference"] is not None
    record = audit["curve"][0]["validation"]
    assert "old_response" in record
    assert "old_reference" in record
    assert record["old_barrier"] >= 0.0


def test_mass_weighted_slot_aggregation_follows_frozen_payload_mass():
    losses = [torch.tensor(1.0), torch.tensor(3.0)]
    base = {"mass": torch.tensor([1.0, 1.0, 1.0, 1.0, 2.0, 8.0])}
    selected = torch.tensor([4, 5], dtype=torch.long)
    group_source = torch.tensor([-1, -1, -1, -1, 0, 1], dtype=torch.long)
    weighted = DSO._aggregate_slot_losses(
        losses, [0, 1], base, selected, group_source, True)
    equal = DSO._aggregate_slot_losses(
        losses, [0, 1], base, selected, group_source, False)
    assert torch.isclose(weighted, torch.tensor(2.6))
    assert torch.isclose(equal, torch.tensor(2.0))
