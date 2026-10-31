import torch

import teacher_guided_message_revision as M
import run_teacher_guided_message_revision as R


def test_assignment_projection_preserves_support_rows_and_is_differentiable():
    assignment = {
        "indices": torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        "logits": torch.tensor([[0.2, 0.1], [0.0, 0.4]], requires_grad=True),
        "distance": torch.tensor([[0.1, 0.2, 0.8], [0.8, 0.2, 0.1]]),
    }
    q, slot_to_raw = M._assignment_matrix(
        assignment, old_rows=1, support_rows=4, candidate_rows=3, device="cpu")
    assert q.shape == (4, 3)
    assert torch.allclose(q.sum(1), torch.ones(4))
    assert torch.allclose(slot_to_raw.sum(1), torch.ones(2))
    loss = q.square().sum()
    loss.backward()
    assert assignment["logits"].grad is not None


def test_materialize_edges_respects_base_row_degree():
    base = torch.tensor([[0, 0, 1, 2], [1, 2, 2, 0]], dtype=torch.long)
    weight = torch.tensor([[0.0, 0.9, 0.8], [0.2, 0.0, 0.7],
                           [0.6, 0.1, 0.0]])
    edge = M._materialize_edges(weight, base)
    assert edge.shape[1] == base.shape[1]
    assert int(torch.bincount(edge[0], minlength=3)[0]) == 2
    assert int(torch.bincount(edge[0], minlength=3)[1]) == 1
    assert int(torch.bincount(edge[0], minlength=3)[2]) == 1


def test_bounded_materialization_preserves_low_degree_base_rows():
    base = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    weight = torch.tensor([[0.0, 0.1, 0.9], [0.8, 0.0, 0.2],
                           [0.3, 0.7, 0.0]])
    edge = M._materialize_edges(weight, base, swap_fraction=0.1)
    assert set(map(tuple, edge.T.tolist())) == set(map(tuple, base.T.tolist()))


def test_old_provenance_creates_balanced_groups():
    class_id = torch.tensor([0, 0, 1, -1, 2, 2])
    groups = M._groups_with_provenance(class_id, old_rows=4)
    assert groups[0].tolist() == [0, 1]
    assert groups[1].tolist() == [2]
    assert groups[2].tolist() == [4, 5]


def test_sealed_allocation_expands_in_row_order():
    values = R._classes_from_allocation({2: 3, 1: 2}, rows=5)
    assert values.tolist() == [1, 1, 2, 2, 2]
    assert R._classes_from_allocation({1: 2}, rows=5) is None


def test_class_mix_keeps_legacy_old_mass_and_adds_new_rows():
    groups = {
        1: torch.tensor([0, 1]),
        2: torch.tensor([2, 3]),
        5: torch.tensor([4, 5, 6]),
        6: torch.tensor([7, 8, 9]),
    }
    rows = M._rows_with_class_mix(groups, batch=8, seed=7, device="cpu",
                                  old_rows=4, old_class_mix=0.25)
    assert len(rows) == 6
    assert int((rows < 4).sum()) == 2
    assert int((rows >= 4).sum()) == 4


def test_old_and_new_rows_with_same_class_are_kept_in_one_group():
    class_id = torch.tensor([2, 2, 3, 2, 4, 4])
    groups = M._groups_with_provenance(class_id, old_rows=3)
    assert groups[2].tolist() == [0, 1, 3]
    assert groups[4].tolist() == [4, 5]


def test_straight_through_forward_matches_materialized_topology():
    base = torch.tensor([[0, 0, 1, 2], [1, 2, 2, 0]], dtype=torch.long)
    mixed = torch.tensor([[0.0, 0.9, 0.8],
                          [0.2, 0.0, 0.7],
                          [0.6, 0.1, 0.0]], requires_grad=True)
    straight, hard = M._straight_through_adjacency(mixed, base, 0.5)
    expected = M._normalize_dense(hard)
    assert torch.allclose(straight.detach(), expected)
    straight.sum().backward()
    assert mixed.grad is not None


def test_new_plasticity_loss_has_gradient_and_handles_single_class():
    graph = torch.randn(3, 5, requires_grad=True)
    text = torch.randn(3, 5, requires_grad=True)
    classes = torch.tensor([0, 0, 1])
    loss = M._new_plasticity_loss(graph, text, classes, margin=0.2)
    assert torch.isfinite(loss)
    loss.backward()
    assert graph.grad is not None
    assert text.grad is not None


def test_baseline_gate_can_fallback_to_identity_memory():
    base = {
        "x": torch.randn(5, 3),
        "text_emb": torch.randn(5, 4),
        "texts": ["a", "b", "c", "d", "e"],
        "mass": torch.ones(5),
        "depth": torch.zeros(5, dtype=torch.long),
        "stage": torch.zeros(5, dtype=torch.long),
        "edge_index": torch.tensor([[0, 1, 2, 3, 4], [1, 2, 0, 4, 3]], dtype=torch.long),
    }
    support = {
        **base,
        "x": torch.cat([base["x"], torch.randn(2, 3)]),
        "text_emb": torch.cat([base["text_emb"], torch.randn(2, 4)]),
        "texts": ["a", "b", "c", "d", "e", "f", "g"],
        "mass": torch.ones(7),
        "depth": torch.zeros(7, dtype=torch.long),
        "stage": torch.zeros(7, dtype=torch.long),
        "edge_index": torch.tensor(
            [[0, 1, 2, 3, 4, 5, 6], [1, 2, 0, 4, 3, 6, 5]], dtype=torch.long),
        "class_id": torch.tensor([-1, -1, -1, 0, 1, 0, 1], dtype=torch.long),
    }
    cfg = __import__("teacher_guided_payload_revision").Config(
        teacher_steps=1, inner_steps=1, outer_steps=1, batch=3, top_r=1,
        teacher_seeds=(101,), validation_teacher_seeds=(102,),
        baseline_gate=True, gate_margin=1e9)
    result, audit, _ = M.optimize(
        base, support, torch.tensor([-1, -1, -1, 0, 1, 0, 1]), 3, 1,
        "cpu", cfg, learn_x=True, learn_p=False, message_mix=0.2,
        swap_fraction=0.1)
    assert audit["gate_fallback"] is True
    assert torch.equal(result["x"], base["x"])
    assert torch.equal(result["edge_index"], base["edge_index"])
