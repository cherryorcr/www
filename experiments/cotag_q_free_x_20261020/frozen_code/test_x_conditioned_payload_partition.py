import torch
import torch.nn.functional as F

import x_conditioned_payload_partition as XPP


class Encoder:
    def encode(self, texts):
        rows = []
        for text in texts:
            value = float(sum(ord(ch) for ch in text) % 97)
            rows.append([value + 1.0, value + 2.0,
                         len(text) + 1.0, 1.0])
        return F.normalize(torch.tensor(rows), dim=1)


def fixture():
    generator = torch.Generator().manual_seed(19)
    y = torch.tensor([0] * 6 + [1] * 6)
    return {
        "ids": torch.arange(100, 112),
        "stage": 2,
        "x": torch.randn(12, 5, generator=generator),
        "y": y,
        "text_emb": F.normalize(torch.randn(12, 4, generator=generator), dim=1),
        "texts": [f"class {int(cls)} source {i} semantic payload"
                  for i, cls in enumerate(y.tolist())],
    }


def test_slot_partition_materializes_row_specific_payloads():
    delta = fixture()
    source = delta["x"]
    x0 = torch.randn(4, 5, generator=torch.Generator().manual_seed(20))
    labels = torch.tensor([0, 0, 1, 1])
    cfg = XPP.PartitionConfig(
        steps=3, hidden=8, lr_x=1e-3, sinkhorn_iters=4,
        text_chars=48, max_sources=2, seed=21)
    block, audit = XPP.build_x_conditioned_slot_partition(
        delta, source, x0, labels, Encoder(), cfg, torch.ones(4), "cpu")
    assert block["x"].shape == x0.shape
    assert block["text_emb"].shape == (4, 4)
    assert len(block["texts"]) == 4
    assert len(set(block["texts"])) == 4
    assert audit["class_shared"] is False
    assert audit["slot_count_by_class"] == {"0": 2, "1": 2}
    assert audit["unique_text_count"] == 4
    assert audit["historical_raw_reads"] == 0
    assert audit["old_memory_updated"] is False
    assert audit["topology_updated"] is False
    assert audit["x_updated"] is True
    assert len(audit["text_source_ids"]) == 4
    assert all(set(ids).issubset(set(delta["ids"].tolist()))
               for ids in audit["text_source_ids"])
    assert max(map(len, block["texts"])) <= 48
    assert audit["x_update_relative_max"] <= cfg.trust_radius + 1e-5


def test_slot_partition_is_reproducible_and_current_delta_only():
    delta = fixture()
    x0 = torch.randn(4, 5, generator=torch.Generator().manual_seed(22))
    labels = torch.tensor([0, 0, 1, 1])
    cfg = XPP.PartitionConfig(
        steps=2, hidden=8, sinkhorn_iters=3,
        text_chars=48, max_sources=2, seed=23)
    first, first_audit = XPP.build_x_conditioned_slot_partition(
        delta, delta["x"], x0, labels, Encoder(), cfg, torch.ones(4), "cpu")
    second, second_audit = XPP.build_x_conditioned_slot_partition(
        delta, delta["x"], x0, labels, Encoder(), cfg, torch.ones(4), "cpu")
    assert torch.equal(first["x"], second["x"])
    assert torch.equal(first["text_emb"], second["text_emb"])
    assert first["texts"] == second["texts"]
    assert first_audit["text_source_ids"] == second_audit["text_source_ids"]
    assert first_audit["current_delta_only"] is True
    assert first_audit["report_test_labels_used_for_construction"] is False


def test_straight_through_sinkhorn_exposes_cost_gradient():
    cost = torch.tensor([[0.1, 0.8], [0.7, 0.2]], requires_grad=True)
    forward = XPP._sinkhorn_plan(cost.detach(), 0.07, 6)
    backward = XPP._sinkhorn_plan(cost, 0.07, 6)
    plan = forward + (backward - backward.detach())
    (plan * cost).sum().backward()
    assert cost.grad is not None
    assert float(cost.grad.abs().sum()) > 0.0


def test_anchor_and_hard_consistency_variants_are_audited():
    delta = fixture()
    x0 = torch.randn(4, 5, generator=torch.Generator().manual_seed(24))
    labels = torch.tensor([0, 0, 1, 1])
    cfg = XPP.PartitionConfig(
        steps=3, hidden=8, sinkhorn_iters=4, text_chars=48,
        max_sources=2, seed=25, straight_through_sinkhorn=True,
        lambda_coherence=0.25, lambda_hard=0.25, hard_refresh_steps=2)
    block, audit = XPP.build_x_conditioned_slot_partition(
        delta, delta["x"], x0, labels, Encoder(), cfg,
        torch.ones(4), "cpu")
    assert block["text_emb"].shape == (4, 4)
    assert audit["transport_gradient"].startswith("straight-through")
    assert audit["coherence_anchor"]["enabled"] is True
    assert audit["hard_consistency"]["enabled"] is True
    assert audit["trace"][-1]["hard"] >= 0.0


def test_slot_audit_uses_normalized_column_mass():
    delta = fixture()
    x0 = torch.randn(4, 5, generator=torch.Generator().manual_seed(26))
    labels = torch.tensor([0, 0, 1, 1])
    cfg = XPP.PartitionConfig(
        steps=1, hidden=8, sinkhorn_iters=3, text_chars=48,
        max_sources=2, seed=27)
    _, audit = XPP.build_x_conditioned_slot_partition(
        delta, delta["x"], x0, labels, Encoder(), cfg,
        torch.ones(4), "cpu")
    for slot in audit["slot_audit"]:
        assert slot["assignment_entropy"] >= 0.0
        assert slot["assignment_effective_sources"] <= 6.0 + 1e-5
        assert all(0.0 <= value <= 1.0
                   for value in slot["top_source_weights"])

