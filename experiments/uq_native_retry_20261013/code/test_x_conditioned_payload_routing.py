import torch
import torch.nn.functional as F

import cat_multimodal_minimal as MM
import x_conditioned_payload_routing as XCR


class Encoder:
    def encode(self, texts):
        rows = []
        for text in texts:
            value = float(sum(ord(ch) for ch in str(text)) % 101)
            rows.append([value + 1.0, value + 3.0,
                         (value % 7.0) + 2.0, (value % 11.0) + 4.0])
        return F.normalize(torch.tensor(rows), dim=1)


def fixture():
    g = torch.Generator().manual_seed(91)
    x = torch.randn(12, 5, generator=g)
    text = F.normalize(torch.randn(12, 4, generator=g), dim=1)
    return {
        "x": x,
        "text_emb": text,
        "y": torch.tensor([0] * 6 + [1] * 6),
        "ids": torch.arange(12) + 1000,
        "texts": [f"class-{int(i // 6)} source-{i} text payload"
                  for i in range(12)],
        "prompt_emb": F.normalize(torch.randn(2, 4, generator=g), dim=1),
        "stage": 1,
    }


def test_routing_materializes_row_specific_real_payloads():
    delta = fixture()
    x0 = torch.randn(6, 5, generator=torch.Generator().manual_seed(92))
    labels = torch.tensor([0, 0, 0, 1, 1, 1])
    block, audit = XCR.attach_x_conditioned_multi_payload(
        delta=delta,
        source_features=delta["x"],
        x0=x0,
        text0=delta["text_emb"][:6],
        base_texts=[f"base-{i}" for i in range(6)],
        base_source_ids=[[1000 + i] for i in range(6)],
        labels=labels,
        encoder=Encoder(),
        cfg=MM.MMConfig(text_sources=3, text_chars=48,
                        support_pool_size=8, chunk_chars=16,
                        payload_max_chunks=4, payload_max_sources=3,
                        seed=17),
        mass=torch.ones(6),
        device="cpu",
        max_slots=3,
        routing_steps=4,
    )
    assert block["x"].shape == x0.shape
    assert len(block["texts"]) == len(labels)
    assert all(len(text) <= 48 for text in block["texts"])
    assert audit["historical_raw_reads"] == 0
    assert audit["old_memory_updated"] is False
    assert audit["topology_updated"] is False
    assert audit["x_updated"] is False
    assert audit["hard_materialization"] is True
    assert len(audit["routing_by_class"]) == 2
    assert all(len(item["probabilities"]) == 3
               for item in audit["routing_by_class"].values())


def test_routing_is_reproducible_and_can_use_multiple_slots():
    delta = fixture()
    x0 = torch.randn(6, 5, generator=torch.Generator().manual_seed(93))
    labels = torch.tensor([0, 0, 0, 1, 1, 1])
    kwargs = dict(
        delta=delta, source_features=delta["x"], x0=x0,
        text0=delta["text_emb"][:6],
        base_texts=[f"base-{i}" for i in range(6)],
        base_source_ids=[[1000 + i] for i in range(6)], labels=labels,
        encoder=Encoder(),
        cfg=MM.MMConfig(text_sources=3, text_chars=48,
                        support_pool_size=8, chunk_chars=16,
                        payload_max_chunks=4, payload_max_sources=3,
                        seed=19),
        mass=torch.ones(6), device="cpu", max_slots=3,
        routing_steps=5)
    first, first_audit = XCR.attach_x_conditioned_multi_payload(**kwargs)
    second, second_audit = XCR.attach_x_conditioned_multi_payload(**kwargs)
    assert first["texts"] == second["texts"]
    assert torch.equal(first["text_emb"], second["text_emb"])
    assert first_audit["unique_payloads_by_class"] == \
        second_audit["unique_payloads_by_class"]


def test_response_coverage_changes_only_candidate_selection():
    delta = fixture()
    x0 = torch.randn(6, 5, generator=torch.Generator().manual_seed(94))
    labels = torch.tensor([0, 0, 0, 1, 1, 1])
    block, audit = XCR.attach_x_conditioned_multi_payload(
        delta=delta, source_features=delta["x"], x0=x0,
        text0=delta["text_emb"][:6],
        base_texts=[f"base-{i}" for i in range(6)],
        base_source_ids=[[1000 + i] for i in range(6)], labels=labels,
        encoder=Encoder(),
        cfg=MM.MMConfig(text_sources=3, text_chars=48,
                        support_pool_size=8, chunk_chars=16,
                        payload_max_chunks=4, payload_max_sources=3,
                        seed=23),
        mass=torch.ones(6), device="cpu", max_slots=3,
        routing_steps=4, candidate_mode="x_response_coverage")
    assert block["texts"]
    assert audit["candidate_selection"] == "x_response_coverage"
    assert audit["x_updated"] is False


def test_coherent_raw_medoid_candidates_are_materialized_from_current_delta():
    delta = fixture()
    x0 = torch.randn(6, 5, generator=torch.Generator().manual_seed(95))
    labels = torch.tensor([0, 0, 0, 1, 1, 1])
    block, audit = XCR.attach_x_conditioned_multi_payload(
        delta=delta, source_features=delta["x"], x0=x0,
        text0=delta["text_emb"][:6],
        base_texts=[f"base-{i}" for i in range(6)],
        base_source_ids=[[1000 + i] for i in range(6)], labels=labels,
        encoder=Encoder(),
        cfg=MM.MMConfig(text_sources=3, text_chars=48,
                        support_pool_size=8, chunk_chars=16,
                        payload_max_chunks=4, payload_max_sources=3,
                        seed=24),
        mass=torch.ones(6), device="cpu", max_slots=3,
        routing_steps=4, candidate_mode="coherent_raw_medoid")
    assert block["texts"]
    assert audit["candidate_selection"] == "coherent_raw_medoid"
    assert audit["historical_raw_reads"] == 0


def test_balanced_hard_materialization_controls_slot_capacity():
    probabilities = torch.tensor([
        [0.90, 0.08, 0.02],
        [0.89, 0.09, 0.02],
        [0.88, 0.10, 0.02],
        [0.70, 0.20, 0.10],
        [0.60, 0.30, 0.10],
    ])
    selected, usage = XCR._balanced_hard_slots(probabilities)
    assert sorted(usage) == [1, 2, 2]
    assert len(selected) == len(probabilities)


def test_cctp_anchor_swap_keeps_materialized_text_auditable():
    delta = fixture()
    x0 = torch.randn(6, 5, generator=torch.Generator().manual_seed(96))
    labels = torch.tensor([0, 0, 0, 1, 1, 1])
    base_texts = [f"base-{i}" for i in range(6)]
    base_ids = [[1000 + i] for i in range(6)]
    block, audit = XCR.attach_x_conditioned_multi_payload(
        delta=delta, source_features=delta["x"], x0=x0,
        text0=delta["text_emb"][:6], base_texts=base_texts,
        base_source_ids=base_ids, labels=labels, encoder=Encoder(),
        cfg=MM.MMConfig(text_sources=3, text_chars=48,
                        support_pool_size=8, chunk_chars=16,
                        payload_max_chunks=4, payload_max_sources=3,
                        seed=25), mass=torch.ones(6), device="cpu",
        max_slots=3, routing_steps=4,
        candidate_mode="coherent_raw_medoid_anchor",
        cctp_anchor_swap=True)
    assert block["texts"]
    assert audit["cctp_anchor_swap"] is True
    assert all(text in base_texts or text.startswith("class-")
               for text in block["texts"])

