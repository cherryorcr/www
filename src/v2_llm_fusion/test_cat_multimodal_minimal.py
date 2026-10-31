import torch
import torch.nn.functional as F

import cat_multimodal_minimal as MM


def fixture():
    g = torch.Generator().manual_seed(11)
    return {
        "x": torch.randn(12, 5, generator=g),
        "text_emb": F.normalize(torch.randn(12, 4, generator=g), dim=1),
        "y": torch.tensor([0] * 6 + [1] * 6),
        "stage": 1,
    }


class Encoder:
    def encode(self, texts):
        rows = []
        for text in texts:
            value = float(sum(ord(ch) for ch in text) % 97)
            rows.append([value + 1.0, value + 2.0, value + 3.0, value + 4.0])
        return F.normalize(torch.tensor(rows), dim=1)


def test_text_only_preserves_x_and_is_reproducible():
    delta = fixture()
    source = delta["x"]
    x0 = torch.randn(4, 5, generator=torch.Generator().manual_seed(12))
    t0 = F.normalize(torch.randn(4, 4, generator=torch.Generator().manual_seed(13)), dim=1)
    labels = torch.tensor([0, 0, 1, 1])
    first, _ = MM.optimize_text_payload(
        delta, source, x0, t0, ["a"] * 4, [[i] for i in range(4)], labels,
        None, MM.MMConfig(steps=3, hidden=8), "text_only", None, "cpu")
    second, _ = MM.optimize_text_payload(
        delta, source, x0, t0, ["a"] * 4, [[i] for i in range(4)], labels,
        None, MM.MMConfig(steps=3, hidden=8), "text_only", None, "cpu")
    assert torch.equal(first["x"], x0)
    assert torch.equal(first["text_emb"], second["text_emb"])


def test_joint_changes_only_new_block():
    delta = fixture()
    x0 = torch.randn(4, 5, generator=torch.Generator().manual_seed(14))
    t0 = F.normalize(torch.randn(4, 4, generator=torch.Generator().manual_seed(15)), dim=1)
    labels = torch.tensor([0, 0, 1, 1])
    block, audit = MM.optimize_text_payload(
        delta, delta["x"], x0, t0, ["a"] * 4, [[i] for i in range(4)], labels,
        None, MM.MMConfig(steps=3, hidden=8), "joint", None, "cpu")
    assert block["x"].shape == x0.shape
    assert block["text_emb"].shape == t0.shape
    assert audit["current_delta_only"] is True


def test_text_aware_cat_updates_x_through_soft_assignment():
    delta = fixture()
    x0 = torch.randn(4, 5, generator=torch.Generator().manual_seed(21))
    t0 = F.normalize(torch.randn(4, 4, generator=torch.Generator().manual_seed(22)), dim=1)
    labels = torch.tensor([0, 0, 1, 1])
    cfg = MM.MMConfig(steps=4, hidden=8, lr_x=.01,
                      lambda_cross=.5, text_sources=2, text_chars=32)
    block, audit = MM.optimize_text_aware_cat(
        delta, delta["x"], x0, t0, ["a"] * 4, [[i] for i in range(4)],
        labels, None, Encoder(), cfg, None, "cpu", materialize="soft")
    assert block["x"].shape == x0.shape
    assert block["text_emb"].shape == t0.shape
    assert audit["constructor"] == "cat_text_aware_soft"
    assert audit["objective"].startswith("CaT graph MMD")
    assert audit["x_update_norm"] > 0
    assert audit["historical_raw_reads"] == 0
    assert all(sum(item["candidate_probabilities"]) > .99
               for item in audit["soft_assignment"])


def test_text_aware_hard_materialization_is_auditable():
    delta = fixture()
    delta["ids"] = torch.arange(len(delta["x"])) + 100
    delta["texts"] = [f"source-{i}" for i in range(len(delta["x"]))]
    x0 = torch.randn(4, 5, generator=torch.Generator().manual_seed(23))
    t0 = F.normalize(torch.randn(4, 4, generator=torch.Generator().manual_seed(24)), dim=1)
    labels = torch.tensor([0, 0, 1, 1])
    block, audit = MM.optimize_text_aware_cat(
        delta, delta["x"], x0, t0, ["a"] * 4, [[i] for i in range(4)],
        labels, None, Encoder(), MM.MMConfig(steps=2, hidden=8,
                                              text_sources=2, text_chars=24),
        None, "cpu", materialize="hard")
    assert len(block["texts"]) == len(labels)
    assert max(map(len, block["texts"])) <= 24
    assert len(audit["hard_materialization"]) == len(labels)
    assert all(len(ids) == 2 for ids in audit["text_source_ids"])
    assert float(audit["soft_hard_cosine_min"]) <= 1.0


def test_text_group_aware_cat_materializes_selected_group():
    delta = fixture()
    delta["ids"] = torch.arange(len(delta["x"])) + 100
    delta["texts"] = [f"source-{i}-semantic payload" for i in range(len(delta["x"]))]
    x0 = torch.randn(4, 5, generator=torch.Generator().manual_seed(31))
    t0 = F.normalize(torch.randn(4, 4, generator=torch.Generator().manual_seed(32)), dim=1)
    labels = torch.tensor([0, 0, 1, 1])
    cfg = MM.MMConfig(steps=3, hidden=8, lr_x=.01, text_sources=2,
                      text_chars=28, text_group_count=3)
    block, audit = MM.optimize_text_group_aware_cat(
        delta, delta["x"], x0, t0, ["a"] * 4, [[i] for i in range(4)],
        labels, None, Encoder(), cfg, None, "cpu")
    assert block["x"].shape == x0.shape
    assert block["text_emb"].shape == t0.shape
    assert audit["constructor"] == "cat_text_group_aware_x"
    assert audit["historical_raw_reads"] == 0
    assert audit["topology_updated"] is False
    assert all(len(ids) == 2 for ids in audit["text_source_ids"])
    assert all(len(text) <= 28 for text in block["texts"])
    assert len(audit["group_bank"]) == 6
    assert float(audit["soft_hard_cosine_min"]) >= -1.0


def test_joint_cat_text_group_updates_x_and_keeps_text_materializable():
    delta = fixture()
    delta["ids"] = torch.arange(len(delta["x"])) + 200
    delta["texts"] = [f"joint-source-{i}-semantic payload"
                      for i in range(len(delta["x"]))]
    delta["prompt_emb"] = F.normalize(torch.randn(
        2, 4, generator=torch.Generator().manual_seed(41)), dim=1)
    x0 = torch.randn(4, 5, generator=torch.Generator().manual_seed(42))
    t0 = F.normalize(torch.randn(4, 4, generator=torch.Generator().manual_seed(43)), dim=1)
    labels = torch.tensor([0, 0, 1, 1])
    cfg = MM.MMConfig(steps=3, hidden=8, lr_x=.005, lr_text=.005,
                      text_sources=2, text_chars=30, text_group_count=3)
    block, audit = MM.optimize_joint_cat_text_groups(
        delta, delta["x"], x0, t0, ["a"] * 4,
        [[200, 201], [200, 201], [206, 207], [206, 207]],
        labels, None, Encoder(), cfg, None, "cpu")
    assert block["x"].shape == x0.shape
    assert block["text_emb"].shape == t0.shape
    assert audit["constructor"] == "cat_joint_text_group"
    assert audit["x_update_norm"] > 0
    assert audit["historical_raw_reads"] == 0
    assert audit["old_memory_updated"] is False
    assert all(len(ids) == 2 for ids in audit["text_source_ids"])
    assert all(len(text) <= 30 for text in block["texts"])
    assert all(item["selected_group_local"] >= 0
               for item in audit["hard_group_materialization"])


def test_gated_joint_cat_text_preserves_cctp_noop_path():
    delta = fixture()
    delta["ids"] = torch.arange(len(delta["x"])) + 300
    delta["texts"] = [f"gated-source-{i}-semantic payload"
                      for i in range(len(delta["x"]))]
    delta["prompt_emb"] = F.normalize(torch.randn(
        2, 4, generator=torch.Generator().manual_seed(51)), dim=1)
    x0 = torch.randn(4, 5, generator=torch.Generator().manual_seed(52))
    t0 = F.normalize(torch.randn(4, 4, generator=torch.Generator().manual_seed(53)), dim=1)
    labels = torch.tensor([0, 0, 1, 1])
    base_ids = [[300, 301], [300, 301], [306, 307], [306, 307]]
    cfg = MM.MMConfig(steps=3, hidden=8, lr_x=.001, text_sources=2,
                      text_chars=30, text_group_count=3,
                      joint_base_logit=2.0, joint_selection_margin=0.05,
                      joint_x_trust_radius=.01)
    block, audit = MM.optimize_gated_joint_cat_text(
        delta, delta["x"], x0, t0, ["a"] * 4, base_ids, labels, None,
        Encoder(), cfg, None, "cpu")
    assert block["x"].shape == x0.shape
    assert block["text_emb"].shape == t0.shape
    assert audit["constructor"] == "cat_gated_joint_text_group"
    assert audit["historical_raw_reads"] == 0
    assert audit["x_update_relative_max"] <= .010001
    assert all(len(ids) == 2 for ids in audit["text_source_ids"])
    assert all(len(text) <= 30 for text in block["texts"])
    assert all("changed_from_cctp" in item
               for item in audit["hard_group_materialization"])


def test_chunk_split_prefers_boundaries_and_respects_limit():
    chunks = MM._split_text_chunks(
        "alpha sentence. beta sentence; gamma sentence.", 16)
    assert len(chunks) >= 3
    assert all(len(item["text"]) <= 16 for item in chunks)
    assert all(item["start"] <= item["end"] for item in chunks)


def test_cat_conditioned_chunk_payloads_are_materialized_and_bounded():
    delta = fixture()
    delta["ids"] = torch.arange(len(delta["x"])) + 500
    delta["texts"] = [
        f"class {int(delta['y'][i])} source {i}. "
        "This is a second semantic sentence for chunk selection."
        for i in range(len(delta["x"]))
    ]
    delta["prompt_emb"] = F.normalize(torch.randn(
        2, 4, generator=torch.Generator().manual_seed(61)), dim=1)
    x0 = torch.randn(4, 5, generator=torch.Generator().manual_seed(62))
    t0 = F.normalize(torch.randn(4, 4,
                                  generator=torch.Generator().manual_seed(63)), dim=1)
    labels = torch.tensor([0, 0, 1, 1])
    base_texts = ["class 0 source 0", "class 0 source 1",
                  "class 1 source 6", "class 1 source 7"]
    base_ids = [[500], [501], [506], [507]]
    cfg = MM.MMConfig(support_pool_size=4, chunk_chars=24,
                      payload_max_chunks=5, payload_max_sources=3,
                      text_chars=54)
    for mode in ("cctp", "text_mmr", "tagsam_chunk", "graph_role",
                 "anchor_mix", "class_shared_tagsam",
                 "class_shared_anchor_mix"):
        block, audit = MM.attach_chunk_payload(
            delta, delta["x"], x0, t0, base_texts, base_ids, labels,
            Encoder(), cfg, mode, None, "cpu")
        assert torch.equal(block["x"], x0)
        assert audit["historical_raw_reads"] == 0
        assert audit["x_updated"] is False
        assert all(len(text) <= 54 for text in block["texts"])
        assert len(audit["candidate_bank"]) == len(labels)
        if mode != "cctp":
            assert all(len(chunks) >= 1
                       for chunks in audit["text_chunk_provenance"])
    assert len(set(block["texts"])) <= 2
    assert all(len(text) <= 54 for text in block["texts"])


def test_two_class_shared_payload_uses_deterministic_x_prototypes():
    delta = fixture()
    delta["ids"] = torch.arange(len(delta["x"])) + 900
    delta["texts"] = [
        f"class {int(delta['y'][i])} source {i}. semantic payload"
        for i in range(len(delta["x"]))
    ]
    x0 = torch.randn(4, 5, generator=torch.Generator().manual_seed(91))
    t0 = F.normalize(torch.randn(4, 4,
                                  generator=torch.Generator().manual_seed(92)), dim=1)
    labels = torch.tensor([0, 0, 1, 1])
    cfg = MM.MMConfig(support_pool_size=4, chunk_chars=24,
                      payload_max_chunks=4, payload_max_sources=2,
                      text_chars=54)
    block, audit = MM.attach_chunk_payload(
        delta, delta["x"], x0, t0,
        ["class 0 source 0", "class 0 source 1",
         "class 1 source 6", "class 1 source 7"],
        [[900], [901], [906], [907]], labels, Encoder(), cfg,
        "class_shared_two_tagsam", None, "cpu")
    assert torch.equal(block["x"], x0)
    assert audit["class_prototype_count"] == 2
    assert len(audit["class_representative_rows"]) == 2
    assert len(audit["class_row_assignments"]) == len(labels)
    assert all(len(text) <= 54 for text in block["texts"])
    assert all(len(chunks) >= 1
               for chunks in audit["text_chunk_provenance"])


def test_representative_only_class_shared_matches_rowwise_output():
    delta = fixture()
    delta["ids"] = torch.arange(len(delta["x"])) + 1200
    delta["texts"] = [
        f"class {int(delta['y'][i])} source {i}. semantic payload"
        for i in range(len(delta["x"]))
    ]
    x0 = torch.randn(4, 5, generator=torch.Generator().manual_seed(121))
    t0 = F.normalize(torch.randn(4, 4,
                                 generator=torch.Generator().manual_seed(122)), dim=1)
    labels = torch.tensor([0, 0, 1, 1])
    base_texts = ["class 0 source 0", "class 0 source 1",
                  "class 1 source 6", "class 1 source 7"]
    base_ids = [[1200], [1201], [1206], [1207]]
    cfg = MM.MMConfig(support_pool_size=4, chunk_chars=24,
                      payload_max_chunks=4, payload_max_sources=2,
                      text_chars=54)
    rowwise, rowwise_audit = MM.attach_chunk_payload(
        delta, delta["x"], x0, t0, base_texts, base_ids, labels,
        Encoder(), cfg, "class_shared_tagsam", None, "cpu")
    fast, fast_audit = MM.attach_class_shared_chunk_payload(
        delta, delta["x"], x0, t0, base_texts, base_ids, labels,
        Encoder(), cfg, None, "cpu")
    assert torch.equal(rowwise["x"], fast["x"])
    assert torch.allclose(rowwise["text_emb"], fast["text_emb"])
    assert rowwise["texts"] == fast["texts"]
    assert rowwise_audit["text_source_ids"] == fast_audit["text_source_ids"]
    assert fast_audit["representative_only_construction"] is True


def test_chunk_payload_support_pool_is_class_restricted():
    delta = fixture()
    delta["ids"] = torch.arange(len(delta["x"])) + 700
    delta["texts"] = [f"text-{i}. extra content" for i in range(len(delta["x"]))]
    x0 = torch.randn(2, 5, generator=torch.Generator().manual_seed(71))
    labels = torch.tensor([0, 1])
    cfg = MM.MMConfig(support_pool_size=3, chunk_chars=20,
                      payload_max_chunks=3, payload_max_sources=2)
    pools = MM._build_catted_chunk_support_pool(
        delta, delta["x"], x0, labels, Encoder(), cfg)
    for pool in pools:
        assert all(int(delta["y"][i]) == int(pool["class"])
                   for i in pool["support_local_indices"])
