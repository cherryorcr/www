from dataclasses import asdict

import pytest
import torch
import torch.nn.functional as F

import run_unified_q_attribution_20261008 as A
from test_unified_q_multimodal import _delta


class Encoder:
    def encode(self, texts):
        rows = [torch.randn(8, generator=torch.Generator().manual_seed(sum(map(ord, text))))
                for text in texts]
        return F.normalize(torch.stack(rows), dim=1)


def fixture():
    torch.manual_seed(4)
    delta = _delta()
    labels = torch.tensor([0, 0, 1, 1])
    features = A.M.propagated_features(delta, (0., 1.), "cpu")
    block, audit = A.M.attach_text(delta, features, delta["x"][[0,1,6,7]], labels,
                                  Encoder(), A.M.Config())
    block.update(labels=labels, text_source_ids=audit["text_source_ids"])
    # Nonuniform masses must survive replacing text.
    block["mass"] = torch.tensor([1., 5., 2., 4.])
    return delta, features, block


@pytest.mark.parametrize("mode,top_k", [("nearest",1),("nearest",5),("random",5)])
def test_text_controls_change_only_materialized_text(mode, top_k):
    delta, features, base = fixture()
    before = base["text_emb"].clone()
    block, audit = A.text_control(delta, features, base, Encoder(), mode, top_k, 31)
    assert A.assert_fixed_content(base, block)["fixed_content_exact"]
    assert torch.equal(before, base["text_emb"])
    assert A.validate_payload(block, delta, Encoder())["same_class_text_sources"]
    assert all(len(sources) == top_k for sources in block["text_source_ids"])
    replay, _ = A.text_control(delta, features, base, Encoder(), mode, top_k, 31)
    assert A.V.memory_hash(block) == A.V.memory_hash(replay)


def test_no_historical_or_cross_class_text_source_is_accepted():
    delta, _, base = fixture()
    for illegal in (999, 9):
        changed = dict(base, text_source_ids=[[illegal], *base["text_source_ids"][1:]])
        with pytest.raises(AssertionError, match="escaped"):
            A.validate_payload(changed, delta, Encoder())


def test_uq_nearest1_uses_exactly_the_primary_candidate_strings():
    delta, features, base = fixture()
    delta["texts"] = [("raw   words " * 80) + str(i) for i in range(12)]
    block, _ = A.text_control(delta, features, base, Encoder(), "nearest", 1, 31,
                               uq_materialization=True)
    candidates = {int(i): A.UQ._append_text(t, 320)
                  for i, t in zip(delta["ids"], delta["texts"])}
    assert block["texts"] == [candidates[v[0]] for v in block["text_source_ids"]]


def test_fixed_content_check_rejects_changed_mass():
    _, _, base = fixture()
    with pytest.raises(AssertionError, match="mass"):
        A.assert_fixed_content(base, dict(base, mass=base["mass"] + 1))


def test_incremental_archives_reconstruct_exact_payload_without_copying_x(tmp_path):
    delta, features, base = fixture()
    A.archive_block(tmp_path, "cat_cctp", 0, base, {})
    variant, audit = A.text_control(delta, features, base, Encoder(), "random", 5, 31)
    saved = A.archive_block(tmp_path, "cat_random5", 0, variant, audit, "cat_cctp")
    assert "x" not in saved["payload"]
    loaded, _ = A.load_block(tmp_path, "cat_random5", 0)
    assert A.V.memory_hash(loaded) == A.V.memory_hash(variant)
    parent = A.append_block(None, loaded)
    full = A.append_block(parent, loaded)
    assert torch.equal(full["x"][:4], parent["x"])
    assert full["text_source_ids"] == loaded["text_source_ids"] * 2


def test_adaptive_receives_historical_configuration_and_identical_cat_x(monkeypatch):
    delta, features, base = fixture()
    seen = {}
    def fake(**kwargs):
        seen.update(kwargs)
        return base, {"text_source_ids": base["text_source_ids"]}
    monkeypatch.setattr(A.XCR, "attach_x_conditioned_multi_payload", fake)
    block, _ = A.adaptive_payload(delta, features, base, Encoder(), 271006, "cpu")
    assert seen["x0"] is base["x"]
    assert seen["adaptive_slots"] and not seen["cctp_anchor_swap"]
    assert seen["candidate_mode"] == "coherent_raw_medoid"
    assert seen["cfg"].seed == 271006 + 76
    A.assert_fixed_content(base, block)


def test_auxiliary_work_cannot_advance_constructor_rng():
    torch.manual_seed(8)
    before = torch.random.get_rng_state().clone()
    with A.preserve_rng():
        torch.rand(32)
    assert torch.equal(before, torch.random.get_rng_state())


def test_legacy_and_prompt_have_identical_initial_architecture_and_memory_operator():
    torch.manual_seed(150)
    old = A.L.U.G.CLIPStudent(4)
    torch.manual_seed(150)
    new = A.P.GraphTextPromptStudent(4)
    for key, value in old.state_dict().items():
        assert torch.equal(value, new.state_dict()[key])
    edges = torch.empty((2,0), dtype=torch.long)
    assert torch.equal(A.L.U.G.norm_adj(edges, 6, "cpu").to_dense(),
                       A.P.normalized_adjacency(edges, 6, "cpu").to_dense())


def test_primary_configuration_remains_frozen():
    assert asdict(A.R.q_config("uq_dm_edgefree", 271006, 30, 2048)) == asdict(
        A.R.I.config("uq_dm_edgefree", 271006, 30, 2048))
