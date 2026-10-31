"""Protocol-critical tests; no data files, labels outside delta, or network."""
from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

import external_continual_mechanisms as M


class Encoder:
    def encode(self, texts):
        return F.normalize(torch.tensor([[len(t), sum(map(ord, t)) % 31, 1.0]
                                         for t in texts], dtype=torch.float32), dim=1)


def fixture():
    gen = torch.Generator().manual_seed(31)
    return {"x": torch.randn(30, 6, generator=gen),
            "y": torch.tensor([8] * 7 + [2] * 23),
            "ids": torch.arange(100, 130), "stage": 1,
            "texts": [f"current node {i} text" for i in range(30)],
            "text_emb": F.normalize(torch.randn(30, 8, generator=gen), dim=1),
            "edge_index": torch.stack((torch.arange(29), torch.arange(1, 30)))}


def test_ratio_imbalance_and_row_labels():
    delta = fixture()
    allocation = M.ratio_allocation(delta["y"], 7)
    assert allocation == {2: 5, 8: 2}
    assert M.row_labels(allocation).tolist() == [2] * 5 + [8] * 2
    assert M.ratio_allocation(delta["y"], 30) == {2: 23, 8: 7}
    with pytest.raises(ValueError):
        M.ratio_allocation(delta["y"], 1)


def test_propagation_deduplicates_and_adds_single_self_loop():
    edges = torch.tensor([[0, 1, 0, 0], [1, 0, 1, 0]])
    adj = M.normalized_adjacency(edges, 3).to_dense()
    torch.testing.assert_close(adj, torch.tensor([[.5, .5, 0.], [.5, .5, 0.], [0., 0., 1.]]))
    delta = {"x": torch.tensor([[1.], [3.], [5.]]), "edge_index": edges}
    torch.testing.assert_close(M.propagated_features(delta, (0., 1.), "cpu"),
                               torch.tensor([[2.], [2.], [5.]]))
    with pytest.raises(ValueError):
        M.normalized_adjacency(torch.tensor([[0], [3]]), 3)


def test_mmd_uses_actual_imbalanced_class_rows():
    real = torch.tensor([[1., 0.]] * 23 + [[0., 1.]] * 7)
    synth = torch.tensor([[1., 0.]] * 5 + [[0., 1.]] * 2)
    real_y = torch.tensor([2] * 23 + [8] * 7)
    synth_y = torch.tensor([2] * 5 + [8] * 2)
    assert M.class_mmd(real, synth, real_y, synth_y) == 0
    assert M.class_mmd(real, synth, real_y, torch.tensor([2] * 4 + [8] * 3)) > 0


def test_cat_is_optimized_synthetic_deterministic_and_source_local():
    torch.set_num_threads(2)
    delta = fixture()
    original = delta["x"].clone()
    cfg = M.Config(cat_steps=100, cat_hidden=32, cat_lr=.01)
    block, audit = M.cat_block(delta, 7, Encoder(), cfg, "cpu")
    repeat, _ = M.cat_block(delta, 7, Encoder(), cfg, "cpu")
    assert torch.equal(block["x"], repeat["x"])
    assert torch.equal(delta["x"], original)
    assert audit["x_update_norm"] > 0
    assert audit["fixed_bank_mmd_final"] < audit["fixed_bank_mmd_initial"]
    assert not block["edge_index"].numel()
    assert audit["row_classes"] == [2] * 5 + [8] * 2
    for cls, sources in zip(audit["row_classes"], audit["text_source_ids"]):
        assert all(int(delta["y"][s - 100]) == cls for s in sources)


@pytest.mark.parametrize("mode", ["tagsam_style", "graph_conditioned",
                                   "graph_text_aligned", "text_center",
                                   "random", "wrong_class"])
def test_tag_text_modes_are_deterministic_fixed_budget_and_source_local(mode):
    torch.set_num_threads(2)
    delta = fixture()
    cfg = M.Config(cat_steps=20, cat_hidden=16, cat_lr=.01,
                   text_mode=mode, text_sources=5, text_chars=80)
    block, audit = M.cat_block(delta, 7, Encoder(), cfg, "cpu")
    repeat, repeat_audit = M.cat_block(delta, 7, Encoder(), cfg, "cpu")
    assert torch.equal(block["x"], repeat["x"])
    assert block["texts"] == repeat["texts"]
    assert audit["text_mode"] == mode
    assert repeat_audit["text_mode"] == mode
    assert all(len(text) <= 80 for text in block["texts"])
    for cls, sources in zip(audit["row_classes"], audit["text_source_ids"]):
        if mode == "wrong_class":
            assert all(int(delta["y"][s - 100]) != cls for s in sources)
        else:
            assert all(int(delta["y"][s - 100]) == cls for s in sources)
    assert audit["text_selection"]


def test_gecc_outputs_propagated_centers_and_freezes_old_memory():
    torch.set_num_threads(2)
    delta = fixture()
    cfg = M.Config(gecc_restarts=2)
    block, audit = M.gecc_block(delta, 2, Encoder(), cfg, "cpu")
    propagated = M.propagated_features(delta, cfg.gecc_weights, "cpu")
    expected = torch.stack([propagated[delta["y"] == c].mean(0) for c in (2, 8)])
    torch.testing.assert_close(block["x"], expected)
    assert not torch.allclose(block["x"], torch.stack([delta["x"][delta["y"] == c].mean(0) for c in (2, 8)]))
    parent_x = block["x"].clone()
    grown = M.append_edge_free(block, block)
    assert torch.equal(grown["x"][:2], parent_x)
    assert len(grown["x"]) == 4
    assert audit["cumulative_raw_reclustering"] is False
    assert not grown["edge_index"].numel()
