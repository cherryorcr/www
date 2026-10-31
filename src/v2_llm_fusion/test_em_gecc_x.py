import torch
import torch.nn.functional as F

import em_gecc_x as E
import external_continual_mechanisms as C
import multimodal_gradient_matching as GM


class Encoder:
    def encode(self, texts):
        values = [[len(text), sum(map(ord, text)) % 37, 1.0,
                   sum(map(ord, text[::-1])) % 19] for text in texts]
        return F.normalize(torch.tensor(values, dtype=torch.float32), dim=1)


def delta(labels, start, stage, seed):
    gen = torch.Generator().manual_seed(seed)
    n = len(labels)
    return {
        "x": torch.randn(n, 6, generator=gen),
        "y": torch.tensor(labels, dtype=torch.long),
        "ids": torch.arange(start, start + n),
        "stage": stage,
        "texts": [f"class {label} current description {i}" for i, label in enumerate(labels)],
        "text_emb": F.normalize(torch.randn(n, 4, generator=gen), dim=1),
        "edge_index": torch.stack((torch.arange(n - 1), torch.arange(1, n))),
    }


def test_em_gecc_x_updates_x_selects_real_text_and_freezes_parent():
    torch.set_num_threads(1)
    encoder = Encoder()
    prompts = F.normalize(torch.randn(3, 4, generator=torch.Generator().manual_seed(12)), dim=1)
    cfg = E.Config(em_rounds=2, m_steps=1, proxy_hidden=8, proxy_out=4,
                   q_sinkhorn_steps=80, seed=23)

    first = delta([0] * 6 + [1] * 6, 100, 0, 31)
    init, init_audit = C.gecc_block(first, 4, encoder,
                                    C.Config(seed=11, gecc_restarts=2,
                                             text_chars=320), "cpu")
    first_labels = C.row_labels(C.ratio_allocation(first["y"], 4))
    first_memory, audit = E.optimize_stage(
        first, None, init["x"], init["text_emb"], first_labels,
        encoder, prompts, cfg, "cpu")
    assert audit["new_x_update_norm"] > 0
    assert first_memory["edge_index"].numel() == 0
    assert audit["slot_text_coverage"] == 1.0
    assert audit["q_row_sum_max_error"] < 1e-3
    for slot, (label, sources) in enumerate(zip(first_memory["labels"].tolist(),
                                                first_memory["text_source_ids"])):
        assert len(sources) == 1
        source = int(sources[0]) - int(first["ids"][0])
        assert int(first["y"][source]) == label
        assert first_memory["texts"][slot] == GM._append_text(first["texts"][source], 320)
    assert init_audit["native_topology"] == "self-loop only"

    second = delta([2] * 10, 200, 1, 32)
    next_init, _ = C.gecc_block(second, 2, encoder,
                                C.Config(seed=11, gecc_restarts=2), "cpu")
    second_labels = C.row_labels(C.ratio_allocation(second["y"], 2))
    before = {key: first_memory[key].clone() for key in
              ("x", "text_emb", "labels", "mass", "depth", "stage")}
    grown, second_audit = E.optimize_stage(
        second, first_memory, next_init["x"], next_init["text_emb"],
        second_labels, encoder, prompts, cfg, "cpu")
    for key, value in before.items():
        assert torch.equal(grown[key][:len(value)], value)
    assert grown["texts"][:len(before["x"])] == first_memory["texts"]
    assert second_audit["old_x_update_norm"] == 0.0
    assert second_audit["historical_raw_reads"] == 0
    assert grown["edge_index"].numel() == 0


def test_q_is_class_masked_and_capacity_balanced():
    scores = torch.tensor([[.8, .1, 7.], [.6, .2, 4.],
                           [8., .1, .7], [5., .2, .6]])
    q = E._sinkhorn(scores, torch.tensor([0, 0, 1, 1]),
                    torch.tensor([0, 0, 1]), .2, 100)
    assert torch.equal(q[:2, 2], torch.zeros(2))
    assert torch.equal(q[2:, :2], torch.zeros(2, 2))
    torch.testing.assert_close(q.sum(1), torch.ones(4), atol=1e-3, rtol=0)
    torch.testing.assert_close(q.sum(0), torch.tensor([1., 1., 2.]),
                               atol=1e-3, rtol=0)
