import pytest
import torch

from prepare_handoff_replay_manifest_20261008 import exact_sources, replay_sources, H
from repair_handoff_cctp_text_20261008 import new_fixed_block


def sample():
    delta = dict(ids=torch.tensor([10, 20, 30]), x=torch.tensor([[1., 0.], [0., 1.], [1., 1.]]),
        text_emb=torch.eye(3), texts=["a", "b", "c"], y=torch.tensor([0, 1, 1]))
    take = torch.tensor([2, 0])
    memory = dict(x=delta["x"][take], text_emb=delta["text_emb"][take], texts=["c", "a"])
    return delta, memory


def test_exact_recovery_preserves_stored_order():
    delta, memory = sample()
    sources, audit = exact_sources(memory, delta, 2)
    assert sources == [30, 10]
    assert audit["historical_raw_reads"] == 0


@pytest.mark.parametrize("key", ["x", "text_emb", "texts"])
def test_mismatched_modality_is_rejected(key):
    delta, memory = sample()
    if key == "texts":
        memory[key][0] = "wrong"
    else:
        memory[key][0, 0] += .01
    with pytest.raises(AssertionError, match="no exact"):
        exact_sources(memory, delta, 2)


def test_ambiguous_duplicate_labels_are_rejected():
    delta, memory = sample()
    delta["x"][1] = delta["x"][0]
    delta["text_emb"][1] = delta["text_emb"][0]
    delta["texts"][1] = delta["texts"][0]
    with pytest.raises(AssertionError, match="ambiguous"):
        exact_sources(memory, delta, 2)


def test_identical_contents_with_same_class_are_explicitly_audited():
    delta, memory = sample()
    delta["x"][1] = delta["x"][0]
    delta["text_emb"][1] = delta["text_emb"][0]
    delta["texts"][1] = delta["texts"][0]
    delta["y"][1] = delta["y"][0]
    _, audit = exact_sources(memory, delta, 2)
    assert audit["duplicate_contents_same_label"] == 1


def test_selector_replay_requires_exact_content_and_declared_seed(monkeypatch):
    delta, memory = sample()
    delta["stage"] = 0
    memory.update(mass=torch.ones(2), depth=torch.zeros(2, dtype=torch.long), stage=torch.zeros(2, dtype=torch.long))
    block = {k: v.clone() if isinstance(v, torch.Tensor) else list(v) for k, v in memory.items()}
    audit = dict(selection_seed=123, selected_source_ids=[2, 0])
    monkeypatch.setattr(H.L.U, "build_real_selection_block", lambda *a: (block, audit))
    cfg = dict(method="random_real", selection_seed=123, approximate=False)
    ids, _ = replay_sources(memory, delta, 2, cfg, "cpu")
    assert ids == [30, 10]
    with pytest.raises(AssertionError, match="seed differs"):
        replay_sources(memory, delta, 2, dict(cfg, selection_seed=124), "cpu")
    block["x"][0, 0] += .01
    with pytest.raises(AssertionError, match="archived x"):
        replay_sources(memory, delta, 2, cfg, "cpu")


def test_fixed_x_repair_cannot_change_features_or_topology():
    _, memory = sample()
    memory.update(mass=torch.ones(2), depth=torch.zeros(2, dtype=torch.long),
        stage=torch.tensor([0, 1]), edge_index=torch.empty(2, 0, dtype=torch.long))
    block = new_fixed_block(memory, torch.tensor([1, 0]), 1)
    assert torch.equal(block["x"], memory["x"][1:])
    assert block["labels"].tolist() == [0]
    block["x"][0, 0] += 1
    assert not torch.equal(block["x"], memory["x"][1:])
    memory["edge_index"] = torch.tensor([[0], [1]])
    with pytest.raises(AssertionError, match="edge-free"):
        new_fixed_block(memory, torch.tensor([1, 0]), 1)
