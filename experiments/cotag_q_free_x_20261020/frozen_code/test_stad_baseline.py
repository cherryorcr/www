"""CPU checks for STAD canonical transfer and the frozen unweighted path."""
import json
from types import SimpleNamespace

import pytest
import torch

import run_stad_graphclip_static as B
import summarize_external_tag_comparison as Q


def test_metric_audit_accepts_only_json_key_conversion():
    fresh = {"per_class_f1": {0: 50.0, 1: 70.0}, "old_f1": None}
    saved = json.loads(json.dumps(fresh))
    assert B.normalized_metrics(fresh) == B.normalized_metrics(saved)
    saved["per_class_f1"]["0"] = 40.0
    assert B.normalized_metrics(fresh) != B.normalized_metrics(saved)


def test_unweighted_path_is_bitwise_unchanged():
    memory = {"x": torch.zeros(4, 3),
              "edge_index": torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])}
    expected = B.G.norm_adj(memory["edge_index"], 4, "cpu").coalesce()
    actual = B.C.training_adjacency(memory, "cpu")
    assert torch.equal(actual.indices(), expected.indices())
    assert torch.equal(actual.values(), expected.values())


def test_weighted_path_never_re_normalizes_or_binarizes():
    memory = {"x": torch.zeros(2, 3), "adj_normalized": True,
              "edge_index": torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]]),
              "adj_values": torch.tensor([.8, .2, .2, .8])}
    actual = B.C.training_adjacency(memory, "cpu").to_dense()
    assert torch.equal(actual, torch.tensor([[.8, .2], [.2, .8]]))
    for field, value in [("adj_normalized", False), ("adj_values", torch.tensor([-1.]*4)),
                         ("adj_values", torch.tensor([float("nan")]*4))]:
        with pytest.raises(ValueError):
            B.C.training_adjacency({**memory, field: value}, "cpu")


def test_sources_exclude_holdout_future_and_duplicates():
    data = {"pool_mask": torch.tensor([True, True, False, True]),
            "y": torch.tensor([0, 1, 0, 2]), "stage_classes": [[0], [1], [2]]}
    B.validate_sources(data, torch.tensor([0]), 0)
    for ids in ([0, 0], [0, 2], [0, 1]):
        with pytest.raises(ValueError):
            B.validate_sources(data, torch.tensor(ids), 0)


def test_llm_fail_closed_and_cached(monkeypatch, tmp_path):
    calls = []
    def post(*args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(raise_for_status=lambda: None,
            json=lambda: {"choices": [{"index": i, "finish_reason": "stop",
                "message": {"content": "[[LABEL]] A [[SUMMARY]] A synthetic description."}}
                for i in range(3)], "usage": {"total_tokens": 10}, "model": "local"})
    monkeypatch.setattr(B.requests, "post", post)
    request = {"model": "local", "seed": 1, "n": 3}
    cache = tmp_path / "generation.json"
    first = B.llm_candidates(request, cache, "http://localhost/v1")
    assert B.llm_candidates(request, cache, "http://localhost/v1") == first
    assert len(calls) == 1
    with pytest.raises(ValueError, match="stale"):
        B.llm_candidates({**request, "seed": 2}, cache, "http://localhost/v1")
    monkeypatch.setattr(B.requests, "post", lambda *a, **k: SimpleNamespace(
        raise_for_status=lambda: None, json=lambda: {"choices": []}))
    with pytest.raises(ValueError, match="incomplete"):
        B.llm_candidates(request, tmp_path / "bad.json", "http://localhost/v1")
    assert not (tmp_path / "bad.json").exists()


def test_weighted_values_are_part_of_support_hash():
    memory = {"x": torch.zeros(2, 3), "adj_values": torch.ones(4)}
    assert B.C.tensor_hash(memory) != B.C.tensor_hash({**memory, "adj_values": torch.zeros(4)})


def test_retention_uses_ratio_of_means():
    rows = []
    for dataset in ("cora", "citeseer"):
        for arm, mean, count in [("v4", 50., 5), ("raw_long_selected", 60., 100)]:
            rows.append({"dataset": dataset, "arm": arm, "final_f1": mean,
                         "rows": count, "final_per_seed": [mean] * 6})
    result = Q.add_reference(rows)
    assert result[0]["retention_percent"] == pytest.approx(100 * 50 / 60)
    assert result[0]["node_ratio_percent"] == 5
    assert result[0]["delta_vs_full_pp"] == -10


def test_summary_rejects_missing_stage():
    with pytest.raises(ValueError, match="stage coverage"):
        Q.collect("cora", "v4", [])
