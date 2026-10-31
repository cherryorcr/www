"""Small tests for the frozen-checkpoint inference adapter."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.metrics import accuracy_score, f1_score

from eval_official_graphclip_report import ego_graph, graph_state, pack_graphs, report_nodes, state_hash
from small_comparison_metrics import score_logits


def test_report_nodes_excludes_training_selector_and_test():
    ids = torch.arange(40)
    data = {"report_dev_mask": ids % 10 == 3, "selector_dev_mask": ids % 10 == 1,
            "test_mask": ids % 10 == 2,
            "pool_mask": ~torch.isin(ids % 10, torch.tensor([1, 2, 3]))}
    assert report_nodes(data).tolist() == [3, 13, 23, 33]
    data["test_mask"][3] = True
    with pytest.raises(ValueError, match="overlaps"):
        report_nodes(data)


def test_graph_checkpoint_prefix_is_required_and_strict():
    with pytest.raises(ValueError, match="no graph_model"):
        graph_state({"weight": torch.zeros(2)})
    linear = torch.nn.Linear(2, 2)
    state = {f"graph_model.{k}": v.clone() for k, v in linear.state_dict().items()}
    state["text_model.unrelated"] = torch.zeros(1)
    linear.load_state_dict(graph_state(state), strict=True)
    del state["graph_model.bias"]
    with pytest.raises(RuntimeError, match="Missing key"):
        linear.load_state_dict(graph_state(state), strict=True)


def test_complete_two_hop_and_batched_centers():
    x = torch.arange(5).float().unsqueeze(1)
    edge = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4], [1, 0, 2, 1, 3, 2, 4, 3]])
    items = [ego_graph(x, edge, node) for node in (0, 4)]
    assert [g.num_nodes for g, _ in items] == [3, 3]
    batch, centers = pack_graphs(items, "cpu")
    assert batch.x[centers].flatten().tolist() == [0, 4]
    assert batch.pe.shape == (6, 32)
    assert torch.isfinite(batch.pe).all()


def test_final_metric_is_node_macro_f1_not_accuracy():
    target = [0, 0, 0, 1]
    logits = torch.tensor([[1., 0.]] * 4)
    result = score_logits(logits, target, [[0], [1]], 1)
    assert result["all"] == result["seen"]
    assert result["all"]["accuracy"] == 75
    assert result["all"]["cumulative_f1"] == pytest.approx(100 * 3 / 7)


def test_state_hash_detects_parameter_and_buffer_changes():
    model = torch.nn.BatchNorm1d(3).eval()
    original = state_hash(model)
    with torch.no_grad():
        model.weight[0] += 1
    changed = state_hash(model)
    assert changed != original
    model.running_mean[0] += 1
    assert state_hash(model) != changed


@pytest.mark.parametrize("dataset", ["cora", "citeseer"])
def test_saved_checkpoint_predictions_and_batch_invariance(dataset):
    root = Path(__file__).resolve().parents[2] / "results/official_graphclip_report_2026_09_17"
    main = root / f"{dataset}.json"
    repeated = root / "audit" / f"{dataset}_batch1.json"
    if not main.exists() or not repeated.exists():
        pytest.skip("inference artifacts not present")
    a, b = json.loads(main.read_text()), json.loads(repeated.read_text())
    p, q = a["predictions"], b["predictions"]
    assert p["node_ids"] == q["node_ids"]
    assert len(set(p["node_ids"])) == a["split_counts"]["report_dev_mask"]
    assert all(node % 10 == 3 for node in p["node_ids"])
    assert p["targets"] == q["targets"]
    assert p["predicted"] == np.asarray(p["logits"]).argmax(1).tolist() == q["predicted"]
    assert np.max(np.abs(np.asarray(p["logits"]) - np.asarray(q["logits"]))) < 1e-5
    assert a["metrics"] == b["metrics"]
    assert a["graph_state_sha256"] == b["graph_state_sha256"]
    assert a["text_state_sha256"] == b["text_state_sha256"]
    for row in (a, b):
        assert row["training_steps"] == 0 and row["model_state_unchanged"]
        assert row["graph_load_strict"] and row["text_load_strict"]
        assert not row["test_labels_used_for_metrics_or_selection"]
    metric = a["final"]["all"]
    assert metric["cumulative_f1"] == pytest.approx(
        f1_score(p["targets"], p["predicted"], average="macro", zero_division=0) * 100)
    assert metric["accuracy"] == pytest.approx(accuracy_score(p["targets"], p["predicted"]) * 100)
