"""Fast structural checks for the five-level small-graph runners."""
from __future__ import annotations

import torch
import pytest

import run_supervised_upper_bound_small as S
import run_tagsam_graphclip_static as T


def test_supervised_shape_and_finite():
    n, d, h, c = 7, 4, 5, 3
    model = S.SupervisedGCN(d, h, c)
    adj = torch.eye(n)
    logits = model.logits(torch.randn(n, d), adj)
    assert logits.shape == (n, c)
    assert torch.isfinite(logits).all()


def test_seen_prompt_projection():
    n, d, h, c = 4, 3, 4, 5
    model = S.SupervisedGCN(d, h, c)
    logits = model.logits(torch.randn(n, d), torch.eye(n))
    seen = [1, 3]
    pred = torch.tensor(seen)[logits[:, seen].argmax(1)]
    assert pred.shape == (n,)
    assert set(pred.tolist()).issubset(set(seen))


def toy_data():
    torch.manual_seed(7)
    n = 12
    x = torch.randn(n, 4)
    mask = lambda ids: torch.isin(torch.arange(n), torch.tensor(ids))
    return {"x": x, "x_device": x, "real_adj": torch.eye(n),
            "y": torch.tensor([0, 1, 2] * 4), "stage_classes": [[0], [1], [2]],
            "pool_mask": mask([0, 1, 2]), "selector_dev_mask": mask([3, 4, 5]),
            "report_dev_mask": mask([6, 7, 8]), "test_mask": mask([9, 10, 11])}


def test_supervised_rejects_holdout_and_future():
    data = toy_data()
    for ids in ([0, 3], [0, 1], [0, 0]):
        with pytest.raises(ValueError):
            S.validate_train_ids(data, torch.tensor(ids), 0)


def test_supervised_search_does_not_score_report(monkeypatch, tmp_path):
    data = toy_data()
    score = S.score_model
    splits = []
    def guarded(*args):
        splits.append(args[3])
        assert args[3] == "selector_dev_mask"
        return score(*args)
    monkeypatch.setattr(S, "score_model", guarded)
    checkpoint = tmp_path / "checkpoint.pt"
    ids = torch.tensor([0, 1, 2])
    row = S.fit(data, ids, 2, 7001, 3, "cpu", eval_every=1,
                checkpoint=checkpoint, evaluate_report=False)
    assert len(splits) == 3 and "metrics" not in row
    path = tmp_path / "result.json"
    S.C.save_json(path, row)
    assert S.load_cached_fit(path, checkpoint, ids) == row
    saved = torch.load(checkpoint, weights_only=False)
    saved["selected_state"]["g1.weight"][0, 0] += 1
    torch.save(saved, checkpoint)
    with pytest.raises(ValueError, match="corrupt"):
        S.load_cached_fit(path, checkpoint, ids)


def test_supervised_checkpoint_reproduces_metrics(tmp_path):
    data = toy_data()
    checkpoint = tmp_path / "checkpoint.pt"
    row = S.fit(data, torch.tensor([0, 1, 2]), 2, 100, 2, "cpu",
                eval_every=1, checkpoint=checkpoint)
    saved = torch.load(checkpoint, weights_only=False)
    model = S.SupervisedGCN(4, 384, 3)
    model.load_state_dict(saved["selected_state"])
    assert S.score_model(model, data, 2, "report_dev_mask", "cpu") == row["selected_metrics"]
    path = tmp_path / "result.json"
    S.C.save_json(path, row)
    assert S.load_cached_fit(path, checkpoint, torch.tensor([0, 1, 2]))["seed"] == 100


def test_static_cumulative_sources_and_budget():
    data = toy_data()
    stages, budgets = T.stage_inputs(data)
    assert [v.tolist() for v in stages] == [[0], [1], [2]]
    assert budgets == [0, 0, 0]
