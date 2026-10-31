"""Focused checks for checkpoint selection and restored supervised diagnostics."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

import run_stad_supervised_sanity as S


class TinyGCN(torch.nn.Module):
    def __init__(self, nfeat, hidden, nclass, **kwargs):
        super().__init__()
        self.linear = torch.nn.Linear(nfeat, nclass)

    def forward(self, x, adj):
        return adj @ self.linear(x)


def fixture():
    x = torch.tensor([[1., 0.], [0., 1.], [.7, .7],
                      [1., .2], [.2, 1.], [.6, .7],
                      [1., .3], [.3, 1.], [.7, .6]])
    data = {"x": x, "x_device": x, "y": torch.tensor([0, 1, 2] * 3),
            "stage_classes": [[0], [1], [2]], "real_adj": torch.eye(9),
            "selector_dev_mask": torch.tensor([False] * 3 + [True] * 3 + [False] * 3),
            "report_dev_mask": torch.tensor([False] * 6 + [True] * 3)}
    support = {"x": x[:3], "edge_index": torch.empty(2, 0, dtype=torch.long)}
    return SimpleNamespace(GCN=TinyGCN), data, support, data["y"][:3]


def test_report_labels_do_not_change_selection_or_weights():
    models, data, support, labels = fixture()
    first, saved = S.fit(models, data, support, labels, 100, "cpu", steps=4)
    changed = deepcopy(data)
    changed["y"][changed["report_dev_mask"]] = (changed["y"][changed["report_dev_mask"]] + 1) % 3
    second, _ = S.fit(models, changed, support, labels, 100, "cpu", steps=4)
    for key in ("selected_steps", "selected_state_sha256", "end_state_sha256", "selection_curve"):
        assert first[key] == second[key]
    S.audit_fit(models, data, support, labels, first, saved, "cpu")


def test_audit_rejects_corrupt_weights():
    models, data, support, labels = fixture()
    row, saved = S.fit(models, data, support, labels, 100, "cpu", steps=4)
    saved["selected_state"]["linear.weight"][0, 0] += .1
    with pytest.raises(ValueError, match="corrupt checkpoint"):
        S.audit_fit(models, data, support, labels, row, saved, "cpu")


def test_test_split_evaluation_is_forbidden():
    models, data, _, _ = fixture()
    with pytest.raises(ValueError, match="test labels"):
        S.score(S.new_model(models, data, "cpu", 100), data, "test_mask", "cpu")
