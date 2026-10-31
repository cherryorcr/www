import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from sklearn.metrics import accuracy_score, f1_score

from audit_p5_metric_scope import pooled_checks
from p5_summary_fields import complete_sum, optional_seconds
from small_comparison_metrics import score_predictions
import summarize_p5_longchain as longchain
import summarize_p5_rate_curve as rate


def legacy_score(target, pred, groups, stage):
    # Exercise the frozen function without loading encoders or training modules.
    source = Path(__file__).parents[1] / "frozen_v1_2026_09_09/run_gate1_continual.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == "eval_stagewise")
    namespace = dict(torch=torch, np=np, f1_score=f1_score,
                     accuracy_score=accuracy_score, STAGE_CLASSES=groups)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    logits = torch.nn.functional.one_hot(torch.tensor(pred), num_classes=max(max(g) for g in groups) + 1)
    model = SimpleNamespace(logits=lambda *args: logits)
    return namespace["eval_stagewise"](
        model, torch.zeros(len(target), 1), None, torch.zeros(1),
        torch.tensor(target), torch.ones(len(target), dtype=torch.bool), stage)


def test_task_slice_excludes_cross_task_false_positives():
    y, pred, groups = [0, 0, 1, 1], [0, 0, 0, 1], [[0], [1]]
    legacy, pooled = legacy_score(y, pred, groups, 1), score_predictions(y, pred, groups, 1)
    assert legacy["old_mean_f1"] == 100.0
    assert pooled["old_f1"] == 80.0
    assert legacy["cumulative_macro_f1"] == pytest.approx(pooled["cumulative_f1"])


def test_legacy_old_is_task_weighted_not_class_weighted():
    groups = [[0], [1, 2], [3]]
    y, pred = [0, 1, 2, 3], [0, 3, 3, 3]
    legacy, pooled = legacy_score(y, pred, groups, 2), score_predictions(y, pred, groups, 2)
    assert legacy["old_mean_f1"] == 50.0
    assert pooled["old_f1"] == pytest.approx(100 / 3)


def test_absent_target_class_changes_cumulative_definition():
    groups = [[0, 1]]
    legacy, pooled = legacy_score([0], [0], groups, 0), score_predictions([0], [0], groups, 0)
    assert legacy["cumulative_macro_f1"] == 100.0
    assert pooled["cumulative_f1"] == 50.0


@pytest.mark.parametrize("stage", [0, 1])
def test_pooled_aggregate_audit_detects_corruption(stage):
    groups = [[0], [1]]
    metric = json.loads(json.dumps(score_predictions([0, 0, 1, 1], [0, 0, 0, 1], groups, stage)))
    assert pooled_checks(metric, groups, stage)
    if stage == 0:
        assert metric["old_f1"] is None
    for field in ("old_f1", "new_f1", "cumulative_f1"):
        corrupted = copy.deepcopy(metric)
        corrupted[field] = (corrupted[field] or 0) + 1
        assert not pooled_checks(corrupted, groups, stage)
    corrupted = copy.deepcopy(metric)
    corrupted["per_task_f1"].append(100)
    assert not pooled_checks(corrupted, groups, stage)


def test_missing_cost_is_not_zero():
    assert optional_seconds({}, "time") is None
    assert optional_seconds({"time": None}, "time") is None
    assert optional_seconds({"time": 0}, "time") == 0.0
    assert complete_sum([1.0, None]) is None
    assert complete_sum([0.0, 1.0]) == 1.0


def test_reports_scope_missing_cost_and_actual_split(tmp_path):
    payload = dict(dataset="fixture", rate=.05, split_mode="frequency", n_stages=3,
                   stage_classes=[[0], [1], [2]], stage_train_counts=[20, 20, 20],
                   budgets=[1, 2, 3], student_batch=16, results=[])
    for stage in range(3):
        payload["results"].append(dict(stage=stage, roots=stage + 1, student_batch=16,
                                       student_seeds=[100], student_steps=50,
                                       student_train_seconds=1.0, cum_dev_f1_mean=50.0,
                                       old_mean_f1_mean=None if stage == 0 else 70.0,
                                       current_f1_mean=40.0))
    (tmp_path / "results.json").write_text(json.dumps(payload), encoding="utf-8")
    stages, runs = longchain.read_runs(tmp_path)
    rates = rate._rows(tmp_path)
    assert runs[0]["total_stage_build_seconds"] is None
    assert runs[0]["total_student_train_seconds"] == 3.0
    for row in [*stages, *runs, *rates]:
        assert row["old_new_metric_scope"] == "task_slice"
        assert row["text_embedding_seconds"] is None
    for row in rates:
        assert row["stage_build_seconds"] is None
    md = tmp_path / "longchain.md"
    longchain.write_md(md, stages, runs)
    text = md.read_text(encoding="utf-8")
    assert "| fixture | 3 | frequency |" in text
    assert "阶段划分为 `base_incremental`" not in text
    assert "task-slice F1" in text
    table = [line for line in text.splitlines() if line.startswith("|")]
    assert table[0].count("|") == table[1].count("|") == table[2].count("|")
    md = tmp_path / "rate.md"
    rate._write_markdown(md, rates)
    assert "task-slice F1" in md.read_text(encoding="utf-8")
    csv_path = tmp_path / "run.csv"
    longchain.write_csv(csv_path, runs)
    assert ",None," not in csv_path.read_text(encoding="utf-8")
