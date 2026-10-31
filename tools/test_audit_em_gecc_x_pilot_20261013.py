"""Protect artifact checks without importing or rerunning any constructor."""
from copy import deepcopy
from pathlib import Path

import pytest
import torch

import audit_em_gecc_x_pilot_20261013 as A


ROOT = Path(__file__).resolve().parents[1] / "results/em_gecc_x_pilot_20261013/run_v2"


@pytest.fixture
def final_cell():
    directory = ROOT / "cora/order19611"
    memory = torch.load(directory / "em_gecc_x_s2_memory.pt", weights_only=False)
    parent = torch.load(directory / "em_gecc_x_s1_memory.pt", weights_only=False)
    protocol = A.read_json(directory / "protocol.json")
    row = next(r for r in A.read_json(directory / "results.json")["rows"]
               if r["arm"] == "em_gecc_x" and r["stage"] == 2)
    return memory, parent, row, protocol


def test_all_completed_artifacts_pass_without_constructors():
    report = A.audit_run(ROOT)
    assert report["all_artifact_checks_passed"]
    assert report["memory_count"] == 30
    assert len(report["raw_artifact_hashes"]) == 35
    assert not report["raw_source_legality_audited"]
    assert all(d["uq_legacy_code"] and d["uq_dm_off"]
               for d in report["datasets"].values())
    em_cells = [c for c in report["cells"] if c["arm"].startswith("em_")]
    assert len(em_cells) == 18
    assert all(c["diagnostics"]["e_round_count"] == 4
               and c["diagnostics"]["e_round_hard_switch_count"] == 0
               and c["diagnostics"]["same_final_hard_sources_across_em_arms"]
               for c in em_cells)


@pytest.mark.parametrize("key,check", [
    ("x", "prefix_tensors"), ("text_emb", "prefix_tensors"),
    ("texts", "prefix_strings"), ("text_source_ids", "prefix_sources"),
])
def test_changed_old_prefix_is_detected(final_cell, key, check):
    memory, parent, row, protocol = final_cell
    changed = deepcopy(memory)
    if isinstance(changed[key], torch.Tensor):
        changed[key][0, 0] += 1
    elif key == "texts":
        changed[key][0] = "altered old text"
    else:
        changed[key][0] = [-999]
    checks = A.check_memory(changed, parent, row, protocol)
    assert not checks[check]
    if key in ("x", "text_emb", "texts"):
        assert not checks["sha256"]


def test_optional_support_provenance_is_checked(final_cell):
    memory, parent, row, protocol = deepcopy(final_cell)
    parent["support_ids"] = [[i] for i in range(len(parent["x"]))]
    memory["support_ids"] = parent["support_ids"] + [
        [i] for i in range(len(parent["x"]), len(memory["x"]))]
    assert A.check_memory(memory, parent, row, protocol)["prefix_sources"]
    memory["support_ids"][0] = [-999]
    assert not A.check_memory(memory, parent, row, protocol)["prefix_sources"]
    del memory["support_ids"]
    assert not A.check_memory(memory, parent, row, protocol)["prefix_sources"]


@pytest.mark.parametrize("centers,expected", [
    ([[4., 0.], [0., 9.]], 0.),
    ([[4., 0.], [-9., 0.]], -1.),
    ([[4., 0.], [9., 0.]], 1.),
    ([[0., 0.], [9., 0.]], 0.),
])
def test_class_center_cosine_is_normalized(centers, expected):
    value = A.class_center_cosine(torch.tensor(centers), torch.tensor([0, 1]))
    assert -1 <= value <= 1
    assert value == pytest.approx(expected)
    assert A.class_center_cosine(torch.ones(2, 2), torch.zeros(2)) is None


def test_hard_source_switches_use_global_ids():
    ids = torch.tensor([100, 103, 109])
    audit = {"e_trace": [{"selected_local_rows": [1, 0]}],
             "selected_text_ids": [103, 100]}
    assert A.final_hard_switches(audit, ids) == {
        "final_e_step_switch_count": 0, "final_e_step_slot_count": 2}
    audit["selected_text_ids"][1] = 109
    assert A.final_hard_switches(audit, ids)["final_e_step_switch_count"] == 1
    audit["selected_text_ids"] = [103]
    with pytest.raises(ValueError, match="source counts differ"):
        A.final_hard_switches(audit, ids)


def test_e_round_switches_track_repeated_changes():
    audit = {"e_trace": [
        {"selected_local_rows": values}
        for values in ([1, 0], [1, 0], [1, 2], [3, 0])]}
    assert A.hard_routing_trace(audit) == {
        "e_round_count": 4, "e_round_hard_switch_count": 3,
        "e_round_slot_comparisons": 6, "e_round_changed_slot_ratio": 1.0}
    audit["e_trace"][-1]["selected_local_rows"] = [3]
    with pytest.raises(ValueError, match="source counts differ"):
        A.hard_routing_trace(audit)
    with pytest.raises(ValueError, match="trace is empty"):
        A.hard_routing_trace({"e_trace": []})


def test_source_legality_checks_current_delta_and_class():
    ids, raw_labels = torch.tensor([1, 3]), torch.tensor([0, 0, 1, 1])
    memory = {"labels": torch.tensor([0, 1]), "text_source_ids": [[1], [3]]}
    assert all(A.source_checks(memory, 0, ids, raw_labels).values())
    memory["text_source_ids"] = [[0], [3]]
    assert not A.source_checks(memory, 0, ids, raw_labels)["source_ids_current_delta"]
    memory["text_source_ids"] = [[3], [1]]
    assert not A.source_checks(memory, 0, ids, raw_labels)["source_labels_match"]
    memory["text_source_ids"] = [[1]]
    assert not A.source_checks(memory, 0, ids, raw_labels)["source_row_count"]


def test_score_change_is_detected(final_cell):
    memory, parent, row, protocol = final_cell
    changed = deepcopy(row)
    changed["scores"]["cumulative_f1"] += 1
    assert not A.check_memory(memory, parent, changed, protocol)["scores_recomputed"]


def test_shared_student_steps_are_checked(monkeypatch):
    original = A.read_json

    def changed(path):
        result = original(path)
        if path.name == "results.json" and path.parent.parent.name == "cora":
            result["rows"][0]["scores"]["student_steps"] += 1
        return result

    monkeypatch.setattr(A, "read_json", changed)
    with pytest.raises(ValueError, match="Student training budgets differ"):
        A.audit_run(ROOT)
