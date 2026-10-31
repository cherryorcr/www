from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_unified_q_revision_20261007 import audit_results


def _fixture():
    protocol = {"dataset": "toy", "budget_mode": "fixed_schedule", "order_seed": 1,
        "construction_seed": 1, "source_hashes": {}, "student": {"name": "toy"},
        "student_steps": 3, "student_batch": 2, "student_seeds": [1, 2],
        "pool_mask_hash": "pool", "report_mask_hash": "report", "groups": [[0]],
        "stage_id_hashes": ["stage"], "budgets": [2], "arms": {},
        "historical_raw_reads_by_constructor": 0, "boundary_enabled": False,
        "evaluation_split": "report-dev only, no test selection"}
    score = {"student_steps": 3, "student_batch": 2, "student_seeds": [1, 2],
        "final_f1": 50, "final_f1_std": 0, "accuracy": 50, "accuracy_std": 0,
        "new_f1": 50, "old_f1": None,
        "per_student": [{"seed": s, "final_f1": 50, "accuracy": 50,
                         "old_f1": None, "new_f1": 50} for s in (1, 2)]}
    row = {"arm": "cat_cctp", "stage": 0, "raw_rows": 10, "memory_rows": 2,
        "new_rows": 2, "checks": {"budget_ok": True, "prefix_ok": True, "finite_ok": True},
        "score": score}
    return {"protocol": protocol, "rows": [row], "status": "complete"}


def _write(root, result, name="cell"):
    folder = root / name
    folder.mkdir()
    (folder / "results.json").write_text(json.dumps(result), encoding="utf-8")
    (folder / "protocol.json").write_text(json.dumps(result["protocol"]), encoding="utf-8")


def test_valid_metric_aggregation(tmp_path):
    _write(tmp_path, _fixture())
    report = audit_results(tmp_path)
    assert report["integrity_passed"]
    assert report["complete_method_cells"] == 1


def test_detects_corrupted_mean(tmp_path):
    result = _fixture()
    result["rows"][0]["score"]["accuracy"] = 90
    _write(tmp_path, result)
    report = audit_results(tmp_path)
    assert not report["integrity_passed"]
    assert any(e["message"] == "accuracy aggregation error" for e in report["errors"])


def test_detects_duplicate_cells(tmp_path):
    _write(tmp_path, _fixture(), "a")
    _write(tmp_path, _fixture(), "b")
    report = audit_results(tmp_path)
    assert any(e["message"] == "duplicate experiment identity" for e in report["errors"])


def test_numerical_warning_is_not_hidden(tmp_path):
    result = _fixture()
    result["protocol"]["arms"] = {"uq_semantic": {}}
    row = deepcopy(result["rows"][0])
    row["arm"] = "uq_semantic"
    row["checks"]["propagation_max_error"] = 0
    row["audit"] = {"text_hard_forward_max_error": 0, "q_row_sum_max_error": .02,
        "historical_raw_reads": 0, "uses_boundary_sketch": False, "x_residual_ratio_max": .05,
        "config": {"trust_radius": .05, "steps": 1}, "loss_weighting": "group_class",
        "real_new_loss_mass": 1, "synthetic_new_loss_mass": 1, "q_update_count": 1,
        "residual_update_count": 1, "q_entropy": 1, "unique_text_ratio": 1,
        "trace": [{"q_gradient_norm": .01, "residual_gradient_norm": .01, "loss": .1}]}
    result["rows"].append(row)
    _write(tmp_path, result)
    report = audit_results(tmp_path)
    assert report["integrity_passed"]
    assert not report["strict_transport_tolerance_met"]
    assert report["numerical_warnings"][0]["observed"] == .02
