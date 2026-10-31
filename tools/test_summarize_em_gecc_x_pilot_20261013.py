"""Prevent pilot summaries from masquerading as official or sealed evidence."""
from copy import deepcopy
from pathlib import Path

import pytest

import summarize_em_gecc_x_pilot_20261013 as S


PARENT = Path(__file__).resolve().parents[1] / "results/em_gecc_x_pilot_20261013"
ROOT = PARENT / "run_v2"


def test_scores_and_qualified_labels_are_preserved():
    rows = S.load_rows(ROOT)
    assert len(rows) == 10
    assert {r["split"] for r in rows} == {"report_dev_only"}
    assert all(r["checks_ok"] for r in rows)
    gecc = next(r for r in rows if r["dataset"] == "cora" and r["method"] == "gecc_hard")
    assert gecc["f1"] == pytest.approx(82.11, abs=.005)
    assert "mechanism adaptation" in gecc["method_label"]
    uq = next(r for r in rows if r["dataset"] == "citeseer" and r["method"] == "unified_q_ef")
    assert uq["f1"] == pytest.approx(68.68, abs=.005)
    assert "DM off / legacy code" in uq["method_label"]
    assert "not frozen paper main" in uq["method_label"]


def test_gate_is_recomputed_from_paired_scores():
    decision = S.decision_summary(ROOT)
    assert decision["source_decision_matches_recomputation"]
    assert not decision["confirmatory_evidence"]
    assert not any(decision["checks"].values())
    assert decision["decision"] == "stop_em_and_continue_unified_q_ef"
    assert decision["metrics"]["mean_dataset_f1_gain_pp"] == pytest.approx(-12.17, abs=.005)


def test_full_audit_is_tied_to_raw_file_fingerprints():
    audit = S.read_json(PARENT / "artifact_audit.json")
    assert audit["raw_source_legality_audited"]
    S.validate_audit(ROOT, audit)
    assert sum(c["diagnostics"].get("final_e_step_switch_count", 0)
               for c in audit["cells"]) == 0


@pytest.mark.parametrize("field", ["raw_artifact_hashes", "all_artifact_checks_passed"])
def test_stale_or_failed_audit_is_rejected(field):
    audit = deepcopy(S.read_json(PARENT / "artifact_audit.json"))
    if field == "raw_artifact_hashes":
        audit[field]["decision.json"] = "wrong"
    else:
        audit[field] = False
    with pytest.raises(ValueError, match="audit"):
        S.validate_audit(ROOT, audit)


def test_raw_run_cannot_be_used_as_output_directory():
    S.check_output_directory(ROOT, PARENT)
    for output in (ROOT, ROOT / "derived"):
        with pytest.raises(ValueError, match="immutable raw run"):
            S.check_output_directory(ROOT, output)


def test_report_retains_negative_result_and_audit_corrections():
    audit = S.read_json(PARENT / "artifact_audit.json")
    report = S.markdown(S.load_rows(ROOT), S.decision_summary(ROOT), audit)
    assert "M-steps used selected hard text embeddings" in report
    assert "source-given-slot entropy" in report
    assert "unnormalized dot product" in report
    assert "not causal proof" in report
    assert "neither establish that main method's superiority" in report
    assert "Final E-step hard-source switches: 0" in report
    assert "Across-round hard-source switches: 0" in report
    assert "unchanged hard routing in 18/18" in report
    assert "in 6/6 dataset/stage cells" in report
    assert "does not imply that soft Q was constant" in report
    assert len(S.diagnostic_rows(audit)) == 6


def test_missing_cell_is_rejected(monkeypatch):
    original = S.read_json

    def changed(path):
        result = original(path)
        if path.name == "results.json":
            result["rows"].pop()
        return result

    monkeypatch.setattr(S, "read_json", changed)
    with pytest.raises(ValueError, match="missing or duplicate cells"):
        S.load_rows(ROOT)
