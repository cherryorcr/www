import importlib.util
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location("attribution_summary", Path(__file__).with_name("summarize_unified_q_attribution_20261008.py"))
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)


def cell(order=9101, status="complete"):
    p = dict(name="unified-q-frozen-attribution-20261008", dataset="wikics",
        budgets=[4,6], arms=list(S.PRIMARY), order_seed=order,
        pool_mask_hash="pool", report_mask_hash="report", student={}, student_seeds=[150,151,152],
        student_steps=2048, student_batch=24, construction_seed=271006,
        q_config={}, source_hashes={}, runner_sha256="runner")
    rows = []
    for stage in range(2):
        for arm in S.PRIMARY:
            value = 70 + (3 if arm == "uq_dm_edgefree" else 2 if arm == "adaptive_coherent" else 0)
            score = dict(final_f1=value, accuracy=value+5, old_f1=value-2,
                         new_f1=value+2, final_f1_std=.2, accuracy_std=.3,
                         per_student=[dict(seed=s,final_f1=value) for s in (150,151,152)])
            rows.append(dict(arm=arm, stage=stage, student_family="prompt", score=score,
                x_hash="cat" if arm != "uq_dm_edgefree" else "q", edge_hash="empty", memory_hash=arm,
                checks=dict(budget_ok=True,prefix_ok=True,finite_ok=True,fixed_content_exact=True),
                audit=dict(payload_validation=dict(current_delta_text_sources=True,same_class_text_sources=True))))
    return dict(protocol=p, rows=rows, status=status)


def write(root, c):
    path = root/f"wikics_o{c['protocol']['order_seed']}"/"results.json"
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(c),encoding="utf-8")


def test_intervals_count_orders_not_student_repeats(tmp_path):
    for order in (9101,9102,9103):
        write(tmp_path,cell(order))
    cells, _ = S.load_cells(tmp_path)
    result = S.summarize(cells)
    assert result["main"][0]["orders"] == 3
    comparison = next(r for r in result["paired"] if r["comparison"] == "uq_dm_edgefree - cat_cctp")
    assert comparison["n"] == 3 and comparison["mean"] == 3


def test_incomplete_cells_are_not_scored(tmp_path):
    write(tmp_path,cell(status="running"))
    cells, incomplete = S.load_cells(tmp_path)
    assert not cells and len(incomplete)==1


def test_fixed_x_violation_is_rejected(tmp_path):
    c = cell()
    next(r for r in c["rows"] if r["arm"] == "adaptive_coherent")["x_hash"] = "different"
    write(tmp_path,c)
    with pytest.raises(AssertionError,match="not fixed"):
        S.load_cells(tmp_path)


def test_student_replay_cannot_change_memory(tmp_path):
    c = cell()
    old = dict(c["rows"][-1],student_family="legacy_loss_matched",memory_hash="changed")
    c["rows"].append(old)
    write(tmp_path,c)
    with pytest.raises(AssertionError,match="different memory"):
        S.load_cells(tmp_path)


def test_duplicate_stages_cannot_count_as_completed(tmp_path):
    c = cell()
    c["rows"].append(c["rows"][0])
    write(tmp_path,c)
    with pytest.raises(AssertionError,match="duplicate"):
        S.load_cells(tmp_path)


def test_stage_means_and_minima_keep_order_as_unit():
    cells = [cell(9102), cell(9103)]
    trajectories = ((50, 90), (60, 80))
    for c, values in zip(cells, trajectories):
        for row in c["rows"]:
            if row["arm"] == "uq_dm_edgefree":
                row["score"]["final_f1"] = values[row["stage"]]
                row["score"]["accuracy"] = values[row["stage"]] + 5
    main = next(r for r in S.summarize(cells)["main"] if r["arm"] == "uq_dm_edgefree")
    assert main["stage_f1"]["mean"] == 70
    assert main["stage_f1"]["sd"] == 0
    assert main["stage_f1"]["n"] == 2
    assert main["stage_accuracy"]["mean"] == 75
    assert main["worst_stage_f1"]["mean"] == 55
    assert main["worst_observed_stage_f1"] == 50
    assert main["final_memory_rows"] == 6
