import importlib.util
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parent))
spec=importlib.util.spec_from_file_location("constructor_summary",Path(__file__).with_name("summarize_unified_q_constructor_attribution_20261008.py"))
S=importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)


def cell(order):
    arms=list(S.CORE)+(["no_text_init","no_payload_gm","stop_text_gradient"] if order==9101 else [])
    p=dict(name="unified-q-constructor-attribution-20261008",dataset="wikics",order_seed=order,
        budgets=[4,6,8,10,12],arms=arms,student={},student_seeds=[150,151,152],
        student_steps=2048,student_batch=24,pool_mask_hash="p",report_mask_hash="r",
        construction_seed=271006,q_config={},source_hashes={},runner_sha256="runner")
    rows=[]
    for arm in p["arms"]:
        for stage in range(5):
            for readout in (["native","common_cat_payload"] if stage==4 else ["native"]):
                value=70+(3 if arm==S.MAIN else 0)
                score=dict(final_f1=value,accuracy=value+5,old_f1=value,new_f1=value,
                    final_f1_std=.2,accuracy_std=.3,per_student=[dict(seed=s,final_f1=value,accuracy=value+5) for s in p["student_seeds"]])
                rows.append(dict(arm=arm,stage=stage,readout=readout,score=score,
                    checks=dict(budget_ok=True,prefix_ok=True,finite_ok=True,fixed_content_exact=True),
                    x_hash=arm,text_hash="common" if readout=="common_cat_payload" else arm,
                    construction_seconds=1,student_seconds=2,memory_rows=p["budgets"][stage],
                    audit=dict(unchanged_main_replay=dict(relative_x_difference=.001,text_source_rows_changed=1))))
    return dict(protocol=p,rows=rows,status="complete")


def write(root,c):
    path=root/f"wikics_o{c['protocol']['order_seed']}"/"results.json"
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(c))


def test_paired_intervals_use_orders_not_student_seeds(tmp_path):
    for order in (9102,9103):
        write(tmp_path,cell(order))
    cells,incomplete=S.load(tmp_path)
    s=S.summary(cells)
    assert len(incomplete)==7
    assert s["contrasts"][0]["n"]==2
    assert s["contrasts"][0]["mean"]==-3
    assert s["contrasts"][0]["ci95"]==[-3,-3]
    assert s["contrasts"][0]["accuracy_paired"]["n"]==2
    assert s["contrasts"][0]["accuracy_paired"]["ci95"]==[-3,-3]
    assert len(s["per_order"])==20


def test_single_order_has_no_paired_ci_or_fake_extra_samples():
    s=S.summary([cell(9102)])
    assert all(r["ci95"] is None and r["accuracy_paired"]["ci95"] is None for r in s["contrasts"])
    assert len(s["per_order"])==10
    assert s["main"][0]["worst_stage_f1"]==73
    assert s["main"][0]["worst_order_f1"]==73
    assert "Mean worst-stage F1" in S.render(s,[])


def test_common_payload_must_be_identical(tmp_path):
    c=cell(9102)
    next(r for r in c["rows"] if r["readout"]=="common_cat_payload")["text_hash"]="different"
    write(tmp_path,c)
    with pytest.raises(AssertionError,match="payload differs"):
        S.load(tmp_path)


def test_complete_flag_does_not_allow_missing_records(tmp_path):
    c=cell(9102)
    c["rows"].pop()
    write(tmp_path,c)
    with pytest.raises(AssertionError,match="partial"):
        S.load(tmp_path)


def test_changed_features_in_payload_readout_rejected(tmp_path):
    c=cell(9102)
    next(r for r in c["rows"] if r["readout"]=="common_cat_payload")["x_hash"]="changed"
    write(tmp_path,c)
    with pytest.raises(AssertionError,match="changed X"):
        S.load(tmp_path)


def test_reference_factorial_signs_and_interaction():
    c=cell(9101)
    values={S.MAIN:76,"no_text_init":71,"no_payload_gm":72,"no_node_text":70}
    for row in c["rows"]:
        if row["arm"] in values:
            v=values[row["arm"]]
            row["score"].update(final_f1=v,accuracy=v+5)
            for seed in row["score"]["per_student"]:
                seed.update(final_f1=v,accuracy=v+5)
    effects={v["effect"]:v for v in S.summary([c])["factorial_effects"]
             if v["readout"]=="native" and v["metric"]=="final_f1"}
    assert effects["init_with_pair"]["mean"]==5
    assert effects["init_without_pair"]["mean"]==2
    assert effects["pair_with_init"]["mean"]==4
    assert effects["pair_without_init"]["mean"]==1
    assert effects["init_pair_interaction"]["mean"]==3
    assert effects["init_pair_interaction"]["paired_student_sd"]==0
