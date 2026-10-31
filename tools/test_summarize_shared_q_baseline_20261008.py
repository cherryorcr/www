import copy

import pytest

import summarize_shared_q_baseline_20261008 as S


def metric(v):
    rows = [dict(seed=seed, final_f1=v, accuracy=v+2, old_f1=v-1, new_f1=v+1) for seed in (150,151,152)]
    return dict(final_f1=v,accuracy=v+2,old_f1=v-1,new_f1=v+1,
                final_f1_std=0,accuracy_std=0,per_student=rows)


def shared_cell(order=9101, delta=1):
    protocol = dict(name="shared-Q-vs-independent-Q-20261008",max_stages=5,
        dataset="wikics",order_seed=order,arms=[S.MAIN,"split_q"],
        student_steps=2048,student_seeds=[150,151,152],budgets=[2,4,6,8,10],q_config=dict(trust_radius=.05))
    rows=[]
    for arm in protocol["arms"]:
        for stage,readout in [(s,"native") for s in range(5)]+[(4,"common_cat_payload")]:
            v=60+stage+(delta if arm==S.MAIN else 0)
            rows.append(dict(arm=arm,stage=stage,readout=readout,score=metric(v),memory_rows=2*(stage+1),
                checks=dict(budget_ok=True,prefix_ok=True,finite_ok=True,fixed_content_exact=True),
                common_payload_hash="identical",construction_seconds=1,student_seconds=2,
                audit=dict(assignment_checks=dict(sinkhorn_feasible=True),historical_raw_reads=0,
                    x_residual_ratio_max=.05,payload_validation=dict(current_delta_text_sources=True,
                    same_class_text_sources=True,text_reencode_max_error=0),
                    assignments={k:dict(cross_class_mass=0,row_error=0,column_relative_error=0) for k in ("qx","qt")},
                    qx_qt_row_l1_mean=.5,qx_qt_top1_disagreement=.3,qx_changed_norm=1,qt_changed_norm=2,assignment_parameter_count=20)))
    return dict(protocol=protocol,rows=rows,status="complete")


def handoff_cell():
    protocol=dict(name="handoff-order-two-Student-paired-20261008",max_stages=5,
        dataset="wikics",arms=[S.MAIN,"random_real"],student_steps=2048,
        student_seeds=[150,151,152],budgets=[2,4,6,8,10])
    rows=[]
    for arm in protocol["arms"]:
        for stage in range(5):
            for student in ("legacy","prompt"):
                score=metric(60+(student=="prompt")*10+(arm==S.MAIN)*2)
                score["student_batch"]=16 if student=="legacy" else 24
                rows.append(dict(arm=arm,stage=stage,student=student,score=score,memory_rows=2*(stage+1),
                    checks=dict(budget_ok=True,prefix_ok=True,finite_ok=True),
                    **{key:arm+str(stage) for key in ("memory_hash","x_hash","text_hash","edge_hash","labels_hash")},
                    construction_seconds=1,student_seconds=2,text_chars_max=320,text_chars_total=1000,nonself_edges=0))
    return dict(protocol=protocol,rows=rows,status="complete")


def test_shared_pairing_uses_orders_not_student_pseudoreplicates():
    cells=[shared_cell(o,d) for o,d in zip((9101,9102,9103),(1,2,3))]
    for c in cells:
        S.audit_shared(c)
    s=S.compute(cells,[])
    effect=next(r for r in s["shared_effects"] if r["readout"]=="native" and r["metric"]=="final_f1")
    assert effect["n"]==3 and effect["mean"]==2 and effect["positive"]==3
    assert len(s["shared_per_order"])==12


def test_duplicate_or_incomplete_chain_rejected():
    for change in (lambda r:r.pop(),lambda r:r.append(copy.deepcopy(r[0]))):
        c=shared_cell()
        change(c["rows"])
        with pytest.raises(AssertionError,match="partial or duplicate"):
            S.audit_shared(c)


def test_common_payload_must_be_identical():
    c=shared_cell()
    c["rows"][-1]["common_payload_hash"]="different"
    with pytest.raises(AssertionError,match="common payload"):
        S.audit_shared(c)


def test_both_q_constraints_are_checked():
    c=shared_cell()
    c["rows"][-1]["audit"]["assignments"]["qt"]["cross_class_mass"]=.01
    with pytest.raises(AssertionError,match="independent assignment"):
        S.audit_shared(c)


def test_same_memory_required_across_students():
    c=handoff_cell()
    S.audit_handoff(c)
    c["rows"][0]["text_hash"]="changed"
    with pytest.raises(AssertionError,match="crossover changed memory"):
        S.audit_handoff(c)


def test_student_effect_not_counted_as_method_gain():
    c=handoff_cell()
    s=S.compute([], [c])
    assert all(r["mean"]==2 for r in s["baseline_paired"])
    assert all(r["f1_prompt_minus_legacy"]==10 for r in s["student_transfer"])


def test_pooled_scores_must_match_seed_scores():
    c=shared_cell()
    c["rows"][0]["score"]["accuracy"]+=1
    with pytest.raises(AssertionError,match="pooled score"):
        S.audit_shared(c)


def repair_pair():
    original = handoff_cell()
    original["protocol"].update(dataset="sportsfit", groups=[[0], [1], [2], [3], [4]],
        pool_mask_hash="pool", report_mask_hash="report", stage_id_hashes=list(range(5)),
        sources={k: k for k in ("graphtext_prompt_student.py", "run_gate1_continual.py", "external_continual_mechanisms.py")})
    original["protocol"]["arms"] = [S.MAIN, "cat_mmd_tag_cctp_top5"]
    for row in original["rows"]:
        if row["arm"] == "random_real":
            row["arm"] = "cat_mmd_tag_cctp_top5"
            row["provenance"] = dict(hash="source" + str(row["stage"]))
    historical = S.mark_historical_cctp(original)
    repair = dict(protocol=copy.deepcopy(historical["protocol"]),
        rows=copy.deepcopy([r for r in historical["rows"] if r["arm"] == "cat_text_center_legacy"]), status="complete")
    repair["protocol"].update(name="handoff-cctp-fixed-X-text-repair-20261008", arms=["cat_cctp_repaired"])
    for row in repair["rows"]:
        row.update(arm="cat_cctp_repaired", text_hash="repaired" + str(row["stage"]),
            memory_hash="repaired" + str(row["stage"]), provenance=dict(source_memory_hash="source" + str(row["stage"])))
        row["checks"]["fixed_content_exact"] = True
    return original, historical, repair


def test_text_repair_preserves_original_and_separates_historical_control():
    original, historical, repair = repair_pair()
    merged = S.merge_text_repair(historical, repair)
    assert "cat_mmd_tag_cctp_top5" in original["protocol"]["arms"]
    assert "cat_cctp_repaired" not in historical["protocol"]["arms"]
    scores = S.compute([], [merged])["baseline_scores"]
    assert all(r["category"] == "historical_control" for r in scores if r["arm"] == "cat_text_center_legacy")
    assert len([r for r in scores if r["arm"] == "cat_cctp_repaired"]) == 2


@pytest.mark.parametrize("change", ["groups", "x_hash", "source_hash", "module"])
def test_text_repair_rejects_nontext_protocol_or_memory_change(change):
    _, original, repair = repair_pair()
    if change == "groups":
        repair["protocol"]["groups"][0] = [99]
    elif change == "x_hash":
        for row in repair["rows"]:
            row["x_hash"] = "different"
    elif change == "source_hash":
        repair["rows"][0]["provenance"]["source_memory_hash"] = "different"
    else:
        repair["protocol"]["sources"]["run_gate1_continual.py"] = "different"
    with pytest.raises(AssertionError):
        S.merge_text_repair(original, repair)
