from dataclasses import asdict, replace

import pytest
import torch

import unified_q_constructor_attribution as C
import unified_q_multimodal as U
from test_unified_q_multimodal import DummyEncoder, _delta
from test_unified_q_distribution import config
import run_unified_q_constructor_attribution_20261008 as R


LABELS = torch.tensor([0, 0, 0, 1, 1, 1])


@pytest.mark.parametrize("arm,keys", [
    (C.MAIN,set()), ("no_node_text",{"text_weight","pair_weight"}),
    ("no_text_init",{"text_weight"}), ("no_payload_gm",{"pair_weight"}),
    ("no_residual",{"train_residual"}), ("no_distribution",{"lambda_distribution"}),
    ("stop_text_gradient",{"text_assignment"}), ("free_x",set())])
def test_controls_change_only_declared_fields(arm,keys):
    cfg = config()
    a,b = asdict(cfg),asdict(C.control_config(cfg,arm))
    assert {k for k in a if a[k] != b[k]} == keys


def test_no_node_payload_is_invariant_to_strings_and_embeddings():
    torch.manual_seed(124)
    delta,prompt = _delta(),torch.randn(2,8)
    other = dict(delta,texts=[f"unrelated content {i*i+42}" for i in range(12)],
                 text_emb=torch.randn_like(delta["text_emb"])*20)
    cfg = config(steps=3)
    a,aa = C.optimize_control(delta,None,LABELS,DummyEncoder(),prompt,cfg,"cpu","no_node_text")
    b,bb = C.optimize_control(other,None,LABELS,DummyEncoder(),prompt,cfg,"cpu","no_node_text")
    torch.testing.assert_close(a["x"],b["x"],rtol=0,atol=1e-7)
    assert aa["q_cost"]["prototype_source_ids"] == bb["q_cost"]["prototype_source_ids"]
    assert a["support_ids"] == b["support_ids"]
    assert not torch.equal(a["text_emb"],b["text_emb"])
    assert aa["config"]["lambda_gradient"] == 1


def test_free_x_has_same_initial_state_and_objective():
    torch.manual_seed(53)
    delta,prompt = _delta(),torch.randn(2,8)
    cfg = config(sinkhorn_max_steps=2048)
    a,aa = C.optimize_control(delta,None,LABELS,DummyEncoder(),prompt,replace(cfg,steps=0),"cpu",C.MAIN)
    b,bb = C.optimize_control(delta,None,LABELS,DummyEncoder(),prompt,replace(cfg,steps=0),"cpu","free_x")
    for key in ("x","text_emb","labels","mass"):
        torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
    _,aa = C.optimize_control(delta,None,LABELS,DummyEncoder(),prompt,replace(cfg,steps=1),"cpu",C.MAIN)
    _,bb = C.optimize_control(delta,None,LABELS,DummyEncoder(),prompt,replace(cfg,steps=1),"cpu","free_x")
    for key in ("loss","gradient_match","distribution_match","propagation","q_anchor"):
        assert aa["trace"][0][key] == pytest.approx(bb["trace"][0][key],abs=1e-7)


def test_free_x_keeps_bound_prefix_and_hard_text():
    torch.manual_seed(52)
    delta,prompt = _delta(),torch.randn(2,8)
    parent,_ = C.optimize_control(delta,None,LABELS,DummyEncoder(),prompt,config(steps=0),"cpu",C.MAIN)
    old = parent["x"].clone()
    block,audit = C.optimize_control(_delta(1,12),parent,LABELS,DummyEncoder(),prompt,config(),"cpu","free_x")
    memory = U.append_memory(parent,block)
    assert torch.equal(memory["x"][:len(old)],old)
    assert audit["x_residual_ratio_max"] <= config().trust_radius + 1e-6
    assert audit["trace"][0]["free_x_gradient_norm"] > 0
    assert audit["text_hard_forward_max_error"] < 1e-6
    assert not block["edge_index"].numel()


def test_projection_is_exact_original_feasible_set():
    torch.manual_seed(4)
    pool,offset = torch.randn(7,5),torch.randn(7,5)
    cap = .05 * pool.norm(dim=1,keepdim=True).clamp_min(1e-3)
    expected = pool + offset * (cap/offset.norm(dim=1,keepdim=True).clamp_min(1e-12)).clamp(max=1.)
    torch.testing.assert_close(C.project_free_x(pool+offset,pool,.05),expected)


def test_no_residual_and_no_distribution_are_real_interventions():
    torch.manual_seed(7)
    delta,prompt = _delta(),torch.randn(2,8)
    for arm in ("no_residual","no_distribution"):
        _,audit = C.optimize_control(delta,None,LABELS,DummyEncoder(),prompt,config(),"cpu",arm)
        if arm == "no_residual":
            assert audit["residual_update_count"] == audit["x_residual_norm"] == 0
        else:
            assert not audit["distribution_matching"]["enabled"]


def test_common_payload_replacement_does_not_change_features():
    torch.manual_seed(12)
    delta,prompt = _delta(),torch.randn(2,8)
    memory,_ = C.optimize_control(delta,None,LABELS,DummyEncoder(),prompt,config(steps=0),"cpu",C.MAIN)
    common = dict(memory,text_emb=torch.zeros_like(memory["text_emb"]),texts=["common"]*6)
    result = R.fixed_payload(memory,common)
    assert torch.equal(result["x"],memory["x"])
    assert torch.equal(result["text_emb"],common["text_emb"])
    with pytest.raises(AssertionError,match="slot labels"):
        R.fixed_payload(memory,dict(common,labels=common["labels"].flip(0)))


def test_common_cat_prefix_needs_no_q_support_ids():
    torch.manual_seed(42)
    block,_ = C.optimize_control(_delta(),None,LABELS,DummyEncoder(),torch.randn(2,8),config(steps=0),"cpu",C.MAIN)
    cat = {k:v for k,v in block.items() if k != "support_ids"}
    common = None
    for _ in range(5):
        common = R.A.append_block(common,cat)
    assert len(common["x"]) == len(common["text_source_ids"]) == 30
    assert "support_ids" not in common


def test_no_node_payload_also_ignores_old_prefix_text():
    torch.manual_seed(64)
    delta,prompt = _delta(),torch.randn(2,8)
    parent,_ = C.optimize_control(delta,None,LABELS,DummyEncoder(),prompt,config(steps=0),"cpu",C.MAIN)
    changed = dict(parent,text_emb=torch.randn_like(parent["text_emb"]),texts=["changed"]*len(LABELS))
    new_delta = _delta(1,12)
    a,_ = C.optimize_control(new_delta,parent,LABELS,DummyEncoder(),prompt,config(steps=2),"cpu","no_node_text")
    b,_ = C.optimize_control(new_delta,changed,LABELS,DummyEncoder(),prompt,config(steps=2),"cpu","no_node_text")
    torch.testing.assert_close(a["x"],b["x"],rtol=0,atol=1e-7)
    assert a["text_source_ids"] == b["text_source_ids"]
