import pytest
import torch

import unified_q_pair_distribution as J


def test_joint_mmd_matches_explicit_product_features_and_gradient():
    torch.manual_seed(41)
    rx, rt = torch.randn(9,4),torch.randn(9,6)
    sy, ry = torch.tensor([0,0,1,1]),torch.tensor([0]*5+[1]*4)
    sx, st = torch.randn(4,4,requires_grad=True),torch.randn(4,6,requires_grad=True)
    reference=J.target(rx,rt,ry,hidden=5,banks=2,chunk=3,seed=19)
    expected=[]
    for wx,wt in zip(reference['wx'],reference['wt']):
        rf=J.features(rx,wx)[:,:,None]*J.features(rt,wt)[:,None,:]
        sf=J.features(sx,wx)[:,:,None]*J.features(st,wt)[:,None,:]
        expected.append(sum((rf[ry==c].mean(0)-sf[sy==c].mean(0)).square().sum() for c in (0,1)))
    expected=torch.stack(expected).mean()
    actual=J.loss(sx,st,sy,reference)
    torch.testing.assert_close(actual,expected)
    for left,right in zip(torch.autograd.grad(actual,(sx,st),retain_graph=True),
                          torch.autograd.grad(expected,(sx,st))):
        torch.testing.assert_close(left,right)


def test_shuffling_changes_only_joint_target_not_marginals_or_inputs():
    torch.manual_seed(24)
    x=torch.randn(40,8)
    text=x.clone()
    y=torch.tensor([0]*20+[1]*20)
    old=x.clone()
    rng=torch.get_rng_state().clone()
    target=J.target(x,text,y,seed=17,hidden=12,chunk=5)
    assert torch.equal(rng,torch.get_rng_state())
    torch.testing.assert_close(old,x,rtol=0,atol=0)
    assert target['pair_shuffle_mmd'] > 1e-4
    assert J.loss(x,text,y,target,'paired') < 1e-12
    assert J.loss(x,text,y,target,'marginal') < 1e-12
    assert J.loss(x,text,y,target,'shuffled') > 1e-4
    for bank,(wx,wt) in enumerate(zip(target['wx'],target['wt'])):
        for index,c in enumerate((0,1)):
            torch.testing.assert_close(target['marginal_x'][bank,index],J.features(x[y==c],wx).mean(0))
            torch.testing.assert_close(target['marginal_t'][bank,index],J.features(text[y==c],wt).mean(0))


def test_chunking_and_invalid_class_contract():
    torch.manual_seed(5)
    x,t,y=torch.randn(15,4),torch.randn(15,6),torch.tensor([0]*8+[1]*7)
    a=J.target(x,t,y,hidden=8,chunk=2,seed=42)
    b=J.target(x,t,y,hidden=8,chunk=500,seed=42)
    for key in ('paired','shuffled','marginal_x','marginal_t'):
        torch.testing.assert_close(a[key],b[key])
    with pytest.raises(ValueError,match='classes'):
        J.loss(x[:8],t[:8],y[:8],a)


def test_joint_modes_leave_initialization_and_payload_pool_unchanged():
    import unified_q_multimodal as U
    from dataclasses import replace
    from test_unified_q_distribution import config
    from test_unified_q_multimodal import DummyEncoder, _delta
    torch.manual_seed(15)
    delta,labels,prompt=_delta(),torch.tensor([0,0,1,1]),torch.randn(2,8)
    records=[]
    states=[]
    for mode in ('paired','marginal','shuffled'):
        cfg=config(steps=0,lambda_joint_distribution=1.,joint_distribution_mode=mode,
                   joint_distribution_hidden=8,joint_distribution_banks=2)
        block,audit=U.optimize_stage(delta,None,labels,DummyEncoder(),prompt,cfg,'cpu',
            use_boundary=False,use_topology=False,diagnostic_callback=lambda state:states.append(state))
        records.append((block,audit))
    for block,audit in records[1:]:
        torch.testing.assert_close(block['x'],records[0][0]['x'],rtol=0,atol=0)
        assert block['texts']==records[0][0]['texts']
        assert block['text_source_ids']==records[0][0]['text_source_ids']
        assert audit['q_cost']==records[0][1]['q_cost']
    for state in states[1:]:
        torch.testing.assert_close(state['q0'],states[0]['q0'],rtol=0,atol=0)


def test_joint_gradient_reaches_q_and_disabled_path_never_builds_target(monkeypatch):
    import unified_q_multimodal as U
    from test_unified_q_distribution import config
    from test_unified_q_multimodal import DummyEncoder, _delta
    torch.manual_seed(11)
    delta,labels,prompt=_delta(),torch.tensor([0,0,1,1]),torch.randn(2,8)
    cfg=config(steps=2,lambda_joint_distribution=1.,joint_distribution_hidden=8)
    _,audit=U.optimize_stage(delta,None,labels,DummyEncoder(),prompt,cfg,'cpu',
        use_boundary=False,use_topology=False)
    assert audit['joint_pair_distribution']['enabled']
    assert audit['trace'][0]['joint_distribution_q_gradient_norm']>0
    assert audit['text_hard_forward_max_error']<1e-4
    def fail(*args,**kwargs):
        raise AssertionError('disabled joint target was built')
    monkeypatch.setattr(J,'target',fail)
    _,audit=U.optimize_stage(delta,None,labels,DummyEncoder(),prompt,config(steps=1),'cpu',
        use_boundary=False,use_topology=False)
    assert not audit['joint_pair_distribution']['enabled']


def test_registered_pair_controls_do_not_include_failed_message_pooling():
    from dataclasses import asdict
    from run_unified_q_gecc_gap_20261009 import candidate_config, R
    base=R.q_config('uq_dm_edgefree',271108,30,2048)
    for phase in ('p2-paired','p2-marginal','p2-shuffled'):
        cfg,fields=candidate_config(base,phase)
        assert cfg.feature_pooling=='raw'
        assert cfg.lambda_joint_distribution==1.
        assert {k for k in asdict(base) if asdict(base)[k]!=asdict(cfg)[k]}==fields
        assert fields <= {'lambda_joint_distribution','joint_distribution_mode'}
