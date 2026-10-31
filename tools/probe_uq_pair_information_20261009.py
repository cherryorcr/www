"""Training-free check for class-conditional graph/text pairing information."""
import json
from pathlib import Path
import sys

ROOT=Path('/home/pc/research/TAG_condensation')
sys.path[:0]=[str(ROOT/'tmp'),str(ROOT/'experiments/uq_gecc_gap_p1_v3_20261009/code'),
             str(ROOT/'code/v2_llm_fusion'),str(ROOT/'code/frozen_v1_2026_09_09')]
import torch
import torch.nn.functional as F
import run_unified_q_text_confirmation_20261008 as C
import unified_q_gap_diagnostics as D
import unified_q_pair_distribution as J

torch.set_num_threads(2)
records=[]
for dataset in ('wikics','sportsfit','bookhis'):
    order=9202 if dataset=='bookhis' else 9201
    folder=ROOT/f'results/uq_text_path_20261008/confirm_{dataset}_o{order}_c271108'
    p=json.loads((folder/'results.json').read_text())['protocol']
    data=C.L.load_large(dataset,'/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip',
        ROOT/f'results/public_baselines_text_cache_20261001/{dataset}','cuda:0',dict(groups=p['groups'],report_dev_mod=3))
    archives={a:C.load_memory(folder/f'final_{a}.pt.gz') for a in ('uq_dm_edgefree','cat_cctp','gecc_hard')}
    for stage,ids in enumerate(C.L.stage_ids(data)):
        delta=C.L.delta_view(data,ids,stage)
        message=C.M.propagated_features(delta,(0.,1.,0.),'cuda:0')
        texts=[C.U._append_text(t,320) for t in delta['texts']]
        text=F.normalize(torch.as_tensor(data['encoder'].encode(texts)).float().to('cuda:0'),dim=1)
        reference=J.target(message,text,delta['y'].to('cuda:0'),seed=271108+1009*stage+3141593)
        scores={}
        for arm,mem in archives.items():
            b=D.edgefree_slice(mem,p['budgets'][stage-1] if stage else 0,p['budgets'][stage])
            scores[arm]={mode:float(J.loss(b['x'].to('cuda:0'),b['text_emb'].to('cuda:0'),
                b['labels'].to('cuda:0'),reference,mode)) for mode in ('paired','marginal','shuffled')}
        records.append(dict(dataset=dataset,stage=stage,classes=len(reference['classes']),
            reference_pair_shuffle_mmd=reference['pair_shuffle_mmd'],archived_joint_error=scores))
    del data,archives
    torch.cuda.empty_cache()
result=dict(scope='training-free current-delta reference probe, no downstream score selection',
    kernel='product of two fixed normalized ReLU random-feature kernels',hidden=64,banks=2,
    threshold=1e-6,passes=sum(r['reference_pair_shuffle_mmd']>1e-6 for r in records),records=records)
out=ROOT/'results/uq_gecc_gap_20261009/pair_information_probe.json'
out.write_text(json.dumps(result,indent=2))
print(json.dumps(result))
