"""Distinguish actual source changes from encoder rounding in a replay."""
import json
from pathlib import Path
import sys
ROOT = Path('/home/pc/research/TAG_condensation')
sys.path[:0] = [str(ROOT / 'experiments/uq_gecc_gap_p0_v2_20261009/code'),
               str(ROOT / 'code/v2_llm_fusion'), str(ROOT / 'code/frozen_v1_2026_09_09')]
import torch
import run_unified_q_text_confirmation_20261008 as C
import unified_q_gap_diagnostics as D
torch.set_num_threads(2)
folder = ROOT / 'results/uq_text_path_20261008/confirm_wikics_o9201_c271108'
p = json.loads((folder / 'results.json').read_text())['protocol']
data = C.L.load_large('wikics', '/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip',
    ROOT / 'results/public_baselines_text_cache_20261001/wikics', 'cuda:0', dict(groups=p['groups'], report_dev_mod=3))
mem = C.load_memory(folder / 'final_gecc_hard.pt.gz')
records = []
for stage, ids in enumerate(C.L.stage_ids(data)):
    delta = C.L.delta_view(data, ids, stage)
    b = D.edgefree_slice(mem, p['budgets'][stage-1] if stage else 0, p['budgets'][stage])
    features = C.M.propagated_features(delta, C.M.Config().gecc_weights, 'cuda:0').cpu()
    v, a = C.A.text_control(delta, features, b, data['encoder'], 'nearest', 5, 271108)
    records.append(dict(stage=stage, texts_equal=b['texts']==v['texts'],
        sources_equal=b['text_source_ids']==v['text_source_ids'],
        embedding_error=float((b['text_emb']-v['text_emb']).abs().max())))
print(json.dumps(records))
(ROOT / 'results/uq_gecc_gap_20261009/gecc_text_replay_probe.json').write_text(json.dumps(records, indent=2))
