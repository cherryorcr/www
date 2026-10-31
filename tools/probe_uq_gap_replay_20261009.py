"""Inspect a failed archive replay without changing the experiment."""
import json
from pathlib import Path
import sys

ROOT = Path('/home/pc/research/TAG_condensation')
sys.path[:0] = [str(ROOT / 'experiments/uq_gecc_gap_p0_20261009/code'),
               str(ROOT / 'code/v2_llm_fusion'), str(ROOT / 'code/frozen_v1_2026_09_09')]
import torch
import run_unified_q_text_confirmation_20261008 as C
import unified_q_gap_diagnostics as D

torch.set_num_threads(2)
source_dir = ROOT / 'results/uq_text_path_20261008/confirm_wikics_o9201_c271108'
source = json.loads((source_dir / 'results.json').read_text())
p = source['protocol']
data = C.L.load_large('wikics', '/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip',
    ROOT / 'results/public_baselines_text_cache_20261001/wikics', 'cuda:0', dict(groups=p['groups'], report_dev_mod=3))
delta = C.L.delta_view(data, C.L.stage_ids(data)[0], 0)
memory = C.load_memory(source_dir / 'final_uq_dm_edgefree.pt.gz')
original = D.edgefree_slice(memory, 0, p['budgets'][0])
records = []
for observed in (False, True):
    measurements = []
    block, audit = C.U.optimize_stage(delta, None, original['labels'], data['encoder'], data['prompt_emb'],
        C.R.q_config('uq_dm_edgefree', 271108, 30, 2048), 'cuda:0', use_boundary=False, use_topology=False,
        diagnostic_callback=(lambda s: measurements.append(D.representation_diagnostic(s))) if observed else None)
    old_audit = next(v['audit'] for v in source['stages'] if v['arm'] == 'uq_dm_edgefree' and v['stage'] == 0)
    records.append(dict(observed=observed, x_max_error=float((block['x'] - original['x']).abs().max()),
        x_relative_error=float((block['x'] - original['x']).norm() / original['x'].norm()),
        text_changed=sum(a != b for a, b in zip(block['texts'], original['texts'])),
        initial_prototypes_equal=audit['q_cost']['prototype_source_ids'] == old_audit['q_cost']['prototype_source_ids'],
        old_trace=old_audit['trace'], new_trace=audit['trace'], diag=measurements))
print(json.dumps(records), flush=True)
(ROOT / 'results/uq_gecc_gap_20261009/replay_probe.json').write_text(json.dumps(records, indent=2))
