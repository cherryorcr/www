"""Read-only inventory for baseline fidelity and publication-year experiments."""
import json
from pathlib import Path
import shutil
import subprocess

import torch

ROOT = Path('/home/pc/research/TAG_condensation')
DATA = Path('/home/pc/research/TAGSAM/data')


def main():
    out = dict(free_gib=shutil.disk_usage(ROOT).free / 2**30)
    out['gpu'] = subprocess.check_output([
        'nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu',
        '--format=csv'], text=True)
    datasets = {}
    for name in ('ogbn_arxiv_full_minilm.pt',
                 'ogbn_arxiv_full_temporal3_delta_t1.pt'):
        path = DATA / name
        value = torch.load(path, map_location='cpu', weights_only=False)
        detail = dict(keys=list(value), bytes=path.stat().st_size)
        for key in value:
            if 'year' in key or 'time' in key or key in ('original_node_ids', 'node_idx'):
                tensor = torch.as_tensor(value[key]).flatten()
                unique, counts = torch.unique(tensor, return_counts=True)
                detail[key] = dict(shape=list(tensor.shape), minimum=int(tensor.min()),
                                   maximum=int(tensor.max()),
                                   counts=dict(zip(unique.tolist(), counts.tolist()))
                                   if len(unique) < 150 else None)
        datasets[name] = detail
    out['datasets'] = datasets
    root = ROOT / 'results'
    out['budget_diagnostics'] = []
    for path in (root / 'cotag_q_four_gap_20261015_v1').rglob('results.json'):
        if 'tagsam' not in str(path):
            continue
        value = json.loads(path.read_text())
        out['budget_diagnostics'].append(dict(path=str(path), status=value.get('status'),
            selector_seconds=value.get('selector_seconds'), rows=[
                {k: row.get(k) for k in ('stage', 'outer_steps', 'response_seconds',
                 'response_peak_mb', 'score')} for row in value.get('rows', [])]))
    out['baseline_archives'] = []
    for path in (root / 'uq_final_evidence_20261012/baselines').glob('tagsam*'):
        result = path / 'results.json'
        if result.exists():
            value = json.loads(result.read_text())
            out['baseline_archives'].append(dict(path=str(path), status=value.get('status'),
                stages=len(value.get('rows', [])), files=[p.name for p in path.iterdir()],
                costs=[{k: r.get(k) for k in ('stage', 'construction_seconds',
                    'construction_peak_mb', 'student_seconds')} for r in value.get('rows', [])]))
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
