"""Reconstruct each compact sidecar and verify frozen base archive and scores."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def audit(root, code):
    sys.path.insert(0, str(code))
    import torch
    import unified_q_pairing_readout as D
    import run_unified_q_gecc_gap_20261009 as G
    checked = []
    for path in sorted(root.glob('*/results.json')):
        result = json.loads(path.read_text())
        if result['status'] != 'complete':
            raise ValueError('incomplete cell')
        source_path = Path(result['protocol']['source']) / 'results.json'
        if hashlib.sha256(source_path.read_bytes()).hexdigest() != result['protocol']['source_results_sha256']:
            raise AssertionError('source results changed')
        source = json.loads(source_path.read_text())
        for arm in ('uq_dm_edgefree', 'cat_cctp', 'gecc_hard'):
            controls = [c for c in result['controls'] if c['arm'] == arm]
            if len(controls) != 2:
                raise AssertionError('missing permutation')
            archive = Path(controls[0]['base_archive'])
            raw_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
            memory = G.C.load_memory(archive)
            for control in controls:
                sidecar = path.parent / f"{arm}_p{control['permutation_seed']}.json"
                info = json.loads(sidecar.read_text())
                if {k:v for k,v in info.items() if k != 'permutation'} != control:
                    raise AssertionError('sidecar differs from results')
                perm = torch.tensor(info['permutation'], dtype=torch.long)
                if not torch.equal(perm, D.within_stage_class_permutation(memory, info['permutation_seed'])):
                    raise AssertionError('wrong permutation seed')
                if raw_sha != info['base_archive_sha256'] or G.V.memory_hash(memory) != info['base_memory_hash']:
                    raise AssertionError('base archive mismatch')
                variant = D.permute_payload(memory, perm)
                digest = G.V.memory_hash(variant)
                if digest != info['memory_hash'] or G.V.hash_tensor(perm) != info['permutation_sha256']:
                    raise AssertionError('reconstruction hash mismatch')
                if hashlib.sha256(json.dumps(variant['text_source_ids']).encode()).hexdigest() != info['source_ids_sha256']:
                    raise AssertionError('provenance mismatch')
                for key, value in info['fixed_hashes'].items():
                    if G.V.hash_tensor(variant[key]) != value:
                        raise AssertionError('fixed field mismatch: ' + key)
                rows = [r for r in result['rows'] if r['arm'] == arm and r['permutation_seed'] == info['permutation_seed']]
                if len(rows) != 2 or any(r['memory_hash'] != digest or r['reused'] for r in rows):
                    raise AssertionError('readout hash mismatch')
                checked.append(dict(cell=path.parent.name, arm=arm, permutation_seed=info['permutation_seed'],
                    memory_hash=digest, sidecar_sha256=hashlib.sha256(sidecar.read_bytes()).hexdigest()))
            for row in (r for r in result['rows'] if r['arm'] == arm and r['permutation_seed'] is None):
                original = next(r for r in source['rows'] if r['arm'] == arm and r['student'] == row['student'])
                if row['score'] != original['score'] or row['memory_hash'] != G.V.memory_hash(memory):
                    raise AssertionError('reused native result mismatch')
    if len(checked) != 36:
        raise AssertionError('expected 36 permutation sidecars')
    out = dict(status='pass', sidecars=len(checked), records=checked)
    (root / 'archive_audit.json').write_text(json.dumps(out, indent=2))
    print(json.dumps(dict(status='pass', sidecars=len(checked))))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--code', type=Path, required=True)
    args = p.parse_args()
    audit(args.root, args.code)
