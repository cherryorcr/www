from pathlib import Path
import os
import torch

root = Path('/home/pc/research/TAG_condensation/results/direct_student_outer_20260928')
for data_path in (Path('/home/pc/research/GraphCLIP/processed_data/cora.pt'),
                   Path('/home/pc/research/GraphCLIP/processed_data/citeseer.pt')):
    if data_path.exists():
        data = torch.load(data_path, map_location='cpu', weights_only=False)
        print('DATA', data_path.name, type(data).__name__,
              (sorted(data.keys()) if isinstance(data, dict) else data.keys if hasattr(data, 'keys') else None))
for name, data_name, old in (('cora_canonical', 'cora.pt', 25),
                             ('cora_order19102', 'cora.pt', 25),
                             ('citeseer_canonical', 'citeseer.pt', 38),
                             ('citeseer_order19102', 'citeseer.pt', 38)):
    raw = torch.load(Path('/home/pc/research/GraphCLIP/processed_data') / data_name,
                     map_location='cpu', weights_only=False)
    support = torch.load(root / 'prep' / name / 'support.pt', map_location='cpu', weights_only=False)
    raw_x = raw.x.float()
    support_x = support['x'].float()[old:]
    counts = []
    for row in support_x:
        counts.append(int(((raw_x - row).abs().max(dim=1).values < 1e-6).sum()))
    print('MATCH', name, 'new=', len(support_x), 'exact=', sum(v == 1 for v in counts),
          'zero=', sum(v == 0 for v in counts), 'multi=', sum(v > 1 for v in counts))
for cache_path in sorted(Path('/home/pc/research/TAG_condensation/results').glob('v4_mainline_*/*/text_emb_pool_only.pt')):
    value = torch.load(cache_path, map_location='cpu', weights_only=False)
    print('CACHE', cache_path, type(value).__name__,
          (list(value.keys()) if isinstance(value, dict) else tuple(value.shape) if hasattr(value, 'shape') else None))
for path in sorted(root.glob('control_*/memory.pt')):
    memory = torch.load(path, map_location='cpu', weights_only=False)
    protocol_path = path.parent / 'protocol.json'
    import json
    protocol = json.loads(protocol_path.read_text())
    name = path.parent.name.replace('control_', '')
    slot = root / ('class_slot_map_' + name) / 'base_slot_class_id.pt'
    support = root / 'prep' / name / 'support.pt'
    prompt = root / ('prompt_support_v2_' + name) / 'support_class_id.pt'
    slot_values = torch.load(slot, map_location='cpu', weights_only=False) if slot.exists() else None
    class_values = torch.load(prompt, map_location='cpu', weights_only=False) if prompt.exists() else None
    print(name, 'rows=', len(memory['x']), 'edges=', len(memory['edge_index'][0]),
          'old=', protocol['old_rows'], 'slot_exists=', slot.exists(),
          'slot_len=', (len(slot_values) if slot_values is not None else None),
          'slot_new_unique=', (sorted(set(slot_values[protocol['old_rows']:].tolist())) if slot_values is not None else None),
          'class_at_old=', (sorted(set(class_values[protocol['old_rows']:].tolist())) if class_values is not None else None),
          'class_at_35=', (sorted(set(class_values[35:].tolist())) if class_values is not None and len(class_values)>35 else None),
          'support_exists=', support.exists(), 'class_exists=', prompt.exists())
