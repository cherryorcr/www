"""Stop only the two verified processes from the invalid precision attempt."""
import json
import os
from pathlib import Path
import signal

ROOT = Path('/home/pc/research/TAG_condensation/results/cotag_fidelity_temporal_20261021/precision_repair')
EXPECTED = {
    406759: ('/experiments/cotag_fidelity_temporal_20261021/batch.py', '--track', 'precision', '--attempt', 'v1'),
    407843: ('/experiments/cotag_fidelity_temporal_20261021/run_temporal_precise.py', '--seed', '271108'),
}

records = []
for pid, tokens in EXPECTED.items():
    path = Path(f'/proc/{pid}/cmdline')
    if not path.exists():
        records.append(dict(pid=pid, status='already_exited'))
        continue
    command = path.read_bytes().decode().replace('\x00', ' ')
    if not all(token in command for token in tokens):
        raise RuntimeError(f'Refusing to signal unrelated PID {pid}')
    os.kill(pid, signal.SIGTERM)
    records.append(dict(pid=pid, command=command, signal='SIGTERM'))
note = dict(status='invalid_precision_intervention',
    reason='Wrapper converted inputs to float64, but the frozen inner function coerced scores back to float32. These results do not test float64 arithmetic.',
    action='Exclude from numerical-repair comparisons; preserve results and logs.', processes=records)
path = ROOT/'INVALID_ATTEMPT.json'
if path.exists():
    raise FileExistsError(path)
path.write_text(json.dumps(note, indent=2))
print(json.dumps(note, indent=2))
