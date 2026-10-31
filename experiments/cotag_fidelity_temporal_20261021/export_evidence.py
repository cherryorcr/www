"""Export compact source evidence, leaving tensor archives on the server."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile


def run(root):
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    target = root/f'evidence_{stamp}.zip'
    files = [(path, path.relative_to(root).as_posix())
             for path in sorted(root.rglob('*')) if path.is_file()
             and path.suffix in ('.json', '.csv', '.md', '.log')]
    files += [(path, 'experiment_source/'+path.name)
              for path in sorted(Path(__file__).parent.iterdir())
              if path.is_file() and path.suffix in ('.py', '.md')]
    inventory = []
    with zipfile.ZipFile(target, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path, relative in files:
            raw = path.read_bytes()
            archive.writestr(relative, raw)
            inventory.append(dict(path=relative, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()))
        archive.writestr('export_manifest.json', json.dumps(dict(exported_utc=stamp, files=inventory,
            note='Tensor checkpoints, data and model weights retained remotely. Includes logs, protocols, metrics, audits and experiment entrypoint source; frozen project dependencies are identified by recorded hashes.'), indent=2))
    print(json.dumps(dict(path=str(target), bytes=target.stat().st_size, files=len(inventory),
                         sha256=hashlib.sha256(target.read_bytes()).hexdigest())))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('/home/pc/research/TAG_condensation/results/cotag_fidelity_temporal_20261021'))
    run(parser.parse_args().root)
