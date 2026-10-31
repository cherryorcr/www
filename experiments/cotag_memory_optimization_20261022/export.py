"""Export this experiment only, with a checksum manifest and a disk floor."""
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

ROOT = Path("/home/pc/research/TAG_condensation")
RUNS = ROOT / "results/cotag_memory_optimization_20261022_v2"
REPORT = ROOT / "results/cotag_memory_optimization_20261022_final"
REPLAY = ROOT / "results/cotag_memory_optimization_20261022_replay"
EXPERIMENT = ROOT / "experiments/cotag_memory_optimization_20261022"
TARGET = ROOT / "results/cotag_memory_optimization_20261022_export.tar.gz"
if TARGET.exists():
    raise FileExistsError(TARGET)
assert json.loads((REPORT / "verification.json").read_text())["complete_streams"] == 12
assert json.loads((REPLAY / "results.json").read_text())["status"] == "complete"
if shutil.disk_usage(ROOT).free < 8 * 2**30:
    raise RuntimeError("8 GiB floor")
files = []
for folder in (RUNS, REPORT, REPLAY, EXPERIMENT, ROOT / "results/cotag_memory_optimization_20261022_v1"):
    files += [p for p in folder.rglob("*") if p.is_file() and "__pycache__" not in p.parts
              and ".pytest_cache" not in p.parts and not p.name.endswith(".tmp")]
manifest = []
for file in sorted(files):
    manifest.append(dict(path=str(file.relative_to(ROOT)), bytes=file.stat().st_size,
                         sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
manifest_path = REPORT / "export_manifest.json"
manifest_path.write_text(json.dumps(manifest, indent=2))
with tarfile.open(TARGET, "w:gz", compresslevel=1) as archive:
    for file in files + [manifest_path]:
        archive.add(file, arcname=str(file.relative_to(ROOT)), recursive=False)
print(json.dumps(dict(archive=str(TARGET), files=len(manifest), bytes=TARGET.stat().st_size,
                      sha256=hashlib.sha256(TARGET.read_bytes()).hexdigest())))
