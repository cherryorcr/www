"""Export only the new experiment; reused streams already have a verified bundle."""
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

ROOT = Path("/home/pc/research/TAG_condensation")
RUNS = ROOT / "results/cotag_resource_comparison_20261022_v1"
REPORT = ROOT / "results/cotag_resource_comparison_20261022_final"
SOURCE = ROOT / "experiments/cotag_resource_comparison_20261022"
TARGET = ROOT / "results/cotag_resource_comparison_20261022_export.tar.gz"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if TARGET.exists():
        raise FileExistsError(TARGET)
    verification = json.loads((REPORT / "verification.json").read_text())
    assert len(verification["statuses"]) == 24
    assert all(r["status"] in ("complete", "failed") for r in verification["statuses"])
    files = []
    for folder in (RUNS, REPORT, SOURCE):
        files += [p for p in folder.rglob("*") if p.is_file() and "__pycache__" not in p.parts
                  and ".pytest_cache" not in p.parts and not p.name.endswith(".tmp")]
    required = sum(p.stat().st_size for p in files)
    if shutil.disk_usage(ROOT).free - required < 8 * 2**30:
        raise RuntimeError("8 GiB export disk floor")
    manifest = [dict(path=p.relative_to(ROOT).as_posix(), bytes=p.stat().st_size,
                     sha256=digest(p)) for p in sorted(files)]
    manifest_path = REPORT / "export_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    with tarfile.open(TARGET, "w:gz", compresslevel=1) as archive:
        for path in files + [manifest_path]:
            archive.add(path, arcname=path.relative_to(ROOT).as_posix(), recursive=False)
    print(json.dumps(dict(archive=str(TARGET), files=len(manifest), bytes=TARGET.stat().st_size,
                         sha256=digest(TARGET))))


if __name__ == "__main__":
    main()
