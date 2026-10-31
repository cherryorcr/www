"""Export reproducible evidence without copying multi-GB source archives."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import zipfile


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(root):
    project = root.parent.parent
    verification = json.loads((root / "analysis/verification_v1.json").read_text())
    assert verification["passed"]
    files = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if "raw" in relative.parts or path.name == "encoded.pt" or path.suffix == ".zip":
            continue
        if path.suffix in (".json", ".jsonl", ".csv", ".md", ".log", ".gz", ".png", ".pdf"):
            files[relative.as_posix()] = path
    here = Path(__file__).parent
    for path in sorted(here.iterdir()):
        if path.is_file() and path.suffix in (".py", ".md"):
            files["experiment_source/" + path.name] = path
    for name in ("common.py", "solver_precision.py", "temporal_protocol.py"):
        files["dependency_source/" + name] = project / "experiments/cotag_fidelity_temporal_20261021" / name
    # Include the frozen code directory, including indirectly imported helpers.
    frozen = project / "experiments/uq_final_evidence_20261012/code"
    for path in sorted(frozen.rglob("*.py")):
        files["frozen_source/" + path.relative_to(frozen).as_posix()] = path
    for run in verification["runs"]:
        for name, raw_path in run["source_paths"].items():
            path = Path(raw_path)
            files["recorded_source/" + name] = path
    if shutil.disk_usage(root).free < sum(p.stat().st_size for p in files.values()) + 8 * 2**30:
        raise RuntimeError("Export would breach the 8 GiB disk floor")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = root / f"evidence_{stamp}.zip"
    inventory = []
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
        for relative, path in files.items():
            archive.write(path, relative, compress_type=(zipfile.ZIP_STORED if path.suffix == ".gz" else zipfile.ZIP_DEFLATED))
            inventory.append(dict(path=relative, bytes=path.stat().st_size, sha256=sha(path)))
        archive.writestr("export_manifest.json", json.dumps(dict(exported_utc=stamp, files=inventory,
            note="Includes all 90 stage memories and prediction archives, selected source records/edges, logs, protocols and code. Raw source ZIPs, encoded.pt and pretrained weights remain on the server with verified hashes."), indent=2))
    print(json.dumps(dict(path=str(output), bytes=output.stat().st_size, files=len(files), sha256=sha(output))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    main(parser.parse_args().root)
