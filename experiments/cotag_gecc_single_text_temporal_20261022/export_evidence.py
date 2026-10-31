"""Export audited text-policy evidence while retaining original comparisons."""
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
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main(root, analysis):
    project = root.parent.parent
    checked = read(analysis / "verification.json")
    assert checked["passed"] and len(checked["results"]) == 9
    assert all(row["passed"] and row["stages"] == 5 for row in checked["results"])
    files = {p.relative_to(root).as_posix(): p for p in sorted(root.rglob("*"))
             if p.is_file() and p.suffix in (".json", ".csv", ".md", ".log", ".gz")}
    here = Path(__file__).resolve().parent
    for path in sorted(here.iterdir()):
        if path.is_file() and path.suffix in (".py", ".md"):
            files["experiment_source/"+path.name] = path
    for folder in ("cotag_temporal_bibliographic_20261022", "cotag_fidelity_temporal_20261021"):
        for path in sorted((project / "experiments" / folder).iterdir()):
            if path.is_file() and path.suffix in (".py", ".md"):
                files[f"dependency_source/{folder}/{path.name}"] = path
    frozen = project / "experiments/uq_final_evidence_20261012/code"
    for path in sorted(frozen.rglob("*.py")):
        files["frozen_source/"+path.relative_to(frozen).as_posix()] = path
    receipts = read(analysis / "sources.json")
    for receipt in receipts["files"]:
        path = Path(receipt["path"])
        assert sha(path) == receipt["sha256"]
        if receipt["variant"] == "gecc_nearest1":
            continue
        p = read(path)["protocol"]
        name = f"reference_results/{p['dataset']}/{receipt['variant']}_c{p['construction_seed']}/"
        files[name+"results.json"] = path
        files[name+"protocol.json"] = path.with_name("protocol.json")
    protocol = read(root / "runs/acm/gecc_nearest1_c271006/protocol.json")
    for name, expected in protocol["source_hashes"].items():
        candidates = [project / "experiments/uq_final_evidence_20261012/code" / name,
                      project / "code/v2_llm_fusion" / name,
                      project / "experiments/cotag_fidelity_temporal_20261021" / name]
        matches = [path for path in candidates if path.is_file() and sha(path) == expected]
        assert matches, name
        files["recorded_source/"+name] = matches[0]
    assert len([name for name in files if name.startswith("runs/") and name.endswith("/memory.pt.gz")]) == 45
    assert len([name for name in files if name.startswith("runs/") and name.endswith("/predictions.pt.gz")]) == 45
    total = sum(path.stat().st_size for path in files.values())
    if shutil.disk_usage(root).free < total + 8 * 2**30:
        raise RuntimeError("Export would breach the 8 GiB free disk floor")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = root / f"evidence_{stamp}.zip"
    inventory = []
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
        for name, path in sorted(files.items()):
            archive.write(path, name, compress_type=(zipfile.ZIP_STORED if path.suffix == ".gz" else zipfile.ZIP_DEFLATED))
            inventory.append(dict(path=name, bytes=path.stat().st_size, sha256=sha(path)))
        archive.writestr("export_manifest.json", json.dumps(dict(exported_utc=stamp, files=inventory,
            note="All 45 new memories and prediction archives, protocols, logs, checked summaries, source code and unchanged reference score snapshots. Original reference memories and raw datasets remain in their previous evidence directories; encoders and raw dataset files are not duplicated."), indent=2))
    print(json.dumps(dict(path=str(destination), bytes=destination.stat().st_size,
                         files=len(files), sha256=sha(destination))), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    args = parser.parse_args()
    main(args.root, args.analysis)
