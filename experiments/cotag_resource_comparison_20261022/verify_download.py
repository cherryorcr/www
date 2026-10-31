"""Verify exported bytes and independently regenerated analysis tables."""
import argparse
import hashlib
import json
from pathlib import Path


def verify(root, analysis):
    exported = root / "results/cotag_resource_comparison_20261022_final"
    manifest = json.loads((exported / "export_manifest.json").read_text())
    for row in manifest:
        path = (root / row["path"]).resolve()
        assert path.is_relative_to(root.resolve())
        assert path.stat().st_size == row["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
    tables = sorted(exported.glob("*.csv"))
    assert len(tables) >= 5
    for path in tables:
        assert path.read_bytes() == (analysis / path.name).read_bytes(), path.name
    expected = json.loads((exported / "verification.json").read_text())
    actual = json.loads((analysis / "verification.json").read_text())
    assert expected == actual
    receipt = dict(manifest_files=len(manifest), byte_identical_tables=[p.name for p in tables],
                   verification_identical=True)
    target = analysis / "download_verification.json"
    target.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    args = parser.parse_args()
    verify(args.root, args.analysis)
