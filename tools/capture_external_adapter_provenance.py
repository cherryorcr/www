"""Record public experiment inputs, package versions and source digest only."""
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform


def evidence(path):
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": h.hexdigest()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    root = Path("/home/pc/research")
    encoder = root / "TAGSAM/graphclip_native_work/minilm_from_graphclip"
    files = [root / "GraphCLIP/processed_data" / f"{d}.pt" for d in ("cora", "citeseer")]
    files += [root / "TAG_condensation/results/p5_rate_curve_small_20260919" /
              f"{d}_r005/text_emb_pool_only.pt" for d in ("cora", "citeseer")]
    files += [v for v in sorted(encoder.iterdir()) if v.is_file()]
    value = {"python": platform.python_version(), "platform": platform.platform(),
             "versions": {name: version(name) for name in
                          ("torch", "transformers", "numpy", "scipy", "scikit-learn", "torch-geometric")},
             "inputs": [evidence(path) for path in files],
             "scope": "actual dataset, pool-only text cache and encoder files; no credentials/environment variables"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2), encoding="utf-8")
    print(json.dumps({"inputs": len(files), "versions": value["versions"]}))


if __name__ == "__main__":
    main()
