#!/usr/bin/env python3
"""Build a deterministic SHA256 inventory for the curated package."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


EXCLUDED = {"PACKAGE_MANIFEST.json"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="package root; defaults to the parent of tools/",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    files = []
    for path in sorted(root.rglob("*"), key=lambda value: value.as_posix()):
        if not path.is_file() or ".git" in path.parts:
            continue
        relative = path.relative_to(root).as_posix()
        if relative in EXCLUDED:
            continue
        files.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )

    payload = {
        "schema": "curated-continual-tag-package-manifest-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": "D:/Adocs/简历/research/graph-condensation/TAGSAM",
        "files": files,
        "totals": {
            "files": len(files),
            "bytes": sum(row["bytes"] for row in files),
        },
    }
    (root / "PACKAGE_MANIFEST.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "written",
                "files": payload["totals"]["files"],
                "bytes": payload["totals"]["bytes"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

