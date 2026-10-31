#!/usr/bin/env python3
"""Verify every file listed in PACKAGE_MANIFEST.json."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="package root; defaults to the parent of tools/",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    manifest_path = root / "PACKAGE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    missing: list[str] = []
    mismatched: list[dict[str, object]] = []
    for row in manifest["files"]:
        relative = row["path"]
        path = root / relative
        if not path.is_file():
            missing.append(relative)
            continue
        actual_size = path.stat().st_size
        actual_sha = sha256(path)
        if actual_size != row["bytes"] or actual_sha != row["sha256"]:
            mismatched.append(
                {
                    "path": relative,
                    "expected_bytes": row["bytes"],
                    "actual_bytes": actual_size,
                    "expected_sha256": row["sha256"],
                    "actual_sha256": actual_sha,
                }
            )

    result = {
        "status": "pass" if not missing and not mismatched else "fail",
        "listed_files": len(manifest["files"]),
        "missing": missing,
        "mismatched": mismatched,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

