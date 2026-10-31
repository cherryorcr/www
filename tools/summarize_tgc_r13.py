"""Summarize R13 with the generic sealed cell/seed accounting."""

import argparse
import json
from pathlib import Path

from summarize_tgc_r12 import summarize


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.root)
    path = args.root / "summary.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["interpretation"] = (
        "R13 is a sealed development gate study: selected_new64 is not promoted "
        "because its new-class and cross-order criteria are not satisfied."
    )
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")
