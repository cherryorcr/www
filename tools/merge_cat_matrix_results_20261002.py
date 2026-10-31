"""Merge per-dataset formal CaT payload runs into one matrix root."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    protocols = [read(p / "protocol.json") for p in args.inputs]
    base = protocols[0]
    for p in protocols[1:]:
        for key in ("name", "orders", "arms", "report_seeds", "config",
                    "student", "memory", "construction_access", "topology",
                    "evaluation", "report_driven_selection"):
            if p[key] != base[key]:
                raise ValueError(f"protocol mismatch: {key}")
    rows = []
    for root in args.inputs:
        rows.extend(read(root / "results.json")["rows"])
    expected = {(d, o, a, s) for d in ("cora", "citeseer")
                for o in base["orders"] for a in base["arms"] for s in range(3)}
    actual = {(r["dataset"], r["order"], r["arm"], int(r["stage"])) for r in rows}
    if actual != expected:
        raise ValueError(f"coverage mismatch: missing={sorted(expected - actual)}")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "protocol.json").write_text(
        json.dumps({**base, "datasets": ["cora", "citeseer"]}, indent=2),
        encoding="utf-8")
    (args.output / "results.json").write_text(
        json.dumps({"protocol": {**base, "datasets": ["cora", "citeseer"]},
                    "rows": rows}, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "output": str(args.output)}))


if __name__ == "__main__":
    main()
