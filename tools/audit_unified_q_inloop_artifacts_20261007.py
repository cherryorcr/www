"""Reload final memories and compare them with the recorded construction artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import run_unified_q_revision_20261007 as V


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--expected-cells", type=int, required=True)
    args = p.parse_args()
    records = []
    for path in sorted(args.root.rglob("results.json")):
        result = json.loads(path.read_text())
        protocol = json.loads(path.with_name("protocol.json").read_text())
        if result["status"] != "complete" or protocol != result["protocol"]:
            raise ValueError(f"incomplete or mismatched protocol: {path}")
        for arm in protocol["arms"]:
            rows = sorted((r for r in result["rows"] if r["arm"] == arm), key=lambda r: r["stage"])
            if [r["stage"] for r in rows] != list(range(len(protocol["groups"]))):
                raise ValueError(f"missing or duplicate stages: {path}/{arm}")
            checkpoint = path.with_name(f"latest_{arm}.pt")
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            memory = saved["memory"]
            if saved["stage"] != rows[-1]["stage"] or V.memory_hash(memory) != rows[-1]["memory_hash"]:
                raise ValueError(f"memory checksum mismatch: {checkpoint}")
            if len(memory["x"]) != protocol["budgets"][-1]:
                raise ValueError("final budget mismatch")
            if any(len(t) > 320 for t in memory["texts"]):
                raise ValueError("text budget violated")
            if not all(torch.isfinite(memory[k]).all() for k in ("x", "text_emb")):
                raise ValueError("nonfinite memory")
            records.append({"dataset": protocol["dataset"], "arm": arm,
                "checkpoint": str(checkpoint), "memory_hash": rows[-1]["memory_hash"],
                "rows": len(memory["x"]), "stages": len(rows), "bytes": checkpoint.stat().st_size,
                "checks": "hash, budget, text length, finite tensors, protocol all passed"})
    if len(records) != args.expected_cells:
        raise ValueError(f"expected {args.expected_cells} cells, got {len(records)}")
    output = {"status": "passed", "complete_cells": len(records), "records": records,
        "limitations": "constructor access restriction is interface-based, not OS-level read tracing"}
    (args.root / "artifact_audit.json").write_text(json.dumps(output, indent=2))
    print(json.dumps({"status": "passed", "complete_cells": len(records),
                      "checkpoint_bytes": sum(r["bytes"] for r in records)}, indent=2))


if __name__ == "__main__":
    main()
