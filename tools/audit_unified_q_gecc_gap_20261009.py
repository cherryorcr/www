"""Audit completed stagewise controls and candidate archives without plotting."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code/v2_llm_fusion"))
import run_unified_q_revision_20261007 as V
from unified_q_gap_diagnostics import edgefree_slice


def load(path):
    with gzip.open(path, "rb") as stream:
        return torch.load(stream, map_location="cpu", weights_only=False)


def audit(root, source_root=None):
    records = []
    for path in sorted(root.glob("*/results.json")):
        cell = json.loads(path.read_text())
        if cell["status"] != "complete":
            continue
        p = cell["protocol"]
        source = Path(p["source"]) if source_root is None else source_root / Path(p["source"]).name
        source_result = json.loads((source / "results.json").read_text())
        if hashlib.sha256((source / "results.json").read_bytes()).hexdigest() != p["source_results_sha256"]:
            raise AssertionError("source results changed")
        if p["phase"] == "p0":
            arms = ("uq_nearest1", "uq_nearest5", "gecc_nearest1", "gecc_nearest5")
        else:
            arms = tuple(r["arm"] for r in cell["rows"] if not r.get("reused") and r["student"] == "prompt_gcn")
        for arm in arms:
            if p["phase"] == "p0":
                base = "uq_dm_edgefree" if arm.startswith("uq_") else "gecc_hard"
                memory = load(source / f"final_{base}.pt.gz")
                sidecar = path.parent / f"payload_{arm}.pt.gz"
                if sidecar.exists():
                    memory.update(load(sidecar))
                elif arm != "gecc_nearest5":
                    raise AssertionError("missing payload sidecar")
                archive = sidecar if sidecar.exists() else source / f"final_{base}.pt.gz"
            else:
                archive = path.parent / f"final_{arm}.pt.gz"
                memory = load(archive)
            rows = [r for r in cell["rows"] if r["arm"] == arm]
            if len(rows) != 2 or {r["student"] for r in rows} != {"prompt_gcn", "prompt_sgc"}:
                raise AssertionError("not the same memory across two architectures")
            digest = V.memory_hash(memory)
            if any(r["memory_hash"] != digest for r in rows):
                raise AssertionError("readout archive differs")
            if memory["edge_index"].numel() or len(memory["x"]) != p["budgets"][-1]:
                raise AssertionError("topology or budget mismatch")
            for stage, end in enumerate(p["budgets"]):
                start = p["budgets"][stage-1] if stage else 0
                block = edgefree_slice(memory, start, end)
                if not (block["stage"] == stage).all() or not torch.isin(block["labels"], torch.tensor(p["groups"][stage])).all():
                    raise AssertionError("stage/label order mismatch")
                if max(map(len, block["texts"])) > 320 or any(not ids for ids in block["text_source_ids"]):
                    raise AssertionError("payload provenance/budget violation")
            records.append(dict(cell=path.parent.name, arm=arm, memory_hash=digest,
                archive=str(archive), sha256=hashlib.sha256(archive.read_bytes()).hexdigest(), rows=len(memory["x"])))
    return dict(status="pass", archives=len(records), records=records,
        limitations="Source membership and re-encoding checked stage-locally by runner. No OS-level data isolation or sealed test.")


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,default=ROOT / "results/uq_gecc_gap_20261009")
    p.add_argument("--source-root",type=Path)
    args=p.parse_args()
    result=audit(args.root,args.source_root)
    (args.root / "archive_audit.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(status=result["status"], archives=result["archives"])))
