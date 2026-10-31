"""Verify the transfer manifest and saved memory chains independently."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import torch


def memory_hash(memory):
    h = hashlib.sha256(json.dumps(memory["texts"], ensure_ascii=True).encode())
    for key in ("x", "text_emb", "labels", "mass", "edge_index"):
        blob = memory[key].detach().cpu().contiguous().numpy().tobytes()
        h.update(hashlib.sha256(blob).hexdigest().encode())
    return h.hexdigest()


def load(path):
    with gzip.open(path, "rb") as stream:
        return torch.load(stream, map_location="cpu", weights_only=False)


def verify(root, artifact_root, output):
    manifest = json.loads((root / "export_manifest.json").read_text())
    for record in manifest:
        path = root / record["path"]
        assert path.stat().st_size == record["size"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
    checked, source_full = 0, 0
    for path in sorted(root.glob("*/results.json")):
        data = json.loads(path.read_text())
        p = data["protocol"]
        q_source = artifact_root / Path(p["memory_source"].split("/results/")[1])
        assert hashlib.sha256(q_source.read_bytes()).hexdigest() == p["memory_source_sha256"]
        full = load(q_source)
        source_full += 1
        budgets = p["base_protocol"]["budgets"]
        groups = p["base_protocol"]["groups"]
        for arm in ("uq_dm_edgefree", "gecc_hard"):
            parent = None
            for stage, budget in enumerate(budgets):
                if arm == "gecc_hard":
                    memory = load(path.parent / f"stage_{stage}_{arm}.pt.gz")
                else:
                    memory = {k: (v[:, :0] if k == "edge_index" else v[:budget]) for k, v in full.items()}
                assert len(memory["texts"]) == budget and not memory["edge_index"].numel()
                for k in ("x", "text_emb", "labels", "mass"):
                    assert len(memory[k]) == budget and torch.isfinite(memory[k]).all()
                row = next(r for r in data["rows"] if r["arm"] == arm and r["stage"] == stage)
                assert memory_hash(memory) == row["memory_hash"]
                old_count = 0 if parent is None else len(parent["texts"])
                assert set(memory["labels"][old_count:].tolist()).issubset(groups[stage])
                if parent is not None:
                    assert memory["texts"][:old_count] == parent["texts"]
                    for k in ("x", "text_emb", "labels", "mass"):
                        assert torch.equal(memory[k][:old_count], parent[k])
                parent = memory
                checked += 1
    result = dict(export_files_verified=len(manifest), saved_memory_stage_views_verified=checked,
                  reused_final_memories_verified=source_full)
    assert not output.exists()
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify(args.root, args.artifact_root, args.output)
