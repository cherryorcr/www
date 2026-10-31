"""Verify downloaded checkpoints directly, independently of runtime flags."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent


def memory_hash(memory):
    digest = hashlib.sha256(json.dumps(memory["texts"], ensure_ascii=True).encode())
    for key in ("x", "text_emb", "labels", "mass", "edge_index"):
        raw = memory[key].detach().cpu().contiguous().numpy().tobytes()
        digest.update(hashlib.sha256(raw).hexdigest().encode())
    return digest.hexdigest()


def load(path):
    with gzip.open(path, "rb") as stream:
        return torch.load(stream, map_location="cpu", weights_only=False)


def main(args):
    verified = []
    for path in sorted(args.root.glob("*/results.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("status") != "complete":
            continue
        protocol = record["protocol"]
        for arm in ("uq_dm_edgefree", "independent_x"):
            rows = [row for row in record["rows"] if row["arm"] == arm]
            previous = None
            for row in rows:
                checkpoint = path.parent / Path(row["checkpoint"]).name
                memory = load(checkpoint)
                n = protocol["budgets"][row["stage"]]
                assert memory_hash(memory) == row["memory_hash"]
                assert all(len(memory[key]) == n for key in ("x", "text_emb", "labels", "mass", "texts"))
                assert all(torch.isfinite(memory[key]).all() for key in ("x", "text_emb", "mass"))
                assert memory["edge_index"].numel() == 0
                if previous is not None:
                    old_n = len(previous["x"])
                    for key in ("x", "text_emb", "labels", "mass"):
                        assert torch.equal(previous[key], memory[key][:old_n])
                    assert previous["texts"] == memory["texts"][:old_n]
                previous = memory
            final_path = path.parent / f"final_{arm}.pt.gz"
            final = load(final_path)
            assert memory_hash(final) == rows[-1]["memory_hash"]
            assert memory_hash(final) == record["sage"][arm]["memory_hash"]
            verified.append(dict(dataset=protocol["dataset"], arm=arm,
                stages=len(rows), memory_hash=memory_hash(final),
                artifact_sha256=hashlib.sha256(final_path.read_bytes()).hexdigest(),
                checks="checkpoint hashes, frozen prefixes, finite arrays, budgets, no edges, same final memory"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(verified=verified), indent=2), encoding="utf-8")
    print(json.dumps(dict(arms_verified=len(verified), stages_verified=sum(row["stages"] for row in verified))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=HERE / "results/cotag_q_free_x_budget100_20261020_v5")
    parser.add_argument("--output", type=Path, default=HERE / "analysis_budget100/artifact_audit.json")
    main(parser.parse_args())
