"""Audit the frozen two-arm recursive semantic confirmation artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics

import torch


def load(path: Path):
    return torch.load(path, map_location="cpu", weights_only=False)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(a: float, b: float, where: str) -> None:
    if abs(float(a) - float(b)) > 1e-8:
        raise AssertionError(f"arithmetic mismatch at {where}: {a} != {b}")


def check_eval(root: Path, cells: list[dict], seeds: list[int]) -> None:
    data = json.loads((root / "results.json").read_text(encoding="utf-8"))
    for cell in data["cells"]:
        cells.append(cell)
        assert set(cell["arms"]) == {"c0", "semantic_perm"}
        c0 = cell["arms"]["c0"]
        sem = cell["arms"]["semantic_perm"]
        assert c0["student_seeds"] == sem["student_seeds"] == seeds
        assert c0["student_steps"] == sem["student_steps"]
        for arm in (c0, sem):
            for row in arm["per_student"]:
                values = row["all_prompt"]["per_class_f1"]
                groups = row["groups"]
                close(row["all_prompt"]["cumulative_f1"], statistics.mean(values.values()), cell["cell"])
                old = [values[str(c)] for group in groups[:-1] for c in group]
                new = [values[str(c)] for c in groups[-1]]
                close(row["all_prompt"]["old_f1"], statistics.mean(old), cell["cell"])
                close(row["all_prompt"]["new_f1"], statistics.mean(new), cell["cell"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--memory-root", type=Path, required=True)
    parser.add_argument("--eval-roots", nargs=2, type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    seeds = [150, 151, 152, 153, 154, 155]
    cells = []
    for root in args.eval_roots:
        check_eval(root, cells, seeds)
    assert len(cells) == 8
    assert len({c["cell"] for c in cells}) == 8
    rows = []
    for cell in cells:
        c0, sem = cell["arms"]["c0"], cell["arms"]["semantic_perm"]
        rows.append({
            "cell": cell["cell"],
            "stage": cell["stage"],
            "cumulative": sem["cumulative_f1_mean"] - c0["cumulative_f1_mean"],
            "old": sem["old_f1_mean"] - c0["old_f1_mean"],
            "new": sem["new_f1_mean"] - c0["new_f1_mean"],
            "c0_sha256": c0["memory_sha256"],
            "semantic_sha256": sem["memory_sha256"],
        })

    artifact = {}
    keys = ("x", "text_emb", "mass", "depth", "stage")
    for row in rows:
        cell = row["cell"]
        path = args.memory_root / cell
        c0 = load(path / "c0_memory.pt")
        sem = load(path / "semantic_perm_memory.pt")
        assert sha256(path / "c0_memory.pt") == row["c0_sha256"]
        assert sha256(path / "semantic_perm_memory.pt") == row["semantic_sha256"]
        assert torch.equal(c0["edge_index"], sem["edge_index"])
        old_n = json.loads((path / "protocol.json").read_text(encoding="utf-8"))["old_rows"]
        assert all(torch.equal(c0[key][:old_n], sem[key][:old_n])
                   for key in keys) if cell.endswith("_s1") else True
        if cell.endswith("_s2"):
            previous = load(args.memory_root / cell.replace("_s2", "_s1") / "semantic_perm_memory.pt")
            assert len(previous["x"]) == old_n
            assert all(torch.equal(previous[key], sem[key][:old_n]) for key in keys)
            assert previous["texts"] == sem["texts"][:old_n]
        build = json.loads((path / "build_audit.json").read_text(encoding="utf-8"))
        assert build["same_c0_topology"] and build["same_parent_prefix"]
        assert build["payload_multiset_preserved"]
        artifact[cell] = {
            "same_c0_topology": True,
            "recursive_parent_prefix": cell.endswith("_s2"),
            "payload_multiset_preserved": True,
            "old_rows": int(old_n),
            "rows": len(c0["x"]),
            "parent_arm": build["parent_arm"],
        }

    summary = {
        "cells": rows,
        "aggregate": {
            "cumulative": statistics.mean(r["cumulative"] for r in rows),
            "old": statistics.mean(r["old"] for r in rows),
            "new": statistics.mean(r["new"] for r in rows),
            "positive_cells": sum(r["cumulative"] > 0 for r in rows),
            "worst_cell": min(r["cumulative"] for r in rows),
        },
        "artifact_checks": artifact,
        "scope": "paired six-seed means on eight fresh Cora/Citeseer cells",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "aggregate": summary["aggregate"]}))


if __name__ == "__main__":
    main()
