"""Independent artifact/checkpoint audit for the C0-M small-graph matrix."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import torch

import run_small_comparison_audited as C


ARMS = ("c0", "naive_refresh", "c0m")


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def close(left: float, right: float, message: str) -> None:
    if not math.isclose(float(left), float(right), rel_tol=1e-9,
                       abs_tol=1e-7):
        raise AssertionError(f"{message}: {left} != {right}")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def audit_memory(cell: Path, stage: int, arm: str, previous: dict | None,
                 c0_memory: dict, expected_rows: int) -> dict:
    path = cell / f"{arm}_s{stage}_memory.pt"
    construction = read(cell / f"{arm}_s{stage}_construction.json")
    memory = torch.load(path, map_location="cpu", weights_only=False)
    require(C.tensor_hash(memory) == construction["tensor_sha256"],
            f"{cell}/{arm}/stage{stage}: tensor hash mismatch")
    require(len(memory["x"]) == expected_rows,
            f"{cell}/{arm}/stage{stage}: row budget mismatch")
    require(int(memory["edge_index"].shape[1]) == construction["edge_entries"],
            f"{cell}/{arm}/stage{stage}: edge count mismatch")
    if previous is not None:
        n = len(previous["x"])
        old_edges = memory["edge_index"][:, (memory["edge_index"] < n).all(0)]
        require(torch.equal(old_edges, previous["edge_index"]),
                f"{cell}/{arm}/stage{stage}: old-old edges changed")
    if arm == "c0m":
        require(torch.equal(memory["core_rows"],
                            (previous or memory)["core_rows"]),
                f"{cell}/c0m/stage{stage}: core membership changed")
        require(len(memory["buffer_rows"]) == len(memory["buffer_anchor_x"]),
                f"{cell}/c0m/stage{stage}: buffer X anchors mismatch")
        require(len(memory["buffer_rows"]) == len(memory["buffer_anchor_text"]),
                f"{cell}/c0m/stage{stage}: buffer text anchors mismatch")
        for row, anchor_x, anchor_t in zip(
                memory["buffer_rows"].tolist(), memory["buffer_anchor_x"],
                memory["buffer_anchor_text"]):
            x_rel = float((memory["x"][row] - anchor_x).norm() /
                          anchor_x.norm().clamp_min(1e-12))
            t_rel = float((memory["text_emb"][row] - anchor_t).norm() /
                          anchor_t.norm().clamp_min(1e-12))
            require(max(x_rel, t_rel) <= 0.25 + 1e-6,
                    f"{cell}/c0m/stage{stage}: anchor radius exceeded")
        if previous is not None:
            for key in ("x", "text_emb", "texts", "mass", "depth", "stage"):
                # C0-M may maintain old buffer rows, but its protected core
                # must remain exactly unchanged.
                core = previous["core_rows"]
                if key == "texts":
                    require([memory[key][int(i)] for i in core.tolist()] ==
                            [previous[key][int(i)] for i in core.tolist()],
                            f"{cell}/c0m/stage{stage}: core text changed")
                else:
                    require(torch.equal(memory[key][core], previous[key][core]),
                            f"{cell}/c0m/stage{stage}: core {key} changed")
        keep = 0 if previous is None else len(previous["x"])
        for key in ("x", "text_emb", "mass", "depth", "stage"):
            require(torch.equal(memory[key][keep:], c0_memory[key][keep:]),
                    f"{cell}/c0m/stage{stage}: appended {key} differs from C0")
        require(memory["texts"][keep:] == c0_memory["texts"][keep:],
                f"{cell}/c0m/stage{stage}: appended texts differ from C0")
    return memory


def audit_cell(cell: Path, expected_seeds: list[int]) -> dict:
    manifest = read(cell / "manifest.json")
    require(tuple(manifest["arms"]) == ARMS, f"{cell}: arm order changed")
    require(manifest["student_seeds"] == expected_seeds,
            f"{cell}: report seed registration changed")
    results = read(cell / "results.json")
    require(results["manifest"] == manifest, f"{cell}: manifest mismatch")
    stages = len(manifest["stage_classes"])
    budgets = manifest["budgets"]
    memories = {}
    cell_rows = []
    for stage in range(stages):
        c0 = audit_memory(cell, stage, "c0",
                          memories.get("c0"),
                          torch.load(cell / f"c0_s{stage}_memory.pt",
                                     map_location="cpu", weights_only=False),
                          int(budgets[stage]))
        memories["c0"] = c0
        for arm in ARMS[1:]:
            memory = audit_memory(cell, stage, arm, memories.get(arm), c0,
                                  int(budgets[stage]))
            require(int(memory["edge_index"].shape[1]) ==
                    int(c0["edge_index"].shape[1]),
                    f"{cell}/{arm}/stage{stage}: edge count differs from C0")
            memories[arm] = memory
        for arm in ARMS:
            construction = read(cell / f"{arm}_s{stage}_construction.json")
            cell_rows.append({
                "dataset": manifest["dataset"], "order": manifest["order"],
                "stage": stage, "arm": arm, "rows": len(memories[arm]["x"]),
                "edge_entries": int(memories[arm]["edge_index"].shape[1]),
                "build_seconds": float(construction["build"].get("build_seconds", 0.0)),
            })
    expected_keys = {(arm, stage, seed) for arm in ARMS
                    for stage in range(stages) for seed in expected_seeds}
    indexed = {}
    for entry in results["results"]:
        for fit in entry["per_seed"]:
            key = (entry["arm"], int(entry["stage"]), int(fit["seed"]))
            require(key not in indexed, f"{cell}: duplicate fit {key}")
            indexed[key] = fit
            path = cell / f"{entry['arm']}_s{entry['stage']}_seed{fit['seed']}.json"
            require(read(path) == fit, f"{path}: fit JSON mismatch")
            memory = memories[entry["arm"]] if int(entry["stage"]) == stages - 1 else torch.load(
                cell / f"{entry['arm']}_s{entry['stage']}_memory.pt",
                map_location="cpu", weights_only=False)
            require(fit["support_sha256"] == C.tensor_hash(memory),
                    f"{path}: support hash mismatch")
            require(fit["support_memory_sha256"] == construction_hash(memory),
                    f"{path}: support memory hash missing/mismatch")
            checkpoint = cell / f"{entry['arm']}_s{entry['stage']}_seed{fit['seed']}.pt"
            require(C.sha(checkpoint) == fit["checkpoint_sha256"],
                    f"{checkpoint}: checkpoint hash mismatch")
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            require(C.tensor_hash(saved["state"]) == fit["end_state_sha256"],
                    f"{checkpoint}: state hash mismatch")
            close(fit["steps"], max(50, min(2048, 16 * len(memory["x"]))),
                  f"{path}: student steps")
    require(set(indexed) == expected_keys,
            f"{cell}: fit key set mismatch ({len(indexed)} vs {len(expected_keys)})")
    return {"dataset": manifest["dataset"], "order": manifest["order"],
            "stages": stages, "budgets": budgets, "fit_count": len(indexed),
            "cell_rows": cell_rows}


def construction_hash(memory: dict) -> str:
    tensor_hash = C.tensor_hash(memory)
    texts = json.dumps(memory.get("texts", []), ensure_ascii=True)
    return hashlib.sha256((tensor_hash + texts).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    cells = []
    for dataset in ("cora", "citeseer"):
        for order in ("canonical", "order19101", "order19102"):
            cell = args.root / dataset / order
            completion = read(cell / "completion.json")
            require(completion["status"] == "complete", f"{cell}: incomplete")
            cells.append(audit_cell(cell, [100, 101, 102, 103, 104, 105]))
    result = {"status": "passed", "cells": cells,
              "fits": sum(cell["fit_count"] for cell in cells),
              "root": str(args.root)}
    output = args.output or args.root / "audit_summary.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    rows = [row for cell in cells for row in cell["cell_rows"]]
    write_csv(output.with_name("audit_construction.csv"), rows)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
