"""Measure role-aware assignment construction cost without Student training.

The measurement reuses the frozen role_perm construction path, but never
reports F1, writes a memory artifact, or selects a candidate.  It separates
the shared two-prototype proposal from the incremental role assignment cost.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

import run_tgc_role_aware_assignment_20260925 as R


def sync(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True),
                    encoding="utf-8")


def measure_cell(data: dict, source_root: Path, dataset: str, order: str,
                 stage: int, device: str, random_seed: int) -> dict:
    cell = source_root / dataset / order
    parent = R.load(cell / f"c0_s{stage - 1}_memory.pt")
    c0 = R.load(cell / f"c0_s{stage}_memory.pt")
    ids, budgets = R.R11.D.set_stream(data, order)
    old_n = len(parent["x"])
    sync(device)
    q = R.R11.D.current_assignment(data, ids[stage], c0, old_n,
                                   stage, device)
    sync(device)
    row_classes = R.B.class_rows_from_c0(q, data["y"][ids[stage]])
    c0_block = {
        key: c0[key][old_n:].clone()
        if isinstance(c0[key], torch.Tensor) else list(c0[key][old_n:])
        for key in R.PAYLOAD_KEYS
    }

    start = time.perf_counter()
    structural_block, structural_audit = R.B.build_two_prototype_block(
        data, ids[stage], q, row_classes, stage, device, c0_block,
    )
    sync(device)
    structural_seconds = time.perf_counter() - start

    start = time.perf_counter()
    mapping, mapping_audit = R.complete_mapping(
        data, ids[stage], q, row_classes, structural_audit["records"],
        c0, old_n, "role_perm", random_seed,
    )
    sync(device)
    assignment_seconds = time.perf_counter() - start

    start = time.perf_counter()
    memory, block, perm_records = R.memory_from_mapping(
        c0, parent, structural_block, structural_audit["records"], mapping,
    )
    R.Base.assert_conserved(
        structural_block, block, structural_audit["records"],
        perm_records, row_classes, c0, parent,
    )
    sync(device)
    materialization_seconds = time.perf_counter() - start
    del memory

    moved = sum(source != target for source, target in mapping.items())
    role_classes = mapping_audit["classes"]
    return {
        "dataset": dataset,
        "order": order,
        "stage": int(stage),
        "budget": int(budgets[stage]),
        "old_rows": int(old_n),
        "new_rows": int(len(c0["x"]) - old_n),
        "proposal_rows": int(len(structural_block["x"])),
        "moved_payloads": int(moved),
        "classes_with_changes": int(sum(a["changed"] > 0 for a in role_classes)),
        "structural_proposal_seconds": float(structural_seconds),
        "role_assignment_seconds": float(assignment_seconds),
        "materialization_seconds": float(materialization_seconds),
        "role_incremental_seconds": float(assignment_seconds + materialization_seconds),
        "construction_seconds": float(
            structural_seconds + assignment_seconds + materialization_seconds),
        "historical_raw_reads": 0,
        "report_labels_used": False,
        "same_c0_topology": True,
        "same_old_prefix": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--minilm", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--datasets", nargs="+", default=["cora", "citeseer"])
    parser.add_argument("--orders", nargs="+",
                        default=["order19605", "order19606", "order19608"])
    parser.add_argument("--stages", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--random-seed", type=int, default=20260925)
    args = parser.parse_args()
    torch.set_num_threads(2)
    data_cache = {
        dataset: R.V1.load_data(
            dataset, args.minilm, args.device, args.cache_root,
        )
        for dataset in args.datasets
    }
    rows = []
    for dataset in args.datasets:
        for order in args.orders:
            for stage in args.stages:
                row = measure_cell(
                    data_cache[dataset], args.source_root, dataset, order,
                    stage, args.device, args.random_seed,
                )
                rows.append(row)
                print(json.dumps({"event": "cost_complete", **row}), flush=True)
    save_json(args.output, {
        "protocol": {
            "name": "tgc-role-aware-assignment-cost-v1",
            "proposal": "frozen two-prototype structural proposal",
            "assignment": "role_perm: joint graph-text + role + mass",
            "student_training": False,
            "report_f1_selection": False,
            "historical_raw_reads": 0,
            "report_labels_used": False,
            "device": args.device,
        },
        "cells": rows,
    })


if __name__ == "__main__":
    main()
