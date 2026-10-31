"""Audit source/memory alignment for the factorial replay."""
from __future__ import annotations

import csv
import argparse
import hashlib
import json
from pathlib import Path

import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code/v2_llm_fusion"))
import run_unified_q_attribution_20261008 as A


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/student_method_factorial_20261010/remote_results"
SOURCE = ROOT / "results/unified_q_attribution_20261008"
OUT = ROOT / "results/student_method_factorial_20261010/audit.json"


def hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=RESULTS)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--gecc-probe-a", type=Path, default=None)
    parser.add_argument("--gecc-probe-b", type=Path, default=None)
    args = parser.parse_args()
    checks = []
    paths = sorted(args.input.glob("*.json"))
    expected_files = {
        f"{dataset}_o{order}.json"
        for dataset in ("wikics", "sportsfit", "bookhis")
        for order in (9101, 9102, 9103)
    }
    actual_files = {path.name for path in paths}
    if actual_files != expected_files:
        raise AssertionError(f"expected exactly 9 order files, got {sorted(actual_files)}")
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "complete":
            raise AssertionError(f"incomplete result: {path}")
        if payload["protocol"].get("inference_operator") != "shared symmetric normalized full transductive graph":
            raise AssertionError(f"unexpected inference operator: {path}")
        dataset = payload["protocol"]["dataset"]
        order = payload["protocol"]["order_seed"]
        source_dir = SOURCE / f"{dataset}_o{order}"
        source_path = source_dir / "results.json"
        source = json.loads(source_path.read_text(encoding="utf-8"))
        if payload["protocol"].get("source_results_sha256") != hash_file(source_path):
            raise AssertionError(f"source results hash mismatch: {path}")
        loaded = payload["protocol"].get("loaded_data_hashes", {})
        for key in ("report_mask_hash", "pool_mask_hash", "stage_id_hashes"):
            if loaded.get(key) != source["protocol"].get(key):
                raise AssertionError(f"loaded {key} mismatch: {path}")
        if len(payload.get("rows", [])) != 4:
            raise AssertionError(f"expected four arms: {path}")
        final_stage = len(payload["protocol"]["budgets"]) - 1
        for row in payload["rows"]:
            arm = row["arm"]
            if row["student"].get("inference_operator") not in (None, "shared data[real_adj]"):
                raise AssertionError(f"legacy operator metadata mismatch: {path} {arm}")
            if arm == "gecc_hard":
                checks.append({"dataset": dataset, "order": order, "arm": arm,
                               "memory_hash_checked": False,
                               "memory_hash_status": "not independently verified; GECC memory is reconstructed in RAM and not archived",
                               "source_reused": False,
                               "prompt_reused": False})
                continue
            source_row = next((value for value in source["rows"]
                               if value.get("arm") == arm
                               and value.get("stage") == final_stage
                               and value.get("student_family") == "prompt"), None)
            if source_row is None:
                raise AssertionError(f"missing source row for {dataset} {order} {arm}")
            if source_row["memory_hash"] != row["memory_hash"]:
                raise AssertionError(f"memory hash mismatch: {dataset} {order} {arm}")
            if not row["student"].get("source_reused"):
                raise AssertionError(f"prompt result not marked reused: {dataset} {order} {arm}")
            checks.append({"dataset": dataset, "order": order, "arm": arm,
                           "memory_hash_checked": True,
                           "source_reused": True,
                           "prompt_reused": True,
                           "source_memory_hash": source_row["memory_hash"],
                           "replay_memory_hash": row["memory_hash"]})
    probe = None
    if args.gecc_probe_a and args.gecc_probe_b:
        probe_a = json.loads(args.gecc_probe_a.read_text(encoding="utf-8"))
        probe_b = json.loads(args.gecc_probe_b.read_text(encoding="utf-8"))
        hash_a = probe_a["method_hashes"]["gecc_hard"]["final_memory_hash"]
        hash_b = probe_b["method_hashes"]["gecc_hard"]["final_memory_hash"]
        if hash_a != hash_b:
            raise AssertionError(f"GECC reproducibility probe mismatch: {hash_a} != {hash_b}")
        probe = {"status": "matched", "hash": hash_a,
                 "probe_a": str(args.gecc_probe_a),
                 "probe_b": str(args.gecc_probe_b)}
    result = {
        "status": "complete",
        "cells": len(checks),
        "existing_memory_hashes_match": all(item["memory_hash_checked"] for item in checks if item["arm"] != "gecc_hard"),
        "gecc_memory_hash_status": "not independently verified; no saved GECC memory",
        "gecc_reproducibility_probe": probe,
        "all_prompt_reuse_flags_match": all((not item["source_reused"] or item["prompt_reused"]) for item in checks),
        "source_runner_hash": hash_file(ROOT / "code/v2_llm_fusion/run_unified_q_attribution_20261008.py"),
        "replay_runner_hash": hash_file(ROOT / "tools/run_student_method_factorial_20261010.py"),
        "checks": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True))


if __name__ == "__main__":
    main()
