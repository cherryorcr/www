"""Final-stage Student x constructor replay for the current fixed protocol.

This runner reuses the frozen attribution blocks for CaT-CCTP, Adaptive
coherent, and Unified-Q.  It constructs GECC in memory with the same current
delta-only adapter, evaluates the final prefix with both Student families,
and writes only a small JSON file.  No synthetic memory archive is written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

import run_unified_q_attribution_20261008 as A


EXISTING_ARMS = ("cat_cctp", "adaptive_coherent", "uq_dm_edgefree")
ALL_ARMS = EXISTING_ARMS + ("gecc_hard",)
SEEDS = (150, 151, 152)
STEPS = 2048
BATCH = 24


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=True, indent=2), encoding="utf-8")
    temporary.replace(path)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reconstruct_existing(source: Path, arm: str, stages: int) -> tuple[dict, list[str]]:
    parent = None
    stage_hashes = []
    for stage in range(stages):
        block, _ = A.load_block(source, arm, stage)
        memory = (A.UQ.append_memory(parent, block)
                  if arm == "uq_dm_edgefree" else A.append_block(parent, block))
        stage_hashes.append(A.V.memory_hash(memory))
        parent = memory
    return parent, stage_hashes


def construct_gecc(data: dict, ids: list[torch.Tensor], budgets: list[int],
                   seed: int, device: str) -> tuple[dict, list[str], list[dict]]:
    parent = None
    stage_hashes = []
    audits = []
    cfg = A.M.Config(seed=seed, cat_steps=3000,
                     text_sources=5, text_mode="nearest")
    for stage, raw_ids in enumerate(ids):
        delta = A.L.delta_view(data, raw_ids, stage)
        previous = 0 if parent is None else len(parent["x"])
        new_rows = int(budgets[stage]) - previous
        allocation = A.M.ratio_allocation(delta["y"], new_rows)
        block, audit = A.M.gecc_block(
            delta, new_rows, data["encoder"], cfg, device)
        block["labels"] = A.M.row_labels(allocation)
        # ``append_block`` carries provenance sidecars across prefixes; the
        # baseline constructor returns them in its audit rather than block.
        block["text_source_ids"] = audit["text_source_ids"]
        parent = A.append_block(parent, block)
        if parent["edge_index"].numel():
            raise AssertionError("GECC adapter produced non-edge-free memory")
        stage_hashes.append(A.V.memory_hash(parent))
        audits.append({
            "stage": stage,
            "new_rows": new_rows,
            "allocation": allocation,
            "construction_seconds": audit.get("construction_seconds"),
            "implementation": audit.get("implementation"),
        })
    return parent, stage_hashes, audits


def legacy_score(memory: dict, data: dict, stage: int, device: str) -> dict:
    """Train Legacy with the same inference operator used by Prompt Student.

    The historical replay accidentally rebuilt the adjacency from raw
    ``edge_index`` here, while Prompt used the loader's ``real_adj``.  That
    changed edge symmetrization/deduplication between Student arms.  Reuse the
    frozen helper, whose only intended difference from Prompt is the training
    objective.
    """
    score = A.legacy_score(memory, data, stage, SEEDS, device, STEPS)
    score["student_family"] = "legacy"
    score["student_name"] = "original row-wise InfoNCE trainer"
    score["source_reused"] = False
    score["inference_operator"] = "shared data[real_adj]"
    return score


def validate_loaded_protocol(data: dict, source_protocol: dict) -> dict:
    """Fail closed if the evaluation split/order differs from the source run."""
    checks = {
        "report_mask_hash": A.V.hash_tensor(data["report_dev_mask"]),
        "pool_mask_hash": A.V.hash_tensor(data["pool_mask"]),
        "stage_id_hashes": [A.V.hash_tensor(value) for value in A.L.stage_ids(data)],
    }
    expected = {
        key: source_protocol.get(key)
        for key in ("report_mask_hash", "pool_mask_hash", "stage_id_hashes")
    }
    for key, value in expected.items():
        if value is None:
            raise AssertionError(f"source protocol missing {key}")
        if checks[key] != value:
            raise AssertionError(f"loaded data changed {key}: {checks[key]} != {value}")
    return checks


def load_prompt_score(source_json: dict, arm: str, final_stage: int) -> dict:
    matches = [row for row in source_json["rows"]
               if row.get("arm") == arm
               and row.get("stage") == final_stage
               and row.get("student_family") == "prompt"]
    if len(matches) != 1:
        raise ValueError(f"missing prompt source row for {arm} stage {final_stage}")
    score = dict(matches[0]["score"])
    score["student_family"] = "prompt"
    score["student_name"] = "GraphTextPromptStudent"
    score["source_reused"] = True
    return score


def prompt_score(memory: dict, data: dict, stage: int, device: str) -> dict:
    with A.preserve_rng():
        score = A.L.score(memory, data, stage, list(SEEDS), device, STEPS, BATCH)
    score["student_family"] = "prompt"
    score["student_name"] = "GraphTextPromptStudent"
    score["source_reused"] = False
    return score


def run(args: argparse.Namespace) -> None:
    root = Path(args.root)
    source = root / "results" / "unified_q_attribution_20261008" / (
        f"{args.dataset}_o{args.order_seed}")
    output_dir = (args.output_dir if args.output_dir is not None else
                  root / "results" / "student_method_factorial_20261010")
    output = output_dir / f"{args.dataset}_o{args.order_seed}.json"
    reference = root / "results" / "unified_q_inloop_cat_20261007" / args.dataset / "protocol.json"
    cache = root / "results" / "public_baselines_text_cache_20261001" / args.dataset
    source_json = json.loads((source / "results.json").read_text(encoding="utf-8"))
    protocol = source_json["protocol"]
    if source_json.get("status") != "complete":
        raise ValueError(f"incomplete source: {source}")
    if not reference.exists() or not cache.exists():
        raise FileNotFoundError(f"missing protocol/cache for {args.dataset}")

    manifest = {"groups": protocol["groups"],
                "report_dev_mod": protocol.get("report_dev_mod", 3)}
    data = A.L.load_large(args.dataset, args.minilm, cache, args.device, manifest)
    loaded_hashes = validate_loaded_protocol(data, protocol)
    ids = A.L.stage_ids(data)
    budgets = [int(v) for v in protocol["budgets"]]
    if len(ids) != len(budgets):
        raise AssertionError("stage/budget mismatch")
    final_stage = len(budgets) - 1

    rows = []
    method_hashes = {}
    construction_audits = {}
    for arm in EXISTING_ARMS:
        memory, stage_hashes = reconstruct_existing(source, arm, len(budgets))
        method_hashes[arm] = {
            "final_memory_hash": A.V.memory_hash(memory),
            "stage_memory_hashes": stage_hashes,
        }
        source_final = next(row for row in source_json["rows"]
                           if row.get("arm") == arm
                           and row.get("stage") == final_stage
                           and row.get("student_family") == "prompt")
        if source_final["memory_hash"] != method_hashes[arm]["final_memory_hash"]:
            raise AssertionError(f"source memory hash mismatch: {arm}")
        rows.append({
            "dataset": args.dataset,
            "order_seed": args.order_seed,
            "arm": arm,
            "method_class": "frozen attribution constructor",
            "stage": final_stage,
            "memory_rows": len(memory["x"]),
            "memory_hash": method_hashes[arm]["final_memory_hash"],
            "student": load_prompt_score(source_json, arm, final_stage),
            "legacy": legacy_score(memory, data, final_stage, args.device),
        })

    gecc_memory, gecc_hashes, gecc_audits = construct_gecc(
        data, ids, budgets, int(protocol["construction_seed"]), args.device)
    method_hashes["gecc_hard"] = {
        "final_memory_hash": A.V.memory_hash(gecc_memory),
        "stage_memory_hashes": gecc_hashes,
    }
    construction_audits["gecc_hard"] = gecc_audits
    rows.append({
        "dataset": args.dataset,
        "order_seed": args.order_seed,
        "arm": "gecc_hard",
        "method_class": "current-delta-only GECC adapter",
        "stage": final_stage,
        "memory_rows": len(gecc_memory["x"]),
        "memory_hash": method_hashes["gecc_hard"]["final_memory_hash"],
        "student": prompt_score(gecc_memory, data, final_stage, args.device),
        "legacy": legacy_score(gecc_memory, data, final_stage, args.device),
    })

    result = {
        "status": "complete",
        "protocol": {
            "name": "student-method-factorial-20261010-final-stage",
            "dataset": args.dataset,
            "order_seed": args.order_seed,
            "construction_seed": protocol["construction_seed"],
            "groups": protocol["groups"],
            "budgets": budgets,
            "evaluation_stage": final_stage,
            "evaluation_split": "report-dev; not sealed test",
            "student_seeds": list(SEEDS),
            "student_steps": STEPS,
            "student_batch": BATCH,
            "inference_operator": protocol.get("inference_operator",
                                                "shared symmetric normalized full transductive graph"),
            "access": protocol.get("access"),
            "historical_raw_reads_by_constructor": 0,
            "source_results": str(source / "results.json"),
            "source_results_sha256": file_hash(source / "results.json"),
            "loaded_data_hashes": loaded_hashes,
            "source_protocol": protocol,
            "gecc_adapter": "current delta only; sklearn hard Lloyd with matched class budget; self-loop memory",
            "legacy_control": "original row-wise InfoNCE trainer; same seeds, steps, batch, report-dev evaluation and data[real_adj] inference operator",
            "prompt_control": "reused from frozen source row for existing arms; rerun for GECC with data[real_adj]",
        },
        "method_hashes": method_hashes,
        "construction_audits": construction_audits,
        "rows": rows,
    }
    write_json(output, result)
    print(json.dumps({"event": "complete", "dataset": args.dataset,
                      "order": args.order_seed, "output": str(output),
                      "arms": [row["arm"] for row in rows]}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("wikics", "sportsfit", "bookhis"), required=True)
    parser.add_argument("--order-seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--root", type=Path, default=Path("/home/pc/research/TAG_condensation"))
    parser.add_argument("--minilm", default="/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    torch.set_num_threads(2)
    run(args)


if __name__ == "__main__":
    main()
