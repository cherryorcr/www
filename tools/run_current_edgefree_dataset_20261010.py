"""Compact current-protocol comparison for an additional large dataset.

The reference JSON supplies the preregistered class stream and cumulative
budgets.  CaT, GECC and Unified-Q are all rebuilt with the same Prompt Student,
current-delta-only access boundary and edge-free memory operator.  No memory
tensors are written to disk.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

import run_unified_q_attribution_20261008 as A


ARMS = ("uq_dm_edgefree", "cat_cctp", "gecc_hard")
STUDENT_SEEDS = (150, 151, 152)
SUPPLEMENTARY_SINKHORN_MAX_STEPS = 4096


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=True, indent=2), encoding="utf-8")
    temp.replace(path)


def construct_block(delta: dict, new_rows: int, arm: str, encoder,
                    prompt_emb: torch.Tensor, parent: dict | None,
                    labels: torch.Tensor, seed: int, device: str):
    if arm == "uq_dm_edgefree":
        # Large supplementary graphs can need a few more normalization
        # iterations for the strict float32 Sinkhorn audit.  Keep the model,
        # seed, temperature, and objective unchanged; only raise the solver
        # cap for this non-primary transfer check.
        cfg = A.R.q_config("uq_dm_edgefree", seed, 30,
                           SUPPLEMENTARY_SINKHORN_MAX_STEPS)
        return A.UQ.optimize_stage(delta, parent, labels, encoder, prompt_emb,
                                   cfg, device, use_boundary=False,
                                   use_topology=False)
    cfg = A.M.Config(seed=seed, cat_steps=3000,
                     text_sources=5, text_mode="nearest")
    block, audit = A.R.make_baseline(delta, new_rows, arm, encoder, seed, device)
    if arm == "gecc_hard":
        block["labels"] = A.M.row_labels(audit["allocation"])
    if "text_source_ids" not in block:
        block["text_source_ids"] = [list(v) for v in audit["text_source_ids"]]
    return block, audit


def run(args: argparse.Namespace) -> dict:
    reference_payload = json.loads(Path(args.reference).read_text(encoding="utf-8"))
    reference = reference_payload.get("protocol", reference_payload)
    groups = [[int(v) for v in group] for group in reference["groups"]]
    budgets = [int(v) for v in reference["budgets"]]
    cache = Path(args.text_cache)
    manifest = {"groups": groups, "report_dev_mod": int(reference.get("report_dev_mod", 3))}
    data = A.L.load_large(args.dataset, args.minilm, cache, args.device, manifest)
    ids = A.L.stage_ids(data)
    if len(ids) != len(budgets):
        raise AssertionError("stage count differs from reference budget")
    prompt_emb = data["prompt_emb"]
    rows = []
    for arm in ARMS:
        parent = None
        stage_scores = []
        construction_seconds = 0.0
        for stage, raw_ids in enumerate(ids):
            delta = A.L.delta_view(data, raw_ids, stage)
            old_rows = 0 if parent is None else len(parent["x"])
            new_rows = budgets[stage] - old_rows
            labels = A.M.row_labels(A.M.ratio_allocation(delta["y"], new_rows))
            started = time.perf_counter()
            block, audit = construct_block(delta, new_rows, arm, data["encoder"],
                                           prompt_emb, parent, labels,
                                           args.construction_seed, args.device)
            parent = A.append_block(parent, block)
            if parent["edge_index"].numel():
                raise AssertionError("edge-free arm produced non-empty edges")
            construction_seconds += time.perf_counter() - started
            stage_scores.append(A.L.score(parent, data, stage,
                                          list(STUDENT_SEEDS), args.device,
                                          args.student_steps, args.batch))
        rows.append({
            "dataset": args.dataset,
            "arm": arm,
            "construction_seed": int(args.construction_seed),
            "memory_rows": int(len(parent["x"])),
            "memory_hash": A.V.memory_hash(parent),
            "construction_seconds": float(construction_seconds),
            "stage_scores": stage_scores,
            "final": stage_scores[-1],
            "historical_raw_reads_by_constructor": 0,
            "protocol_status": "current Prompt Student; edge-free; report-dev",
        })
        del parent
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    result = {
        "status": "complete",
        "protocol": {
            "name": "current-edgefree-additional-dataset-20261010",
            "dataset": args.dataset,
            "groups": groups,
            "budgets": budgets,
            "construction_seed": int(args.construction_seed),
            "student": "GraphTextPromptStudent",
            "student_seeds": list(STUDENT_SEEDS),
            "student_steps": int(args.student_steps),
            "student_batch": int(args.batch),
            "evaluation_split": "report-dev; not sealed test",
            "historical_raw_reads_by_constructor": 0,
            "topology": "edge-free/self-loop operator",
            "unified_q_sinkhorn_max_steps": SUPPLEMENTARY_SINKHORN_MAX_STEPS,
            "reference_protocol": str(args.reference),
        },
        "rows": rows,
    }
    write_json(Path(args.output), result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("arxiv", "cora", "citeseer"), required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--text-cache", type=Path, required=True)
    parser.add_argument("--minilm", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--construction-seed", type=int, default=271006)
    parser.add_argument("--student-steps", type=int, default=2048)
    parser.add_argument("--batch", type=int, default=24)
    args = parser.parse_args()
    torch.set_num_threads(2)
    result = run(args)
    print(json.dumps({"event": "complete", "dataset": args.dataset,
                      "output": str(args.output), "rows": len(result["rows"])},
                     ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
