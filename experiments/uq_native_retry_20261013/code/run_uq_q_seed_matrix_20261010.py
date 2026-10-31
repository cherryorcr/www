"""Seeded attribution matrix for shared-Q and residual controls.

This runner is intentionally separate from the sealed experiments.  It keeps
the protocol fixed and changes only the registered constructor factor:

* ``uq_shared_q``: the current in-loop Unified-Q constructor;
* ``uq_split_q``: independent feature and text assignments;
* ``uq_no_residual``: the current constructor with the bounded X residual
  disabled.

Only JSON metrics and memory hashes are written.  Synthetic memories remain
in process memory and are never copied to the persistent result directory.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import time

import torch

import run_unified_q_constructor_attribution_20261008 as B
import shared_q_controls as C

A, R, L, V, M, P, U = B.A, B.R, B.L, B.V, B.M, B.P, B.U

ARMS = ("uq_shared_q", "uq_split_q", "uq_no_residual")


def file_hash(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def config(arm: str, seed: int, steps: int, sinkhorn_max_steps: int):
    base = R.q_config("uq_dm_edgefree", seed, steps, sinkhorn_max_steps)
    if arm == "uq_no_residual":
        return replace(base, train_residual=False)
    if arm in ("uq_shared_q", "uq_split_q"):
        return base
    raise ValueError(arm)


def make_protocol(args, reference, data, groups, ids, budgets):
    cfg = config("uq_shared_q", args.construction_seed, args.q_steps,
                 args.sinkhorn_max_steps)
    modules = (B, A, R, L, V, M, P, U, C)
    return {
        "name": "uq-q-seed-attribution-20261010",
        "dataset": args.dataset,
        "groups": groups,
        "budgets": budgets,
        "budget_mode": "fixed_schedule",
        "order_seed": args.order_seed,
        "construction_seed": args.construction_seed,
        "arms": list(ARMS),
        "student": P.model_config(),
        "student_seeds": args.student_seeds,
        "student_steps": args.student_steps,
        "student_batch": 24,
        "q_steps": args.q_steps,
        "sinkhorn_max_steps": args.sinkhorn_max_steps,
        "q_config": asdict(cfg),
        "q_split_control": "C.optimize(shared=False); same initial logits, targets, constraints and budget",
        "no_residual_control": "same q_config with train_residual=False",
        "evaluation_split": "report-dev; not sealed test",
        "access": "current delta plus own frozen synthetic prefix; no historical raw reads",
        "topology": "edge-free",
        "pool_mask_hash": V.hash_tensor(data["pool_mask"]),
        "report_mask_hash": V.hash_tensor(data["report_dev_mask"]),
        "stage_id_hashes": [V.hash_tensor(v) for v in ids],
        "source_hashes": {Path(m.__file__).name: file_hash(m.__file__) for m in modules},
        "runner_sha256": file_hash(__file__),
        "reference_protocol_hash": file_hash(args.reference),
        "no_parameter_search": True,
        "sealed_primary_unchanged": True,
        "reference_report_dev_mod": reference.get("report_dev_mod", 3),
    }


def compact(audit: dict) -> dict:
    return R.compact_audit(audit)


def run(args):
    reference = L.read(args.reference)
    groups = V.choose_groups(reference, args.order_seed)
    data = L.load_large(
        args.dataset,
        args.minilm,
        args.text_cache,
        args.device,
        {"groups": groups, "report_dev_mod": reference.get("report_dev_mod", 3)},
    )
    ids = L.stage_ids(data)
    budgets, floors = V.choose_budgets(
        reference, list(map(len, ids)), groups, "fixed_schedule", 0.05
    )
    protocol = make_protocol(args, reference, data, groups, ids, budgets)
    protocol["class_floor_adjustments"] = floors
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "results.json"
    if path.exists():
        result = json.loads(path.read_text(encoding="utf-8"))
        if result["protocol"] != protocol:
            raise ValueError("existing output has a different registered protocol")
    else:
        result = {"protocol": protocol, "rows": [], "status": "running"}
    V.write_json(path, result)

    for arm in ARMS:
        parent = None
        for stage, raw_ids in enumerate(ids):
            delta = L.delta_view(data, raw_ids, stage)
            delta["prompt_emb"] = data["prompt_emb"]
            old_n = 0 if parent is None else len(parent["x"])
            new_n = int(budgets[stage]) - old_n
            labels = M.row_labels(M.ratio_allocation(delta["y"], new_n))
            cfg = config(arm, args.construction_seed, args.q_steps,
                         args.sinkhorn_max_steps)
            existing = [r for r in result["rows"] if r["arm"] == arm and r["stage"] == stage]
            if existing:
                if len(existing) != 1:
                    raise ValueError("duplicate stage row")
                parent = None
                break

            if args.device.startswith("cuda"):
                torch.cuda.reset_peak_memory_stats(args.device)
            V.sync(args.device)
            started = time.perf_counter()
            with B.A.preserve_rng():
                if arm == "uq_split_q":
                    block, audit = C.optimize(
                        delta, parent, labels, data["encoder"], data["prompt_emb"],
                        cfg, args.device, shared=False,
                    )
                else:
                    block, audit = U.optimize_stage(
                        delta, parent, labels, data["encoder"], data["prompt_emb"],
                        cfg, args.device, use_boundary=False, use_topology=False,
                    )
            V.sync(args.device)
            build_seconds = time.perf_counter() - started
            peak_mb = (
                torch.cuda.max_memory_allocated(args.device) / 2**20
                if args.device.startswith("cuda") else None
            )
            memory = U.append_memory(parent, block)
            checks = V.validate_memory(memory, parent, budgets[stage])
            if memory["edge_index"].numel():
                raise AssertionError("attribution matrix unexpectedly produced synthetic edges")
            if "q_row_sum_max_error" in audit:
                checks.update(R.assignment_checks(audit))
            if arm == "uq_no_residual" and audit.get("x_residual_norm", 0.0) != 0.0:
                raise AssertionError("no-residual arm updated residual")

            V.sync(args.device)
            score_started = time.perf_counter()
            with B.A.preserve_rng():
                score = L.score(
                    memory, data, stage, args.student_seeds, args.device,
                    args.student_steps, 24,
                )
            V.sync(args.device)
            record = {
                "arm": arm,
                "stage": stage,
                "raw_rows": len(raw_ids),
                "new_rows": new_n,
                "memory_rows": len(memory["x"]),
                "score": score,
                "checks": checks,
                "audit": compact(audit),
                "construction_seconds": build_seconds,
                "student_seconds": time.perf_counter() - score_started,
                "construction_peak_mb": peak_mb,
                "memory_hash": V.memory_hash(memory),
                "x_hash": R.digest(memory["x"]),
                "text_hash": R.digest(memory["text_emb"]),
            }
            result["rows"].append(record)
            V.write_json(path, result)
            print(json.dumps({
                "event": "evaluated",
                "dataset": args.dataset,
                "order": args.order_seed,
                "seed": args.construction_seed,
                "arm": arm,
                "stage": stage,
                "f1": score["final_f1"],
                "accuracy": score["accuracy"],
            }), flush=True)
            parent = memory

    result["status"] = "complete"
    V.write_json(path, result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("wikics", "sportsfit", "bookhis"), required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--text-cache", type=Path, required=True)
    parser.add_argument("--minilm", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--order-seed", type=int, required=True)
    parser.add_argument("--construction-seed", type=int, required=True)
    parser.add_argument("--q-steps", type=int, default=30)
    parser.add_argument("--sinkhorn-max-steps", type=int, default=2048)
    parser.add_argument("--student-steps", type=int, default=2048)
    parser.add_argument("--student-seeds", nargs="+", type=int, default=[150, 151, 152])
    args = parser.parse_args()
    torch.set_num_threads(2)
    run(args)


if __name__ == "__main__":
    main()
