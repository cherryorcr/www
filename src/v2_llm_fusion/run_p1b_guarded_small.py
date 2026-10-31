"""Run the P1-B fixed-budget guarded response experiment on small graphs.

This wrapper reuses the audited multi-order cost runner, but registers a
separate response arm and a matched random-transfer control.  It never
changes the total memory budget or the canonical student protocol.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import torch

import run_small_order_cost_confirmation as B
import run_unified_v3_continual as U


B.__file__ = __file__
B.U = U
B.ARMS = ["dispersion", "response_guarded", "matched_random_transfer"]
B.P1X_ARMS = ["response_guarded", "matched_random_transfer"]


def p1b_args():
    return SimpleNamespace(
        response_base_allocation="dispersion",
        text_recipe="joint_g5_c320",
        text_operator="concat",
        text_diversity_lambda=0.5,
        dual_prototype_alpha=0.5,
        topology_mode="frozen_old",
        cross_degree=1,
        refresh_mode="frozen_old",
        refresh_ratio=0.1,
        refresh_alpha=0.25,
        selection_seeds=[401, 402],
        selection_batches=[16],
        response_old_weight=0.25,
        response_risk_z=1.0,
        response_marginal_rounds=1,
        response_marginal_threshold=0.5,
        response_min_horizon_gain=0.0,
        response_min_lcb=0.0,
        text_recipe_name="joint_g5_c320",
        text_diversity_lambda_name=0.5,
        dual_prototype_alpha_name=0.5,
        x_operator="mean",
    )


B.v4_args = p1b_args


def construct_synthetic(data, arm, stages, budgets, out, device, canonical):
    memories, audits, old = [], [], None
    args = p1b_args()
    for stage, ids in enumerate(stages):
        path = out / f"{arm}_s{stage}_memory.pt"
        audit_path = out / f"{arm}_s{stage}_construction.json"
        if path.exists() and audit_path.exists():
            mem = torch.load(path, map_location="cpu", weights_only=False)
            audit = json.loads(audit_path.read_text())
            if B.C.tensor_hash(mem) != audit["tensor_sha256"]:
                raise ValueError(f"stale P1-B memory artifact: {path}")
        else:
            keep = len(old["x"]) if old is not None else 0
            count = int(budgets[stage]) - int(keep)
            if count <= 0:
                raise ValueError("P1-B requires a growing fixed-rate budget")
            stat = {"response_seconds": 0.0, "response_calls": 0}
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            if arm == "response_guarded":
                with B.response_meter(device, stat):
                    block, selection = U.build_response_marginal_block(
                        old, data, ids, count, keep, stage, args, device)
                selection["p1b_guarded"] = True
                selection["fallback_rule"] = "dispersion_on_gate_failure"
            elif arm == "matched_random_transfer":
                block, selection = U.build_matched_random_transfer_block(
                    data, ids, count, stage, args, device)
                selection["p1b_guarded"] = True
            else:
                block = U.build_stage_block(
                    data, ids, count, data["encoder"], stage,
                    "joint_g5_c320", "concat", device, "dispersion",
                    data["stage_classes"][stage], 0.5, 0.5,
                    x_operator="mean")
                selection = {"mode": "dispersion_control"}
            mem = U.SCA.append_memory(
                old, block, keep, topology_mode="frozen_old", cross_degree=1,
                refresh_mode="frozen_old", device=device)
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            runs, steps = U._selection_cost(selection)
            audit = {
                "arm": arm,
                "stage": stage,
                "keep": int(keep),
                "new": int(count),
                "build_seconds": elapsed,
                "selector_seconds": stat["response_seconds"],
                "non_response_build_seconds": elapsed - stat["response_seconds"],
                "selector_runs": int(runs),
                "selector_steps": int(steps),
                "selection": selection,
                "peak_allocated_mb": torch.cuda.max_memory_allocated(device) / 2**20,
                "tensor_sha256": B.C.tensor_hash(mem),
                "payload": B.payload_stats(mem),
                "total_budget_unchanged": len(mem["x"]) == int(budgets[stage]),
                "synthetic_only": True,
            }
            if canonical and arm == "dispersion":
                audit["canonical_regression"] = B.check_canonical_memory(
                    data["dataset"], arm, stage, mem)
            torch.save(mem, path)
            B.C.save_json(audit_path, audit)
        B.C.assert_frozen(old, mem)
        if len(mem["x"]) != int(budgets[stage]):
            raise AssertionError("P1-B synthetic budget mismatch")
        memories.append(mem)
        audits.append(audit)
        old = mem
    return memories, audits


B.construct_synthetic = construct_synthetic


if __name__ == "__main__":
    B.main()
