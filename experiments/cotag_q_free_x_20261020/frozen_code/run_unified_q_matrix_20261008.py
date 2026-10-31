"""Compact same-protocol baseline and component-ablation matrix.

This runner keeps the current arm's synthetic memory in RAM and writes only
JSON rows. It is intended for disk-constrained confirmation runs; it does not
change the frozen In-loop CaT-Q constructor.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys
import time

import torch

import unified_q_multimodal as UQ
import external_continual_mechanisms as M
import graphtext_prompt_student as P
import puma_tag_adapter as PU
import run_unified_q_large_smoke_20261007 as L
# Legacy dataset imports prepend their directory; keep the frozen snapshot
# ahead of that directory for all constructor/experiment modules.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_unified_q_matched_baselines_20261007 as B
import run_unified_q_inloop_cat_20261007 as I
import run_unified_q_revision_20261007 as V


BASELINE_ARMS = (
    "cat_cctp",
    "random_coreset",
    "kcenter_coreset",
    "gecc_hard",
    "puma_tag",
)

ABLATION_ARMS = (
    "uq_dm_edgefree",
    "uq_no_gm",
    "uq_no_dm",
    "uq_fixed_q",
    "uq_frozen_text",
    "uq_no_prop",
    "uq_no_residual",
    "uq_no_text_init",
    "uq_detached_text",
    "uq_no_q_anchor",
    "uq_no_x_anchor",
    "uq_dm_only",
)


def q_config(arm: str, seed: int, steps: int, sinkhorn_max: int) -> UQ.Config:
    base = I.config("uq_dm_edgefree", seed, steps, sinkhorn_max)
    if arm == "uq_no_gm":
        return replace(base, lambda_gradient=0.0)
    if arm == "uq_no_dm":
        return replace(base, lambda_distribution=0.0)
    if arm == "uq_fixed_q":
        return replace(base, train_q=False)
    if arm == "uq_frozen_text":
        return replace(base, text_assignment="frozen")
    if arm == "uq_no_prop":
        return replace(base, lambda_propagation=0.0)
    if arm == "uq_no_residual":
        return replace(base, train_residual=False)
    if arm == "uq_no_text_init":
        return replace(base, text_weight=0.0)
    if arm == "uq_detached_text":
        return replace(base, text_assignment="detached")
    if arm == "uq_no_q_anchor":
        return replace(base, lambda_q_anchor=0.0)
    if arm == "uq_no_x_anchor":
        return replace(base, lambda_x_anchor=0.0)
    if arm == "uq_dm_only":
        return replace(base, lambda_gradient=0.0, lambda_propagation=0.0,
                       lambda_q_anchor=0.0, lambda_x_anchor=0.0)
    if arm == "uq_dm_edgefree":
        return base
    raise ValueError(f"unknown Q arm: {arm}")


def digest(value: torch.Tensor) -> str:
    value = value.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def compact_audit(audit: dict) -> dict:
    output = {}
    for key, value in audit.items():
        if key in {"support_ids", "text_materialization", "text_source_ids"}:
            continue
        output[key] = value
    return output


def make_baseline(delta: dict, rows: int, arm: str, encoder,
                  seed: int, device: str) -> tuple[dict, dict]:
    if arm == "cat_cctp":
        cfg = M.Config(seed=seed, cat_steps=3000,
                       text_sources=5, text_mode="nearest")
        block, audit = M.cat_block(delta, rows, encoder, cfg, device)
        block["labels"] = M.row_labels(M.ratio_allocation(delta["y"], rows))
        return block, audit
    if arm in {"random_coreset", "kcenter_coreset"}:
        return B.coreset(delta, rows, arm, encoder, seed, device)
    if arm == "gecc_hard":
        cfg = M.Config(seed=seed, cat_steps=3000,
                       text_sources=5, text_mode="nearest")
        block, audit = M.gecc_block(delta, rows, encoder, cfg, device)
        block["labels"] = M.row_labels(audit["allocation"])
        return block, audit
    if arm == "puma_tag":
        cfg = PU.PUMAConfig(seed=seed)
        block, audit = PU.build_puma_block(
            delta, rows, encoder, cfg, device, "puma_full")
        block["labels"] = M.row_labels(
            audit["cat_distribution_matching"]["allocation"])
        return block, audit
    raise ValueError(f"unknown baseline arm: {arm}")


def completed_arm(rows: list[dict], arm: str, stages: int) -> bool:
    done = [row["stage"] for row in rows if row["arm"] == arm]
    if done and sorted(done) != list(range(stages)):
        raise RuntimeError("compact runner cannot resume a partial arm; preserve it and use a fresh output")
    return bool(done)


def assignment_checks(audit: dict) -> dict:
    # Summing the full zero-padded Q and its class-local blocks can differ by
    # float32 rounding. The solver tolerance itself remains unchanged.
    error = max(audit["q_row_sum_max_error"], audit["q_column_relative_error"])
    local_error = max(audit["sinkhorn_max_row_error"],
                      audit["sinkhorn_max_column_relative_error"])
    tolerance = 1e-4
    rounding = 8 * torch.finfo(torch.float32).eps
    return dict(sinkhorn_feasible=local_error <= tolerance and error <= tolerance + rounding,
                sinkhorn_block_error=local_error, sinkhorn_full_error=error,
                sinkhorn_solver_tolerance=tolerance,
                full_matrix_rounding_allowance=rounding,
                numerical_warning=error > tolerance)


def protocol(args, data: dict, groups: list[list[int]], ids: list[torch.Tensor],
             budgets: list[int], arms: list[str]) -> dict:
    source_modules = (M, P, PU, L, B, V, UQ, I, I.D)
    hashes = {Path(m.__file__).name: hashlib.sha256(
        Path(m.__file__).read_bytes()).hexdigest() for m in source_modules}
    return {
        "name": "unified-q-matrix-20261008",
        "dataset": args.dataset,
        "groups": groups,
        "budgets": budgets,
        "budget_mode": args.budget_mode,
        "order_seed": args.order_seed,
        "construction_seed": args.construction_seed,
        "arms": arms,
        "student": P.model_config(),
        "student_seeds": args.student_seeds,
        "student_steps": args.student_steps,
        "student_batch": 24,
        "report_dev_mod": L.read(args.reference).get("report_dev_mod", 3),
        "report_mask_hash": V.hash_tensor(data["report_dev_mask"]),
        "pool_mask_hash": V.hash_tensor(data["pool_mask"]),
        "stage_id_hashes": [V.hash_tensor(v) for v in ids],
        "evaluation_split": "report-dev; not sealed test",
        "access": "current delta plus frozen synthetic prefix",
        "historical_raw_reads_by_constructor": 0,
        "same_student_and_budget": True,
        "official_reproduction": False,
        "source_hashes": hashes,
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "q_configs": {a: asdict(q_config(a, args.construction_seed,
                        args.q_steps, args.sinkhorn_max_steps))
                      for a in arms if a in ABLATION_ARMS},
        "raw_eval_graph": "full transductive graph",
        "boundary_enabled": False,
        "persistence": "JSON metrics and tensor hashes; memory held in RAM, not recoverable from hashes",
    }


def run(args: argparse.Namespace) -> None:
    for module in (UQ, P, M, PU, B, I, V):
        if Path(module.__file__).resolve().parent != Path(__file__).resolve().parent:
            raise RuntimeError(f"module escaped the frozen snapshot: {module.__file__}")
    reference = L.read(args.reference)
    groups = V.choose_groups(reference, args.order_seed)
    manifest = {"groups": groups, "report_dev_mod": reference.get("report_dev_mod", 3)}
    data = L.load_large(args.dataset, args.minilm, args.text_cache,
                        args.device, manifest)
    ids = L.stage_ids(data)
    budgets, _ = V.choose_budgets(reference, list(map(len, ids)), groups,
                                  args.budget_mode, 0.05)
    arms = list(args.arms)
    config = protocol(args, data, groups, ids, budgets, arms)
    args.output.mkdir(parents=True, exist_ok=True)
    output_path = args.output / "results.json"
    if output_path.exists():
        saved = json.loads(output_path.read_text(encoding="utf-8"))
        if saved["protocol"] != config:
            raise ValueError("existing output protocol differs")
        result = saved
    else:
        result = {"protocol": config, "rows": [], "status": "running"}
    for arm in arms:
        if completed_arm(result["rows"], arm, len(ids)):
            continue
        parent = None
        for stage, stage_ids in enumerate(ids):
            delta = L.delta_view(data, stage_ids, stage)
            old_n = 0 if parent is None else len(parent["x"])
            new_n = int(budgets[stage]) - old_n
            labels = M.row_labels(M.ratio_allocation(delta["y"], new_n))
            if arm in BASELINE_ARMS:
                cfg = None
            else:
                cfg = q_config(arm, args.construction_seed, args.q_steps,
                                args.sinkhorn_max_steps)
            if args.device.startswith("cuda"):
                torch.cuda.reset_peak_memory_stats(args.device)
            V.sync(args.device)
            started = time.perf_counter()
            if arm in BASELINE_ARMS:
                block, audit = make_baseline(
                    delta, new_n, arm, data["encoder"],
                    args.construction_seed, args.device)
            else:
                block, audit = UQ.optimize_stage(
                    delta, parent, labels, data["encoder"], data["prompt_emb"],
                    cfg, args.device, use_boundary=False, use_topology=False)
            memory = (L.append_edge_free(parent, block) if arm in BASELINE_ARMS
                      else UQ.append_memory(parent, block))
            V.sync(args.device)
            build_seconds = time.perf_counter() - started
            peak = (torch.cuda.max_memory_allocated(args.device) / 2**20
                    if args.device.startswith("cuda") else None)
            checks = V.validate_memory(memory, parent, budgets[stage])
            if cfg is not None:
                checks.update(assignment_checks(audit))
                if not checks["sinkhorn_feasible"] or audit["text_hard_forward_max_error"] > 1e-4:
                    V.write_json(args.output / f"failure_{arm}_s{stage}.json",
                                 dict(checks=checks, audit=compact_audit(audit)))
                    raise AssertionError(f"assignment or text invariant failed: {checks}")
            if memory["edge_index"].numel():
                raise AssertionError("edge-free matrix acquired non-self-loop edges")
            started = time.perf_counter()
            score = L.score(memory, data, stage, args.student_seeds,
                            args.device, args.student_steps, 24)
            V.sync(args.device)
            student_seconds = time.perf_counter() - started
            row = {
                "arm": arm,
                "stage": stage,
                "raw_rows": len(stage_ids),
                "new_rows": new_n,
                "memory_rows": int(budgets[stage]),
                "score": score,
                "construction_seconds": build_seconds,
                "student_seconds": student_seconds,
                "construction_peak_mb": peak,
                "memory_hash": V.memory_hash(memory),
                "x_hash": digest(memory["x"]),
                "text_hash": digest(memory["text_emb"]),
                "audit": compact_audit(audit),
                "checks": checks,
                "method_class": ("coreset_reference" if "coreset" in arm else
                                 "external_adapter" if arm in BASELINE_ARMS else
                                 "primary" if arm == "uq_dm_edgefree" else "component_ablation"),
                "config": asdict(cfg) if cfg is not None else None,
            }
            result["rows"].append(row)
            result["rows"] = sorted(result["rows"],
                                     key=lambda value: (arms.index(value["arm"]), value["stage"]))
            V.write_json(output_path, result)
            print(json.dumps({"event": "evaluated", "dataset": args.dataset,
                              "arm": arm, "stage": stage,
                              "f1": score["final_f1"],
                              "accuracy": score["accuracy"]}), flush=True)
            parent = memory
        # Free the previous arm's graph/memory before beginning the next arm.
        parent = None
    result["status"] = "complete"
    V.write_json(output_path, result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True,
                        choices=("wikics", "sportsfit", "bookhis", "arxiv"))
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--text-cache", type=Path, required=True)
    parser.add_argument("--minilm", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--arms", nargs="+",
                        choices=BASELINE_ARMS + ABLATION_ARMS,
                        default=list(BASELINE_ARMS + ABLATION_ARMS))
    parser.add_argument("--order-seed", type=int, default=9101)
    parser.add_argument("--construction-seed", type=int, default=271006)
    parser.add_argument("--budget-mode", choices=("fixed_schedule",),
                        default="fixed_schedule")
    parser.add_argument("--student-seeds", nargs="+", type=int,
                        default=[150, 151, 152])
    parser.add_argument("--student-steps", type=int, default=2048)
    parser.add_argument("--q-steps", type=int, default=30)
    parser.add_argument("--sinkhorn-max-steps", type=int, default=2048)
    args = parser.parse_args()
    torch.set_num_threads(2)
    run(args)


if __name__ == "__main__":
    main()
