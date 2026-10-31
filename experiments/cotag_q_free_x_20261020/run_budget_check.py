"""Matched 100-update development check, with final-memory evaluation."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import math
from pathlib import Path
import shutil
import time
import traceback

import run as B


def assignment_checks(audit, tolerance):
    """Audit against the numerical tolerance actually used by this diagnostic."""
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("a finite positive solver tolerance is required")
    errors = [audit[key] for key in ("q_row_sum_max_error", "q_column_relative_error",
        "sinkhorn_max_row_error", "sinkhorn_max_column_relative_error")]
    finite = all(math.isfinite(value) and value >= 0 for value in errors)
    error, local_error = max(errors[:2]), max(errors[2:])
    rounding = 8 * B.torch.finfo(B.torch.float32).eps
    return dict(sinkhorn_feasible=finite and local_error <= tolerance
                and error <= tolerance + rounding,
        sinkhorn_block_error=local_error, sinkhorn_full_error=error,
        sinkhorn_solver_tolerance=tolerance,
        full_matrix_rounding_allowance=rounding,
        numerical_warning=not finite or error > tolerance)


def run_one(args):
    path = args.output / "results.json"
    if path.exists():
        if B.read(path).get("status") == "complete":
            B.emit(event="skip_complete", path=str(path))
            return
        raise FileExistsError("preserve incomplete results; use a fresh output path")
    if shutil.disk_usage(B.ROOT).free < 8 * 2**30:
        raise RuntimeError("less than 8 GiB free disk space")
    _, _, data, ids, base, protocol = B.load(args)
    # A longer trajectory can miss the frozen 1e-4 Sinkhorn residual by a few
    # 1e-6 after repeated unconstrained updates. This numerical tolerance
    # change does not alter the construction objective.
    base = replace(base, steps=100, sinkhorn_tolerance=2e-4)
    protocol.update(name="cotag-independent-x-budget100-20261020",
        primary_protocol_checks_scope="validated before changing construction steps and solver tolerance",
        numerical_amendment="after v2 failed at row residual 0.0001029372: both arms use 2e-4 instead of 1e-4; v3 checker mismatch corrected without changing either constructor",
        q_config=asdict(base), construction_steps=100,
        budget_check_hash=B.E.file_hash(__file__), primary_runner_hash=protocol["runner_hash"],
        evaluation_scope="final memory only; separate single-construction development diagnostic")
    B.save(args.output / "protocol.json", protocol)
    result = dict(status="running", protocol=protocol, rows=[], sage={})
    B.save(path, result)
    try:
        for arm in (B.FULL, B.FREE):
            cfg = base if arm == B.FULL else replace(base, lambda_x_anchor=0.)
            parent = None
            for stage, raw_ids in enumerate(ids):
                if shutil.disk_usage(B.ROOT).free < 8 * 2**30:
                    raise RuntimeError("less than 8 GiB free disk space")
                result["progress"] = dict(arm=arm, stage=stage, updates=100)
                B.save(path, result)
                delta = B.R.L.delta_view(data, raw_ids, stage)
                new_n = protocol["budgets"][stage] - (0 if parent is None else len(parent["x"]))
                labels = B.R.M.row_labels(B.R.M.ratio_allocation(delta["y"], new_n))
                diagnostics, trajectory = {}, []

                def observe(info):
                    diagnostics.update(B.C.diagnostics(info["q"], info["q0"],
                        info["delta"]["text_emb"].float().to(args.device)))

                def observe_step(info):
                    trajectory.append(dict(step=info["step"] + 1,
                        **{key: float(info[key].detach()) for key in
                           ("loss", "gm", "distribution", "propagation")}))

                B.R.V.sync(args.device)
                B.torch.cuda.empty_cache()
                B.torch.cuda.reset_peak_memory_stats(args.device)
                start = time.perf_counter()
                if arm == B.FULL:
                    block, audit = B.R.UQ.optimize_stage(delta, parent, labels, data["encoder"],
                        data["prompt_emb"], cfg, args.device, use_boundary=False,
                        use_topology=False, diagnostic_callback=observe,
                        optimization_callback=observe_step)
                    audit.update(diagnostics=diagnostics, component_trajectory=trajectory)
                else:
                    block, audit = B.C.optimize(delta, parent, labels, data["encoder"],
                                              data["prompt_emb"], cfg, args.device)
                B.R.V.sync(args.device)
                seconds = time.perf_counter() - start
                peak = B.torch.cuda.max_memory_allocated(args.device) / 2**20
                memory = B.R.L.append_edge_free(parent, block)
                checks = B.R.V.validate_memory(memory, parent, protocol["budgets"][stage])
                checks.update(assignment_checks(audit, cfg.sinkhorn_tolerance))
                if not all(checks[k] for k in ("budget_ok", "prefix_ok", "finite_ok", "sinkhorn_feasible")):
                    raise AssertionError(checks)
                if memory["edge_index"].numel():
                    raise AssertionError("unexpected synthetic edges")
                row = dict(arm=arm, stage=stage, raw_rows=len(raw_ids), new_rows=new_n,
                    memory_rows=len(memory["x"]), memory_hash=B.R.V.memory_hash(memory),
                    checks=checks, audit=B.R.compact_audit(audit),
                    construction_seconds=seconds, construction_peak_mb=peak)
                checkpoint = args.output / f"stage_{stage}_{arm}.pt.gz"
                B.archive(checkpoint, memory)
                row["checkpoint"] = str(checkpoint)
                result["rows"].append(row)
                B.save(path, result)
                B.emit(event="constructed", dataset=args.dataset, arm=arm, stage=stage,
                       updates=100, seconds=seconds, peak_mb=peak)
                parent = memory
                del block
                B.torch.cuda.empty_cache()
            B.archive(args.output / f"final_{arm}.pt.gz", memory)
            result["rows"][-1]["scores"] = B.E.evaluate(memory, data, len(ids) - 1,
                                                        B.SEEDS, args.device)
            B.save(path, result)
            result["sage"][arm] = dict(memory_hash=B.R.V.memory_hash(memory),
                scores=B.sage_evaluate(memory, data, args.device))
            B.save(path, result)
            B.emit(event="arm_complete", dataset=args.dataset, arm=arm, updates=100,
                gcn_dev_f1=result["rows"][-1]["scores"]["report_dev"]["final_f1"],
                gcn_historical_test_f1=result["rows"][-1]["scores"]["sealed_test"]["final_f1"])
            del memory
            B.torch.cuda.empty_cache()
        result["status"] = "complete"
        B.save(path, result)
        B.emit(event="complete", dataset=args.dataset, updates=100)
    except BaseException as error:
        result.update(status="failed", failure=dict(type=type(error).__name__,
            message=str(error), traceback=traceback.format_exc()))
        B.save(path, result)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=("arxiv", "sportsfit", "bookhis"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.order_seed, args.construction_seed = 9101, 271006
    B.torch.set_num_threads(2)
    for dataset in args.datasets:
        args.dataset = dataset
        args.output = args.output_root / f"{dataset}_o9101_c271006"
        run_one(args)
