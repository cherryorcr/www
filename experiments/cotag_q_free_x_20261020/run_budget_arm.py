"""Finish a single longer-budget arm without repeating a failed construction."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from pathlib import Path
import shutil
import time
import traceback

import run as B
import run_budget_check as K


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
    cfg = replace(base, steps=100, sinkhorn_tolerance=2e-4,
                  lambda_x_anchor=0. if args.arm == B.FREE else base.lambda_x_anchor)
    protocol.update(name="cotag-independent-x-budget100-single-arm-20261021",
        arms=[args.arm], q_config=asdict(cfg), construction_steps=100,
        primary_protocol_checks_scope="validated before changing steps, solver tolerance and arm-specific X anchor",
        numerical_amendment="same 2e-4 assignment tolerance as v4; see BUDGET_CHECK.md; no objective or learning-rate retuning",
        budget_check_hash=B.E.file_hash(__file__),
        assignment_checker_hash=B.E.file_hash(K.__file__),
        primary_runner_hash=protocol["runner_hash"],
        trigger="v4 SportsFit full arm OOM; finish independent-X arm without repeating failed full arm",
        evaluation_scope="final memory only; separate single-construction development diagnostic")
    B.save(args.output / "protocol.json", protocol)
    result = dict(status="running", protocol=protocol, rows=[], sage={})
    B.save(path, result)
    parent = None
    try:
        for stage, raw_ids in enumerate(ids):
            if shutil.disk_usage(B.ROOT).free < 8 * 2**30:
                raise RuntimeError("less than 8 GiB free disk space")
            delta = B.R.L.delta_view(data, raw_ids, stage)
            new_n = protocol["budgets"][stage] - (0 if parent is None else len(parent["x"]))
            labels = B.R.M.row_labels(B.R.M.ratio_allocation(delta["y"], new_n))
            result["progress"] = dict(arm=args.arm, stage=stage, updates=100)
            B.save(path, result)
            B.emit(event="construction_start", dataset=args.dataset, **result["progress"])
            diagnostic, trajectory = {}, []

            def observe(info):
                diagnostic.update(B.C.diagnostics(info["q"], info["q0"],
                    info["delta"]["text_emb"].float().to(args.device)))

            def observe_step(info):
                trajectory.append(dict(step=info["step"] + 1,
                    **{key: float(info[key].detach()) for key in
                       ("loss", "gm", "distribution", "propagation")}))

            B.R.V.sync(args.device)
            B.torch.cuda.empty_cache()
            B.torch.cuda.reset_peak_memory_stats(args.device)
            start = time.perf_counter()
            if args.arm == B.FREE:
                block, audit = B.C.optimize(delta, parent, labels, data["encoder"],
                                           data["prompt_emb"], cfg, args.device)
            else:
                block, audit = B.R.UQ.optimize_stage(delta, parent, labels, data["encoder"],
                    data["prompt_emb"], cfg, args.device, use_boundary=False,
                    use_topology=False, diagnostic_callback=observe,
                    optimization_callback=observe_step)
                audit.update(diagnostics=diagnostic, component_trajectory=trajectory)
            B.R.V.sync(args.device)
            seconds = time.perf_counter() - start
            peak = B.torch.cuda.max_memory_allocated(args.device) / 2**20
            memory = B.R.L.append_edge_free(parent, block)
            checks = B.R.V.validate_memory(memory, parent, protocol["budgets"][stage])
            checks.update(K.assignment_checks(audit, cfg.sinkhorn_tolerance))
            if not all(checks[key] for key in ("budget_ok", "prefix_ok", "finite_ok", "sinkhorn_feasible")):
                raise AssertionError(checks)
            if memory["edge_index"].numel():
                raise AssertionError("unexpected synthetic edges")
            row = dict(arm=args.arm, stage=stage, raw_rows=len(raw_ids), new_rows=new_n,
                memory_rows=len(memory["x"]), memory_hash=B.R.V.memory_hash(memory),
                checks=checks, audit=B.R.compact_audit(audit),
                construction_seconds=seconds, construction_peak_mb=peak)
            result["rows"].append(row)
            B.save(path, result)
            B.emit(event="constructed", dataset=args.dataset, arm=args.arm, stage=stage,
                   updates=100, seconds=seconds, peak_mb=peak)
            parent = memory
            del block
            B.torch.cuda.empty_cache()
        B.archive(args.output / f"final_{args.arm}.pt.gz", memory)
        result["rows"][-1]["scores"] = B.E.evaluate(memory, data, len(ids) - 1,
                                                    B.SEEDS, args.device)
        B.save(path, result)
        result["sage"][args.arm] = dict(memory_hash=B.R.V.memory_hash(memory),
            scores=B.sage_evaluate(memory, data, args.device))
        result["status"] = "complete"
        B.save(path, result)
        B.emit(event="arm_complete", dataset=args.dataset, arm=args.arm, updates=100,
            gcn_dev_f1=result["rows"][-1]["scores"]["report_dev"]["final_f1"],
            gcn_historical_test_f1=result["rows"][-1]["scores"]["sealed_test"]["final_f1"])
    except BaseException as error:
        result.update(status="failed", failure=dict(type=type(error).__name__,
            message=str(error), traceback=traceback.format_exc()))
        B.save(path, result)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=("arxiv", "sportsfit", "bookhis"), required=True)
    parser.add_argument("--arm", choices=(B.FULL, B.FREE), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.order_seed, args.construction_seed = 9101, 271006
    B.torch.set_num_threads(2)
    for dataset in args.datasets:
        args.dataset = dataset
        args.output = args.output_root / f"{dataset}_{args.arm}_o9101_c271006"
        run_one(args)
