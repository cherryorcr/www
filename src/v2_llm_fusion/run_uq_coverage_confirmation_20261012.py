"""Matched report-dev comparison for Q-conditioned semantic coverage.

This is a report-dev follow-up to the frozen Unified-Q text projection
confirmation. The new arm changes the hard real-text projection; because
hard text participates in training, Q and X may subsequently change too.
All other objectives, initialization, Student and budget settings are shared.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import time

import torch

import run_unified_q_attribution_20261008 as A
import unified_q_coverage_diagnostics as D
import unified_q_payload_projection as Projection

R, L, V, M, UQ = A.R, A.L, A.V, A.M, A.UQ
ARMS = ("uq_dm_edgefree", "uq_qmedoid", "uq_qcoverage", "cat_cctp",
        "adaptive_coherent", "gecc_hard")


def validate_arms(arms):
    if not arms or len(set(arms)) != len(arms):
        raise ValueError("arms must be nonempty and distinct")
    if "adaptive_coherent" in arms and (
            "cat_cctp" not in arms or arms.index("cat_cctp") > arms.index("adaptive_coherent")):
        raise ValueError("adaptive coherent requires cat_cctp before it")


def method_class(arm):
    if arm == "uq_dm_edgefree":
        return "primary"
    if arm == "adaptive_coherent":
        return "internal_historical_candidate"
    return "internal_text_projection" if arm.startswith("uq_") else "mechanism_adapted_baseline"


def protect_existing(path):
    if path.exists():
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("status") not in ("complete", "smoke_complete"):
            raise RuntimeError("partial output preserved; retry in a fresh directory")


def digest_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def q_configs(seed: int, steps: int, sinkhorn_max: int):
    base = R.q_config("uq_dm_edgefree", seed, steps, sinkhorn_max)
    medoid = replace(base, text_materialization="q_text_medoid")
    coverage = replace(base, text_materialization="q_coverage")
    return base, medoid, coverage


def score(memory, data, stage, seeds, device, steps):
    with A.preserve_rng():
        return L.score(memory, data, stage, seeds, device, steps, 24)


def run(args: argparse.Namespace) -> None:
    validate_arms(list(args.arms))
    path = args.output / "results.json"
    protect_existing(path)
    if args.max_stages < 1 or args.student_steps < 1 or not args.student_seeds:
        raise ValueError("stages, Student steps and seeds must be positive/nonempty")
    reference = L.read(args.reference)
    groups = V.choose_groups(reference, args.order_seed)
    data = L.load_large(args.dataset, args.minilm, args.text_cache, args.device,
                        dict(groups=groups, report_dev_mod=reference.get("report_dev_mod", 3)))
    ids = L.stage_ids(data)
    raw_sizes = list(map(len, ids))
    budgets, floors = V.choose_budgets(reference, raw_sizes, groups, "fixed_rate", .05)
    base_cfg, medoid_cfg, coverage_cfg = q_configs(
        args.construction_seed, args.q_steps, args.sinkhorn_max_steps)
    arms = list(args.arms)
    protocol = {
        "name": "uq-q-coverage-confirmation-20261012-v2",
        "dataset": args.dataset,
        "groups": groups,
        "order_seed": args.order_seed,
        "construction_seed": args.construction_seed,
        "arms": arms,
        "budgets": budgets,
        "budget_mode": "fixed_rate_current_legal_pool",
        "memory_rate": .05,
        "actual_final_rate": budgets[-1] / sum(raw_sizes),
        "stage_raw_rows": raw_sizes,
        "pool_mask_hash": V.hash_tensor(data["pool_mask"]),
        "report_mask_hash": V.hash_tensor(data["report_dev_mask"]),
        "test_mask_hash": V.hash_tensor(data["test_mask"]),
        "stage_id_hashes": [V.hash_tensor(v) for v in ids],
        "class_floor_adjustments": floors,
        "student": A.P.model_config(),
        "student_seeds": list(args.student_seeds),
        "student_steps": args.student_steps,
        "student_batch": 24,
        "evaluated_stage": min(args.max_stages, len(ids)) - 1,
        "max_stages": args.max_stages,
        "evaluation_split": "report-dev only; sealed test not used",
        "access": "current-delta-only; each arm owns a frozen synthetic prefix",
        "historical_raw_reads_by_constructor": 0,
        "synthetic_topology": "edge-free/self-loop-only",
        "candidate_change": "q_coverage changes hard text selection during construction; Q/X may respond to that selection",
        "coverage_rule": "Q-weighted slot compatibility plus bounded greedy class slot-center coverage without replacement",
        "coverage_weight": .35,
        "diagnostics": "read-only final Q; no dense Q stored; timing includes diagnostics with their cost recorded separately",
        "cost_accounting": "adaptive coherent includes the reused CaT construction cost; loading and shared encoder/cache preparation excluded",
        "q_configs": {
            "base": asdict(base_cfg),
            "q_medoid": asdict(medoid_cfg),
            "q_coverage": asdict(coverage_cfg),
        },
        "source_hashes": {},
        "runner_sha256": digest_file(Path(__file__)),
        "official_reproduction": False,
        "method_selection_from_test": False,
    }
    modules = (A, R, L, V, M, UQ, A.XCR, A.P, D, Projection, R.I, R.I.D)
    protocol["source_hashes"] = {Path(m.__file__).name: digest_file(Path(m.__file__))
                                  for m in modules}
    args.output.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("protocol") != protocol:
            raise ValueError("existing output has a different protocol; use a fresh directory")
        if existing.get("status") in ("complete", "smoke_complete"):
            print(json.dumps({"event": "already_complete", "output": str(path)}), flush=True)
            return
    result = {"protocol": protocol, "stages": [], "rows": [], "status": "running",
              "runtime": {"device": args.device, "torch": torch.__version__,
                          "cost_measurement": "shared research server; elapsed times are not exclusive-device benchmarks"}}
    V.write_json(args.output / "protocol.json", protocol)
    V.write_json(path, result)
    parents = {arm: None for arm in arms}
    final_stage = min(args.max_stages, len(ids)) - 1
    for stage, raw_ids in enumerate(ids[:args.max_stages]):
        delta = L.delta_view(data, raw_ids, stage)
        delta["prompt_emb"] = data["prompt_emb"]
        previous_n = budgets[stage - 1] if stage else 0
        if any((0 if parent is None else len(parent["x"])) != previous_n
               for parent in parents.values()):
            raise AssertionError("arms have different frozen-prefix budgets")
        new_n = budgets[stage] - previous_n
        labels = M.row_labels(M.ratio_allocation(delta["y"], new_n))
        features = None
        blocks = {}
        stage_costs = {}
        for arm in arms:
            V.sync(args.device)
            is_cuda = torch.device(args.device).type == "cuda"
            if is_cuda:
                torch.cuda.reset_peak_memory_stats(args.device)
            started = time.perf_counter()
            diagnostics = D.CoverageDiagnostics()
            if arm.startswith("uq_"):
                cfg = {"uq_dm_edgefree": base_cfg, "uq_qmedoid": medoid_cfg,
                       "uq_qcoverage": coverage_cfg}[arm]
                block, audit = UQ.optimize_stage(
                    delta, parents[arm], labels, data["encoder"],
                    data["prompt_emb"], cfg, args.device,
                    use_boundary=False, use_topology=False,
                    diagnostic_callback=diagnostics)
                audit["coverage_diagnostics"] = diagnostics.result
            elif arm == "cat_cctp":
                block, audit = R.make_baseline(
                    delta, new_n, arm, data["encoder"], args.construction_seed, args.device)
                block["text_source_ids"] = audit["text_source_ids"]
            elif arm == "gecc_hard":
                block, audit = R.make_baseline(
                    delta, new_n, arm, data["encoder"], args.construction_seed, args.device)
                block["text_source_ids"] = audit["text_source_ids"]
            elif arm == "adaptive_coherent":
                if features is None:
                    features = M.propagated_features(
                        delta, (0.0, 1.0), args.device).cpu()
                if "cat_cctp" not in blocks:
                    raise AssertionError("adaptive coherent requires the paired CaT block")
                block, audit = A.adaptive_payload(
                    delta, features, blocks["cat_cctp"], data["encoder"],
                    args.construction_seed, args.device)
            else:
                raise ValueError(f"unknown arm {arm}")
            V.sync(args.device)
            construction_seconds = time.perf_counter() - started
            peak_mb = torch.cuda.max_memory_allocated(args.device) / 2**20 if is_cuda else None
            reused_seconds, reused_peak = stage_costs.get("cat_cctp", (0.0, None)) if arm == "adaptive_coherent" else (0.0, None)
            total_seconds = construction_seconds + reused_seconds
            total_peak = max(v for v in (peak_mb, reused_peak) if v is not None) if is_cuda else None
            stage_costs[arm] = (total_seconds, total_peak)
            if arm.startswith("uq_"):
                checks = R.assignment_checks(audit)
                if not checks["sinkhorn_feasible"]:
                    raise AssertionError(f"assignment invariant failed: {checks}")
                if audit["text_hard_forward_max_error"] > 1e-4:
                    raise AssertionError("hard text forward/materialization mismatch")
            else:
                checks = {}
            memory = (UQ.append_memory(parents[arm], block)
                      if arm.startswith("uq_") else A.append_block(parents[arm], block))
            checks.update(V.validate_memory(memory, parents[arm], budgets[stage]))
            if memory["edge_index"].numel():
                raise AssertionError(f"edge-free arm acquired edges: {arm}")
            with A.preserve_rng():
                checks.update(A.validate_payload(block, delta, data["encoder"]))
            blocks[arm] = block
            stage_record = dict(arm=arm, stage=stage, raw_rows=len(raw_ids),
                                new_rows=new_n, memory_rows=len(memory["x"]),
                                construction_seconds=construction_seconds,
                                construction_seconds_including_reused=total_seconds,
                                reused_cat_seconds=reused_seconds,
                                diagnostic_seconds=diagnostics.seconds,
                                construction_peak_mb=total_peak,
                                memory_hash=V.memory_hash(memory), checks=checks,
                                audit=R.compact_audit(audit))
            result["stages"].append(stage_record)
            print(json.dumps({"event": "constructed", "dataset": args.dataset,
                "arm": arm, "stage": stage, "seconds": total_seconds,
                "peak_mb": total_peak}), flush=True)
            if stage == final_stage:
                started = time.perf_counter()
                scores = score(memory, data, stage, args.student_seeds,
                               args.device, args.student_steps)
                result["rows"].append(dict(arm=arm, stage=stage,
                    memory_rows=len(memory["x"]), memory_hash=V.memory_hash(memory),
                    score=scores,
                    construction_seconds=sum(s["construction_seconds_including_reused"]
                                             for s in result["stages"] if s["arm"] == arm),
                    construction_peak_mb=(max(s["construction_peak_mb"] for s in result["stages"]
                                              if s["arm"] == arm) if is_cuda else None),
                    student_seconds=time.perf_counter() - started,
                    unique_texts=len(set(memory["texts"])),
                    method_class=method_class(arm)))
                print(json.dumps({"event": "evaluated", "dataset": args.dataset,
                    "order": args.order_seed, "seed": args.construction_seed,
                    "arm": arm, "f1": scores["final_f1"],
                    "accuracy": scores["accuracy"]}), flush=True)
            parents[arm] = memory
            del memory
            torch.cuda.empty_cache()
            V.write_json(path, result)
        del blocks
    result["status"] = "complete" if args.max_stages >= len(ids) else "smoke_complete"
    V.write_json(path, result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("wikics", "sportsfit", "bookhis", "arxiv"), required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--text-cache", type=Path, required=True)
    parser.add_argument("--minilm", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--order-seed", type=int, required=True)
    parser.add_argument("--construction-seed", type=int, default=271108)
    parser.add_argument("--q-steps", type=int, default=30)
    parser.add_argument("--sinkhorn-max-steps", type=int, default=4096)
    parser.add_argument("--student-steps", type=int, default=2048)
    parser.add_argument("--student-seeds", nargs="+", type=int, default=[150, 151, 152])
    parser.add_argument("--max-stages", type=int, default=5)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    torch.set_num_threads(2)
    run(parser.parse_args())
