"""Fixed-training-budget sensitivity panel using the frozen EF constructor.

The frozen module directory is supplied through PYTHONPATH. Only the node
retention ratio changes; sealed results and constructor settings stay intact.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import torch

import run_unified_q_matrix_20261008 as R


ARMS = ("uq_dm_edgefree", "cat_cctp", "gecc_hard")


def digest(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args) -> None:
    ref = R.L.read(args.reference)
    groups = ref["groups"]
    manifest = {"groups": groups, "report_dev_mod": ref.get("report_dev_mod", 3)}
    data = R.L.load_large(args.dataset, args.minilm, args.text_cache,
                         args.device, manifest)
    ids = R.L.stage_ids(data)
    sizes = list(map(len, ids))
    budgets, floors = R.V.choose_budgets(ref, sizes, groups, "fixed_rate", args.rate)
    baseline_budgets, _ = R.V.choose_budgets(ref, sizes, groups, "fixed_rate", 0.05)
    reference_matches_rate = baseline_budgets == ref["budgets"]
    if not reference_matches_rate and not args.allow_legacy_budget_reference:
        raise AssertionError("canonical 5% schedule does not match this construction pool")
    for field, actual in (("pool_mask_hash", R.V.hash_tensor(data["pool_mask"])),
                          ("report_mask_hash", R.V.hash_tensor(data["report_dev_mask"]))):
        if field in ref and ref[field] != actual:
            raise AssertionError(f"reference {field} mismatch")
    if "stage_id_hashes" in ref and ref["stage_id_hashes"] != [R.V.hash_tensor(v) for v in ids]:
        raise AssertionError("reference stage membership mismatch")
    cfg = R.q_config("uq_dm_edgefree", args.construction_seed, 30, 4096)
    protocol = {
        "name": "frozen-ef-fixed-rate-sensitivity-20261012",
        "dataset": args.dataset,
        "rate": args.rate,
        "budget_mode": "fixed_rate",
        "budgets": budgets,
        "stage_raw_rows": sizes,
        "class_floor_adjustments": floors,
        "reference_budgets": ref["budgets"],
        "reference_matches_current_pool_5pct": reference_matches_rate,
        "current_pool_5pct_budgets": baseline_budgets,
        "reference_actual_final_rate": ref["budgets"][-1] / sum(sizes),
        "actual_final_rate": budgets[-1] / sum(sizes),
        "reference_budget_note": ("verified current-pool 5% reference" if reference_matches_rate
                                  else "legacy pre-report-dev schedule; excluded from this fixed-rate curve"),
        "groups": groups,
        "order_seed": ref.get("order_seed"),
        "construction_seed": args.construction_seed,
        "student_seeds": [150, 151, 152],
        "student_steps": 2048,
        "student_batch": 24,
        "student": R.P.model_config(),
        "q_config": asdict(cfg),
        "source_hashes": {Path(m.__file__).name: digest(m.__file__)
                          for m in (R, R.UQ, R.M, R.P, R.L, R.V, R.I, R.I.D)},
        "runner_hash": digest(__file__),
        "pool_mask_hash": R.V.hash_tensor(data["pool_mask"]),
        "report_mask_hash": R.V.hash_tensor(data["report_dev_mask"]),
        "stage_id_hashes": [R.V.hash_tensor(v) for v in ids],
        "evaluation_split": "report-dev; sensitivity only, no method selection",
        "access": "current-delta-only; frozen synthetic prefix",
        "historical_raw_reads_by_constructor": 0,
        "topology": "edge-free synthetic memory; full raw transductive inference",
        "arms": list(ARMS),
        "official_reproduction": False,
        "persistence": "JSON metrics and hashes only; synthetic memory held in RAM",
        "solver_note": "4096 iteration cap at unchanged 1e-4 tolerance; no loss/temperature retuning",
    }
    if min(budgets) < 24:
        raise AssertionError("a rate cell cannot use the common effective batch of 24")
    path = args.output / "results.json"
    if path.exists():
        result = json.loads(path.read_text(encoding="utf-8"))
        if result["protocol"] != protocol:
            raise ValueError("existing result has a different protocol")
    else:
        result = {"protocol": protocol, "rows": [], "status": "running"}
    R.V.write_json(path, result)
    for arm in ARMS:
        done = [r for r in result["rows"] if r["arm"] == arm]
        if len(done) == len(ids):
            continue
        if done:
            raise RuntimeError("partial arm has no saved memory; preserve it and use a fresh retry directory")
        parent = None
        for stage, raw_ids in enumerate(ids):
            delta = R.L.delta_view(data, raw_ids, stage)
            old_n = 0 if parent is None else len(parent["x"])
            new_n = budgets[stage] - old_n
            labels = R.M.row_labels(R.M.ratio_allocation(delta["y"], new_n))
            R.V.sync(args.device)
            torch.cuda.reset_peak_memory_stats(args.device)
            started = time.perf_counter()
            if arm == "uq_dm_edgefree":
                block, audit = R.UQ.optimize_stage(delta, parent, labels,
                    data["encoder"], data["prompt_emb"], cfg, args.device,
                    use_boundary=False, use_topology=False)
                memory = R.UQ.append_memory(parent, block)
            else:
                block, audit = R.make_baseline(delta, new_n, arm, data["encoder"],
                                              args.construction_seed, args.device)
                memory = R.L.append_edge_free(parent, block)
            R.V.sync(args.device)
            build_seconds = time.perf_counter() - started
            peak_mb = torch.cuda.max_memory_allocated(args.device) / 2**20
            checks = R.V.validate_memory(memory, parent, budgets[stage])
            if memory["edge_index"].numel():
                raise AssertionError("EF memory acquired synthetic edges")
            if arm == "uq_dm_edgefree":
                checks.update(R.assignment_checks(audit))
                if not checks["sinkhorn_feasible"] or audit["text_hard_forward_max_error"] > 1e-4:
                    R.V.write_json(args.output / f"failure_{arm}_s{stage}.json",
                                   {"checks": checks, "audit": R.compact_audit(audit)})
                    raise AssertionError("assignment/text invariant failed")
            print(json.dumps({"event": "constructed", "dataset": args.dataset,
                "rate": args.rate, "arm": arm, "stage": stage,
                "seconds": build_seconds, "peak_mb": peak_mb}), flush=True)
            started = time.perf_counter()
            score = R.L.score(memory, data, stage, [150, 151, 152],
                              args.device, 2048, 24)
            R.V.sync(args.device)
            result["rows"].append({
                "arm": arm, "stage": stage, "raw_rows": len(raw_ids),
                "new_rows": new_n, "memory_rows": budgets[stage],
                "effective_student_batch": min(24, len(memory["x"])),
                "score": score, "construction_seconds": build_seconds,
                "student_seconds": time.perf_counter() - started,
                "construction_peak_mb": peak_mb,
                "memory_hash": R.V.memory_hash(memory), "checks": checks,
                "audit": R.compact_audit(audit),
            })
            R.V.write_json(path, result)
            print(json.dumps({"event": "evaluated", "dataset": args.dataset,
                "rate": args.rate, "arm": arm, "stage": stage,
                "f1": score["final_f1"], "accuracy": score["accuracy"]}), flush=True)
            parent = memory
        del parent, memory, block
        torch.cuda.empty_cache()
    result["status"] = "complete"
    R.V.write_json(path, result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("arxiv", "sportsfit", "bookhis"), required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--text-cache", type=Path, required=True)
    parser.add_argument("--minilm", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rate", type=float, choices=(0.01, 0.05, 0.10), required=True)
    parser.add_argument("--allow-legacy-budget-reference", action="store_true",
                        help="Record a legacy schedule mismatch; recompute every rate from the current pool")
    parser.add_argument("--construction-seed", type=int, default=271006)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(2)
    run(args)
