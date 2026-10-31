"""Controlled Unified-Q ablations; no evaluation-dependent method selection."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import time

import torch

import unified_q_multimodal as UQ
import graphtext_prompt_student as P
import run_unified_q_revision_20261007 as V
import run_unified_q_large_smoke_20261007 as L
import external_continual_mechanisms as M


ARMS = ("cat_cctp", "uq_30", "uq_64", "uq_128", "uq_strict", "uq_init",
        "uq_q_fixed", "uq_r_fixed", "uq_feature_only", "uq_edge_free",
        "uq_proxy384", "uq_prop384", "uq_v3", "uq_old_bounded", "uq_old_unbounded")


def config(arm, seed, steps=30, max_sinkhorn=512):
    base = UQ.Config(seed=seed, steps=steps, sinkhorn_tolerance=1e-4,
                    sinkhorn_max_steps=max_sinkhorn)
    if arm == "uq_30":
        return replace(base, sinkhorn_tolerance=0)
    if arm in ("uq_64", "uq_128"):
        return replace(base, sinkhorn_steps=int(arm.split("_")[1]), sinkhorn_tolerance=0)
    if arm == "uq_init":
        return replace(base, steps=0)
    if arm == "uq_q_fixed":
        return replace(base, train_q=False)
    if arm == "uq_r_fixed":
        return replace(base, train_residual=False)
    if arm == "uq_feature_only":
        return replace(base, pair_weight=0.0, text_assignment="frozen")
    if arm in ("uq_proxy384", "uq_prop384", "uq_v3", "uq_old_bounded", "uq_old_unbounded"):
        base = replace(base, proxy_hidden=384, proxy_out=384)
    if arm in ("uq_prop384", "uq_v3", "uq_old_bounded", "uq_old_unbounded"):
        base = replace(base, propagation_metric="cosine")
    if arm in ("uq_v3", "uq_old_bounded", "uq_old_unbounded"):
        base = replace(base, topology_refine_swaps=2)
    if arm == "uq_old_bounded":
        base = replace(base, old_update_radius=0.01)
    if arm == "uq_old_unbounded":
        base = replace(base, old_update_radius=-1.0)
    if arm not in ARMS:
        raise ValueError(arm)
    return base


def validate(memory, parent, budget, cfg):
    if parent is None or cfg.old_update_radius == 0:
        return V.validate_memory(memory, parent, budget)
    check_parent = dict(parent, x=memory["x"][:len(parent["x"])])
    checks = V.validate_memory(memory, check_parent, budget)
    change = (check_parent["x"] - parent["x"]).norm(dim=1) / parent["x"].norm(dim=1).clamp_min(1e-3)
    checks["old_x_relative_change_max"] = float(change.max())
    if cfg.old_update_radius > 0 and change.max() > cfg.old_update_radius + 1e-5:
        raise AssertionError("old X trust radius violated")
    checks["prefix_ok"] = False
    checks["prefix_except_x_ok"] = True
    return checks


def compact_audit(audit):
    # Raw supports remain in the latest memory, not repeated in every JSON row.
    result = {k: v for k, v in audit.items() if k not in (
        "support_ids", "candidate_delta_ids", "text_materialization", "text_source_ids")}
    if "text_materialization" in audit:
        result["text_source_hash"] = hashlib.sha256(json.dumps(
            audit["text_materialization"], sort_keys=True).encode()).hexdigest()
    return result


def run(args, config_fn=config, topology_fn=None,
        protocol_name="unified-q-controlled-diagnostics-v3",
        primary="uq_v3", extra_modules=()):
    if Path(UQ.__file__).resolve().parent != Path(__file__).resolve().parent:
        raise RuntimeError("diagnostic constructor must come from the runner snapshot")
    reference = L.read(args.reference)
    groups = V.choose_groups(reference, args.order_seed)
    manifest = {"groups": groups, "report_dev_mod": reference.get("report_dev_mod", 3)}
    data = L.load_large(args.dataset, args.minilm, args.text_cache, args.device, manifest)
    ids = L.stage_ids(data)
    budgets, floors = V.choose_budgets(reference, list(map(len, ids)), groups,
                                      args.budget_mode, 0.05)
    import unified_q_sparsification as SP
    modules = (UQ, P, M, L, V, SP, *extra_modules)
    source_hashes = {Path(m.__file__).name: hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest() for m in modules}
    source_hashes[Path(__file__).name] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    protocol = {"name": protocol_name, "dataset": args.dataset,
        "groups": groups, "budgets": budgets, "budget_mode": args.budget_mode,
        "order_seed": args.order_seed if args.order_seed is not None else reference.get("order_seed", 9101),
        "construction_seed": args.construction_seed, "student": P.model_config(),
        "student_steps": args.student_steps, "student_batch": 24, "student_seeds": args.student_seeds,
        "report_dev_mod": manifest["report_dev_mod"], "class_floor_adjustments": floors,
        "evaluation_split": "report-dev; not unseen-class or independent final test",
        "report_mask_hash": V.hash_tensor(data["report_dev_mask"]),
        "pool_mask_hash": V.hash_tensor(data["pool_mask"]),
        "stage_id_hashes": [V.hash_tensor(v) for v in ids],
        "source_hashes": source_hashes, "primary_preregistered": primary,
        "arms": {a: asdict(config_fn(a, args.construction_seed, args.q_steps, args.sinkhorn_max_steps)) for a in args.arms},
        "access": "current delta plus own synthetic parent; no historical raw constructor access",
        "boundary_enabled": False, "old_update_arms_are_separate_protocol": True,
        "raw_eval_graph": "full transductive graph", "max_stages": args.max_stages}
    path = args.output / "results.json"
    result = json.loads(path.read_text()) if path.exists() else {"protocol": protocol, "rows": [], "status": "running"}
    if protocol != result["protocol"]:
        raise ValueError("output already exists with another code or protocol")
    V.write_json(args.output / "protocol.json", protocol)
    for arm in args.arms:
        cfg = config_fn(arm, args.construction_seed, args.q_steps, args.sinkhorn_max_steps)
        checkpoint = args.output / f"latest_{arm}.pt"
        done = [r for r in result["rows"] if r["arm"] == arm]
        parent = None
        if done:
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            parent = saved["memory"]
            if saved["stage"] != len(done) - 1 or V.memory_hash(parent) != done[-1]["memory_hash"]:
                raise ValueError("checkpoint does not match completed stages")
        for stage in range(len(done), min(len(ids), args.max_stages)):
            delta = L.delta_view(data, ids[stage], stage)
            old_n = 0 if parent is None else len(parent["x"])
            new_n = budgets[stage] - old_n
            labels = M.row_labels(M.ratio_allocation(delta["y"], new_n))
            if args.device.startswith("cuda"):
                torch.cuda.reset_peak_memory_stats(args.device)
            V.sync(args.device)
            started = time.perf_counter()
            if arm == "cat_cctp":
                block, audit = M.cat_block(delta, new_n, data["encoder"],
                    M.Config(seed=args.construction_seed, cat_steps=3000,
                             text_sources=5, text_mode="nearest"), args.device)
                block["labels"] = labels
                memory = L.append_edge_free(parent, block)
            else:
                block, audit = UQ.optimize_stage(delta, parent, labels, data["encoder"],
                    data["prompt_emb"], cfg, args.device, use_boundary=False,
                    use_topology=(topology_fn(arm) if topology_fn is not None else
                                  arm not in ("uq_feature_only", "uq_edge_free")))
                memory = UQ.append_memory(parent, block)
            V.sync(args.device)
            construction_seconds = time.perf_counter() - started
            peak = torch.cuda.max_memory_allocated(args.device) / 2**20 if args.device.startswith("cuda") else None
            checks = validate(memory, parent, budgets[stage], cfg)
            if arm != "cat_cctp":
                expected = P.normalized_adjacency(memory["edge_index"], len(memory["x"]), "cpu").to_dense()
                actual = UQ._hard_adjacency(block["edge_index"], parent, len(memory["x"]), "cpu")
                checks["propagation_max_error"] = float((actual - expected).abs().max())
                if checks["propagation_max_error"] > 1e-6:
                    raise AssertionError("materialized graph propagation differs")
                checks["sinkhorn_feasible"] = max(audit["sinkhorn_max_row_error"],
                    audit["sinkhorn_max_column_relative_error"]) <= 1e-4
                if cfg.sinkhorn_tolerance and not checks["sinkhorn_feasible"]:
                    raise AssertionError("transport constraint violated")
            print(json.dumps({"event": "constructed", "arm": arm, "stage": stage,
                "construction_seconds": construction_seconds, "peak_mb": peak}), flush=True)
            V.sync(args.device)
            started = time.perf_counter()
            scored = L.score(memory, data, stage, args.student_seeds, args.device, args.student_steps, 24)
            V.sync(args.device)
            student_seconds = time.perf_counter() - started
            temporary = checkpoint.with_suffix(".tmp")
            torch.save({"stage": stage, "memory": memory}, temporary)
            temporary.replace(checkpoint)
            result["rows"].append({"arm": arm, "stage": stage, "raw_rows": len(ids[stage]),
                "new_rows": new_n, "memory_rows": budgets[stage], "score": scored,
                "construction_seconds": construction_seconds, "student_seconds": student_seconds,
                "construction_peak_mb": peak, "memory_hash": V.memory_hash(memory),
                "checks": checks, "audit": compact_audit(audit)})
            V.write_json(path, result)
            print(json.dumps({"event": "evaluated", "arm": arm, "stage": stage,
                "f1": scored["final_f1"], "accuracy": scored["accuracy"]}), flush=True)
            parent = memory
    result["status"] = "complete" if args.max_stages >= len(ids) else "partial-smoke"
    V.write_json(path, result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=("wikics", "bookhis", "sportsfit", "arxiv"))
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--text-cache", type=Path, required=True)
    parser.add_argument("--minilm", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--budget-mode", choices=("fixed_rate", "fixed_schedule"), default="fixed_rate")
    parser.add_argument("--order-seed", type=int)
    parser.add_argument("--construction-seed", type=int, default=271006)
    parser.add_argument("--student-seeds", nargs="+", type=int, default=[150, 151, 152])
    parser.add_argument("--student-steps", type=int, default=2048)
    parser.add_argument("--q-steps", type=int, default=30)
    parser.add_argument("--sinkhorn-max-steps", type=int, default=512)
    parser.add_argument("--max-stages", type=int, default=5)
    args = parser.parse_args()
    torch.set_num_threads(2)
    run(args)


if __name__ == "__main__":
    main()
