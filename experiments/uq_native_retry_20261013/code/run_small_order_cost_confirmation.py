"""Frozen-method order confirmation and synchronized cost accounting.

No method search: regenerate v4 for each preregistered class order. Keep the
archived canonical selector, student arithmetic, and reporting implementation.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import subprocess
import time
from types import SimpleNamespace

import torch

import run_small_comparison_audited as C

U, G, SCA, P = C.U, C.G, C.SCA, C.P
ARMS = [a for a in C.CORE_ARMS if a != "concat64"]
P1X_ARMS = ["mean_x", "weighted_x", "moment_x"]
ORDER_SEEDS = [1701, 1702]


def stage_orders(dataset):
    sizes = [2, 2, 3] if dataset == "cora" else [2, 2, 2]
    orders = {}
    for name, seed in [("canonical", None)] + [(f"order{s}", s) for s in ORDER_SEEDS]:
        classes = list(range(sum(sizes)))
        if seed is not None:
            random.Random(seed).shuffle(classes)
        offset, stages = 0, []
        for size in sizes:
            stages.append(classes[offset:offset + size])
            offset += size
        orders[name] = stages
    return orders


def set_stages(data, groups):
    groups = P.canonicalize_class_groups(groups)
    classes = sorted(set(data["y"].tolist()))
    if sorted(c for group in groups for c in group) != classes or not all(groups):
        raise ValueError("stages must partition all classes exactly once")
    data["stage_classes"] = [list(g) for g in groups]
    G.STAGE_CLASSES = data["stage_classes"]
    SCA.G.STAGE_CLASSES = data["stage_classes"]
    ids = [torch.sort(torch.where(
        data["pool_mask"] & torch.isin(data["y"], torch.tensor(g)))[0]).values
           for g in groups]
    if any(len(x) == 0 for x in ids):
        raise ValueError("empty source stage")
    budgets = [math.floor(.05 * sum(map(len, ids[:s + 1]))) for s in range(len(ids))]
    if min(budgets) < P.STUDENT.batch:
        raise ValueError("confirmation must preserve effective student batch")
    return ids, budgets


def v4_args():
    return SimpleNamespace(response_base_allocation="dispersion", text_recipe="joint_g5_c320",
        text_operator="concat", text_diversity_lambda=.5, dual_prototype_alpha=.5,
        topology_mode="frozen_old", cross_degree=1, refresh_mode="frozen_old",
        refresh_ratio=.1, refresh_alpha=.25, selection_seeds=[300, 301, 302],
        selection_batches=[8, 16, 32], response_old_weight=.25, response_risk_z=1.,
        response_marginal_rounds=2, response_marginal_threshold=0.,
        response_min_horizon_gain=0., response_min_lcb=0.)


def payload_stats(memory):
    tensors = {k: v.numel() * v.element_size() for k, v in memory.items()
               if isinstance(v, torch.Tensor)}
    texts = sum(len(t.encode("utf-8")) for t in memory.get("texts", []))
    return {"rows": len(memory["x"]), "edge_entries": memory["edge_index"].shape[1],
            "tensor_bytes": tensors, "text_utf8_bytes": texts,
            "logical_payload_bytes": sum(tensors.values()) + texts,
            "byte_scope": "all stored tensor fields plus UTF-8 texts; excludes Python/serialization overhead"}


def lock_json(path, value):
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError(f"frozen configuration mismatch: {path}")
    C.save_json(path, value)


def check_archived_fit(row, ref, arm, stage):
    for key in ("support_sha256", "steps", "rows"):
        if row[key] != ref[key]:
            raise AssertionError(f"canonical input regression: {arm} {stage} {key}")
    chained = arm.endswith("finetune") and stage > 0
    if not chained and row["start_state_sha256"] != ref["start_state_sha256"]:
        raise AssertionError("canonical initialization regression")
    equal = row["end_state_sha256"] == ref["end_state_sha256"]
    raw_training = arm.startswith("raw_") or chained
    if not equal and not raw_training:
        raise AssertionError(f"canonical synthetic student regression: {arm} {stage}")
    return {"arm": arm, "stage": stage, "seed": row["seed"],
            "input_match": True, "end_state_bitwise_equal": equal,
            "fresh_f1": row["metrics"]["all"]["cumulative_f1"],
            "archived_f1": ref["metrics"]["all"]["cumulative_f1"],
            "fresh_state": row["end_state_sha256"], "archived_state": ref["end_state_sha256"]}


def lock_plan_with_validation_amendment(out, plan, enabled):
    path = out / "plan.json"
    if path.exists() and json.loads(path.read_text()) != plan:
        if not enabled:
            raise ValueError("explicit validation amendment required")
        previous = json.loads(path.read_text())
        comparable = json.loads(json.dumps(plan))
        runner = str(Path(__file__))
        comparable["code_sha256"][runner] = previous["code_sha256"][runner]
        if comparable != previous:
            raise ValueError("amendment cannot alter algorithms, data, seeds, or protocol")
        diagnostic = out.parent / "cora/raw_regression_diagnostic.json"
        evidence = json.loads(diagnostic.read_text())
        repeats = evidence["repeats"]
        if (len({r["end_state_sha256"] for r in repeats}) < 2
                or len({r["support_sha256"] for r in repeats}) != 1
                or len({r["start_state_sha256"] for r in repeats}) != 1):
            raise ValueError("numerical non-reproducibility was not established")
        snapshot = out.parent / "runner_v1_snapshot.py"
        if C.sha(snapshot) != previous["code_sha256"][runner]:
            raise ValueError("original runner source snapshot missing")
        lock_json(out / "validation_amendment.json", {
            "original_plan_sha256": C.sha(path), "current_runner_sha256": C.sha(__file__),
            "original_runner_sha256": C.sha(snapshot), "diagnostic_sha256": C.sha(diagnostic),
            "reason": "identical raw support/initialization produced distinct GPU training endpoints in repeated fits",
            "policy": "strict synthetic regression; raw endpoint drift recorded; input/initialization and checkpoint reload checks retained",
            "algorithm_or_hyperparameters_changed": False})
        return previous
    lock_json(path, plan)
    return plan


@contextmanager
def response_meter(device, stats):
    original = SCA.evaluate_response_curve
    def measured(*args, **kwargs):
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        result = original(*args, **kwargs)
        torch.cuda.synchronize(device)
        stats["response_seconds"] += time.perf_counter() - start
        stats["response_calls"] += 1
        print(json.dumps({"event": "response", "calls": stats["response_calls"],
                          "seconds": stats["response_seconds"]}), flush=True)
        return result
    SCA.evaluate_response_curve = measured
    try:
        yield
    finally:
        SCA.evaluate_response_curve = original


def check_canonical_memory(dataset, arm, stage, memory):
    reference = C.artifact_dir(dataset, arm) / f"append_fixed_s{stage}_memory.pt"
    old = torch.load(reference, map_location="cpu", weights_only=False)
    if C.tensor_hash(old) != C.tensor_hash(memory) or old["texts"] != memory["texts"]:
        raise AssertionError(f"canonical {arm} memory regression: stage {stage}")
    return {"source": str(reference), "sha256": C.sha(reference), "equal": True}


def construct_synthetic(data, arm, stages, budgets, out, device, canonical):
    memories, audits, old = [], [], None
    for stage, ids in enumerate(stages):
        path = out / f"{arm}_s{stage}_memory.pt"
        audit_path = out / f"{arm}_s{stage}_construction.json"
        if path.exists() and audit_path.exists():
            mem = torch.load(path, map_location="cpu", weights_only=False)
            audit = json.loads(audit_path.read_text())
            if C.tensor_hash(mem) != audit["tensor_sha256"]:
                raise ValueError("stale memory")
        else:
            keep = len(old["x"]) if old is not None else 0
            count = budgets[stage] - keep
            stat = {"response_seconds": 0., "response_calls": 0}
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            start = time.perf_counter()
            selection = None
            if arm == "v4":
                with response_meter(device, stat):
                    block, selection = U.build_response_marginal_block(
                        old, data, ids, count, keep, stage, v4_args(), device)
            else:
                x_operator = {
                    "dispersion": "mean",
                    "mean_x": "mean",
                    "weighted_x": "weighted",
                    "moment_x": "moment",
                }.get(arm, "mean")
                block = U.build_stage_block(data, ids, count, data["encoder"], stage,
                    "joint_g5_c320", "concat", device, "dispersion",
                    data["stage_classes"][stage], .5, .5,
                    x_operator=x_operator)
            mem = SCA.append_memory(old, block, keep, topology_mode="frozen_old",
                                    cross_degree=1, refresh_mode="frozen_old", device=device)
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start
            runs, steps = U._selection_cost(selection)
            audit = {"arm": arm, "stage": stage, "keep": keep, "new": count,
                "build_seconds": elapsed, "selector_seconds": stat["response_seconds"],
                "non_response_build_seconds": elapsed - stat["response_seconds"],
                "selector_runs": runs, "selector_steps": steps, "selection": selection,
                "peak_allocated_mb": torch.cuda.max_memory_allocated(device) / 2**20,
                "tensor_sha256": C.tensor_hash(mem), "payload": payload_stats(mem)}
            if canonical and arm in ("v4", "dispersion"):
                audit["canonical_regression"] = check_canonical_memory(data["dataset"], arm, stage, mem)
            torch.save(mem, path)
            C.save_json(audit_path, audit)
        C.assert_frozen(old, mem)
        if len(mem["x"]) != budgets[stage]:
            raise AssertionError("synthetic budget mismatch")
        memories.append(mem)
        audits.append(audit)
        old = mem
    return memories, audits


def prepare(dataset, minilm, device, out):
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    data = U.load_unified(dataset, U.DATA_CFG[dataset]["path"], minilm, device)
    torch.cuda.synchronize(device)
    load_seconds = time.perf_counter() - start
    # Warm shared kernels once; this is not method-specific training.
    data["encoder"].encode(data["prompts"])
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    ids = torch.where(data["pool_mask"])[0]
    emb = data["encoder"].encode([data["texts"][int(i)] for i in ids])
    data["text_emb"] = torch.zeros((len(data["x"]), 384), dtype=emb.dtype)
    data["text_emb"][ids] = emb
    torch.cuda.synchronize(device)
    encode_seconds = time.perf_counter() - start
    archived = torch.load(C.artifact_dir(dataset, "v4") / "text_emb_pool_only.pt",
                          map_location="cpu", weights_only=False)
    if not torch.equal(archived, data["text_emb"]):
        raise AssertionError("fresh pool encoding differs from frozen source cache")
    data["prompt_emb"] = data["encoder"].encode(data["prompts"])
    data["real_adj"] = G.norm_adj(data["edge_index"], len(data["x"]), device)
    data["x_device"] = data["x"].to(device)
    C.save_json(out / "preprocessing.json", {"load_seconds": load_seconds,
        "pool_text_encoding_seconds": encode_seconds, "pool_rows": len(ids),
        "source_cache_bitwise_equal": True,
        "source_hash": C.sha(U.DATA_CFG[dataset]["path"]),
        "scope": "shared warm-kernel preprocessing once per dataset, not charged once per arm/order; source encoding excludes holdouts"})
    return data


def verify_fit(memory, row, checkpoint, data, stage, device):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = G.CLIPStudent(memory["x"].shape[1]).to(device)
    model.load_state_dict(saved["state"])
    if C.tensor_hash(saved["state"]) != row["end_state_sha256"]:
        raise AssertionError("checkpoint hash mismatch")
    normalized = lambda value: json.loads(json.dumps(value))
    if normalized(C.evaluate(model, data, stage)) != normalized(row["metrics"]):
        raise AssertionError("report metrics mismatch on reload")
    if saved["selected_state"] is not None:
        model.load_state_dict(saved["selected_state"])
        if normalized(C.evaluate(model, data, stage)) != normalized(row["selected_metrics"]):
            raise AssertionError("selected metrics mismatch on reload")
    return True


def run_order(data, name, groups, out, device, arms, seeds, plan_hash):
    out.mkdir(parents=True, exist_ok=True)
    stages, budgets = set_stages(data, groups)
    lock_json(out / "manifest.json", {"plan_sha256": plan_hash, "order": name,
        "dataset": data["dataset"], "stage_classes": groups,
        "pool_counts": list(map(len, stages)), "budgets": budgets,
        "enumeration_protocol": P.ENUMERATION_PROTOCOL,
        "class_order_policy": "sorted-within-stage; stage-order-preserved",
        "stage_ids_sha256": hashlib.sha256(json.dumps(
            [v.tolist() for v in stages], sort_keys=True).encode()).hexdigest(),
        "seed_policy": {"student": list(seeds),
                        "method": P.METHOD_SEED,
                        "selection": list(v4_args().selection_seeds)},
        "seeds": seeds, "arms": arms, "selector": vars(v4_args()),
        "student_steps": [P.student_steps(k) for k in budgets],
        "test_labels_evaluated": False, "report_used_for_selection": False})
    supports, all_audits, results, costs = {}, {}, [], {}
    regressions = []
    for arm in arms:
        cost_path = out / f"{arm}_cost.json"
        cached_cost = json.loads(cost_path.read_text()) if cost_path.exists() else None
        prior_construction = list(out.glob(f"{arm}_s*_construction.json"))
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        if arm in ("v4_init_finetune", "raw_finetune"):
            source = "v4" if arm == "v4_init_finetune" else "raw_equal"
            if source not in supports:
                raise ValueError("fine-tune initializer must precede it")
            memories = [supports[source][0]] + [C.raw_support(data, ids, s)
                                                for s, ids in enumerate(stages) if s]
            audits = []
        elif arm in ("v4", "dispersion", *P1X_ARMS):
            memories, audits = construct_synthetic(data, arm, stages, budgets, out, device,
                                                   name == "canonical")
        else:
            memories, audits = C.construct(data, arm, stages, budgets, out, device)
        torch.cuda.synchronize(device)
        built_seconds = time.perf_counter() - started
        cost = cached_cost or {"build_chain_seconds_with_io": built_seconds,
            "fresh_chain_measurement": not bool(prior_construction),
            "stage_build_seconds_sum": sum(a.get("build_seconds", 0.) for a in audits),
            "build_peak_allocated_mb": torch.cuda.max_memory_allocated(device) / 2**20,
            "selector_seconds": sum(a.get("selector_seconds", 0.) for a in audits),
            "selector_runs": sum(a.get("selector_runs", 0) for a in audits),
            "selector_steps": sum(a.get("selector_steps", 0) for a in audits),
            "memory_payload": [payload_stats(m) for m in memories],
            "build_scope": "algorithm plus response and memory/audit IO; shared preprocessing excluded",
            "x_operator": ("weighted" if arm == "weighted_x" else
                           "moment" if arm == "moment_x" else "mean"),
            "initializer_dependency": ("v4 stage0" if arm == "v4_init_finetune" else
                                       "raw_equal stage0" if arm == "raw_finetune" else None)}
        if not cached_cost:
            C.save_json(cost_path, cost)
        supports[arm], all_audits[arm] = memories, audits
        stage_scores, checks = [], 0
        for stage, memory in enumerate(memories):
            scores = []
            steps = 8192 if arm == "raw_long" else P.student_steps(budgets[stage])
            for seed in seeds:
                key = f"{arm}_s{stage}_seed{seed}"
                rowpath, checkpoint = out / f"{key}.json", out / f"{key}.pt"
                if rowpath.exists() and checkpoint.exists():
                    row = json.loads(rowpath.read_text())
                    if row["support_sha256"] != C.tensor_hash(memory) or row["steps"] != steps:
                        raise ValueError("stale training cache")
                else:
                    initial = None
                    if arm.endswith("finetune") and stage > 0:
                        initial = torch.load(out / f"{arm}_s{stage-1}_seed{seed}.pt",
                                             map_location="cpu", weights_only=False)["state"]
                    torch.cuda.synchronize(device)
                    fit_start = time.perf_counter()
                    row = C.fit(memory, seed, steps, device, data, stage, initial,
                                arm == "raw_long", checkpoint)
                    torch.cuda.synchronize(device)
                    row["fit_wall_seconds_including_report_io"] = time.perf_counter() - fit_start
                    if initial is not None and C.tensor_hash(initial) != row["start_state_sha256"]:
                        raise AssertionError("fine-tune initialization mismatch")
                    C.save_json(rowpath, row)
                checks += int(verify_fit(memory, row, checkpoint, data, stage, device))
                # Canonical final results must remain identical to archived runs.
                if name == "canonical" and arm in ("v4", "dispersion"):
                    refpath = C.RESULTS / "small_comparison_audited_r2_2026_09_16" / data["dataset"] / f"{key}.json"
                    ref = json.loads(refpath.read_text())
                    regressions.append(check_archived_fit(row, ref, arm, stage))
                    C.save_json(out / "canonical_regression.json", regressions)
                scores.append(row)
            result = {"arm": arm, "stage": stage, "rows": len(memory["x"]),
                      "per_seed": scores, "metrics": C.aggregate(scores)}
            if arm == "raw_long":
                result["selected_metrics"] = C.aggregate([dict(r, metrics=r["selected_metrics"]) for r in scores])
            results.append(result)
            stage_scores.extend(scores)
            C.save_json(out / "progress.json", {"results": results, "audits": all_audits})
            print(json.dumps({"event": "stage_done", "dataset": data["dataset"], "order": name,
                              "arm": arm, "stage": stage, "f1": result["metrics"]["all"]["cumulative_f1"]}), flush=True)
        cost.update(student_seconds=sum(r["seconds"] for r in stage_scores),
            student_wall_seconds=sum(r["fit_wall_seconds_including_report_io"] for r in stage_scores),
            student_steps=sum(r["steps"] for r in stage_scores), student_fits=len(stage_scores),
            student_peak_allocated_mb=max(r["peak_allocated_mb"] for r in stage_scores),
            checkpoint_rechecks=checks)
        C.save_json(cost_path, cost)
        costs[arm] = cost
    C.save_json(out / "results.json", {"results": results, "audits": all_audits, "costs": costs})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cora", "citeseer"], required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--arms", nargs="+", default=None,
                    help="subset of registered comparison arms; defaults to all frozen arms")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--minilm", default=str(Path(U.__file__).parent / "minilm_from_graphclip"))
    ap.add_argument("--resume-validation-amendment", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(2)
    selected_arms = list(ARMS if args.arms is None else args.arms)
    allowed_arms = set(ARMS) | set(P1X_ARMS)
    unknown_arms = sorted(set(selected_arms) - allowed_arms)
    if unknown_arms:
        raise ValueError(f"unknown or disabled comparison arms: {unknown_arms}")
    if not selected_arms:
        raise ValueError("at least one comparison arm is required")
    args.output.mkdir(parents=True, exist_ok=True)
    modules = [__file__, C.__file__, U.__file__, G.__file__, SCA.__file__, P.__file__,
               Path(__file__).with_name("small_comparison_metrics.py")]
    plan = {"version": "p1x-order-cost-v1", "dataset": args.dataset,
        "orders": stage_orders(args.dataset), "arms": selected_arms, "student_seeds": list(range(100, 106)),
        "selector": vars(v4_args()), "rate": .05, "raw_steps": 8192,
        "code_sha256": {str(p): C.sha(p) for p in modules},
        "batch": P.STUDENT.batch, "learning_rate": P.STUDENT.lr,
        "test_labels_evaluated": False, "external_reconstruction": False,
        "python": platform.python_version(), "torch": torch.__version__,
        "device": args.device, "gpu": torch.cuda.get_device_name(args.device),
        "num_threads": 2, "prompt_scope": "all primary; seen secondary",
        "cost_policy": "one process per GPU, fresh algorithm construction; shared preprocessing separately; cached resume never masquerades as fresh construction"}
    plan = lock_plan_with_validation_amendment(args.output, plan, args.resume_validation_amendment)
    C.save_json(args.output / "hardware_start.json", {"nvidia_smi": subprocess.check_output(
        ["nvidia-smi"], text=True), "unix_time": time.time()})
    data = prepare(args.dataset, args.minilm, args.device, args.output)
    for name, groups in plan["orders"].items():
        run_order(data, name, groups, args.output / name, args.device,
                  selected_arms, plan["student_seeds"], C.sha(args.output / "plan.json"))
    C.save_json(args.output / "acceptance.json", {"complete": True,
        "orders": list(plan["orders"]), "arms": selected_arms,
        "reloaded_fits": len(plan["orders"]) * len(selected_arms) * 3 * 6,
        "canonical_regression": "v4/dispersion memories and synthetic-trained student states strict; raw endpoint drift recorded with unchanged input and scratch initialization",
        "test_labels_evaluated": False})
    print("ORDER_COST_CONFIRMATION_DONE", flush=True)


if __name__ == "__main__":
    main()
