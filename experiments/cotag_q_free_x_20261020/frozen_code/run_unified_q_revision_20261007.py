"""Preregistered Unified-Q correctness, weighting, initialization and ablations.

The evaluator is unchanged. Constructor variants use only current delta and
their own frozen synthetic parent. Outputs are resumable and never mix budget
policies, orders or construction seeds in a single cell.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

# Some legacy loaders prepend their source directory to sys.path. Load the
# versioned constructor/Student first so they cannot silently select old code.
import unified_q_multimodal as UQ
import graphtext_prompt_student as P
import external_continual_mechanisms as M
import run_unified_q_large_smoke_20261007 as L


ARMS = ("cat_cctp", "uq_consistent", "uq_balanced", "uq_semantic",
        "uq_text_frozen", "uq_topology_frozen", "uq_edge_free", "uq_alternating")


def arm_config(arm: str, seed: int, steps: int) -> UQ.Config:
    base = UQ.Config(seed=seed, steps=steps, materialize_top_k=1)
    if arm == "uq_consistent":
        return replace(base, init_mode="legacy_spread", loss_weighting="node_mean")
    if arm == "uq_balanced":
        return replace(base, init_mode="legacy_spread")
    if arm == "uq_text_frozen":
        return replace(base, text_assignment="frozen")
    if arm == "uq_topology_frozen":
        return replace(base, topology_assignment="frozen")
    if arm == "uq_alternating":
        return replace(base, optimization="alternating")
    if arm not in ("cat_cctp", "uq_semantic", "uq_edge_free"):
        raise ValueError(arm)
    return base


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=True, indent=2), encoding="utf-8")
    temporary.replace(path)


def hash_tensor(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def memory_hash(memory: dict) -> str:
    digest = hashlib.sha256(json.dumps(memory["texts"], ensure_ascii=True).encode())
    for key in ("x", "text_emb", "labels", "mass", "edge_index"):
        digest.update(hash_tensor(memory[key]).encode())
    return digest.hexdigest()


def choose_groups(reference: dict, order_seed: int | None) -> list[list[int]]:
    if order_seed is None:
        return reference["groups"]
    sizes = [len(g) for g in reference["groups"]]
    classes = sorted(c for g in reference["groups"] for c in g)
    if len(set(classes)) != len(classes):
        raise ValueError("runner requires disjoint class-incremental groups")
    shuffled = np.random.default_rng(order_seed).permutation(classes).tolist()
    groups, offset = [], 0
    for count in sizes:
        groups.append(shuffled[offset:offset + count])
        offset += count
    return groups


def choose_budgets(reference: dict, stage_sizes: list[int], groups: list[list[int]],
                   mode: str, rate: float) -> tuple[list[int], list[dict]]:
    if mode == "fixed_schedule":
        return [int(v) for v in reference["budgets"]], []
    total = previous = 0
    result, floors = [], []
    for stage, (size, group) in enumerate(zip(stage_sizes, groups)):
        total += size
        target = int(round(rate * total))
        budget = max(target, previous + len(group))
        if budget > previous + size:
            raise ValueError("class floor cannot fit current delta")
        if budget != target:
            floors.append({"stage": stage, "rate_target": target, "actual": budget})
        result.append(budget)
        previous = budget
    return result, floors


def validate_memory(memory: dict, parent: dict | None, budget: int) -> dict:
    for key in ("x", "text_emb", "labels", "mass", "texts"):
        if len(memory[key]) != budget:
            raise AssertionError(f"budget mismatch: {key}")
    for key in ("x", "text_emb", "mass"):
        if not bool(torch.isfinite(memory[key]).all()):
            raise AssertionError(f"nonfinite memory: {key}")
    edges = memory["edge_index"]
    if edges.numel() and (int(edges.min()) < 0 or int(edges.max()) >= budget):
        raise AssertionError("edge endpoint outside memory")
    if parent is not None:
        n = len(parent["x"])
        for key in ("x", "text_emb", "labels", "mass"):
            if not torch.equal(memory[key][:n], parent[key]):
                raise AssertionError(f"frozen prefix changed: {key}")
        if memory["texts"][:n] != parent["texts"]:
            raise AssertionError("frozen prefix text changed")
        if not torch.equal(edges[:, (edges < n).all(0)], parent["edge_index"]):
            raise AssertionError("frozen old-old edges changed")
    return {"budget_ok": True, "prefix_ok": True, "finite_ok": True}


def sync(device: str) -> None:
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def run(args) -> dict:
    if Path(UQ.__file__).resolve().parent != Path(__file__).resolve().parent:
        raise RuntimeError("runner imported a different constructor source tree")
    reference = L.read(args.reference)
    groups = choose_groups(reference, args.order_seed)
    manifest = {"groups": groups, "report_dev_mod": reference.get("report_dev_mod", 3)}
    data = L.load_large(args.dataset, args.minilm, args.text_cache, args.device, manifest)
    ids = L.stage_ids(data)
    budgets, floor_adjustments = choose_budgets(reference, list(map(len, ids)), groups,
                                               args.budget_mode, args.memory_rate)
    source_hashes = {Path(module.__file__).name: hashlib.sha256(
        Path(module.__file__).read_bytes()).hexdigest() for module in (UQ, P, M, L)}
    source_hashes[Path(__file__).name] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    protocol = {
        "name": "unified-q-revision-v2-20261007", "dataset": args.dataset,
        "groups": groups, "budgets": budgets, "budget_mode": args.budget_mode,
        "memory_rate_target": args.memory_rate if args.budget_mode == "fixed_rate" else None,
        "class_floor_adjustments": floor_adjustments,
        "order_seed": args.order_seed if args.order_seed is not None else reference.get("order_seed"),
        "construction_seed": args.construction_seed,
        "student": P.model_config(), "student_steps": args.student_steps,
        "student_batch": args.student_batch, "student_seeds": args.student_seeds,
        "evaluation_split": "report-dev only, no test selection",
        "report_mask_hash": hash_tensor(data["report_dev_mask"]),
        "pool_mask_hash": hash_tensor(data["pool_mask"]),
        "stage_id_hashes": [hash_tensor(v) for v in ids],
        "access": "current delta + own frozen synthetic parent; no old raw features/text",
        "historical_raw_reads_by_constructor": 0, "boundary_enabled": False,
        "raw_evaluation_graph": "full transductive graph; labels excluded from construction",
        "source_hashes": source_hashes,
        "arms": {a: asdict(arm_config(a, args.construction_seed, args.q_steps))
                 for a in args.arms if a != "cat_cctp"},
        "cat_config": asdict(M.Config(seed=args.construction_seed, cat_steps=args.cat_steps,
                                      text_sources=5, text_mode="nearest")),
        "primary_candidate_preregistered": "uq_semantic",
        "alternating_budget": "same outer-loss/backward count; half Q and half R updates",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "results.json"
    if path.exists():
        result = json.loads(path.read_text(encoding="utf-8"))
        if result["protocol"] != protocol:
            raise ValueError("existing output has different protocol/code; use a new directory")
    else:
        result = {"protocol": protocol, "rows": [], "status": "running"}
    write_json(args.output / "protocol.json", protocol)
    for arm in args.arms:
        done = [r for r in result["rows"] if r["arm"] == arm]
        checkpoint_path = args.output / f"latest_{arm}.pt"
        parent = None
        if done:
            saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if saved["stage"] != max(r["stage"] for r in done):
                raise ValueError("checkpoint/result stage mismatch")
            parent = saved["memory"]
            if memory_hash(parent) != done[-1]["memory_hash"]:
                raise ValueError("checkpoint memory hash mismatch")
        for stage in range(len(done), min(len(ids), args.max_stages)):
            delta = L.delta_view(data, ids[stage], stage)
            old_n = 0 if parent is None else len(parent["x"])
            new_n = budgets[stage] - old_n
            labels = M.row_labels(M.ratio_allocation(delta["y"], new_n))
            gpu = str(args.device).startswith("cuda")
            if gpu:
                torch.cuda.reset_peak_memory_stats(args.device)
            sync(args.device)
            start = time.perf_counter()
            if arm == "cat_cctp":
                block, audit = M.cat_block(delta, new_n, data["encoder"],
                    M.Config(seed=args.construction_seed, cat_steps=args.cat_steps,
                             text_sources=5, text_mode="nearest"), args.device)
                block["labels"] = labels
                memory = L.append_edge_free(parent, block)
            else:
                block, audit = UQ.optimize_stage(delta, parent, labels, data["encoder"],
                    data["prompt_emb"], arm_config(arm, args.construction_seed, args.q_steps),
                    args.device, use_boundary=False, use_topology=arm != "uq_edge_free")
                memory = UQ.append_memory(parent, block)
            sync(args.device)
            construction_seconds = time.perf_counter() - start
            peak_mb = torch.cuda.max_memory_allocated(args.device) / 2**20 if gpu else None
            checks = validate_memory(memory, parent, budgets[stage])
            if arm != "cat_cctp":
                actual = UQ._hard_adjacency(block["edge_index"], parent, len(memory["x"]), "cpu")
                expected = P.normalized_adjacency(memory["edge_index"], len(memory["x"]), "cpu").to_dense()
                checks["propagation_max_error"] = float((actual - expected).abs().max())
                if checks["propagation_max_error"] > 1e-6:
                    raise AssertionError("constructor/reloaded Student adjacency differs")
            print(json.dumps({"event": "constructed", "arm": arm, "stage": stage,
                              "seconds": construction_seconds, "peak_mb": peak_mb}), flush=True)
            if gpu:
                torch.cuda.reset_peak_memory_stats(args.device)
            sync(args.device)
            start = time.perf_counter()
            score = L.score(memory, data, stage, args.student_seeds, args.device,
                            args.student_steps, args.student_batch)
            sync(args.device)
            student_seconds = time.perf_counter() - start
            student_peak = torch.cuda.max_memory_allocated(args.device) / 2**20 if gpu else None
            temporary = checkpoint_path.with_suffix(".tmp")
            torch.save({"stage": stage, "memory": memory}, temporary)
            temporary.replace(checkpoint_path)
            result["rows"].append({"arm": arm, "stage": stage, "raw_rows": len(ids[stage]),
                "new_rows": new_n, "memory_rows": budgets[stage],
                "construction_seconds": construction_seconds, "student_seconds": student_seconds,
                "construction_peak_mb": peak_mb, "student_peak_mb": student_peak,
                "score": score, "audit": audit, "checks": checks, "memory_hash": memory_hash(memory)})
            write_json(path, result)
            parent = memory
            print(json.dumps({"event": "evaluated", "arm": arm, "stage": stage,
                              "f1": score["final_f1"], "accuracy": score["accuracy"]}), flush=True)
    result["status"] = "complete" if args.max_stages >= len(ids) else "partial-smoke"
    write_json(path, result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("wikics", "bookhis", "sportsfit", "arxiv"), required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--text-cache", type=Path, required=True)
    parser.add_argument("--minilm", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--budget-mode", choices=("fixed_schedule", "fixed_rate"), default="fixed_schedule")
    parser.add_argument("--memory-rate", type=float, default=0.05)
    parser.add_argument("--order-seed", type=int)
    parser.add_argument("--construction-seed", type=int, default=271006)
    parser.add_argument("--student-seeds", nargs="+", type=int, default=[150, 151, 152])
    parser.add_argument("--student-steps", type=int, default=2048)
    parser.add_argument("--student-batch", type=int, default=24)
    parser.add_argument("--q-steps", type=int, default=30)
    parser.add_argument("--cat-steps", type=int, default=3000)
    parser.add_argument("--max-stages", type=int, default=5)
    args = parser.parse_args()
    if not 0 < args.memory_rate <= 1 or args.max_stages < 1:
        parser.error("invalid memory rate/stage count")
    torch.set_num_threads(2)
    run(args)


if __name__ == "__main__":
    main()
