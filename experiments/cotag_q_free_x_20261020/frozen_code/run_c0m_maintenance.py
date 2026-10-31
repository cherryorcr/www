"""Frozen small-graph C0-M core-buffer maintenance experiment."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import time

import torch

import c0m_memory_maintenance as M
import run_content_gap_attribution as A
import run_p0_random_synthetic as R
import run_small_comparison_audited as C
import run_student_compatible_append as SCA
import run_unified_v3_continual as U

G, P = C.G, C.P
ARMS = ["c0", "naive_refresh", "c0m"]


def save_json(path: Path, value: dict) -> None:
    C.save_json(path, value)


def payload(memory: dict) -> dict:
    tensor_bytes = {key: int(value.numel() * value.element_size())
                    for key, value in memory.items()
                    if isinstance(value, torch.Tensor)}
    text_bytes = sum(len(str(t).encode("utf-8")) for t in memory["texts"])
    return {"tensor_bytes": tensor_bytes, "text_utf8_bytes": text_bytes,
            "logical_bytes": sum(tensor_bytes.values()) + text_bytes,
            "rows": int(len(memory["x"])),
            "edge_entries": int(memory["edge_index"].shape[1])}


def stage_groups(dataset: str, order: str) -> list[list[int]]:
    return A.stage_groups(dataset, 3, order)


def set_stream(data: dict, groups: list[list[int]], rate: float):
    return A.set_stream(data, groups, rate)


def memory_hash(memory: dict) -> str:
    tensor_hash = C.tensor_hash(memory)
    text_hash = json.dumps(memory.get("texts", []), ensure_ascii=True)
    return hashlib.sha256((tensor_hash + text_hash).encode()).hexdigest()


def adjacency_edges(memory: dict) -> int:
    return int(memory["edge_index"].shape[1])


def build_c0(data: dict, old: dict | None, ids: torch.Tensor,
             stage: int, keep: int, new_rows: int, device: str) -> tuple[dict, dict]:
    started = time.perf_counter()
    block = U.build_stage_block(data, ids, new_rows, data["encoder"], stage,
                                "joint_g5_c320", "concat", device,
                                "dispersion", data["stage_classes"][stage])
    memory = SCA.append_memory(old, block, keep,
                               topology_mode="frozen_old", cross_degree=1,
                               refresh_mode="frozen_old", device=device)
    return memory, {"build_seconds": float(time.perf_counter() - started),
                    "rows": int(len(memory["x"])),
                    "new_rows": int(new_rows),
                    "payload": payload(memory),
                    "memory_sha256": memory_hash(memory)}


def fit(memory: dict, data: dict, stage: int, seed: int, arm: str,
        out: Path, device: str) -> dict:
    key = f"{arm}_s{stage}_seed{seed}"
    row_path, checkpoint = out / f"{key}.json", out / f"{key}.pt"
    if row_path.exists() and checkpoint.exists():
        row = json.loads(row_path.read_text())
        if row["support_sha256"] != C.tensor_hash(memory):
            raise AssertionError("stale C0-M fit")
        return row
    steps = P.student_steps(len(memory["x"]))
    started = time.perf_counter()
    row = C.fit(memory, seed, steps, device, data, stage, checkpoint=checkpoint)
    row.update(arm=arm, stage=int(stage), seed=int(seed),
               support_memory_sha256=memory_hash(memory),
               wall_fit_seconds=float(time.perf_counter() - started),
               effective_batch=min(int(P.STUDENT.batch), len(memory["x"])))
    row["metrics"] = json.loads(json.dumps(row["metrics"]))
    row["checkpoint_sha256"] = C.sha(checkpoint)
    save_json(row_path, row)
    return row


def run_cell(data: dict, groups: list[list[int]], order: str, args, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    stage_ids, budgets = set_stream(data, groups, args.rate)
    cfg = M.MaintenanceConfig()
    manifest = {
        "name": "c0m-core-buffer-maintenance-v1",
        "dataset": data["dataset"], "order": order,
        "stage_classes": groups, "rate": args.rate,
        "budgets": budgets, "arms": ARMS,
        "student_seeds": list(args.seeds),
        "student": {"batch": int(P.STUDENT.batch), "lr": float(P.STUDENT.lr),
                     "steps": "max(50,min(2048,16*rows))"},
        "maintenance_config": cfg.dict(),
        "negative_control": "naive_refresh = existing low-mass alpha=0.25 refresh",
        "fixed": ["C0 incoming block", "rows", "student", "batch", "steps",
                  "report seeds", "report-dev", "old-old topology"],
        "test_labels_evaluated": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    save_json(out / "manifest.json", manifest)
    memories = {arm: None for arm in ARMS}
    results, constructions = [], []
    for stage, ids in enumerate(stage_ids):
        budget, keep = budgets[stage], 0 if stage == 0 else budgets[stage - 1]
        new_rows = budget - keep
        c0_memory, c0_audit = build_c0(
            data, memories["c0"], ids, stage, keep, new_rows, args.device)
        memories["c0"] = c0_memory
        arm_audits = {"c0": c0_audit}
        for arm in ("naive_refresh", "c0m"):
            started = time.perf_counter()
            # Build the incoming block inside the measured region.  C0's
            # reference timing includes this same construction, so the
            # maintenance arms must include it too.
            if arm == "naive_refresh":
                block = U.build_stage_block(
                    data, ids, new_rows, data["encoder"], stage,
                    "joint_g5_c320", "concat", args.device, "dispersion",
                    data["stage_classes"][stage])
                memory = SCA.append_memory(
                    memories[arm], block, keep, topology_mode="frozen_old",
                    cross_degree=1, refresh_mode="bounded_refresh",
                    refresh_ratio=0.10, refresh_alpha=0.25,
                    device=args.device)
                build_audit = {"build_seconds": float(time.perf_counter() - started),
                               "operator": "naive-low-mass-refresh",
                               "payload": payload(memory),
                               "memory_sha256": memory_hash(memory)}
            else:
                block = U.build_stage_block(
                    data, ids, new_rows, data["encoder"], stage,
                    "joint_g5_c320", "concat", args.device, "dispersion",
                    data["stage_classes"][stage])
                memory, maintenance = M.append(
                    memories[arm], block, keep, cfg, device=args.device)
                M.assert_invariants(memories[arm], block, memory,
                                    maintenance, cfg)
                build_audit = {"build_seconds": float(time.perf_counter() - started),
                               "operator": "core-buffer-redundancy-novelty-refresh",
                               "maintenance": maintenance,
                               "payload": payload(memory),
                               "memory_sha256": memory_hash(memory)}
            if len(memory["x"]) != budget:
                raise AssertionError(f"{arm} budget mismatch")
            if adjacency_edges(memory) != adjacency_edges(c0_memory):
                raise AssertionError(f"{arm} edge count differs from C0")
            memories[arm] = memory
            torch.save(memory, out / f"{arm}_s{stage}_memory.pt")
            save_json(out / f"{arm}_s{stage}_construction.json", {
                "arm": arm, "stage": stage, "keep": keep,
                "new_rows": new_rows,
                "memory_sha256": memory_hash(memory),
                "tensor_sha256": C.tensor_hash(memory),
                "payload": payload(memory), "edge_entries": adjacency_edges(memory),
                "build": build_audit,
            })
            arm_audits[arm] = build_audit
        # Save C0 after other arms so the current stage is the shared reference.
        torch.save(c0_memory, out / f"c0_s{stage}_memory.pt")
        save_json(out / f"c0_s{stage}_construction.json", {
            "arm": "c0", "stage": stage, "keep": keep,
            "new_rows": new_rows, "memory_sha256": memory_hash(c0_memory),
            "tensor_sha256": C.tensor_hash(c0_memory),
            "payload": payload(c0_memory), "edge_entries": adjacency_edges(c0_memory),
            "build": c0_audit,
        })
        constructions.append({"stage": stage, "arms": arm_audits,
                              "budgets": budgets, "keep": keep,
                              "new_rows": new_rows})
        for arm in ARMS:
            scores = [fit(memories[arm], data, stage, seed, arm, out, args.device)
                      for seed in args.seeds]
            results.append({"arm": arm, "stage": stage, "rows": len(memories[arm]["x"]),
                            "steps": P.student_steps(len(memories[arm]["x"])),
                            "metrics": C.aggregate(scores), "per_seed": scores})
            save_json(out / "progress.json", {"manifest": manifest,
                                               "results": results,
                                               "constructions": constructions})
            print(json.dumps({"dataset": data["dataset"], "order": order,
                              "stage": stage, "arm": arm,
                              "f1": results[-1]["metrics"]["all"][
                                  "cumulative_f1"]["mean"]}), flush=True)
    save_json(out / "results.json", {"manifest": manifest, "results": results,
                                     "constructions": constructions})
    save_json(out / "completion.json", {"status": "complete",
        "fits": len(results) * len(args.seeds),
        "expected_fits": len(ARMS) * len(stage_ids) * len(args.seeds),
        "completed_at": datetime.now(timezone.utc).isoformat()})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cora", "citeseer"], default=None)
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--minilm", default=(
        "/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip"))
    ap.add_argument("--text-cache-root", default=(
        "/home/pc/research/TAG_condensation/results/p5_rate_curve_small_20260919"))
    ap.add_argument("--rate", type=float, default=0.05)
    ap.add_argument("--order", nargs="+", default=["canonical", "order19101", "order19102"])
    ap.add_argument("--seeds", type=int, nargs="+", default=list(P.STUDENT_SEEDS))
    args = ap.parse_args()
    torch.set_num_threads(2)
    datasets = [args.dataset] if args.dataset else ["cora", "citeseer"]
    root = Path(args.output)
    for dataset in datasets:
        cache = Path(args.text_cache_root) / f"{dataset}_r005" / "text_emb_pool_only.pt"
        data = R.load_data(dataset, args.minilm, args.device, cache)
        for order in args.order:
            groups = stage_groups(dataset, order)
            run_cell(data, groups, order, args, root / dataset / order)
    print("C0M_EXPERIMENT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
