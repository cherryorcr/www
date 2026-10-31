"""Detailed old/new F1 reevaluation for frozen role-aware memories.

This runner never constructs or selects a memory.  It reloads immutable C0,
identity and role_perm artifacts, retrains the original Student, and records
per-class all-prompt and seen-prompt F1 so that current-stage F1 is not used as
a proxy for new-class plasticity.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

import run_tgc_joint_student_response as V1
import run_unified_v3_continual as U
import run_tgc_r4_relational as R4


def load(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True),
                    encoding="utf-8")


def set_order_groups(dataset: str, order: str) -> list[list[int]]:
    groups = R4.R.A.stage_groups(dataset, 3, order)
    U.G.STAGE_CLASSES = groups
    return groups


@torch.no_grad()
def detailed_scores(model, data: dict, groups: list[list[int]], stage: int) -> dict:
    logits = model.logits(
        data["x_device"], data["real_adj"],
        data["prompt_emb"].to(data["x_device"].device),
    ).cpu()
    y = data["y"][data["report_dev_mask"]].cpu().numpy()
    logits = logits[data["report_dev_mask"]].numpy()
    seen_classes = [int(c) for group in groups[:stage + 1] for c in group]
    allowed = np.asarray(sorted(seen_classes), dtype=np.int64)
    active = np.isin(y, allowed)
    y, logits = y[active], logits[active]
    per_class = {}
    for cls in seen_classes:
        pred = logits.argmax(1)
        tp = np.sum((y == cls) & (pred == cls))
        fp = np.sum((y != cls) & (pred == cls))
        fn = np.sum((y == cls) & (pred != cls))
        denom = 2 * tp + fp + fn
        per_class[str(cls)] = float(200 * tp / denom) if denom else 0.0

    def scope_scores(scope: str) -> dict:
        if scope == "all":
            pred = logits.argmax(1)
        else:
            pred = allowed[logits[:, allowed].argmax(1)]
        scores = {}
        for cls in seen_classes:
            tp = np.sum((y == cls) & (pred == cls))
            fp = np.sum((y != cls) & (pred == cls))
            fn = np.sum((y == cls) & (pred != cls))
            denom = 2 * tp + fp + fn
            scores[str(cls)] = float(200 * tp / denom) if denom else 0.0
        old = [scores[str(c)] for group in groups[:stage]
               for c in group]
        new = [scores[str(c)] for c in groups[stage]]
        return {
            "cumulative_f1": float(np.mean(list(scores.values()))),
            "old_f1": float(np.mean(old)) if old else None,
            "new_f1": float(np.mean(new)) if new else None,
            "per_class_f1": scores,
            "accuracy": float(np.mean(y == pred) * 100),
        }

    return {"all_prompt": scope_scores("all"),
            "seen_prompt": scope_scores("seen"),
            "n": int(len(y)),
            "groups": groups[:stage + 1]}


def evaluate_cell(cell: Path, data: dict, groups: list[list[int]], stage: int,
                  seeds: list[int], device: str, arms: list[str]) -> dict:
    out = {"cell": cell.name, "stage": stage, "arms": {}}
    for arm in arms:
        memory = load(cell / f"{arm}_memory.pt")
        rows = []
        started = time.perf_counter()
        for seed in seeds:
            steps = U.protocol_v3.student_steps(len(memory["x"]))
            model = U.G.train_student(
                U.G.memory_to_trainable(memory), int(seed), device, steps,
                batch=U.protocol_v3.STUDENT.batch,
            )
            rows.append({
                "seed": int(seed),
                "steps": int(steps),
                **detailed_scores(model, data, groups, stage),
            })
            del model
        all_rows = [r["all_prompt"] for r in rows]
        out["arms"][arm] = {
            "memory_sha256": str(__import__("hashlib").sha256(
                (cell / f"{arm}_memory.pt").read_bytes()).hexdigest()),
            "student_seeds": seeds,
            "student_steps": int(U.protocol_v3.student_steps(len(memory["x"]))),
            "wall_seconds_total": float(time.perf_counter() - started),
            "cumulative_f1_mean": float(np.mean([r["cumulative_f1"] for r in all_rows])),
            "old_f1_mean": float(np.mean([r["old_f1"] for r in all_rows
                                           if r["old_f1"] is not None])),
            "new_f1_mean": float(np.mean([r["new_f1"] for r in all_rows
                                           if r["new_f1"] is not None])),
            "per_student": rows,
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--minilm", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--datasets", nargs="+", default=["cora", "citeseer"])
    parser.add_argument("--orders", nargs="+",
                        default=["order19605", "order19606", "order19608"])
    parser.add_argument("--stages", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--arms", nargs="+",
                        default=["c0", "identity", "role_perm"])
    parser.add_argument("--report-seeds", nargs="+", type=int,
                        default=list(range(150, 156)))
    args = parser.parse_args()
    torch.set_num_threads(2)
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "name": "tgc-role-aware-detailed-evaluation-v1",
        "input_root": str(args.input_root),
        "datasets": args.datasets,
        "orders": args.orders,
        "stages": args.stages,
        "arms": args.arms,
        "report_seeds": args.report_seeds,
        "selection": False,
        "report_labels_used_after_construction": True,
        "metric": "all-prompt per-class old/new F1 on report-dev",
    }
    save_json(args.output_root / "protocol.json", manifest)
    data_cache = {}
    for dataset in args.datasets:
        data_cache[dataset] = V1.load_data(
            dataset, args.minilm, args.device, args.cache_root,
        )
    results = []
    for dataset in args.datasets:
        data = data_cache[dataset]
        for order in args.orders:
            groups = set_order_groups(dataset, order)
            for stage in args.stages:
                cell = args.input_root / f"{dataset}_{order}_s{stage}"
                if not (cell / "results.json").exists():
                    raise FileNotFoundError(cell / "results.json")
                cell_result = evaluate_cell(
                    cell, data, groups, stage, args.report_seeds,
                    args.device, args.arms,
                )
                results.append(cell_result)
                save_json(args.output_root / f"{dataset}_{order}_s{stage}.json",
                          cell_result)
                print(json.dumps({"event": "detailed_complete",
                                  "cell": cell.name}), flush=True)
    save_json(args.output_root / "results.json", {"protocol": manifest,
                                                    "cells": results})


if __name__ == "__main__":
    main()
