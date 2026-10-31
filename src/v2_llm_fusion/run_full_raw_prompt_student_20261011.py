"""Full-raw upper reference under the current Prompt GraphText Student.

This is an offline reference, not a legal continual constructor.  At each
stage it gives the Student every cumulative construction-pool raw node and
the induced raw graph, while keeping the stage order, masks, optimizer
budget, batch size, and prompt-only inference identical to the 5% runs.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

import graphtext_prompt_student as P
import run_unified_v3_continual as U


def save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True), encoding="utf-8")


def read(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("protocol", payload)


def induced_edges(edge_index: torch.Tensor, ids: torch.Tensor,
                  total_nodes: int) -> torch.Tensor:
    ids = ids.long().cpu()
    remap = torch.full((int(total_nodes),), -1, dtype=torch.long)
    remap[ids] = torch.arange(len(ids), dtype=torch.long)
    edges = edge_index.long().cpu()
    if edges.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long)
    valid = (remap[edges[0]] >= 0) & (remap[edges[1]] >= 0)
    return torch.stack((remap[edges[0, valid]], remap[edges[1, valid]]))


def load_large(dataset: str, minilm: str, cache: Path,
               device: str, manifest: dict) -> dict:
    data = U.load_unified(
        dataset, U.DATA_CFG[dataset]["path"], minilm, device,
        n_stages=len(manifest["groups"]), split_mode="base_incremental",
        base_fraction=0.4,
        report_dev_mod=int(manifest.get("report_dev_mod", 3)),
    )
    cache_path = cache / "text_emb_pool_only.pt" if cache.is_dir() else cache
    text_emb = torch.load(cache_path, map_location="cpu", weights_only=False).float()
    if tuple(text_emb.shape) != (len(data["x"]), 384):
        raise ValueError(f"unexpected text cache shape: {tuple(text_emb.shape)}")
    excluded = ~data["pool_mask"]
    if bool(excluded.any()) and float(text_emb[excluded].abs().max()) != 0.0:
        raise AssertionError("text cache contains non-pool rows")
    data["text_emb"] = text_emb
    data["stage_classes"] = [[int(v) for v in group]
                              for group in manifest["groups"]]
    data["prompt_emb"] = data["encoder"].encode(data["prompts"])
    data["real_adj"] = P.normalized_adjacency(
        data["edge_index"], len(data["x"]), device)
    data["x_device"] = data["x"].float().to(device)
    return data


def get_stage_ids(data: dict) -> list[torch.Tensor]:
    output = []
    for group in data["stage_classes"]:
        classes = torch.tensor(group, dtype=torch.long)
        ids = torch.where(data["pool_mask"] & torch.isin(data["y"], classes))[0]
        output.append(torch.sort(ids).values)
    return output


def score_memory(memory: dict, data: dict, stage: int, seeds: list[int],
                 device: str, steps: int, batch: int) -> dict:
    prompt = data["prompt_emb"].float().to(device)
    y = data["y"].long().cpu()
    report = data["report_dev_mask"].bool().cpu()
    seen = [int(c) for group in data["stage_classes"][:stage + 1]
            for c in group]
    mask = report & torch.isin(y, torch.tensor(seen, dtype=torch.long))
    scores = []
    for seed in seeds:
        model = P.train_prompt_student(
            memory, prompt, int(seed), device, int(steps),
            lr=0.001, batch=int(batch))
        with torch.no_grad():
            logits = model.logits(data["x_device"], data["real_adj"], prompt)
            pred = logits.argmax(1).detach().cpu()
        yy = y[mask].numpy()
        pp = pred[mask].numpy()
        macro = 100.0 * f1_score(yy, pp, average="macro",
                                 labels=seen, zero_division=0)
        acc = 100.0 * accuracy_score(yy, pp)
        per_class = f1_score(yy, pp, average=None, labels=seen,
                             zero_division=0)
        class_f1 = {cls: float(value * 100.0)
                    for cls, value in zip(seen, per_class)}
        old = [int(c) for group in data["stage_classes"][:stage]
               for c in group]
        new = [int(c) for c in data["stage_classes"][stage]]
        scores.append({
            "seed": int(seed),
            "final_f1": float(macro),
            "accuracy": float(acc),
            "old_f1": float(np.mean([class_f1.get(c, 0.0) for c in old])) if old else None,
            "new_f1": float(np.mean([class_f1.get(c, 0.0) for c in new])),
        })
        del model
    return {
        "student_steps": int(steps),
        "student_batch": int(batch),
        "student_seeds": [int(v) for v in seeds],
        "final_f1": float(np.mean([v["final_f1"] for v in scores])),
        "final_f1_std": float(np.std([v["final_f1"] for v in scores], ddof=1)) if len(scores) > 1 else 0.0,
        "accuracy": float(np.mean([v["accuracy"] for v in scores])),
        "accuracy_std": float(np.std([v["accuracy"] for v in scores], ddof=1)) if len(scores) > 1 else 0.0,
        "old_f1": (float(np.mean([v["old_f1"] for v in scores]))
                   if scores and scores[0]["old_f1"] is not None else None),
        "new_f1": float(np.mean([v["new_f1"] for v in scores])),
        "per_student": scores,
    }


def raw_memory(data: dict, ids: torch.Tensor, stage: int,
               edge_free: bool = False) -> dict:
    ids = ids.long().cpu()
    if not bool(data["pool_mask"][ids].all()):
        raise AssertionError("full-raw reference attempted to read outside pool")
    return {
        "x": data["x"][ids].float(),
        "text_emb": data["text_emb"][ids].float(),
        "labels": data["y"][ids].long(),
        "mass": torch.ones(len(ids), dtype=torch.float32),
        "depth": torch.zeros(len(ids), dtype=torch.long),
        "stage": torch.full((len(ids),), int(stage), dtype=torch.long),
        "texts": [data["texts"][int(i)] for i in ids.tolist()],
        "edge_index": (torch.empty((2, 0), dtype=torch.long)
                       if edge_free else
                       induced_edges(data["edge_index"], ids, len(data["x"]))),
    }


def run(args: argparse.Namespace) -> dict:
    manifest = read(args.protocol)
    data = load_large(args.dataset, args.minilm, args.text_cache,
                      args.device, manifest)
    stage_ids = get_stage_ids(data)
    budgets = [int(v) for v in manifest["budgets"]]
    if len(stage_ids) != len(budgets):
        raise AssertionError("stage groups and budgets have different lengths")

    rows = []
    stage_range = [len(stage_ids) - 1] if args.only_final else range(len(stage_ids))
    for stage in stage_range:
        source = torch.cat(stage_ids[:stage + 1])
        memory = raw_memory(data, source, stage, edge_free=bool(args.edge_free))
        started = time.perf_counter()
        score = score_memory(memory, data, stage, args.student_seeds,
                             args.device, args.student_steps,
                             args.student_batch)
        elapsed = time.perf_counter() - started
        row = {
            "stage": int(stage),
            "raw_rows": int(len(source)),
            "budget_reference": int(budgets[stage]),
            "compression_ratio_vs_raw": float(budgets[stage] / len(source)),
            "score": score,
            "historical_raw_access": True,
            "rewrites_previous_memory": True,
            "induced_edge_count": int(memory["edge_index"].shape[1]),
            "edge_free": bool(args.edge_free),
            "student_seconds": float(elapsed),
        }
        rows.append(row)
        print(json.dumps({
            "event": "evaluated",
            "dataset": args.dataset,
            "stage": int(stage),
            "raw_rows": int(len(source)),
            "budget": int(budgets[stage]),
            "f1": score["final_f1"],
            "accuracy": score["accuracy"],
        }), flush=True)

    result = {
        "name": "full-raw-prompt-student-reference-2026-10-11",
        "dataset": args.dataset,
        "protocol": manifest,
        "student": P.model_config(),
        "student_steps": int(args.student_steps),
        "student_batch": int(args.student_batch),
        "student_seeds": [int(v) for v in args.student_seeds],
        "construction": ("all cumulative legal raw pool nodes with self-loop-only memory"
                         if args.edge_free else
                         "all cumulative legal raw pool nodes with induced raw edges"),
        "edge_free": bool(args.edge_free),
        "historical_raw_access": True,
        "rewrites_previous_memory": True,
        "test_accessed": False,
        "rows": rows,
    }
    save(args.output / f"{args.dataset}_full_raw_prompt_student.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True,
                        choices=["wikics", "bookhis", "sportsfit", "arxiv"])
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--text-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minilm", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--student-steps", type=int, default=2048)
    parser.add_argument("--student-batch", type=int, default=24)
    parser.add_argument("--student-seeds", nargs="+", type=int,
                        default=[150, 151, 152])
    parser.add_argument("--only-final", action="store_true",
                        help="evaluate only the final cumulative raw support")
    parser.add_argument("--edge-free", action="store_true",
                        help="train with self-loop-only memory, matching edge-free condensation")
    args = parser.parse_args()
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    run(args)


if __name__ == "__main__":
    main()
