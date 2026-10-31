"""Large-graph smoke runner for the current Unified-Q constructor.

This is deliberately a feasibility runner, not a new algorithm.  It reuses
``unified_q_multimodal.optimize_stage`` and changes only the large TAGBench
loader, base+4 stage manifest, and sparse full-graph prompt-only evaluation.
No historical raw features or text are passed to the constructor.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

import external_continual_mechanisms as M
import graphtext_prompt_student as P
import run_unified_v3_continual as U
import unified_q_multimodal as UQ


def read(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("protocol", payload)


def save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True), encoding="utf-8")


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
    text_path = cache / "text_emb_pool_only.pt"
    text_emb = torch.load(text_path, map_location="cpu", weights_only=False).float()
    if tuple(text_emb.shape) != (len(data["x"]), 384):
        raise ValueError(f"unexpected text cache shape: {tuple(text_emb.shape)}")
    excluded = ~data["pool_mask"]
    if bool(excluded.any()) and float(text_emb[excluded].abs().max()) != 0.0:
        raise AssertionError("text cache contains non-pool rows")
    data["text_emb"] = text_emb
    data["stage_classes"] = [[int(v) for v in group] for group in manifest["groups"]]
    data["prompt_emb"] = data["encoder"].encode(data["prompts"])
    data["real_adj"] = P.normalized_adjacency(
        data["edge_index"], len(data["x"]), device)
    data["x_device"] = data["x"].float().to(device)
    return data


def stage_ids(data: dict) -> list[torch.Tensor]:
    output = []
    for group in data["stage_classes"]:
        classes = torch.tensor(group, dtype=torch.long)
        ids = torch.where(data["pool_mask"] & torch.isin(data["y"], classes))[0]
        output.append(torch.sort(ids).values)
    return output


def delta_view(data: dict, ids: torch.Tensor, stage: int) -> dict:
    ids = ids.long().cpu()
    return {
        "ids": ids,
        "stage": int(stage),
        "x": data["x"][ids].float(),
        "y": data["y"][ids].long(),
        "text_emb": data["text_emb"][ids].float(),
        "texts": [data["texts"][int(i)] for i in ids.tolist()],
        "edge_index": induced_edges(data["edge_index"], ids, len(data["x"])),
        "boundary_edge_index": torch.empty((2, 0), dtype=torch.long),
    }


def score(memory: dict, data: dict, stage: int, seeds: list[int],
          device: str, steps: int, batch: int, trainer=None) -> dict:
    prompt = data["prompt_emb"].float().to(device)
    y = data["y"].long().cpu()
    report = data["report_dev_mask"].bool().cpu()
    seen = [int(c) for group in data["stage_classes"][:stage + 1]
            for c in group]
    seen_tensor = torch.tensor(seen, dtype=torch.long)
    mask = report & torch.isin(y, seen_tensor)
    scores = []
    for seed in seeds:
        train = P.train_prompt_student if trainer is None else trainer
        model = train(
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


def append_edge_free(parent: dict | None, block: dict) -> dict:
    if parent is None:
        return block
    return {
        "x": torch.cat((parent["x"], block["x"])),
        "text_emb": torch.cat((parent["text_emb"], block["text_emb"])),
        "labels": torch.cat((parent["labels"], block["labels"])),
        "mass": torch.cat((parent["mass"], block["mass"])),
        "depth": torch.cat((parent["depth"], block["depth"])),
        "stage": torch.cat((parent["stage"], block["stage"])),
        "texts": list(parent["texts"]) + list(block["texts"]),
        "edge_index": torch.empty((2, 0), dtype=torch.long),
    }


def run(args: argparse.Namespace) -> dict:
    manifest = read(args.protocol)
    data = load_large(args.dataset, args.minilm, args.text_cache,
                      args.device, manifest)
    ids = stage_ids(data)
    budgets = [int(v) for v in manifest["budgets"]]
    if len(ids) != len(budgets):
        raise AssertionError("stage groups and budgets have different lengths")
    cfg = UQ.Config(
        steps=int(args.q_steps),
        proxy_seeds=(2711,),
        materialize_top_k=1,
        straight_through_text=True,
        hard_topology_forward=True,
        seed=271006,
    )
    cat_cfg = M.Config(seed=271006, cat_steps=int(args.cat_steps),
                       text_sources=5, text_mode="nearest")
    parents = {arm: None for arm in args.arms}
    rows = []
    for stage, stage_ids_value in enumerate(ids):
        delta = delta_view(data, stage_ids_value, stage)
        for arm in args.arms:
            parent = parents[arm]
            old_rows = 0 if parent is None else len(parent["x"])
            new_rows = budgets[stage] - old_rows
            if new_rows <= 0 or new_rows > len(stage_ids_value):
                raise AssertionError(f"invalid new rows for {arm} at stage {stage}: {new_rows}")
            labels = M.row_labels(M.ratio_allocation(delta["y"], new_rows))
            started = time.perf_counter()
            if arm == "cat_x":
                block, audit = M.cat_block(
                    delta, new_rows, data["encoder"], cat_cfg, args.device)
                block["labels"] = labels.clone()
                memory = append_edge_free(parent, block)
            elif arm == "unified_q":
                block, audit = UQ.optimize_stage(
                    delta, parent, labels, data["encoder"], data["prompt_emb"],
                    cfg, args.device, use_boundary=False, use_topology=True)
                memory = UQ.append_memory(parent, block)
            else:
                raise ValueError(arm)
            construction_seconds = time.perf_counter() - started
            print(json.dumps({"event": "constructed", "dataset": args.dataset,
                              "stage": stage, "arm": arm,
                              "raw_rows": len(delta["x"]),
                              "memory_rows": len(memory["x"]),
                              "construction_seconds": construction_seconds}), flush=True)
            evaluated = score(memory, data, stage, args.student_seeds,
                              args.device, args.student_steps, args.student_batch)
            print(json.dumps({"event": "evaluated", "dataset": args.dataset,
                              "stage": stage, "arm": arm,
                              "f1": evaluated["final_f1"],
                              "accuracy": evaluated["accuracy"]}), flush=True)
            rows.append({"arm": arm, "stage": stage, "raw_rows": len(delta["x"]),
                         "new_rows": int(new_rows), "memory_rows": len(memory["x"]),
                         "construction_seconds": construction_seconds,
                         "score": evaluated, "audit": audit})
            parents[arm] = memory
    result = {
        "dataset": args.dataset,
        "protocol": manifest,
        "method": "Unified-Q",
        "historical_raw_reads": 0,
        "construction": "current delta + parent synthetic memory; no boundary sketch",
        "config": {"q_steps": int(args.q_steps),
                   "student_steps": int(args.student_steps),
                   "student_batch": int(args.student_batch),
                   "student_seeds": [int(v) for v in args.student_seeds]},
        "rows": rows,
    }
    save(args.output / f"{args.dataset}_unified_q_smoke.json", result)
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
    parser.add_argument("--q-steps", type=int, default=5)
    parser.add_argument("--cat-steps", type=int, default=3000)
    parser.add_argument("--student-steps", type=int, default=128)
    parser.add_argument("--student-batch", type=int, default=24)
    parser.add_argument("--student-seeds", nargs="+", type=int, default=[150, 151])
    parser.add_argument("--arms", nargs="+", choices=["cat_x", "unified_q"],
                        default=["cat_x", "unified_q"])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    run(args)


if __name__ == "__main__":
    main()
