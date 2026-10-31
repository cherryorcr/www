"""Evaluate sealed C0 and one GC-SR memory on the canonical Student."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

import run_p0_random_synthetic as P0
import run_content_gap_attribution as CAG
import run_unified_v3_continual as U


def load(path: Path):
    return torch.load(path, map_location="cpu", weights_only=False)


def fit(memory: dict, data: dict, stage: int, seeds: list[int], device: str) -> dict:
    rows = []
    started = time.perf_counter()
    for seed in seeds:
        steps = U.protocol_v3.student_steps(len(memory["x"]))
        model = U.G.train_student(memory, int(seed), device, steps,
                                  batch=U.protocol_v3.STUDENT.batch)
        score = U.G.eval_stagewise(model, data["x_device"], data["real_adj"],
                                   data["prompt_emb"], data["y"],
                                   data["report_dev_mask"], stage)
        rows.append({"seed": int(seed), "steps": int(steps), **score})
        del model
    return {
        "rows": rows,
        "mean_final_f1": float(np.mean([r["cumulative_macro_f1"] for r in rows])),
        "std_final_f1": float(np.std([r["cumulative_macro_f1"] for r in rows])),
        "mean_accuracy": float(np.mean([r["cumulative_accuracy"] for r in rows])),
        "mean_new_f1": float(np.mean([r["current_f1"] for r in rows])),
        "mean_old_f1": float(np.mean([r["old_mean_f1"] for r in rows
                                      if r.get("old_mean_f1") is not None])),
        "per_stage_mean": [float(np.mean([r["per_stage"][s]["macro_f1"]
                                           for r in rows]))
                           for s in range(stage + 1)],
        "student_seconds": float(time.perf_counter() - started),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("cora", "citeseer"), required=True)
    parser.add_argument("--order", default="canonical")
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--data", type=Path, default=None,
                        help="optional local raw dataset path")
    parser.add_argument("--text-cache", type=Path, default=None,
                        help="optional pool-only MiniLM node embedding cache")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--minilm", required=True)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=list(range(150, 156)))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"immutable output exists: {args.output}")
    if args.data is None:
        data = P0.load_data(
            args.dataset, args.minilm, args.device,
            args.cache_root / f"{args.dataset}_r005" / "text_emb_pool_only.pt")
    else:
        data = U.load_unified(args.dataset, str(args.data), args.minilm,
                              args.device, n_stages=3, split_mode="canonical")
        cache = args.text_cache
        if cache is None:
            cache = args.cache_root / f"{args.dataset}_r005" / "text_emb_pool_only.pt"
        text_emb = torch.load(cache, map_location="cpu", weights_only=False)
        if tuple(text_emb.shape) != (len(data["x"]), 384):
            raise ValueError(f"invalid text cache shape: {tuple(text_emb.shape)}")
        data["text_emb"] = text_emb
        data["prompt_emb"] = data["encoder"].encode(data["prompts"])
        data["real_adj"] = U.G.norm_adj(data["edge_index"], len(data["x"]), args.device)
        data["x_device"] = data["x"].float().to(args.device)
    groups = CAG.stage_groups(args.dataset, 3, args.order)
    CAG.set_stream(data, groups, 0.05)
    base, candidate = load(args.base), load(args.candidate)
    result = {
        "protocol": {
            "name": "gc-sr-stage-eval-v1", "dataset": args.dataset,
            "order": args.order, "stage": args.stage, "seeds": args.seeds,
            "student": "canonical two-layer GCN + bidirectional InfoNCE",
            "report_mask": "report-dev", "selection_by_report": False,
        },
        "base": fit(base, data, args.stage, args.seeds, args.device),
        "candidate": fit(candidate, data, args.stage, args.seeds, args.device),
    }
    result["delta"] = {
        "final_f1_pp": result["candidate"]["mean_final_f1"] - result["base"]["mean_final_f1"],
        "accuracy_pp": result["candidate"]["mean_accuracy"] - result["base"]["mean_accuracy"],
        "new_f1_pp": result["candidate"]["mean_new_f1"] - result["base"]["mean_new_f1"],
        "old_f1_pp": result["candidate"]["mean_old_f1"] - result["base"]["mean_old_f1"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps({"event": "gc_sr_eval_complete", "output": str(args.output),
                      "delta": result["delta"]}), flush=True)


if __name__ == "__main__":
    main()
