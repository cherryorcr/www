"""Evaluate all cross-edge topology arms for one continual cell."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

import run_content_gap_attribution as CAG
import run_unified_v3_continual as U


def load(path: Path):
    return torch.load(path, map_location="cpu", weights_only=False)


def fit(memory: dict, data: dict, stage: int, seeds: list[int], device: str,
        checkpoint_dir: Path | None = None) -> dict:
    rows = []
    training_seconds = 0.0
    started = time.perf_counter()
    for seed in seeds:
        steps = U.protocol_v3.student_steps(len(memory["x"]))
        if str(device).startswith("cuda"):
            torch.cuda.synchronize(device)
        fit_started = time.perf_counter()
        model = U.G.train_student(memory, int(seed), device, steps,
                                  batch=U.protocol_v3.STUDENT.batch)
        if str(device).startswith("cuda"):
            torch.cuda.synchronize(device)
        fit_seconds = time.perf_counter() - fit_started
        training_seconds += fit_seconds
        score = U.G.eval_stagewise(
            model, data["x_device"], data["real_adj"], data["prompt_emb"],
            data["y"], data["report_dev_mask"], stage)
        row = {"seed": int(seed), "steps": int(steps),
               "training_seconds": float(fit_seconds), **score}
        rows.append(row)
        if checkpoint_dir is not None:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            path = checkpoint_dir / f"seed{seed}.pt"
            if path.exists():
                raise FileExistsError(f"immutable checkpoint exists: {path}")
            torch.save({"state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "seed": int(seed), "stage": int(stage), "steps": int(steps),
                        "metrics": score}, path)
            print(json.dumps({"event": "student_saved", "stage": stage,
                              "seed": seed, "checkpoint": str(path),
                              "f1": score["cumulative_macro_f1"]}), flush=True)
        del model
    old_values = [r["old_mean_f1"] for r in rows
                  if r.get("old_mean_f1") is not None]
    return {
        "rows": rows,
        "mean_final_f1": float(np.mean([r["cumulative_macro_f1"] for r in rows])),
        "std_final_f1": float(np.std([r["cumulative_macro_f1"] for r in rows])),
        "mean_accuracy": float(np.mean([r["cumulative_accuracy"] for r in rows])),
        "mean_new_f1": float(np.mean([r["current_f1"] for r in rows])),
        "mean_old_f1": float(np.mean(old_values)) if old_values else None,
        "stage_mean_f1": float(np.mean([
            np.mean([item["macro_f1"] for item in r["per_stage"]])
            for r in rows])),
        "student_seconds": float(time.perf_counter() - started),
        "training_only_seconds": float(training_seconds),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("cora", "citeseer"), required=True)
    parser.add_argument("--order", required=True)
    parser.add_argument("--stage", type=int, choices=(1, 2), required=True)
    parser.add_argument("--construction-root", required=True, type=Path)
    parser.add_argument("--semantic-root", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--text-cache", required=True, type=Path)
    parser.add_argument("--minilm", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=list(range(150, 156)))
    parser.add_argument("--modes", nargs="+",
                        default=["c0", "none", "random", "semantic", "message"])
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"immutable output exists: {args.output}")
    data = U.load_unified(args.dataset, str(args.data), args.minilm,
                          args.device, n_stages=3, split_mode="canonical")
    text_emb = load(args.text_cache)
    data["text_emb"] = text_emb
    data["prompt_emb"] = data["encoder"].encode(data["prompts"])
    data["real_adj"] = U.G.norm_adj(data["edge_index"], len(data["x"]), args.device)
    data["x_device"] = data["x"].float().to(args.device)
    groups = CAG.stage_groups(args.dataset, 3, args.order)
    CAG.set_stream(data, groups, 0.05)
    cell = args.construction_root / f"{args.dataset}_{args.order}_s{args.stage}"
    base_path = (args.semantic_root /
                 f"{args.dataset}_{args.order}_s{args.stage}" /
                 "semantic_perm_memory.pt")
    memories = {"c0": load(cell / "c0" / "tgr_message_memory.pt")}
    for mode in args.modes:
        if mode != "c0":
            memories[mode] = load(cell / mode / "tgr_message_memory.pt")
    result = {"protocol": {
        "name": "tgr-current-delta-cross-edge-eval-v1",
        "dataset": args.dataset, "order": args.order, "stage": int(args.stage),
        "modes": list(args.modes), "seeds": list(args.seeds),
        "student": "canonical two-layer GCN + bidirectional InfoNCE",
        "report_mask": "report-dev", "selection_by_report": False,
        "base_semantic_recursive": str(base_path),
    }, "arms": {}}
    result["arms"]["c0"] = fit(memories["c0"], data, args.stage,
                               args.seeds, args.device)
    base = result["arms"]["c0"]
    for mode in args.modes:
        if mode == "c0":
            continue
        if mode == "none":
            # Strict no-op construction is exactly C0.  Reusing the fitted
            # result avoids duplicating stochastic Student training while
            # retaining an explicit control arm in the report.
            value = json.loads(json.dumps(base))
            value["exact_c0_control"] = True
            value["delta_vs_c0"] = {
                "final_f1_pp": 0.0, "accuracy_pp": 0.0,
                "new_f1_pp": 0.0, "old_f1_pp": 0.0,
                "stage_mean_f1_pp": 0.0,
            }
            result["arms"][mode] = value
            continue
        value = fit(memories[mode], data, args.stage, args.seeds, args.device)
        value["delta_vs_c0"] = {
            # The canonical evaluator already returns percentages (e.g. 52.08),
            # so a difference of 0.72 is 0.72 percentage points.  Do not
            # multiply by 100 a second time.
            "final_f1_pp": value["mean_final_f1"] - base["mean_final_f1"],
            "accuracy_pp": value["mean_accuracy"] - base["mean_accuracy"],
            "new_f1_pp": value["mean_new_f1"] - base["mean_new_f1"],
            "old_f1_pp": value["mean_old_f1"] - base["mean_old_f1"],
            "stage_mean_f1_pp": value["stage_mean_f1"] - base["stage_mean_f1"],
        }
        result["arms"][mode] = value
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps({"event": "cross_edge_eval_complete", "output": str(args.output),
                      "deltas": {k: v.get("delta_vs_c0") for k, v in result["arms"].items()
                                  if k != "c0"}}), flush=True)


if __name__ == "__main__":
    main()
