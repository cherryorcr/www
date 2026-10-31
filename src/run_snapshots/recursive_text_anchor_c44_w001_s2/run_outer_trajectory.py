"""Fixed-outer-step student response curves for continual TAG diagnostics.

The condensation selector is deliberately disabled.  ASM snapshots are saved
at a pre-registered set of outer steps and every snapshot is retrained by
independent students.  The development checkpoint and the final training
endpoint are both reported so a single validation student cannot choose the
continual artifact.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from native_core import DATA_KEYS, clean_data
from protocol_data import concat_graphs
from run_native_experiments import Experiment, dump, file_sha


def load_graph(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=True)


def build_args(base_config: dict, output: str, prepared: str, device: str | None,
               seed: int, model_seeds: list[int], check_every: int,
               outer_steps: int, outer_eval_every: int, text_anchor_weight: float = 0.0,
               text_anchor_scope: str = "all") -> argparse.Namespace:
    config = dict(base_config)
    config.update({
        "output": output,
        "prepared": prepared,
        "seed": int(seed),
        "model_seeds": model_seeds,
        "device": device or config.get("device", "cuda:0"),
        "check_every": int(check_every),
        "outer_steps": int(outer_steps),
        "outer_eval_every": int(outer_eval_every),
        "text_anchor_weight": float(text_anchor_weight),
        "text_anchor_scope": text_anchor_scope,
        "confirm": False,
        "phase": "continual",
        "stages": 3,
        "arms": [],
        "diagnostic_mode": "fixed_outer_trajectory",
    })
    return argparse.Namespace(**config)


def assert_memory(memory: dict, manifest_sha: str, max_stage: int) -> dict:
    metadata = memory.get("metadata", {})
    if metadata.get("source_manifest_sha256") != manifest_sha:
        raise ValueError("memory belongs to a different prepared manifest")
    if metadata.get("training_labels_used"):
        raise ValueError("memory metadata claims labels were used")
    if int(memory["stage"].max()) > max_stage:
        raise ValueError("memory contains a future stage")
    return clean_data(memory, "cpu") | {"metadata": metadata}


def selected_metrics(exp: Experiment, model, stage: int) -> dict:
    metric = exp.metric(model, stage, "checkpoint")
    return {
        "macro_f1": metric["macro_f1"],
        "accuracy": metric["accuracy"],
        "per_stage": metric["per_stage"],
        "old_mean_f1": metric["old_mean_f1"],
        "current_f1": metric["current_f1"],
    }


def evaluate_snapshots(exp: Experiment, root: Path, name: str, stage: int,
                       outer_steps: list[int], model_seeds: list[int]) -> dict:
    trajectory = []
    for outer_step in outer_steps:
        path = root / f"{name}_s{stage}_outer_{outer_step}.pt"
        if not path.exists():
            raise FileNotFoundError(f"missing fixed outer-step snapshot: {path}")
        artifact = load_graph(path)
        runs = []
        for model_seed in model_seeds:
            model, info = exp.fit(artifact, stage, int(model_seed), save=False)
            selected = selected_metrics(exp, model, stage)
            runs.append({
                "seed": int(model_seed),
                "selected_step": info["selected_step"],
                "best_checkpoint_f1": info["best_checkpoint_f1"],
                "selected_checkpoint": selected,
                "endpoint_checkpoint": info["endpoint_checkpoint"],
                "step0_checkpoint": info["step0_checkpoint"],
                "best_minus_step0": info["best_minus_step0"],
            })
            del model
        selected_values = [run["selected_checkpoint"]["macro_f1"] for run in runs]
        endpoint_values = [run["endpoint_checkpoint"]["macro_f1"] for run in runs]
        first = runs[0]["selected_checkpoint"]
        trajectory.append({
            "outer_step": int(outer_step),
            "artifact": path.name,
            "artifact_sha256": file_sha(path),
            "x_norm_mean": artifact.get("metadata", {}).get("x_norm_mean"),
            "x_norm_max": artifact.get("metadata", {}).get("x_norm_max"),
            "x_l2": artifact.get("metadata", {}).get("x_l2"),
            "selected_student_mean_macro_f1": float(torch.tensor(selected_values).mean()),
            "selected_student_std_macro_f1": float(torch.tensor(selected_values).std(unbiased=False)),
            "endpoint_student_mean_macro_f1": float(torch.tensor(endpoint_values).mean()),
            "endpoint_student_std_macro_f1": float(torch.tensor(endpoint_values).std(unbiased=False)),
            "selected_student_per_stage_mean": [
                float(torch.tensor([run["selected_checkpoint"]["per_stage"][j]["macro_f1"] for run in runs]).mean())
                for j in range(stage + 1)
            ],
            "selected_student_old_mean_f1": first["old_mean_f1"] if len(runs) == 1 else float(torch.tensor([
                run["selected_checkpoint"]["old_mean_f1"] for run in runs
                if run["selected_checkpoint"]["old_mean_f1"] is not None
            ]).mean()) if stage else None,
            "selected_student_current_f1_mean": float(torch.tensor([
                run["selected_checkpoint"]["current_f1"] for run in runs
            ]).mean()),
            "student_runs": runs,
        })
    return {"outer_steps": outer_steps, "trajectory": trajectory}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", required=True,
                        help="completed recursive run_config.json")
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--old-artifact", required=True,
                        help="fixed stage-1 memory artifact")
    parser.add_argument("--output", required=True)
    parser.add_argument("--stage", type=int, default=2)
    parser.add_argument("--variant", choices=("joint", "protected", "new_only"), default="joint")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--model-seeds", type=int, nargs="+", default=[100, 101])
    parser.add_argument("--check-every", type=int, default=32)
    parser.add_argument("--outer-steps", type=int, default=100)
    parser.add_argument("--outer-eval-every", type=int, default=20)
    parser.add_argument("--text-anchor-weight", type=float, default=0.0)
    parser.add_argument("--text-anchor-scope", choices=("all", "optimized"), default="all")
    args = parser.parse_args()

    base_path = Path(args.base_config)
    base = json.loads(base_path.read_text(encoding="utf-8"))
    seed = int(base.get("seed", 44) if args.seed is None else args.seed)
    exp = Experiment(build_args(base, args.output, args.prepared, args.device, seed,
                                args.model_seeds, args.check_every, args.outer_steps,
                                args.outer_eval_every, args.text_anchor_weight,
                                args.text_anchor_scope))
    stages = [exp.load_stage(i) for i in range(max(3, args.stage + 1))]
    manifest_sha = file_sha(exp.prepared / "manifest.json")
    old_path = Path(args.old_artifact)
    old = assert_memory(load_graph(old_path), manifest_sha, args.stage - 1)
    if int(old["stage"].max()) != args.stage - 1:
        raise ValueError("old artifact must end at stage-1")
    delta = stages[args.stage]
    total_budget = math.floor(exp.args.rate * sum(exp.counts[: args.stage + 1]))
    new_k = total_budget - len(old["x"])
    if new_k < 2:
        raise ValueError("new block budget is too small")

    delta_teacher = exp.teacher(delta, args.stage, f"trajectory_{args.variant}_delta")
    compact = exp.initializer(delta, new_k, args.stage, delta_teacher)
    source = concat_graphs((old, delta))
    initial = concat_graphs((old, compact))
    name = f"trajectory_{args.variant}"
    extra = {"text_anchor_weight": args.text_anchor_weight,
             "text_anchor_scope": args.text_anchor_scope}
    if args.variant == "protected":
        extra["optimize_rows"] = initial["stage"] == args.stage
    elif args.variant == "new_only":
        source = delta
        initial = compact

    teacher = exp.teacher(source, args.stage, f"trajectory_{args.variant}_joint")
    started = time.time()
    exp.optimize(source, initial, teacher, args.stage, name,
                 extra_config=extra, select_validation=False, retain_outer_steps=True)
    outer_steps = list(range(0, args.outer_steps + 1, args.outer_eval_every))
    curve = evaluate_snapshots(exp, exp.output, name, args.stage, outer_steps, args.model_seeds)
    curve.update({"text_anchor_weight": args.text_anchor_weight,
                  "text_anchor_scope": args.text_anchor_scope})
    dump(exp.output / "trajectory.json", curve)
    dump(exp.output / "diagnostic_inputs.json", {
        "mode": "fixed_outer_trajectory",
        "variant": args.variant,
        "confirmation_accessed": False,
        "base_config": str(base_path),
        "base_config_sha256": file_sha(base_path),
        "old_artifact": str(old_path),
        "old_artifact_sha256": file_sha(old_path),
        "prepared_manifest_sha256": manifest_sha,
        "stage": args.stage,
        "seed": seed,
        "model_seeds": args.model_seeds,
        "outer_steps": outer_steps,
        "text_anchor_weight": args.text_anchor_weight,
        "text_anchor_scope": args.text_anchor_scope,
        "outer_selector": "disabled; fixed pre-registered snapshots",
        "student_selection": "checkpoint-best and endpoint both recorded",
        "training_labels_used": False,
        "elapsed_seconds": time.time() - started,
    })
    dump(exp.output / "completed.json", {"status": "complete", "confirmation_accessed": False,
                                           "elapsed_seconds": time.time() - started})


if __name__ == "__main__":
    main()
