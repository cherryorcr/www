"""Read-only Student trajectory response diagnostic for a memory pair.

The diagnostic compares a frozen control memory with one candidate memory under
the same Student initialization, Adam optimizer, mini-batch schedule, and
checkpoint horizons.  At each checkpoint it measures the sealed construction
support response (old/new InfoNCE) and pairs that response with the already
computed report-dev F1 delta for the same Student seed.

This script deliberately does not update a memory, select a candidate, or read
report labels during training.  Report F1 is loaded only after the response
trajectory has been written, and is used for offline correlation analysis.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

import direct_student_outer as DSO


def load(path: Path):
    return torch.load(path, map_location="cpu", weights_only=False)


def _json_hash(value) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=True).encode()
    return hashlib.sha256(payload).hexdigest()


def _rankdata(values: Iterable[float]) -> np.ndarray:
    """Average-rank values without requiring scipy in the remote env."""
    values = np.asarray(list(values), dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def _corr(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2 or len(set(left)) < 2 or len(set(right)) < 2:
        return None
    return float(np.corrcoef(np.asarray(left), np.asarray(right))[0, 1])


def _spearman(left: list[float], right: list[float]) -> float | None:
    return _corr(_rankdata(left).tolist(), _rankdata(right).tolist())


def _same_non_x(control: dict, candidate: dict) -> dict:
    if len(control["x"]) != len(candidate["x"]):
        raise ValueError("control and candidate row counts differ")
    checks = {}
    for key in ("text_emb", "texts", "edge_index", "mass", "depth", "stage"):
        if key not in control or key not in candidate:
            continue
        left, right = control[key], candidate[key]
        if isinstance(left, torch.Tensor):
            equal = torch.equal(left, right)
        else:
            equal = left == right
        checks[key] = bool(equal)
        if not equal:
            raise ValueError(f"candidate changes non-X field: {key}")
    return checks


def _batch_schedule(n: int, steps: int, batch: int, seed: int) -> list[list[int]]:
    generator = torch.Generator(device="cpu").manual_seed(int(seed) * 7 + 1)
    schedule = []
    for _ in range(int(steps)):
        rows = torch.randperm(n, generator=generator)[:min(int(batch), n)]
        schedule.append([int(value) for value in rows.tolist()])
    return schedule


def _support_measure(model: torch.nn.Module, support: dict,
                     support_adj: torch.Tensor, plan: dict,
                     cfg: DSO.Config, device: str) -> dict:
    params = dict(model.named_parameters())
    with torch.no_grad():
        losses = DSO.support_losses(
            params, support, support_adj, plan, "validation", cfg)
    output = {key: float(value.detach().cpu()) for key, value in losses.items()}
    if "old" not in output or "new" not in output or "total" not in output:
        raise ValueError("sealed support plan did not produce old/new/total loss")
    return output


def _trajectory(memory: dict, support: dict, support_adj: torch.Tensor,
                plan: dict, seed: int, checkpoints: list[int], batch: int,
                lr: float, device: str) -> dict:
    if checkpoints != sorted(set(checkpoints)) or checkpoints[0] != 0:
        raise ValueError("checkpoints must be sorted, unique, and start at zero")
    max_steps = checkpoints[-1]
    model = DSO._student(seed, memory["x"].shape[1],
                         memory["text_emb"].shape[1], device)
    model.train()
    x = memory["x"].float().to(device)
    text = memory["text_emb"].float().to(device)
    adjacency = DSO.fixed_norm_adj(memory["edge_index"], len(x), device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))
    schedule = _batch_schedule(len(x), max_steps, batch, seed)
    records = []
    checkpoint_set = set(checkpoints)
    for step in range(max_steps + 1):
        if step in checkpoint_set:
            response = _support_measure(
                model, support, support_adj, plan, DSO.Config(
                    support_batch=batch), device)
            records.append({"step": int(step), **response})
        if step == max_steps:
            break
        rows = torch.tensor(schedule[step], dtype=torch.long, device=device)
        graph = model.encode_graph(x, adjacency)[rows]
        encoded = model.encode_text(text[rows])
        scale = model.logit_scale.exp().clamp(max=100)
        logits = scale * graph @ encoded.T
        target = torch.arange(len(rows), device=device)
        loss = (torch.nn.functional.cross_entropy(logits, target)
                + torch.nn.functional.cross_entropy(logits.T, target)) / 2
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return {
        "seed": int(seed),
        "schedule_sha256": _json_hash(schedule),
        "trajectory": records,
    }


def _f1_pairs(eval_record: dict, seeds: list[int]) -> dict[int, dict]:
    base = {int(row["seed"]): row for row in eval_record["base"]["rows"]}
    candidate = {int(row["seed"]): row for row in eval_record["candidate"]["rows"]}
    result = {}
    for seed in seeds:
        if seed not in base or seed not in candidate:
            raise ValueError(f"evaluation record lacks paired seed {seed}")
        result[seed] = {
            "final_f1_pp": float(candidate[seed]["cumulative_macro_f1"]
                                  - base[seed]["cumulative_macro_f1"]),
            "accuracy_pp": float(candidate[seed]["cumulative_accuracy"]
                                  - base[seed]["cumulative_accuracy"]),
            "new_f1_pp": float(candidate[seed]["current_f1"]
                                - base[seed]["current_f1"]),
            "old_f1_pp": float(candidate[seed]["old_mean_f1"]
                                - base[seed]["old_mean_f1"]),
        }
    return result


def _correlation_summary(observations: list[dict],
                         checkpoints: list[int]) -> list[dict]:
    """Summarize response/F1 alignment for pooled paired observations."""
    correlations = []
    for step in checkpoints:
        rows = [row for row in observations if int(row["step"]) == int(step)]
        item = {"step": int(step), "n": len(rows)}
        for response_key in ("old", "new", "total"):
            for metric in ("final_f1_pp", "new_f1_pp", "old_f1_pp"):
                left = [-float(row[response_key]) for row in rows]
                right = [float(row[metric]) for row in rows]
                item[f"spearman_neg_{response_key}_vs_{metric}"] = _spearman(left, right)
                item[f"pearson_neg_{response_key}_vs_{metric}"] = _corr(left, right)
            signs = [(-float(row[response_key]) >= 0) ==
                     (float(row["final_f1_pp"]) >= 0) for row in rows]
            item[f"sign_agreement_neg_{response_key}_vs_final"] = (
                float(np.mean(signs)) if signs else None)
        correlations.append(item)
    return correlations


def analyze_cell(control_path: Path, candidate_path: Path, support_path: Path,
                 eval_path: Path, output_path: Path, old_rows: int,
                 seeds: list[int], checkpoints: list[int], batch: int,
                 lr: float, plan_seed: int, device: str) -> dict:
    control, candidate, support = (load(control_path), load(candidate_path),
                                   load(support_path))
    DSO.validate_memory(control, "control")
    DSO.validate_memory(candidate, "candidate")
    DSO.validate_memory(support, "support")
    fields = _same_non_x(control, candidate)
    if old_rows < 2 or old_rows >= len(support["x"]):
        raise ValueError("old_rows does not leave a valid support delta")
    cfg = DSO.Config(support_batch=batch, plan_seed=plan_seed)
    plan = DSO.make_support_plan(support, old_rows, batch, plan_seed)
    support_adj = DSO.fixed_norm_adj(support["edge_index"], len(support["x"]), device)
    eval_record = json.loads(eval_path.read_text(encoding="utf-8"))
    f1_pairs = _f1_pairs(eval_record, seeds)
    trajectories = {}
    for seed in seeds:
        control_traj = _trajectory(control, support, support_adj, plan, seed,
                                   checkpoints, batch, lr, device)
        candidate_traj = _trajectory(candidate, support, support_adj, plan, seed,
                                     checkpoints, batch, lr, device)
        if control_traj["schedule_sha256"] != candidate_traj["schedule_sha256"]:
            raise AssertionError("control/candidate batch schedules differ")
        by_step = {}
        for left, right in zip(control_traj["trajectory"], candidate_traj["trajectory"]):
            if left["step"] != right["step"]:
                raise AssertionError("checkpoint mismatch")
            by_step[str(left["step"])] = {
                "control": left,
                "candidate": right,
                "delta": {
                    key: float(right[key] - left[key])
                    for key in ("old", "new", "total")
                },
            }
        trajectories[str(seed)] = {
            "f1_delta": f1_pairs[seed],
            "schedule_sha256": control_traj["schedule_sha256"],
            "checkpoints": by_step,
        }
    pooled = []
    for step in checkpoints:
        key = str(step)
        for seed in seeds:
            response = trajectories[str(seed)]["checkpoints"][key]["delta"]
            f1 = trajectories[str(seed)]["f1_delta"]
            pooled.append({"seed": seed, "step": step, **response, **f1})
    correlations = _correlation_summary(pooled, checkpoints)
    result = {
        "protocol": {
            "name": "student-trajectory-response-v1",
            "memory_comparison": "control vs x-provenance r=0.10b",
            "student": "canonical two-layer GCN + bidirectional InfoNCE",
            "optimizer": "Adam",
            "batch": int(batch), "lr": float(lr),
            "checkpoints": [int(value) for value in checkpoints],
            "seeds": [int(value) for value in seeds],
            "plan_seed": int(plan_seed),
            "support_split": "sealed validation old/new support; no labels",
            "historical_raw_reads": 0,
            "memory_updated": False,
            "report_f1_used_after_trajectory": True,
        },
        "inputs": {
            "control": str(control_path), "candidate": str(candidate_path),
            "support": str(support_path), "evaluation": str(eval_path),
            "control_sha256": DSO.memory_hash(control),
            "candidate_sha256": DSO.memory_hash(candidate),
            "support_sha256": DSO.memory_hash(support),
            "non_x_fields_equal": fields,
            "old_rows": int(old_rows),
        },
        "trajectories": trajectories,
        "correlations_pooled_over_paired_student_seeds": correlations,
        "paired_observations": pooled,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(f"immutable output exists: {output_path}")
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=True,
                                     allow_nan=False), encoding="utf-8")
    return result


def _cell_args(root: Path, cell: str, output_prefix: str) -> tuple[Path, Path, Path, Path, Path]:
    return (
        root / f"control_{cell}" / "memory.pt",
        root / f"provenance_x_r10b_{cell}" / "memory.pt",
        root / "prep" / cell / "support.pt",
        root / f"eval_provenance_x_r10b_{cell}.json",
        root / f"{output_prefix}_{cell}.json",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cell", nargs="+",
                        default=["cora_canonical", "cora_order19102",
                                 "citeseer_canonical", "citeseer_order19102"])
    parser.add_argument("--old-rows", type=int, default=25)
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[150, 151, 152, 153, 154, 155])
    parser.add_argument("--checkpoints", nargs="+", type=int, default=[0, 2, 4, 8])
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--plan-seed", type=int, default=9320)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-prefix", default="trajectory_response_r10b_v2")
    args = parser.parse_args()
    summary = []
    pooled_observations = []
    for cell in args.cell:
        paths = _cell_args(args.root, cell, args.output_prefix)
        result = analyze_cell(*paths, args.old_rows, args.seeds,
                              args.checkpoints, args.batch, args.lr,
                              args.plan_seed, args.device)
        pooled_observations.extend({**row, "cell": cell}
                                   for row in result["paired_observations"])
        summary.append({
            "cell": cell,
            "output": str(paths[-1]),
            "correlations": result["correlations_pooled_over_paired_student_seeds"],
        })
        print(json.dumps({"event": "trajectory_cell_complete", "cell": cell,
                          "output": str(paths[-1])}), flush=True)
    cell_means = []
    for cell in args.cell:
        for step in args.checkpoints:
            rows = [row for row in pooled_observations
                    if row["cell"] == cell and int(row["step"]) == int(step)]
            if not rows:
                continue
            cell_means.append({
                "cell": cell, "step": int(step),
                **{key: float(np.mean([row[key] for row in rows]))
                   for key in ("old", "new", "total", "final_f1_pp",
                               "new_f1_pp", "old_f1_pp")},
            })
    summary_path = args.root / f"{args.output_prefix}_summary.json"
    summary_path.write_text(json.dumps({
        "protocol": "student-trajectory-response-v1",
        "cells": summary,
        "aggregate_correlations_pooled_24_seed_pairs": _correlation_summary(
            pooled_observations, args.checkpoints),
        "aggregate_correlations_cell_means_4_cells": _correlation_summary(
            cell_means, args.checkpoints),
    }, indent=2, ensure_ascii=True, allow_nan=False), encoding="utf-8")
    print(json.dumps({"event": "trajectory_matrix_complete",
                      "summary": str(summary_path)}), flush=True)


if __name__ == "__main__":
    main()
