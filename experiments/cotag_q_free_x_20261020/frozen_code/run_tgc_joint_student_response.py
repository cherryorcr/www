"""Minimal shared-assignment student-response selector.

This runner is the first post-GraphGPS experiment on the original GCN
student.  A single assignment P materializes X, text and a binary topology;
the selector compares complete candidates with short GCN retraining on the
selector-dev split.  Report-dev scores are computed only after the decision
is sealed and are never used to choose a candidate.

The initial implementation is deliberately stage-wise: every stage uses the
sealed C0 parent as its historical prefix.  This isolates candidate selection
from recursive rollout.  Recursive selected-parent rollout is a follow-up
only if this smoke matrix passes its response and report checks.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import torch

import run_p0_random_synthetic as P0
import run_student_compatible_append as SCA
import run_tgc_r11_c0_anchor as R11
import run_tgc_r5_joint as V
import run_unified_v3_continual as U
import tgc_shared_assignment as J


DATASETS = ("cora", "citeseer")
DEFAULT_SOURCE = Path(
    "/home/pc/research/TAG_condensation/results/tgc_r10_joint_confirmation_20260923b"
)
DEFAULT_CACHE = Path(
    "/home/pc/research/TAG_condensation/results/p5_rate_curve_small_20260919"
)
DEFAULT_SELECTION_SEEDS = (300, 301)
DEFAULT_REPORT_SEEDS = tuple(range(150, 156))
DEFAULT_SELECTION_STEPS = (128, 256)


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True), encoding="utf-8")


def tensor_hash(value: torch.Tensor) -> str:
    return J.R.C.tensor_hash({"value": value.detach().cpu()})


def memory_hash(memory: dict) -> str:
    return J.T.memory_hash(memory)


def load_data(dataset: str, minilm: str, device: str, cache_root: Path) -> dict:
    cache = cache_root / f"{dataset}_r005" / "text_emb_pool_only.pt"
    return P0.load_data(dataset, minilm, device, cache)


def append_c0_parent(parent: dict, block: dict, device: str) -> dict:
    return R11.append_parent(parent, block, device)


def response_summary(response: dict, reference: dict,
                     old_weight: float = 0.25,
                     variance_weight: float = 0.10) -> dict:
    """Compute selection-only gains from selector-dev response curves."""
    new_gain = float(response["tail_mean"] - reference["tail_mean"])
    old = response.get("old_tail_mean")
    ref_old = reference.get("old_tail_mean")
    old_delta = 0.0 if old is None or ref_old is None else float(old - ref_old)
    variance = float(response["tail_std"] + response["cumulative_tail_std"])
    utility = new_gain + float(old_weight) * old_delta - float(variance_weight) * variance

    horizon_gains = []
    for candidate_grid, reference_grid in zip(response["grids"], reference["grids"]):
        current_gain = float(
            candidate_grid["current_f1"]["mean"]
            - reference_grid["current_f1"]["mean"]
        )
        candidate_old = candidate_grid.get("old_mean_f1")
        reference_old = reference_grid.get("old_mean_f1")
        old_gain = 0.0
        if candidate_old is not None and reference_old is not None:
            old_gain = float(candidate_old["mean"] - reference_old["mean"])
        horizon_gains.append(current_gain + float(old_weight) * old_gain)
    return {
        "new_gain": new_gain,
        "old_delta": old_delta,
        "variance_penalty": variance,
        "utility": float(utility),
        "minimum_horizon_gain": float(min(horizon_gains)) if horizon_gains else 0.0,
        "horizon_gains": horizon_gains,
    }


def build_candidates(data: dict, source: Path, order: str, stage: int,
                     device: str) -> tuple[dict[str, dict], dict]:
    """Rebuild shared-P candidates and compare them with sealed artifacts."""
    if stage <= 0:
        c0 = load(source / f"{data['dataset']}" / order / "c0_s0_memory.pt")
        return {"fallback_content": c0, "p_initial": c0,
                "anchor025": c0, "joint": c0}, {
                    "stage": stage, "source_rebuild": False,
                    "reason": "stage-0 has no incoming delta"}

    cell = source / data["dataset"] / order
    c0 = load(cell / f"c0_s{stage}_memory.pt")
    parent = load(cell / f"c0_s{stage - 1}_memory.pt")
    c0_block = load(cell / f"c0_s{stage}_block.pt")
    ids, _ = R11.D.set_stream(data, order)
    delta = J.T.raw_delta_block(data, ids[stage], stage)
    old_n = len(parent["x"])
    q = R11.D.current_assignment(data, ids[stage], c0, old_n, stage, device)
    labels = data["y"][ids[stage]].cpu()
    p_initial = R11.initial_assignment(q, labels, V.CFG, device)
    p_joint = load(cell / f"shared_s{stage}_assignment.pt").float()
    if p_joint.shape != p_initial.shape:
        raise AssertionError("sealed joint assignment shape mismatch")

    base = append_c0_parent(parent, c0_block, device)
    # The lifted support must be defined from the current C0 source partition.
    support = J.M.lifted_support(parent, delta, base, q)
    p_anchor = 0.75 * p_initial + 0.25 * p_joint
    assignments = {
        "p_initial": p_initial,
        "anchor025": p_anchor,
        "joint": p_joint,
    }
    candidates = {}
    audits = {
        "stage": stage,
        "source_cell": str(cell),
        "parent_memory_sha256": memory_hash(parent),
        "c0_memory_sha256": memory_hash(c0),
        "delta_rows": len(delta["x"]),
        "q_sha256": tensor_hash(q),
        "p_initial_sha256": tensor_hash(p_initial),
        "p_joint_sha256": tensor_hash(p_joint),
        "p_anchor_sha256": tensor_hash(p_anchor),
        "source_rebuild": True,
        "candidate_audits": {},
    }
    for name, p in assignments.items():
        coarse = R11.coarse_from_assignment(support, p, old_n, device)
        material, material_audit = J.materialize(base, delta, p, coarse,
                                                 data["encoder"], V.CFG)
        memory = material["joint"]
        candidates[name] = memory
        audit = {
            "assignment_sha256": tensor_hash(p),
            "memory_sha256": memory_hash(memory),
            "materialization": material_audit,
            "binary_edge_entries": int(memory["edge_index"].shape[1]),
            "old_edge_entries": int(parent["edge_index"].shape[1]),
            "shared_assignment_used_for_x_text_topology": True,
        }
        sealed_name = name
        sealed_path = cell / f"{sealed_name}_s{stage}_memory.pt"
        if sealed_path.exists():
            sealed = load(sealed_path)
            audit["matches_sealed_artifact"] = memory_hash(sealed) == memory_hash(memory)
            audit["sealed_memory_sha256"] = memory_hash(sealed)
        audits["candidate_audits"][name] = audit

    # R10 content is the prescribed all-or-nothing fallback.  It is kept as a
    # reference only; it is not treated as a shared-P candidate.
    fallback = load(cell / f"content_s{stage}_memory.pt")
    candidates["fallback_content"] = fallback
    audits["fallback_content_memory_sha256"] = memory_hash(fallback)
    audits["fallback_content_is_shared_candidate"] = False
    return candidates, audits


def choose_candidate(candidates: dict[str, dict], data: dict, stage: int,
                     selection_seeds: list[int], selection_steps: list[int],
                     device: str, old_weight: float, variance_weight: float,
                     retention_tol: float, min_horizon_gain: float) -> tuple[str, dict]:
    """Select a whole candidate from selector-dev response only."""
    reports = {}
    # The fallback is the reference candidate.  It is also an eligible choice,
    # so an unsafe shared-P candidate automatically falls back as a whole.
    reference_name = "fallback_content"
    reference_response = SCA.evaluate_response_curve(
        candidates[reference_name], stage, selection_seeds, [16], selection_steps,
        device, data["x_device"], data["real_adj"], data["prompt_emb"],
        data["y"], data["selector_dev_mask"])
    for name, memory in candidates.items():
        if name == reference_name:
            response = reference_response
        else:
            response = SCA.evaluate_response_curve(
                memory, stage, selection_seeds, [16], selection_steps, device,
                data["x_device"], data["real_adj"], data["prompt_emb"],
                data["y"], data["selector_dev_mask"])
        summary = response_summary(response, reference_response,
                                   old_weight, variance_weight)
        reports[name] = {
            "candidate": name,
            "memory_sha256": memory_hash(memory),
            "response": response,
            **summary,
            "eligible": bool(
                summary["old_delta"] >= -float(retention_tol)
                and summary["minimum_horizon_gain"] >= -float(min_horizon_gain)
            ),
        }

    eligible = [r for r in reports.values() if r["eligible"]]
    if not eligible:
        selected = reference_name
        reason = "no_shared_candidate_passed_retention_and_horizon_gate"
    else:
        selected_row = max(
            eligible,
            key=lambda row: (float(row["utility"]), float(row["new_gain"]),
                             float(row["old_delta"]), row["candidate"]),
        )
        selected = str(selected_row["candidate"])
        # Never displace the fallback for a negative aggregate utility.
        if float(selected_row["utility"]) < 0.0:
            selected = reference_name
            reason = "best_shared_candidate_has_negative_utility"
        else:
            reason = "best_eligible_shared_candidate"
    return selected, {
        "reference_candidate": reference_name,
        "selection_seeds": [int(v) for v in selection_seeds],
        "selection_steps": [int(v) for v in selection_steps],
        "selection_batches": [16],
        "old_weight": float(old_weight),
        "variance_weight": float(variance_weight),
        "retention_tolerance": float(retention_tol),
        "minimum_horizon_gain": float(min_horizon_gain),
        "selected_candidate": selected,
        "decision_reason": reason,
        "candidates": reports,
        "report_labels_used": False,
        "selection_mask": "selector_dev only",
    }


def report_candidate(memory: dict, data: dict, stage: int,
                     seeds: list[int], device: str) -> dict:
    rows = []
    started = time.perf_counter()
    for seed in seeds:
        steps = U.protocol_v3.student_steps(len(memory["x"]))
        model = U.G.train_student(U.G.memory_to_trainable(memory), int(seed),
                                  device, steps, batch=U.protocol_v3.STUDENT.batch)
        score = U.G.eval_stagewise(
            model, data["x_device"], data["real_adj"], data["prompt_emb"],
            data["y"], data["report_dev_mask"], stage)
        rows.append({"seed": int(seed), "steps": int(steps), **score})
        del model
    return {
        "memory_sha256": memory_hash(memory),
        "student_seeds": [int(v) for v in seeds],
        "student_steps": int(U.protocol_v3.student_steps(len(memory["x"]))),
        "student_batch": int(U.protocol_v3.STUDENT.batch),
        "report_mask": "report_dev",
        "report_labels_used_after_selection": True,
        "wall_seconds": float(time.perf_counter() - started),
        "cumulative_f1_mean": float(np.mean([r["cumulative_macro_f1"] for r in rows])),
        "cumulative_f1_std": float(np.std([r["cumulative_macro_f1"] for r in rows])),
        "current_f1_mean": float(np.mean([r["current_f1"] for r in rows])),
        "old_mean_f1_mean": (float(np.mean([r["old_mean_f1"] for r in rows]))
                             if stage > 0 else None),
        "per_stage_mean": [float(np.mean([r["per_stage"][s]["macro_f1"]
                                           for r in rows]))
                           for s in range(stage + 1)],
        "per_student": rows,
    }


def run_cell(args, dataset: str, order: str) -> None:
    out = Path(args.output) / dataset / order
    if (out / "results.json").exists():
        raise FileExistsError(f"immutable output already exists: {out}")
    out.mkdir(parents=True, exist_ok=True)
    data = load_data(dataset, args.minilm, args.device, Path(args.cache_root))
    ids, budgets = R11.D.set_stream(data, order)
    source = Path(args.source_root)
    stages = [int(v) for v in args.stages]
    protocol = {
        "name": "tgc-shared-assignment-gcn-response-v1",
        "dataset": dataset,
        "order": order,
        "stages": stages,
        "rate": 0.05,
        "source_root": str(source),
        "student": "original two-layer GCN + trainable text projection + bidirectional InfoNCE",
        "graphgps": "diagnostic only; not used",
        "candidates": ["fallback_content", "p_initial", "anchor025", "joint"],
        "shared_assignment": "one P materializes X, top-5/320-char re-encoded text, and binary topology",
        "p_anchor": "0.75*p_initial + 0.25*p_joint",
        "selection": "selector-dev GCN response; no report/test labels",
        "selection_seeds": [int(v) for v in args.selection_seeds],
        "selection_steps": [int(v) for v in args.selection_steps],
        "report_seeds": [int(v) for v in args.report_seeds],
        "historical_raw_reads": 0,
        "recursive_rollout": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    save_json(out / "protocol.json", protocol)

    results = []
    for stage in stages:
        if stage < 0 or stage >= len(ids):
            raise ValueError(f"invalid stage {stage}")
        candidates, build_audit = build_candidates(data, source, order, stage,
                                                   args.device)
        save_json(out / f"s{stage}_build_audit.json", build_audit)
        selected, selection = choose_candidate(
            candidates, data, stage, list(args.selection_seeds),
            list(args.selection_steps), args.device, args.old_weight,
            args.variance_weight, args.retention_tolerance,
            args.min_horizon_gain)
        save_json(out / f"s{stage}_selection.json", selection)
        stage_result = {
            "stage": stage,
            "budget": int(budgets[stage]),
            "selected_candidate": selected,
            "candidate_rows": {name: int(len(memory["x"]))
                                for name, memory in candidates.items()},
            "selection": selection,
            "report": {},
        }
        for name in ("fallback_content", "p_initial", "anchor025", "joint"):
            stage_result["report"][name] = report_candidate(
                candidates[name], data, stage, list(args.report_seeds), args.device)
            torch.save(candidates[name], out / f"{name}_s{stage}_memory.pt")
        stage_result["selected_report"] = stage_result["report"][selected]
        results.append(stage_result)
        save_json(out / "progress.json", {"results": results})
        print(json.dumps({
            "event": "stage_complete",
            "dataset": dataset,
            "order": order,
            "stage": stage,
            "selected": selected,
            "selection_reason": selection["decision_reason"],
            "selected_report_f1": stage_result["selected_report"]["cumulative_f1_mean"],
            "fallback_report_f1": stage_result["report"]["fallback_content"]["cumulative_f1_mean"],
        }), flush=True)
    save_json(out / "results.json", {
        "protocol": protocol,
        "manifest": {"stage_classes": data["stage_classes"],
                     "budgets": [int(v) for v in budgets],
                     "stage_source_ids": [v.tolist() for v in ids]},
        "results": results,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--order", required=True)
    parser.add_argument("--stages", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE))
    parser.add_argument("--cache-root", default=str(DEFAULT_CACHE))
    parser.add_argument("--minilm", default="/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--selection-seeds", nargs="+", type=int,
                        default=list(DEFAULT_SELECTION_SEEDS))
    parser.add_argument("--selection-steps", nargs="+", type=int,
                        default=list(DEFAULT_SELECTION_STEPS))
    parser.add_argument("--report-seeds", nargs="+", type=int,
                        default=list(DEFAULT_REPORT_SEEDS))
    parser.add_argument("--old-weight", type=float, default=0.25)
    parser.add_argument("--variance-weight", type=float, default=0.10)
    parser.add_argument("--retention-tolerance", type=float, default=0.50)
    parser.add_argument("--min-horizon-gain", type=float, default=0.50)
    args = parser.parse_args()
    torch.set_num_threads(2)
    run_cell(args, args.dataset, args.order)
    print("TGC_SHARED_ASSIGNMENT_GCN_RESPONSE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
