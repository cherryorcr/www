"""Prompt-aware paired selector for shared-assignment candidates.

This selector keeps the V3 construction and original GCN student unchanged.
It replaces absolute response ranking with two label-free signals:

* class-prompt margins on the current delta and old synthetic prefix;
* candidate-minus-fallback paired response using the same student seed and
  training horizon.

Node labels are not used by the selector. Class names are already part of the
zero-shot task protocol and are used only to form prompt groups.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch

import run_tgc_joint_student_response as V1
import run_tgc_joint_student_response_v2 as V2
import run_tgc_joint_student_response_v3 as V3
import run_unified_v3_continual as U
import tgc_shared_assignment as J


DATASETS = V3.DATASETS
DEFAULT_SELECTION_SEEDS = (300, 301, 302)
DEFAULT_REPORT_SEEDS = V3.DEFAULT_REPORT_SEEDS


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True), encoding="utf-8")


def _summary(values):
    array = np.asarray(values, dtype=float)
    return {"mean": float(array.mean()), "std": float(array.std()),
            "min": float(array.min()), "max": float(array.max()),
            "n": int(array.size)}


def _prompt_stats(model, x, adj, prompt_emb, old_prompt_ids,
                  current_prompt_ids, focus: str,
                  rows: torch.Tensor | None = None) -> dict:
    """Compute prompt margins without accessing node labels."""
    graph = model.encode_graph(x, adj)
    if rows is not None:
        graph = graph[rows.to(graph.device)]
    prompts = model.encode_text(prompt_emb.to(x.device))
    scores = graph @ prompts.T
    old_ids = torch.tensor(old_prompt_ids, device=x.device, dtype=torch.long)
    current_ids = torch.tensor(current_prompt_ids, device=x.device, dtype=torch.long)
    if not len(old_ids) or not len(current_ids):
        raise ValueError("prompt groups must both be non-empty")

    old_top = scores[:, old_ids].max(1).values
    current_top = scores[:, current_ids].max(1).values
    if focus == "new":
        gap = current_top - old_top
        primary = scores[:, current_ids]
    elif focus == "old":
        gap = old_top - current_top
        primary = scores[:, old_ids]
    else:
        raise ValueError(focus)

    if primary.shape[1] >= 2:
        top = primary.topk(2, dim=1).values
        within_margin = top[:, 0] - top[:, 1]
    else:
        within_margin = scores.topk(2, dim=1).values
        within_margin = within_margin[:, 0] - within_margin[:, 1]
    return {
        "prompt_gap": float(gap.mean().detach()),
        "prompt_margin": float(within_margin.mean().detach()),
    }


def _fit_response(memory: dict, delta: dict, old_n: int, data: dict,
                  stage: int, seed: int, steps: int, device: str) -> dict:
    syn = U.G.memory_to_trainable(memory)
    syn_x, syn_text = syn["x"].to(device), syn["text_emb"].to(device)
    syn_adj = U.G.norm_adj(syn["edge_index"], len(syn_x), device)
    old_ids = torch.arange(max(0, int(old_n)))
    delta_x = delta["x"].to(device)
    delta_text = delta["text_emb"].to(device)
    delta_adj = U.G.norm_adj(delta["edge_index"], len(delta_x), device)
    delta_ids = torch.arange(len(delta_x))
    old_prompt_ids = [int(c) for group in data["stage_classes"][:stage]
                      for c in group]
    current_prompt_ids = [int(c) for c in data["stage_classes"][stage]]

    model = U.G.train_student(
        syn, int(seed), device, int(steps), batch=U.protocol_v3.STUDENT.batch)
    with torch.no_grad():
        old_loss, old_margin = V3._stats(model, syn_x, syn_adj, syn_text, old_ids)
        delta_loss, delta_margin = V3._stats(
            model, delta_x, delta_adj, delta_text, delta_ids)
        old_prompt = _prompt_stats(
            model, syn_x,
            syn_adj, data["prompt_emb"], old_prompt_ids,
            current_prompt_ids, "old", rows=old_ids)
        delta_prompt = _prompt_stats(
            model, delta_x, delta_adj, data["prompt_emb"],
            old_prompt_ids, current_prompt_ids, "new")
    del model
    return {
        "seed": int(seed), "steps": int(steps),
        "old_loss": old_loss, "old_margin": old_margin,
        "delta_loss": delta_loss, "delta_margin": delta_margin,
        "old_prompt_gap": old_prompt["prompt_gap"],
        "old_prompt_margin": old_prompt["prompt_margin"],
        "delta_prompt_gap": delta_prompt["prompt_gap"],
        "delta_prompt_margin": delta_prompt["prompt_margin"],
    }


def evaluate_response(memory: dict, delta: dict, old_n: int, data: dict,
                      stage: int, seeds: list[int], steps_curve: list[int],
                      device: str) -> dict:
    grids = []
    for steps in steps_curve:
        rows = [_fit_response(memory, delta, old_n, data, stage, seed,
                              int(steps), device) for seed in seeds]
        grids.append({
            "steps": int(steps),
            "old_loss": _summary([r["old_loss"] for r in rows]),
            "old_margin": _summary([r["old_margin"] for r in rows]),
            "delta_loss": _summary([r["delta_loss"] for r in rows]),
            "delta_margin": _summary([r["delta_margin"] for r in rows]),
            "old_prompt_gap": _summary([r["old_prompt_gap"] for r in rows]),
            "old_prompt_margin": _summary([r["old_prompt_margin"] for r in rows]),
            "delta_prompt_gap": _summary([r["delta_prompt_gap"] for r in rows]),
            "delta_prompt_margin": _summary([r["delta_prompt_margin"] for r in rows]),
            "per_seed": rows,
        })
    return {"steps": [int(v) for v in steps_curve],
            "grids": grids, "seeds": [int(v) for v in seeds]}


def _paired_point(candidate: dict, reference: dict) -> dict:
    """Aggregate one candidate/reference horizon with paired seeds."""
    paired = []
    for cand, ref in zip(candidate["per_seed"], reference["per_seed"]):
        new_score = ((cand["delta_prompt_gap"] - ref["delta_prompt_gap"])
                     + 0.5 * (cand["delta_prompt_margin"]
                              - ref["delta_prompt_margin"]))
        old_score = ((cand["old_prompt_gap"] - ref["old_prompt_gap"])
                     + 0.25 * (cand["old_prompt_margin"]
                               - ref["old_prompt_margin"]))
        nce_gain = ref["delta_loss"] - cand["delta_loss"]
        margin_gain = cand["delta_margin"] - ref["delta_margin"]
        point = 0.60 * new_score + 0.30 * old_score + 0.05 * nce_gain + 0.05 * margin_gain
        paired.append({"seed": cand["seed"], "new_score": new_score,
                       "old_score": old_score, "nce_gain": nce_gain,
                       "margin_gain": margin_gain, "point": point})
    return {
        "new_score": _summary([r["new_score"] for r in paired]),
        "old_score": _summary([r["old_score"] for r in paired]),
        "nce_gain": _summary([r["nce_gain"] for r in paired]),
        "margin_gain": _summary([r["margin_gain"] for r in paired]),
        "point": _summary([r["point"] for r in paired]),
        "per_seed": paired,
    }


def response_summary_prompt_paired(response: dict, reference: dict,
                                   weights: list[float] | None = None,
                                   variance_weight: float = 0.10) -> dict:
    if len(response["grids"]) != len(reference["grids"]):
        raise ValueError("response/reference curve length mismatch")
    if weights is None:
        weights = [0.15, 0.30, 0.55] if len(response["grids"]) == 3 else None
    if weights is None or len(weights) != len(response["grids"]):
        weights = [1.0 / len(response["grids"])] * len(response["grids"])
    weights = np.asarray(weights, dtype=float)
    weights /= weights.sum()
    horizons = [_paired_point(cand, ref)
                for cand, ref in zip(response["grids"], reference["grids"])]
    weighted = float(np.dot(weights, [h["point"]["mean"] for h in horizons]))
    terminal = float(horizons[-1]["point"]["mean"])
    terminal_old = float(horizons[-1]["old_score"]["mean"])
    uncertainty = float(np.dot(weights, [h["point"]["std"] for h in horizons]))
    return {
        "curve_weights": weights.tolist(),
        "horizons": horizons,
        "weighted_gain": weighted,
        "terminal_gain": terminal,
        "terminal_old_delta": terminal_old,
        "variance_penalty": uncertainty,
        "utility": weighted - float(variance_weight) * uncertainty,
    }


def choose_candidate_prompt_paired(
        candidates, delta, data, stage, old_n, seeds, steps, device,
        variance_weight, retention_tolerance, terminal_gain_tolerance):
    reference_name = "fallback_content"
    reference_response = evaluate_response(
        candidates[reference_name], delta, old_n, data, stage, seeds, steps, device)
    reports = []
    for name, memory in candidates.items():
        response = (reference_response if name == reference_name else
                    evaluate_response(memory, delta, old_n, data, stage,
                                      seeds, steps, device))
        summary = response_summary_prompt_paired(
            response, reference_response, variance_weight=variance_weight)
        reports.append({"candidate": name, "memory_sha256": V1.memory_hash(memory),
                        "response": response, **summary})
    fallback = next(r for r in reports if r["candidate"] == reference_name)
    fallback.update({"weighted_gain": 0.0, "terminal_gain": 0.0,
                     "terminal_old_delta": 0.0, "variance_penalty": 0.0,
                     "utility": 0.0, "eligible": True})
    for row in reports:
        if row["candidate"] == reference_name:
            continue
        row["eligible"] = bool(
            row["terminal_old_delta"] >= -float(retention_tolerance)
            and row["terminal_gain"] >= -float(terminal_gain_tolerance)
            and row["utility"] >= 0.0)
    eligible = [r for r in reports if r["eligible"]]
    chosen = max(eligible, key=lambda r: (r["utility"], r["terminal_gain"],
                                           r["candidate"])) if eligible else fallback
    selected = chosen["candidate"]
    return selected, {
        "selector_version": "class-prompt-margin-paired-response-v1",
        "reference_candidate": reference_name,
        "stage": int(stage), "old_rows": int(old_n),
        "selection_seeds": [int(v) for v in seeds],
        "selection_steps": [int(v) for v in steps],
        "variance_weight": float(variance_weight),
        "retention_tolerance": float(retention_tolerance),
        "terminal_gain_tolerance": float(terminal_gain_tolerance),
        "selected_candidate": selected,
        "decision_reason": "best_prompt_paired_eligible_candidate" if selected != reference_name
                           else "fallback_reference_or_no_eligible_candidate",
        "candidates": {r["candidate"]: r for r in reports},
        "report_labels_used": False, "selector_labels_used": False,
        "response_inputs": "class prompts + paired current-delta/old-memory response",
        "construction_unchanged_from_v3": True,
    }


def run_cell(args, dataset, order):
    out = Path(args.output) / dataset / order
    if (out / "results.json").exists():
        raise FileExistsError(f"immutable output already exists: {out}")
    out.mkdir(parents=True, exist_ok=True)
    data = V1.load_data(dataset, args.minilm, args.device, Path(args.cache_root))
    ids, budgets = V1.R11.D.set_stream(data, order)
    source = Path(args.source_root)
    protocol = {
        "name": "tgc-shared-assignment-gcn-prompt-paired-v1",
        "dataset": dataset, "order": order, "stages": list(args.stages),
        "student": "original two-layer GCN + bidirectional InfoNCE",
        "construction": "unchanged shared-assignment candidates from V3",
        "selector": "class-prompt margin + paired candidate-minus-fallback response",
        "selection_seeds": [int(v) for v in args.selection_seeds],
        "response_steps": "128/512/canonical per stage",
        "report_not_used_for_selection": True,
        "selector_labels_used": False,
        "recursive_rollout": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    save_json(out / "protocol.json", protocol)
    results = []
    for stage in args.stages:
        candidates, build_audit = V1.build_candidates(
            data, source, order, int(stage), args.device)
        save_json(out / f"s{stage}_build_audit.json", build_audit)
        delta = J.T.raw_delta_block(data, ids[int(stage)], int(stage))
        old_n = 0 if int(stage) == 0 else int(budgets[int(stage) - 1])
        steps = V2.stage_curve(len(candidates["fallback_content"]["x"]),
                               args.early_steps, args.middle_steps)
        selected, selection = choose_candidate_prompt_paired(
            candidates, delta, data, int(stage), old_n,
            list(args.selection_seeds), steps, args.device,
            args.variance_weight, args.retention_tolerance,
            args.terminal_gain_tolerance)
        save_json(out / f"s{stage}_selection.json", selection)
        report = {}
        for name in ("fallback_content", "p_initial", "anchor025", "joint"):
            report[name] = V3.V1.report_candidate(
                candidates[name], data, int(stage), list(args.report_seeds), args.device)
            torch.save(candidates[name], out / f"{name}_s{stage}_memory.pt")
        report_order = V2.rank_order({
            name: float(value["cumulative_f1_mean"])
            for name, value in report.items()})
        row = {"stage": int(stage), "budget": int(budgets[int(stage)]),
               "response_steps": steps, "selected_candidate": selected,
               "selection": selection, "report": report,
               "report_order": report_order,
               "selected_report_f1": float(report[selected]["cumulative_f1_mean"]),
               "best_report_candidate": report_order[0],
               "selection_regret_pp": float(
                   report[report_order[0]]["cumulative_f1_mean"]
                   - report[selected]["cumulative_f1_mean"])}
        results.append(row)
        save_json(out / "progress.json", {"results": results})
        print(json.dumps({"event": "stage_complete", "dataset": dataset,
                          "order": order, "stage": int(stage),
                          "selected": selected, "report_order": report_order,
                          "regret_pp": row["selection_regret_pp"]}), flush=True)
    save_json(out / "results.json", {
        "protocol": protocol,
        "manifest": {"stage_classes": data["stage_classes"],
                     "budgets": [int(v) for v in budgets],
                     "stage_source_ids": [v.tolist() for v in ids]},
        "results": results})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--order", required=True)
    parser.add_argument("--stages", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-root", default=str(V1.DEFAULT_SOURCE))
    parser.add_argument("--cache-root", default=str(V1.DEFAULT_CACHE))
    parser.add_argument("--minilm", default="/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--selection-seeds", nargs="+", type=int,
                        default=list(DEFAULT_SELECTION_SEEDS))
    parser.add_argument("--report-seeds", nargs="+", type=int,
                        default=list(DEFAULT_REPORT_SEEDS))
    parser.add_argument("--early-steps", type=int, default=128)
    parser.add_argument("--middle-steps", type=int, default=512)
    parser.add_argument("--variance-weight", type=float, default=0.10)
    parser.add_argument("--retention-tolerance", type=float, default=0.02)
    parser.add_argument("--terminal-gain-tolerance", type=float, default=0.02)
    args = parser.parse_args()
    torch.set_num_threads(2)
    run_cell(args, args.dataset, args.order)
    print("TGC_SHARED_ASSIGNMENT_GCN_PROMPT_PAIRED_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
