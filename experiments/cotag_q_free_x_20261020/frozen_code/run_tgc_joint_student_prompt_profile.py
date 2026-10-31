"""Per-class prompt-profile selector with uncertainty-aware paired gating.

The construction and original GCN student are inherited unchanged from V3.
This experiment addresses the prompt-paired false-rejection problem by:

* retaining a separate prompt-margin profile for every seen class;
* aggregating current-class gains with mean and lower-quantile terms;
* using paired candidate-vs-fallback seed variation rather than an absolute
  score threshold;
* selecting only candidates whose old-class lower-bound is non-inferior and
  whose robust utility is positive.

No node labels or report metrics are used by the selector.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch

import run_tgc_joint_student_prompt_paired as P
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


def _quantile(values, q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=float), q))


def _prompt_profile(model, x, adj, prompt_emb, prompt_ids,
                    seen_prompt_ids, rows: torch.Tensor | None = None,
                    top_fraction: float = 0.20) -> dict:
    """Return a label-free per-prompt top-quantile margin profile."""
    graph = model.encode_graph(x, adj)
    if rows is not None:
        graph = graph[rows.to(graph.device)]
    prompts = model.encode_text(prompt_emb.to(x.device))
    scores = graph @ prompts.T
    seen = torch.tensor(seen_prompt_ids, device=x.device, dtype=torch.long)
    selected = torch.tensor(prompt_ids, device=x.device, dtype=torch.long)
    if not len(seen) or not len(selected) or scores.shape[0] == 0:
        raise ValueError("prompt profile groups must be non-empty")
    seen_scores = scores[:, seen]
    values = {}
    top_count = max(1, int(np.ceil(float(top_fraction) * scores.shape[0])))
    for prompt_id in selected.tolist():
        local = int((seen == prompt_id).nonzero(as_tuple=False)[0])
        positive = seen_scores[:, local]
        if seen_scores.shape[1] > 1:
            negative = seen_scores.masked_fill(
                torch.nn.functional.one_hot(
                    torch.tensor(local, device=x.device), seen_scores.shape[1]
                ).bool().unsqueeze(0), -1e9).max(1).values
        else:
            negative = torch.zeros_like(positive)
        margin = positive - negative
        top = torch.topk(margin, min(top_count, len(margin)), sorted=False).values
        values[str(prompt_id)] = float(top.mean().detach())
    return values


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
    seen_prompt_ids = old_prompt_ids + current_prompt_ids

    model = U.G.train_student(
        syn, int(seed), device, int(steps), batch=U.protocol_v3.STUDENT.batch)
    with torch.no_grad():
        old_loss, old_margin = V3._stats(model, syn_x, syn_adj, syn_text, old_ids)
        delta_loss, delta_margin = V3._stats(
            model, delta_x, delta_adj, delta_text, delta_ids)
        old_profile = _prompt_profile(
            model, syn_x, syn_adj, data["prompt_emb"], old_prompt_ids,
            seen_prompt_ids, rows=old_ids)
        delta_profile = _prompt_profile(
            model, delta_x, delta_adj, data["prompt_emb"], current_prompt_ids,
            seen_prompt_ids)
    del model
    return {
        "seed": int(seed), "steps": int(steps),
        "old_loss": old_loss, "old_margin": old_margin,
        "delta_loss": delta_loss, "delta_margin": delta_margin,
        "old_profile": old_profile, "delta_profile": delta_profile,
    }


def evaluate_response(memory: dict, delta: dict, old_n: int, data: dict,
                      stage: int, seeds: list[int], steps_curve: list[int],
                      device: str) -> dict:
    grids = []
    for steps in steps_curve:
        rows = [_fit_response(memory, delta, old_n, data, stage, seed,
                              int(steps), device) for seed in seeds]
        old_ids = sorted(rows[0]["old_profile"])
        delta_ids = sorted(rows[0]["delta_profile"])
        grids.append({
            "steps": int(steps),
            "old_loss": _summary([r["old_loss"] for r in rows]),
            "old_margin": _summary([r["old_margin"] for r in rows]),
            "delta_loss": _summary([r["delta_loss"] for r in rows]),
            "delta_margin": _summary([r["delta_margin"] for r in rows]),
            "old_profile": {key: _summary([r["old_profile"][key] for r in rows])
                            for key in old_ids},
            "delta_profile": {key: _summary([r["delta_profile"][key] for r in rows])
                              for key in delta_ids},
            "per_seed": rows,
        })
    return {"steps": [int(v) for v in steps_curve],
            "grids": grids, "seeds": [int(v) for v in seeds]}


def _paired_profile_point(candidate: dict, reference: dict,
                          old_keys: list[str], new_keys: list[str]) -> dict:
    paired = []
    for cand, ref in zip(candidate["per_seed"], reference["per_seed"]):
        new_gains = [cand["delta_profile"][key] - ref["delta_profile"][key]
                     for key in new_keys]
        old_gains = [cand["old_profile"][key] - ref["old_profile"][key]
                     for key in old_keys]
        new_mean = float(np.mean(new_gains))
        new_q25 = _quantile(new_gains, 0.25)
        old_mean = float(np.mean(old_gains)) if old_gains else 0.0
        old_q25 = _quantile(old_gains, 0.25) if old_gains else 0.0
        new_score = 0.7 * new_q25 + 0.3 * new_mean
        old_score = 0.7 * old_q25 + 0.3 * old_mean
        nce_gain = ref["delta_loss"] - cand["delta_loss"]
        margin_gain = cand["delta_margin"] - ref["delta_margin"]
        point = new_score + 0.25 * old_score + 0.05 * nce_gain + 0.05 * margin_gain
        paired.append({"seed": cand["seed"], "new_by_class": new_gains,
                       "old_by_class": old_gains, "new_mean": new_mean,
                       "new_q25": new_q25, "old_mean": old_mean,
                       "old_q25": old_q25, "new_score": new_score,
                       "old_score": old_score, "nce_gain": nce_gain,
                       "margin_gain": margin_gain, "point": point})
    return {
        "new_score": _summary([r["new_score"] for r in paired]),
        "old_score": _summary([r["old_score"] for r in paired]),
        "point": _summary([r["point"] for r in paired]),
        "lcb_new": float(np.mean([r["new_score"] for r in paired])
                         - np.std([r["new_score"] for r in paired])),
        "lcb_old": float(np.mean([r["old_score"] for r in paired])
                         - np.std([r["old_score"] for r in paired])),
        "lcb_point": float(np.mean([r["point"] for r in paired])
                           - np.std([r["point"] for r in paired])),
        "per_seed": paired,
    }


def response_summary_profile(response: dict, reference: dict,
                             variance_weight: float = 0.10) -> dict:
    if len(response["grids"]) != len(reference["grids"]):
        raise ValueError("response/reference curve length mismatch")
    weights = np.asarray([0.15, 0.30, 0.55]
                         if len(response["grids"]) == 3 else
                         [1.0 / len(response["grids"])] * len(response["grids"]),
                         dtype=float)
    weights /= weights.sum()
    horizons = []
    for candidate, ref in zip(response["grids"], reference["grids"]):
        old_keys = sorted(candidate["old_profile"])
        new_keys = sorted(candidate["delta_profile"])
        horizons.append(_paired_profile_point(candidate, ref, old_keys, new_keys))
    mean_point = [h["point"]["mean"] for h in horizons]
    weighted = float(np.dot(weights, mean_point))
    uncertainty = float(np.dot(weights, [h["point"]["std"] for h in horizons]))
    terminal = horizons[-1]
    return {
        "curve_weights": weights.tolist(), "horizons": horizons,
        "weighted_gain": weighted,
        "terminal_gain": float(terminal["point"]["mean"]),
        "terminal_old_delta": float(terminal["old_score"]["mean"]),
        "terminal_new_lcb": float(terminal["lcb_new"]),
        "terminal_old_lcb": float(terminal["lcb_old"]),
        "terminal_lcb": float(terminal["lcb_point"]),
        "variance_penalty": uncertainty,
        "utility": weighted - float(variance_weight) * uncertainty,
    }


def choose_candidate_profile(candidates, delta, data, stage, old_n, seeds,
                             steps, device, variance_weight):
    reference_name = "fallback_content"
    reference_response = evaluate_response(
        candidates[reference_name], delta, old_n, data, stage, seeds, steps, device)
    reports = []
    for name, memory in candidates.items():
        response = (reference_response if name == reference_name else
                    evaluate_response(memory, delta, old_n, data, stage,
                                      seeds, steps, device))
        summary = response_summary_profile(response, reference_response,
                                           variance_weight=variance_weight)
        reports.append({"candidate": name, "memory_sha256": V1.memory_hash(memory),
                        "response": response, **summary})
    fallback = next(r for r in reports if r["candidate"] == reference_name)
    fallback.update({"weighted_gain": 0.0, "terminal_gain": 0.0,
                     "terminal_old_delta": 0.0, "terminal_new_lcb": 0.0,
                     "terminal_old_lcb": 0.0, "terminal_lcb": 0.0,
                     "variance_penalty": 0.0, "utility": 0.0,
                     "eligible": True})
    for row in reports:
        if row["candidate"] == reference_name:
            continue
        # Non-inferiority is defined in the paired response scale: historical
        # prompt profile must not have a negative one-sigma lower bound, and
        # the robust combined lower bound must beat the fallback anchor.
        row["eligible"] = bool(
            row["terminal_old_lcb"] >= 0.0 and row["terminal_lcb"] >= 0.0)
    eligible = [r for r in reports if r["eligible"]]
    chosen = max(eligible, key=lambda r: (r["terminal_lcb"],
                                           r["utility"], r["candidate"])) \
        if eligible else fallback
    selected = chosen["candidate"]
    return selected, {
        "selector_version": "per-class-prompt-profile-paired-lcb-v1",
        "reference_candidate": reference_name, "stage": int(stage),
        "old_rows": int(old_n), "selection_seeds": [int(v) for v in seeds],
        "selection_steps": [int(v) for v in steps],
        "variance_weight": float(variance_weight),
        "uncertainty_rule": "one-standard-deviation lower bound over paired seeds",
        "eligibility_rule": "terminal_old_lcb >= 0 and terminal_lcb >= 0",
        "selected_candidate": selected,
        "decision_reason": "best_profile_lcb_candidate" if selected != reference_name
                           else "fallback_reference_or_no_noninferior_candidate",
        "candidates": {r["candidate"]: r for r in reports},
        "report_labels_used": False, "selector_labels_used": False,
        "response_inputs": "per-class prompt margins + paired delta/old response",
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
        "name": "tgc-shared-assignment-gcn-prompt-profile-v1",
        "dataset": dataset, "order": order, "stages": list(args.stages),
        "student": "original two-layer GCN + bidirectional InfoNCE",
        "construction": "unchanged shared-assignment candidates from V3",
        "selector": "per-class prompt margin + paired one-sigma LCB gate",
        "selection_seeds": [int(v) for v in args.selection_seeds],
        "response_steps": "128/512/canonical per stage",
        "report_not_used_for_selection": True,
        "selector_labels_used": False, "recursive_rollout": False,
        "independent_order": order != "order19601",
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
        selected, selection = choose_candidate_profile(
            candidates, delta, data, int(stage), old_n,
            list(args.selection_seeds), steps, args.device, args.variance_weight)
        save_json(out / f"s{stage}_selection.json", selection)
        report = {}
        for name in ("fallback_content", "p_initial", "anchor025", "joint"):
            report[name] = V1.report_candidate(
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
    args = parser.parse_args()
    torch.set_num_threads(2)
    run_cell(args, args.dataset, args.order)
    print("TGC_SHARED_ASSIGNMENT_GCN_PROMPT_PROFILE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
