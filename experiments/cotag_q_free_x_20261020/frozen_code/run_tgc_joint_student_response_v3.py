"""Label-free student-response selector for shared-assignment candidates.

V2 showed that selector-dev F1 can rank candidates differently from the
canonical report split.  V3 therefore uses only response signals available
from the legal construction inputs: InfoNCE loss and diagonal-vs-negative
margin on the current raw delta and the old synthetic prefix.  Report-dev
F1 remains a post-selection audit and is never part of the gate.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import run_tgc_joint_student_response as V1
import run_tgc_joint_student_response_v2 as V2
import run_unified_v3_continual as U
import tgc_shared_assignment as J


DATASETS = V1.DATASETS
DEFAULT_SELECTION_SEEDS = (300, 301, 302)
DEFAULT_REPORT_SEEDS = V1.DEFAULT_REPORT_SEEDS


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True), encoding="utf-8")


def _stats(model, x, adj, text, rows):
    rows = rows.to(x.device)
    graph = model.encode_graph(x, adj)[rows]
    encoded_text = model.encode_text(text[rows])
    logits = model.logit_scale.exp().clamp(max=100) * (graph @ encoded_text.T)
    labels = torch.arange(len(rows), device=x.device)
    loss = (F.cross_entropy(logits, labels) +
            F.cross_entropy(logits.T, labels)) / 2
    diagonal = logits.diag()
    if len(rows) > 1:
        negative = logits.masked_fill(torch.eye(len(rows), device=x.device).bool(), -1e9)
        margin = (diagonal - negative.max(1).values).mean()
    else:
        margin = diagonal.mean()
    return float(loss.detach()), float(margin.detach())


def _summary(values):
    array = np.asarray(values, dtype=float)
    return {"mean": float(array.mean()), "std": float(array.std()),
            "min": float(array.min()), "max": float(array.max()),
            "n": int(array.size)}


def evaluate_label_free_response(memory: dict, delta: dict, old_n: int,
                                seeds: list[int], steps_curve: list[int],
                                device: str) -> dict:
    """Evaluate legal raw-delta and old-memory response without labels."""
    syn = U.G.memory_to_trainable(memory)
    syn_x, syn_text = syn["x"].to(device), syn["text_emb"].to(device)
    syn_adj = U.G.norm_adj(syn["edge_index"], len(syn_x), device)
    old_ids = torch.arange(max(0, int(old_n)))
    new_ids = torch.arange(max(0, int(old_n)), len(syn_x))
    delta_x = delta["x"].to(device)
    delta_text = delta["text_emb"].to(device)
    delta_adj = U.G.norm_adj(delta["edge_index"], len(delta_x), device)
    delta_ids = torch.arange(len(delta_x))
    curve = []
    for steps in steps_curve:
        rows = []
        for seed in seeds:
            model = U.G.train_student(syn, int(seed), device, int(steps),
                                      batch=U.protocol_v3.STUDENT.batch)
            with torch.no_grad():
                old_loss, old_margin = _stats(model, syn_x, syn_adj, syn_text, old_ids)
                new_loss, new_margin = _stats(model, syn_x, syn_adj, syn_text, new_ids)
                delta_loss, delta_margin = _stats(
                    model, delta_x, delta_adj, delta_text, delta_ids)
            rows.append({"seed": int(seed), "steps": int(steps),
                         "old_loss": old_loss, "old_margin": old_margin,
                         "new_memory_loss": new_loss,
                         "new_memory_margin": new_margin,
                         "delta_loss": delta_loss,
                         "delta_margin": delta_margin})
            del model
        curve.append({
            "steps": int(steps),
            "old_loss": _summary([r["old_loss"] for r in rows]),
            "old_margin": _summary([r["old_margin"] for r in rows]),
            "new_memory_loss": _summary([r["new_memory_loss"] for r in rows]),
            "new_memory_margin": _summary([r["new_memory_margin"] for r in rows]),
            "delta_loss": _summary([r["delta_loss"] for r in rows]),
            "delta_margin": _summary([r["delta_margin"] for r in rows]),
            "per_seed": rows,
        })
    return {"steps": [int(v) for v in steps_curve], "grids": curve,
            "seeds": [int(v) for v in seeds]}


def response_summary_v3(response: dict, reference: dict,
                        weights: list[float] | None = None,
                        old_weight: float = 0.25,
                        margin_weight: float = 0.25,
                        variance_weight: float = 0.10) -> dict:
    grids, ref_grids = response["grids"], reference["grids"]
    if len(grids) != len(ref_grids):
        raise ValueError("response/reference curve length mismatch")
    if weights is None:
        weights = [0.15, 0.30, 0.55] if len(grids) == 3 else None
    if weights is None or len(weights) != len(grids):
        weights = [1.0 / len(grids)] * len(grids)
    weights = np.asarray(weights, dtype=float)
    weights /= weights.sum()
    point, old_gain, variance = [], [], []
    for cand, ref in zip(grids, ref_grids):
        new_loss_gain = float(ref["delta_loss"]["mean"] - cand["delta_loss"]["mean"])
        new_margin_gain = float(cand["delta_margin"]["mean"] - ref["delta_margin"]["mean"])
        old_loss_gain = float(ref["old_loss"]["mean"] - cand["old_loss"]["mean"])
        old_margin_gain = float(cand["old_margin"]["mean"] - ref["old_margin"]["mean"])
        new_gain = new_loss_gain + float(margin_weight) * new_margin_gain
        old_value = old_loss_gain + float(margin_weight) * old_margin_gain
        point.append(new_gain + float(old_weight) * old_value)
        old_gain.append(old_value)
        variance.append(float(cand["delta_loss"]["std"] + cand["delta_margin"]["std"]
                              + cand["old_loss"]["std"] + cand["old_margin"]["std"]))
    weighted = float(np.dot(weights, np.asarray(point)))
    terminal = float(point[-1])
    terminal_old = float(old_gain[-1])
    uncertainty = float(np.dot(weights, np.asarray(variance)))
    return {"curve_weights": weights.tolist(), "point_gains": point,
            "old_gains": old_gain, "weighted_gain": weighted,
            "terminal_gain": terminal, "terminal_old_delta": terminal_old,
            "variance_penalty": uncertainty,
            "utility": weighted - float(variance_weight) * uncertainty}


def choose_candidate_v3(candidates, delta, data, stage, old_n, seeds, steps,
                        device, old_weight, margin_weight, variance_weight,
                        retention_tolerance, terminal_gain_tolerance):
    reference_name = "fallback_content"
    reference_response = evaluate_label_free_response(
        candidates[reference_name], delta, old_n, seeds, steps, device)
    reports = []
    for name, memory in candidates.items():
        response = (reference_response if name == reference_name else
                    evaluate_label_free_response(memory, delta, old_n, seeds,
                                                 steps, device))
        summary = response_summary_v3(
            response, reference_response, old_weight=old_weight,
            margin_weight=margin_weight, variance_weight=variance_weight)
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
    chosen = max(eligible, key=lambda r: (r["utility"], r["terminal_gain"], r["candidate"])) \
        if eligible else fallback
    selected = chosen["candidate"]
    return selected, {
        "selector_version": "label-free-delta-old-response-v3",
        "reference_candidate": reference_name,
        "stage": int(stage), "old_rows": int(old_n),
        "selection_seeds": [int(v) for v in seeds],
        "selection_steps": [int(v) for v in steps],
        "old_weight": float(old_weight), "margin_weight": float(margin_weight),
        "variance_weight": float(variance_weight),
        "retention_tolerance": float(retention_tolerance),
        "terminal_gain_tolerance": float(terminal_gain_tolerance),
        "selected_candidate": selected,
        "decision_reason": "best_label_free_eligible_candidate" if selected != reference_name
                           else "fallback_reference_or_no_eligible_candidate",
        "candidates": {r["candidate"]: r for r in reports},
        "report_labels_used": False, "selector_labels_used": False,
        "response_inputs": "current raw delta + old synthetic memory only",
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
        "name": "tgc-shared-assignment-gcn-response-v3",
        "dataset": dataset, "order": order, "stages": list(args.stages),
        "student": "original two-layer GCN + bidirectional InfoNCE",
        "graphgps": "diagnostic only; not used",
        "selector": "label-free current-delta and old-memory InfoNCE loss/margin",
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
        candidates, build_audit = V1.build_candidates(data, source, order,
                                                      int(stage), args.device)
        save_json(out / f"s{stage}_build_audit.json", build_audit)
        delta = J.T.raw_delta_block(data, ids[int(stage)], int(stage))
        old_n = 0 if int(stage) == 0 else int(budgets[int(stage) - 1])
        steps = V2.stage_curve(len(candidates["fallback_content"]["x"]),
                               args.early_steps, args.middle_steps)
        selected, selection = choose_candidate_v3(
            candidates, delta, data, int(stage), old_n,
            list(args.selection_seeds), steps, args.device, args.old_weight,
            args.margin_weight, args.variance_weight,
            args.retention_tolerance, args.terminal_gain_tolerance)
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
    parser.add_argument("--old-weight", type=float, default=0.25)
    parser.add_argument("--margin-weight", type=float, default=0.25)
    parser.add_argument("--variance-weight", type=float, default=0.10)
    parser.add_argument("--retention-tolerance", type=float, default=0.50)
    parser.add_argument("--terminal-gain-tolerance", type=float, default=0.50)
    args = parser.parse_args()
    torch.set_num_threads(2)
    run_cell(args, args.dataset, args.order)
    print("TGC_SHARED_ASSIGNMENT_GCN_RESPONSE_V3_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
