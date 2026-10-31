"""Stage-aware response selector for the shared-assignment candidates.

This is a selector-only follow-up to ``run_tgc_joint_student_response.py``.
The materialization path and original GCN student remain unchanged.  The
selector uses an early/mid/terminal response curve, gives the terminal point
the largest weight, and records selector-vs-report ranking regret after the
report-only evaluation.  Report labels never influence the decision.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch

import run_student_compatible_append as SCA
import run_tgc_joint_student_response as V1
import run_unified_v3_continual as U


DATASETS = V1.DATASETS
DEFAULT_SELECTION_SEEDS = (300, 301, 302)
DEFAULT_REPORT_SEEDS = V1.DEFAULT_REPORT_SEEDS


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True), encoding="utf-8")


def stage_curve(total_rows: int, early: int = 128, middle: int = 512) -> list[int]:
    terminal = U.protocol_v3.student_steps(int(total_rows))
    return sorted(set(max(50, min(int(terminal), int(v)))
                   for v in (early, middle, terminal)))


def _old_gain(candidate_grid: dict, reference_grid: dict) -> float:
    candidate = candidate_grid.get("old_mean_f1")
    reference = reference_grid.get("old_mean_f1")
    if candidate is None or reference is None:
        return 0.0
    return float(candidate["mean"] - reference["mean"])


def response_summary_v2(response: dict, reference: dict,
                        weights: list[float] | None = None,
                        old_weight: float = 0.25,
                        variance_weight: float = 0.10,
                        rank_penalty_weight: float = 0.25) -> dict:
    """Use a terminal-weighted curve; early points are diagnostic, not vetoes."""
    grids = response["grids"]
    ref_grids = reference["grids"]
    if len(grids) != len(ref_grids):
        raise ValueError("response/reference curve length mismatch")
    if weights is None:
        weights = [0.15, 0.30, 0.55] if len(grids) == 3 else None
    if weights is None or len(weights) != len(grids):
        weights = [1.0 / len(grids)] * len(grids)
    weights = np.asarray(weights, dtype=float)
    weights /= weights.sum()
    point_gains, old_gains, variances = [], [], []
    for candidate_grid, reference_grid in zip(grids, ref_grids):
        new_gain = float(candidate_grid["current_f1"]["mean"]
                         - reference_grid["current_f1"]["mean"])
        old_gain = _old_gain(candidate_grid, reference_grid)
        current_std = float(candidate_grid["current_f1"].get("std", 0.0))
        candidate_old = candidate_grid.get("old_mean_f1")
        old_std = 0.0 if candidate_old is None else float(candidate_old.get("std", 0.0))
        point_gains.append(new_gain + float(old_weight) * old_gain)
        old_gains.append(old_gain)
        variances.append(current_std + old_std)
    weighted_gain = float(np.dot(weights, np.asarray(point_gains)))
    weighted_old = float(np.dot(weights, np.asarray(old_gains)))
    variance = float(np.dot(weights, np.asarray(variances)))
    terminal_gain = float(point_gains[-1])
    terminal_old = float(old_gains[-1])
    return {
        "curve_weights": weights.tolist(),
        "point_gains": point_gains,
        "old_gains": old_gains,
        "weighted_gain": weighted_gain,
        "weighted_old_delta": weighted_old,
        "terminal_gain": terminal_gain,
        "terminal_old_delta": terminal_old,
        "variance_penalty": variance,
        "rank_penalty": 0.0,
        "utility": weighted_gain - float(variance_weight) * variance,
    }


def choose_candidate_v2(candidates: dict[str, dict], data: dict, stage: int,
                        selection_seeds: list[int], selection_steps: list[int],
                        device: str, old_weight: float,
                        variance_weight: float, rank_penalty_weight: float,
                        retention_tolerance: float,
                        terminal_gain_tolerance: float) -> tuple[str, dict]:
    reference_name = "fallback_content"
    reference_response = SCA.evaluate_response_curve(
        candidates[reference_name], stage, selection_seeds, [16], selection_steps,
        device, data["x_device"], data["real_adj"], data["prompt_emb"],
        data["y"], data["selector_dev_mask"])
    reports = []
    for name, memory in candidates.items():
        response = (reference_response if name == reference_name else
                    SCA.evaluate_response_curve(
                        memory, stage, selection_seeds, [16], selection_steps,
                        device, data["x_device"], data["real_adj"],
                        data["prompt_emb"], data["y"],
                        data["selector_dev_mask"]))
        summary = response_summary_v2(response, reference_response,
                                       old_weight=old_weight,
                                       variance_weight=variance_weight)
        reports.append({"candidate": name, "memory_sha256": V1.memory_hash(memory),
                        "response": response, **summary})

    # Ranking consistency is computed only from selector response conditions.
    SCA.add_rank_consistency(reports, selection_steps, [16])
    for row in reports:
        score = float(row["rank_consistency"]["score"])
        row["rank_penalty"] = float(rank_penalty_weight) * (1.0 - score)
        row["utility"] = float(row["utility"] - row["rank_penalty"])

    fallback = next(row for row in reports if row["candidate"] == reference_name)
    fallback.update({"weighted_gain": 0.0, "weighted_old_delta": 0.0,
                     "terminal_gain": 0.0, "terminal_old_delta": 0.0,
                     "variance_penalty": 0.0, "rank_penalty": 0.0,
                     "utility": 0.0, "eligible": True})
    for row in reports:
        if row["candidate"] == reference_name:
            continue
        row["eligible"] = bool(
            row["terminal_old_delta"] >= -float(retention_tolerance)
            and row["terminal_gain"] >= -float(terminal_gain_tolerance)
            and row["utility"] >= 0.0
        )

    eligible = [row for row in reports if row["eligible"]]
    selected_row = max(
        eligible,
        key=lambda row: (float(row["utility"]), float(row["terminal_gain"]),
                         float(row["weighted_gain"]), row["candidate"]),
    ) if eligible else fallback
    selected = str(selected_row["candidate"])
    reason = ("best_terminal_weighted_eligible_candidate"
              if selected != reference_name else
              "fallback_reference_or_no_eligible_candidate")
    return selected, {
        "selector_version": "stage-aware-terminal-weighted-v2",
        "reference_candidate": reference_name,
        "selection_seeds": [int(v) for v in selection_seeds],
        "selection_steps": [int(v) for v in selection_steps],
        "selection_batches": [16],
        "old_weight": float(old_weight),
        "variance_weight": float(variance_weight),
        "rank_penalty_weight": float(rank_penalty_weight),
        "retention_tolerance": float(retention_tolerance),
        "terminal_gain_tolerance": float(terminal_gain_tolerance),
        "selected_candidate": selected,
        "decision_reason": reason,
        "candidates": {row["candidate"]: row for row in reports},
        "report_labels_used": False,
        "selection_mask": "selector_dev only",
    }


def rank_order(values: dict[str, float]) -> list[str]:
    return [name for name, _ in sorted(values.items(),
                                        key=lambda item: (item[1], item[0]),
                                        reverse=True)]


def rank_correlation(left: list[str], right: list[str]) -> float:
    names = sorted(set(left) | set(right))
    if len(names) < 2:
        return 1.0
    lrank = {name: left.index(name) for name in names}
    rrank = {name: right.index(name) for name in names}
    diff2 = sum((lrank[name] - rrank[name]) ** 2 for name in names)
    n = len(names)
    return float(1.0 - 6.0 * diff2 / (n * (n * n - 1)))


def run_cell(args, dataset: str, order: str) -> None:
    out = Path(args.output) / dataset / order
    if (out / "results.json").exists():
        raise FileExistsError(f"immutable output already exists: {out}")
    out.mkdir(parents=True, exist_ok=True)
    data = V1.load_data(dataset, args.minilm, args.device, Path(args.cache_root))
    ids, budgets = V1.R11.D.set_stream(data, order)
    source = Path(args.source_root)
    protocol = {
        "name": "tgc-shared-assignment-gcn-response-v2",
        "dataset": dataset, "order": order, "stages": list(args.stages),
        "student": "original two-layer GCN + bidirectional InfoNCE",
        "graphgps": "diagnostic only; not used",
        "selector": "terminal-weighted stage-aware response curve with ranking penalty",
        "selection_seeds": [int(v) for v in args.selection_seeds],
        "response_steps": "128/512/canonical per stage",
        "report_seeds": [int(v) for v in args.report_seeds],
        "report_not_used_for_selection": True,
        "recursive_rollout": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    save_json(out / "protocol.json", protocol)
    results = []
    for stage in args.stages:
        candidates, build_audit = V1.build_candidates(data, source, order,
                                                      int(stage), args.device)
        save_json(out / f"s{stage}_build_audit.json", build_audit)
        steps = stage_curve(len(candidates["fallback_content"]["x"]),
                            args.early_steps, args.middle_steps)
        selected, selection = choose_candidate_v2(
            candidates, data, int(stage), list(args.selection_seeds), steps,
            args.device, args.old_weight, args.variance_weight,
            args.rank_penalty_weight, args.retention_tolerance,
            args.terminal_gain_tolerance)
        selection["stage"] = int(stage)
        save_json(out / f"s{stage}_selection.json", selection)
        report = {}
        for name in ("fallback_content", "p_initial", "anchor025", "joint"):
            report[name] = V1.report_candidate(
                candidates[name], data, int(stage), list(args.report_seeds),
                args.device)
            torch.save(candidates[name], out / f"{name}_s{stage}_memory.pt")
        selector_order = rank_order({
            name: float(value["utility"])
            for name, value in selection["candidates"].items()
        })
        report_order = rank_order({
            name: float(value["cumulative_f1_mean"])
            for name, value in report.items()
        })
        ranking_audit = {
            "selector_order": selector_order,
            "report_order": report_order,
            "spearman_like_rank_correlation": rank_correlation(selector_order,
                                                                  report_order),
            "selected_candidate": selected,
            "selected_report_f1": float(report[selected]["cumulative_f1_mean"]),
            "best_report_candidate": report_order[0],
            "best_report_f1": float(report[report_order[0]]["cumulative_f1_mean"]),
            "selection_regret_pp": float(
                report[report_order[0]]["cumulative_f1_mean"]
                - report[selected]["cumulative_f1_mean"]),
            "report_labels_used_for_decision": False,
        }
        save_json(out / f"s{stage}_ranking_audit.json", ranking_audit)
        row = {"stage": int(stage), "budget": int(budgets[stage]),
               "response_steps": steps, "selected_candidate": selected,
               "selection": selection, "report": report,
               "ranking_audit": ranking_audit}
        results.append(row)
        save_json(out / "progress.json", {"results": results})
        print(json.dumps({"event": "stage_complete", "dataset": dataset,
                          "order": order, "stage": int(stage),
                          "selected": selected,
                          "selector_order": selector_order,
                          "report_order": report_order,
                          "rank_corr": ranking_audit[
                              "spearman_like_rank_correlation"],
                          "regret_pp": ranking_audit["selection_regret_pp"]}),
              flush=True)
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
    parser.add_argument("--variance-weight", type=float, default=0.10)
    parser.add_argument("--rank-penalty-weight", type=float, default=0.25)
    parser.add_argument("--retention-tolerance", type=float, default=0.50)
    parser.add_argument("--terminal-gain-tolerance", type=float, default=0.50)
    args = parser.parse_args()
    torch.set_num_threads(2)
    run_cell(args, args.dataset, args.order)
    print("TGC_SHARED_ASSIGNMENT_GCN_RESPONSE_V2_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
