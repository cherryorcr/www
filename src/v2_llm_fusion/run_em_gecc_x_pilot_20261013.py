"""Matched small-graph pilot for the experimental EM-GECC-X constructor.

This runner is isolated from the frozen Unified-Q-EF confirmation path. It
uses one fixed order and construction seed as a go/no-go screen, reports only
development-split metrics, and never reads sealed-test labels for selection.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

import em_gecc_x as EM
import external_continual_mechanisms as M
import graphtext_prompt_student as PS
import run_semantic_recursive_unified_baselines_20260926 as BASE
import run_unified_q_multimodal_20261006 as QRUN
import unified_q_multimodal as UQ


ARMS = ("gecc_hard", "unified_q_ef", "em_gecc_x", "em_no_gm",
        "em_no_geometry")
DATASETS = ("cora", "citeseer")
PILOT_GATE = {
    "comparison": "final-stage report-dev; compare EM-GECC-X with the stronger-F1 of GECC-hard and Unified-Q-EF per dataset",
    "continue_if": {
        "mean_dataset_paired_macro_f1_gain_pp_min": 1.0,
        "worst_dataset_paired_macro_f1_gain_pp_min": -0.5,
        "mean_dataset_paired_accuracy_gain_pp_min": 0.0,
        "worst_dataset_paired_accuracy_gain_pp_min": -1.0,
        "mean_dataset_paired_macro_f1_gain_vs_no_gm_pp_min": 0.5,
    },
    "otherwise": "stop this prototype and continue the Unified-Q-EF paper line",
    "scope": "screening gate only; one order and construction seed are not confirmatory evidence",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_hash(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(list(value.shape)).encode("ascii"))
    digest.update(value.tobytes())
    return digest.hexdigest()


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=True),
                    encoding="utf-8")
    temp.replace(path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def memory_hash(memory: dict) -> str:
    return QRUN._memory_hash(memory)


def add_provenance_fields(block: dict, audit: dict) -> dict:
    """Adapt legacy GECC blocks to the current memory provenance schema."""
    rows = len(block["x"])
    block.setdefault("support_ids", [[] for _ in range(rows)])
    block.setdefault("text_source_ids", audit.get(
        "text_source_ids", [[] for _ in range(rows)]))
    if len(block["support_ids"]) != rows or len(block["text_source_ids"]) != rows:
        raise AssertionError("invalid provenance length in adapted GECC block")
    return block


def validate_memory(memory: dict, parent: dict | None, budget: int) -> dict:
    QRUN._check_memory(memory, parent, budget, preserve_parent=True)
    if memory["edge_index"].numel():
        raise AssertionError("pilot requires edge-free synthetic memory")
    old_n = 0 if parent is None else len(parent["x"])
    if parent is not None:
        for key in ("x", "text_emb", "labels", "mass", "depth", "stage"):
            if not torch.equal(memory[key][:old_n], parent[key]):
                raise AssertionError(f"frozen prefix changed: {key}")
    return {"budget_ok": len(memory["x"]) == int(budget),
            "prefix_ok": parent is None or all(torch.equal(
                memory[key][:old_n], parent[key]) for key in
                ("x", "text_emb", "labels", "mass", "depth", "stage")),
            "edge_free": memory["edge_index"].numel() == 0,
            "finite": all(bool(torch.isfinite(memory[key].float()).all())
                          for key in ("x", "text_emb", "mass"))}


def build_arm(arm: str, delta: dict, parent: dict | None,
              slot_labels: torch.Tensor, encoder, prompt_emb: torch.Tensor,
              device: str, gecc_cfg: M.Config, uq_cfg: UQ.Config,
              em_cfg: EM.Config) -> tuple[dict, dict]:
    rows = len(slot_labels)
    if arm == "gecc_hard":
        block, audit = M.gecc_block(delta, rows, encoder, gecc_cfg, device)
        block = add_provenance_fields(block, audit)
        block["labels"] = slot_labels.detach().cpu().long()
        memory = UQ.append_memory(parent, block)
    elif arm == "unified_q_ef":
        block, audit = UQ.optimize_stage(
            delta, parent, slot_labels, encoder, prompt_emb, uq_cfg, device,
            use_boundary=False, use_topology=False)
        memory = UQ.append_memory(parent, block)
    elif arm in ("em_gecc_x", "em_no_gm", "em_no_geometry"):
        initial, init_audit = M.gecc_block(
            delta, rows, encoder, gecc_cfg, device)
        use_gm = arm != "em_no_gm"
        use_geometry = arm != "em_no_geometry"
        memory, audit = EM.optimize_stage(
            delta, parent, initial["x"], initial["text_emb"], slot_labels,
            encoder, prompt_emb, em_cfg, device,
            use_gradient_matching=use_gm, use_geometry=use_geometry)
        audit["gecc_initialization_audit"] = init_audit
    else:
        raise ValueError(f"unknown arm: {arm}")
    return memory, audit


def evaluate(memory: dict, data: dict, stage: int, seeds: list[int],
             steps: int, device: str) -> dict:
    rows = []
    for seed in seeds:
        model = PS.train_prompt_student(
            memory, data["prompt_emb"], int(seed), device, steps,
            lr=0.001, batch=24)
        score = BASE.score_model(model, data, data["stage_classes"], stage)
        rows.append({"seed": int(seed), **score})
        del model
    result = {"student_seeds": list(seeds), "student_steps": int(steps),
              "split": "report_dev_only", "per_student": rows}
    for key in ("cumulative_f1", "accuracy", "old_f1", "new_f1"):
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        result[key] = float(np.mean(values)) if values else None
        result[key + "_std"] = (float(np.std(values, ddof=1))
                                 if len(values) > 1 else 0.0 if values else None)
    return result


def make_protocol(args, dataset: str, data: dict, groups: list,
                  stage_ids: list[torch.Tensor], budgets: list[int],
                  gecc_cfg: M.Config, uq_cfg: UQ.Config,
                  em_cfg: EM.Config) -> dict:
    return {
        "name": "em-gecc-x-small-graph-go-no-go-pilot-20261013-v1",
        "status": "report-development pilot; one order and construction seed",
        "dataset": dataset,
        "order": args.order,
        "stage_classes": groups,
        "stage_raw_rows": [len(ids) for ids in stage_ids],
        "budgets": budgets,
        "memory_rate": 0.05,
        "budget_rule": "floor(0.05 * cumulative current legal-pool rows), with the frozen order helper's class feasibility rule",
        "construction_seed": args.construction_seed,
        "student_seeds": list(args.student_seeds),
        "student_steps": "same protocol_v3.student_steps(memory_rows) for every arm",
        "student": PS.model_config(),
        "text_encoder": str(args.minilm),
        "arms": list(ARMS),
        "arm_definition": {
            "gecc_hard": "current-delta GECC hard clustering and its native hard text payload",
            "unified_q_ef": "frozen Unified-Q-EF constructor; no synthetic edges",
            "em_gecc_x": "GECC X initialization, alternating class-balanced E-step and free-new-X GM M-step with Q-conditioned geometry",
            "em_no_gm": "same EM path without Student gradient matching",
            "em_no_geometry": "same EM path without the Q-conditioned geometry tether",
        },
        "access": "current raw delta only plus each arm's frozen synthetic prefix; zero historical raw reads",
        "old_memory": "frozen; no old X or payload updates",
        "synthetic_topology": "none for every arm",
        "selection": "fixed before evaluation; report-dev only; test split not evaluated",
        "precommitted_go_no_go_gate": PILOT_GATE,
        "official_reproduction": False,
        "protocol_hashes": {
            name: sha256_file(path) for name, path in {
                "em_gecc_x.py": EM.__file__,
                "external_continual_mechanisms.py": M.__file__,
                "unified_q_multimodal.py": UQ.__file__,
                "graphtext_prompt_student.py": PS.__file__,
                "run_unified_q_multimodal_20261006.py": QRUN.__file__,
                "protocol_v3.py": BASE.U.protocol_v3.__file__,
            }.items()
        },
        "configs": {"gecc": asdict(gecc_cfg), "unified_q": asdict(uq_cfg),
                    "em_gecc_x": asdict(em_cfg)},
        "masks": {
            "pool": tensor_hash(data["pool_mask"]),
            "report_dev": tensor_hash(data["report_dev_mask"]),
        },
    }


def run_dataset(args, dataset: str, output_root: Path,
                gecc_cfg: M.Config, uq_cfg: UQ.Config,
                em_cfg: EM.Config) -> None:
    data = BASE.load_data(dataset, args.minilm, args.device, args.cache_root)
    encoded_prompts = data["encoder"].encode(data["prompts"])
    data["prompt_emb"] = (encoded_prompts if isinstance(encoded_prompts, torch.Tensor)
                          else torch.as_tensor(encoded_prompts, dtype=torch.float32))
    stage_ids, budgets = BASE.set_stream(data, args.order)
    groups = data["stage_classes"]
    output = output_root / dataset / args.order
    output.mkdir(parents=True, exist_ok=True)
    protocol = make_protocol(args, dataset, data, groups, stage_ids, budgets,
                             gecc_cfg, uq_cfg, em_cfg)
    protocol_path = output / "protocol.json"
    result_path = output / "results.json"
    if protocol_path.exists():
        if read_json(protocol_path) != protocol:
            raise ValueError(f"existing pilot protocol differs: {output}")
    else:
        save_json(protocol_path, protocol)
    if result_path.exists():
        result = read_json(result_path)
        if result.get("protocol") != protocol:
            raise ValueError(f"existing result protocol differs: {output}")
    else:
        result = {"protocol": protocol, "rows": [], "status": "running"}

    completed = {(row["arm"], int(row["stage"])) for row in result["rows"]}
    for arm in ARMS:
        parent = None
        for stage, ids in enumerate(stage_ids):
            memory_path = output / f"{arm}_s{stage}_memory.pt"
            key = (arm, stage)
            if key in completed:
                if not memory_path.exists():
                    raise FileNotFoundError(f"completed memory missing: {memory_path}")
                parent = torch.load(memory_path, map_location="cpu", weights_only=False)
                continue
            delta = QRUN.delta_view(data, ids, stage,
                                    torch.cat(stage_ids[:stage]) if stage else
                                    torch.empty(0, dtype=torch.long))
            new_rows = int(budgets[stage] - (len(parent["x"]) if parent is not None else 0))
            slot_labels = M.row_labels(M.ratio_allocation(delta["y"], new_rows))
            if str(args.device).startswith("cuda"):
                torch.cuda.synchronize(args.device)
                torch.cuda.reset_peak_memory_stats(args.device)
            started = time.perf_counter()
            memory, audit = build_arm(
                arm, delta, parent, slot_labels, data["encoder"],
                data["prompt_emb"], args.device, gecc_cfg, uq_cfg, em_cfg)
            if str(args.device).startswith("cuda"):
                torch.cuda.synchronize(args.device)
                peak_mb = float(torch.cuda.max_memory_allocated(args.device) / 2**20)
            else:
                peak_mb = None
            construction_seconds = float(time.perf_counter() - started)
            checks = validate_memory(memory, parent, budgets[stage])
            steps = int(args.student_steps or
                        BASE.U.protocol_v3.student_steps(len(memory["x"])))
            scores = evaluate(memory, data, stage, args.student_seeds, steps,
                              args.device)
            torch.save(memory, memory_path)
            row = {
                "arm": arm, "stage": int(stage),
                "stage_classes": list(groups[stage]),
                "raw_rows": int(len(ids)), "new_rows": new_rows,
                "memory_rows": int(len(memory["x"])),
                "construction_seconds": construction_seconds,
                "construction_peak_mb": peak_mb,
                "memory_sha256": memory_hash(memory),
                "parent_sha256": memory_hash(parent) if parent is not None else None,
                "checks": checks,
                "audit": audit,
                "scores": scores,
            }
            result["rows"].append(row)
            result["rows"].sort(key=lambda item: (ARMS.index(item["arm"]),
                                                  int(item["stage"])))
            save_json(result_path, result)
            parent = memory
            print(json.dumps({"event": "arm_stage_complete", "dataset": dataset,
                              "arm": arm, "stage": stage,
                              "report_f1": scores["cumulative_f1"],
                              "accuracy": scores["accuracy"],
                              "construction_seconds": construction_seconds},
                             ensure_ascii=True), flush=True)
    result["status"] = "complete"
    save_json(result_path, result)


def paired_delta(candidate: dict, reference: dict, metric: str) -> dict:
    candidate_rows = {int(row["seed"]): float(row[metric])
                      for row in candidate["scores"]["per_student"]}
    reference_rows = {int(row["seed"]): float(row[metric])
                      for row in reference["scores"]["per_student"]}
    seeds = sorted(candidate_rows.keys() & reference_rows.keys())
    if not seeds:
        raise ValueError(f"no paired Student seeds for {metric}")
    differences = [candidate_rows[seed] - reference_rows[seed]
                   for seed in seeds]
    return {
        "n": len(seeds),
        "student_seeds": seeds,
        "mean": float(np.mean(differences)),
        "std": float(np.std(differences, ddof=1)) if len(differences) > 1 else 0.0,
        "per_seed": differences,
    }


def pilot_decision(dataset_summaries: list[dict]) -> dict:
    f1_gains = [row["f1_gain_vs_best_baseline"]["mean"]
                for row in dataset_summaries]
    acc_gains = [row["accuracy_gain_vs_best_baseline"]["mean"]
                 for row in dataset_summaries]
    gm_gains = [row["f1_gain_vs_no_gm"]["mean"]
                for row in dataset_summaries]
    checks = {
        "mean_f1_gain": float(np.mean(f1_gains)) >= 1.0,
        "worst_dataset_f1": min(f1_gains) >= -0.5,
        "mean_accuracy_gain": float(np.mean(acc_gains)) >= 0.0,
        "worst_dataset_accuracy": min(acc_gains) >= -1.0,
        "gradient_matching_contribution": float(np.mean(gm_gains)) >= 0.5,
    }
    return {
        "gate": PILOT_GATE,
        "checks": checks,
        "metrics": {
            "mean_dataset_f1_gain_pp": float(np.mean(f1_gains)),
            "worst_dataset_f1_gain_pp": float(min(f1_gains)),
            "mean_dataset_accuracy_gain_pp": float(np.mean(acc_gains)),
            "worst_dataset_accuracy_gain_pp": float(min(acc_gains)),
            "mean_dataset_f1_gain_vs_no_gm_pp": float(np.mean(gm_gains)),
        },
        "decision": ("continue_em_multiseed" if all(checks.values())
                     else "stop_em_and_continue_unified_q_ef"),
        "confirmatory_evidence": False,
    }


def summarize_and_decide(args) -> dict:
    summaries = []
    for dataset in args.datasets:
        result_path = args.output_root / dataset / args.order / "results.json"
        result = read_json(result_path)
        if result.get("status") != "complete":
            raise ValueError(f"incomplete pilot result: {result_path}")
        final_stage = max(int(row["stage"]) for row in result["rows"])
        final_rows = {row["arm"]: row for row in result["rows"]
                      if int(row["stage"]) == final_stage}
        baselines = ("gecc_hard", "unified_q_ef")
        best_baseline = max(
            baselines,
            key=lambda arm: float(final_rows[arm]["scores"]["cumulative_f1"]))
        summaries.append({
            "dataset": dataset,
            "stage": final_stage,
            "best_baseline_by_report_dev_f1": best_baseline,
            "f1_gain_vs_best_baseline": paired_delta(
                final_rows["em_gecc_x"], final_rows[best_baseline],
                "cumulative_f1"),
            "accuracy_gain_vs_best_baseline": paired_delta(
                final_rows["em_gecc_x"], final_rows[best_baseline],
                "accuracy"),
            "f1_gain_vs_no_gm": paired_delta(
                final_rows["em_gecc_x"], final_rows["em_no_gm"],
                "cumulative_f1"),
            "f1_gain_vs_no_geometry": paired_delta(
                final_rows["em_gecc_x"], final_rows["em_no_geometry"],
                "cumulative_f1"),
            "final_stage_scores": {
                arm: final_rows[arm]["scores"] for arm in ARMS
            },
        })
    decision = pilot_decision(summaries)
    decision["protocol_name"] = "em-gecc-x-small-graph-go-no-go-pilot-20261013-v1"
    decision["dataset_summaries"] = summaries
    decision_path = args.output_root / "decision.json"
    if decision_path.exists() and read_json(decision_path) != decision:
        raise ValueError(f"existing pilot decision differs: {decision_path}")
    if not decision_path.exists():
        save_json(decision_path, decision)
    return decision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--minilm", required=True)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS,
                        default=list(DATASETS))
    parser.add_argument("--order", default="order19611")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--construction-seed", type=int, default=271006)
    parser.add_argument("--student-seeds", nargs="+", type=int,
                        default=[150, 151, 152])
    parser.add_argument("--student-steps", type=int, default=None)
    parser.add_argument("--gecc-restarts", type=int, default=20)
    parser.add_argument("--uq-steps", type=int, default=30)
    parser.add_argument("--em-rounds", type=int, default=4)
    parser.add_argument("--em-m-steps", type=int, default=5)
    args = parser.parse_args()
    torch.set_num_threads(2)
    gecc_cfg = M.Config(seed=args.construction_seed,
                        gecc_restarts=args.gecc_restarts,
                        text_chars=320)
    uq_cfg = UQ.Config(steps=args.uq_steps, proxy_seeds=(2711,),
                       materialize_top_k=1, straight_through_text=True,
                       hard_topology_forward=True)
    em_cfg = EM.Config(seed=args.construction_seed, em_rounds=args.em_rounds,
                       m_steps=args.em_m_steps, text_chars=320)
    args.output_root.mkdir(parents=True, exist_ok=True)
    for dataset in args.datasets:
        run_dataset(args, dataset, args.output_root, gecc_cfg, uq_cfg, em_cfg)
    decision = summarize_and_decide(args)
    print(json.dumps({"event": "pilot_gate_complete",
                      "decision": decision["decision"],
                      "metrics": decision["metrics"]}, ensure_ascii=True),
          flush=True)


if __name__ == "__main__":
    main()
