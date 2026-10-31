"""Read-only checks of the completed EM pilot; never import its constructors."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


ARMS = ("gecc_hard", "unified_q_ef", "em_gecc_x", "em_no_gm", "em_no_geometry")
TENSORS = ("x", "text_emb", "labels", "mass", "depth", "stage")
LEGACY_UQ_HASH = "f621d4575a9bb660fad8031e4d7135e2570cc72c02d16145c08aa989bee99525"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def memory_hash(memory: dict) -> str:
    digest = hashlib.sha256(json.dumps(memory["texts"], ensure_ascii=True).encode())
    for key in (*TENSORS, "edge_index"):
        value = memory[key].detach().cpu().contiguous()
        digest.update(f"{key}:{value.dtype}:{list(value.shape)}".encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def tensor_hash(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(list(value.shape)).encode("ascii"))
    digest.update(value.tobytes())
    return digest.hexdigest()


def check_memory(memory: dict, parent: dict | None, row: dict,
                 protocol: dict) -> dict[str, bool]:
    n, stage = len(memory["x"]), int(row["stage"])
    old_n = 0 if parent is None else len(parent["x"])
    checks = {
        "sha256": memory_hash(memory) == row["memory_sha256"],
        "parent_sha256": row["parent_sha256"] == (
            None if parent is None else memory_hash(parent)),
        "budget": n == row["memory_rows"] == protocol["budgets"][stage],
        "new_rows": n - old_n == row["new_rows"],
        "raw_rows": row["raw_rows"] == protocol["stage_raw_rows"][stage],
        "finite": all(bool(torch.isfinite(memory[k]).all()) for k in TENSORS),
        "edge_free": tuple(memory["edge_index"].shape) == (2, 0),
        "row_shapes": all(len(memory[k]) == n for k in TENSORS)
            and len(memory["texts"]) == len(memory["text_source_ids"]) == n,
        "prefix_tensors": parent is None or all(
            torch.equal(memory[k][:old_n], parent[k]) for k in TENSORS),
        "prefix_strings": parent is None or memory["texts"][:old_n] == parent["texts"],
        "prefix_sources": parent is None or all(
            k in memory and memory[k][:old_n] == parent[k]
            for k in ("text_source_ids", "support_ids") if k in parent),
        "new_stage": bool((memory["stage"][old_n:] == stage).all()),
        "new_classes": set(memory["labels"][old_n:].tolist())
            == set(protocol["stage_classes"][stage]),
        "student_seeds": row["scores"]["student_seeds"] == protocol["student_seeds"]
            == [s["seed"] for s in row["scores"]["per_student"]],
        "report_dev_only": row["scores"]["split"] == "report_dev_only",
        "reported_checks": all(row["checks"].values()),
        "scores_recomputed": all(
            math.isclose(row["scores"][key], statistics.mean(
                s[key] for s in row["scores"]["per_student"]), abs_tol=1e-9)
            and math.isclose(row["scores"][key + "_std"], statistics.stdev(
                s[key] for s in row["scores"]["per_student"]), abs_tol=1e-9)
            for key in ("cumulative_f1", "accuracy")),
    }
    return checks


def class_center_cosine(x: torch.Tensor, labels: torch.Tensor) -> float | None:
    centers = torch.stack([x[labels == c].mean(0) for c in torch.unique(labels)])
    if len(centers) < 2:
        return None
    centers = F.normalize(centers.float(), dim=1)
    off = ~torch.eye(len(centers), dtype=torch.bool)
    return float((centers @ centers.T)[off].mean())


def source_checks(memory: dict, old_n: int, ids: torch.Tensor,
                  raw_labels: torch.Tensor) -> dict[str, bool]:
    lookup = {int(i): int(raw_labels[int(i)]) for i in ids}
    sources = memory["text_source_ids"][old_n:]
    labels = memory["labels"][old_n:].tolist()
    return {
        "source_row_count": len(sources) == len(labels),
        "source_ids_current_delta": all(len(s) > 0 and all(i in lookup for i in s)
                                        for s in sources),
        "source_labels_match": all(all(lookup.get(i) == label for i in s)
                                   for s, label in zip(sources, labels)),
    }


def final_hard_switches(audit: dict, ids: torch.Tensor) -> dict:
    last = ids[torch.tensor(audit["e_trace"][-1]["selected_local_rows"])].tolist()
    saved = audit["selected_text_ids"]
    if len(last) != len(saved):
        raise ValueError("last-M-step and final E-step source counts differ")
    return {"final_e_step_switch_count": sum(a != b for a, b in zip(last, saved)),
            "final_e_step_slot_count": len(saved)}


def hard_routing_trace(audit: dict) -> dict:
    selections = [step["selected_local_rows"] for step in audit["e_trace"]]
    if not selections or not selections[0]:
        raise ValueError("hard E-step trace is empty")
    slots = len(selections[0])
    if any(len(selection) != slots for selection in selections):
        raise ValueError("hard E-step trace source counts differ")
    changed = set()
    switches = 0
    for previous, current in zip(selections, selections[1:]):
        for slot, (left, right) in enumerate(zip(previous, current)):
            if left != right:
                switches += 1
                changed.add(slot)
    return {
        "e_round_count": len(selections),
        "e_round_hard_switch_count": switches,
        "e_round_slot_comparisons": (len(selections) - 1) * slots,
        "e_round_changed_slot_ratio": len(changed) / slots,
    }


def raw_graph_context(graph_root: Path, code_root: Path, dataset: str,
                      protocol: dict) -> tuple[dict, dict]:
    path = code_root / "protocol_v3.py"
    expected = protocol["protocol_hashes"][path.name]
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError("raw-source audit requires the recorded protocol module")
    spec = importlib.util.spec_from_file_location("em_pilot_audit_protocol", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    graph = torch.load(graph_root / f"{dataset}.pt", map_location="cpu", weights_only=False)
    y = graph.y.reshape(-1).long()
    ids = torch.arange(len(y))
    _, report, _, pool = module.split_masks_v4(ids)
    groups = protocol["stage_classes"]
    stage_ids = [torch.where(pool & torch.isin(y, torch.tensor(c)))[0] for c in groups]
    checks = {
        "pool_hash": tensor_hash(pool) == protocol["masks"]["pool"],
        "report_hash": tensor_hash(report) == protocol["masks"]["report_dev"],
        "stage_raw_counts": [len(v) for v in stage_ids] == protocol["stage_raw_rows"],
        "exact_5pct_schedule": [int(.05 * sum(len(v) for v in stage_ids[:i + 1]))
                                for i in range(len(stage_ids))] == protocol["budgets"],
    }
    if not all(checks.values()):
        raise ValueError(f"raw graph/protocol mismatch: {checks}")
    return {"stage_ids": stage_ids, "y": y, "module": module}, checks


def audit_run(root: Path, graph_root: Path | None = None,
              code_root: Path | None = None) -> dict:
    if (graph_root is None) != (code_root is None):
        raise ValueError("graph-root and code-root must be provided together")
    cells, datasets = [], {}
    fingerprints = {"decision.json": file_hash(root / "decision.json")}
    for dataset in ("cora", "citeseer"):
        directory = root / dataset / "order19611"
        protocol = read_json(directory / "protocol.json")
        result = read_json(directory / "results.json")
        for name in ("protocol.json", "results.json"):
            path = directory / name
            fingerprints[path.relative_to(root).as_posix()] = file_hash(path)
        if result["protocol"] != protocol or result["status"] != "complete":
            raise ValueError(f"incomplete or inconsistent results: {dataset}")
        keyed = {(r["arm"], r["stage"]): r for r in result["rows"]}
        if len(keyed) != len(result["rows"]) or len(keyed) != len(ARMS) * len(protocol["budgets"]):
            raise ValueError(f"missing or duplicate cells: {dataset}")
        context, graph_checks = (None, {}) if graph_root is None else raw_graph_context(
            graph_root, code_root, dataset, protocol)
        datasets[dataset] = {
            "raw_graph_checks": graph_checks,
            "recorded_code_hashes": protocol["protocol_hashes"],
            "uq_legacy_code": protocol["protocol_hashes"]["unified_q_multimodal.py"] == LEGACY_UQ_HASH,
            "uq_dm_off": protocol["configs"]["unified_q"]["lambda_distribution"] == 0
                and protocol["configs"]["unified_q"]["lambda_joint_distribution"] == 0,
            "budgets": protocol["budgets"],
            "common_student_steps": [],
        }
        for stage in range(len(protocol["budgets"])):
            steps = {keyed[(arm, stage)]["scores"]["student_steps"] for arm in ARMS}
            if len(steps) != 1 or (context and next(iter(steps)) !=
                                  context["module"].student_steps(protocol["budgets"][stage])):
                raise ValueError("Student training budgets differ")
            datasets[dataset]["common_student_steps"].append(next(iter(steps)))
        memories = {key: torch.load(directory / f"{key[0]}_s{key[1]}_memory.pt",
                    map_location="cpu", weights_only=False) for key in keyed}
        for arm, stage in keyed:
            path = directory / f"{arm}_s{stage}_memory.pt"
            fingerprints[path.relative_to(root).as_posix()] = file_hash(path)
        for arm in ARMS:
            parent = None
            for stage in range(len(protocol["budgets"])):
                row, memory = keyed[(arm, stage)], memories[(arm, stage)]
                old_n = 0 if parent is None else len(parent["x"])
                checks = check_memory(memory, parent, row, protocol)
                if context:
                    checks.update(source_checks(memory, old_n, context["stage_ids"][stage], context["y"]))
                audit = row["audit"]
                diagnostics = {"normalized_new_class_center_cosine": class_center_cosine(
                    memory["x"][old_n:], memory["labels"][old_n:])}
                if arm.startswith("em_"):
                    initial = memories[("gecc_hard", stage)]["x"][old_n:]
                    drift = (memory["x"][old_n:] - initial).norm()
                    checks["same_gecc_initialization_record"] = (
                        audit["gecc_initialization_audit"]["clustering"] ==
                        keyed[("gecc_hard", stage)]["audit"]["clustering"])
                    checks["x_drift_recomputed"] = math.isclose(
                        float(drift), audit["new_x_update_norm"], rel_tol=1e-5, abs_tol=1e-5)
                    checks["selected_sources_saved"] = [s[0] for s in
                        memory["text_source_ids"][old_n:]] == audit["selected_text_ids"]
                    checks["unique_hard_sources"] = len(set(audit["selected_text_ids"])) == row["new_rows"]
                    checks["m_step_count"] = len(audit["trace"]) == audit["total_m_steps"] == 20
                    diagnostics.update({
                        "x_drift_relative_to_gecc": float(drift / initial.norm().clamp_min(1e-8)),
                        "source_given_slot_entropy_normalized": audit["q_entropy_normalized"],
                        "diagnostic_soft_hard_cosine": audit["soft_hard_text_cosine_mean"],
                        "legacy_unnormalized_center_dot_product": audit["class_center_cosine_mean"],
                        "slot_text_coverage": audit["slot_text_coverage"],
                        "distinct_payload_ratio": len(set(memory["texts"][old_n:])) / row["new_rows"],
                        "selected_text_pairwise_cosine": audit["selected_text_pairwise_cosine_mean"],
                        "reference_new_row_fraction": row["raw_rows"] / (old_n + row["raw_rows"]),
                        "synthetic_new_row_fraction": row["new_rows"] / row["memory_rows"],
                        "first_m_objective": audit["trace"][0]["objective"],
                        "last_m_objective": audit["trace"][-1]["objective"],
                        "same_final_hard_sources_across_em_arms": all(
                            keyed[(candidate, stage)]["audit"]["selected_text_ids"]
                            == audit["selected_text_ids"]
                            for candidate in ARMS if candidate.startswith("em_")),
                    })
                    diagnostics.update(hard_routing_trace(audit))
                    if context:
                        diagnostics.update(final_hard_switches(audit, context["stage_ids"][stage]))
                cells.append({"dataset": dataset, "arm": arm, "stage": stage,
                              "checks": checks, "diagnostics": diagnostics})
                parent = memory
    passed = all(all(c["checks"].values()) for c in cells)
    return {
        "scope": "post-hoc artifact audit; two small graphs, one order, one construction seed; not a training run",
        "all_artifact_checks_passed": passed,
        "memory_count": len(cells),
        "raw_source_legality_audited": graph_root is not None,
        "raw_artifact_hashes": fingerprints,
        "limitations": ["No encoder rerun; source legality does not independently reproduce text embeddings.",
                        "Artifact checks do not measure runtime historical raw reads.",
                        "The legacy Unified-Q pilot is not the frozen paper constructor.",
                        "Soft/hard cosine is diagnostic only; M-steps used hard text embeddings."],
        "datasets": datasets,
        "cells": cells,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--graph-root", type=Path)
    parser.add_argument("--code-root", type=Path)
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    if output == root or root in output.parents:
        raise ValueError("audit output must be outside the immutable raw run")
    report = audit_run(root, args.graph_root, args.code_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps({k: report[k] for k in (
        "memory_count", "all_artifact_checks_passed", "raw_source_legality_audited")}))
    if not report["all_artifact_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
