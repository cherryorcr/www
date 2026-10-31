"""Run the four remaining CoTAG-Q evidence tracks without touching the paper.

Tracks:
  1. fresh sealed confirmation with an unused original-id modulo split;
  2. fixed-character text-budget controls on frozen primary memories;
  3. construction/Student/inference resource decomposition;
  4. aggregation of the existing no-graph control and the new tracks.

The script is deployed to the remote TAG_condensation checkout. It writes only
under results/uq_four_priority_20261017 and never changes frozen artifacts.
"""
from __future__ import annotations

import argparse
import csv
import gc
import gzip
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score


ROOT = Path("/home/pc/research/TAG_condensation")
FROZEN = ROOT / "experiments/uq_final_evidence_20261012/code"
sys.path.insert(0, str(FROZEN))
sys.path.insert(1, str(ROOT / "code/v2_llm_fusion"))

import graphtext_prompt_student as P  # noqa: E402
import run_unified_q_large_smoke_20261007 as L  # noqa: E402
import run_unified_q_matrix_20261008 as R  # noqa: E402
import run_uq_final_confirmation_20261012 as E  # noqa: E402


SEEDS = (271006, 271108, 271109)
STUDENT_SEEDS = (150, 151, 152)
DATASETS = ("arxiv", "sportsfit", "bookhis")
ARMS = ("uq_dm_edgefree", "cat_cctp", "gecc_hard")


def jwrite(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=True), encoding="utf-8")
    tmp.replace(path)


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_gzip(path: Path) -> dict:
    with gzip.open(path, "rb") as stream:
        return torch.load(stream, map_location="cpu", weights_only=False)


def sync(device: str) -> None:
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def fresh_split_masks(ids: torch.Tensor, report_dev_mod: int | None = 3):
    """Use modulo 4 as a new held-out split; selector/report stay unchanged."""
    selector = (ids % 10) == 1
    report = ((ids % 10) == int(report_dev_mod)
              if report_dev_mod is not None else selector & False)
    test = (ids % 10) == 4
    pool = ~selector & ~report & ~test
    return selector, report, test, pool


def load_fresh_data(dataset: str, minilm: str, cache: Path, device: str,
                    groups: list[list[int]]) -> dict:
    """Load the raw graph with the fresh split and zero text outside its pool."""
    # The loader calls this module-level protocol function. The replacement is
    # local to this process and never changes the frozen source file.
    original = L.U.protocol_v3.split_masks_v4
    L.U.protocol_v3.split_masks_v4 = fresh_split_masks
    try:
        data = L.U.load_unified(
            dataset, L.U.DATA_CFG[dataset]["path"], minilm, device,
            n_stages=len(groups), split_mode="base_incremental",
            base_fraction=0.4, report_dev_mod=3)
    finally:
        L.U.protocol_v3.split_masks_v4 = original
    text_emb = torch.load(cache / "text_emb_pool_only.pt",
                          map_location="cpu", weights_only=False).float()
    if tuple(text_emb.shape) != (len(data["x"]), 384):
        raise ValueError(f"unexpected text cache shape: {tuple(text_emb.shape)}")
    # Reuse the old cache where possible, then fill rows that become legal only
    # under the fresh split (notably the old modulo-2 holdout). This keeps the
    # new split genuinely independent without putting the full text cache on
    # disk a second time.
    text_emb[~data["pool_mask"]] = 0.0
    missing = data["pool_mask"] & (text_emb.abs().sum(1) == 0)
    missing_ids = torch.where(missing)[0].tolist()
    for start in range(0, len(missing_ids), 256):
        batch_ids = missing_ids[start:start + 256]
        encoded = torch.as_tensor(
            data["encoder"].encode([data["texts"][int(i)] for i in batch_ids])
        ).float().cpu()
        text_emb[torch.tensor(batch_ids, dtype=torch.long)] = encoded
    data["text_emb"] = text_emb
    data["stage_classes"] = [[int(v) for v in group] for group in groups]
    data["prompt_emb"] = data["encoder"].encode(data["prompts"])
    data["real_adj"] = P.normalized_adjacency(data["edge_index"],
                                               len(data["x"]), device)
    data["x_device"] = data["x"].float().to(device)
    return data


def stage_ids(data: dict) -> list[torch.Tensor]:
    return [torch.sort(torch.where(
        data["pool_mask"] & torch.isin(data["y"], torch.tensor(group))
    )[0]).values for group in data["stage_classes"]]


def score_fresh(memory: dict, data: dict, stage: int, device: str) -> dict:
    prompt = data["prompt_emb"].float().to(device)
    y = data["y"].long().cpu()
    seen = [int(c) for group in data["stage_classes"][:stage + 1] for c in group]
    old = [int(c) for group in data["stage_classes"][:stage] for c in group]
    new = [int(c) for c in data["stage_classes"][stage]]
    masks = {
        "report_dev": data["report_dev_mask"].bool().cpu()
        & torch.isin(y, torch.tensor(seen)),
        "fresh_sealed": data["test_mask"].bool().cpu()
        & torch.isin(y, torch.tensor(seen)),
    }
    rows = {name: [] for name in masks}
    for seed in STUDENT_SEEDS:
        sync(device)
        started = time.perf_counter()
        model = P.train_prompt_student(memory, prompt, int(seed), device,
                                        2048, lr=.001, batch=24)
        sync(device)
        train_seconds = time.perf_counter() - started
        with torch.no_grad():
            pred = model.logits(data["x_device"], data["real_adj"], prompt)
            pred = pred.argmax(1).detach().cpu()
        for name, mask in masks.items():
            yy, pp = y[mask].numpy(), pred[mask].numpy()
            per = f1_score(yy, pp, labels=seen, average=None, zero_division=0)
            class_f1 = {cls: float(value * 100.0)
                        for cls, value in zip(seen, per)}
            rows[name].append({
                "seed": int(seed),
                "final_f1": float(100 * f1_score(
                    yy, pp, labels=seen, average="macro", zero_division=0)),
                "accuracy": float(100 * accuracy_score(yy, pp)),
                "old_f1": float(np.mean([class_f1[c] for c in old])) if old else None,
                "new_f1": float(np.mean([class_f1[c] for c in new])),
                "student_seconds": train_seconds,
            })
        del model
        torch.cuda.empty_cache() if str(device).startswith("cuda") else None

    output = {}
    for name, values in rows.items():
        output[name] = {
            "split": name,
            "student_steps": 2048,
            "student_batch": 24,
            "student_seeds": list(STUDENT_SEEDS),
            "eval_rows": int(masks[name].sum()),
            "per_student": values,
        }
        for key in ("final_f1", "accuracy", "old_f1", "new_f1"):
            nums = [v[key] for v in values if v[key] is not None]
            output[name][key] = float(np.mean(nums)) if nums else None
            output[name][key + "_std"] = (float(np.std(nums, ddof=1))
                                            if len(nums) > 1 else 0.0)
    return output


def run_fresh_cell(args) -> None:
    reference_path = Path(args.reference)
    reference = R.L.read(reference_path)
    groups = R.V.choose_groups(reference, args.order_seed)
    data = load_fresh_data(args.dataset, args.minilm, Path(args.text_cache),
                           args.device, groups)
    ids = stage_ids(data)
    sizes = list(map(len, ids))
    budgets, floors = R.V.choose_budgets(reference, sizes, groups,
                                         "fixed_rate", .05)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    protocol = {
        "name": "cotag-fresh-sealed-mod4-20261017-v1",
        "dataset": args.dataset, "order_seed": args.order_seed,
        "construction_seed": args.construction_seed, "groups": groups,
        "stage_raw_rows": sizes, "budgets": budgets,
        "memory_rate": .05, "budget_mode": "fixed_rate_current_legal_pool",
        "class_floor_adjustments": floors, "test_mod": 4,
        "selector_dev_mod": 1, "report_dev_mod": 3,
        "fresh_test_definition": "original_id % 10 == 4",
        "method_selection_from_fresh_test": False,
        "access": "current-delta-only; frozen synthetic prefix; no historical raw reads",
        "synthetic_topology": "none",
        "student": P.model_config(), "student_seeds": list(STUDENT_SEEDS),
        "student_steps": 2048, "student_batch": 24,
        "pool_mask_hash": R.V.hash_tensor(data["pool_mask"]),
        "report_mask_hash": R.V.hash_tensor(data["report_dev_mask"]),
        "fresh_test_mask_hash": R.V.hash_tensor(data["test_mask"]),
        "stage_id_hashes": [R.V.hash_tensor(v) for v in ids],
        "source_hashes": {
            Path(m.__file__).name: sha_file(Path(m.__file__))
            for m in (R, R.L, R.M, R.P, R.UQ, R.V, R.L.U)
        },
    }
    path = output / "results.json"
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved.get("protocol") != protocol:
            raise ValueError(f"protocol mismatch in {path}")
        if saved.get("status") == "complete":
            print(json.dumps({"event": "skip_complete", "output": str(path)}), flush=True)
            return
    result = {"protocol": protocol, "rows": [], "status": "running"}
    parent_by_arm = {arm: None for arm in ARMS}
    for arm in ARMS:
        parent = None
        for stage, raw_ids in enumerate(ids):
            delta = R.L.delta_view(data, raw_ids, stage)
            old_n = 0 if parent is None else len(parent["x"])
            new_n = budgets[stage] - old_n
            labels = R.M.row_labels(R.M.ratio_allocation(delta["y"], new_n))
            R.V.sync(args.device)
            torch.cuda.reset_peak_memory_stats(args.device)
            started = time.perf_counter()
            if arm == "uq_dm_edgefree":
                block, audit = R.UQ.optimize_stage(
                    delta, parent, labels, data["encoder"], data["prompt_emb"],
                    R.q_config(arm, args.construction_seed, 30, 4096),
                    args.device, use_boundary=False, use_topology=False)
                memory = R.UQ.append_memory(parent, block)
            else:
                block, audit = R.make_baseline(
                    delta, new_n, arm, data["encoder"],
                    args.construction_seed, args.device)
                block["labels"] = labels
                memory = R.L.append_edge_free(parent, block)
            R.V.sync(args.device)
            construction_seconds = time.perf_counter() - started
            peak_mb = torch.cuda.max_memory_allocated(args.device) / 2**20
            checks = R.V.validate_memory(memory, parent, budgets[stage])
            if memory["edge_index"].numel():
                raise AssertionError("fresh edge-free confirmation acquired edges")
            if arm == "uq_dm_edgefree":
                checks.update(R.assignment_checks(audit))
            scores = score_fresh(memory, data, stage, args.device)
            row = {
                "arm": arm, "stage": stage, "raw_rows": len(raw_ids),
                "new_rows": new_n, "memory_rows": len(memory["x"]),
                "construction_seconds": construction_seconds,
                "construction_peak_mb": peak_mb,
                "memory_hash": R.V.memory_hash(memory),
                "scores": scores, "checks": checks,
                "audit": R.compact_audit(audit),
            }
            result["rows"].append(row)
            jwrite(path, result)
            print(json.dumps({"event": "fresh_stage_done", "dataset": args.dataset,
                              "seed": args.construction_seed, "arm": arm,
                              "stage": stage,
                              "f1": scores["fresh_sealed"]["final_f1"],
                              "accuracy": scores["fresh_sealed"]["accuracy"],
                              "construction_seconds": construction_seconds,
                              "peak_mb": peak_mb}), flush=True)
            parent = memory
            del memory
            torch.cuda.empty_cache()
    result["status"] = "complete"
    jwrite(path, result)


def stage_block(memory: dict, ids: torch.Tensor) -> dict:
    ids = ids.long().cpu()
    block = {key: memory[key][ids].clone()
             for key in ("x", "text_emb", "labels", "mass", "depth", "stage")}
    block["texts"] = [memory["texts"][int(i)] for i in ids.tolist()]
    block["edge_index"] = torch.empty((2, 0), dtype=torch.long)
    return block


def text_variant(memory: dict, data: dict, ids: list[torch.Tensor], seed: int,
                 mode: str, sources: int) -> dict:
    text_emb = memory["text_emb"].clone()
    texts = list(memory["texts"])
    source_ids = []
    for stage, raw_ids in enumerate(ids):
        slot_ids = torch.where(memory["stage"].long() == int(stage))[0]
        block = stage_block(memory, slot_ids)
        delta = R.L.delta_view(data, raw_ids, stage)
        cfg = R.M.Config(seed=seed, text_sources=sources,
                         text_mode=mode, text_chars=320)
        altered, audit = R.M.attach_text(delta, delta["x"], block["x"],
                                          block["labels"], data["encoder"], cfg)
        text_emb[slot_ids] = altered["text_emb"].float().cpu()
        for local, global_id in enumerate(slot_ids.tolist()):
            texts[global_id] = altered["texts"][local]
        source_ids.extend(audit["text_source_ids"])
    variant = dict(memory, text_emb=text_emb, texts=texts,
                   text_source_ids=source_ids)
    if not torch.equal(variant["x"], memory["x"]):
        raise AssertionError("text control changed X")
    if variant["labels"] is not memory["labels"] and not torch.equal(variant["labels"], memory["labels"]):
        raise AssertionError("text control changed labels")
    return variant


def reconstruct_primary_memory(reference: dict, data: dict, ids: list[torch.Tensor],
                               dataset: str, construction_seed: int,
                               device: str) -> tuple[dict, str, str | None]:
    """Rebuild a missing frozen primary archive without changing the protocol."""
    budgets, _ = R.V.choose_budgets(reference, list(map(len, ids)),
                                    data["stage_classes"], "fixed_rate", .05)
    parent = None
    for stage, raw_ids in enumerate(ids):
        delta = R.L.delta_view(data, raw_ids, stage)
        old_n = 0 if parent is None else len(parent["x"])
        new_n = budgets[stage] - old_n
        labels = R.M.row_labels(R.M.ratio_allocation(delta["y"], new_n))
        block, _ = R.UQ.optimize_stage(
            delta, parent, labels, data["encoder"], data["prompt_emb"],
            R.q_config("uq_dm_edgefree", construction_seed, 30, 4096),
            device, use_boundary=False, use_topology=False)
        parent = R.UQ.append_memory(parent, block)
    expected = None
    for row in reference.get("rows", []):
        if row.get("arm") == "uq_dm_edgefree" and int(row.get("stage", -1)) == len(ids) - 1:
            expected = row.get("memory_hash")
    actual = R.V.memory_hash(parent)
    source = "protocol-matched-reconstruction"
    if expected and expected != actual:
        source = "protocol-matched-reconstruction-hash-drift"
    return parent, source, expected


def run_text_budget(args) -> None:
    reference = R.L.read(Path(args.reference))
    groups = R.V.choose_groups(reference, args.order_seed)
    data, ids = L.load_large(args.dataset, args.minilm, Path(args.text_cache),
                             args.device, {"groups": groups, "report_dev_mod": 3}), None
    # The loader returns a data dict; stage IDs are derived after the assignment
    # to keep the text-control path explicit.
    ids = L.stage_ids(data)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for seed in SEEDS:
        cell = (ROOT / "results/uq_final_evidence_20261012/confirmation"
                / f"{args.dataset}_o{args.order_seed}_c{seed}")
        memory_path = cell / "final_uq_dm_edgefree.pt.gz"
        memory_source = "archived_final_memory"
        expected_memory_hash = None
        if memory_path.exists():
            memory = load_gzip(memory_path)
        else:
            seed_reference_path = cell / "results.json"
            seed_reference = (R.L.read(seed_reference_path)
                              if seed_reference_path.exists() else reference)
            memory, memory_source, expected_memory_hash = reconstruct_primary_memory(
                seed_reference, data, ids, args.dataset, seed, args.device)
        base_hash = R.V.memory_hash(memory)
        controls = [
            ("primary", None, 1),
            ("nearest1", "nearest", 1),
            ("nearest5", "nearest", 5),
            ("random1", "random", 1),
            ("random5", "random", 5),
        ]
        cell_rows = []
        for name, mode, sources in controls:
            variant = memory if mode is None else text_variant(
                memory, data, ids, seed, mode, sources)
            scores = E.evaluate(variant, data, len(ids) - 1,
                                list(STUDENT_SEEDS), args.device)
            chars = [len(str(v)) for v in variant["texts"]]
            words = [len(str(v).split()) for v in variant["texts"]]
            cell_rows.append({
                "dataset": args.dataset, "order_seed": args.order_seed,
                "construction_seed": seed, "arm": name,
                "memory_hash": base_hash,
                "memory_source": memory_source,
                "expected_memory_hash": expected_memory_hash,
                "x_hash": R.V.hash_tensor(variant["x"]),
                "labels_hash": R.V.hash_tensor(variant["labels"]),
                "fixed_char_budget": 320, "max_chars": max(chars),
                "mean_chars": float(np.mean(chars)),
                "mean_words": float(np.mean(words)),
                "unique_text_ratio": len(set(variant["texts"])) / len(variant["texts"]),
                "source_budget": sources,
                "scores": scores,
            })
            print(json.dumps({"event": "text_budget_done", "dataset": args.dataset,
                              "seed": seed, "arm": name,
                              "f1": scores["sealed_test"]["final_f1"]}), flush=True)
        records.append({"construction_seed": seed, "rows": cell_rows})
        del memory
        torch.cuda.empty_cache()
    protocol = {
        "name": "cotag-text-budget-fairness-20261017-v1",
        "dataset": args.dataset, "order_seed": args.order_seed,
        "source": "frozen primary final_uq_dm_edgefree memories",
        "controls": ["primary", "nearest1", "nearest5", "random1", "random5"],
        "character_budget": 320, "same_X_labels_mass_stage_edges": True,
        "evaluation": "same original report-dev and sealed-test masks; storage-only",
        "method_selection_from_test": False,
    }
    jwrite(output / "results.json", {"protocol": protocol, "records": records,
                                     "status": "complete"})


def measure_student(memory: dict, data: dict, stage: int, device: str) -> dict:
    prompt = data["prompt_emb"].float().to(device)
    rows = []
    for seed in STUDENT_SEEDS:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        sync(device)
        start = time.perf_counter()
        model = P.train_prompt_student(memory, prompt, int(seed), device,
                                        2048, lr=.001, batch=24)
        sync(device)
        train_seconds = time.perf_counter() - start
        train_peak = torch.cuda.max_memory_allocated(device) / 2**20
        torch.cuda.reset_peak_memory_stats(device)
        sync(device)
        start = time.perf_counter()
        with torch.no_grad():
            _ = model.logits(data["x_device"], data["real_adj"], prompt)
        sync(device)
        inference_seconds = time.perf_counter() - start
        inference_peak = torch.cuda.max_memory_allocated(device) / 2**20
        rows.append({"seed": int(seed), "student_train_seconds": train_seconds,
                     "student_train_peak_mb": train_peak,
                     "full_graph_inference_seconds": inference_seconds,
                     "full_graph_inference_peak_mb": inference_peak})
        del model
        torch.cuda.empty_cache()
    return {"per_student": rows,
            "student_train_seconds_mean": float(np.mean([r["student_train_seconds"] for r in rows])),
            "student_train_peak_mb_max": float(max(r["student_train_peak_mb"] for r in rows)),
            "full_graph_inference_seconds_mean": float(np.mean([r["full_graph_inference_seconds"] for r in rows])),
            "full_graph_inference_peak_mb_max": float(max(r["full_graph_inference_peak_mb"] for r in rows))}


def run_resource(args) -> None:
    reference = R.L.read(Path(args.reference))
    groups = R.V.choose_groups(reference, args.order_seed)
    data = L.load_large(args.dataset, args.minilm, Path(args.text_cache),
                        args.device, {"groups": groups, "report_dev_mod": 3})
    data_rows = {
        "dataset": args.dataset, "order_seed": args.order_seed,
        "construction_seed": args.construction_seed,
        "gpu_after_loader_allocated_mb": torch.cuda.memory_allocated(args.device) / 2**20,
        "gpu_after_loader_reserved_mb": torch.cuda.memory_reserved(args.device) / 2**20,
        "raw_x_rows": len(data["x"]), "raw_feature_dim": int(data["x"].shape[1]),
        "raw_edge_count": int(data["edge_index"].shape[1]),
        "arms": [],
    }
    cell = (ROOT / "results/uq_final_evidence_20261012/confirmation"
            / f"{args.dataset}_o{args.order_seed}_c{args.construction_seed}")
    result_json = json.loads((cell / "results.json").read_text(encoding="utf-8"))
    for arm in ARMS:
        archive = cell / f"final_{arm}.pt.gz"
        memory_source = "archived_final_memory"
        if archive.exists():
            start = time.perf_counter()
            memory = load_gzip(archive)
            load_seconds = time.perf_counter() - start
        else:
            # Some older BookHis cells kept hashes/metrics but omitted the
            # compressed final memory. Reconstruct that frozen cell in RAM
            # solely for resource measurement, then verify its final hash.
            start = time.perf_counter()
            ids = L.stage_ids(data)
            budgets, _ = R.V.choose_budgets(reference, list(map(len, ids)),
                                            groups, "fixed_rate", .05)
            parent = None
            for stage, raw_ids in enumerate(ids):
                delta = R.L.delta_view(data, raw_ids, stage)
                old_n = 0 if parent is None else len(parent["x"])
                new_n = budgets[stage] - old_n
                labels = R.M.row_labels(R.M.ratio_allocation(delta["y"], new_n))
                if arm == "uq_dm_edgefree":
                    block, _ = R.UQ.optimize_stage(
                        delta, parent, labels, data["encoder"], data["prompt_emb"],
                        R.q_config(arm, args.construction_seed, 30, 4096),
                        args.device, use_boundary=False, use_topology=False)
                    parent = R.UQ.append_memory(parent, block)
                else:
                    block, _ = R.make_baseline(delta, new_n, arm,
                                               data["encoder"], args.construction_seed,
                                               args.device)
                    block["labels"] = labels
                    parent = R.L.append_edge_free(parent, block)
            memory = parent
            load_seconds = time.perf_counter() - start
            memory_source = "frozen-cell-in-memory-reconstruction"
            expected = [r for r in result_json["rows"] if r["arm"] == arm
                        and int(r["stage"]) == len(ids) - 1]
            reconstructed_hash = R.V.memory_hash(memory)
            expected_hash = expected[0].get("memory_hash") if expected else None
            if expected_hash and reconstructed_hash != expected_hash:
                # The old cell did not retain its final archive. Keep the
                # resource measurement, but expose the endpoint drift instead
                # of presenting the reconstruction as the sealed artifact.
                memory_source = "protocol-matched-reconstruction-hash-drift"
        source_rows = [r for r in result_json["rows"] if r["arm"] == arm]
        construction_seconds = float(sum(r.get("construction_seconds", 0.0)
                                         for r in source_rows))
        construction_peak = float(max(r.get("construction_peak_mb", 0.0)
                                      for r in source_rows))
        payload_bytes = int(sum(v.numel() * v.element_size()
                                for v in memory.values()
                                if isinstance(v, torch.Tensor)))
        student = measure_student(memory, data, len(data["stage_classes"]) - 1,
                                  args.device)
        archive_bytes = int(archive.stat().st_size) if archive.exists() else 0
        data_rows["arms"].append({
            "arm": arm, "memory_rows": len(memory["x"]),
            "archive_bytes": archive_bytes,
            "cpu_tensor_payload_bytes": payload_bytes,
            "memory_load_seconds": load_seconds,
            "memory_source": memory_source,
            "reconstructed_memory_hash": R.V.memory_hash(memory),
            "archived_memory_hash": (expected_hash if not archive.exists() else None),
            "construction_seconds_sum": construction_seconds,
            "construction_peak_mb_max": construction_peak,
            "student": student,
        })
        del memory
        torch.cuda.empty_cache()
        print(json.dumps({"event": "resource_arm_done", "dataset": args.dataset,
                          "arm": arm, "construction_peak_mb": construction_peak,
                          "student_peak_mb": student["student_train_peak_mb_max"],
                          "inference_peak_mb": student["full_graph_inference_peak_mb_max"]}), flush=True)
    jwrite(Path(args.output) / "results.json", {"status": "complete", **data_rows})


def aggregate(args) -> None:
    root = Path(args.output)
    rows = []
    # Existing no-graph cells are imported without modification.
    ng_root = ROOT / "results/uq_priority_controls_20261014/no_graph"
    for path in sorted(ng_root.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        final = payload["no_graph"]["rows"][-1]
        score = final["scores"]["sealed_test"]
        rows.append({"track": "no_graph", "dataset": payload["dataset"],
                     "seed": payload["construction_seed"],
                     "f1": score["final_f1"], "accuracy": score["accuracy"],
                     "source": str(path)})
    for dataset in DATASETS:
        fresh_root = root / "fresh" / dataset
        if fresh_root.exists():
            for p in sorted(fresh_root.glob("o*_c*/results.json")):
                payload = json.loads(p.read_text(encoding="utf-8"))
                for row in payload.get("rows", []):
                    score = row["scores"]["fresh_sealed"]
                    rows.append({"track": "fresh_sealed", "dataset": dataset,
                                 "seed": payload["protocol"]["construction_seed"],
                                 "order_seed": payload["protocol"]["order_seed"],
                                 "arm": row["arm"], "stage": row["stage"],
                                 "f1": score["final_f1"], "accuracy": score["accuracy"],
                                 "construction_seconds": row["construction_seconds"],
                                 "construction_peak_mb": row["construction_peak_mb"],
                                 "source": str(p)})
        p = root / "text_budget" / dataset / "results.json"
        if p.exists():
            payload = json.loads(p.read_text(encoding="utf-8"))
            for record in payload["records"]:
                for row in record["rows"]:
                    score = row["scores"]["sealed_test"]
                    rows.append({"track": "text_budget", "dataset": dataset,
                                 "seed": record["construction_seed"],
                                 "arm": row["arm"], "f1": score["final_f1"],
                                 "accuracy": score["accuracy"],
                                 "max_chars": row["max_chars"],
                                 "mean_chars": row["mean_chars"],
                                 "source": str(p)})
        p = root / "resource" / dataset / "results.json"
        if p.exists():
            payload = json.loads(p.read_text(encoding="utf-8"))
            for arm in payload["arms"]:
                rows.append({"track": "resource", "dataset": dataset,
                             "seed": payload["construction_seed"],
                             "arm": arm["arm"],
                             "construction_peak_mb": arm["construction_peak_mb_max"],
                             "construction_seconds": arm["construction_seconds_sum"],
                             "student_peak_mb": arm["student"]["student_train_peak_mb_max"],
                             "student_seconds": arm["student"]["student_train_seconds_mean"],
                             "inference_peak_mb": arm["student"]["full_graph_inference_peak_mb_max"],
                             "inference_seconds": arm["student"]["full_graph_inference_seconds_mean"],
                             "source": str(p)})
    out_json = root / "summary.json"
    jwrite(out_json, {"status": "complete", "rows": rows,
                      "notes": {
                          "no_graph": "existing exact-5% constructor control; downstream uses common full raw graph",
                          "text_budget": "fixed 320-character cap; storage-only on frozen primary X/labels",
                          "fresh_sealed": "fresh modulo-4 test requires per-cell aggregation from fresh/ outputs",
                          "resource": "construction costs from frozen rows; Student and full-graph inference measured on representative cell",
                      }})
    with (root / "summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        fields = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    md = ["# CoTAG-Q four-priority experiment summary", "",
          "This report is additive and does not modify the paper or frozen primary tables.", ""]
    for track in sorted({row["track"] for row in rows}):
        md += [f"## {track}", ""]
        subset = [row for row in rows if row["track"] == track]
        if not subset:
            md.append("No completed rows.")
        else:
            keys = sorted({key for row in subset for key in row
                           if key not in {"track", "source"}})
            md.append("| " + " | ".join(keys) + " |")
            md.append("| " + " | ".join(["---"] * len(keys)) + " |")
            for row in subset:
                md.append("| " + " | ".join(str(row.get(key, "")) for key in keys) + " |")
        md.append("")
    (root / "summary.md").write_text("\n".join(md), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("fresh", "text_budget", "resource", "aggregate"), required=True)
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--text-cache", type=Path)
    parser.add_argument("--minilm")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--order-seed", type=int, default=9101)
    parser.add_argument("--construction-seed", type=int, default=271006)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.cuda.set_device(args.device) if str(args.device).startswith("cuda") else None
    if args.mode in ("fresh", "text_budget", "resource"):
        for required in (args.dataset, args.reference, args.text_cache, args.minilm):
            if required is None:
                parser.error("dataset/reference/text-cache/minilm are required for this mode")
    if args.mode == "fresh":
        run_fresh_cell(args)
    elif args.mode == "text_budget":
        run_text_budget(args)
    elif args.mode == "resource":
        run_resource(args)
    else:
        aggregate(args)


if __name__ == "__main__":
    main()
