"""Read-only sealed-test evaluation for frozen Unified-Q memories.

The constructor is deliberately absent from this runner.  It discovers
completed confirmation cells, reloads their archived final memories, verifies
the protocol/split hashes, and evaluates the same memory with prompt-only GCN,
SGC, and GraphSAGE students on the sealed ``original_id % 10 == 2`` mask.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import graphtext_prompt_student as GCN
import graphtext_sgc_transfer as SGC
import graphtext_sage_transfer as SAGE
import run_unified_v3_continual as U


ARMS = ("uq_qmedoid", "uq_dm_edgefree", "cat_cctp", "gecc_hard")
STUDENTS = ("prompt_gcn", "prompt_sgc", "prompt_sage")


def hash_tensor(value: torch.Tensor) -> str:
    return hashlib.sha256(
        value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def memory_hash(memory: dict) -> str:
    digest = hashlib.sha256(
        json.dumps(memory["texts"], ensure_ascii=True).encode("utf-8"))
    for key in ("x", "text_emb", "labels", "mass", "edge_index"):
        digest.update(hash_tensor(memory[key]).encode("ascii"))
    return digest.hexdigest()


def validate_memory(memory: dict, budget: int) -> None:
    for key in ("x", "text_emb", "labels", "mass", "texts"):
        if len(memory[key]) != int(budget):
            raise AssertionError(f"budget mismatch: {key}")
    for key in ("x", "text_emb", "mass"):
        if not bool(torch.isfinite(memory[key]).all()):
            raise AssertionError(f"nonfinite memory: {key}")
    edges = memory["edge_index"]
    if edges.numel() and (int(edges.min()) < 0 or int(edges.max()) >= int(budget)):
        raise AssertionError("edge endpoint outside memory")


def load_large(dataset: str, minilm: str, cache: Path, device: str,
               manifest: dict) -> dict:
    """Load the same raw data/splits as the frozen confirmation loader."""
    data = U.load_unified(
        dataset, U.DATA_CFG[dataset]["path"], minilm, device,
        n_stages=len(manifest["groups"]), split_mode="base_incremental",
        base_fraction=0.4,
        report_dev_mod=int(manifest.get("report_dev_mod", 3)),
    )
    text_path = cache / "text_emb_pool_only.pt"
    text_emb = torch.load(text_path, map_location="cpu", weights_only=False).float()
    if tuple(text_emb.shape) != (len(data["x"]), 384):
        raise ValueError(f"unexpected text cache shape: {tuple(text_emb.shape)}")
    excluded = ~data["pool_mask"]
    if bool(excluded.any()) and float(text_emb[excluded].abs().max()) != 0.0:
        raise AssertionError("text cache contains non-pool rows")
    data["text_emb"] = text_emb
    data["stage_classes"] = [[int(v) for v in group]
                              for group in manifest["groups"]]
    data["prompt_emb"] = data["encoder"].encode(data["prompts"])
    data["real_adj"] = GCN.normalized_adjacency(
        data["edge_index"], len(data["x"]), device)
    data["x_device"] = data["x"].float().to(device)
    return data


def stage_ids(data: dict) -> list[torch.Tensor]:
    output = []
    for group in data["stage_classes"]:
        classes = torch.tensor(group, dtype=torch.long)
        ids = torch.where(data["pool_mask"] & torch.isin(data["y"], classes))[0]
        output.append(torch.sort(ids).values)
    return output


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_cache_specs(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"cache must use DATASET=PATH syntax: {value}")
        name, raw_path = value.split("=", 1)
        result[name.strip().lower()] = Path(raw_path)
    return result


def load_memory(path: Path) -> dict:
    with gzip.open(path, "rb") as stream:
        return torch.load(stream, map_location="cpu", weights_only=False)


def discover_cells(roots: list[Path]) -> list[tuple[Path, dict]]:
    found = {}
    for root in roots:
        if not root.exists():
            raise FileNotFoundError(root)
        for protocol_path in root.rglob("protocol.json"):
            cell = protocol_path.parent
            protocol = read_json(protocol_path)
            if protocol.get("name") != "uq-single-hard-projection-confirmation-v1":
                continue
            result_path = cell / "results.json"
            if not result_path.exists():
                continue
            result = read_json(result_path)
            if result.get("status") != "complete":
                continue
            key = (protocol["dataset"], int(protocol["order_seed"]),
                   int(protocol["construction_seed"]))
            for arm in ARMS:
                archive = cell / f"final_{arm}.pt.gz"
                if not archive.exists():
                    raise FileNotFoundError(f"missing {archive}")
            previous = found.get(key)
            if previous is not None and previous[0] != cell:
                raise RuntimeError(f"duplicate sealed cell {key}: {previous[0]} and {cell}")
            found[key] = (cell, protocol)
    return [found[key] for key in sorted(found)]


def stage_mask(data: dict, protocol: dict, split_name: str) -> tuple[torch.Tensor, list[int], list[int], list[int]]:
    mask_key = {"test": "test_mask", "report_dev": "report_dev_mask"}[split_name]
    mask = data[mask_key].bool().cpu()
    groups = [[int(v) for v in group] for group in protocol["groups"]]
    seen = [c for group in groups for c in group]
    old = [c for group in groups[:-1] for c in group]
    new = list(groups[-1])
    labels = data["y"].long().cpu()
    return mask & torch.isin(labels, torch.tensor(seen)), seen, old, new


def score_memory(memory: dict, data: dict, protocol: dict, student: str,
                 student_seeds: list[int], device: str, steps: int,
                 batch: int, split_name: str) -> dict:
    if memory["edge_index"].numel():
        raise AssertionError("sealed evaluator expects edge-free memory")
    prompt = data["prompt_emb"].float().to(device)
    y = data["y"].long().cpu()
    mask, seen, old, new = stage_mask(data, protocol, split_name)
    train = {
        "prompt_gcn": GCN.train_prompt_student,
        "prompt_sgc": SGC.train_sgc_student,
        "prompt_sage": SAGE.train_sage_student,
    }[student]
    scores = []
    for seed in student_seeds:
        started = time.perf_counter()
        model = train(memory, prompt, int(seed), device, int(steps),
                      lr=0.001, batch=int(batch))
        with torch.no_grad():
            logits = model.logits(data["x_device"], data["real_adj"], prompt)
            pred = logits.argmax(1).detach().cpu()
        yy = y[mask].numpy()
        pp = pred[mask].numpy()
        per_class = f1_score(yy, pp, average=None, labels=seen,
                             zero_division=0)
        class_f1 = {cls: float(value * 100.0)
                    for cls, value in zip(seen, per_class)}
        scores.append({
            "seed": int(seed),
            "final_f1": float(100.0 * f1_score(
                yy, pp, average="macro", labels=seen, zero_division=0)),
            "accuracy": float(100.0 * accuracy_score(yy, pp)),
            "old_f1": (float(np.mean([class_f1.get(c, 0.0) for c in old]))
                       if old else None),
            "new_f1": float(np.mean([class_f1.get(c, 0.0) for c in new])),
            "rows": int(mask.sum()),
            "seconds": float(time.perf_counter() - started),
        })
        del model
    def mean(key):
        values = [row[key] for row in scores if row[key] is not None]
        return float(np.mean(values)) if values else None
    def std(key):
        values = [row[key] for row in scores if row[key] is not None]
        return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return {
        "split": split_name,
        "student": student,
        "student_steps": int(steps),
        "student_batch": int(batch),
        "student_seeds": [int(v) for v in student_seeds],
        "final_f1": mean("final_f1"),
        "final_f1_std": std("final_f1"),
        "accuracy": mean("accuracy"),
        "accuracy_std": std("accuracy"),
        "old_f1": mean("old_f1"),
        "new_f1": mean("new_f1"),
        "eval_rows": int(mask.sum()),
        "seen_classes": seen,
        "old_classes": old,
        "new_classes": new,
        "per_student": scores,
    }


def resolve_cache(dataset: str, cache_specs: dict[str, Path]) -> Path:
    if dataset not in cache_specs:
        raise KeyError(f"missing --cache {dataset}=...")
    path = cache_specs[dataset]
    if path.is_file():
        path = path.parent
    if not (path / "text_emb_pool_only.pt").exists():
        raise FileNotFoundError(path / "text_emb_pool_only.pt")
    return path


def verify_data(data: dict, protocol: dict) -> dict:
    expected_report = protocol.get("report_mask_hash")
    expected_pool = protocol.get("pool_mask_hash")
    actual = {
        "report_mask_hash": hash_tensor(data["report_dev_mask"]),
        "test_mask_hash": hash_tensor(data["test_mask"]),
        "pool_mask_hash": hash_tensor(data["pool_mask"]),
        "stage_id_hashes": [hash_tensor(v) for v in stage_ids(data)],
    }
    original_ids = data["original_ids"].long().cpu()
    expected_test = (original_ids % 10) == 2
    if not torch.equal(data["test_mask"].bool().cpu(), expected_test):
        raise AssertionError("sealed test mask is not original_id % 10 == 2")
    masks = [data["selector_dev_mask"].bool().cpu(),
             data["report_dev_mask"].bool().cpu(),
             data["test_mask"].bool().cpu(),
             data["pool_mask"].bool().cpu()]
    if any(bool((masks[i] & masks[j]).any())
           for i in range(len(masks)) for j in range(i + 1, len(masks))):
        raise AssertionError("sealed/data masks overlap")
    if not bool(torch.stack(masks).sum(0).eq(1).all()):
        raise AssertionError("sealed/data masks do not cover every node exactly once")
    if expected_report and actual["report_mask_hash"] != expected_report:
        raise AssertionError("report-dev mask differs from construction protocol")
    if expected_pool and actual["pool_mask_hash"] != expected_pool:
        raise AssertionError("construction pool differs from construction protocol")
    if actual["stage_id_hashes"] != protocol.get("stage_id_hashes"):
        raise AssertionError("arrival stage ids differ from construction protocol")
    if int(data["test_mask"].sum()) == 0:
        raise AssertionError("sealed test mask is empty")
    return actual


def run(args):
    roots = [Path(v) for v in args.source_root]
    cells = discover_cells(roots)
    cache_specs = parse_cache_specs(args.cache)
    args.output.mkdir(parents=True, exist_ok=True)
    output_path = args.output / "results.json"
    config = {
        "name": "uq-sealed-test-evaluator-v1",
        "source_roots": [str(v.resolve()) for v in roots],
        "arms": list(ARMS),
        "students": list(args.students),
        "student_seeds": [int(v) for v in args.student_seeds],
        "split": "sealed test: original_id % 10 == 2",
        "construction_reused": True,
        "constructor_invoked": False,
        "test_used_for_construction": False,
        "selection_or_tuning_on_test": False,
        "batch": int(args.batch),
        "steps_override": args.student_steps,
        "cell_count": len(cells),
        "cache_specs": {k: str(v) for k, v in cache_specs.items()},
        "evaluator_sha256": sha256_file(Path(__file__)),
    }
    if output_path.exists():
        saved = read_json(output_path)
        if saved.get("config") != config:
            raise ValueError("existing sealed output protocol differs")
        result = saved
    else:
        result = {"config": config, "cells": [], "rows": [], "status": "running"}
    done = {(row["dataset"], int(row["order_seed"]),
             int(row["construction_seed"]), row["arm"], row["student"])
            for row in result["rows"]}
    for cell, protocol in cells:
        key = (protocol["dataset"], int(protocol["order_seed"]),
               int(protocol["construction_seed"]))
        data = load_large(protocol["dataset"], args.minilm,
                          resolve_cache(protocol["dataset"], cache_specs),
                          args.device,
                          {"groups": protocol["groups"],
                           "report_dev_mod": protocol.get("report_dev_mod", 3)})
        data_audit = verify_data(data, protocol)
        cell_key = {"dataset": key[0], "order_seed": key[1],
                    "construction_seed": key[2]}
        if not any(v == cell_key for v in result["cells"]):
            result["cells"].append(dict(cell_key, source=str(cell.resolve()),
                                         data_audit=data_audit,
                                         source_result_sha256=sha256_file(cell / "results.json")))
            result["cells"] = sorted(result["cells"],
                                      key=lambda v: (v["dataset"], v["order_seed"], v["construction_seed"]))
        steps = int(args.student_steps or protocol["student_steps"])
        for arm in ARMS:
            archive = cell / f"final_{arm}.pt.gz"
            memory = load_memory(archive)
            archive_hash = sha256_file(archive)
            archive_memory_hash = memory_hash(memory)
            expected = [v for v in json.loads((cell / "results.json").read_text(encoding="utf-8")).get("rows", [])
                        if v.get("arm") == arm]
            known_hashes = {v.get("memory_hash") for v in expected}
            if known_hashes and archive_memory_hash not in known_hashes:
                raise AssertionError(f"archive hash mismatch for {archive}")
            validate_memory(memory, int(protocol["budgets"][-1]))
            for student in args.students:
                row_key = (*key, arm, student)
                if row_key in done:
                    continue
                score = score_memory(memory, data, protocol, student,
                                     args.student_seeds, args.device, steps,
                                     args.batch, "test")
                row = dict(cell_key, arm=arm, student=student,
                           archive_sha256=archive_hash,
                           memory_hash=archive_memory_hash,
                           memory_rows=len(memory["x"]),
                           unique_texts=len(set(memory["texts"])),
                           archive_bytes=archive.stat().st_size,
                           score=score,
                           checks=dict(memory_unchanged=True,
                                       construction_reused=True,
                                       test_mask_hash=data_audit["test_mask_hash"],
                                       report_mask_hash=data_audit["report_mask_hash"]))
                result["rows"].append(row)
                result["rows"] = sorted(result["rows"], key=lambda v: (
                    v["dataset"], v["order_seed"], v["construction_seed"],
                    ARMS.index(v["arm"]), list(args.students).index(v["student"])))
                done.add(row_key)
                output_path.write_text(json.dumps(result, indent=2, ensure_ascii=True), encoding="utf-8")
                print(json.dumps({"event": "sealed_evaluated", **cell_key,
                                  "arm": arm, "student": student,
                                  "f1": score["final_f1"],
                                  "accuracy": score["accuracy"]}), flush=True)
            del memory
        del data
    result["status"] = "complete"
    result["config"]["completed_rows"] = len(result["rows"])
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=True), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", action="append", required=True)
    parser.add_argument("--cache", action="append", required=True,
                        help="dataset=directory containing text_emb_pool_only.pt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minilm", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--students", nargs="+", choices=STUDENTS,
                        default=list(STUDENTS))
    parser.add_argument("--student-seeds", nargs="+", type=int,
                        default=[150, 151, 152])
    parser.add_argument("--student-steps", type=int, default=None)
    parser.add_argument("--batch", type=int, default=24)
    torch.set_num_threads(2)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
