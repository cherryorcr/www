"""Run PUMA-inspired TAG adaptation under the frozen large-graph protocol.

The output is intentionally summary-oriented: memories remain in RAM between
stages and only JSON audits/digests are written.  This keeps the experiment
restartable without filling the remote disk with repeated ``memory.pt`` files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

import external_continual_mechanisms as M
import puma_tag_adapter as PUMA
import run_public_continual_baselines_large_20261001 as L
import run_unified_v3_continual as U


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=True),
                         encoding="utf-8")
    temporary.replace(path)


def digest_memory(memory: dict) -> str:
    digest = hashlib.sha256()
    for key in ("x", "text_emb", "mass", "depth", "stage", "edge_index"):
        value = memory[key].detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    for text in memory["texts"]:
        encoded = str(text).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def append_edge_free(parent: dict | None, block: dict) -> dict:
    if parent is None:
        out = {key: (value.clone() if isinstance(value, torch.Tensor)
                     else list(value) if isinstance(value, list) else value)
               for key, value in block.items()}
    else:
        out = {key: torch.cat((parent[key], block[key]))
               for key in ("x", "text_emb", "mass", "depth", "stage")}
        out["texts"] = list(parent["texts"]) + list(block["texts"])
    out["edge_index"] = torch.empty((2, 0), dtype=torch.long)
    return out


def run_dataset(dataset: str, args: argparse.Namespace, cfg: PUMA.PUMAConfig) -> dict:
    source = args.source_root / f"{dataset}_base4"
    manifest = L.source_root_manifest(args.source_root, dataset, args.text_cache_root)
    groups = L.stage_groups(manifest)
    budgets = [int(v) for v in manifest["budgets"]]
    data = U.load_unified(dataset, U.DATA_CFG[dataset]["path"], args.minilm,
                          args.device, n_stages=len(groups),
                          split_mode="base_incremental", base_fraction=0.4,
                          report_dev_mod=int(manifest.get("report_dev_mod", 3)))
    if args.order_seed is not None:
        groups = L.permute_stage_groups(groups, data, int(args.order_seed))
    cache_source = args.text_cache_root / dataset
    if not (cache_source / "text_emb_pool_only.pt").exists():
        cache_source = args.text_cache_root / f"{dataset}_base4"
    data = L.set_pool_text_embeddings(data, cache_source)
    L.configure_eval(data, groups, args.device)
    ids = L.stage_ids(data, groups)
    expected_counts = manifest.get("stage_train_counts")
    if args.order_seed is None and expected_counts is not None:
        if [len(v) for v in ids] != [int(v) for v in expected_counts]:
            raise AssertionError(f"{dataset}: source stage counts differ")

    result_dir = args.output_root / f"{dataset}_base4"
    result_dir.mkdir(parents=True, exist_ok=True)
    records = []
    parent = None
    for stage, (delta_ids, budget) in enumerate(zip(ids, budgets)):
        new_rows = int(budget) - (len(parent["x"]) if parent is not None else 0)
        if new_rows <= 0 or new_rows > len(delta_ids):
            raise AssertionError(f"{dataset}/stage{stage}: invalid new rows {new_rows}")
        delta = L.delta_view(data, delta_ids, stage)
        started = time.perf_counter()
        block, audit = PUMA.build_puma_block(
            delta, new_rows, data["encoder"], cfg, args.device, args.arm)
        memory = append_edge_free(parent, block)
        L.check_memory(memory, int(budget), parent, allow_edges=False)
        build_seconds = time.perf_counter() - started
        score = L.student_score(memory, stage, data, args.student_seeds,
                                args.device, args.student_batch,
                                args.student_steps)
        record = {
            "dataset": dataset,
            "stage": int(stage),
            "arm": args.arm,
            "memory_rows": int(len(memory["x"])),
            "memory_sha256": digest_memory(memory),
            "construction_seconds": float(build_seconds),
            "construction_audit": audit,
            "score": score,
        }
        records.append(record)
        save_json(result_dir / "progress.json", {"records": records})
        print(json.dumps({"event": "puma_stage_complete", "dataset": dataset,
                          "stage": stage, "arm": args.arm,
                          "f1": score["final_f1"],
                          "accuracy": score["accuracy"],
                          "pseudo_rows": audit["pseudo_rows"]}), flush=True)
        parent = memory

    final = records[-1]["score"]
    overall = {
        "final_f1": final["final_f1"],
        "final_f1_std": final["final_f1_std"],
        "final_accuracy": final["accuracy"],
        "final_accuracy_std": final["accuracy_std"],
        "accuracy": final["accuracy"],
        "accuracy_std": final["accuracy_std"],
        "old_f1": final["old_f1"],
        "new_f1": final["new_f1"],
        "stage_avg_f1": final["stage_avg_f1"],
        "stage_avg_accuracy": final["stage_avg_accuracy"],
        "worst_stage_f1": float(min(v["score"]["final_f1"] for v in records)),
        "construction_seconds": float(sum(v["construction_seconds"] for v in records)),
        "student_seconds": float(sum(v["score"]["student_seconds"] for v in records)
                                  / len(args.student_seeds)),
    }
    protocol = {
        "name": "puma-inspired-tag-adapted-large-20261006",
        "dataset": dataset,
        "stage_protocol": "same frozen base+4 groups and 5% cumulative budget",
        "groups": groups,
        "budgets": budgets,
        "arm": args.arm,
        "student": "same Prompt zero-shot Student as expanded strict table",
        "student_seeds": args.student_seeds,
        "historical_raw_reads": 0,
        "report_test_labels_used_for_construction": False,
        "data_access": "current delta plus previous synthetic parent only",
        "puma_mechanisms": ["one-time propagation", "confidence-filtered pseudo labels",
                            "edge-free memory", "scratch Student per stage"],
        "tag_text_attachment": "current-delta nearest text, top-5 sources, bounded re-encoding",
        "official_reproduction": False,
        "order_seed": args.order_seed,
    }
    output = {"protocol": protocol, "dataset": dataset,
              "records": records, "overall": {args.arm: overall}}
    save_json(result_dir / "results.json", output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+",
                        default=["arxiv", "sportsfit", "wikics", "bookhis"])
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--text-cache-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--minilm", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--arm", choices=PUMA.PUMA_ARMS, default="puma_full")
    parser.add_argument("--student-seeds", nargs="+", type=int,
                        default=[100, 101, 102, 103, 104, 105])
    parser.add_argument("--student-batch", type=int, default=16)
    parser.add_argument("--student-steps", type=int, default=None)
    parser.add_argument("--construction-seed", type=int, default=44)
    parser.add_argument("--pseudo-threshold", type=float, default=0.80)
    parser.add_argument("--order-seed", type=int, default=None)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    cfg = PUMA.PUMAConfig(seed=int(args.construction_seed),
                           pseudo_threshold=float(args.pseudo_threshold))
    outputs = [run_dataset(dataset, args, cfg) for dataset in args.datasets]
    table = []
    for value in outputs:
        table.append({"dataset": value["dataset"], "arm": args.arm,
                      **value["overall"][args.arm]})
    summary = {"protocol": {"name": "puma-inspired-tag-adapted-large-20261006",
                             "datasets": args.datasets, "arm": args.arm},
               "rows": table}
    save_json(args.output_root / "summary.json", summary)
    lines = ["# PUMA-inspired TAG adapted large-graph results", "",
             "The arm is a mechanism adaptation, not an official native PUMA reproduction.", "",
             "| Dataset | Final F1 | Accuracy | Old F1 | New F1 | Stage avg F1 | Worst stage F1 | Build s | Student s/seed |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in table:
        lines.append("|{dataset}|{final_f1:.2f}|{final_accuracy:.2f}|{old_f1:.2f}|{new_f1:.2f}|{stage_avg_f1:.2f}|{worst_stage_f1:.2f}|{construction_seconds:.1f}|{student_seconds:.1f}|".format(**row))
    (args.output_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"event": "complete", "output": str(args.output_root),
                      "rows": table}), flush=True)


if __name__ == "__main__":
    main()
