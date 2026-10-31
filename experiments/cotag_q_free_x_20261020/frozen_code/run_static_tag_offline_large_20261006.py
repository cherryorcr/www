"""Full-history static TAG reference under the shared large-graph Student.

This is intentionally an offline/reference protocol.  At every stage it may
read the complete legal raw pool seen so far and rebuild all synthetic rows;
therefore it is not a strict continual competitor.  The constructor is the
project's scalable joint graph-text distribution matcher, not an official
TAGSAM or STAD end-to-end reproduction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

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


def run_dataset(dataset: str, args: argparse.Namespace) -> dict:
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
    if args.order_seed is None and manifest.get("stage_train_counts") is not None:
        if [len(v) for v in ids] != [int(v) for v in manifest["stage_train_counts"]]:
            raise AssertionError(f"{dataset}: source stage counts differ")

    result_dir = args.output_root / f"{dataset}_base4"
    result_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for stage, budget in enumerate(budgets):
        source_ids = torch.cat(ids[:stage + 1])
        started = time.perf_counter()
        memory = U.condense_static_gpu(
            data["x"][source_ids], data["text_emb"][source_ids],
            [data["texts"][int(i)] for i in source_ids.tolist()], int(budget),
            data["encoder"], args.device, stage=stage)
        build_seconds = time.perf_counter() - started
        if len(memory["x"]) != int(budget):
            raise AssertionError("static offline budget mismatch")
        score = L.student_score(memory, stage, data, args.student_seeds,
                                args.device, args.student_batch,
                                args.student_steps)
        audit = {
            "method": "Static-TAG-full-history-joint",
            "stage": int(stage),
            "source_rows": int(len(source_ids)),
            "source_ids_sha256": hashlib.sha256(
                source_ids.cpu().numpy().tobytes()).hexdigest(),
            "historical_raw_reads": "full seen legal raw pool",
            "old_memory_updated": True,
            "rebuild_from_scratch": True,
            "topology": "text similarity topology materialized by static constructor",
            "text_attachment": "joint graph-text cluster center with bounded source-text fusion",
            "official_tagsam_or_stad_reproduction": False,
        }
        record = {
            "dataset": dataset, "stage": int(stage),
            "arm": "static_tag_full_history",
            "memory_rows": int(len(memory["x"])),
            "memory_sha256": digest_memory(memory),
            "construction_seconds": float(build_seconds),
            "construction_audit": audit,
            "score": score,
        }
        records.append(record)
        save_json(result_dir / "progress.json", {"records": records})
        print(json.dumps({"event": "static_stage_complete", "dataset": dataset,
                          "stage": stage, "f1": score["final_f1"],
                          "accuracy": score["accuracy"],
                          "source_rows": len(source_ids)}), flush=True)

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
        "name": "static-tag-full-history-reference-large-20261006",
        "dataset": dataset,
        "stage_protocol": "same base+4 groups and 5% cumulative budget",
        "groups": groups, "budgets": budgets,
        "arm": "static_tag_full_history",
        "student": "same Prompt zero-shot Student as strict table",
        "student_seeds": args.student_seeds,
        "historical_raw_reads": "full seen legal raw pool at every stage",
        "strict_continual": False,
        "official_reproduction": False,
        "comparison_role": "offline/reference upper-bound style comparator",
        "order_seed": args.order_seed,
    }
    output = {"protocol": protocol, "dataset": dataset,
              "records": records, "overall": {"static_tag_full_history": overall}}
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
    parser.add_argument("--student-seeds", nargs="+", type=int,
                        default=[100, 101, 102, 103, 104, 105])
    parser.add_argument("--student-batch", type=int, default=16)
    parser.add_argument("--student-steps", type=int, default=None)
    parser.add_argument("--order-seed", type=int, default=None)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    outputs = [run_dataset(dataset, args) for dataset in args.datasets]
    rows = [{"dataset": value["dataset"],
             **value["overall"]["static_tag_full_history"]}
            for value in outputs]
    summary = {"protocol": {"name": "static-tag-full-history-reference-large-20261006",
                             "datasets": args.datasets}, "rows": rows}
    save_json(args.output_root / "summary.json", summary)
    lines = ["# Static TAG full-history reference large-graph results", "",
             "This is an offline/reference comparator and is not a strict continual method.", "",
             "| Dataset | Final F1 | Accuracy | Old F1 | New F1 | Stage avg F1 | Worst stage F1 | Build s | Student s/seed |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append("|{dataset}|{final_f1:.2f}|{final_accuracy:.2f}|{old_f1:.2f}|{new_f1:.2f}|{stage_avg_f1:.2f}|{worst_stage_f1:.2f}|{construction_seconds:.1f}|{student_seconds:.1f}|".format(**row))
    (args.output_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"event": "complete", "output": str(args.output_root),
                      "rows": rows}), flush=True)


if __name__ == "__main__":
    main()
