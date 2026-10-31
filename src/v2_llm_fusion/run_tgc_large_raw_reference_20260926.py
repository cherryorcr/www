"""Run same-student full-raw references for the frozen large-graph protocol.

The raw support may read all legal pool nodes seen so far and their induced
training edges.  It is an offline reference, not a continual constructor.
``raw_equal`` uses the canonical synthetic-memory step count for the stage;
``raw_long`` uses a fixed extended step cap and is reported separately.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import torch

import protocol_v3
import run_unified_v3_continual as U


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def induced_edges(edge_index: torch.Tensor, ids: torch.Tensor,
                  total_nodes: int) -> torch.Tensor:
    ids = ids.long().cpu()
    remap = torch.full((int(total_nodes),), -1, dtype=torch.long)
    remap[ids] = torch.arange(len(ids), dtype=torch.long)
    edges = edge_index.long().cpu()
    valid = (remap[edges[0]] >= 0) & (remap[edges[1]] >= 0)
    return torch.stack((remap[edges[0, valid]], remap[edges[1, valid]]))


def load_pool_embeddings(data: dict, c0_root: Path) -> dict:
    cache = c0_root / "text_emb_pool_only.pt"
    if cache.exists():
        text_emb = torch.load(cache, map_location="cpu", weights_only=False)
    else:
        pool_ids = torch.where(data["pool_mask"])[0]
        pool_emb = data["encoder"].encode(
            [data["texts"][int(i)] for i in pool_ids]
        )
        text_emb = torch.zeros((len(data["x"]), 384), dtype=pool_emb.dtype)
        text_emb[pool_ids] = pool_emb
    if tuple(text_emb.shape) != (len(data["x"]), 384):
        raise ValueError(f"invalid text cache shape: {tuple(text_emb.shape)}")
    excluded = ~data["pool_mask"]
    if bool(excluded.any()) and float(text_emb[excluded].abs().max()) != 0.0:
        raise AssertionError("raw reference text cache contains holdout rows")
    data["text_emb"] = text_emb.float()
    return data


def raw_support(data: dict, ids: torch.Tensor, stage: int) -> dict:
    ids = ids.long().cpu()
    if not bool(data["pool_mask"][ids].all()):
        raise AssertionError("raw reference attempted to read outside pool")
    return {
        "x": data["x"][ids].float(),
        "text_emb": data["text_emb"][ids].float(),
        "texts": [data["texts"][int(i)] for i in ids],
        "edge_index": induced_edges(data["edge_index"], ids, len(data["x"])),
        "mass": torch.ones(len(ids)),
        "depth": torch.zeros(len(ids), dtype=torch.long),
        "stage": torch.full((len(ids),), int(stage), dtype=torch.long),
    }


def run_dataset(args: argparse.Namespace) -> dict:
    c0_root = Path(args.c0_root)
    output = Path(args.output_root) / f"{args.dataset}_base4"
    output.mkdir(parents=True, exist_ok=True)
    final_path = output / "results.json"
    if final_path.exists():
        raise FileExistsError(f"immutable output exists: {final_path}")
    manifest = json.loads((c0_root / "results.json").read_text(encoding="utf-8"))
    data = U.load_unified(
        args.dataset, args.data, args.minilm, args.device,
        n_stages=5, split_mode="base_incremental", base_fraction=0.4,
        report_dev_mod=int(manifest["report_dev_mod"]),
    )
    if data["stage_classes"] != manifest["stage_classes"]:
        raise AssertionError("raw reference class split differs from C0")
    data = load_pool_embeddings(data, c0_root)
    data["prompt_emb"] = data["encoder"].encode(data["prompts"])
    data["real_adj"] = U.G.norm_adj(data["edge_index"], len(data["x"]), args.device)
    data["x_device"] = data["x"].to(args.device)
    stage_train = []
    for classes in data["stage_classes"]:
        ids = torch.where(
            torch.isin(data["y"], torch.tensor(classes)) & data["pool_mask"]
        )[0]
        stage_train.append(torch.sort(ids).values)
    expected = [int(v) for v in manifest["stage_train_counts"]]
    if [len(v) for v in stage_train] != expected:
        raise AssertionError("raw reference stage counts differ from C0")
    budgets = [int(v) for v in manifest["budgets"]]
    student_seeds = [int(v) for v in args.student_seeds]
    results = []
    for stage in range(5):
        source = torch.cat(stage_train[:stage + 1])
        memory = raw_support(data, source, stage)
        step_modes = {
            "raw_equal": protocol_v3.student_steps(budgets[stage]),
            "raw_long": int(args.raw_long_steps),
        }
        for arm, steps in step_modes.items():
            started = time.perf_counter()
            student_args = SimpleNamespace(
                student_seeds=student_seeds,
                student_batch=int(args.student_batch),
                student_steps=int(steps),
            )
            metrics = U.run_students(memory, stage, data, student_args, args.device)
            row = {
                "arm": arm,
                "stage": int(stage),
                "raw_rows": int(len(source)),
                "budget_reference": int(budgets[stage]),
                "steps": int(steps),
                "construction_source": "all legal cumulative raw pool",
                "historical_raw_access": True,
                "rewrites_previous_memory": True,
                "induced_edge_count": int(memory["edge_index"].shape[1]),
                "wall_seconds": float(time.perf_counter() - started),
                **metrics,
            }
            results.append(row)
            print(json.dumps({
                "dataset": args.dataset,
                "arm": arm,
                "stage": stage,
                "cum_dev_f1_mean": row["cum_dev_f1_mean"],
                "steps": steps,
            }), flush=True)
    output_manifest = {
        "name": "tgc-large-raw-reference-2026-09-26",
        "dataset": args.dataset,
        "protocol": "base+4; fixed-rate 5%; report-dev mod 3",
        "arms": ["raw_equal", "raw_long"],
        "student": "original two-layer GCN + trainable text projection + bidirectional InfoNCE",
        "student_batch": int(args.student_batch),
        "student_seeds": student_seeds,
        "raw_long_steps": int(args.raw_long_steps),
        "rate": float(manifest["rate"]),
        "report_dev_mod": int(manifest["report_dev_mod"]),
        "stage_classes": data["stage_classes"],
        "stage_train_counts": [int(len(v)) for v in stage_train],
        "budgets": budgets,
        "source_hash_sha256": sha256_file(args.data),
        "c0_manifest_sha256": sha256_file(c0_root / "results.json"),
        "construction_labels": "none; raw support is an offline data reference",
        "test_accessed": False,
        "results": results,
    }
    (output / "results.json").write_text(
        json.dumps(output_manifest, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    return output_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("arxiv", "bookchild", "sportsfit"), required=True)
    parser.add_argument("--c0-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--data", default=None)
    parser.add_argument("--minilm", default="/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--student-seeds", type=int, nargs="+", default=[100, 101, 102, 103, 104, 105])
    parser.add_argument("--student-batch", type=int, default=16)
    parser.add_argument("--raw-long-steps", type=int, default=8192)
    args = parser.parse_args()
    if args.data is None:
        args.data = U.DATA_CFG[args.dataset]["path"]
    torch.set_num_threads(2)
    started = time.perf_counter()
    run_dataset(args)
    print(json.dumps({"event": "complete", "elapsed_seconds": time.perf_counter() - started}), flush=True)


if __name__ == "__main__":
    main()
