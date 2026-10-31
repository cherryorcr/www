"""Large-TAG node-classification comparison for Unified-Q topology.

The edge-free arm and Q-derived-topology arm share the exact same large-graph
stream, budgets, prompt Student and evaluation code.  Only JSON metrics are
persisted to keep the experiment usable on a nearly full remote volume.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

import external_continual_mechanisms as M
import run_unified_q_large_smoke_20261007 as L
import unified_q_multimodal as UQ


ARMS = ("unified_q_edge_free", "unified_q")


def save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=True), encoding="utf-8")
    tmp.replace(path)


def run(args: argparse.Namespace) -> dict:
    manifest = L.read(args.protocol)
    data = L.load_large(args.dataset, args.minilm, args.text_cache,
                        args.device, manifest)
    ids = L.stage_ids(data)
    budgets = [int(v) for v in manifest["budgets"]]
    cfg = UQ.Config(steps=int(args.q_steps), proxy_seeds=(2711,),
                    materialize_top_k=1, straight_through_text=True,
                    hard_topology_forward=True, seed=271006)
    cat_cfg = M.Config(seed=271006, cat_steps=3000, text_sources=5,
                       text_mode="nearest")
    parents = {arm: None for arm in ARMS}
    rows = []
    for stage, stage_ids_value in enumerate(ids):
        delta = L.delta_view(data, stage_ids_value, stage)
        for arm in ARMS:
            parent = parents[arm]
            old_rows = 0 if parent is None else len(parent["x"])
            new_rows = budgets[stage] - old_rows
            labels = M.row_labels(M.ratio_allocation(delta["y"], new_rows))
            started = time.perf_counter()
            block, audit = UQ.optimize_stage(
                delta, parent, labels, data["encoder"], data["prompt_emb"],
                cfg, args.device, use_boundary=False,
                use_topology=(arm == "unified_q"))
            memory = UQ.append_memory(parent, block) if arm == "unified_q" else L.append_edge_free(parent, block)
            build_seconds = time.perf_counter() - started
            score = L.score(memory, data, stage, args.student_seeds,
                            args.device, args.student_steps, args.student_batch)
            edge_entries = int(memory.get("edge_index", torch.empty((2, 0))).shape[1])
            rows.append({"dataset": args.dataset, "arm": arm, "stage": stage,
                         "raw_rows": len(delta["x"]), "new_rows": int(new_rows),
                         "memory_rows": len(memory["x"]),
                         "construction_seconds": build_seconds,
                         "directed_edge_entries": edge_entries,
                         "undirected_edges": edge_entries // 2,
                         "score": score,
                         "audit": {k: v for k, v in audit.items()
                                   if k not in {"support_ids", "text_materialization", "text_source_ids"}}})
            parents[arm] = memory
            print(json.dumps({"event": "evaluated", "dataset": args.dataset,
                              "arm": arm, "stage": stage,
                              "f1": score["final_f1"],
                              "accuracy": score["accuracy"]}), flush=True)
    result = {"protocol": {
        "name": "unified-q-large-topology-compare-20261011",
        "dataset": args.dataset, "arms": list(ARMS),
        "source_protocol": str(args.protocol),
        "memory_budget": "copied from frozen large reference",
        "construction_access": "current delta plus frozen synthetic prefix; no historical raw reads",
        "student": "same prompt-only GraphText Student for both arms",
        "topology": "Q-derived new-new edges for unified_q; no saved edges for edge-free",
    }, "rows": rows}
    save(args.output / f"{args.dataset}_results.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("wikics", "bookhis", "sportsfit", "arxiv"), required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--text-cache", type=Path, required=True)
    parser.add_argument("--minilm", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--q-steps", type=int, default=30)
    parser.add_argument("--student-steps", type=int, default=2048)
    parser.add_argument("--student-batch", type=int, default=24)
    parser.add_argument("--student-seeds", nargs="+", type=int, default=[150, 151, 152])
    args = parser.parse_args()
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    run(args)


if __name__ == "__main__":
    main()
