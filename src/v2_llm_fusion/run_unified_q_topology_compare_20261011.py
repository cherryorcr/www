"""Sealed comparison of edge-free Unified-Q and topology-enabled branches.

This wrapper deliberately keeps the existing Unified-Q constructor unchanged.
It exposes only the three topology arms needed for the cross-task study and
writes a separate output root, so the frozen edge-free matrix is never
overwritten.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import run_unified_q_multimodal_20261006 as R
import unified_q_multimodal as UQ


ARMS = ("unified_q_edge_free", "unified_q", "unified_q_boundary")


def save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=True), encoding="utf-8")
    tmp.replace(path)


def summarize(root: Path, dataset: str, order: str) -> dict:
    cell = root / dataset / order
    construction = R.read(cell / "construction.json")
    rows = []
    for item in construction["records"]:
        memory = R.torch.load(cell / item["memory_file"], map_location="cpu", weights_only=False)
        edge = memory.get("edge_index")
        entries = int(edge.shape[1]) if edge is not None else 0
        undirected = entries // 2
        n = max(1, len(memory["x"]))
        degree = R.torch.bincount(edge[0], minlength=n) if entries else R.torch.zeros(n, dtype=R.torch.long)
        rows.append({
            "dataset": dataset,
            "order": order,
            "arm": item["arm"],
            "stage": int(item["stage"]),
            "memory_rows": n,
            "directed_edge_entries": entries,
            "undirected_edges": undirected,
            "mean_degree": float(degree.float().mean()),
            "max_degree": int(degree.max()) if len(degree) else 0,
            "construction_seconds": item.get("construction_seconds"),
            "memory_file": item["memory_file"],
        })
    return {"rows": rows, "protocol": R.read(root / "protocol.json")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--minilm", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--datasets", nargs="+", default=["cora", "citeseer"])
    parser.add_argument("--orders", nargs="+", default=["order19611"])
    parser.add_argument("--report-seeds", nargs="+", type=int, default=[150, 151, 152])
    parser.add_argument("--q-steps", type=int, default=30)
    parser.add_argument("--student-steps", type=int, default=2048)
    parser.add_argument("--phase", choices=("all", "construct", "evaluate"), default="all")
    args = parser.parse_args()
    args.arms = list(ARMS)
    args.cat_steps = 0
    args.gm_steps = 0
    torch = R.torch
    torch.set_num_threads(2)

    uq_cfg = UQ.Config(
        steps=int(args.q_steps),
        proxy_seeds=(2711,),
        materialize_top_k=1,
        straight_through_text=True,
        hard_topology_forward=True,
        seed=271006,
    )
    cat_cfg = R.M.Config(seed=271006, cat_steps=3000,
                         text_sources=5, text_mode="nearest")
    gm_cfg = R.GM.Config(steps=30, gradient_matching_mode="decoupled",
                         topology_mode="none", proxy_seeds=(2711,))
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = R._protocol(args, cat_cfg, gm_cfg, uq_cfg)
    manifest["name"] = "unified-q-topology-compare-20261011"
    manifest["arms"] = list(ARMS)
    manifest["scope"] = "topology branch versus edge-free main method"
    protocol_path = args.output_root / "protocol.json"
    if protocol_path.exists():
        if R.read(protocol_path) != manifest:
            raise ValueError("existing topology output root has a different protocol")
    else:
        save(protocol_path, manifest)

    data = {}
    for name in args.datasets:
        value = R.BASE.load_data(name, args.minilm, args.device, args.cache_root)
        value["prompt_emb"] = value["encoder"].encode(value["prompts"])
        data[name] = value

    phases = ("construct", "evaluate") if args.phase == "all" else (args.phase,)
    for phase in phases:
        for name in args.datasets:
            for order in args.orders:
                if phase == "construct":
                    R.construct_cell(data[name], order, args, cat_cfg, gm_cfg, uq_cfg)
                else:
                    R.evaluate_cell(data[name], order, args)
    if args.phase in ("all", "construct"):
        summaries = []
        for name in args.datasets:
            for order in args.orders:
                summaries.extend(summarize(args.output_root, name, order)["rows"])
        save(args.output_root / "topology_summary.json", {"protocol": manifest, "rows": summaries})
    if args.phase == "all":
        rows = []
        for name in args.datasets:
            for order in args.orders:
                rows.extend(R.read(args.output_root / name / order / "results.json")["rows"])
        save(args.output_root / "results.json", {"protocol": manifest, "rows": rows})
    print("UNIFIED_Q_TOPOLOGY_COMPARE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
