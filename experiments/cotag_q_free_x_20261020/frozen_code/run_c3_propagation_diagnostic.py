"""C3 controlled propagation diagnostic for continual synthetic memory.

This experiment keeps the C0 block, rows, text, class order and student fixed.
Only the cross-block message operator changes:

* current_bidir: C0 bidirectional text-nearest cross edges, degree 1;
* new_receive_old: only (new, old) edges, so new rows receive old messages;
* old_receive_new: only (old, new) edges, so old rows receive new messages;
* bounded_bidir: bidirectional cross edges with block-local normalization and
  a bounded cross weight, preserving within-block normalization.

The edge convention follows ``adj @ x`` in the project: row i receives the
feature of column j.  The diagnostic is deliberately separate from P3-B and
does not modify the frozen v4 runner.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import run_p3b_topology_diagnostic as B

G, SCA, P = B.G, B.SCA, B.P
C, O, U = B.C, B.O, B.U

ARMS = ("current_bidir", "new_receive_old", "old_receive_new", "bounded_bidir")
METHOD_SEED = 9301
CROSS_WEIGHT = 0.25


def save_json(path: Path, value) -> None:
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def sync(device: str) -> None:
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def _nearest_pairs(old: dict, block: dict, stage: int) -> torch.Tensor:
    """Return [old_id, new_id] pairs using the fixed C0 text selector."""
    old_n, new_n = len(old["x"]), len(block["x"])
    if old_n == 0 or new_n == 0:
        return torch.empty((2, 0), dtype=torch.long)
    old_t = F.normalize(old["text_emb"].float(), dim=1)
    new_t = F.normalize(block["text_emb"].float(), dim=1)
    nearest = (new_t @ old_t.T).argmax(1)
    new_ids = torch.arange(new_n, dtype=torch.long)
    return torch.stack((nearest.long(), new_ids), dim=0)


def _cross_edges(old: dict, block: dict, variant: str, stage: int) -> torch.Tensor:
    pairs = _nearest_pairs(old, block, stage)
    if pairs.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long)
    old_ids, new_ids = pairs[0], pairs[1] + len(old["x"])
    old_to_new = torch.stack((old_ids, new_ids), dim=0)
    new_to_old = torch.stack((new_ids, old_ids), dim=0)
    if variant == "current_bidir" or variant == "bounded_bidir":
        return torch.cat((old_to_new, new_to_old), dim=1)
    if variant == "new_receive_old":
        return new_to_old
    if variant == "old_receive_new":
        return old_to_new
    raise ValueError(f"unknown C3 variant: {variant}")


def _raw_local_degree(edge_index: torch.Tensor, n: int) -> torch.Tensor:
    """Count local support entries, including one self-loop per row."""
    edges = edge_index.long().cpu()
    loops = torch.arange(n, dtype=torch.long).repeat(2, 1)
    support = torch.cat((edges, loops), dim=1)
    degree = torch.zeros(n, dtype=torch.float32)
    degree.scatter_add_(0, support[0], torch.ones(support.shape[1]))
    return degree.clamp_min(1.0)


def _weighted_support(edge_index: torch.Tensor, n: int,
                      device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    """Return the project's standard normalized support on one block."""
    normalized = G.norm_adj(edge_index.long().cpu(), n, device)
    normalized = normalized.coalesce().cpu()
    return normalized.indices().long(), normalized.values().float()


def _append_bounded_support(old: dict, block: dict, cross: torch.Tensor,
                            cross_weight: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Preserve old weighted support and append a bounded current block.

    ``old`` may already carry explicit normalized values from a previous C3
    append.  Those values are kept bitwise; only the new block support and
    this round's cross messages are added.  This avoids rebuilding the whole
    history at stage 2 and prevents duplicate self-loops.
    """
    old_n = len(old["x"])
    new_n = len(block["x"])
    if "adj_values" in old:
        old_edges = old["edge_index"].long().cpu()
        old_values = old["adj_values"].float().cpu()
    else:
        old_edges, old_values = _weighted_support(
            old["edge_index"], old_n, "cpu")
    new_edges, new_values = _weighted_support(
        block["edge_index"], new_n, "cpu")
    new_edges = new_edges + old_n

    if cross.numel() == 0:
        return (torch.cat((old_edges, new_edges), dim=1),
                torch.cat((old_values, new_values), dim=0))

    old_degree = _raw_local_degree(old["edge_index"], old_n)
    new_degree = _raw_local_degree(block["edge_index"], new_n)
    src, dst = cross[0].long().cpu(), cross[1].long().cpu()
    src_old = src < old_n
    dst_old = dst < old_n
    src_degree = torch.empty(len(src), dtype=torch.float32)
    dst_degree = torch.empty(len(dst), dtype=torch.float32)
    src_degree[src_old] = old_degree[src[src_old]]
    src_degree[~src_old] = new_degree[src[~src_old] - old_n]
    dst_degree[dst_old] = old_degree[dst[dst_old]]
    dst_degree[~dst_old] = new_degree[dst[~dst_old] - old_n]
    cross_values = torch.full((cross.shape[1],), float(cross_weight))
    cross_values = cross_values / torch.sqrt(src_degree * dst_degree)
    return (torch.cat((old_edges, new_edges, cross), dim=1),
            torch.cat((old_values, new_values, cross_values), dim=0))


def append_c3(old: dict | None, block: dict, variant: str, stage: int,
              device: str) -> dict:
    if variant not in ARMS:
        raise ValueError(f"unknown C3 variant: {variant}")
    if old is None:
        out = dict(block)
        out["edge_index"] = SCA.memory_edges(block, device=device)
        out["topology_variant"] = variant
        return out
    old_n = len(old["x"])
    old_edges = SCA.memory_edges(old, device=device)
    new_edges = SCA.memory_edges(block, device=device) + old_n
    cross = _cross_edges(old, block, variant, stage)
    edge_index = torch.cat((old_edges.cpu(), new_edges.cpu(), cross), dim=1).long()
    out = {
        key: (old[key] + block[key] if key == "texts"
              else torch.cat((old[key], block[key])))
        for key in ("x", "text_emb", "texts", "mass", "depth", "stage")
    }
    out["edge_index"] = edge_index
    out["topology_variant"] = variant
    out["cross_edge_count"] = int(cross.shape[1])
    if variant == "bounded_bidir":
        weighted_edges, values = _append_bounded_support(
            old, block, cross, CROSS_WEIGHT)
        out["edge_index"] = weighted_edges
        out["adj_values"] = values
        out["adj_normalized"] = True
    return out


def _adjacency(memory: dict, device: str) -> torch.Tensor:
    return C.training_adjacency(memory, device).to_dense().detach().cpu()


def topology_metrics(previous: dict | None, current: dict, device: str) -> dict:
    if previous is None:
        return {"old_rows": 0, "new_rows": len(current["x"]),
                "old_propagation_relative_drift": 0.0,
                "old_total_x_propagation_relative_drift": 0.0,
                "cross_normalized_l1": 0.0, "cross_normalized_fro": 0.0}
    old_n = len(previous["x"])
    full = _adjacency(current, device)
    prior = _adjacency(previous, device)
    old_block = full[:old_n, :old_n]
    cross = full[:old_n, old_n:]
    x_old = previous["x"].float().cpu()
    x_new = current["x"][old_n:].float().cpu()
    base = prior @ x_old
    old_only = old_block @ x_old
    cross_message = cross @ x_new
    denom = base.norm().clamp_min(1e-12)
    return {
        "old_rows": old_n,
        "new_rows": len(current["x"]) - old_n,
        "old_propagation_relative_drift": float((old_only - base).norm() / denom),
        "old_total_x_propagation_relative_drift": float(
            (old_only + cross_message - base).norm() / denom),
        "cross_normalized_l1": float(cross.abs().sum()),
        "cross_normalized_fro": float(cross.norm()),
        "cross_x_message_relative": float(cross_message.norm() / denom),
        "edge_count": int(current["edge_index"].shape[1]),
        "weighted_support": bool("adj_values" in current),
    }


def fit_one(memory: dict, previous: dict | None, data: dict, stage: int,
            arm: str, seed: int, out: Path, device: str) -> dict:
    key = f"{arm}_s{stage}_seed{seed}"
    row_path, checkpoint = out / f"{key}.json", out / f"{key}.pt"
    steps = P.student_steps(len(memory["x"]))
    if row_path.exists() and checkpoint.exists():
        row = json.loads(row_path.read_text())
    else:
        sync(device)
        started = time.perf_counter()
        row = C.fit(memory, seed, steps, device, data, stage, None, False, checkpoint)
        sync(device)
        row["fit_wall_seconds_including_report_io"] = time.perf_counter() - started
        row["topology_variant"] = arm
        row["stage"] = stage
        row["old_embedding_drift"] = B.trained_old_embedding_drift(
            memory, previous, checkpoint, data, device)
        save_json(row_path, row)
    O.verify_fit(memory, row, checkpoint, data, stage, device)
    return row


def aggregate(rows: list[dict]) -> dict:
    def stat(key: str):
        vals = [r["metrics"]["all"][key] for r in rows
                if r["metrics"]["all"][key] is not None]
        return None if not vals else {"mean": float(np.mean(vals)),
                                      "std": float(np.std(vals)), "n": len(vals)}
    return {"final_f1": stat("cumulative_f1"), "old_f1": stat("old_f1"),
            "new_f1": stat("new_f1"), "accuracy": stat("accuracy"),
            "old_embedding_drift": {
                "mean": float(np.mean([r["old_embedding_drift"] for r in rows])),
                "std": float(np.std([r["old_embedding_drift"] for r in rows])),
            }}


def run_order(data: dict, groups: list[list[int]], arms: list[str],
              seeds: list[int], out: Path, device: str) -> list[dict]:
    out.mkdir(parents=True, exist_ok=True)
    blocks, block_audits = B.build_blocks(data, groups, "c3_canonical", out, device)
    all_results, topology_all, costs = [], {}, {}
    for arm in arms:
        memories, topology, old = [], [], None
        started_arm = time.perf_counter()
        for stage, block in enumerate(blocks):
            memory = append_c3(old, block, arm, stage, device)
            torch.save(memory, out / f"{arm}_s{stage}_memory.pt")
            topology.append(topology_metrics(old, memory, device))
            memories.append(memory)
            old = memory
        seed_rows_by_stage = []
        for stage, memory in enumerate(memories):
            seed_rows = [fit_one(memory, memories[stage - 1] if stage else None,
                                 data, stage, arm, seed, out, device)
                         for seed in seeds]
            result = {"arm": arm, "stage": stage, "rows": len(memory["x"]),
                      "metrics": aggregate(seed_rows),
                      "topology": topology[stage], "per_seed": seed_rows}
            seed_rows_by_stage.append(result)
            all_results.append(result)
            save_json(out / "progress.json", {"results": all_results,
                                               "topology": topology_all})
            print(json.dumps({"event": "stage_done", "arm": arm,
                              "stage": stage,
                              "f1": result["metrics"]["final_f1"]["mean"]}),
                  flush=True)
        topology_all[arm] = topology
        save_json(out / f"{arm}_topology.json", topology)
        flat = [r for s in seed_rows_by_stage for r in s["per_seed"]]
        costs[arm] = {
            "topology_seconds": time.perf_counter() - started_arm,
            "student_seconds": float(sum(r["seconds"] for r in flat)),
            "student_wall_seconds": float(sum(
                r["fit_wall_seconds_including_report_io"] for r in flat)),
            "student_steps": int(sum(r["steps"] for r in flat)),
            "student_fits": len(flat),
        }
    save_json(out / "results.json", {"results": all_results,
                                      "topology": topology_all,
                                      "costs": costs})
    return all_results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cora", "citeseer"], required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    ap.add_argument("--minilm", default="/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip")
    args = ap.parse_args()
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    data = B.prepare(args.dataset, args.minilm, args.device, args.output)
    groups = B.stage_orders(args.dataset)["canonical"]
    results = run_order(data, groups, list(args.arms), list(B.STUDENT_SEEDS),
                        args.output / "canonical", args.device)
    plan = {
        "version": "c3-propagation-diagnostic-v1",
        "dataset": args.dataset, "stage_classes": groups,
        "arms": list(args.arms), "student_seeds": list(B.STUDENT_SEEDS),
        "rate": 0.05, "allocation": "dispersion", "x_operator": "mean",
        "text_recipe": "joint_g5_c320", "cross_degree": 1,
        "cross_weight": CROSS_WEIGHT,
        "edge_convention": "adj[row] receives column feature",
        "bounded_norm": "within-block degree excludes cross edges; cross weight attenuated",
        "test_labels_evaluated": False,
        "student_steps": "protocol_v3.student_steps(rows)",
        "code_sha256": {str(Path(p)): B.sha(Path(p)) for p in
                         [__file__, B.__file__, C.__file__, O.__file__, U.__file__,
                          SCA.__file__, P.__file__]},
        "results": {"rows": len(results)},
    }
    save_json(args.output / "plan.json", plan)
    save_json(args.output / "acceptance.json", {
        "complete": True, "arms": list(args.arms), "orders": ["canonical"],
        "fits": len(args.arms) * 3 * len(B.STUDENT_SEEDS),
        "test_labels_evaluated": False,
        "control": "fixed C0; propagation operator only",
    })
    print("C3_DONE", flush=True)


if __name__ == "__main__":
    main()
