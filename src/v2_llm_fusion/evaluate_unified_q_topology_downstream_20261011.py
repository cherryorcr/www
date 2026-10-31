"""Leakage-controlled downstream comparison for Unified-Q topology branches.

The constructor is rerun on a fixed train-edge split.  Held-out undirected
edges are scored after mapping raw nodes to their materialized synthetic slot.
All arms use the same feature score; topology arms receive one additional
binary synthetic-edge signal.  This is a small, auditable downstream probe,
not a claim that edge-free memory can train a GAE without edges.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

import run_unified_q_multimodal_20261006 as R
import unified_q_multimodal as UQ


ARMS = ("unified_q_edge_free", "unified_q", "unified_q_boundary")


def save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=True), encoding="utf-8")
    tmp.replace(path)


def undirected(edge: torch.Tensor, n: int) -> torch.Tensor:
    if edge is None or not edge.numel():
        return torch.empty((2, 0), dtype=torch.long)
    edge = edge.long().cpu()
    edge = edge[:, edge[0] != edge[1]]
    lo, hi = torch.minimum(edge[0], edge[1]), torch.maximum(edge[0], edge[1])
    keys = torch.unique(lo * int(n) + hi, sorted=True)
    return torch.stack((keys // int(n), keys % int(n)))


def bidirectional(edge: torch.Tensor) -> torch.Tensor:
    return torch.cat((edge, edge.flip(0)), dim=1) if edge.numel() else edge


def split_edges(data: dict, seed: int) -> dict:
    pool = data["pool_mask"].bool().cpu()
    n = len(data["x"])
    pos = undirected(data["edge_index"], n)
    keep = pool[pos[0]] & pool[pos[1]]
    pos = pos[:, keep]
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(pos.shape[1])
    n_test = max(1, int(round(len(perm) * 0.20)))
    n_val = max(1, int(round(len(perm) * 0.10)))
    test_pos = pos[:, perm[:n_test]]
    val_pos = pos[:, perm[n_test:n_test + n_val]]
    train_pos = pos[:, perm[n_test + n_val:]]
    allowed = pool.nonzero().flatten().numpy()
    u, v = np.triu_indices(len(allowed), 1)
    cand = np.stack((allowed[u], allowed[v]))
    pos_keys = set((pos[0] * n + pos[1]).tolist())
    mask = np.asarray([(int(a) * n + int(b)) not in pos_keys
                       for a, b in cand.T], dtype=bool)
    neg = torch.from_numpy(cand[:, mask]).long()
    neg_perm = rng.permutation(neg.shape[1])
    test_neg = neg[:, neg_perm[:len(test_pos.T)]]
    val_neg = neg[:, neg_perm[len(test_pos.T):len(test_pos.T) + len(val_pos.T)]]
    train_neg = neg[:, neg_perm[len(test_pos.T) + len(val_pos.T):
                               len(test_pos.T) + len(val_pos.T) + len(train_pos.T)]]
    return {"train_pos": train_pos, "val_pos": val_pos, "test_pos": test_pos,
            "train_neg": train_neg, "val_neg": val_neg, "test_neg": test_neg,
            "seed": int(seed), "pool_nodes": int(pool.sum()),
            "full_pos": pos}


def replace_delta_graph(data: dict, edge_index: torch.Tensor) -> dict:
    out = dict(data)
    out["edge_index"] = bidirectional(edge_index)
    return out


def stage_ids(data: dict, order: str):
    ids, budgets = R.BASE.set_stream(data, order)
    return ids, budgets


def slot_lookup(memory: dict, n_raw: int) -> torch.Tensor:
    lookup = torch.full((int(n_raw),), -1, dtype=torch.long)
    for slot, support in enumerate(memory.get("support_ids", [])):
        for raw_id in support:
            if 0 <= int(raw_id) < n_raw:
                lookup[int(raw_id)] = int(slot)
    return lookup


def pair_scores(memory: dict, pairs: torch.Tensor, lookup: torch.Tensor,
                topology_weight: float = 0.25) -> tuple[torch.Tensor, torch.Tensor]:
    slots = lookup[pairs]
    fused = torch.cat((F.normalize(memory["x"].float(), dim=1),
                       F.normalize(memory["text_emb"].float(), dim=1)), dim=1)
    score = (fused[slots[0]] * fused[slots[1]]).sum(1) / math.sqrt(2.0)
    edges = undirected(memory.get("edge_index", torch.empty((2, 0))), len(fused))
    edge_keys = set((edges[0] * len(fused) + edges[1]).tolist())
    lo, hi = torch.minimum(slots[0], slots[1]), torch.maximum(slots[0], slots[1])
    has_edge = torch.tensor([int(a) * len(fused) + int(b) in edge_keys
                             for a, b in zip(lo.tolist(), hi.tolist())], dtype=torch.float32)
    return score + float(topology_weight) * has_edge, has_edge


def metrics(pos: torch.Tensor, neg: torch.Tensor) -> dict:
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    s = np.r_[pos.numpy(), neg.numpy()]
    return {"auc": float(roc_auc_score(y, s)),
            "ap": float(average_precision_score(y, s)),
            "positives": int(len(pos)), "negatives": int(len(neg))}


def structure_metrics(memory: dict, data: dict, test_pos: torch.Tensor,
                      lookup: torch.Tensor) -> dict:
    edge = undirected(memory.get("edge_index", torch.empty((2, 0))), len(memory["x"]))
    edge_keys = set((edge[0] * len(memory["x"]) + edge[1]).tolist())
    mapped = lookup[test_pos]
    valid = (mapped >= 0).all(0)
    if not bool(valid.any()):
        return {"heldout_edge_recall": 0.0, "degree_pearson": 0.0,
                "synthetic_undirected_edges": int(edge.shape[1]),
                "synthetic_directed_edge_entries": int(memory.get("edge_index", torch.empty((2, 0))).shape[1])}
    mapped = mapped[:, valid]
    lo, hi = torch.minimum(mapped[0], mapped[1]), torch.maximum(mapped[0], mapped[1])
    hit = torch.tensor([int(a) * len(memory["x"]) + int(b) in edge_keys
                        for a, b in zip(lo.tolist(), hi.tolist())])
    raw_deg = torch.bincount(undirected(data["edge_index"], len(data["x"])).reshape(-1),
                             minlength=len(data["x"]))
    syn_deg = torch.bincount(edge.reshape(-1), minlength=len(memory["x"])) if edge.numel() else torch.zeros(len(memory["x"]), dtype=torch.long)
    valid_nodes = lookup >= 0
    sums = torch.zeros(len(memory["x"])); counts = torch.zeros(len(memory["x"]))
    sums.index_add_(0, lookup[valid_nodes], raw_deg[valid_nodes].float())
    counts.index_add_(0, lookup[valid_nodes], torch.ones(int(valid_nodes.sum())))
    mean_raw = sums / counts.clamp_min(1)
    corr = float(np.corrcoef(mean_raw.numpy(), syn_deg.float().numpy())[0, 1]) if len(memory["x"]) > 1 and float(mean_raw.std()) > 0 and float(syn_deg.float().std()) > 0 else 0.0
    return {"heldout_edge_recall": float(hit.float().mean()),
            "degree_pearson": corr,
            "synthetic_undirected_edges": int(edge.shape[1]),
            "synthetic_directed_edge_entries": int(memory.get("edge_index", torch.empty((2, 0))).shape[1])}


def construct(args, data: dict, split: dict, order: str, out: Path) -> dict:
    ids, budgets = stage_ids(data, order)
    train_data = replace_delta_graph(data, split["train_pos"])
    uq_cfg = UQ.Config(steps=args.q_steps, proxy_seeds=(2711,), materialize_top_k=1,
                       straight_through_text=True, hard_topology_forward=True,
                       seed=271006)
    cat_cfg = R.M.Config(seed=271006, cat_steps=3000, text_sources=5, text_mode="nearest")
    gm_cfg = R.GM.Config(steps=30, gradient_matching_mode="decoupled",
                         topology_mode="none", proxy_seeds=(2711,))
    local_args = argparse.Namespace(**vars(args))
    local_args.arms = list(ARMS)
    local_args.output_root = out
    local_args.cat_steps = 3000
    local_args.gm_steps = 30
    R.construct_cell(train_data, order, local_args, cat_cfg, gm_cfg, uq_cfg)
    return {"ids": ids, "budgets": budgets, "root": out / data["dataset"] / order}


def evaluate(args, data: dict, split: dict, order: str, root: Path) -> list[dict]:
    ids, budgets = stage_ids(data, order)
    rows = []
    for arm in ARMS:
        for stage, budget in enumerate(budgets):
            memory = torch.load(root / f"{arm}_s{stage}_memory.pt", map_location="cpu", weights_only=False)
            seen = torch.cat(ids[:stage + 1])
            keep_pos = torch.isin(split["test_pos"][0], seen) & torch.isin(split["test_pos"][1], seen)
            keep_neg = torch.isin(split["test_neg"][0], seen) & torch.isin(split["test_neg"][1], seen)
            pos = split["test_pos"][:, keep_pos]
            neg = split["test_neg"][:, keep_neg]
            lookup = slot_lookup(memory, len(data["x"]))
            valid_pos = (lookup[pos] >= 0).all(0)
            valid_neg = (lookup[neg] >= 0).all(0)
            pos = pos[:, valid_pos]
            neg = neg[:, valid_neg]
            pos_score, pos_topology = pair_scores(memory, pos, lookup)
            neg_score, neg_topology = pair_scores(memory, neg, lookup)
            pos_feature, _ = pair_scores(memory, pos, lookup, topology_weight=0.0)
            neg_feature, _ = pair_scores(memory, neg, lookup, topology_weight=0.0)
            row = {"dataset": data["dataset"], "order": order, "arm": arm,
                   "stage": stage, "memory_rows": int(budget),
                   "metrics": metrics(pos_score, neg_score),
                   "metrics_feature_only": metrics(pos_feature, neg_feature),
                   "topology_positive_rate": float(pos_topology.mean()) if len(pos_topology) else 0.0,
                   "topology_negative_rate": float(neg_topology.mean()) if len(neg_topology) else 0.0,
                   "structure": structure_metrics(memory, data, pos, lookup),
                   "construction_graph": "train-edge split only",
                   "test_edges_used_for_construction": False}
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("cora", "citeseer"), required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--minilm", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--order", default="order19611")
    parser.add_argument("--split-seed", type=int, default=29001)
    parser.add_argument("--q-steps", type=int, default=30)
    args = parser.parse_args()
    torch.set_num_threads(2)
    data = R.BASE.load_data(args.dataset, args.minilm, args.device, args.cache_root)
    data["prompt_emb"] = data["encoder"].encode(data["prompts"])
    split = split_edges(data, args.split_seed)
    structure_data = replace_delta_graph(data, split["train_pos"])
    out = args.output / "construction"
    out.mkdir(parents=True, exist_ok=True)
    save(args.output / "edge_split.json", {k: (v.tolist() if isinstance(v, torch.Tensor) else v)
         for k, v in split.items() if k != "full_pos"})
    construct(args, data, split, args.order, out)
    rows = evaluate(args, structure_data, split, args.order, out / args.dataset / args.order)
    save(args.output / "results.json", {"protocol": {
        "name": "unified-q-topology-downstream-20261011",
        "dataset": args.dataset, "order": args.order,
        "split_seed": args.split_seed, "arms": list(ARMS),
        "construction_edges": "train_pos only; held-out positives and negatives are never passed to constructor",
        "downstream_score": "fused X/text cosine + fixed 0.25 synthetic-edge indicator",
    }, "rows": rows})
    print("UNIFIED_Q_TOPOLOGY_DOWNSTREAM_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
