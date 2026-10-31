"""Minimal block-preserving joint-kernel append for the Phase-2 gate.

This module deliberately changes only the topology of a newly appended block.
Synthetic X, fused/re-encoded text, row budget, mass pruning, student, and
training schedule remain owned by the frozen C0 runner.  The graph component
is a source-only one-hop summary computed on the current delta graph; it never
reads historical raw rows or holdout labels.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


KERNEL_WEIGHTS = (0.4, 0.3, 0.3)  # X, text, source-local graph summary


def induced_edges(edge_index: torch.Tensor, ids: torch.Tensor,
                  total: int) -> torch.Tensor:
    ids = torch.sort(ids.long().cpu()).values
    if len(ids) == 0:
        return torch.empty((2, 0), dtype=torch.long)
    remap = torch.full((int(total),), -1, dtype=torch.long)
    remap[ids] = torch.arange(len(ids), dtype=torch.long)
    local = remap[edge_index.long().cpu()]
    keep = (local >= 0).all(0)
    return local[:, keep].contiguous()


def _normalise(value: torch.Tensor) -> torch.Tensor:
    return F.normalize(value.float(), dim=1)


def source_local_graph_summary(D: dict, ids: torch.Tensor,
                               joint: torch.Tensor) -> torch.Tensor:
    """Return a source-only one-hop joint representation for ``ids``.

    The graph is induced on the current incoming delta only.  Thus the
    summary can use current source edges but cannot accidentally reread old
    raw features, future classes, selector-dev, report-dev, or test nodes.
    """
    ids = ids.long().cpu()
    if not torch.equal(ids, torch.sort(ids).values) or len(ids.unique()) != len(ids):
        raise ValueError("source IDs must be sorted and unique")
    if "pool_mask" in D and not bool(D["pool_mask"][ids].all()):
        raise ValueError("graph signatures require pool-only source nodes")
    local_edges = induced_edges(D["edge_index"], ids, len(D["x"]))
    n = len(ids)
    if n == 0:
        return torch.empty_like(joint)
    # Add self loops and symmetric normalization without moving source data to
    # the student device.  This is a deterministic construction-only kernel.
    import run_gate1_continual as G

    adj = G.norm_adj(local_edges, n, "cpu")
    propagated = torch.sparse.mm(adj, joint.float().cpu())
    return _normalise(0.5 * joint.float().cpu() + 0.5 * propagated)


def _kernel_matrix(left: dict, right: dict) -> torch.Tensor:
    wx, wt, wg = KERNEL_WEIGHTS
    x = _normalise(left["x"]) @ _normalise(right["x"]).T
    t = _normalise(left["text_emb"]) @ _normalise(right["text_emb"]).T
    g = _normalise(left["graph_signature"]) @ _normalise(right["graph_signature"]).T
    return wx * x + wt * t + wg * g


def _top_undirected(score: torch.Tensor, count: int) -> torch.Tensor:
    """Select deterministic highest-scoring undirected pairs."""
    n = int(score.shape[0])
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((float(score[i, j]), i, j))
    count = min(max(0, int(count)), len(pairs))
    pairs.sort(key=lambda item: (-item[0], item[1], item[2]))
    chosen = pairs[:count]
    if not chosen:
        return torch.empty((2, 0), dtype=torch.long)
    forward = torch.tensor([[i for _, i, _ in chosen],
                            [j for _, _, j in chosen]], dtype=torch.long)
    return torch.cat((forward, forward.flip(0)), dim=1)


def kernel_edges(block: dict, degree: float = 1.0) -> torch.Tensor:
    n = len(block["x"])
    if n < 2:
        return torch.empty((2, 0), dtype=torch.long)
    count = min(max(0, int(n * float(degree))), n * (n - 1) // 2)
    return _top_undirected(_kernel_matrix(block, block), count)


def append_kernel_topology(old: dict, block: dict, degree: float = 1.0,
                           cross_degree: int = 1) -> torch.Tensor:
    old_n, new_n = len(old["x"]), len(block["x"])
    old_edges = old.get("edge_index")
    if old_edges is None:
        old_edges = kernel_edges(old, degree)
    new_edges = kernel_edges(block, degree)
    if new_edges.numel():
        new_edges = new_edges + old_n
    parts = [old_edges.long().cpu().contiguous()]
    if new_edges.numel():
        parts.append(new_edges.long().cpu().contiguous())
    if old_n and new_n and int(cross_degree) > 0:
        scores = _kernel_matrix(block, old)
        width = min(int(cross_degree), old_n)
        cross = []
        for new_id in range(new_n):
            order = sorted(range(old_n),
                           key=lambda old_id: (-float(scores[new_id, old_id]),
                                               old_id))[:width]
            for old_id in order:
                cross.append((old_id, old_n + new_id))
        if cross:
            forward = torch.tensor(cross, dtype=torch.long).T
            parts.extend((forward, forward.flip(0)))
    return torch.cat(parts, dim=1).long().contiguous()


def build_joint_block(U: Any, D: dict, ids: torch.Tensor, rows: int,
                      encoder: Any, stage: int, device: str,
                      allocation_mode: str = "dispersion",
                      recipe: str = "joint_g5_c320", base: dict | None = None) -> dict:
    """Build C0-equivalent rows plus source-local graph signatures."""
    ids = torch.sort(ids.long().cpu()).values
    classes = sorted(int(c) for c in D["stage_classes"][stage])
    allocation = U.allocate_rows_by_class(D, ids, classes, int(rows),
                                          allocation_mode)
    if base is None:
        base = U.build_stage_block(D, ids, rows, encoder, stage, recipe,
                                  "concat", device, allocation_mode, classes)
    all_x = D["x"][ids].float()
    all_t = D["text_emb"][ids].float()
    all_joint = torch.cat((F.normalize(all_x, dim=1), all_t), dim=1)
    all_graph = source_local_graph_summary(D, ids, all_joint)
    position = {int(source_id): index for index, source_id in enumerate(ids.tolist())}

    graph_sigs, source_groups = [], []
    for cls in classes:
        class_ids = torch.sort(ids[D["y"][ids] == int(cls)]).values
        row_count = min(int(allocation[int(cls)]), len(class_ids))
        if row_count <= 0:
            continue
        x = D["x"][class_ids].float()
        t = D["text_emb"][class_ids].float()
        joint = torch.cat((F.normalize(x, dim=1), t), dim=1)
        assign = U.SCA.cluster_assign(joint, torch.ones(len(joint)), row_count,
                                       device=device)
        local_positions = torch.tensor([position[int(v)] for v in class_ids],
                                       dtype=torch.long)
        for cluster_id in range(int(assign.max().item()) + 1):
            members = torch.where(assign == cluster_id)[0]
            row = len(graph_sigs)
            if not torch.equal(x[members].mean(0), base["x"][row]):
                raise AssertionError("source partition does not reconstruct C0 X")
            if float(base["mass"][row]) != len(members):
                raise AssertionError("source partition does not reconstruct C0 mass")
            graph_sigs.append(all_graph[local_positions[members]].mean(0))
            source_groups.append(class_ids[members].tolist())
    if not graph_sigs or len(graph_sigs) != len(base["x"]):
        raise ValueError("joint-kernel block has no rows")
    out = {key: value.clone() if isinstance(value, torch.Tensor) else list(value)
           if isinstance(value, list) else value for key, value in base.items()}
    out["graph_signature"] = _normalise(torch.stack(graph_sigs))
    # Source memberships are an offline audit record, not persistent memory.
    out["audit_source_groups"] = source_groups
    return out


def shrink_joint(memory: dict, keep: int) -> dict:
    if keep >= len(memory["x"]):
        return {key: value.clone() if isinstance(value, torch.Tensor) else list(value)
                if isinstance(value, list) else value for key, value in memory.items()}
    order = torch.argsort(memory["mass"], descending=True, stable=True)[:int(keep)]
    order = torch.sort(order).values
    out = {key: memory[key][order]
           for key in ("x", "text_emb", "mass", "depth", "stage",
                       "graph_signature")}
    out["texts"] = [memory["texts"][int(i)] for i in order]
    edges = memory.get("edge_index")
    if edges is not None:
        remap = torch.full((len(memory["x"]),), -1, dtype=torch.long)
        remap[order] = torch.arange(len(order), dtype=torch.long)
        edges = edges.long().cpu()
        valid = (remap[edges[0]] >= 0) & (remap[edges[1]] >= 0)
        out["edge_index"] = remap[edges[:, valid]].contiguous()
    return out


def _assert_old_rows(old: dict, kept: dict, new: dict) -> None:
    n = len(kept["x"])
    for key in ("x", "text_emb", "mass", "depth", "stage",
                "graph_signature"):
        if not torch.equal(kept[key], new[key][:n]):
            raise AssertionError(f"old retained field changed: {key}")
    if kept["texts"] != new["texts"][:n]:
        raise AssertionError("old retained text changed")
    old_edges = kept.get("edge_index")
    if old_edges is not None:
        new_old_edges = new["edge_index"]
        new_old_edges = new_old_edges[:, (new_old_edges < n).all(0)]
        if not torch.equal(old_edges, new_old_edges):
            raise AssertionError("old-old topology changed")


def append_joint(old: dict | None, block: dict, keep: int,
                 degree: float = 1.0, cross_degree: int = 1) -> tuple[dict, dict]:
    if old is not None and keep != len(old["x"]):
        raise ValueError("strict append cannot prune or refresh old rows")
    if old is None:
        out = {key: value.clone() if isinstance(value, torch.Tensor) else list(value)
               if isinstance(value, list) else value for key, value in block.items()}
        out["edge_index"] = kernel_edges(out, degree)
        out.pop("audit_source_groups", None)
        return out, {"old_rows": 0, "kept_rows": 0, "new_rows": len(block["x"]),
                     "old_old_edges_unchanged": True}
    kept = shrink_joint(old, int(keep))
    out = {
        key: (kept[key] + block[key] if key == "texts"
              else torch.cat((kept[key], block[key])))
        for key in ("x", "text_emb", "mass", "depth", "stage",
                    "graph_signature")
    }
    out["texts"] = kept["texts"] + list(block["texts"])
    out["edge_index"] = append_kernel_topology(kept, block, degree,
                                                cross_degree)
    _assert_old_rows(old, kept, out)
    if not torch.equal(_kernel_matrix(kept, kept), _kernel_matrix(
            {k: out[k][:len(kept["x"])] for k in
             ("x", "text_emb", "graph_signature")}, kept)):
        raise AssertionError("old-old relation kernel changed")
    return out, {
        "old_rows": int(len(old["x"])),
        "kept_rows": int(len(kept["x"])),
        "new_rows": int(len(block["x"])),
        "total_rows": int(len(out["x"])),
        "old_old_edges_unchanged": True,
        "kernel_weights": list(KERNEL_WEIGHTS),
        "degree": float(degree),
        "cross_degree": int(cross_degree),
    }
