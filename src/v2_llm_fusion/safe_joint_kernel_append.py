"""Label-free topology and signature controls; synthetic C0 rows stay fixed."""
from __future__ import annotations

import json
import math

import torch
import torch.nn.functional as F


WEIGHTS = {
    "joint_kernel": (.4, .3, .3), "kernel_x": (1., 0., 0.),
    "kernel_t": (0., 1., 0.), "kernel_g": (0., 0., 1.),
    "kernel_xt": (4 / 7, 3 / 7, 0.),
}
VARIANTS = tuple(WEIGHTS) + ("degree_balanced", "coverage", "operator_stable")
SIGNATURE_SEED = 20260920


def copy_value(value):
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, list):
        return list(value)
    if isinstance(value, dict):
        return dict(value)
    return value


def kernel_matrix(left, right, weights):
    result = torch.zeros((len(left["x"]), len(right["x"])))
    for key, weight in zip(("x", "text_emb", "graph_signature"), weights):
        if weight:
            result += weight * (F.normalize(left[key].float(), dim=1)
                                @ F.normalize(right[key].float(), dim=1).T)
    return result


def edge_tensor(pairs):
    if not pairs:
        return torch.empty((2, 0), dtype=torch.long)
    edge = torch.tensor(pairs, dtype=torch.long).T.contiguous()
    return torch.cat((edge, edge.flip(0)), dim=1)


def edge_redundancy(score, i, j, chosen):
    """Cosine of edge midpoints in the positive-semidefinite joint kernel."""
    if not chosen:
        return 0.
    denom1 = (score[i, i] + score[j, j] + 2 * score[i, j]).clamp_min(1e-12)
    return max(float((score[i, a] + score[i, b] + score[j, a] + score[j, b]) /
                     (denom1 * (score[a, a] + score[b, b] + 2 * score[a, b])
                      .clamp_min(1e-12)).sqrt()) for a, b in chosen)


def select_internal(block, weights, variant, degree=1.):
    score = kernel_matrix(block, block, weights)
    n = len(score)
    target = min(max(0, int(n * degree)), n * (n - 1) // 2)
    pairs = sorted([(float(score[i, j]), i, j) for i in range(n)
                    for j in range(i + 1, n)], key=lambda p: (-p[0], p[1], p[2]))
    if variant not in ("degree_balanced", "coverage"):
        return edge_tensor([(i, j) for _, i, j in pairs[:target]])
    if degree != 1.:
        raise ValueError("degree-constrained controls are registered only at degree=1")
    # A path forest, without premature cycles, can always be completed to a
    # Hamiltonian path and closed. This enforces cap=2 AND exactly n pairs.
    parent, counts, chosen = list(range(n)), [0] * n, []
    endpoints = torch.tensor([(i, j) for _, i, j in pairs], dtype=torch.long)
    redundancy = torch.zeros(len(pairs))
    if len(pairs):
        a, b = endpoints[:, 0], endpoints[:, 1]
        midpoint_norm = (score[a, a] + score[b, b] + 2 * score[a, b]).clamp_min(1e-12)

    def root(i):
        while i != parent[i]:
            i = parent[i]
        return i

    while len(chosen) < target:
        allowed = []
        penalties = redundancy.tolist()
        for index, (base, i, j) in enumerate(pairs):
            if counts[i] >= 2 or counts[j] >= 2 or (i, j) in chosen:
                continue
            if root(i) == root(j) and len(chosen) < n - 1:
                continue
            value = base
            if variant == "coverage":
                value -= .15 * penalties[index]
            allowed.append((value, base, -i, -j, i, j))
        if not allowed:
            raise AssertionError("degree-2 path completion failed")
        *_, i, j = max(allowed)
        chosen.append((i, j))
        counts[i] += 1
        counts[j] += 1
        parent[root(i)] = root(j)
        if variant == "coverage":
            norm = (score[i, i] + score[j, j] + 2 * score[i, j]).clamp_min(1e-12)
            current = (score[a, i] + score[a, j] + score[b, i] + score[b, j]) / (midpoint_norm * norm).sqrt()
            redundancy = current if len(chosen) == 1 else torch.maximum(redundancy, current)
    return edge_tensor(chosen)


def cross_pairs(old, block, weights, variant):
    old_n, new_n = len(old["x"]), len(block["x"])
    score = kernel_matrix(block, old, weights)
    if variant == "operator_stable":
        if old_n < 2:
            raise ValueError("equal-edge one-way control requires at least two old rows")
        pairs = [(old_n + i, j) for i in range(new_n) for j in
                 sorted(range(old_n), key=lambda j: (-float(score[i, j]), j))[:2]]
        return torch.tensor(pairs, dtype=torch.long).T.contiguous(), {
            "cross_direction": "new_receives_old", "cross_capacity": None}
    balanced = variant in ("degree_balanced", "coverage")
    capacity = max(1, math.ceil(new_n / old_n)) if balanced else None
    counts, pairs = [0] * old_n, []
    for i in range(new_n):
        eligible = [j for j in range(old_n) if capacity is None or counts[j] < capacity]
        j = min(eligible, key=lambda j: (-float(score[i, j]), j))
        counts[j] += 1
        pairs.append((j, old_n + i))
    return edge_tensor(pairs), {"cross_direction": "bidirectional", "cross_capacity": capacity}


def append_variant(old, block, keep, variant, degree=1.):
    if variant not in VARIANTS:
        raise ValueError(variant)
    if old is not None and keep != len(old["x"]):
        raise ValueError("strict append cannot prune or refresh old rows")
    weights = WEIGHTS.get(variant, WEIGHTS["joint_kernel"])
    internal = select_internal(block, weights, variant, degree)
    audit = {"variant": variant, "weights": list(weights), "old_rows": keep,
             "new_rows": len(block["x"]), "internal_edges": internal.shape[1], "cross_edges": 0}
    if old is None:
        out = {k: copy_value(v) for k, v in block.items() if k != "audit_source_groups"}
        out["edge_index"] = internal
        return out, audit
    keys = ("x", "text_emb", "mass", "depth", "stage", "graph_signature")
    out = {key: torch.cat((old[key], block[key])) for key in keys}
    out["texts"] = old["texts"] + block["texts"]
    if old.get("signature_config") != block.get("signature_config"):
        raise ValueError("signature space changed across stages")
    if "signature_config" in block:
        out["signature_config"] = dict(block["signature_config"])
    cross, cross_audit = cross_pairs(old, block, weights, variant)
    out["edge_index"] = torch.cat((old["edge_index"], internal + keep, cross), dim=1)
    for key in keys:
        if not torch.equal(old[key], out[key][:keep]):
            raise AssertionError(f"old field changed: {key}")
    audit.update(cross_audit, cross_edges=cross.shape[1])
    return out, audit


def compress_signature(block, mode):
    out = {k: copy_value(v) for k, v in block.items()}
    sig = block["graph_signature"].float()
    if mode == "fp16":
        out["graph_signature"] = sig.half()
    elif mode == "sketch64":
        generator = torch.Generator().manual_seed(SIGNATURE_SEED)
        projection = torch.randint(0, 2, (sig.shape[1], 64), generator=generator).float()
        projection = (2 * projection - 1) / math.sqrt(64)
        out["graph_signature"] = F.normalize(sig @ projection, dim=1)
    elif mode != "fp32":
        raise ValueError(mode)
    out["signature_config"] = {"mode": mode, "seed": SIGNATURE_SEED,
                               "source_dim": sig.shape[1], "stored_dim": out["graph_signature"].shape[1]}
    return out


def signature_metadata_bytes(memory):
    if "signature_config" not in memory:
        return 0
    return len(json.dumps(memory["signature_config"], sort_keys=True, separators=(",", ":")).encode())
