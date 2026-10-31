"""Deterministic constructive initializer for static TAG condensation.

Motivation (STATIC_DIAG 2026-09-08): frozen compact quality varies 49.9-80.1
dev Macro-F1 across condensation seeds while the prepared data is identical.
The variance is manufactured by arbitrary randomness in the constructor
(multinomial k-means++ seeding; teacher-dependent learned topology), not by
the data. A deterministic constructor therefore removes cross-seed variance
by construction, leaving only student-seed noise.

Components (all label-free, all deterministic):
- greedy mass-weighted farthest-point seeding (maxmin), started from the
  point with the largest mass (ties: lowest index);
- standard Lloyd refinement on the joint normalized X/text space;
- per-cluster medoid gives the representative text/origin (real, readable);
- synthetic X is the mass-weighted cluster mean (``centroid``) or the
  medoid's real X (``real``);
- topology: deterministic mutual-kNN in frozen BERT CLS text space with the
  same edge budget (degree * n) as the audited link-predictor path. No
  teacher, no training, no RNG.
"""
from __future__ import annotations

from typing import Any

import torch

from protocol_data import _as_graph, _normalize, subset_graph


def _joint(source: dict[str, Any]) -> torch.Tensor:
    return torch.cat((_normalize(source["x"].detach().cpu()),
                      _normalize(source["text_emb"].detach().cpu())), dim=1)


def deterministic_select(source: dict[str, Any], k: int, x_mode: str = "centroid",
                         lloyd_iters: int = 25) -> dict[str, Any]:
    """Deterministic compact selection; same inputs always give same output."""
    _as_graph(source)
    n = int(source["x"].shape[0])
    if not 1 < k <= n:
        raise ValueError(f"k must be in [2, {n}]")
    if x_mode not in ("centroid", "real"):
        raise ValueError("x_mode must be centroid or real")
    joint = _joint(source)
    mass = source["weights"].detach().float().cpu()
    if not torch.isfinite(joint).all() or (mass <= 0).any():
        raise ValueError("selection requires finite features and positive mass")

    # Greedy mass-weighted maxmin seeding. First center: max mass, tie -> min index.
    first = int(torch.argmax(mass))  # argmax returns first maximal index
    centers_idx = [first]
    dist = torch.cdist(joint, joint[first:first + 1]).flatten().square() * mass
    for _ in range(1, k):
        dist[torch.tensor(centers_idx)] = -1.0
        nxt = int(torch.argmax(dist))
        centers_idx.append(nxt)
        newd = torch.cdist(joint, joint[nxt:nxt + 1]).flatten().square() * mass
        dist = torch.minimum(dist, newd)
    centers_idx = torch.tensor(centers_idx, dtype=torch.long)

    centers = joint[centers_idx].clone()
    for _ in range(lloyd_iters):
        assignment = torch.cdist(joint, centers).argmin(dim=1)
        assignment[centers_idx] = torch.arange(k)
        updated = []
        for cluster in range(k):
            members = torch.where(assignment == cluster)[0]
            updated.append((joint[members] * mass[members, None]).sum(0) / mass[members].sum())
        new_centers = torch.stack(updated)
        if torch.allclose(new_centers, centers, atol=1e-6, rtol=0):
            break
        centers = new_centers

    assignment = torch.cdist(joint, centers).argmin(dim=1)
    assignment[centers_idx] = torch.arange(k)
    representatives = []
    for cluster in range(k):
        members = torch.where(assignment == cluster)[0]
        local = torch.cdist(joint[members], centers[cluster:cluster + 1]).flatten()
        representatives.append(int(members[int(torch.argmin(local))]))
    selected = torch.tensor(sorted(set(representatives)), dtype=torch.long)
    if len(selected) < k:
        # Duplicated medoids across clusters: fill with unused maxmin points.
        used = set(selected.tolist())
        pool = [i for i in range(n) if i not in used]
        fill_d = torch.cdist(joint[pool], joint[selected]).min(dim=1).values * mass[pool]
        order = torch.argsort(fill_d, descending=True)
        for j in order[: k - len(selected)]:
            used.add(pool[int(j)])
        selected = torch.tensor(sorted(used), dtype=torch.long)

    # Recompute assignment against final representatives for X/weights.
    assignment = torch.cdist(joint, joint[selected]).argmin(dim=1)
    assignment[selected] = torch.arange(len(selected))
    out_x, out_w = [], []
    for cluster in range(len(selected)):
        members = torch.where(assignment == cluster)[0]
        m = mass[members]
        out_w.append(m.sum())
        if x_mode == "centroid":
            out_x.append((source["x"].detach().cpu()[members].float() * m[:, None]).sum(0) / m.sum())
        else:
            out_x.append(source["x"].detach().cpu()[selected[cluster]].float())
    result = subset_graph(source, selected)
    result["x"] = torch.stack(out_x).to(source["x"].device)
    result["weights"] = torch.stack(out_w).to(source["weights"].device)
    result["metadata"] = {"initializer": "deterministic_maxmin_lloyd_medoid",
                          "x_mode": x_mode, "lloyd_iters": lloyd_iters,
                          "deterministic": True, "text_selection": "real medoid text"}
    return result


def knn_topology(initial: dict[str, Any], degree: float = 1.5) -> dict[str, Any]:
    """Deterministic symmetric kNN links in frozen text-embedding space.

    Edge budget matches the audited link-predictor path: ``int(n * degree)``
    undirected pairs plus self loops. Pair score is cosine similarity of the
    frozen BERT CLS embeddings; ties resolve by fixed pair order.
    """
    result = {key: (value.clone() if torch.is_tensor(value) else list(value) if isinstance(value, list) else value)
              for key, value in initial.items()}
    n = len(result["x"])
    loops = torch.arange(n).repeat(2, 1)
    count = min(int(n * degree), n * (n - 1) // 2)
    if n < 2 or count < 1:
        result["edge_index"] = loops
        return result
    text = _normalize(result["text_emb"].detach().cpu())
    pairs = torch.triu_indices(n, n, offset=1)
    score = (text[pairs[0]] * text[pairs[1]]).sum(1)
    chosen = pairs[:, torch.topk(score, count).indices]
    result["edge_index"] = torch.cat((chosen, chosen.flip(0), loops), 1)
    meta = dict(result.get("metadata", {}))
    meta.update({"topology": {"kind": "deterministic_text_knn", "degree": degree,
                              "undirected_pairs": int(count)}, "deterministic_topology": True})
    result["metadata"] = meta
    return result
