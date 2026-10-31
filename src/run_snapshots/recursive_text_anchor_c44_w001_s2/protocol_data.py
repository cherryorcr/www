"""Small, label-free data containers used by the native TAG v1 protocol.

Labels deliberately live outside these graph dictionaries.  The preparation
script may use labels to define the class-stream split, while a condensation
runner receives only the dictionaries returned here.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import torch


GRAPH_KEYS = ("x", "edge_index", "text_emb", "texts", "weights", "origin_ids", "stage")


def _as_graph(data: dict[str, Any]) -> dict[str, Any]:
    missing = [key for key in GRAPH_KEYS if key not in data]
    if missing:
        raise KeyError(f"graph is missing keys: {missing}")
    n = int(data["x"].shape[0])
    if data["edge_index"].shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E]")
    if data["edge_index"].numel() and (
        int(data["edge_index"].min()) < 0 or int(data["edge_index"].max()) >= n
    ):
        raise ValueError("edge_index has an out-of-range endpoint")
    for key in ("text_emb", "weights", "origin_ids", "stage"):
        if int(data[key].shape[0]) != n:
            raise ValueError(f"{key} has a different node count")
    if len(data["texts"]) != n:
        raise ValueError("texts has a different node count")
    return data


def _empty_like(data: dict[str, Any], n: int = 0) -> dict[str, Any]:
    return {
        "x": data["x"].new_empty((n, data["x"].shape[1])),
        "edge_index": data["edge_index"].new_empty((2, 0)),
        "text_emb": data["text_emb"].new_empty((n, data["text_emb"].shape[1])),
        "texts": [],
        "weights": data["weights"].new_empty((n,)),
        "origin_ids": data["origin_ids"].new_empty((n,), dtype=torch.long),
        "stage": data["stage"].new_empty((n,), dtype=torch.long),
    }


def subset_graph(data: dict[str, Any], indices: Sequence[int] | torch.Tensor) -> dict[str, Any]:
    """Return an induced, re-indexed graph on ``indices``.

    The function keeps all non-label fields and drops edges crossing the
    subset boundary.  Duplicate indices are rejected because they make the
    origin-to-local mapping ambiguous.
    """
    _as_graph(data)
    idx = torch.as_tensor(indices, dtype=torch.long, device=data["x"].device).flatten()
    n = int(data["x"].shape[0])
    if idx.numel() and (int(idx.min()) < 0 or int(idx.max()) >= n):
        raise IndexError("subset index is out of bounds")
    if idx.unique().numel() != idx.numel():
        raise ValueError("subset indices must be unique")
    if idx.numel() == 0:
        return _empty_like(data)
    remap = torch.full((n,), -1, dtype=torch.long, device=idx.device)
    remap[idx] = torch.arange(idx.numel(), device=idx.device)
    edges = data["edge_index"].to(idx.device)
    keep = (remap[edges[0]] >= 0) & (remap[edges[1]] >= 0)
    new_edges = torch.stack((remap[edges[0, keep]], remap[edges[1, keep]]), dim=0)
    return {
        "x": data["x"].index_select(0, idx).clone(),
        "edge_index": new_edges.to(data["edge_index"].device),
        "text_emb": data["text_emb"].index_select(0, idx).clone(),
        "texts": [data["texts"][int(i)] for i in idx.cpu().tolist()],
        "weights": data["weights"].index_select(0, idx).clone(),
        "origin_ids": data["origin_ids"].index_select(0, idx).clone(),
        "stage": data["stage"].index_select(0, idx).clone(),
    }


def concat_graphs(graphs: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Concatenate graphs as a disjoint union, offsetting edge indices."""
    items = list(graphs)
    if not items:
        raise ValueError("concat_graphs requires at least one graph")
    for graph in items:
        _as_graph(graph)
    x, text_emb, weights, origin_ids, stages, texts, edges = [], [], [], [], [], [], []
    offset = 0
    for graph in items:
        n = int(graph["x"].shape[0])
        x.append(graph["x"])
        text_emb.append(graph["text_emb"])
        weights.append(graph["weights"])
        origin_ids.append(graph["origin_ids"])
        stages.append(graph["stage"])
        texts.extend(graph["texts"])
        edges.append(graph["edge_index"] + offset)
        offset += n
    return {
        "x": torch.cat(x, dim=0),
        "edge_index": torch.cat(edges, dim=1),
        "text_emb": torch.cat(text_emb, dim=0),
        "texts": texts,
        "weights": torch.cat(weights, dim=0),
        "origin_ids": torch.cat(origin_ids, dim=0),
        "stage": torch.cat(stages, dim=0),
    }


def _normalize(value: torch.Tensor) -> torch.Tensor:
    return value.float() / value.float().norm(dim=1, keepdim=True).clamp_min(1e-12)


def select_initial(
    source: dict[str, Any],
    k: int,
    seed: int,
    mode: str = "random",
) -> dict[str, Any]:
    """Make a compact initial graph using normalized joint X/text k-means.

    ``joint_medoid`` runs a deterministic, small k-means over concatenated
    normalized feature/text vectors and chooses a real text/origin per cluster.
    Synthetic X uses weighted cluster means; each text embedding is the frozen
    encoding of its selected readable source text. ``random`` keeps the exact
    selected source X/text and only assigns representative mass by proximity.
    This is a practical simplification of TAGSAM's CMI text
    selection, not the paper's CMI implementation.
    """
    _as_graph(source)
    n = int(source["x"].shape[0])
    if not 0 < k <= n:
        raise ValueError(f"k must be in [1, {n}]")
    if mode not in {"random", "joint_medoid"}:
        raise ValueError("mode must be random or joint_medoid")
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    joint = torch.cat((_normalize(source["x"].detach().cpu()), _normalize(source["text_emb"].detach().cpu())), dim=1)
    mass = source["weights"].detach().float().cpu()
    if not torch.isfinite(joint).all() or not torch.isfinite(mass).all() or (mass <= 0).any():
        raise ValueError("selection requires finite features and strictly positive mass")
    if mode == "random":
        centers_idx = torch.randperm(n, generator=gen)[:k]
    else:
        # k-means++-style seeded initialization keeps the routine dependency-free.
        centers_idx = [int(torch.multinomial(mass, 1, generator=gen))]
        for _ in range(1, k):
            centers = joint[centers_idx]
            dist = torch.cdist(joint, centers).square().min(dim=1).values * mass
            dist[torch.tensor(centers_idx)] = 0
            if dist.sum() <= 1e-20:
                unused = torch.ones(n, dtype=torch.bool)
                unused[torch.tensor(centers_idx)] = False
                # Identical embeddings are valid; choose a distinct source row.
                chosen = torch.where(unused)[0][torch.randint(n - len(centers_idx), (1,), generator=gen)]
                centers_idx.append(int(chosen))
            else:
                centers_idx.append(int(torch.multinomial(dist, 1, generator=gen)))
        centers_idx = torch.tensor(centers_idx, dtype=torch.long)
    centers = joint[centers_idx].clone()
    for _ in range(12 if mode == "joint_medoid" else 0):
        assignment = torch.cdist(joint, centers).argmin(dim=1)
        # Distinct anchors guarantee nonempty, disjoint clusters, including ties.
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
        representatives.append(
            members[torch.cdist(joint[members], centers[cluster:cluster + 1]).flatten().argmin()]
            if mode == "joint_medoid" else centers_idx[cluster]
        )
    selected_local = torch.stack(representatives).long()
    # Reassign to the actual distinct representatives, not obsolete centroids.
    assignment = torch.cdist(joint, joint[selected_local]).argmin(dim=1)
    assignment[selected_local] = torch.arange(k)
    out_x, out_weights = [], []
    for cluster in range(k):
        members = torch.where(assignment == cluster)[0]
        member_mass = mass[members]
        total = member_mass.sum()
        mean_x = (source["x"].detach().cpu()[members].float() * member_mass[:, None]).sum(0) / total
        out_x.append(mean_x)
        out_weights.append(total)
    # A selected induced graph is an explicit, conservative topology proxy.
    induced = subset_graph(source, selected_local)
    if mode == "joint_medoid":
        induced["x"] = torch.stack(out_x).to(source["x"].device)
    induced["weights"] = torch.stack(out_weights).to(source["weights"].device)
    return induced
