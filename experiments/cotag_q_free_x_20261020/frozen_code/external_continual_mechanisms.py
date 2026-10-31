"""CaT/GECC mechanisms adapted to a current-delta TAG memory interface.

Equation reimplementations, not official end-to-end reproductions.  See the
pinned upstream commits and adaptation table in the experiment protocol.
Constructors receive a sliced delta, never the full evaluation container.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import random

import numpy as np
from sklearn.cluster import KMeans
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class Config:
    seed: int = 270927
    cat_steps: int = 3000
    cat_hidden: int = 512
    cat_lr: float = 0.001
    gecc_restarts: int = 20
    gecc_weights: tuple[float, ...] = (0.0, 0.3, 0.9)
    gecc_balance: float = 1.0
    text_sources: int = 5
    text_chars: int = 320
    # The default preserves the previously frozen CaT adapter exactly.  The
    # other modes are deterministic TAG-specific text-selection controls.
    text_mode: str = "nearest"
    text_diversity: float = 0.15
    text_ridge: float = 1e-2


TEXT_MODES = ("nearest", "tagsam_style", "graph_conditioned",
              "graph_text_aligned", "text_center", "random",
              "wrong_class", "text_center_soft")


def ratio_allocation(labels: torch.Tensor, rows: int) -> dict[int, int]:
    """Largest-remainder class-ratio allocation with a feasible class floor."""
    classes, counts = torch.unique(labels.cpu().long(), sorted=True,
                                   return_counts=True)
    if not 1 <= rows <= len(labels) or rows < len(classes):
        raise ValueError("budget must cover each incoming class and fit delta")
    quota = torch.ones(len(classes), dtype=torch.long)
    remaining = rows - len(classes)
    while remaining:
        capacity = counts - quota
        eligible = capacity > 0
        weights = counts.double() * eligible
        fractional = remaining * weights / weights.sum()
        add = torch.minimum(fractional.floor().long(), capacity)
        quota += add
        remaining -= int(add.sum())
        if remaining:
            priority = torch.argsort(fractional - fractional.floor(),
                                     descending=True, stable=True)
            for j in priority.tolist():
                if quota[j] < counts[j] and remaining:
                    quota[j] += 1
                    remaining -= 1
    return dict(zip(classes.tolist(), quota.tolist()))


def row_labels(allocation: dict[int, int]) -> torch.Tensor:
    return torch.tensor([c for c in sorted(allocation)
                         for _ in range(allocation[c])], dtype=torch.long)


def normalized_adjacency(edges: torch.Tensor, n: int,
                         device: str = "cpu") -> torch.Tensor:
    """Simple undirected graph, one self loop, symmetric normalization."""
    edges = edges.long().cpu()
    if edges.numel() and (int(edges.min()) < 0 or int(edges.max()) >= n):
        raise ValueError("delta edge outside current-delta rows")
    edges = edges[:, edges[0] != edges[1]]
    index = torch.cat((edges, edges.flip(0), torch.arange(n).repeat(2, 1)), 1)
    keys = torch.unique(index[0] * n + index[1], sorted=True)
    index = torch.stack((keys // n, keys % n)).to(device)
    degree = torch.bincount(index[0], minlength=n).float().clamp_min(1)
    value = degree[index[0]].rsqrt() * degree[index[1]].rsqrt()
    return torch.sparse_coo_tensor(index, value, (n, n), device=device).coalesce()


def propagated_features(delta: dict, weights: tuple[float, ...],
                        device: str) -> torch.Tensor:
    x = delta["x"].float().to(device)
    adj = normalized_adjacency(delta["edge_index"], len(x), device)
    out = weights[0] * x
    for weight in weights[1:]:
        x = torch.sparse.mm(adj, x)
        out = out + weight * x
    return out


def random_features(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    # Official two-layer Encoder excludes its last classification layer.
    return F.normalize(F.relu(x @ weight), dim=1)


def class_mmd(real: torch.Tensor, synth: torch.Tensor, real_labels: torch.Tensor,
              synth_labels: torch.Tensor) -> torch.Tensor:
    losses = []
    for cls in torch.unique(synth_labels, sorted=True):
        a, b = real[real_labels == cls], synth[synth_labels == cls]
        if not len(a) or not len(b):
            raise ValueError("MMD label block lacks real or synthetic support")
        losses.append((a.mean(0) - b.mean(0)).square().sum())
    return torch.stack(losses).sum()


def _fit_graph_text_map(source_features: torch.Tensor,
                        text_emb: torch.Tensor,
                        ridge: float) -> tuple[torch.Tensor, float]:
    """Fit a current-delta-only graph-to-text linear alignment map.

    This is a construction-time diagnostic map, not a downstream Student
    parameter.  The bias is unregularized so that centered text embeddings do
    not force the graph features through the origin.
    """
    graph = F.normalize(source_features.float(), dim=1)
    text = F.normalize(text_emb.float(), dim=1)
    ones = torch.ones((len(graph), 1), dtype=graph.dtype,
                      device=graph.device)
    design = torch.cat((graph, ones), dim=1)
    gram = design.T @ design
    reg = torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
    reg[-1, -1] = 0.0
    rhs = design.T @ text
    mapping = torch.linalg.solve(gram + float(ridge) * reg, rhs)
    predicted = F.normalize(design @ mapping, dim=1)
    fit_cosine = float((predicted * text).sum(1).mean())
    return mapping, fit_cosine


def _select_text_sources(delta: dict, source_features: torch.Tensor,
                         x_row: torch.Tensor, cls: int, cfg: Config,
                         graph_text_map: torch.Tensor | None = None,
                         row_index: int = 0):
    """Select current-delta text sources with a fixed, auditable rule.

    ``nearest`` is the frozen historical behavior.  The two new modes do not
    optimize against downstream F1: they use only current-delta graph/text
    embeddings and an optional class prompt.  This keeps the first TAG
    extension focused on text construction rather than Student response.
    """
    mode = cfg.text_mode
    if mode not in TEXT_MODES:
        raise ValueError(f"unknown text mode: {mode}")
    if mode == "wrong_class":
        candidates = torch.where(delta["y"] != cls)[0].long()
    else:
        candidates = torch.where(delta["y"] == cls)[0].long()
    if not len(candidates):
        raise ValueError(f"no current-delta text candidates for class {cls}")
    k = min(int(cfg.text_sources), len(candidates))

    if mode in ("random", "wrong_class"):
        generator = torch.Generator(device="cpu").manual_seed(
            int(cfg.seed) + 1009 * int(delta["stage"]) +
            9176 * int(cls) + 53 * int(row_index))
        chosen_local = torch.randperm(len(candidates), generator=generator)[:k]
        chosen = candidates[chosen_local]
        return chosen, {
            "graph_score": None,
            "text_score": None,
            "prompt_score": None,
            "selection_score": [0.0] * len(chosen),
            "control": mode,
        }

    graph = F.normalize(source_features[candidates].float(), dim=1)
    query = F.normalize(x_row.float().view(1, -1), dim=1).squeeze(0)
    graph_score = graph @ query
    if mode == "nearest":
        order = torch.argsort(-graph_score, stable=True)[:k]
        chosen = candidates[order]
        return chosen, {
            "graph_score": graph_score[order].tolist(),
            "text_score": None,
            "prompt_score": None,
            "selection_score": graph_score[order].tolist(),
        }

    text_emb = delta.get("text_emb")
    if text_emb is None:
        raise ValueError(f"text mode {mode} requires delta text_emb")
    text = F.normalize(text_emb[candidates].float(), dim=1)
    class_text = F.normalize(text.mean(0, keepdim=True), dim=1).squeeze(0)
    text_score = text @ class_text

    prompt_score = torch.zeros_like(text_score)
    prompt_emb = delta.get("prompt_emb")
    if prompt_emb is not None and int(cls) < len(prompt_emb):
        prompt = F.normalize(prompt_emb[int(cls)].float().view(1, -1), dim=1)
        prompt_score = (text @ prompt.squeeze(0)).to(text_score)

    if mode in ("text_center", "text_center_soft"):
        score = text_score
    elif mode == "tagsam_style":
        # A fixed attribute/text matching proxy: graph coverage plus semantic
        # centrality.  This is labeled mechanism-inspired, not official
        # TAGSAM reproduction.
        score = 0.55 * graph_score + 0.35 * text_score + 0.10 * prompt_score
    elif mode == "graph_conditioned":
        # Graph-conditioned text: first identify the graph-compatible text
        # region, then score candidates against its soft text prototype.
        prior = torch.softmax(graph_score / 0.25, dim=0)
        graph_text_target = F.normalize(
            (prior.unsqueeze(1) * text).sum(0, keepdim=True), dim=1
        ).squeeze(0)
        cross_score = text @ graph_text_target
        score = 0.55 * graph_score + 0.30 * cross_score \
            + 0.10 * text_score + 0.05 * prompt_score
    else:
        if graph_text_map is None:
            raise ValueError("graph_text_aligned requires a fitted map")
        # Fit a graph-to-text relation on the current delta only.  The graph
        # query is a soft prototype of the graph-compatible source region;
        # the map converts that prototype into the frozen text space.
        graph_norm = F.normalize(source_features[candidates].float(), dim=1)
        prior = torch.softmax(graph_score / 0.25, dim=0)
        target_graph = F.normalize(
            (prior.unsqueeze(1) * graph_norm).sum(0, keepdim=True), dim=1
        )
        target_aug = torch.cat(
            (target_graph, torch.ones((1, 1), dtype=target_graph.dtype)), dim=1
        )
        aligned_target = F.normalize(target_aug @ graph_text_map, dim=1)
        cross_score = (text @ aligned_target.squeeze(0)).to(text_score)
        score = 0.55 * graph_score + 0.40 * cross_score \
            + 0.05 * prompt_score

    # Greedy MMR keeps the fixed top-k budget while avoiding five near-
    # duplicate snippets for a synthetic row.
    available = torch.ones(len(candidates), dtype=torch.bool)
    chosen_local = []
    for _ in range(k):
        adjusted = score.clone()
        if chosen_local:
            selected_text = text[torch.tensor(chosen_local)]
            redundancy = (text @ selected_text.T).max(dim=1).values
            adjusted = adjusted - float(cfg.text_diversity) * redundancy
        adjusted[~available] = -torch.inf
        best = int(torch.argmax(adjusted).item())
        chosen_local.append(best)
        available[best] = False
    selected = torch.tensor(chosen_local, dtype=torch.long)
    chosen = candidates[selected]
    return chosen, {
        "graph_score": graph_score[selected].tolist(),
        "text_score": text_score[selected].tolist(),
        "prompt_score": prompt_score[selected].tolist(),
        "selection_score": score[selected].tolist(),
    }


def attach_text(delta: dict, source_features: torch.Tensor, x: torch.Tensor,
                labels: torch.Tensor, encoder, cfg: Config) -> tuple[dict, dict]:
    """Attach fixed-budget current-delta text under an auditable selector."""
    texts, mass, provenance = [], [], []
    soft_embeddings = []
    selection_audit = []
    source_features, x = source_features.cpu(), x.cpu()
    graph_text_map = None
    graph_text_fit = None
    if cfg.text_mode == "graph_text_aligned":
        if delta.get("text_emb") is None:
            raise ValueError("graph_text_aligned requires delta text_emb")
        graph_text_map, graph_text_fit = _fit_graph_text_map(
            source_features, delta["text_emb"].cpu(), cfg.text_ridge)
    for row, cls in enumerate(labels.tolist()):
        chosen, scores = _select_text_sources(
            delta, source_features, x[row], int(cls), cfg, graph_text_map,
            row_index=row)
        candidate_count = int((delta["y"] == int(cls)).sum())
        # Count separators inside the fixed budget.  The previous adapter
        # divided the budget by k but could exceed it by k-1 spaces.
        separator_budget = max(0, len(chosen) - 1)
        per = max(1, (cfg.text_chars - separator_budget) // len(chosen))
        texts.append(" ".join(str(delta["texts"][i])[:per]
                              for i in chosen.tolist()))
        if len(texts[-1]) > cfg.text_chars:
            texts[-1] = texts[-1][:cfg.text_chars]
        provenance.append(delta["ids"][chosen].tolist())
        mass.append(float(candidate_count / int((labels == cls).sum())))
        selection_audit.append(scores)
        if cfg.text_mode == "text_center_soft":
            # Continuous current-delta class prototype. It preserves the
            # auditable top-k text provenance while giving Student a direct
            # weighted embedding instead of only a concatenated string.
            candidates = torch.where(delta["y"] == int(cls))[0].long()
            candidate_text = F.normalize(delta["text_emb"][candidates].float(), dim=1)
            center = F.normalize(candidate_text.mean(0, keepdim=True), dim=1).squeeze(0)
            weights = torch.softmax((candidate_text @ center) / 0.07, dim=0)
            prototype = F.normalize((weights.unsqueeze(1) * candidate_text).sum(0), dim=0)
            soft_embeddings.append(prototype)
    text_embedding = (torch.stack(soft_embeddings).cpu()
                      if cfg.text_mode == "text_center_soft"
                      else encoder.encode(texts).cpu())
    block = {"x": x.detach().float(), "text_emb": text_embedding,
             "texts": texts, "mass": torch.tensor(mass),
             "depth": torch.zeros(len(x), dtype=torch.long),
             "stage": torch.full((len(x),), delta["stage"], dtype=torch.long),
             "edge_index": torch.empty((2, 0), dtype=torch.long)}
    return block, {"text_source_ids": provenance, "row_classes": labels.tolist(),
                   "text_chars_max": max(map(len, texts)),
                   "text_adapter": f"{cfg.text_mode} / top-{cfg.text_sources} / "
                                   f"{cfg.text_chars} chars + re-encode",
                   "embedding_mode": ("weighted current-delta class prototype"
                                      if cfg.text_mode == "text_center_soft"
                                      else "frozen encoder of concatenated sources"),
                   "soft_prototype_temperature": (0.07
                                                   if cfg.text_mode == "text_center_soft"
                                                   else None),
                   "text_mode": cfg.text_mode,
                   "text_selection": selection_audit,
                   "text_diversity": float(cfg.text_diversity),
                   "text_ridge": float(cfg.text_ridge),
                   "graph_text_fit_cosine": graph_text_fit}


def cat_block(delta: dict, rows: int, encoder, cfg: Config,
              device: str) -> tuple[dict, dict]:
    """CaT official Encoder/CGM objective, common TAG text attachment."""
    if cfg.cat_steps < 1:
        raise ValueError("CaT requires positive optimization steps")
    allocation = ratio_allocation(delta["y"], rows)
    labels = row_labels(allocation)
    seed = cfg.seed + 1009 * delta["stage"]
    py_rng = random.Random(seed)
    initial_ids = []
    for cls, count in allocation.items():
        candidates = torch.where(delta["y"] == cls)[0].tolist()
        initial_ids.extend(py_rng.choices(candidates, k=count))
    initial = delta["x"][initial_ids].float().to(device)
    synth = torch.nn.Parameter(initial.clone())
    propagated = propagated_features(delta, (0.0, 1.0), device)
    real_labels, synth_labels = delta["y"].to(device), labels.to(device)
    gen = torch.Generator(device=device).manual_seed(seed)
    optimizer = torch.optim.Adam([synth], lr=cfg.cat_lr)
    bound = math.sqrt(6.0 / (initial.shape[1] + cfg.cat_hidden))
    trace = []
    # A fixed diagnostic bank compares endpoints in the same random spaces.
    check_gen = torch.Generator(device=device).manual_seed(seed + 999983)
    check_bank = [torch.empty((initial.shape[1], cfg.cat_hidden), device=device)
                  .uniform_(-bound, bound, generator=check_gen) for _ in range(4)]

    @torch.no_grad()
    def diagnostic(value):
        return float(torch.stack([
            class_mmd(random_features(propagated, w), random_features(value, w),
                      real_labels, synth_labels) for w in check_bank]).mean())

    before = diagnostic(initial)
    for step in range(cfg.cat_steps):
        weight = torch.empty((initial.shape[1], cfg.cat_hidden), device=device)
        weight.uniform_(-bound, bound, generator=gen)
        with torch.no_grad():
            target = random_features(propagated, weight)
        loss = class_mmd(target, random_features(synth, weight),
                         real_labels, synth_labels)
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite CaT MMD")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 0 or (step + 1) % 100 == 0 or step == cfg.cat_steps - 1:
            trace.append({"step": step + 1, "random_space_loss": float(loss.detach())})
    after = diagnostic(synth)
    block, audit = attach_text(delta, propagated, synth, labels, encoder, cfg)
    audit.update(allocation=allocation, construction_seed=seed,
                 initialization="randomChoice with replacement, then optimized synthetic X",
                 optimizer_steps=cfg.cat_steps, hidden=cfg.cat_hidden, lr=cfg.cat_lr,
                 mmd_class_weighting="equal sum as official cgm.py (paper Eq.8 has ratio weights)",
                 fixed_bank_mmd_initial=before, fixed_bank_mmd_final=after,
                 random_encoder_trace=trace,
                 x_update_norm=float((synth.detach() - initial).norm()),
                 native_topology="self-loop only; implicit loops supplied by Student")
    return block, audit


def gecc_block(delta: dict, rows: int, encoder, cfg: Config,
               device: str) -> tuple[dict, dict]:
    """GECC hard-clustering variant, restricted to current delta access.

    Disjoint class increments contain no new data for an old class.  Historical
    centroids therefore persist without a raw refresh.  This does not reproduce
    GECC's cumulative-raw evolving experiment or its tuned fuzzy variant.
    """
    allocation = ratio_allocation(delta["y"], rows)
    propagated = propagated_features(delta, cfg.gecc_weights, device).cpu()
    centers, records = [], []
    for cls, k in allocation.items():
        local = torch.where(delta["y"] == cls)[0]
        values = propagated[local].numpy()
        best = None
        for rep in range(cfg.gecc_restarts):
            seed = cfg.seed + 1009 * delta["stage"] + rep
            km = KMeans(n_clusters=k, init="random", n_init=1, max_iter=300,
                        tol=0.0, random_state=seed).fit(values)
            sizes = np.bincount(km.labels_, minlength=k)
            balance = float(np.sum((sizes - len(values) / k) ** 2))
            objective = float(km.inertia_) + cfg.gecc_balance * balance
            if best is None or objective < best[0]:
                best = (objective, km, rep, balance)
        objective, km, rep, balance = best
        centers.append(torch.from_numpy(km.cluster_centers_))
        records.append({"class": cls, "rows": k, "restart": rep,
                        "sse": float(km.inertia_), "balance_penalty": balance,
                        "objective": objective})
    x = torch.cat(centers)
    block, audit = attach_text(delta, propagated, x, row_labels(allocation), encoder, cfg)
    audit.update(allocation=allocation, clustering=records,
                 propagated_weights=list(cfg.gecc_weights),
                 restarts=cfg.gecc_restarts, implementation="sklearn hard Lloyd + balanced restart score",
                 old_centroids="carried in memory; no old-class raw or same-class new samples",
                 cumulative_raw_reclustering=False, native_topology="self-loop only")
    return block, audit


def append_edge_free(parent: dict | None, block: dict) -> dict:
    if block["edge_index"].numel() or (parent and parent["edge_index"].numel()):
        raise ValueError("edge-free method contains nonself topology")
    if parent is None:
        return block
    out = {key: torch.cat((parent[key], block[key]))
           for key in ("x", "text_emb", "mass", "depth", "stage")}
    out["texts"] = list(parent["texts"]) + list(block["texts"])
    out["edge_index"] = torch.empty((2, 0), dtype=torch.long)
    return out
