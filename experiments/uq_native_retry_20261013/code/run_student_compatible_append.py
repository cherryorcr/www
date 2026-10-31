"""Student-Compatible Append for continual GraphCLIP TAG condensation.

The historical synthetic memory is never reconstructed from raw history.  At
each stage, several deterministic compact blocks are built from the current
 delta only.  Candidates are screened with independent seeds and batch sizes,
 using a multi-step student response curve, current-class plasticity,
 historical-class retention, and cross-condition ranking stability.  The
 selected block is then evaluated with the canonical v3 student protocol.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve()
for candidate in (HERE.parent, HERE.parent / "frozen_v1_2026_09_09",
                  HERE.parent.parent / "frozen_v1_2026_09_09"):
    if (candidate / "run_gate1_continual.py").exists():
        sys.path.insert(0, str(candidate))
        break
v2_dir = HERE.parent / "v2_llm_fusion"
if v2_dir.exists():
    sys.path.insert(1, str(v2_dir))
import protocol_v3
import run_gate1_continual as G
try:
    from run_multiscale_asm import maxmin_lloyd_gpu
except ImportError:
    maxmin_lloyd_gpu = None


RECIPE_SPECS = {
    "joint_g3_c192": {"space": "joint", "g": 3, "chars": 192},
    "joint_g5_c320": {"space": "joint", "g": 5, "chars": 320},
    "joint_g8_c512": {"space": "joint", "g": 8, "chars": 512},
    "xonly_g5_c320": {"space": "xonly", "g": 5, "chars": 320},
}


def text_audit(texts: list[str], tokenizer=None, operator: str = "concat",
               recipe: str = "joint_g5_c320") -> dict:
    """Record comparable text information for every final memory artifact."""
    lengths = [len(str(v)) for v in texts]
    token_total = None
    if tokenizer is not None and texts:
        encoded = tokenizer([str(v) for v in texts], add_special_tokens=False,
                            padding=False, truncation=False)
        token_total = int(sum(len(ids) for ids in encoded["input_ids"]))
    return {
        "operator": str(operator),
        "recipe": str(recipe),
        "rows": int(len(texts)),
        "char_total": int(sum(lengths)),
        "char_mean": float(np.mean(lengths)) if lengths else 0.0,
        "char_max": int(max(lengths)) if lengths else 0,
        "token_total": token_total,
        "token_mean": (float(token_total / len(texts))
                        if token_total is not None and texts else None),
        "empty_rows": int(sum(length == 0 for length in lengths)),
    }


def _bounded_concat(parts: list[str], chars: int) -> str:
    """Join snippets under an exact character budget."""
    budget = max(0, int(chars))
    if budget == 0 or not parts:
        return ""
    text = " ".join(str(v) for v in parts)
    return text[:budget]


def _select_text_members(joint: torch.Tensor, members: torch.Tensor,
                         source_text_emb: torch.Tensor, g: int,
                         operator: str, diversity_lambda: float) -> torch.Tensor:
    """Select a bounded text set deterministically, optionally with MMR."""
    count = max(1, min(int(g), int(len(members))))
    center = joint[members].mean(0, keepdim=True)
    center_dist = torch.cdist(joint[members], center).flatten()
    if operator not in {"diverse_concat", "dual_proto"}:
        return members[torch.argsort(center_dist, stable=True)[:count]]

    text_points = F.normalize(source_text_emb[members].float(), dim=1)
    text_center = F.normalize(text_points.mean(0, keepdim=True), dim=1)
    remaining = list(range(len(members)))
    first = int(torch.argmin(center_dist).item())
    selected = [first]
    remaining.remove(first)
    lam = max(0.0, min(1.0, float(diversity_lambda)))
    while remaining and len(selected) < count:
        best = max(remaining, key=lambda local: (
            lam * float((text_points[local] * text_center[0]).sum())
            - (1.0 - lam) * max(float((text_points[local] *
                                      text_points[j]).sum())
                                 for j in selected),
            -int(local)))
        selected.append(best)
        remaining.remove(best)
    return members[torch.tensor(selected, dtype=torch.long)]


def delta_difficulty(x: torch.Tensor, text_emb: torch.Tensor) -> float:
    """Estimate incoming-delta difficulty without labels or test data."""
    parts = []
    for value in (x.float(), text_emb.float()):
        z = F.normalize(value, dim=1)
        center = F.normalize(z.mean(0, keepdim=True), dim=1)
        parts.append(float((1.0 - (z * center).sum(1)).mean().clamp(0.0, 1.0)))
    return float(max(0.0, min(1.0, sum(parts) / max(1, len(parts)))))


def text_knn_edges(text_emb: torch.Tensor, degree: float = protocol_v3.DEGREE) -> torch.Tensor:
    """Build the deterministic text topology used by a standalone block."""
    k = len(text_emb)
    pairs = torch.triu_indices(k, k, offset=1)
    if pairs.shape[1] == 0:
        return torch.empty((2, 0), dtype=torch.long)
    normalized = F.normalize(text_emb.float(), dim=1)
    score = (normalized[pairs[0]] * normalized[pairs[1]]).sum(1)
    count = min(max(0, int(k * degree)), pairs.shape[1])
    if count == 0:
        return torch.empty((2, 0), dtype=torch.long)
    chosen = pairs[:, torch.topk(score, count, sorted=True).indices]
    return torch.cat((chosen, chosen.flip(0)), 1).contiguous()


def memory_edges(memory: dict, device: str | None = None) -> torch.Tensor:
    """Read an artifact topology, rebuilding legacy artifacts when needed."""
    edges = memory.get("edge_index")
    if edges is None:
        return G.text_topology(memory["text_emb"], device=device)
    return edges.long().cpu().contiguous()


def append_topology(old_memory: dict, block: dict, cross_degree: int = 1,
                    device: str | None = None) -> torch.Tensor:
    """Preserve old-old edges and add deterministic new/cross edges."""
    old_n, new_n = len(old_memory["x"]), len(block["x"])
    old_edges = memory_edges(old_memory, device=device)
    new_edges = memory_edges(block, device=device)
    if new_edges.numel():
        new_edges = new_edges + old_n
    parts = [old_edges, new_edges]
    if old_n and new_n and cross_degree > 0:
        old_t = F.normalize(old_memory["text_emb"].float(), dim=1)
        new_t = F.normalize(block["text_emb"].float(), dim=1)
        sim = new_t @ old_t.T
        width = min(int(cross_degree), old_n)
        nearest = torch.topk(sim, width, dim=1, sorted=True).indices
        new_ids = torch.arange(new_n).unsqueeze(1).expand_as(nearest)
        cross = torch.stack((nearest, old_n + new_ids), dim=0).reshape(2, -1)
        parts.extend((cross, cross.flip(0)))
    return torch.cat(parts, dim=1).long().contiguous()


def global_topology(memory: dict, device: str | None = None) -> torch.Tensor:
    """Rebuild all edges from the current artifact for the legacy control."""
    return G.text_topology(memory["text_emb"], device=device)


def configure_dataset(dataset: str) -> tuple[str, str]:
    if dataset == "cora":
        return ("/home/pc/research/GraphCLIP/processed_data/cora.pt",
                "/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip")
    if dataset == "citeseer":
        G.CLASS_NAMES = ["agents", "machine learning", "information retrieval",
                         "databases", "human computer interaction",
                         "artificial intelligence"]
        G.PROMPTS = [f"a research paper about {name}" for name in G.CLASS_NAMES]
        G.STAGE_CLASSES = [[0, 1], [2, 3], [4, 5]]
        return ("/home/pc/research/GraphCLIP/processed_data/citeseer.pt",
                "/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip")
    raise ValueError(f"unsupported dataset: {dataset}")


def encode_cluster_text(joint: torch.Tensor, members: torch.Tensor,
                        texts: list[str], encoder, operator: str,
                        chars: int, g: int,
                        source_text_emb: torch.Tensor | None = None,
                        diversity_lambda: float = 0.5,
                        dual_alpha: float = 0.5) -> tuple[str, torch.Tensor]:
    """Construct a cluster text prototype with an explicit operator."""
    if source_text_emb is None:
        source_text_emb = encoder.encode(texts)
    if operator == "concat":
        # Keep the v4 default compatible; strict token controls live in P2.
        fused = G.fuse_new_only(joint, members, texts, budget_chars=chars, g=g)
        return fused, encoder.encode([fused])[0]
    center = joint[members].mean(0, keepdim=True)
    distance = torch.cdist(joint[members], center).flatten()
    chosen = _select_text_members(
        joint, members, source_text_emb, g, operator, diversity_lambda)
    # Allocate the fixed budget across selected snippets and cap once more
    # after joining so all operators expose the same character budget.
    per = max(1, int(chars) // max(1, len(chosen)))
    snippets = [texts[int(i)][:per] for i in chosen]
    fused = _bounded_concat(snippets, chars)
    if operator in {"concat", "diverse_concat"}:
        return fused, encoder.encode([fused])[0]
    if operator == "no_reencode":
        return fused, F.normalize(source_text_emb[chosen].mean(0), dim=0)
    if operator == "no_fuse":
        text = texts[int(chosen[0])][:max(1, int(chars))]
        return text, encoder.encode([text])[0]
    if operator == "no_text":
        return "", torch.zeros_like(source_text_emb[0])
    chosen_emb = source_text_emb[chosen].float()
    chosen_local = torch.tensor(
        [int(torch.where(members == value)[0][0]) for value in chosen],
        dtype=torch.long)
    chosen_dist = distance[chosen_local]
    if operator == "mean_pool":
        weights = torch.full((len(chosen_emb),),
                             1.0 / max(1, len(chosen_emb)))
        pooled = (chosen_emb * weights[:, None]).sum(0)
    elif operator in {"weighted_pool", "set_pool"}:
        logits = -chosen_dist
        if operator == "set_pool":
            logits = logits - 0.5 * chosen_dist.square()
        weights = torch.softmax(logits, dim=0)
        pooled = (chosen_emb * weights[:, None]).sum(0)
    elif operator == "dual_proto":
        center_text = texts[int(chosen[0])][:max(1, int(chars) // 2)]
        diverse = _bounded_concat(
            [texts[int(i)] for i in chosen[1:]], max(1, int(chars) // 2))
        center_emb = encoder.encode([center_text])[0]
        diverse_emb = (F.normalize(chosen_emb[1:].mean(0), dim=0)
                       if len(chosen_emb) > 1 else center_emb)
        alpha = max(0.0, min(1.0, float(dual_alpha)))
        pooled = alpha * center_emb + (1.0 - alpha) * diverse_emb
    else:
        raise ValueError(f"unknown text operator: {operator}")
    return fused, F.normalize(pooled, dim=0)


def cluster_assign(points: torch.Tensor, mass: torch.Tensor, k: int,
                   device: str | None = None) -> torch.Tensor:
    """Use GPU chunked Lloyd for large blocks and exact Lloyd for small ones."""
    if len(points) >= getattr(G, "LARGE_CLUSTER_THRESHOLD", 20000):
        if maxmin_lloyd_gpu is None:
            raise RuntimeError("large-block clustering requires run_multiscale_asm")
        run_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        return maxmin_lloyd_gpu(points, int(k), run_device, iters=5, chunk=8192)
    return G.weighted_cluster(points, mass, int(k))


X_OPERATORS = ("mean", "weighted", "moment")


def synthesize_x(x: torch.Tensor, joint: torch.Tensor, members: torch.Tensor,
                 operator: str = "mean") -> torch.Tensor:
    """Construct one synthetic feature without copying a source node.

    The P1-X comparison keeps clustering, text and topology fixed and varies
    only this function.  ``weighted`` is a bounded centrality-weighted convex
    combination.  ``moment`` adds a deterministic, scale-controlled offset in
    the dominant within-cluster feature direction to counter variance collapse
    caused by a plain mean.  Both operators fall back to Mean-X for singleton
    or degenerate clusters.
    """
    if operator not in X_OPERATORS:
        raise ValueError(f"unknown x operator: {operator}")
    values = x[members].float()
    mean = values.mean(0)
    if len(values) <= 1 or operator == "mean":
        return mean
    if operator == "weighted":
        center = joint[members].float().mean(0, keepdim=True)
        distance = (joint[members].float() - center).pow(2).sum(1)
        scale = distance.std(unbiased=False).clamp_min(1e-6)
        logits = -(distance - distance.mean()) / scale
        weights = torch.softmax(logits, dim=0)
        # Keep the operator from collapsing to a single source row while
        # retaining a deterministic preference for central representatives.
        uniform = torch.full_like(weights, 1.0 / len(weights))
        weights = 0.5 * weights + 0.5 * uniform
        return (values * weights[:, None]).sum(0)

    centered = values - mean
    variance = centered.pow(2).mean(0)
    if float(variance.max()) <= 1e-12:
        return mean
    # Power iteration avoids an expensive full covariance eigendecomposition
    # and is deterministic because the initial direction is fixed.
    direction = torch.zeros(values.shape[1], dtype=values.dtype,
                            device=values.device)
    direction[0] = 1.0
    for _ in range(8):
        direction = centered.T @ (centered @ direction)
        norm = direction.norm()
        if float(norm) <= 1e-12:
            return mean
        direction = direction / norm
    pivot = int(torch.argmax(direction.abs()).item())
    if float(direction[pivot]) < 0:
        direction = -direction
    principal_std = torch.sqrt((centered @ direction).pow(2).mean())
    # A half-standard-deviation offset preserves scale while adding only a
    # bounded amount of within-cluster spread.
    candidate = mean + 0.5 * principal_std * direction
    target_norm = values.norm(dim=1).mean()
    candidate_norm = candidate.norm()
    if float(candidate_norm) > 1e-12 and float(target_norm) > 1e-12:
        candidate = candidate * (target_norm / candidate_norm)
    return candidate


def build_compact_block(x: torch.Tensor, text_emb: torch.Tensor,
                        texts: list[str], k: int, encoder,
                        recipe_name: str, stage: int,
                        text_operator: str = "concat",
                        device: str | None = None,
                        diversity_lambda: float = 0.5,
                        dual_alpha: float = 0.5,
                        x_operator: str = "mean") -> dict:
    """Build one deterministic delta-only compact block."""
    spec = RECIPE_SPECS[recipe_name]
    joint = torch.cat((F.normalize(x.float(), dim=1), text_emb), dim=1)
    if spec["space"] == "joint":
        points = joint
    elif spec["space"] == "xonly":
        points = F.normalize(x.float(), dim=1)
    else:
        raise ValueError(f"unknown candidate space: {spec['space']}")

    assign = cluster_assign(points, torch.ones(len(x)), int(k), device=device)
    syn_x, fused, encoded_parts, mass = [], [], [], []
    for j in range(int(assign.max()) + 1):
        members = torch.where(assign == j)[0]
        syn_x.append(synthesize_x(x, joint, members, x_operator))
        text, emb = encode_cluster_text(joint, members, texts, encoder,
                                        text_operator, spec["chars"], spec["g"],
                                        source_text_emb=text_emb,
                                        diversity_lambda=diversity_lambda,
                                        dual_alpha=dual_alpha)
        fused.append(text)
        encoded_parts.append(emb)
        mass.append(float(len(members)))
    syn_x = torch.stack(syn_x)
    if text_operator == "concat":
        encoded = encoder.encode(fused)
    else:
        encoded = torch.stack(encoded_parts)
    return {
        "x": syn_x,
        "text_emb": encoded,
        "texts": fused,
        "mass": torch.tensor(mass),
        "depth": torch.zeros(len(syn_x), dtype=torch.long),
        "stage": torch.full((len(syn_x),), stage, dtype=torch.long),
        "x_operator": x_operator,
    }


def bounded_refresh(old_memory: dict, block: dict, ratio: float,
                    alpha: float) -> tuple[dict, dict]:
    """Refresh only low-mass historical rows while preserving old topology."""
    def clone_value(value):
        """Clone tensor/list metadata without assuming every value is a tensor."""
        if isinstance(value, torch.Tensor):
            return value.clone()
        if isinstance(value, list):
            return value.copy()
        if isinstance(value, dict):
            return {key: clone_value(item) for key, item in value.items()}
        if isinstance(value, tuple):
            return tuple(clone_value(item) for item in value)
        return value

    out = {key: clone_value(value) for key, value in old_memory.items()}
    count = min(int(round(len(out["x"]) * max(0.0, min(1.0, ratio)))),
               len(block["x"]))
    if count <= 0 or len(out["x"]) == 0:
        return out, {"refreshed": 0, "ratio": float(ratio), "alpha": float(alpha)}
    low = torch.argsort(out["mass"], stable=True)[:count]
    old_x = F.normalize(out["x"].float(), dim=1)
    new_x = F.normalize(block["x"].float(), dim=1)
    old_t = F.normalize(out["text_emb"].float(), dim=1)
    new_t = F.normalize(block["text_emb"].float(), dim=1)
    sim = 0.5 * (old_x[low] @ new_x.T) + 0.5 * (old_t[low] @ new_t.T)
    nearest = sim.argmax(1)
    a = max(0.0, min(1.0, float(alpha)))
    for row, src in zip(low.tolist(), nearest.tolist()):
        out["x"][row] = (1.0 - a) * out["x"][row] + a * block["x"][src]
        out["text_emb"][row] = F.normalize(
            (1.0 - a) * out["text_emb"][row] + a * block["text_emb"][src], dim=0)
        out["mass"][row] = out["mass"][row] + a * block["mass"][src]
        if a >= 0.5:
            out["texts"][row] = block["texts"][src]
    return out, {"refreshed": int(count), "ratio": float(ratio),
                 "alpha": float(alpha), "rows": [int(i) for i in low.tolist()]}


def append_memory(old_memory: dict | None, block: dict, keep: int,
                  topology_mode: str = "frozen_old",
                  cross_degree: int = 1,
                  refresh_mode: str = "frozen_old",
                  refresh_ratio: float = 0.10,
                  refresh_alpha: float = 0.25,
                  device: str | None = None) -> dict:
    if old_memory is None:
        out = dict(block)
        out["edge_index"] = memory_edges(block, device=device)
        return out
    old = old_memory if int(keep) == len(old_memory["x"]) else G.shrink_by_mass(old_memory, int(keep))
    refresh_meta = {"refreshed": 0, "ratio": 0.0, "alpha": 0.0}
    if refresh_mode == "bounded_refresh":
        old, refresh_meta = bounded_refresh(old, block, refresh_ratio, refresh_alpha)
    elif refresh_mode not in {"frozen_old", "rescore_only"}:
        raise ValueError(f"unknown refresh mode: {refresh_mode}")
    out = {
        key: (old[key] + block[key] if key == "texts"
              else torch.cat((old[key], block[key])))
        for key in ("x", "text_emb", "texts", "mass", "depth", "stage")
    }
    if topology_mode == "frozen_old":
        out["edge_index"] = append_topology(old, block, cross_degree, device=device)
    elif topology_mode == "global":
        out["edge_index"] = global_topology(out, device=device)
    else:
        raise ValueError(f"unknown topology mode: {topology_mode}")
    out["refresh_metadata"] = refresh_meta
    return out


def _summary(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "n": int(arr.size),
    }


def evaluate_grid(memory: dict, stage: int, seeds: list[int], batches: list[int],
                  steps: int, device: str, xg: torch.Tensor,
                  real_adj: torch.Tensor, prompt_emb: torch.Tensor,
                  y: torch.Tensor, dev_mask: torch.Tensor) -> dict:
    """Evaluate one artifact over a seed x batch short-retraining grid."""
    syn = G.memory_to_trainable(memory)
    rows = []
    for batch in batches:
        for seed in seeds:
            model = G.train_student(syn, seed, device, steps, batch=batch)
            score = G.eval_stagewise(model, xg, real_adj, prompt_emb, y,
                                     dev_mask, stage)
            rows.append({
                "seed": int(seed),
                "batch": int(batch),
                "current_f1": float(score["current_f1"]),
                "old_mean_f1": (float(score["old_mean_f1"])
                                 if stage > 0 else None),
                "cumulative_f1": float(score["cumulative_macro_f1"]),
                "per_stage": score["per_stage"],
            })
            del model

    current = [r["current_f1"] for r in rows]
    cumulative = [r["cumulative_f1"] for r in rows]
    old = [r["old_mean_f1"] for r in rows if r["old_mean_f1"] is not None]
    by_batch = {}
    for batch in batches:
        subset = [r["current_f1"] for r in rows if r["batch"] == batch]
        by_batch[str(batch)] = _summary(subset)
    return {
        "steps": int(steps),
        "seeds": list(seeds),
        "batches": list(batches),
        "current_f1": _summary(current),
        "cumulative_f1": _summary(cumulative),
        "old_mean_f1": _summary(old) if old else None,
        "current_by_batch": by_batch,
        "per_student": rows,
    }


def evaluate_response_curve(memory: dict, stage: int, seeds: list[int],
                            batches: list[int], steps_curve: list[int],
                            device: str, xg: torch.Tensor,
                            real_adj: torch.Tensor, prompt_emb: torch.Tensor,
                            y: torch.Tensor, dev_mask: torch.Tensor) -> dict:
    """Evaluate a candidate at several training horizons."""
    curve = []
    for steps in steps_curve:
        grid = evaluate_grid(memory, stage, seeds, batches, int(steps), device,
                             xg, real_adj, prompt_emb, y, dev_mask)
        curve.append(grid)
    means = np.asarray([row["current_f1"]["mean"] for row in curve], dtype=float)
    cumulative_means = np.asarray(
        [row["cumulative_f1"]["mean"] for row in curve], dtype=float)
    if len(curve) == 1:
        auc = float(means[0])
        cumulative_auc = float(cumulative_means[0])
    else:
        xx = np.asarray(steps_curve, dtype=float)
        span = max(float(xx[-1] - xx[0]), 1.0)
        trapezoid = getattr(np, "trapezoid", None)
        if trapezoid is not None:
            auc = float(trapezoid(means, xx) / span)
            cumulative_auc = float(trapezoid(cumulative_means, xx) / span)
        else:
            auc = float(np.sum((means[:-1] + means[1:])
                               * 0.5 * np.diff(xx)) / span)
            cumulative_auc = float(np.sum((cumulative_means[:-1]
                                           + cumulative_means[1:])
                                          * 0.5 * np.diff(xx)) / span)
    tail_start = max(0, len(curve) - max(2, len(curve) // 2))
    tail = curve[tail_start:]
    old_values = [r["old_mean_f1"]["mean"] for r in tail
                  if r["old_mean_f1"] is not None]
    return {
        "steps": [int(s) for s in steps_curve],
        "grids": curve,
        "response_auc": auc,
        "cumulative_auc": cumulative_auc,
        "tail_mean": float(np.mean([r["current_f1"]["mean"] for r in tail])),
        "tail_std": float(np.mean([r["current_f1"]["std"] for r in tail])),
        "cumulative_tail_mean": float(np.mean(
            [r["cumulative_f1"]["mean"] for r in tail])),
        "cumulative_tail_std": float(np.mean(
            [r["cumulative_f1"]["std"] for r in tail])),
        "old_tail_mean": (float(np.mean(old_values)) if old_values else None),
    }


def add_rank_consistency(reports: list[dict], steps_curve: list[int],
                         batches: list[int]) -> None:
    """Add condition-wise relative ranking diagnostics to each report."""
    names = [r["candidate"] for r in reports]
    conditions = []
    for step_index, _ in enumerate(steps_curve):
        for batch in batches:
            scores = {}
            for report in reports:
                grid = report["response"]["grids"][step_index]
                scores[report["candidate"]] = grid["current_by_batch"][str(batch)]["mean"]
            ranking = sorted(names, key=lambda name: (scores[name], name), reverse=True)
            conditions.append({name: rank + 1 for rank, name in enumerate(ranking)})

    pooled = sorted(reports,
                    key=lambda r: (r["response"]["tail_mean"],
                                   r["response"]["response_auc"], r["candidate"]),
                    reverse=True)
    pooled_rank = {r["candidate"]: i + 1 for i, r in enumerate(pooled)}
    max_rank_span = max(1, len(names) - 1)
    for report in reports:
        name = report["candidate"]
        ranks = np.asarray([c[name] for c in conditions], dtype=float)
        rank_std = float(ranks.std())
        rank_stability = max(0.0, 1.0 - rank_std / max_rank_span)
        agreements = []
        for other in names:
            if other == name:
                continue
            pooled_sign = np.sign(pooled_rank[other] - pooled_rank[name])
            for condition in conditions:
                condition_sign = np.sign(condition[other] - condition[name])
                if pooled_sign != 0 and condition_sign != 0:
                    agreements.append(float(pooled_sign == condition_sign))
        pairwise_agreement = float(np.mean(agreements)) if agreements else 1.0
        consistency = 0.5 * rank_stability + 0.5 * pairwise_agreement
        report["rank_consistency"] = {
            "mean_rank": float(ranks.mean()),
            "rank_std": rank_std,
            "rank_stability": float(rank_stability),
            "pairwise_agreement": pairwise_agreement,
            "score": float(consistency),
            "pooled_rank": int(pooled_rank[name]),
            "conditions": len(conditions),
        }


def dominates(a: dict, b: dict) -> bool:
    """Three-objective dominance: new F1 and retention up, instability down."""
    no_worse = (
        a["new_mean"] >= b["new_mean"]
        and a["old_delta"] >= b["old_delta"]
        and a["new_std"] <= b["new_std"]
    )
    strictly = (
        a["new_mean"] > b["new_mean"]
        or a["old_delta"] > b["old_delta"]
        or a["new_std"] < b["new_std"]
    )
    return bool(no_worse and strictly)


def pareto_front(rows: list[dict]) -> list[str]:
    return [row["candidate"] for row in rows
            if not any(dominates(other, row)
                       for other in rows if other is not row)]


def resolve_selection_curve(stage: int, total_budget: int, args) -> list[int]:
    """Resolve a stage-aware response curve under the canonical budget."""
    final_steps = protocol_v3.student_steps(total_budget)
    if args.selection_steps is not None:
        requested = args.selection_steps
    else:
        bases = args.selection_base_by_stage
        if bases is None:
            base = max(128, final_steps // 4)
        elif len(bases) == 1:
            base = int(bases[0])
        elif len(bases) == len(G.STAGE_CLASSES):
            base = int(bases[stage])
        else:
            raise ValueError("--selection-base-by-stage expects one value or one per stage")
        requested = [int(round(base * multiplier))
                     for multiplier in args.selection_curve_multipliers]
    steps = [max(50, min(final_steps, int(step))) for step in requested]
    return sorted(set(steps))


def select_diverse_beam(expansions: list[dict], width: int) -> list[dict]:
    """Keep the best child from each parent before filling the beam globally."""
    ranked = sorted(expansions,
                    key=lambda item: (item["score"], item["path"]),
                    reverse=True)
    mandatory = []
    for parent_index in sorted({item["parent_index"] for item in ranked}):
        group = [item for item in ranked
                 if item["parent_index"] == parent_index]
        mandatory.append(group[0])
    chosen = {id(item) for item in mandatory}
    remainder = [item for item in ranked if id(item) not in chosen]
    selected = mandatory + remainder
    return sorted(selected[:width],
                  key=lambda item: (item["score"], item["path"]),
                  reverse=True)


def run_path_beam(args, stage_train, budgets, x, y, texts, pool,
                  pool_text_emb, encoder, prompt_emb, real_adj, xg,
                  dev_mask, device, out) -> list[dict]:
    """Search over several complete memory trajectories instead of one greedy path."""
    beams = [{"memory": None, "path": [], "score": 0.0}]
    results = []
    width = max(2, int(args.path_beam_width))

    for stage, ids in enumerate(stage_train):
        k_t = budgets[stage]
        local = torch.searchsorted(pool, ids)
        dx, de = x[ids], pool_text_emb[local]
        dt = [texts[int(i)] for i in ids]
        expansions = []
        parent_groups = []
        for parent_index, parent in enumerate(beams):
            old_memory = parent["memory"]
            if stage == 0:
                keep, k_new = 0, k_t
                old_for_stage = None
            else:
                difficulty = delta_difficulty(dx, de)
                quota = (protocol_v3.dynamic_append_quota(
                    k_t, len(old_memory["x"]), difficulty,
                    len(G.STAGE_CLASSES[stage]))
                    if args.dynamic_rows else protocol_v3.append_quota(
                        k_t, len(old_memory["x"]), len(G.STAGE_CLASSES[stage])))
                keep, k_new = quota
                old_for_stage = G.shrink_by_mass(old_memory, keep)
                if len(old_for_stage["x"]) != keep:
                    raise RuntimeError("beam append quota did not preserve old rows")

            selection_curve = resolve_selection_curve(stage, k_t, args)
            if old_for_stage is not None:
                old_reference = evaluate_response_curve(
                    old_for_stage, stage, args.selection_seeds,
                    args.selection_batches, selection_curve, device, xg,
                    real_adj, prompt_emb, y, dev_mask)
                old_reference_mean = old_reference["old_tail_mean"]
            else:
                old_reference = None
                old_reference_mean = 0.0

            candidate_reports = []
            candidate_memories = {}
            for recipe in args.candidate_recipes:
                block = build_compact_block(dx, de, dt, k_new, encoder,
                                            recipe, stage,
                                            text_operator=args.text_operator)
                memory = append_memory(
                    old_memory, block, keep,
                    topology_mode=args.topology_mode,
                    cross_degree=args.cross_degree,
                    refresh_mode=args.refresh_mode,
                    refresh_ratio=args.refresh_ratio,
                    refresh_alpha=args.refresh_alpha)
                candidate_memories[recipe] = memory
                response = evaluate_response_curve(
                    memory, stage, args.selection_seeds, args.selection_batches,
                    selection_curve, device, xg, real_adj, prompt_emb, y,
                    dev_mask)
                new_mean = response["tail_mean"]
                new_std = response["tail_std"]
                old_mean = response["old_tail_mean"]
                old_delta = 0.0 if old_mean is None else old_mean - old_reference_mean
                candidate_reports.append({
                    "candidate": recipe,
                    "spec": RECIPE_SPECS[recipe],
                    "roots": len(memory["x"]),
                    "keep_old_rows": int(keep),
                    "new_rows": int(k_new),
                    "new_mean": float(new_mean),
                    "new_std": float(new_std),
                    "old_mean": None if old_mean is None else float(old_mean),
                    "old_delta": float(old_delta),
                    "response": response,
                })

            add_rank_consistency(candidate_reports, selection_curve,
                                 args.selection_batches)
            for report in candidate_reports:
                consistency = report["rank_consistency"]["score"]
                report["rank_consistency_bonus"] = float(
                    args.rank_consistency_weight * 5.0 * consistency)
                report["robust_utility"] = float(
                    0.5 * report["new_mean"]
                    + 0.5 * report["response"]["response_auc"]
                    - 0.5 * report["new_std"]
                    + 0.5 * report["old_delta"]
                    + report["rank_consistency_bonus"])
            front = pareto_front(candidate_reports)
            eligible = [r for r in candidate_reports
                        if r["old_delta"] >= -args.retention_tol
                        and r["rank_consistency"]["score"] >= args.min_rank_consistency]
            if not eligible:
                eligible = [r for r in candidate_reports
                            if r["old_delta"] >= -args.retention_tol]
            eligible_front = [r for r in eligible if r["candidate"] in front]
            choice_pool = (eligible_front
                           or [r for r in candidate_reports if r["candidate"] in front]
                           or candidate_reports)
            ranked = sorted(
                choice_pool,
                key=lambda r: (r["robust_utility"], r["new_mean"],
                               -r["new_std"], r["candidate"]),
                reverse=True,
            )
            # Greedy selection can collapse a Pareto front to one item.  A
            # beam must still retain alternatives for path-level recovery.
            ranked_names = {r["candidate"] for r in ranked}
            extras = sorted(
                [r for r in candidate_reports
                 if r["candidate"] not in ranked_names
                 and r["old_delta"] >= -args.retention_tol],
                key=lambda r: (r["robust_utility"], r["new_mean"],
                               -r["new_std"], r["candidate"]),
                reverse=True,
            )
            ranked.extend(extras)
            # Keep at least two local alternatives so a close early decision
            # cannot eliminate every trajectory that may recover later.  On
            # the terminal stage, optional lookahead evaluates every local
            # candidate at full selection budget before beam pruning.
            if (stage == len(stage_train) - 1
                    and args.terminal_lookahead_all):
                local_keep = len(ranked)
            else:
                local_keep = min(len(ranked), max(2, width))
            parent_groups.append({
                "parent_index": int(parent_index),
                "parent_path": list(parent["path"]),
                "parent_score": float(parent["score"]),
                "selection_steps": [int(s) for s in selection_curve],
                "old_reference": old_reference,
                "pareto_front": front,
                "eligible_pareto_front": [r["candidate"] for r in eligible_front],
                "candidates": candidate_reports,
            })
            for rank, report in enumerate(ranked[:local_keep], start=1):
                recipe = report["candidate"]
                memory = candidate_memories[recipe]
                lineage = "_".join(parent["path"] + [recipe]) or recipe
                torch.save(memory, out / f"beam_s{stage}_p{parent_index}_{rank}_{lineage}_memory.pt")
                expansions.append({
                    "memory": memory,
                    "path": parent["path"] + [recipe],
                    "score": float(parent["score"] + report["robust_utility"]),
                    "parent_index": int(parent_index),
                    "local_rank": int(rank),
                    "report": report,
                })

        beams = select_diverse_beam(expansions, width)
        terminal_confirmation = None
        if stage == len(stage_train) - 1:
            terminal_rows = []
            terminal_pool = (expansions if args.terminal_lookahead_all
                             else beams)
            for item in terminal_pool:
                terminal_steps = protocol_v3.student_steps(
                    len(item["memory"]["x"]), override=args.student_steps)
                terminal_grid = evaluate_grid(
                    item["memory"], stage, args.selection_seeds,
                    [args.student_batch], terminal_steps, device, xg,
                    real_adj, prompt_emb, y, dev_mask)
                terminal_mean = terminal_grid["cumulative_f1"]["mean"]
                terminal_std = terminal_grid["cumulative_f1"]["std"]
                item["terminal_grid"] = terminal_grid
                item["terminal_score"] = float(terminal_mean - 0.25 * terminal_std)
                terminal_rows.append({
                    "path": item["path"],
                    "beam_score": float(item["score"]),
                    "terminal_score": float(item["terminal_score"]),
                    "terminal_cumulative_f1": float(terminal_mean),
                    "terminal_cumulative_std": float(terminal_std),
                })
            beams = sorted(
                terminal_pool,
                key=lambda item: (item["terminal_score"], item["score"], item["path"]),
                reverse=True,
            )[:width]
            terminal_confirmation = terminal_rows
        selected = beams[0]
        selected_memory = selected["memory"]
        torch.save(selected_memory, out / f"student_compatible_append_s{stage}_memory.pt")
        final_steps = protocol_v3.student_steps(
            len(selected_memory["x"]), override=args.student_steps)
        final_grid = evaluate_grid(
            selected_memory, stage, args.student_seeds, [args.student_batch],
            final_steps, device, xg, real_adj, prompt_emb, y, dev_mask)
        row = {
            "stage": stage,
            "selected_candidate": selected["path"][-1],
            "selected_path": selected["path"],
            "path_score": selected["score"],
            "path_beam_width": width,
            "text_operator": args.text_operator,
            "refresh_mode": args.refresh_mode,
            "refresh_ratio": float(args.refresh_ratio),
            "refresh_alpha": float(args.refresh_alpha),
            "dynamic_rows": bool(args.dynamic_rows),
            "beam_paths": [
                {"path": item["path"], "score": item["score"],
                 **({"terminal_score": item["terminal_score"],
                    "terminal_cumulative_f1": item["terminal_grid"]["cumulative_f1"]["mean"]}
                    if "terminal_score" in item else {})}
                for item in beams
            ],
            "terminal_confirmation": terminal_confirmation,
            "beam_expansions": len(expansions),
            "parent_groups": parent_groups,
            "final": final_grid,
        }
        results.append(row)
        print(json.dumps({
            "stage": stage,
            "selected": selected["path"][-1],
            "path": selected["path"],
            "beam_paths": [{"path": b["path"], "score": b["score"],
                            **({"terminal_score": b["terminal_score"],
                                "terminal_cumulative_f1": b["terminal_grid"]["cumulative_f1"]["mean"]}
                               if "terminal_score" in b else {})}
                           for b in beams],
            "final_cumulative_f1": final_grid["cumulative_f1"]["mean"],
        }), flush=True)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["cora", "citeseer"], required=True)
    default_data, default_minilm = configure_dataset("cora")
    parser.add_argument("--data", default=None)
    parser.add_argument("--minilm", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rate", type=float, default=0.05)
    parser.add_argument("--candidate-recipes", nargs="+",
                        default=list(RECIPE_SPECS))
    parser.add_argument("--selection-seeds", type=int, nargs="+",
                        default=[200, 201, 202])
    parser.add_argument("--selection-batches", type=int, nargs="+",
                        default=[8, 16, 32])
    parser.add_argument("--selection-steps", type=int, nargs="+", default=None,
                        help="explicit response-curve steps; one or more values")
    parser.add_argument("--selection-base-by-stage", type=int, nargs="+",
                        default=None,
                        help="stage-specific curve base, or one shared base")
    parser.add_argument("--selection-curve-multipliers", type=float, nargs="+",
                        default=[0.5, 1.0, 2.0])
    parser.add_argument("--retention-tol", type=float, default=2.0)
    parser.add_argument("--min-rank-consistency", type=float, default=0.5)
    parser.add_argument("--rank-consistency-weight", type=float, default=0.25)
    parser.add_argument("--topology-mode", choices=["frozen_old", "global"],
                        default="frozen_old")
    parser.add_argument("--cross-degree", type=int, default=1)
    parser.add_argument("--path-beam-width", type=int, default=1,
                        help="keep this many complete memory paths; >1 enables beam search")
    parser.add_argument("--terminal-lookahead-all", action="store_true",
                        help="at the final stage, full-step evaluate all local candidates before beam pruning")
    parser.add_argument("--text-operator", choices=["concat", "diverse_concat",
                                                     "dual_proto", "mean_pool",
                                                     "weighted_pool", "set_pool",
                                                     "no_reencode", "no_fuse", "no_text"],
                        default="concat")
    parser.add_argument("--refresh-mode", choices=["frozen_old", "rescore_only",
                                                     "bounded_refresh"],
                        default="frozen_old")
    parser.add_argument("--refresh-ratio", type=float, default=0.10)
    parser.add_argument("--refresh-alpha", type=float, default=0.25)
    parser.add_argument("--dynamic-rows", action="store_true",
                        help="allocate a bounded extra fraction of memory to difficult deltas")
    parser.add_argument("--student-seeds", type=int, nargs="+",
                        default=list(protocol_v3.STUDENT_SEEDS))
    parser.add_argument("--student-batch", type=int,
                        default=protocol_v3.STUDENT.batch)
    parser.add_argument("--student-steps", type=int, default=None)
    args = parser.parse_args()

    if args.data is None:
        args.data = default_data if args.dataset == "cora" else configure_dataset(args.dataset)[0]
    if args.minilm is None:
        args.minilm = default_minilm if args.dataset == "cora" else configure_dataset(args.dataset)[1]
    unknown = sorted(set(args.candidate_recipes) - set(RECIPE_SPECS))
    if unknown:
        raise ValueError(f"unknown candidate recipes: {unknown}")

    # Apply dataset-specific class names before loading the shared runner APIs.
    configure_dataset(args.dataset)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "results.json").exists():
        raise FileExistsError("existing runs are immutable")

    data = torch.load(args.data, map_location="cpu", weights_only=False)
    x, y, texts = data.x, data.y, list(data.raw_texts)
    n = len(x)
    dev_mask, _, pool_mask = protocol_v3.split_masks(torch.arange(n))
    pool = torch.where(pool_mask)[0]
    device = args.device
    encoder = G.FrozenMiniLM(args.minilm, device)
    pool_texts = [texts[int(i)] for i in pool]
    pool_text_emb = encoder.encode(pool_texts)
    prompt_emb = encoder.encode(G.PROMPTS)
    real_adj = G.norm_adj(data.edge_index, n, device)
    xg = x.float().to(device)

    stage_train = []
    for cls in G.STAGE_CLASSES:
        mask = torch.isin(y, torch.tensor(cls)) & pool_mask
        stage_train.append(torch.where(mask)[0])
    cum_counts = np.cumsum([len(s) for s in stage_train])
    budgets = [math.floor(args.rate * c) for c in cum_counts]
    print(json.dumps({"stage_train_counts": [len(s) for s in stage_train],
                      "budgets": budgets}), flush=True)

    if args.path_beam_width > 1:
        results = run_path_beam(
            args, stage_train, budgets, x, y, texts, pool, pool_text_emb,
            encoder, prompt_emb, real_adj, xg, dev_mask, device, out)
        json.dump({
            **protocol_v3.config_dict(
                dataset=args.dataset,
                rate=args.rate,
                student_batch=args.student_batch,
                student_steps=(args.student_steps if args.student_steps is not None
                                else "protocol-scaled"),
                student_seeds=args.student_seeds,
            ),
            "method": ("student_compatible_append_path_beam_terminal_lookahead"
                        if args.terminal_lookahead_all
                        else "student_compatible_append_path_beam"),
            "candidate_specs": RECIPE_SPECS,
            "candidate_recipes": args.candidate_recipes,
            "selection_seeds": args.selection_seeds,
            "selection_batches": args.selection_batches,
            "selection_steps": args.selection_steps,
            "selection_base_by_stage": args.selection_base_by_stage,
            "selection_curve_multipliers": args.selection_curve_multipliers,
            "retention_tolerance": args.retention_tol,
            "min_rank_consistency": args.min_rank_consistency,
            "rank_consistency_weight": args.rank_consistency_weight,
            "topology_mode": args.topology_mode,
            "cross_degree": args.cross_degree,
            "path_beam_width": int(args.path_beam_width),
            "terminal_lookahead_all": bool(args.terminal_lookahead_all),
            "stage_train_counts": [len(s) for s in stage_train],
            "budgets": budgets,
            "dev_rule": "original_id % 10 == 1",
            "test_rule": "original_id % 10 == 2",
            "test_accessed": False,
            "test_source_sealed": True,
            "old_memory_content_preserved": True,
            "results": results,
        }, open(out / "results.json", "w"), indent=1)
        print("DONE", flush=True)
        return

    selected_memory = None
    results = []
    for stage, ids in enumerate(stage_train):
        k_t = budgets[stage]
        local = torch.searchsorted(pool, ids)
        dx, de = x[ids], pool_text_emb[local]
        dt = [texts[int(i)] for i in ids]
        if stage == 0:
            keep, k_new = 0, k_t
            old_for_stage = None
        else:
            difficulty = delta_difficulty(dx, de)
            quota = (protocol_v3.dynamic_append_quota(
                k_t, len(selected_memory["x"]), difficulty,
                len(G.STAGE_CLASSES[stage]))
                if args.dynamic_rows else protocol_v3.append_quota(
                    k_t, len(selected_memory["x"]), len(G.STAGE_CLASSES[stage])))
            keep, k_new = quota
            old_for_stage = G.shrink_by_mass(selected_memory, keep)
            if len(old_for_stage["x"]) != keep:
                raise RuntimeError("append quota did not preserve the requested old-row count")

        selection_curve = resolve_selection_curve(stage, k_t, args)

        old_reference = None
        if old_for_stage is not None:
            old_reference_curve = evaluate_response_curve(
                old_for_stage, stage, args.selection_seeds,
                args.selection_batches, selection_curve, device, xg,
                real_adj, prompt_emb, y, dev_mask)
            old_reference = old_reference_curve
            old_reference_mean = old_reference_curve["old_tail_mean"]
        else:
            old_reference_mean = 0.0
            old_reference_curve = None

        candidate_reports = []
        candidate_memories = {}
        for recipe in args.candidate_recipes:
            block = build_compact_block(dx, de, dt, k_new, encoder, recipe, stage,
                                        text_operator=args.text_operator)
            memory = append_memory(selected_memory, block, keep,
                                   topology_mode=args.topology_mode,
                                   cross_degree=args.cross_degree,
                                   refresh_mode=args.refresh_mode,
                                   refresh_ratio=args.refresh_ratio,
                                   refresh_alpha=args.refresh_alpha)
            candidate_memories[recipe] = memory
            torch.save(memory, out / f"sca_s{stage}_{recipe}_memory.pt")
            response = evaluate_response_curve(
                memory, stage, args.selection_seeds, args.selection_batches,
                selection_curve, device, xg, real_adj, prompt_emb, y, dev_mask)
            new_mean = response["tail_mean"]
            new_std = response["tail_std"]
            old_mean = response["old_tail_mean"]
            old_delta = 0.0 if old_mean is None else old_mean - old_reference_mean
            report = {
                "candidate": recipe,
                "spec": RECIPE_SPECS[recipe],
                "roots": len(memory["x"]),
                "keep_old_rows": int(keep),
                "new_rows": int(k_new),
                "new_mean": float(new_mean),
                "new_std": float(new_std),
                "old_mean": None if old_mean is None else float(old_mean),
                "old_delta": float(old_delta),
                "response": response,
            }
            candidate_reports.append(report)

        add_rank_consistency(candidate_reports, selection_curve,
                             args.selection_batches)
        for report in candidate_reports:
            consistency = report["rank_consistency"]["score"]
            report["rank_consistency_bonus"] = float(
                args.rank_consistency_weight * 5.0 * consistency)
            report["robust_utility"] = float(
                0.5 * report["new_mean"]
                + 0.5 * report["response"]["response_auc"]
                - 0.5 * report["new_std"]
                + 0.5 * report["old_delta"]
                + report["rank_consistency_bonus"])
        front = pareto_front(candidate_reports)
        eligible = [r for r in candidate_reports
                    if r["old_delta"] >= -args.retention_tol
                    and r["rank_consistency"]["score"] >= args.min_rank_consistency]
        if not eligible:
            eligible = [r for r in candidate_reports
                        if r["old_delta"] >= -args.retention_tol]
        eligible_front = [r for r in eligible if r["candidate"] in front]
        pool_for_choice = eligible_front or [r for r in candidate_reports
                                             if r["candidate"] in front] or candidate_reports
        selected_report = sorted(
            pool_for_choice,
            key=lambda r: (r["robust_utility"], r["new_mean"],
                           -r["new_std"], r["candidate"]),
            reverse=True,
        )[0]
        selected_name = selected_report["candidate"]
        selected_memory = candidate_memories[selected_name]
        torch.save(selected_memory, out / f"student_compatible_append_s{stage}_memory.pt")

        final_steps = protocol_v3.student_steps(
            len(selected_memory["x"]), override=args.student_steps)
        final_grid = evaluate_grid(
            selected_memory, stage, args.student_seeds, [args.student_batch],
            final_steps, device, xg, real_adj, prompt_emb, y, dev_mask)
        row = {
            "stage": stage,
            "selected_candidate": selected_name,
            "selected_spec": RECIPE_SPECS[selected_name],
            "roots": len(selected_memory["x"]),
            "keep_old_rows": int(keep),
            "new_rows": int(k_new),
            "selection_steps": [int(s) for s in selection_curve],
            "selection_seeds": list(args.selection_seeds),
            "selection_batches": list(args.selection_batches),
            "retention_tolerance": float(args.retention_tol),
            "min_rank_consistency": float(args.min_rank_consistency),
            "rank_consistency_weight": float(args.rank_consistency_weight),
            "topology_mode": args.topology_mode,
            "cross_degree": int(args.cross_degree),
            "pareto_front": front,
            "eligible_pareto_front": [r["candidate"] for r in eligible_front],
            "old_reference": old_reference,
            "candidates": candidate_reports,
            "final": final_grid,
        }
        results.append(row)
        print(json.dumps({"stage": stage, "selected": selected_name,
                          "pareto": front,
                          "final_current_f1": final_grid["current_f1"]["mean"],
                          "final_cumulative_f1": final_grid["cumulative_f1"]["mean"]}),
              flush=True)

    json.dump({
        **protocol_v3.config_dict(
            dataset=args.dataset,
            rate=args.rate,
            student_batch=args.student_batch,
            student_steps=(args.student_steps if args.student_steps is not None
                            else "protocol-scaled"),
            student_seeds=args.student_seeds,
        ),
        "method": "student_compatible_append",
        "candidate_specs": RECIPE_SPECS,
        "candidate_recipes": args.candidate_recipes,
        "selection_seeds": args.selection_seeds,
        "selection_batches": args.selection_batches,
        "selection_steps": args.selection_steps,
        "selection_base_by_stage": args.selection_base_by_stage,
        "selection_curve_multipliers": args.selection_curve_multipliers,
        "retention_tolerance": args.retention_tol,
        "min_rank_consistency": args.min_rank_consistency,
        "rank_consistency_weight": args.rank_consistency_weight,
        "topology_mode": args.topology_mode,
        "cross_degree": args.cross_degree,
        "path_beam_width": int(args.path_beam_width),
        "text_operator": args.text_operator,
        "refresh_mode": args.refresh_mode,
        "refresh_ratio": float(args.refresh_ratio),
        "refresh_alpha": float(args.refresh_alpha),
        "dynamic_rows": bool(args.dynamic_rows),
        "stage_train_counts": [len(s) for s in stage_train],
        "budgets": budgets,
        "dev_rule": "original_id % 10 == 1",
        "test_rule": "original_id % 10 == 2",
        "test_accessed": False,
        "test_source_sealed": True,
        "old_memory_content_preserved": True,
        "results": results,
    }, open(out / "results.json", "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
