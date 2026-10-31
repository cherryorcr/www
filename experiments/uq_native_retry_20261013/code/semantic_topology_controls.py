"""Deterministic, equal-edge semantic controls; no labels or task feedback."""
from __future__ import annotations

import numpy as np


def validate_scores(scores):
    scores = np.asarray(scores)
    if scores.ndim != 2 or scores.shape[0] != scores.shape[1]:
        raise ValueError("expected square similarity matrix")
    if not np.isfinite(scores).all() or not np.allclose(scores, scores.T, atol=1e-6):
        raise ValueError("expected finite symmetric similarities")
    return scores


def pair_array(pairs):
    return np.asarray(sorted(pairs), dtype=np.int64).reshape(-1, 2).T.copy()


def budget_knn(scores):
    """Preserve symmetrized 1-NN; fill to n pairs from symmetrized 2-NN."""
    scores = validate_scores(scores)
    n = len(scores)
    target = min(n, n * (n - 1) // 2)
    if n < 2:
        return pair_array([]), dict(target=target, candidate_pairs=0, mandatory_pairs=0)
    ranked = scores.copy()
    np.fill_diagonal(ranked, -np.inf)
    neighbors = np.argsort(-ranked, axis=1, kind="stable")[:, :min(2, n - 1)]
    first = {tuple(sorted((i, int(row[0])))) for i, row in enumerate(neighbors)}
    candidates = {tuple(sorted((i, int(j)))) for i, row in enumerate(neighbors) for j in row}
    chosen = set(first)
    extra = sorted(candidates - first, key=lambda e: (-float(scores[e]), *e))
    chosen.update(extra[:target - len(chosen)])
    assert len(chosen) == target and len(chosen) >= len(first)
    return pair_array(chosen), dict(target=target, candidate_pairs=len(candidates), mandatory_pairs=len(first))


def local_cycles(scores):
    """Greedy cap-2 edges allowing early cycles; one residual insertion if needed.

    Unlike the single-ring rule, this never rejects an edge just because it
    closes a component. It is a heuristic cycle cover, not an optimal 2-factor.
    """
    scores = validate_scores(scores)
    n = len(scores)
    if n < 3:
        pairs, info = budget_knn(scores)
        return pairs, dict(info, repairs=0)
    u, v = np.triu_indices(n, 1)
    order = np.argsort(-scores[u, v], kind="stable")
    degree = np.zeros(n, dtype=np.int64)
    chosen = set()
    for at in order:
        a, b = int(u[at]), int(v[at])
        if degree[a] < 2 and degree[b] < 2:
            chosen.add((a, b))
            degree[a] += 1
            degree[b] += 1
            if len(chosen) == n:
                break
    residual = np.flatnonzero(degree < 2).tolist()
    repairs = 0
    if residual:
        # With all pairs considered, only an isolated vertex or an isolated
        # two-vertex path can remain. Insert it into the least costly cycle edge.
        assert len(chosen) == n - 1
        assert (len(residual) == 1 and degree[residual[0]] == 0 or
                len(residual) == 2 and np.all(degree[residual] == 1) and tuple(residual) in chosen)
        options = []
        for a, b in sorted(chosen):
            if a in residual or b in residual:
                continue
            r = residual[0]
            endpoints = [(r, r)] if len(residual) == 1 else [tuple(residual), tuple(residual[::-1])]
            for left, right in endpoints:
                additions = tuple(sorted((tuple(sorted((a, left))), tuple(sorted((b, right))))))
                gain = float(scores[a, left]) + float(scores[b, right]) - float(scores[a, b])
                options.append((-gain, a, b, additions))
        _, a, b, additions = min(options)
        chosen.remove((a, b))
        chosen.update(additions)
        repairs = 1
    pairs = pair_array(chosen)
    assert pairs.shape[1] == n
    assert np.all(np.bincount(pairs.ravel(), minlength=n) == 2)
    return pairs, dict(target=n, candidate_pairs=n * (n - 1) // 2, repairs=repairs)


SELECTORS = {"knn": budget_knn, "cycles": local_cycles}
