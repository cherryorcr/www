"""Sparse negatives and exact greedy topology for large LP validation."""
from __future__ import annotations

import heapq
import numpy as np
import torch

import run_link_prediction_transfer as L
import safe_joint_kernel_append as S


def nonedges(left, right, n, count, forbidden, rng):
    left, right = np.asarray(left), np.asarray(right)
    if count == 0:
        return torch.empty((2, 0), dtype=torch.long)
    same = np.array_equal(left, right)
    capacity = len(left) * (len(left) - 1) // 2 if same else len(left) * len(right)
    if count > capacity:
        raise ValueError("negative sample exceeds stratum capacity")
    forbidden = np.unique(np.asarray(forbidden, dtype=np.int64))
    picked = np.empty(0, dtype=np.int64)
    for _ in range(1000):
        remaining = count - len(picked)
        if remaining == 0:
            return torch.from_numpy(np.stack((picked // n, picked % n)))
        size = max(1024, min(1000000, 3 * remaining))
        a, b = rng.choice(left, size), rng.choice(right, size)
        keep = a != b
        keys = np.unique(np.minimum(a[keep], b[keep]) * n + np.maximum(a[keep], b[keep]))
        exclude = np.union1d(forbidden, picked)
        indices = np.searchsorted(exclude, keys)
        padded = np.r_[exclude, np.iinfo(np.int64).max]
        keys = keys[padded[indices] != keys]
        rng.shuffle(keys)
        picked = np.r_[picked, keys[:remaining]]
    raise ValueError("negative rejection sampler exhausted; graph may be dense")


def split_edges(edge, arrival, seed):
    n = len(arrival)
    positive = L.pairs(edge, n)
    positive = positive[:, (arrival[positive] >= 0).all(0)]
    forbidden = (positive[0] * n + positive[1]).numpy()
    rng = np.random.default_rng(seed)
    ap = arrival[positive].sort(dim=0).values
    buckets = {key: [] for key in ("train_pos", "val_pos", "test_pos", "train_neg", "val_neg", "test_neg")}
    for first in range(int(arrival.max()) + 1):
        for last in range(first, int(arrival.max()) + 1):
            pidx = torch.where((ap[0] == first) & (ap[1] == last))[0].numpy()
            rng.shuffle(pidx)
            nv = nt = int(.1 * len(pidx))
            negatives = nonedges(torch.where(arrival == first)[0].numpy(),
                                 torch.where(arrival == last)[0].numpy(), n, len(pidx), forbidden, rng)
            for name, subset in (("val", slice(0, nv)), ("test", slice(nv, nv + nt)),
                                 ("train", slice(nv + nt, None))):
                buckets[name + "_pos"].append(positive[:, pidx[subset]])
                buckets[name + "_neg"].append(negatives[:, subset])
    result = {key: torch.cat(value, dim=1) for key, value in buckets.items()}
    result.update(arrival=arrival, seed=seed, full_pos=positive)
    L.validate_split(result)
    return result


def training_negatives(edges, n, seed):
    count = min(n * (n - 1) // 2 - edges.shape[1], max(4096, edges.shape[1] * 16))
    return nonedges(np.arange(n), np.arange(n), n, count,
                    (edges[0] * n + edges[1]).numpy(), np.random.default_rng(seed))


def select_internal(block, weights, variant):
    """Same greedy cap-2/path rule and tie breaks as the frozen reference.

    Coverage penalties are monotone after the first edge. A sorted reservoir
    plus a lazy heap avoids rescoring every possible edge at every step.
    """
    score = S.kernel_matrix(block, block, weights).numpy()
    n = len(score)
    if n < 3:
        return S.select_internal(block, weights, variant)
    u, v = np.triu_indices(n, 1)
    base = score[u, v]
    parent, counts, chosen = list(range(n)), [0] * n, []

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def allowed(index):
        i, j = int(u[index]), int(v[index])
        return counts[i] < 2 and counts[j] < 2 and root(i) != root(j)

    def choose(index):
        i, j = int(u[index]), int(v[index])
        chosen.append((i, j))
        counts[i] += 1
        counts[j] += 1
        parent[root(i)] = root(j)

    if variant == "degree_balanced":
        for index in np.argsort(-base, kind="stable"):
            if allowed(index):
                choose(index)
                if len(chosen) == n - 1:
                    break
    elif variant == "coverage":
        first = int(np.argmax(base))
        choose(first)
        a, b = chosen[0]
        norms = np.maximum(score[u, u] + score[v, v] + 2 * base, np.float32(1e-12))
        norm = np.maximum(score[a, a] + score[b, b] + np.float32(2) * score[a, b], np.float32(1e-12))
        penalty = (score[u, a] + score[u, b] + score[v, a] + score[v, b]) / np.sqrt(norms * norm)
        bound = base.astype(np.float64) - .15 * penalty
        # NumPy would otherwise do this subtraction in float32 in older releases.
        bound = base.astype(np.float64) - .15 * penalty.astype(np.float64)
        order = np.lexsort((np.arange(len(u)), -base, -bound))
        heap, at = [], 0
        while len(chosen) < n - 1:
            while at < len(order) and not allowed(order[at]):
                at += 1
            while heap and not allowed(heap[0][2]):
                heapq.heappop(heap)
            initial = None if at == len(order) else int(order[at])
            key = None if initial is None else (-float(bound[initial]), -float(base[initial]), initial)
            if heap and (key is None or heap[0][:3] < key):
                _, _, index, checked = heapq.heappop(heap)
            elif initial is not None:
                index, checked = initial, 1
                at += 1
            else:
                raise AssertionError("greedy path exhausted")
            if checked == len(chosen):
                choose(index)
                continue
            i, j = int(u[index]), int(v[index])
            a, b = np.asarray(chosen[checked:], dtype=np.int64).T
            norm = np.maximum(score[a, a] + score[b, b] + 2 * score[a, b], np.float32(1e-12))
            current = (score[i, a] + score[i, b] + score[j, a] + score[j, b]) / np.sqrt(norms[index] * norm)
            penalty[index] = max(penalty[index], float(current.max()))
            value = float(base[index]) - .15 * float(penalty[index])
            heapq.heappush(heap, (-value, -float(base[index]), index, len(chosen)))
    else:
        raise ValueError(variant)
    ends = [i for i, count in enumerate(counts) if count == 1]
    assert len(chosen) == n - 1 and len(ends) == 2
    chosen.append(tuple(sorted(ends)))
    return S.edge_tensor(chosen)


def append(old, block, weights, variant):
    internal = select_internal(block, weights, variant)
    if old is None:
        return {**{k: S.copy_value(v) for k, v in block.items()}, "edge_index": internal}
    keep = len(old["x"])
    out = {k: torch.cat((old[k], v)) if isinstance(v, torch.Tensor) else old[k] + v
           for k, v in block.items()}
    cross, _ = S.cross_pairs(old, block, weights, variant)
    out["edge_index"] = torch.cat((old["edge_index"], internal + keep, cross), dim=1)
    return out
