"""Bounded propagation-improving two-switches with an exact degree invariant."""
from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def refine_degree_preserving(edge_index, weights, x, target, max_swaps=2,
                             candidate_limit=64, neighbor_limit=4):
    n, device = len(x), x.device
    adjacency = torch.zeros((n, n), dtype=torch.bool, device=device)
    if edge_index.numel():
        adjacency[edge_index[0].to(device), edge_index[1].to(device)] = True
    adjacency = adjacency | adjacency.T
    adjacency.fill_diagonal_(False)
    original_degree = adjacency.sum(1)
    degree = original_degree.float() + 1
    target = F.normalize(target.to(device), dim=1)
    scaled = (adjacency.float() + torch.eye(n, device=device)) / (degree[:, None] * degree[None, :]).sqrt()
    messages = scaled @ x
    error = (F.normalize(messages, dim=1) - target).square().sum(1)
    before = float(error.mean())
    accepted = 0
    for _ in range(max_swaps):
        tri = torch.triu_indices(n, n, 1, device=device)
        allowed = (~adjacency[tri[0], tri[1]]) & (weights[tri[0], tri[1]] > 0)
        tri = tri[:, allowed]
        if not tri.numel():
            break
        rank = torch.argsort(weights[tri[0], tri[1]], descending=True, stable=True)[:candidate_limit]
        proposals = []
        adj_cpu = adjacency.cpu()
        for u, v in tri[:, rank].T.cpu().tolist():
            aa = torch.where(adj_cpu[u])[0].tolist()[:neighbor_limit]
            bb = torch.where(adj_cpu[v])[0].tolist()[:neighbor_limit]
            for a in aa:
                for b in bb:
                    if len({u, v, a, b}) == 4 and not adj_cpu[a, b] and weights[a, b] > 0:
                        proposals.append((u, a, v, b))
        if not proposals:
            break
        ids = torch.tensor(proposals, dtype=torch.long, device=device)
        u, a, v, b = ids.T
        def contribution(left, right):
            return x[right] / (degree[left] * degree[right]).sqrt()[:, None]
        delta = torch.stack((contribution(u, v) - contribution(u, a),
                             contribution(a, b) - contribution(a, u),
                             contribution(v, u) - contribution(v, b),
                             contribution(b, a) - contribution(b, v)), dim=1)
        changed = messages[ids] + delta
        proposed_error = (F.normalize(changed, dim=2) - target[ids]).square().sum(2)
        improvement = (error[ids] - proposed_error).sum(1)
        best = int(improvement.argmax())
        if float(improvement[best]) <= 1e-8:
            break
        ui, ai, vi, bi = ids[best].tolist()
        for left, right in ((ui, ai), (vi, bi)):
            adjacency[left, right] = adjacency[right, left] = False
        for left, right in ((ui, vi), (ai, bi)):
            adjacency[left, right] = adjacency[right, left] = True
        messages[ids[best]] = changed[best]
        error[ids[best]] = proposed_error[best]
        accepted += 1
    if not torch.equal(adjacency.sum(1), original_degree):
        raise AssertionError("two-switch changed node degree")
    final_scaled = (adjacency.float() + torch.eye(n, device=device)) / (degree[:, None] * degree[None, :]).sqrt()
    after = float((F.normalize(final_scaled @ x, dim=1) - target).square().sum(1).mean())
    if after > before + 1e-6:
        raise AssertionError("accepted swap increased propagation error")
    return torch.nonzero(adjacency, as_tuple=False).T.cpu().contiguous(), {
        "degree_preserved": True, "accepted_swaps": accepted,
        "propagation_before": before, "propagation_after": after,
        "degree_reference": "Q-induced top-k graph, not original raw graph"}
