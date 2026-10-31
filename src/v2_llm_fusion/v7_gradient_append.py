"""v7-A gradient-compatible append candidates and low-cost selector."""
from __future__ import annotations

from copy import deepcopy
import math

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn.functional as F

import v6_local_marginal as V6

C, U, SCA, G, P = V6.C, V6.U, V6.SCA, V6.G, V6.P
GRADIENT_STEPS = (0, 1, 2)
SELECTION_GROUPS = ((7300, 7301), (7400, 7401))
REPORT_SEEDS = tuple(range(9300, 9306))
LAMBDA_DYN = 0.5


def _integer_budget(weights: torch.Tensor, budget: int, minimum: int = 0,
                    capacities: list[int] | None = None) -> list[int]:
    weights = weights.float().clamp_min(1e-8)
    n = len(weights)
    if n == 0 or budget <= 0:
        return [0] * n
    caps = torch.tensor(capacities if capacities is not None else [budget] * n)
    if len(caps) != n or (caps < 0).any() or int(caps.sum()) < budget:
        raise ValueError("insufficient subcluster capacity")
    minimum = min(int(minimum), int(budget) // n)
    alloc = torch.minimum(torch.full((n,), minimum, dtype=torch.long), caps)
    while int(alloc.sum()) < budget:
        left = int(budget) - int(alloc.sum())
        active = alloc < caps
        raw = left * weights * active / (weights * active).sum()
        increment = torch.minimum(torch.floor(raw).long(), caps - alloc)
        alloc += increment
        left = budget - int(alloc.sum())
        eligible = sorted((i for i in range(n) if alloc[i] < caps[i]),
                          key=lambda i: (-float(raw[i] - torch.floor(raw[i])), i))
        for index in eligible[:left]:
            alloc[index] += 1
    return alloc.tolist()


def class_stats(data: dict, ids: torch.Tensor, classes: list[int]) -> list[dict]:
    stats = []
    for cls in sorted(int(c) for c in classes):
        members = ids[data["y"][ids] == cls]
        points = V6.joint_points(data, members)
        center = points.mean(0, keepdim=True)
        dispersion = float((points - center).square().sum() / max(1, len(points)))
        stats.append({"class": cls, "count": int(len(members)),
                      "dispersion": dispersion})
    return stats


def affected_pair(data: dict, ids: torch.Tensor, classes: list[int]) -> tuple[int, int]:
    stats = class_stats(data, ids, classes)
    if len(stats) < 2:
        raise ValueError("v7-A requires at least two new classes")
    donor = max(stats, key=lambda r: (r["count"], -r["class"]))["class"]
    recipient = max((r for r in stats if r["class"] != donor),
                    key=lambda r: (r["dispersion"], -r["class"]))["class"]
    return int(donor), int(recipient)


def multi_prototype_records(data: dict, ids: torch.Tensor,
                            classes: list[int], class_rows: dict[int, int],
                            device) -> list[dict]:
    """Create synthetic-only multi-prototype source partitions.

    The source ids are retained only for offline audit. Every deployed row is
    synthesized as a mean of its assigned source members.
    """
    records = []
    for cls in sorted(int(c) for c in classes):
        members = ids[data["y"][ids] == cls]
        target = int(class_rows[cls])
        if target <= 0 or len(members) == 0:
            continue
        q = min(target, max(1, min(4, int(math.sqrt(max(1, len(members) / 64.0))) + 1)))
        points = V6.joint_points(data, members)
        assign = SCA.cluster_assign(points, torch.ones(len(points)), q, device)
        subclusters = []
        for sub in range(q):
            local = members[assign == sub]
            if len(local):
                p = V6.joint_points(data, local)
                center = p.mean(0, keepdim=True)
                dispersion = float((p - center).square().sum() / max(1, len(p)))
                subclusters.append((local, dispersion))
        weights = torch.tensor([
            math.sqrt(len(local)) * (1.0 + dispersion)
            for local, dispersion in subclusters], dtype=torch.float32)
        rows = _integer_budget(weights, target, minimum=1 if target >= len(subclusters) else 0,
                               capacities=[len(local) for local, _ in subclusters])
        for (local, _), row_count in zip(subclusters, rows):
            row_count = min(int(row_count), len(local))
            if row_count <= 0:
                continue
            if row_count == 1:
                pieces = [local]
            else:
                split = SCA.cluster_assign(V6.joint_points(data, local),
                                            torch.ones(len(local)), row_count, device)
                pieces = [local[split == j] for j in range(row_count)]
            for piece in pieces:
                if len(piece) == 0:
                    continue
                records.append({"class": cls, "ids": piece})
    if len(records) != sum(int(v) for v in class_rows.values()):
        raise AssertionError("multi-prototype row allocation mismatch")
    return records


def align_class_slots(data: dict, base_records: list[dict], records: list[dict]) -> list[dict]:
    """Align equal-quota candidates to C0 without using student performance."""
    aligned = deepcopy(base_records)
    for cls in sorted({r["class"] for r in base_records}):
        slots = [i for i, r in enumerate(base_records) if r["class"] == cls]
        proposals = [r for r in records if r["class"] == cls]
        if len(slots) != len(proposals):
            raise ValueError("alignment requires equal class quotas")
        anchors = torch.stack([V6.joint_points(data, base_records[i]["ids"]).mean(0) for i in slots])
        centers = torch.stack([V6.joint_points(data, r["ids"]).mean(0) for r in proposals])
        rows, cols = linear_sum_assignment(torch.cdist(anchors, centers).square().numpy())
        for row, col in zip(rows, cols):
            aligned[slots[int(row)]] = proposals[int(col)]
    return aligned


def make_candidates(data: dict, ids: torch.Tensor, base: dict,
                    stage: int, device) -> tuple[dict, list[dict]]:
    classes = [int(c) for c in data["stage_classes"][stage]]
    allocation = U.allocate_rows_by_class(data, ids, classes, len(base["x"]), "dispersion")
    base_records = V6.source_clusters(data, ids, allocation, device)
    V6.validate_base(data, ids, base_records, base, stage, device)
    output = {"c0": {"block": base, "records": base_records, "family": "c0",
                     "geometry_error": V6.geometry_error(data, base_records),
                     "changed_slots": [], "allocation": allocation,
                     "candidate_rule": "frozen C0"}}
    skipped = []

    try:
        donor, recipient = affected_pair(data, ids, classes)
        edit_result = V6.local_edit(data, base_records, donor, recipient, device)
        if edit_result is None:
            raise ValueError("deterministic affected pair has no valid split/merge")
        local, edit = edit_result
        proposal = V6.aligned_recluster(data, ids, base_records, local, edit, device)
        V6.validate_partition(data, ids, proposal, classes)
        block, changed = V6.candidate_block(data, base, base_records, proposal, stage)
        output["recluster"] = {
            "block": block, "records": proposal, "family": "recluster",
            "edit": edit, "changed_slots": changed,
            "allocation": {int(c): sum(r["class"] == c for r in proposal) for c in classes},
            "geometry_error": V6.geometry_error(data, proposal),
            "candidate_rule": "deterministic highest-count donor / highest-dispersion recipient",
        }
    except (AssertionError, ValueError) as exc:
        skipped.append({"candidate": "recluster", "reason": str(exc)})

    try:
        multi = multi_prototype_records(data, ids, classes, allocation, device)
        multi = align_class_slots(data, base_records, multi)
        V6.validate_partition(data, ids, multi, classes)
        block, changed = V6.candidate_block(data, base, base_records, multi, stage)
        output["multi_prototype"] = {
            "block": block, "records": multi, "family": "multi_prototype",
            "changed_slots": changed, "allocation": allocation,
            "geometry_error": V6.geometry_error(data, multi),
            "candidate_rule": "class-wise joint subclusters with complexity allocation",
        }
    except (AssertionError, ValueError) as exc:
        skipped.append({"candidate": "multi_prototype", "reason": str(exc)})
    return output, skipped


def info_nce_loss(model, memory: dict, rows: torch.Tensor, device, adjacency=None):
    x = memory["x"].to(device)
    text = memory["text_emb"].to(device)
    adj = C.training_adjacency(memory, device) if adjacency is None else adjacency
    graph = model.encode_graph(x, adj)[rows]
    tt = model.encode_text(text[rows])
    logits = model.logit_scale.exp().clamp(max=100) * graph @ tt.T
    labels = torch.arange(len(rows), device=device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


def gradient_curve(memory: dict, seed: int, device,
                   steps: int = 2, keep_state: bool = False) -> dict:
    if steps != 2:
        raise ValueError("v7-A is frozen to two optimizer updates")
    torch.manual_seed(seed)
    model = G.CLIPStudent(memory["x"].shape[1]).to(device)
    initial = C.tensor_hash(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=P.STUDENT.lr)
    generator = torch.Generator().manual_seed(seed * 7 + 1)
    adjacency = C.training_adjacency(memory, device)
    inputs = {"x": memory["x"].to(device), "text_emb": memory["text_emb"].to(device)}
    observations = []
    for step in range(steps + 1):
        rows = torch.randperm(len(memory["x"]), generator=generator)[:min(P.STUDENT.batch, len(memory["x"]))].to(device)
        optimizer.zero_grad(set_to_none=True)
        model.train()
        loss = info_nce_loss(model, inputs, rows, device, adjacency)
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite gradient loss")
        grads = torch.autograd.grad(loss, tuple(model.parameters()), create_graph=False)
        flat = torch.cat([g.detach().float().reshape(-1) for g in grads])
        if not torch.isfinite(flat).all():
            raise FloatingPointError("nonfinite gradient")
        observations.append({"step": step, "loss": float(loss.detach().cpu()),
                             "gradient_norm": float(flat.norm().detach().cpu()),
                             "batch_rows": rows.cpu(),
                             "gradient": flat.detach().cpu()})
        for param, grad in zip(model.parameters(), grads):
            param.grad = grad.detach()
        if step < steps:
            optimizer.step()
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    result = {"seed": seed, "observations": observations, "optimizer_updates": steps,
              "start_state_sha256": initial, "end_state_sha256": C.tensor_hash(state),
              "parameter_shapes": {n: list(p.shape) for n, p in model.named_parameters()},
              "representation": "full flattened parameter gradients; no random projection"}
    if keep_state:
        result["state"] = state
    return result


def gradient_score(curve: dict, base_curve: dict, base_geometry: float,
                   geometry: float) -> dict:
    if (len(curve["observations"]) != 3 or len(base_curve["observations"]) != 3
            or not np.isfinite([base_geometry, geometry]).all()):
        raise ValueError("invalid gradient observations or geometry")
    dyn = []
    for current, base in zip(curve["observations"], base_curve["observations"]):
        a, b = current["gradient"], base["gradient"]
        if a.shape != b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError("invalid gradient vectors")
        distance = 0. if torch.equal(a, b) else float(1.0 - F.cosine_similarity(a[None], b[None]).item())
        dyn.append(min(2., max(0., distance)))
    coverage_gain = float((base_geometry - geometry) / max(abs(base_geometry), 1e-8))
    score = coverage_gain - LAMBDA_DYN * float(np.mean(dyn))
    return {"coverage_gain": coverage_gain, "dynamic_distance": dyn,
            "dynamic_distance_mean": float(np.mean(dyn)), "score": score}


def select_gradient(scores: dict[str, dict[int, dict]], names: list[str],
                    groups=SELECTION_GROUPS) -> dict:
    if "c0" not in names:
        raise ValueError("missing C0 fallback")
    picks = []
    group_scores = {}
    flat = sum((list(g) for g in groups), [])
    if set(flat) & set(REPORT_SEEDS) or len(flat) != len(set(flat)):
        raise ValueError("selection seeds must be unique and disjoint from report")
    if not all(np.isfinite(scores[n][s]["score"]) for n in names for s in flat):
        raise ValueError("nonfinite selection score")
    for group in groups:
        values = {name: float(np.mean([scores[name][seed]["score"] for seed in group]))
                  for name in names}
        group_scores[str(group)] = values
        picks.append(max(names, key=lambda n: (values[n], n == "c0", n)))
    selected = picks[0] if len(set(picks)) == 1 else "c0"
    score_gain = float(np.mean([scores[selected][seed]["score"] for seed in flat]))
    accepted = selected != "c0" and score_gain > 0.0
    return {"selected": selected if accepted else "c0", "accepted": bool(accepted),
            "group_picks": picks, "score_gain": score_gain,
            "group_scores": group_scores,
            "lambda_dyn": LAMBDA_DYN,
            "scope": "selection-only; two-step gradient sketch; no report metrics"}
