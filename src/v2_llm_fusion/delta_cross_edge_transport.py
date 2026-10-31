"""Bounded new/old cross-edge transport for a current raw delta.

The module is intentionally independent from X/P revision.  It builds the
support graph used by a Teacher from ``old synthetic memory + current raw
delta`` and changes only the cross block.  Historical raw nodes are never
needed unless a sealed ``raw_id -> old synthetic row`` provenance map is
explicitly supplied.
"""
from __future__ import annotations

from collections import defaultdict
import copy
import math

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def _clone(value):
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, list):
        return list(value)
    if isinstance(value, dict):
        return {key: _clone(item) for key, item in value.items()}
    return copy.deepcopy(value)


def _norm_adj(edge_index: torch.Tensor, n: int) -> torch.Tensor:
    edge_index = edge_index.long().cpu()
    matrix = torch.zeros((int(n), int(n)), dtype=torch.float32)
    if edge_index.numel():
        matrix.index_put_((edge_index[0], edge_index[1]),
                          torch.ones(edge_index.shape[1]), accumulate=True)
    matrix.fill_diagonal_(1.0)
    degree = matrix.sum(1).clamp_min(1.0)
    inv = degree.rsqrt()
    return inv[:, None] * matrix * inv[None, :]


def _symmetric_edges(pairs: set[tuple[int, int]]) -> torch.Tensor:
    values = []
    for left, right in sorted(pairs):
        if int(left) == int(right):
            continue
        values.extend(((int(left), int(right)), (int(right), int(left))))
    if not values:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(values, dtype=torch.long).T.contiguous()


def _deduplicate_edges(edge_index: torch.Tensor) -> torch.Tensor:
    pairs = {(int(left), int(right))
             for left, right in edge_index.long().cpu().T.tolist()
             if int(left) != int(right)}
    return torch.tensor(sorted(pairs), dtype=torch.long).T.contiguous() \
        if pairs else torch.empty((2, 0), dtype=torch.long)


def build_raw_to_old_slot(records: list[dict], old_rows: int) -> dict[int, int]:
    """Build a sealed raw-id to old synthetic-row map from audit records."""
    result: dict[int, int] = {}
    for row, record in enumerate(records):
        if row >= int(old_rows):
            break
        values = record.get("source_original_ids", [])
        if not isinstance(values, list):
            raise ValueError("source_original_ids must be a list")
        for raw_id in values:
            raw_id = int(raw_id)
            previous = result.get(raw_id)
            if previous is not None and previous != row:
                raise ValueError(f"raw id {raw_id} maps to multiple old rows")
            result[raw_id] = int(row)
    return result


def _base_edges(old: dict, delta: dict) -> torch.Tensor:
    old_n = len(old["x"])
    old_edges = old["edge_index"].long().cpu()
    new_edges = delta["edge_index"].long().cpu()
    if new_edges.numel():
        new_edges = new_edges + int(old_n)
    return _deduplicate_edges(torch.cat((old_edges, new_edges), dim=1))


def _joint_scores(old: dict, delta: dict) -> torch.Tensor:
    old_x = F.normalize(old["x"].float(), dim=1)
    new_x = F.normalize(delta["x"].float(), dim=1)
    old_t = F.normalize(old["text_emb"].float(), dim=1)
    new_t = F.normalize(delta["text_emb"].float(), dim=1)
    return 0.5 * old_x @ new_x.T + 0.5 * old_t @ new_t.T


def _message_scores(old: dict, delta: dict) -> torch.Tensor:
    """Content-aware one-hop message compatibility without labels."""
    old_adj = _norm_adj(old["edge_index"], len(old["x"]))
    new_adj = _norm_adj(delta["edge_index"], len(delta["x"]))
    old_msg = F.normalize(old_adj @ old["x"].float(), dim=1)
    new_msg = F.normalize(new_adj @ delta["x"].float(), dim=1)
    scores = old_msg @ new_msg.T
    # Keep a small direct-content term so isolated delta nodes are not scored
    # only by their self-loop message.
    scores = 0.75 * scores + 0.25 * (
        F.normalize(old["x"].float(), dim=1) @
        F.normalize(delta["x"].float(), dim=1).T)
    return scores


def _incident_candidates(raw_edge_index: torch.Tensor,
                         current_ids: torch.Tensor,
                         raw_to_old_slot: dict[int, int] | None,
                         old_rows: int, new_rows: int) -> dict[int, dict[int, int]]:
    if raw_to_old_slot is None:
        return {}
    current_ids = current_ids.long().cpu()
    current_pos = {int(raw_id): int(pos)
                   for pos, raw_id in enumerate(current_ids.tolist())}
    counts: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for left, right in raw_edge_index.long().cpu().T.tolist():
        left, right = int(left), int(right)
        if left in current_pos and right in raw_to_old_slot:
            new_pos, old_slot = current_pos[left], int(raw_to_old_slot[right])
        elif right in current_pos and left in raw_to_old_slot:
            new_pos, old_slot = current_pos[right], int(raw_to_old_slot[left])
        else:
            continue
        if 0 <= new_pos < int(new_rows) and 0 <= old_slot < int(old_rows):
            counts[new_pos][old_slot] += 1
    return {int(new_pos): dict(values) for new_pos, values in counts.items()}


def _choose(scores: torch.Tensor, degree: int, mode: str, seed: int,
            incident: dict[int, dict[int, int]] | None = None,
            fill: bool = False) -> tuple[set[tuple[int, int]], dict]:
    old_rows, new_rows = scores.shape
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    pairs: set[tuple[int, int]] = set()
    audits = []
    balanced_load = [0 for _ in range(old_rows)]
    balanced_cap = (math.ceil(new_rows * int(degree) / old_rows)
                    if mode == "message_balanced" and old_rows else None)
    for new_pos in range(new_rows):
        if mode.startswith("incident"):
            allowed = sorted((incident or {}).get(new_pos, {}).keys())
            counts = (incident or {}).get(new_pos, {})
        else:
            allowed = list(range(old_rows))
            counts = {}
        fallback_used = False
        if not allowed and fill:
            allowed = list(range(old_rows))
            fallback_used = True
        if not allowed:
            selected = []
        elif mode == "random":
            order = torch.randperm(len(allowed), generator=generator).tolist()
            selected = [allowed[index] for index in order[:min(degree, len(allowed))]]
        elif mode == "message_balanced":
            remaining = set(allowed)
            selected = []
            for _ in range(min(int(degree), len(remaining))):
                available = [value for value in remaining
                             if balanced_load[value] < int(balanced_cap)]
                pool = available if available else list(remaining)
                choice = min(pool, key=lambda value:
                             (-float(scores[value, new_pos]), value))
                selected.append(choice)
                remaining.remove(choice)
                balanced_load[choice] += 1
        elif mode.startswith("incident") and "semantic" not in mode and "message" not in mode:
            selected = sorted(allowed, key=lambda value: (-counts[value], value))[:degree]
        else:
            selected = sorted(allowed,
                              key=lambda value: (-float(scores[value, new_pos]), value))[:degree]
        for old_pos in selected:
            pairs.add((int(old_pos), int(new_pos)))
        audits.append({"new_pos": int(new_pos), "candidate_count": len(allowed),
                       "selected_old": [int(v) for v in selected],
                       "fallback_used": bool(fallback_used)})
    selection = {"rows": audits, "edge_pairs": len(pairs)}
    if mode == "message_balanced":
        selection["capacity_cap"] = int(balanced_cap)
        selection["distinct_old"] = int(sum(value > 0 for value in balanced_load))
        selection["max_old_load"] = int(max(balanced_load, default=0))
    return pairs, selection


def make_support(old: dict, delta: dict, mode: str = "none",
                 cross_degree: int = 1, seed: int = 0,
                 raw_edge_index: torch.Tensor | None = None,
                 current_ids: torch.Tensor | None = None,
                 raw_to_old_slot: dict[int, int] | None = None,
                 fill_incident: bool = False) -> tuple[dict, dict]:
    """Append a delta and add a bounded cross-edge block.

    ``semantic`` and ``message`` use only old synthetic/current delta payloads.
    ``incident_*`` additionally use current-delta incident edges and a sealed
    raw-to-synthetic provenance map.  No historical raw features are read.
    """
    valid = {"none", "random", "semantic", "message", "message_balanced", "incident",
             "incident_semantic", "incident_message"}
    if mode not in valid:
        raise ValueError(f"unknown cross-edge mode: {mode}")
    old_rows, new_rows = len(old["x"]), len(delta["x"])
    if cross_degree < 0:
        raise ValueError("cross_degree must be non-negative")
    base = _base_edges(old, delta)
    if mode == "none" or cross_degree == 0 or old_rows == 0 or new_rows == 0:
        cross_pairs, selection = set(), {"rows": [], "edge_pairs": 0}
    else:
        if mode in {"semantic", "incident_semantic"}:
            scores = _joint_scores(old, delta)
        elif mode in {"message", "message_balanced", "incident_message"}:
            scores = _message_scores(old, delta)
        else:
            scores = torch.zeros((old_rows, new_rows))
        incident = None
        if mode.startswith("incident"):
            if raw_edge_index is None or current_ids is None:
                raise ValueError("incident modes require raw edges and current ids")
            if raw_to_old_slot is None:
                raise ValueError("incident modes require sealed raw-to-old provenance")
            incident = _incident_candidates(raw_edge_index, current_ids,
                                            raw_to_old_slot, old_rows, new_rows)
        cross_pairs, selection = _choose(
            scores, int(cross_degree), mode, int(seed), incident,
            fill=bool(fill_incident))
    cross = _symmetric_edges({(old_pos, old_rows + new_pos)
                               for old_pos, new_pos in cross_pairs})
    support = {key: _clone(value) for key, value in old.items()}
    for key in ("x", "text_emb", "mass", "depth", "stage"):
        support[key] = torch.cat((old[key], delta[key]), dim=0)
    support["texts"] = list(old["texts"]) + list(delta["texts"])
    support["edge_index"] = _deduplicate_edges(torch.cat((base, cross), dim=1))
    if "class_id" in old or "class_id" in delta:
        old_class = old.get("class_id", torch.full((old_rows,), -1, dtype=torch.long))
        new_class = delta.get("class_id", torch.full((new_rows,), -1, dtype=torch.long))
        support["class_id"] = torch.cat((old_class.long(), new_class.long()))
    if "current_delta_ids" in delta:
        support["current_delta_ids"] = delta["current_delta_ids"].clone()
    audit = {
        "mode": mode,
        "cross_degree": int(cross_degree),
        "old_rows": int(old_rows), "new_rows": int(new_rows),
        "base_edge_count": int(base.shape[1]),
        "cross_edge_pairs": int(len(cross_pairs)),
        "cross_edge_count_directed": int(cross.shape[1]),
        "support_edge_count": int(support["edge_index"].shape[1]),
        "selection": selection,
        "provenance_rows": int(len(raw_to_old_slot or {})),
        "historical_raw_reads": 0,
        "report_labels_used": False,
    }
    if mode.startswith("incident"):
        audit["incident_candidate_new_rows"] = int(
            sum(bool(values) for values in (selection.get("rows") or [])))
    return support, audit


def _partition_base_edges(edge_index: torch.Tensor, old_rows: int):
    old_old: set[tuple[int, int]] = set()
    new_new: set[tuple[int, int]] = set()
    cross_by_new: dict[int, set[int]] = defaultdict(set)
    for left, right in edge_index.long().cpu().T.tolist():
        left, right = int(left), int(right)
        if left < int(old_rows) and right < int(old_rows):
            old_old.add((left, right))
        elif left >= int(old_rows) and right >= int(old_rows):
            new_new.add((left, right))
        elif left < int(old_rows) <= right:
            cross_by_new[right].add(left)
        elif right < int(old_rows) <= left:
            cross_by_new[left].add(right)
    return old_old, new_new, cross_by_new


def materialize_cross_only(base_edge: torch.Tensor, projected: torch.Tensor,
                           old_rows: int, preserve_capacity: bool = True
                           ) -> tuple[torch.Tensor, dict]:
    """Replace only cross edges using a projected support message kernel.

    The old-old and new-new edge sets are copied exactly.  Each new row keeps
    its C0 cross-edge count; optional old-endpoint capacities are initialized
    from C0 and respected whenever possible.  This is the hard materializer
    used by the topology-only causal experiment, so X/P/text never change.
    """
    old_rows = int(old_rows)
    total = int(projected.shape[0])
    if projected.ndim != 2 or projected.shape[0] != projected.shape[1]:
        raise ValueError("projected kernel must be square")
    if total < old_rows:
        raise ValueError("projected kernel is smaller than old prefix")
    old_old, new_new, cross_by_new = _partition_base_edges(base_edge, old_rows)
    old_capacity = {old: 0 for old in range(old_rows)}
    for values in cross_by_new.values():
        for old in values:
            old_capacity[old] += 1
    remaining = dict(old_capacity)
    selected_by_new: dict[int, list[int]] = {}
    score_records = []
    for new_node in range(old_rows, total):
        target = sorted(cross_by_new.get(new_node, set()))
        need = len(target)
        if need == 0:
            selected_by_new[new_node] = []
            continue
        scores = projected[:old_rows, new_node].float().cpu()
        ranked = sorted(range(old_rows),
                        key=lambda old: (-float(scores[old]), int(old)))
        chosen: list[int] = []
        if preserve_capacity:
            chosen.extend(old for old in ranked if remaining.get(old, 0) > 0)
            chosen = chosen[:need]
        else:
            chosen = ranked[:need]
        # A zero/empty projected candidate must not delete a C0 connection.
        # Fill remaining slots first from the original endpoints, then from
        # the highest-scoring unused rows.
        for old in target:
            if len(chosen) >= need:
                break
            if old not in chosen:
                chosen.append(old)
        for old in ranked:
            if len(chosen) >= need:
                break
            if old not in chosen:
                chosen.append(old)
        chosen = sorted(chosen[:need])
        for old in chosen:
            remaining[old] = max(0, remaining.get(old, 0) - 1)
        selected_by_new[new_node] = chosen
        score_records.append({
            "new_node": int(new_node), "target_degree": int(need),
            "base_old": target, "selected_old": chosen,
            "selected_score": [float(scores[old]) for old in chosen],
        })
    cross_pairs = {(old, new) for new, values in selected_by_new.items()
                   for old in values}
    result = _deduplicate_edges(torch.cat((
        torch.tensor(sorted(old_old), dtype=torch.long).T
        if old_old else torch.empty((2, 0), dtype=torch.long),
        torch.tensor(sorted(new_new), dtype=torch.long).T
        if new_new else torch.empty((2, 0), dtype=torch.long),
        _symmetric_edges(cross_pairs),
    ), dim=1))
    audit = {
        "old_old_edges": len(old_old),
        "new_new_edges": len(new_new),
        "cross_pairs_before": int(sum(len(v) for v in cross_by_new.values())),
        "cross_pairs_after": int(len(cross_pairs)),
        "old_old_preserved": True,
        "new_new_preserved": True,
        "per_new_cross_degree_preserved": all(
            len(selected_by_new.get(node, [])) == len(values)
            for node, values in cross_by_new.items()),
        "old_capacity_overflow": int(sum(max(0, old_capacity[old] - remaining[old])
                                         for old in old_capacity)),
        "rows": score_records,
    }
    return result, audit


def materialize_cross_only_global(base_edge: torch.Tensor,
                                  projected: torch.Tensor,
                                  old_rows: int) -> tuple[torch.Tensor, dict]:
    """Materialize a degree-one cross block with global old-slot assignment.

    The existing materializer processes new rows in index order.  This
    variant keeps every old endpoint's C0 cross-edge capacity, but solves one
    global linear assignment over all new rows and the duplicated old-capacity
    slots.  The current frozen protocol has one cross edge per new row; fail
    explicitly for larger cross degree instead of silently allowing duplicate
    old endpoints within one new row.
    """
    old_rows = int(old_rows)
    total = int(projected.shape[0])
    if projected.ndim != 2 or projected.shape[0] != projected.shape[1]:
        raise ValueError("projected kernel must be square")
    if total < old_rows:
        raise ValueError("projected kernel is smaller than old prefix")

    old_old, new_new, cross_by_new = _partition_base_edges(base_edge, old_rows)
    targets = {int(node): sorted(values)
               for node, values in cross_by_new.items()}
    invalid = {node: len(values) for node, values in targets.items()
               if len(values) != 1}
    if invalid:
        raise ValueError(
            "global cross assignment currently requires one C0 cross edge "
            f"per new row; invalid degrees={invalid}")

    old_capacity = {old: 0 for old in range(old_rows)}
    for values in targets.values():
        for old in values:
            old_capacity[old] += 1
    new_nodes = sorted(targets)
    old_slots = [old for old in range(old_rows)
                 for _ in range(old_capacity[old])]
    if len(new_nodes) != len(old_slots):
        raise ValueError("old capacity does not match new cross-edge slots")
    if not new_nodes:
        result = _deduplicate_edges(torch.cat((
            torch.tensor(sorted(old_old), dtype=torch.long).T
            if old_old else torch.empty((2, 0), dtype=torch.long),
            torch.tensor(sorted(new_new), dtype=torch.long).T
            if new_new else torch.empty((2, 0), dtype=torch.long),
        ), dim=1))
        return result, {
            "old_old_edges": len(old_old), "new_new_edges": len(new_new),
            "cross_pairs_before": 0, "cross_pairs_after": 0,
            "old_old_preserved": True, "new_new_preserved": True,
            "per_new_cross_degree_preserved": True,
            "old_capacity_exact": True,
            "assignment_method": "global_capacity_linear_sum_assignment",
            "global_objective": 0.0, "rows": [],
        }

    scores = torch.stack([
        projected[old_slots, new_node].detach().float().cpu()
        for new_node in new_nodes
    ], dim=0).double()
    if not bool(torch.isfinite(scores).all()):
        raise ValueError("projected cross scores contain non-finite values")
    row_indices, column_indices = linear_sum_assignment((-scores).numpy())
    if len(row_indices) != len(new_nodes):
        raise ValueError("global assignment did not cover every new row")

    selected_by_new = {}
    score_records = []
    for row_index, column_index in zip(row_indices.tolist(),
                                       column_indices.tolist()):
        new_node = new_nodes[int(row_index)]
        old_node = old_slots[int(column_index)]
        selected_by_new[new_node] = [old_node]
        score_records.append({
            "new_node": int(new_node), "target_degree": 1,
            "base_old": targets[new_node],
            "selected_old": [int(old_node)],
            "selected_score": [float(scores[row_index, column_index])],
        })
    if set(selected_by_new) != set(new_nodes):
        raise ValueError("global assignment omitted a new cross-edge row")

    cross_pairs = {(old, new) for new, values in selected_by_new.items()
                   for old in values}
    result = _deduplicate_edges(torch.cat((
        torch.tensor(sorted(old_old), dtype=torch.long).T
        if old_old else torch.empty((2, 0), dtype=torch.long),
        torch.tensor(sorted(new_new), dtype=torch.long).T
        if new_new else torch.empty((2, 0), dtype=torch.long),
        _symmetric_edges(cross_pairs),
    ), dim=1))
    selected_counts = {old: 0 for old in range(old_rows)}
    for old, _ in cross_pairs:
        selected_counts[old] += 1
    audit = {
        "old_old_edges": len(old_old),
        "new_new_edges": len(new_new),
        "cross_pairs_before": int(sum(len(v) for v in targets.values())),
        "cross_pairs_after": int(len(cross_pairs)),
        "old_old_preserved": True,
        "new_new_preserved": True,
        "per_new_cross_degree_preserved": all(
            len(selected_by_new.get(node, [])) == len(values)
            for node, values in targets.items()),
        "old_capacity_exact": selected_counts == old_capacity,
        "assignment_method": "global_capacity_linear_sum_assignment",
        "global_objective": float(scores[row_indices, column_indices].sum()),
        "rows": sorted(score_records, key=lambda value: value["new_node"]),
    }
    return result, audit
