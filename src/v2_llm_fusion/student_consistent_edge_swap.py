"""Bounded Student-consistent topology swaps.

This module is a diagnostic operator for the current Semantic-Recursive
memory.  It does not change X, text, rows, or old-old/new-new edges.  The
only editable part is the old-new cross block.  Candidate changes are
two-switches between endpoint slots, so the old endpoint load and every new
row's cross degree are conserved exactly, including rows with cross degree
greater than one.

The selector uses a short, fixed Student response rather than a projected
message proxy.  It is deliberately conservative: a swap must improve the
new-block one-step loss reduction, stay within the old-block regression
guard, and not worsen the total post-update loss.  Any failed gate returns
the input topology unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import sys
from typing import Callable, Mapping

import torch
import torch.nn.functional as F


def _canonical_student_module():
    """Load the frozen canonical Student without importing it in unit tests."""
    try:
        import run_gate1_continual as module
        return module
    except ModuleNotFoundError as error:
        if error.name != "run_gate1_continual":
            raise
    here = Path(__file__).resolve().parent
    frozen = here.parent / "frozen_v1_2026_09_09"
    if not frozen.exists():
        raise ModuleNotFoundError(
            "run_gate1_continual is unavailable and the frozen Student path "
            f"does not exist: {frozen}")
    sys.path.insert(0, str(frozen))
    import run_gate1_continual as module
    return module


@dataclass(frozen=True)
class Config:
    """Small fixed-budget configuration for one topology diagnostic."""

    max_swaps: int = 1
    student_steps: int = 1
    student_lr: float = 1e-3
    batch: int = 16
    new_gain_margin: float = 0.0
    max_old_gain_drop: float = 0.0
    max_total_loss_increase: float = 0.0
    new_weight: float = 1.0
    old_weight: float = 0.5
    min_seed_fraction: float = 0.0

    def __post_init__(self) -> None:
        if self.max_swaps < 0 or self.student_steps < 1:
            raise ValueError("max_swaps and student_steps must be valid")
        if self.student_lr <= 0 or self.batch < 0:
            raise ValueError("student_lr and batch must be positive/non-negative")
        for name in ("new_gain_margin", "max_old_gain_drop",
                     "max_total_loss_increase", "new_weight", "old_weight",
                     "min_seed_fraction"):
            value = float(getattr(self, name))
            if not torch.isfinite(torch.tensor(value)) or value < 0:
                raise ValueError(f"invalid {name}")
        if self.min_seed_fraction > 1:
            raise ValueError("min_seed_fraction must be at most one")
        if self.new_weight <= 0:
            raise ValueError("new_weight must be positive")

    def dict(self) -> dict:
        return asdict(self)


def _as_pairs(edge_index: torch.Tensor) -> set[tuple[int, int]]:
    """Return an undirected pair set without changing the caller tensor."""
    edge_index = torch.as_tensor(edge_index, dtype=torch.long).cpu()
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E]")
    pairs: set[tuple[int, int]] = set()
    for left, right in edge_index.T.tolist():
        left, right = int(left), int(right)
        if left == right:
            continue
        pairs.add(tuple(sorted((left, right))))
    return pairs


def _cross_map(edge_index: torch.Tensor, old_rows: int) -> dict[int, tuple[int, ...]]:
    """Extract ``new_row -> old_row slots`` from an undirected edge index."""
    result: dict[int, list[int]] = {}
    for left, right in sorted(_as_pairs(edge_index)):
        if left < old_rows <= right:
            result.setdefault(right, []).append(left)
    return {node: tuple(sorted(values)) for node, values in result.items()}


def _validate_cross_blocks(
        base_edge: torch.Tensor, candidate_edge: torch.Tensor,
        old_rows: int) -> tuple[dict[int, tuple[int, ...]], dict[int, set[int]]]:
    """Check the candidate pool and return its base map and alternatives."""
    if old_rows <= 0:
        raise ValueError("old_rows must be positive")
    base = _cross_map(base_edge, old_rows)
    candidate_pairs = _as_pairs(candidate_edge)
    alternatives: dict[int, set[int]] = {node: set() for node in base}
    for left, right in sorted(candidate_pairs):
        if left < old_rows <= right:
            if right not in alternatives:
                raise ValueError("candidate pool changes the new-row set")
            alternatives[right].add(left)
    if any(not values for values in alternatives.values()):
        raise ValueError("candidate pool omits a base new-row cross edge")
    return base, alternatives


def enumerate_two_switches(base_edge: torch.Tensor, candidate_edge: torch.Tensor,
                           old_rows: int) -> list[dict[int, tuple[int, ...]]]:
    """Enumerate legal endpoint swaps from a bounded candidate pool.

    A returned mapping differs from the base in two new rows.  One endpoint
    slot is exchanged between each row.  The operation therefore preserves
    every old endpoint's capacity and each row's cross degree exactly, while
    rejecting swaps that would create a duplicate edge within one row.
    """
    base, alternatives = _validate_cross_blocks(base_edge, candidate_edge,
                                                 int(old_rows))
    nodes = sorted(base)
    result: list[dict[int, tuple[int, ...]]] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    for index, first in enumerate(nodes):
        for second in nodes[index + 1:]:
            for old_first in base[first]:
                for old_second in base[second]:
                    if old_first == old_second:
                        continue
                    if old_second not in alternatives[first]:
                        continue
                    if old_first not in alternatives[second]:
                        continue
                    first_slots = set(base[first])
                    second_slots = set(base[second])
                    first_slots.remove(old_first)
                    second_slots.remove(old_second)
                    if old_second in first_slots or old_first in second_slots:
                        continue
                    first_slots.add(old_second)
                    second_slots.add(old_first)
                    mapping = dict(base)
                    mapping[first] = tuple(sorted(first_slots))
                    mapping[second] = tuple(sorted(second_slots))
                    key = tuple(sorted(mapping.items()))
                    if key not in seen:
                        result.append(mapping)
                        seen.add(key)
    return result


def _normalize_mapping(mapping: Mapping[int, object]) -> dict[int, tuple[int, ...]]:
    """Normalize scalar degree-one and iterable multi-slot mappings."""
    result: dict[int, tuple[int, ...]] = {}
    for node, values in mapping.items():
        if isinstance(values, int):
            values = (values,)
        elif torch.is_tensor(values) and values.ndim == 0:
            values = (int(values.item()),)
        else:
            values = tuple(int(value) for value in values)
        if len(set(values)) != len(values):
            raise ValueError("a new row cannot contain duplicate old endpoints")
        result[int(node)] = tuple(sorted(values))
    return result


def materialize_cross_swap(base_edge: torch.Tensor, mapping: Mapping[int, object],
                           old_rows: int) -> torch.Tensor:
    """Replace only the cross block and preserve both internal blocks."""
    base_edge = torch.as_tensor(base_edge, dtype=torch.long).cpu()
    base_map = _cross_map(base_edge, int(old_rows))
    mapping = _normalize_mapping(mapping)
    if set(mapping) != set(base_map):
        raise ValueError("mapping must cover exactly the base new-row set")
    if any(len(mapping[node]) != len(base_map[node]) for node in base_map):
        raise ValueError("mapping changes a new row's cross degree")
    base_values = sorted(value for values in base_map.values() for value in values)
    result_values = sorted(value for values in mapping.values() for value in values)
    if base_values != result_values:
        raise ValueError("mapping changes old endpoint capacity")
    base_pairs = _as_pairs(base_edge)
    internal = sorted((left, right) for left, right in base_pairs
                      if not (left < old_rows <= right))
    cross = sorted((old, new) for new, olds in mapping.items() for old in olds)
    pairs = internal + cross
    directed = pairs + [(right, left) for left, right in pairs]
    if not directed:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(directed, dtype=torch.long).T.contiguous()


def topology_audit(base_edge: torch.Tensor, result_edge: torch.Tensor,
                   old_rows: int) -> dict:
    """Report the invariants that make the operator safe to compare."""
    old_rows = int(old_rows)
    base_pairs, result_pairs = _as_pairs(base_edge), _as_pairs(result_edge)
    base_old_old = {p for p in base_pairs if p[1] < old_rows}
    result_old_old = {p for p in result_pairs if p[1] < old_rows}
    base_new_new = {p for p in base_pairs if p[0] >= old_rows}
    result_new_new = {p for p in result_pairs if p[0] >= old_rows}
    base_map, result_map = _cross_map(base_edge, old_rows), _cross_map(result_edge, old_rows)
    base_values = [value for values in base_map.values() for value in values]
    result_values = [value for values in result_map.values() for value in values]
    base_load = torch.bincount(torch.tensor(base_values, dtype=torch.long),
                               minlength=old_rows) if base_map else torch.zeros(old_rows, dtype=torch.long)
    result_load = torch.bincount(torch.tensor(result_values, dtype=torch.long),
                                 minlength=old_rows) if result_map else torch.zeros(old_rows, dtype=torch.long)
    changed = 0
    for node in base_map:
        changed += len(set(base_map.get(node, ())).symmetric_difference(
            set(result_map.get(node, ()))))
    changed //= 2
    return {
        "old_old_preserved": base_old_old == result_old_old,
        "new_new_preserved": base_new_new == result_new_new,
        "new_row_set_preserved": set(base_map) == set(result_map),
        "per_new_cross_degree_preserved": (
            set(base_map) == set(result_map) and all(
                len(base_map[node]) == len(result_map[node]) for node in base_map)),
        "old_endpoint_capacity_preserved": torch.equal(base_load, result_load),
        "base_cross_pairs": sum(len(values) for values in base_map.values()),
        "result_cross_pairs": sum(len(values) for values in result_map.values()),
        "changed_cross_pairs": int(changed),
    }


def _clone_with_edge(memory: Mapping[str, object], edge_index: torch.Tensor) -> dict:
    result = dict(memory)
    result["edge_index"] = edge_index
    return result


def _metric(stats: Mapping[str, float], name: str) -> float:
    aliases = {"new": ("new_gain", "mean_new_gain"),
               "old": ("old_gain", "mean_old_gain"),
               "total": ("total_after", "mean_total_after")}
    for key in aliases[name]:
        if key in stats:
            return float(stats[key])
    raise KeyError(f"student evaluator did not return {name} metric")


def _seed_guard(base_stats: Mapping[str, object], trial_stats: Mapping[str, object],
                config: Config) -> tuple[bool, dict]:
    """Apply an optional majority-of-seeds guard to a candidate.

    The aggregate gate remains the primary compatibility rule.  When the
    evaluator exposes per-seed records, this additional guard prevents one
    unusually favorable average from accepting a topology that harms most
    selector seeds.  A zero fraction keeps the historical test-double API.
    """
    required = float(config.min_seed_fraction)
    if required <= 0:
        return True, {"enabled": False}
    base_records = base_stats.get("records")
    trial_records = trial_stats.get("records")
    if not isinstance(base_records, list) or not isinstance(trial_records, list):
        raise ValueError("min_seed_fraction requires evaluator records")
    base_by_seed = {int(row["seed"]): row for row in base_records}
    trial_by_seed = {int(row["seed"]): row for row in trial_records}
    seeds = sorted(set(base_by_seed) & set(trial_by_seed))
    if not seeds:
        raise ValueError("evaluator records have no matching seeds")
    accepted = 0
    per_seed = []
    for seed in seeds:
        base = base_by_seed[seed]
        trial = trial_by_seed[seed]
        delta_new = float(trial["new_gain"]) - float(base["new_gain"])
        delta_old = float(trial["old_gain"]) - float(base["old_gain"])
        delta_total = float(trial["total_after"]) - float(base["total_after"])
        legal = (delta_new >= float(config.new_gain_margin)
                 and delta_old >= -float(config.max_old_gain_drop)
                 and delta_total <= float(config.max_total_loss_increase))
        accepted += int(legal)
        per_seed.append({"seed": seed, "delta_new_gain": delta_new,
                         "delta_old_gain": delta_old,
                         "delta_total_after": delta_total,
                         "legal": bool(legal)})
    fraction = accepted / len(seeds)
    return fraction >= required, {
        "enabled": True,
        "required_fraction": required,
        "legal_seed_count": accepted,
        "seed_count": len(seeds),
        "legal_fraction": fraction,
        "per_seed": per_seed,
    }


def select_swaps(
        base_memory: Mapping[str, object], candidate_edge: torch.Tensor,
        old_rows: int, config: Config,
        evaluate: Callable[[Mapping[str, object]], Mapping[str, float]]) -> tuple[dict, dict]:
    """Select bounded swaps using a caller-supplied Student evaluator.

    ``evaluate`` must use the same initialization, rows, optimizer and short
    horizon for every candidate.  It may be the canonical one-step InfoNCE
    evaluator or a deterministic test double.  The function never sees raw
    graph data and never uses report labels.
    """
    base_edge = torch.as_tensor(base_memory["edge_index"], dtype=torch.long).cpu()
    base_stats = dict(evaluate(base_memory))
    current_edge = base_edge
    current_memory = _clone_with_edge(base_memory, current_edge)
    current_stats = base_stats
    accepted: list[dict] = []
    all_trials: list[dict] = []

    for round_index in range(int(config.max_swaps)):
        best = None
        candidates = enumerate_two_switches(current_edge, candidate_edge,
                                             int(old_rows))
        base_new = _metric(current_stats, "new")
        base_old = _metric(current_stats, "old")
        base_total = _metric(current_stats, "total")
        for mapping in candidates:
            edge = materialize_cross_swap(current_edge, mapping, int(old_rows))
            if torch.equal(edge, current_edge):
                continue
            trial = _clone_with_edge(base_memory, edge)
            stats = dict(evaluate(trial))
            delta_new = _metric(stats, "new") - base_new
            delta_old = _metric(stats, "old") - base_old
            delta_total = _metric(stats, "total") - base_total
            aggregate_legal = (delta_new >= float(config.new_gain_margin)
                     and delta_old >= -float(config.max_old_gain_drop)
                     and delta_total <= float(config.max_total_loss_increase))
            seed_legal, seed_audit = _seed_guard(current_stats, stats, config)
            legal = aggregate_legal and seed_legal
            score = (float(config.new_weight) * delta_new
                     + float(config.old_weight) * delta_old)
            record = {"round": round_index, "delta_new_gain": delta_new,
                      "delta_old_gain": delta_old, "delta_total_after": delta_total,
                      "score": score, "aggregate_legal": bool(aggregate_legal),
                      "seed_legal": bool(seed_legal), "seed_guard": seed_audit,
                      "legal": bool(legal),
                      "mapping": {str(k): list(v) for k, v in sorted(mapping.items())}}
            all_trials.append(record)
            if legal and (best is None or score > best[0]):
                best = (score, mapping, edge, stats, record)
        if best is None:
            break
        _, mapping, current_edge, current_stats, record = best
        accepted.append(record)
        current_memory = _clone_with_edge(base_memory, current_edge)

    fallback = not accepted
    if fallback:
        result_edge = base_edge
        result_stats = base_stats
    else:
        result_edge = current_edge
        result_stats = current_stats
    audit = {
        "operator": "student_consistent_bounded_cross_edge_swap",
        "config": config.dict(),
        "candidate_count": len(all_trials),
        "accepted_count": len(accepted),
        "accepted": accepted,
        "fallback_to_input": fallback,
        "base_stats": base_stats,
        "result_stats": result_stats,
        "topology": topology_audit(base_edge, result_edge, int(old_rows)),
        "report_labels_used": False,
        "historical_raw_reads": 0,
    }
    if not all(audit["topology"][key] for key in (
            "old_old_preserved", "new_new_preserved", "new_row_set_preserved",
            "per_new_cross_degree_preserved", "old_endpoint_capacity_preserved")):
        raise AssertionError("student-consistent swap violated topology invariants")
    return _clone_with_edge(base_memory, result_edge), audit


def evaluate_one_step(memory: Mapping[str, object], old_rows: int,
                      seeds: tuple[int, ...], device: str = "cpu",
                      config: Config = Config()) -> dict:
    """Evaluate the canonical GCN/InfoNCE response for a fixed short horizon."""
    # Import lazily so the pure topology helpers remain usable without the
    # transformer/data-loading stack in unit tests.
    G = _canonical_student_module()

    x = torch.as_tensor(memory["x"], dtype=torch.float32).to(device)
    text = torch.as_tensor(memory["text_emb"], dtype=torch.float32).to(device)
    edge = torch.as_tensor(memory["edge_index"], dtype=torch.long)
    if len(x) != len(text) or old_rows < 1 or old_rows >= len(x):
        raise ValueError("invalid memory or old_rows")
    adjacency = G.norm_adj(edge, len(x), device)

    def rows_for(seed: int, start: int, end: int) -> torch.Tensor:
        values = torch.arange(start, end, device=device)
        if config.batch <= 0 or len(values) <= config.batch:
            return values
        generator = torch.Generator(device="cpu").manual_seed(int(seed) * 7919 + start)
        return values[torch.randperm(len(values), generator=generator,
                                     device="cpu")[:config.batch].to(device)]

    records = []
    for seed in tuple(int(v) for v in seeds):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            model = G.CLIPStudent(x.shape[1], text_dim=text.shape[1]).to(device)
        old_batch = rows_for(seed, 0, int(old_rows))
        new_batch = rows_for(seed + 1, int(old_rows), len(x))
        train_rows = torch.cat((old_batch, new_batch))

        def loss(rows: torch.Tensor) -> torch.Tensor:
            graph = model.encode_graph(x, adjacency)[rows]
            text_z = model.encode_text(text[rows])
            logits = model.logit_scale.exp().clamp(max=100) * (graph @ text_z.T)
            target = torch.arange(len(rows), device=device)
            return (F.cross_entropy(logits, target)
                    + F.cross_entropy(logits.T, target)) / 2

        def group_loss(rows: torch.Tensor) -> torch.Tensor:
            if len(rows) == 0:
                return x.new_zeros(())
            return loss(rows)

        with torch.no_grad():
            old_before = float(group_loss(old_batch))
            new_before = float(group_loss(new_batch))
            total_before = float(group_loss(train_rows))
        optimizer = torch.optim.Adam(model.parameters(), lr=float(config.student_lr))
        for _ in range(int(config.student_steps)):
            value = loss(train_rows)
            optimizer.zero_grad(set_to_none=True)
            value.backward()
            optimizer.step()
        with torch.no_grad():
            old_after = float(group_loss(old_batch))
            new_after = float(group_loss(new_batch))
            total_after = float(group_loss(train_rows))
        records.append({
            "seed": seed,
            "old_gain": old_before - old_after,
            "new_gain": new_before - new_after,
            "total_after": total_after,
            "old_before": old_before, "new_before": new_before,
            "old_after": old_after, "new_after": new_after,
        })
        del model
    if not records:
        raise ValueError("at least one Student seed is required")
    return {
        "old_gain": float(sum(row["old_gain"] for row in records) / len(records)),
        "new_gain": float(sum(row["new_gain"] for row in records) / len(records)),
        "total_after": float(sum(row["total_after"] for row in records) / len(records)),
        "records": records,
        "student_steps": int(config.student_steps),
        "student_seeds": list(seeds),
    }
