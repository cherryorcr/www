"""Prepare a deterministic raw-to-synthetic-slot provenance map.

The map is built only from the previous synthetic memory and the current
support delta.  It never loads a historical raw graph.  A synthetic slot has
capacity ``base["mass"]``; current raw rows are assigned within their class
to slots using a joint graph-text cosine score, with exact capacity
conservation.  The result is metadata for the direct Student outer optimizer,
not a modified memory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def load(path: Path):
    return torch.load(path, map_location="cpu", weights_only=False)


def _joint_payload(x: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
    if x.ndim != 2 or text.ndim != 2 or len(x) != len(text):
        raise ValueError("X and text payloads must be aligned rank-2 tensors")
    x = F.normalize(x.float(), dim=1)
    text = F.normalize(text.float(), dim=1)
    return F.normalize(torch.cat((x, text), dim=1), dim=1)


def _capacity_values(mass: torch.Tensor, slot_rows: torch.Tensor) -> list[int]:
    values = mass[slot_rows].float()
    rounded = values.round().long()
    if bool((values - rounded.float()).abs().gt(1e-4).any()):
        raise ValueError("provenance mapping requires integer synthetic masses")
    if bool((rounded <= 0).any()):
        raise ValueError("every provenance slot must have positive mass")
    return [int(value) for value in rounded.tolist()]


def _balanced_assignment(scores: torch.Tensor,
                         capacities: list[int]) -> tuple[list[int], list[float]]:
    """Assign every row to one slot while preserving each slot capacity.

    The global score order is deterministic under ties (raw row, then slot).
    This is deliberately a bounded construction-time assignment, rather than
    a search over historical graph instances.
    """
    if scores.ndim != 2 or scores.shape[1] != len(capacities):
        raise ValueError("score matrix and slot capacities are misaligned")
    if sum(capacities) != scores.shape[0]:
        raise ValueError("slot capacities do not conserve current raw mass")
    pairs = [
        (float(value), int(raw_id), int(slot_id))
        for raw_id, row in enumerate(scores.detach().cpu().tolist())
        for slot_id, value in enumerate(row)
    ]
    pairs.sort(key=lambda item: (-item[0], item[1], item[2]))
    remaining = list(capacities)
    assignment = [-1] * int(scores.shape[0])
    similarity = [float("nan")] * int(scores.shape[0])
    for value, raw_id, slot_id in pairs:
        if assignment[raw_id] >= 0 or remaining[slot_id] <= 0:
            continue
        assignment[raw_id] = slot_id
        similarity[raw_id] = value
        remaining[slot_id] -= 1
        if not any(remaining):
            break
    if any(value < 0 for value in assignment) or any(remaining):
        raise AssertionError("capacity-constrained assignment left rows unmatched")
    return assignment, similarity


def build_provenance_map(base: dict, support: dict, old_rows: int,
                         support_class_id: torch.Tensor,
                         base_slot_class_id: torch.Tensor) -> tuple[dict, dict]:
    """Return support slot ids, base group ids, and an audit record."""
    if old_rows < 2 or old_rows >= len(support["x"]):
        raise ValueError("old_rows must leave a current support delta")
    if old_rows > len(base["x"]):
        raise ValueError("old_rows exceeds candidate memory")
    if len(support_class_id) != len(support["x"]):
        raise ValueError("support class ids must align with support rows")
    if len(base_slot_class_id) != len(base["x"]):
        raise ValueError("base slot class ids must align with base rows")
    if bool((support_class_id[:old_rows] >= 0).any()):
        raise ValueError("old support rows must have class id -1")
    if bool((support_class_id[old_rows:] < 0).any()):
        raise ValueError("current support rows need nonnegative class ids")
    if bool((base_slot_class_id[:old_rows] >= 0).any()):
        raise ValueError("old base rows must have slot class id -1")
    if bool((base_slot_class_id[old_rows:] < 0).any()):
        raise ValueError("new base rows need slot class ids")

    support_slot_id = torch.full((len(support["x"]),), -1, dtype=torch.long)
    base_slot_group_id = torch.full((len(base["x"]),), -1, dtype=torch.long)
    base_slot_group_id[old_rows:] = torch.arange(
        len(base["x"]) - old_rows, dtype=torch.long)

    classes = sorted(set(int(value) for value in
                         support_class_id[old_rows:].tolist()))
    slot_classes = set(int(value) for value in
                       base_slot_class_id[old_rows:].tolist())
    if set(classes) != slot_classes:
        raise ValueError(
            f"support classes {classes} and slot classes {sorted(slot_classes)} differ")
    rows_audit = []
    for class_id in classes:
        raw_rows = torch.where(support_class_id == int(class_id))[0]
        raw_rows = raw_rows[raw_rows >= int(old_rows)]
        slot_rows = torch.where(base_slot_class_id == int(class_id))[0]
        slot_rows = slot_rows[slot_rows >= int(old_rows)]
        capacities = _capacity_values(base["mass"], slot_rows)
        if sum(capacities) != len(raw_rows):
            raise ValueError(
                f"class {class_id} slot capacity {sum(capacities)} does not "
                f"match current raw mass {len(raw_rows)}")
        raw_joint = _joint_payload(support["x"][raw_rows],
                                   support["text_emb"][raw_rows])
        slot_joint = _joint_payload(base["x"][slot_rows],
                                    base["text_emb"][slot_rows])
        scores = raw_joint @ slot_joint.T
        assignment, similarities = _balanced_assignment(scores, capacities)
        slot_group = (slot_rows - int(old_rows)).tolist()
        support_slot_id[raw_rows] = torch.tensor(
            [slot_group[value] for value in assignment], dtype=torch.long)
        rows_audit.append({
            "class_id": int(class_id),
            "raw_rows": int(len(raw_rows)),
            "slot_rows": [int(value) for value in slot_rows.tolist()],
            "slot_group_ids": [int(value) for value in slot_group],
            "slot_capacities": capacities,
            "assigned_counts": [assignment.count(index)
                                 for index in range(len(capacities))],
            "capacity_conserved": True,
            "mean_joint_cosine": float(sum(similarities) / len(similarities)),
            "min_joint_cosine": float(min(similarities)),
            "max_joint_cosine": float(max(similarities)),
        })

    if bool((support_slot_id[old_rows:] < 0).any()):
        raise AssertionError("not every current support row received a slot")
    audit = {
        "name": "x-provenance-slot-map-v1",
        "old_rows": int(old_rows),
        "new_base_rows": int(len(base["x"]) - old_rows),
        "current_support_rows": int(len(support["x"]) - old_rows),
        "classes": rows_audit,
        "joint_payload": "l2-normalized X concatenated with l2-normalized text",
        "assignment": "deterministic capacity-constrained descending cosine",
        "capacity_source": "base.mass",
        "historical_raw_reads": 0,
        "report_labels_used": False,
        "old_support_slot_id": -1,
        "old_base_slot_group_id": -1,
        "mass_conservation": True,
    }
    return {
        "support_slot_id": support_slot_id,
        "base_slot_group_id": base_slot_group_id,
    }, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--support", required=True, type=Path)
    parser.add_argument("--support-class-id", required=True, type=Path)
    parser.add_argument("--base-slot-class-id", required=True, type=Path)
    parser.add_argument("--old-rows", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"immutable output exists: {args.output}")
    base, support = load(args.base), load(args.support)
    support_class_id = load(args.support_class_id)
    base_slot_class_id = load(args.base_slot_class_id)
    if not all(isinstance(value, torch.Tensor) for value in
               (support_class_id, base_slot_class_id)):
        raise TypeError("class metadata artifacts must contain tensors")
    result, audit = build_provenance_map(
        base, support, args.old_rows, support_class_id.long(),
        base_slot_class_id.long())
    args.output.mkdir(parents=True, exist_ok=False)
    torch.save(result["support_slot_id"], args.output / "support_slot_id.pt")
    torch.save(result["base_slot_group_id"],
               args.output / "base_slot_group_id.pt")
    (args.output / "audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps({
        "event": "x_provenance_slot_map_complete",
        "output": str(args.output),
        "current_support_rows": audit["current_support_rows"],
        "new_base_rows": audit["new_base_rows"],
        "classes": [row["class_id"] for row in audit["classes"]],
        "mass_conservation": True,
    }), flush=True)


if __name__ == "__main__":
    main()
