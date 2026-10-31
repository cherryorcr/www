"""Build a verified synthetic-slot-to-class map for class-aware X outer steps."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def dispersion_allocation(support: dict, class_id: torch.Tensor,
                          old_rows: int, budget: int) -> dict[int, int]:
    classes = sorted(set(int(value) for value in class_id[old_rows:].tolist()))
    if not classes or any(value < 0 for value in classes):
        raise ValueError("current support rows must have nonnegative class ids")
    counts, dispersions = [], []
    for value in classes:
        rows = torch.where(class_id[old_rows:] == value)[0] + old_rows
        counts.append(max(1, len(rows)))
        if len(rows) <= 1:
            dispersions.append(0.0)
            continue
        x = F.normalize(support["x"][rows].float(), dim=1)
        text = F.normalize(support["text_emb"][rows].float(), dim=1)
        dx = 1.0 - (x @ x.mean(0, keepdim=True).T).mean()
        dt = 1.0 - (text @ text.mean(0, keepdim=True).T).mean()
        dispersions.append(float(((dx + dt) * 0.5).clamp(0.0, 1.0)))
    weights = torch.tensor(counts, dtype=torch.float32).sqrt() * (
        1.0 + torch.tensor(dispersions, dtype=torch.float32))
    minimum = 1 if budget >= len(classes) else 0
    minimum = min(minimum, budget // len(classes))
    remaining = budget - minimum * len(classes)
    raw = remaining * weights / weights.sum().clamp_min(1e-12)
    allocated = torch.floor(raw).long() + minimum
    left = int(budget - allocated.sum())
    if left > 0:
        order = torch.argsort(raw - torch.floor(raw), descending=True).tolist()
        for index in order[:left]:
            allocated[index] += 1
    result = dict(zip(classes, (int(value) for value in allocated.tolist())))
    if sum(result.values()) != budget:
        raise AssertionError("dispersion allocation does not match synthetic rows")
    return result


def build_slot_classes(base: dict, support: dict, old_rows: int,
                       support_class_id: torch.Tensor) -> tuple[torch.Tensor, dict]:
    if len(support_class_id) != len(support["x"]):
        raise ValueError("support class ids are not aligned to support rows")
    if old_rows < 2 or old_rows >= len(support["x"]):
        raise ValueError("invalid old-row boundary")
    if len(base["x"]) < old_rows:
        raise ValueError("base is shorter than its old prefix")
    if bool((support_class_id[:old_rows] >= 0).any()):
        raise ValueError("old synthetic rows must remain unlabeled in this artifact")
    if bool((support_class_id[old_rows:] < 0).any()):
        raise ValueError("current raw support rows need class ids")

    new_rows = len(base["x"]) - old_rows
    allocation = dispersion_allocation(
        support, support_class_id, old_rows, new_rows)
    slot_class_id = torch.full((len(base["x"]),), -1, dtype=torch.long)
    cursor = old_rows
    rows_audit = []
    for class_id, count in allocation.items():
        source_rows = torch.where(support_class_id[old_rows:] == class_id)[0]
        source_count = len(source_rows)
        slot_mass = float(base["mass"][cursor:cursor + count].sum())
        if abs(slot_mass - source_count) > 1e-4:
            raise ValueError(
                f"class {class_id} slot mass {slot_mass} does not match "
                f"current support count {source_count}; refusing guessed mapping")
        slot_class_id[cursor:cursor + count] = int(class_id)
        rows_audit.append({
            "class_id": int(class_id),
            "source_rows": int(source_count),
            "slot_start": int(cursor),
            "slot_end_exclusive": int(cursor + count),
            "slot_count": int(count),
            "slot_mass": slot_mass,
            "mass_matches_source_count": True,
        })
        cursor += count
    if cursor != len(base["x"]):
        raise AssertionError("slot class map does not cover the new memory block")
    audit = {
        "name": "class-aware-x-slot-map-v1",
        "allocation_mode": "dispersion",
        "old_rows": int(old_rows),
        "new_rows": int(new_rows),
        "class_allocation": {str(key): int(value)
                             for key, value in allocation.items()},
        "classes": rows_audit,
        "mapping_verified_by": "per-class synthetic mass equals current raw support count",
        "historical_raw_reads": 0,
        "report_labels_used": False,
    }
    return slot_class_id, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--support", required=True, type=Path)
    parser.add_argument("--support-class-id", required=True, type=Path)
    parser.add_argument("--old-rows", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"immutable output exists: {args.output}")
    load = lambda path: torch.load(path, map_location="cpu", weights_only=False)
    base, support, class_id = load(args.base), load(args.support), load(args.support_class_id)
    if not all(isinstance(value, torch.Tensor) for value in (base["x"], support["x"], class_id)):
        raise TypeError("input artifact has an unexpected tensor type")
    slot_class_id, audit = build_slot_classes(
        base, support, args.old_rows, class_id.long())
    args.output.mkdir(parents=True, exist_ok=False)
    torch.save(slot_class_id, args.output / "base_slot_class_id.pt")
    (args.output / "audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps({"event": "class_aware_x_slot_map_complete",
                      "output": str(args.output),
                      "class_allocation": audit["class_allocation"],
                      "mass_verified": True}), flush=True)


if __name__ == "__main__":
    main()
