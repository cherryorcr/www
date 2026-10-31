"""Run one auditable GC-SR X-only construction arm.

The runner deliberately accepts prepared ``base`` and ``support`` memory
artifacts.  This keeps source locality explicit: the caller must materialize
the legal mixed support (previous synthetic memory + current raw delta), while
this process only edits X and never reads report/test labels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import gc_sr_gradient as GC


MODES = {
    "output": dict(gradient_weight=0.0, output_weight=1.0,
                    old_anchor_weight=0.0),
    "gradient": dict(gradient_weight=1.0, output_weight=0.0,
                     old_anchor_weight=0.0),
    "joint": dict(gradient_weight=1.0, output_weight=1.0,
                  old_anchor_weight=0.0),
    "gradient_anchor": dict(gradient_weight=1.0, output_weight=0.0,
                             old_anchor_weight=0.05),
    # Emphasize the current-stage block while retaining a weak full-support
    # alignment term. This is the first direct plasticity diagnostic; it is
    # not selected by report-dev scores.
    "joint_new": dict(gradient_weight=1.0, output_weight=1.0,
                       new_gradient_weight=1.0, old_gradient_weight=0.25,
                       old_anchor_weight=0.0),
}


def load(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True),
                    encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--support", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--stage", required=True, type=int)
    parser.add_argument("--old-rows", required=True, type=int)
    parser.add_argument("--editable-rows", required=True, type=int, nargs="+")
    parser.add_argument("--mode", choices=sorted(MODES), required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--outer-steps", type=int, default=8)
    parser.add_argument("--outer-lr", type=float, default=0.01)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--old-radius", type=float, default=0.05)
    parser.add_argument("--new-radius", type=float, default=0.20)
    parser.add_argument("--train-seeds", type=int, nargs="+", default=[9100, 9101])
    parser.add_argument("--validation-seeds", type=int, nargs="+", default=[9110, 9111])
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"immutable output exists: {args.output}")
    base, support = load(args.base), load(args.support)
    cfg = GC.Config(
        outer_steps=args.outer_steps,
        outer_lr=args.outer_lr,
        batch=args.batch,
        old_radius=args.old_radius,
        new_radius=args.new_radius,
        train_seeds=tuple(args.train_seeds),
        validation_seeds=tuple(args.validation_seeds),
        **MODES[args.mode],
    )
    result, audit = GC.optimize(
        base, support, args.editable_rows, args.old_rows,
        args.stage, args.device, cfg,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(result, args.output / "memory.pt")
    audit.update({
        "base_file": str(args.base),
        "support_file": str(args.support),
        "mode": args.mode,
        "device": args.device,
        "source_schema": "prepared legal support; no raw historical access in optimizer",
    })
    save_json(args.output / "audit.json", audit)
    save_json(args.output / "protocol.json", {
        "name": "gc-sr-x-only-v1",
        "mode": args.mode,
        "stage": args.stage,
        "old_rows": args.old_rows,
        "editable_rows": args.editable_rows,
        "text": "frozen",
        "budget": "frozen",
        "topology": "frozen",
        "student": "canonical two-layer GCN + bidirectional InfoNCE",
        "report_labels_used": False,
        "historical_raw_reads": 0,
    })
    print(json.dumps({"event": "gc_sr_complete", "mode": args.mode,
                      "output": str(args.output),
                      "max_relative_displacement": audit["max_relative_displacement"]}),
          flush=True)


if __name__ == "__main__":
    main()
