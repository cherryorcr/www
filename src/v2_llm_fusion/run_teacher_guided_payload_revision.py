"""Run Teacher/P/X revision on a Semantic-Recursive stage artifact."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import run_content_gap_attribution as CAG
import teacher_guided_payload_revision as TGR
import protocol_v3


def load(path: Path):
    return torch.load(path, map_location="cpu", weights_only=False)


def raw_fields(raw):
    if isinstance(raw, dict):
        x = raw["node_f"].float()
        texts = list(raw.get("text_list", []))
        names = [str(v) for v in raw["all_labels"]]
        name_to_id = {name: i for i, name in enumerate(names)}
        y = torch.tensor([name_to_id[str(v)] for v in raw["label_list"]],
                         dtype=torch.long)
        edge = raw["edge_index"].long()
    else:
        x = raw.x.float()
        texts = list(raw.raw_texts)
        y = raw.y.long()
        edge = raw.edge_index.long()
    return x, texts, y, edge


def induced_edges(edge_index: torch.Tensor, ids: torch.Tensor,
                  total: int) -> torch.Tensor:
    remap = torch.full((int(total),), -1, dtype=torch.long)
    remap[ids] = torch.arange(len(ids), dtype=torch.long)
    local = remap[edge_index]
    return local[:, (local >= 0).all(0)].contiguous()


def make_delta(raw, text_emb: torch.Tensor, ids: torch.Tensor,
               stage: int) -> dict:
    x, texts, y, edge = raw_fields(raw)
    return {
        "x": x[ids],
        "text_emb": text_emb[ids].float(),
        "texts": [str(texts[int(i)]) for i in ids],
        "class_id": y[ids].long(),
        "mass": torch.ones(len(ids)),
        "depth": torch.zeros(len(ids), dtype=torch.long),
        "stage": torch.full((len(ids),), int(stage), dtype=torch.long),
        "edge_index": induced_edges(edge, ids, len(x)),
    }


def concat_support(old: dict, delta: dict) -> dict:
    n = len(old["x"])
    return {
        "x": torch.cat((old["x"], delta["x"])),
        "text_emb": torch.cat((old["text_emb"], delta["text_emb"])),
        "texts": list(old["texts"]) + list(delta["texts"]),
        "class_id": torch.cat((torch.full((n,), -1, dtype=torch.long),
                                delta["class_id"])),
        "mass": torch.cat((old["mass"], delta["mass"])),
        "depth": torch.cat((old["depth"], delta["depth"])),
        "stage": torch.cat((old["stage"], delta["stage"])),
        "edge_index": torch.cat((old["edge_index"], delta["edge_index"] + n),
                                 dim=1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("cora", "citeseer"), required=True)
    parser.add_argument("--order", required=True)
    parser.add_argument("--stage", type=int, choices=(1, 2), required=True)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--text-cache", required=True, type=Path)
    parser.add_argument("--c0-root", required=True, type=Path)
    parser.add_argument("--c0-file", type=Path, default=None,
                        help="Stage-0 parent memory file; overrides --c0-root for stage 1.")
    parser.add_argument("--semantic-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--teacher-steps", type=int, default=4)
    parser.add_argument("--inner-steps", type=int, default=2)
    parser.add_argument("--outer-steps", type=int, default=4)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--top-r", type=int, default=8)
    parser.add_argument("--teacher-seeds", nargs="+", type=int, default=[9300, 9301])
    parser.add_argument("--validation-teacher-seeds", nargs="+", type=int, default=[9310])
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"immutable output exists: {args.output}")

    raw = load(args.data)
    x, _, _, _ = raw_fields(raw)
    text_emb = load(args.text_cache)
    if len(text_emb) != len(x):
        raise ValueError("text cache is not node aligned")
    groups = CAG.stage_groups(args.dataset, 3, args.order)
    _, _, _, pool = protocol_v3.split_masks_v4(
        torch.arange(len(x)), protocol_v3.REPORT_DEV_MOD)
    ids = torch.where(pool & torch.isin(
        raw_fields(raw)[2], torch.tensor(groups[args.stage])))[0].long()
    current = make_delta(raw, text_emb, ids, args.stage)
    current_cell = args.semantic_root / f"{args.dataset}_{args.order}_s{args.stage}"
    parent_cell = args.semantic_root / f"{args.dataset}_{args.order}_s{args.stage - 1}"
    base = load(current_cell / "semantic_perm_memory.pt")
    if args.stage == 1:
        parent_path = args.c0_file
        if parent_path is None:
            parent_path = args.c0_root / args.dataset / args.order / "c0_s0_memory.pt"
        parent = load(parent_path)
    else:
        parent_path = parent_cell / "semantic_perm_memory.pt"
        parent = load(parent_cell / "semantic_perm_memory.pt")
    old_rows = len(parent["x"])
    if len(base["x"]) <= old_rows:
        raise ValueError("Semantic-Recursive base has no new rows")
    support = concat_support(parent, current)
    cfg = TGR.Config(
        teacher_steps=args.teacher_steps,
        inner_steps=args.inner_steps,
        outer_steps=args.outer_steps,
        batch=args.batch,
        top_r=args.top_r,
        teacher_seeds=tuple(args.teacher_seeds),
        validation_teacher_seeds=tuple(args.validation_teacher_seeds),
    )
    result, audit, assignment = TGR.optimize(
        base, support, support["class_id"], old_rows, args.stage,
        args.device, cfg)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(support, args.output / "support.pt")
    torch.save(result, args.output / "tgr_memory.pt")
    (args.output / "assignment.json").write_text(
        json.dumps(assignment, indent=2, ensure_ascii=True), encoding="utf-8")
    (args.output / "audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=True), encoding="utf-8")
    (args.output / "protocol.json").write_text(json.dumps({
        "name": "teacher-guided-payload-revision-v1",
        "dataset": args.dataset, "order": args.order, "stage": args.stage,
        "base": str(current_cell / "semantic_perm_memory.pt"),
        "parent": str(parent_path),
        "current_delta_rows": int(len(ids)), "base_rows": int(len(base["x"])),
        "old_rows": int(old_rows), "new_rows": int(len(base["x"]) - old_rows),
        "teacher": "canonical two-layer GCN + bidirectional InfoNCE trained on legal support",
        "assignment": "sparse top-r joint graph-text soft P with class-consistency penalty",
        "optimized": ["X", "P", "short Student trajectory"],
        "fixed": ["text_emb", "texts", "budget", "edge_index", "Student architecture"],
        "historical_raw_reads": "current delta only",
        "report_labels_used": False,
    }, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps({"event": "teacher_guided_payload_revision_complete",
                      "dataset": args.dataset, "order": args.order,
                      "stage": args.stage, "output": str(args.output),
                      "selected_outer_step": audit["selected_outer_step"],
                      "max_relative_displacement": audit["max_relative_displacement"],
                      "top_r": assignment["top_r"]}), flush=True)


if __name__ == "__main__":
    main()
