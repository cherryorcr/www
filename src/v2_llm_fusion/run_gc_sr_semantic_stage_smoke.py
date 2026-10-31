"""Run GC-SR X-only arms on a real Semantic-Recursive stage artifact."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import gc_sr_gradient as GC
import protocol_v3
import run_content_gap_attribution as CAG


def load(path: Path):
    return torch.load(path, map_location="cpu", weights_only=False)


def raw_fields(raw):
    if isinstance(raw, dict):
        x = raw["node_f"]
        texts = raw.get("text_list")
        labels = [str(v) for v in raw["all_labels"]]
        label_to_id = {name: i for i, name in enumerate(labels)}
        y = torch.tensor([label_to_id[str(v)] for v in raw["label_list"]],
                         dtype=torch.long)
        edge = raw["edge_index"]
    else:
        x = raw.x
        texts = list(raw.raw_texts)
        y = raw.y.long()
        edge = raw.edge_index
    return x.float(), list(texts), y, edge.long()


def induced_edges(edge_index: torch.Tensor, ids: torch.Tensor, total: int) -> torch.Tensor:
    remap = torch.full((int(total),), -1, dtype=torch.long)
    remap[ids] = torch.arange(len(ids), dtype=torch.long)
    local = remap[edge_index]
    return local[:, (local >= 0).all(0)].contiguous()


def make_delta(raw, text_emb: torch.Tensor, ids: torch.Tensor, stage: int) -> dict:
    x, texts, _, edge = raw_fields(raw)
    return {
        "x": x[ids],
        "text_emb": text_emb[ids].float(),
        "texts": [str(texts[int(i)]) for i in ids],
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
        "mass": torch.cat((old["mass"], delta["mass"])),
        "depth": torch.cat((old["depth"], delta["depth"])),
        "stage": torch.cat((old["stage"], delta["stage"])),
        "edge_index": torch.cat((old["edge_index"], delta["edge_index"] + n), dim=1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("cora", "citeseer"), required=True)
    parser.add_argument("--order", required=True)
    parser.add_argument("--stage", type=int, choices=(1, 2), required=True)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--text-cache", required=True, type=Path)
    parser.add_argument("--c0-root", required=True, type=Path)
    parser.add_argument("--semantic-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", choices=("output", "gradient", "joint", "gradient_anchor", "joint_new", "new_only"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-rows", type=int, default=8)
    parser.add_argument("--max-old-rows", type=int, default=4)
    parser.add_argument("--selection-mode", choices=("first", "sensitivity", "all", "new"), default="all")
    parser.add_argument("--outer-steps", type=int, default=4)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--train-seeds", nargs="+", type=int, default=[9100])
    parser.add_argument("--validation-seeds", nargs="+", type=int, default=[9110])
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"immutable output exists: {args.output}")
    raw = load(args.data)
    x, _, y, _ = raw_fields(raw)
    text_emb = load(args.text_cache)
    if len(text_emb) != len(x):
        raise ValueError("text cache is not node aligned")
    groups = CAG.stage_groups(args.dataset, 3, args.order)
    _, _, _, pool = protocol_v3.split_masks_v4(torch.arange(len(x)), protocol_v3.REPORT_DEV_MOD)
    ids = torch.where(pool & torch.isin(y, torch.tensor(groups[args.stage])))[0].long()
    current = make_delta(raw, text_emb, ids, args.stage)
    current_cell = args.semantic_root / f"{args.dataset}_{args.order}_s{args.stage}"
    parent_cell = args.semantic_root / f"{args.dataset}_{args.order}_s{args.stage - 1}"
    base = load(current_cell / "semantic_perm_memory.pt")
    if args.stage == 1:
        parent = load(args.c0_root / args.dataset / args.order / "c0_s0_memory.pt")
    else:
        parent = load(parent_cell / "semantic_perm_memory.pt")
    support = concat_support(parent, current)
    old_n = len(parent["x"])
    new_n = len(base["x"]) - old_n
    if new_n <= 0 or len(base["x"]) != len(support["x"][:old_n]) + new_n:
        raise ValueError("semantic base row accounting is invalid")
    if args.selection_mode == "all":
        editable = list(range(len(base["x"])))
        selection_audit = {"mode": "all_x_rows", "selected_rows": editable}
    elif args.selection_mode == "new":
        editable = list(range(old_n, len(base["x"])))
        selection_audit = {"mode": "all_new_rows_only", "selected_rows": editable}
    elif args.selection_mode == "sensitivity":
        editable, selection_audit = GC.select_editable_rows(
            base, old_n, args.max_old_rows, args.max_new_rows,
            args.device, seed=9120, batch=0)
    else:
        editable = list(range(old_n, old_n + min(new_n, args.max_new_rows)))
        editable += list(range(min(old_n, args.max_old_rows)))
        selection_audit = {"mode": "first_rows", "selected_rows": editable}
    weights = {"output": (0.0, 1.0, 0.0, 0.0, 0.0),
               "gradient": (1.0, 0.0, 0.0, 0.0, 0.0),
               "joint": (1.0, 1.0, 0.0, 0.0, 0.0),
               "gradient_anchor": (1.0, 0.0, 0.0, 0.0, 0.05),
               "joint_new": (1.0, 1.0, 1.0, 0.25, 0.0)}
    weights["new_only"] = (0.0, 0.0, 1.0, 0.0, 0.0)
    gw, ow, nw, oldw, aw = weights[args.mode]
    cfg = GC.Config(outer_steps=args.outer_steps, eval_every=1, batch=args.batch,
                    train_seeds=tuple(args.train_seeds),
                    validation_seeds=tuple(args.validation_seeds),
                    gradient_weight=gw, output_weight=ow,
                    new_gradient_weight=nw, old_gradient_weight=oldw,
                    old_anchor_weight=aw)
    diagnostic = GC.diagnostic(base, support, args.device, cfg)
    result, audit = GC.optimize(base, support, editable, old_n, args.stage,
                                args.device, cfg)
    audit["selection"] = selection_audit
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(support, args.output / "support.pt")
    torch.save(result, args.output / "gc_sr_memory.pt")
    (args.output / "diagnostic.json").write_text(json.dumps(diagnostic, indent=2, ensure_ascii=True), encoding="utf-8")
    (args.output / "audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=True), encoding="utf-8")
    (args.output / "protocol.json").write_text(json.dumps({
        "name": "gc-sr-semantic-stage-smoke-v1", "dataset": args.dataset,
        "order": args.order, "stage": args.stage, "mode": args.mode,
        "base": str(current_cell / "semantic_perm_memory.pt"),
        "parent": str(parent), "current_delta_rows": int(len(ids)),
        "base_rows": int(len(base["x"])), "old_rows": int(old_n),
        "editable_rows": editable, "text": "fixed", "budget": "fixed",
        "topology": "fixed semantic-recursive/C0 topology",
        "student": "canonical two-layer GCN + bidirectional InfoNCE",
        "historical_raw_reads": "current delta only", "report_labels_used": False,
    }, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps({"event": "gc_sr_semantic_stage_smoke_complete",
                      "dataset": args.dataset, "order": args.order,
                      "stage": args.stage, "mode": args.mode,
                      "output": str(args.output),
                      "diagnostic_mean_gradient_cosine": diagnostic["mean_gradient_cosine"],
                      "max_relative_displacement": audit["max_relative_displacement"]}), flush=True)


if __name__ == "__main__":
    main()
