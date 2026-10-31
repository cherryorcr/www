"""Prepare and run a small GC-SR stage smoke from local sealed artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import gc_sr_gradient as GC
import protocol_v3


def induced_edges(edge_index: torch.Tensor, ids: torch.Tensor, total: int) -> torch.Tensor:
    remap = torch.full((int(total),), -1, dtype=torch.long)
    remap[ids] = torch.arange(len(ids), dtype=torch.long)
    edge = edge_index.long()
    local = remap[edge]
    return local[:, (local >= 0).all(0)].contiguous()


def load(path: Path):
    return torch.load(path, map_location="cpu", weights_only=False)


def delta(data: dict, text_emb: torch.Tensor, ids: torch.Tensor, stage: int) -> dict:
    x = data["x"] if isinstance(data, dict) else data.x
    texts = (data.get("texts", data.get("text_list"))
             if isinstance(data, dict) else list(data.raw_texts))
    edges = data["edge_index"] if isinstance(data, dict) else data.edge_index
    return {
        "x": x[ids].float(),
        "text_emb": text_emb[ids].float(),
        "texts": [str(texts[int(i)]) for i in ids],
        "mass": torch.ones(len(ids)),
        "depth": torch.zeros(len(ids), dtype=torch.long),
        "stage": torch.full((len(ids),), int(stage), dtype=torch.long),
        "edge_index": induced_edges(edges, ids, len(x)),
    }


def concat_support(old: dict, current: dict) -> dict:
    old_n = len(old["x"])
    edge = torch.cat((old["edge_index"],
                      current["edge_index"] + old_n), dim=1)
    return {
        "x": torch.cat((old["x"], current["x"])),
        "text_emb": torch.cat((old["text_emb"], current["text_emb"])),
        "texts": list(old["texts"]) + list(current["texts"]),
        "mass": torch.cat((old["mass"], current["mass"])),
        "depth": torch.cat((old["depth"], current["depth"])),
        "stage": torch.cat((old["stage"], current["stage"])),
        "edge_index": edge,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", required=True, type=Path)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--text-cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-new-rows", type=int, default=4)
    parser.add_argument("--max-old-rows", type=int, default=2)
    parser.add_argument("--outer-steps", type=int, default=1)
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"immutable output exists: {args.output}")
    manifest = json.loads((args.cell / "manifest.json").read_text(encoding="utf-8"))
    raw = load(args.data)
    text_emb = load(args.text_cache)
    raw_x = raw["node_f"] if isinstance(raw, dict) else raw.x
    if not isinstance(text_emb, torch.Tensor) or len(text_emb) != len(raw_x):
        raise ValueError("text cache must be a full node-aligned tensor")
    old = load(args.cell / f"c0_s{args.stage - 1}_memory.pt")
    base = load(args.cell / f"c0_s{args.stage}_memory.pt")
    if "stage_source_ids" in manifest:
        ids = torch.tensor(manifest["stage_source_ids"][args.stage], dtype=torch.long)
    else:
        names = ([str(v) for v in raw["all_labels"]]
                 if isinstance(raw, dict) else [str(v) for v in sorted(set(raw.y.tolist()))])
        label_to_id = {name: i for i, name in enumerate(names)}
        y = (torch.tensor([label_to_id[str(v)] for v in raw["label_list"]],
                          dtype=torch.long)
             if isinstance(raw, dict) else raw.y.long())
        _, _, _, pool = protocol_v3.split_masks_v4(
            torch.arange(len(y)), protocol_v3.REPORT_DEV_MOD)
        classes = torch.tensor(manifest["stage_classes"][args.stage], dtype=torch.long)
        ids = torch.where(pool & torch.isin(y, classes))[0]
    current = delta(raw, text_emb, ids, args.stage)
    support = concat_support(old, current)
    old_n = len(old["x"])
    new_n = len(base["x"]) - old_n
    if new_n <= 0:
        raise ValueError("current base contains no appended rows")
    editable = list(range(old_n, old_n + min(new_n, args.max_new_rows)))
    editable += list(range(min(old_n, args.max_old_rows)))
    cfg = GC.Config(outer_steps=args.outer_steps, eval_every=1, batch=args.batch,
                    train_seeds=(9100,), validation_seeds=(9110,),
                    gradient_weight=1.0, output_weight=0.0,
                    old_anchor_weight=0.05)
    diagnostic = GC.diagnostic(base, support, args.device, cfg)
    result, audit = GC.optimize(base, support, editable, old_n, args.stage,
                                args.device, cfg)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(support, args.output / "support.pt")
    torch.save(result, args.output / "gc_sr_memory.pt")
    (args.output / "diagnostic.json").write_text(
        json.dumps(diagnostic, indent=2, ensure_ascii=True), encoding="utf-8")
    (args.output / "audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=True), encoding="utf-8")
    (args.output / "protocol.json").write_text(json.dumps({
        "name": "gc-sr-stage-smoke-v1", "cell": str(args.cell),
        "stage": args.stage, "base": f"c0_s{args.stage}_memory.pt",
        "parent": f"c0_s{args.stage - 1}_memory.pt",
        "current_delta_rows": int(len(ids)), "base_rows": int(len(base["x"])),
        "old_rows": int(old_n), "editable_rows": editable,
        "text": "fixed", "budget": "fixed", "topology": "fixed",
        "student": "canonical GCN + bidirectional InfoNCE",
        "historical_raw_reads": "current delta only; no historical raw source",
        "report_labels_used": False,
    }, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps({"event": "gc_sr_stage_smoke_complete",
                      "output": str(args.output),
                      "editable_rows": editable,
                      "diagnostic_mean_gradient_cosine": diagnostic["mean_gradient_cosine"],
                      "max_relative_displacement": audit["max_relative_displacement"]}),
          flush=True)


if __name__ == "__main__":
    main()
