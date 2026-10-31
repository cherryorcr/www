"""Attach construction-time new-class metadata to a sealed direct-outer support.

The support tensor intentionally contains no labels.  This helper creates a
separate audit artifact containing only the current incoming delta class ids
and the public class-prompt embeddings needed by the prompt-aware outer loss.
It does not modify the sealed memory or support tensors.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import protocol_v3
import run_content_gap_attribution as CAG
import run_unified_v3_continual as U


def load(path: Path):
    return torch.load(path, map_location="cpu", weights_only=False)


def raw_fields(raw: dict):
    if isinstance(raw, dict):
        x = raw["node_f"].float()
        texts = list(raw.get("text_list", raw.get("texts", [])))
        names = [str(value) for value in raw["all_labels"]]
        lookup = {name: index for index, name in enumerate(names)}
        y = torch.tensor([lookup[str(value)] for value in raw["label_list"]],
                         dtype=torch.long)
    else:
        x = raw.x.float()
        texts = list(raw.raw_texts)
        y = raw.y.long()
    return x, texts, y


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("cora", "citeseer"), required=True)
    parser.add_argument("--order", required=True)
    parser.add_argument("--stage", type=int, required=True)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--text-cache", required=True, type=Path)
    parser.add_argument("--support", required=True, type=Path)
    parser.add_argument("--old-rows", required=True, type=int)
    parser.add_argument("--minilm", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"immutable output exists: {args.output}")

    support = load(args.support)
    raw = load(args.data)
    text_cache = load(args.text_cache)
    x, _, y = raw_fields(raw)
    groups = CAG.stage_groups(args.dataset, 3, args.order)
    _, _, _, pool = protocol_v3.split_masks_v4(
        torch.arange(len(y)), protocol_v3.REPORT_DEV_MOD)
    classes = torch.tensor(groups[args.stage], dtype=torch.long)
    expected = len(support["x"]) - int(args.old_rows)
    ids = torch.where(pool & torch.isin(y, classes))[0].long()
    selection_mode = "stage_pool_filter"
    if len(ids) != expected:
        # Some sealed cells were prepared by an older order-specific selector.
        # Recover labels from the actual support rows, matching both payload
        # tensors so a duplicate X row cannot silently receive a wrong class.
        if len(text_cache) != len(x):
            raise ValueError("text cache is not aligned to raw rows")
        lookup = {}
        for index in range(len(x)):
            key = (x[index].contiguous().numpy().tobytes()
                   + text_cache[index].contiguous().numpy().tobytes())
            lookup.setdefault(key, []).append(index)
        matched = []
        for index in range(int(args.old_rows), len(support["x"])):
            key = (support["x"][index].contiguous().numpy().tobytes()
                   + support["text_emb"][index].contiguous().numpy().tobytes())
            candidates = lookup.get(key, [])
            if len(candidates) != 1:
                raise ValueError(
                    f"support row {index} has {len(candidates)} raw matches")
            matched.append(candidates[0])
        ids = torch.tensor(matched, dtype=torch.long)
        selection_mode = "support_payload_exact_match"
    class_id = torch.full((len(support["x"]),), -1, dtype=torch.long)
    class_id[int(args.old_rows):] = y[ids]

    unified = U.load_unified(args.dataset, str(args.data), args.minilm, "cpu",
                             n_stages=3, split_mode="canonical")
    prompt_emb = unified["encoder"].encode(unified["prompts"]).cpu()
    prompt_ids = torch.arange(len(prompt_emb), dtype=torch.long)

    args.output.mkdir(parents=True, exist_ok=False)
    torch.save(class_id, args.output / "support_class_id.pt")
    torch.save(prompt_emb, args.output / "prompt_emb.pt")
    torch.save(prompt_ids, args.output / "prompt_ids.pt")
    (args.output / "audit.json").write_text(json.dumps({
        "name": "direct-student-prompt-support-v1",
        "dataset": args.dataset,
        "order": args.order,
        "stage": int(args.stage),
        "old_rows": int(args.old_rows),
        "current_delta_rows": int(len(ids)),
        "selection_mode": selection_mode,
        "current_raw_ids": [int(value) for value in ids],
        "incoming_classes": [int(value) for value in classes],
        "support_class_id_negative_old_rows": int((class_id < 0).sum()),
        "prompt_count": int(len(prompt_emb)),
        "source": "current stage class protocol plus public class prompts",
        "report_labels_used": False,
        "historical_raw_reads": 0,
    }, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps({
        "event": "direct_student_prompt_support_complete",
        "output": str(args.output),
        "current_delta_rows": int(len(ids)),
        "incoming_classes": [int(value) for value in classes],
    }), flush=True)


if __name__ == "__main__":
    main()
