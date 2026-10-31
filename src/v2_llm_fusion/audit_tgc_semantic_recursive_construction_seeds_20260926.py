"""Audit construction-seed invariance for semantic-recursive P0 runs.

The current constructor is intended to be deterministic.  This audit reruns
only block construction and semantic payload assignment for several declared
construction seeds, without training a Student or selecting by F1.  Identical
stage hashes are the expected result; a difference is a protocol failure that
must be investigated before treating the large-graph gain as confirmed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import run_tgc_role_aware_assignment_20260925 as Role
import run_tgc_semantic_recursive_large_20260926 as R
import run_tgc_two_prototype_smoke as Smoke
import run_unified_v3_continual as U


def load(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True), encoding="utf-8")


def run(args: argparse.Namespace) -> dict:
    root = Path(args.c0_root)
    manifest = json.loads((root / "results.json").read_text(encoding="utf-8"))
    groups = [[int(v) for v in group] for group in manifest["stage_classes"]]
    data = U.load_unified(
        args.dataset,
        args.data,
        args.minilm,
        args.device,
        n_stages=5,
        split_mode="base_incremental",
        base_fraction=0.4,
        report_dev_mod=int(manifest["report_dev_mod"]),
    )
    data["stage_classes"] = groups
    U.G.STAGE_CLASSES = groups
    text_cache = Path(args.text_cache)
    data["text_emb"] = load(text_cache).float()
    data["prompt_emb"] = data["encoder"].encode(data["prompts"])
    ids_by_stage = []
    for classes in groups:
        ids = torch.where(
            data["pool_mask"] & torch.isin(
                data["y"], torch.tensor(classes, dtype=torch.long))
        )[0]
        ids_by_stage.append(torch.sort(ids).values)
    budgets = [int(v) for v in manifest["budgets"]]
    all_hashes = {}
    all_moves = {}
    for seed in args.construction_seeds:
        parent = load(root / "append_fixed_s0_memory.pt")
        hashes = [R.memory_sha256(parent)]
        moves = [0]
        for stage in range(1, 5):
            c0 = load(root / f"append_fixed_s{stage}_memory.pt")
            old_n = len(parent["x"])
            allocation = {
                int(key): int(value)
                for key, value in manifest["results"][stage][
                    "allocated_new_rows_by_class"].items()
            }
            q, row_classes = R.build_source_assignment(
                data, ids_by_stage[stage], allocation, args.device)
            c0_block = {
                key: (c0[key][old_n:].clone()
                      if isinstance(c0[key], torch.Tensor)
                      else list(c0[key][old_n:]))
                for key in R.PAYLOAD_KEYS
            }
            structural, audit = Smoke.build_two_prototype_block(
                data, ids_by_stage[stage], q, row_classes, stage,
                args.device, c0_block)
            mapping, _ = Role.complete_mapping(
                data, ids_by_stage[stage], q, row_classes,
                audit["records"], c0, old_n, "semantic_perm", int(seed))
            semantic_block = R.semantic_block_from_mapping(structural, mapping)
            semantic = R.candidate_from_parent(c0, parent, semantic_block)
            R.assert_candidate(c0, parent, semantic_block, semantic,
                               structural, mapping)
            hashes.append(R.memory_sha256(semantic))
            moves.append(int(sum(source != target
                                 for source, target in mapping.items())))
            parent = semantic
        all_hashes[str(seed)] = hashes
        all_moves[str(seed)] = moves

    reference_seed = str(args.construction_seeds[0])
    reference = all_hashes[reference_seed]
    invariant = all(values == reference for values in all_hashes.values())
    result = {
        "dataset": args.dataset,
        "c0_root": str(root),
        "stage_classes": groups,
        "construction_seeds": [int(v) for v in args.construction_seeds],
        "stage_memory_sha256": all_hashes,
        "mapping_moves_by_seed": all_moves,
        "bitwise_seed_invariant": bool(invariant),
        "student_trained": False,
        "f1_used_for_construction": False,
        "historical_raw_reads": 0,
    }
    save_json(Path(args.output), result)
    print(json.dumps(result, ensure_ascii=True), flush=True)
    if not invariant:
        raise SystemExit("construction seed changed semantic memory hash")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("arxiv", "bookchild", "sportsfit"), required=True)
    parser.add_argument("--c0-root", required=True)
    parser.add_argument("--text-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--data", default=None)
    parser.add_argument("--minilm", default="/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--construction-seeds", type=int, nargs="+",
                        default=[9001, 9002, 9003])
    args = parser.parse_args()
    if args.data is None:
        args.data = U.DATA_CFG[args.dataset]["path"]
    torch.set_num_threads(2)
    run(args)


if __name__ == "__main__":
    main()
