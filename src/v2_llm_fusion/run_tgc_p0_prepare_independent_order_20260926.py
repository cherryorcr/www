"""Prepare a frozen C0 parent for P0 independent-order confirmation.

This script only rebuilds the existing C0 ``dispersion + append_fixed``
constructor under a pre-registered class order.  It does not change the
semantic-recursive method or train a student.  The resulting manifest is
consumed by ``run_tgc_semantic_recursive_large_20260926.py``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import run_student_compatible_append as SCA
import run_unified_v3_continual as U


FIELDS = ("x", "text_emb", "mass", "depth", "stage")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True), encoding="utf-8")


def fresh_groups(groups: list[list[int]], seed: int) -> list[list[int]]:
    flat = sorted(int(cls) for group in groups for cls in group)
    if len(set(flat)) != len(flat) or any(not group for group in groups):
        raise ValueError("invalid reference class partition")
    shuffled = np.random.default_rng(int(seed)).permutation(flat).tolist()
    offsets = np.cumsum([0] + [len(group) for group in groups]).tolist()
    return [sorted(shuffled[offsets[i]:offsets[i + 1]])
            for i in range(len(groups))]


def clone_block(block: dict) -> dict:
    return {
        key: (value.clone() if isinstance(value, torch.Tensor)
              else list(value) if isinstance(value, list) else value)
        for key, value in block.items()
        if key in FIELDS or key == "texts"
    }


def memory_digest(memory: dict) -> str:
    digest = hashlib.sha256()
    for key in ("x", "text_emb", "mass", "depth", "stage", "edge_index"):
        value = memory[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(value.numpy().tobytes())
    for text in memory["texts"]:
        encoded = str(text).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def run(args: argparse.Namespace) -> dict:
    canonical_root = Path(args.canonical_root)
    canonical_manifest = json.loads(
        (canonical_root / "results.json").read_text(encoding="utf-8"))
    if canonical_manifest.get("arms") != ["append_fixed"]:
        raise ValueError("canonical root must be the frozen append_fixed C0 source")
    if int(canonical_manifest.get("n_stages", 0)) != 5:
        raise ValueError("P0 requires the frozen base+4 five-stage protocol")

    data = U.load_unified(
        args.dataset,
        args.data,
        args.minilm,
        args.device,
        n_stages=5,
        split_mode="base_incremental",
        base_fraction=0.4,
        report_dev_mod=int(canonical_manifest["report_dev_mod"]),
    )
    cache = canonical_root / "text_emb_pool_only.pt"
    if not cache.exists():
        raise FileNotFoundError(cache)
    data["text_emb"] = load(cache).float()
    if len(data["text_emb"]) != len(data["x"]):
        raise AssertionError("text embedding cache has an invalid row count")
    if bool(data["text_emb"][~data["pool_mask"]].abs().any()):
        raise AssertionError("holdout text rows were encoded")

    reference_groups = [[int(v) for v in group]
                        for group in canonical_manifest["stage_classes"]]
    groups = fresh_groups(reference_groups, args.order_seed)
    data["stage_classes"] = groups
    U.G.STAGE_CLASSES = groups

    ids_by_stage = []
    stage_counts = []
    for classes in groups:
        ids = torch.where(
            data["pool_mask"] & torch.isin(
                data["y"], torch.tensor(classes, dtype=torch.long))
        )[0]
        ids = torch.sort(ids).values
        ids_by_stage.append(ids)
        stage_counts.append(int(len(ids)))

    budgets = []
    cumulative = 0
    for count in stage_counts:
        cumulative += int(count)
        budgets.append(int(U.protocol_v3.growing_budget(
            cumulative, rate=float(canonical_manifest["rate"]))))

    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    records = []
    previous = None
    memory_hashes = []
    for stage, (classes, ids, budget) in enumerate(
            zip(groups, ids_by_stage, budgets)):
        keep = 0 if previous is None else len(previous["x"])
        new_rows = int(budget - keep)
        if new_rows <= 0:
            raise AssertionError(f"non-growing independent budget at stage {stage}")
        block = U.build_stage_block(
            data,
            ids,
            new_rows,
            data["encoder"],
            stage,
            "joint_g5_c320",
            "concat",
            args.device,
            "dispersion",
            classes,
            0.5,
            0.5,
            x_operator="mean",
        )
        plain = clone_block(block)
        memory = SCA.append_memory(
            previous,
            plain,
            keep,
            topology_mode="frozen_old",
            cross_degree=1,
            refresh_mode="frozen_old",
            device=args.device,
        )
        if len(memory["x"]) != budget:
            raise AssertionError(f"budget mismatch at stage {stage}")
        if previous is not None:
            for key in FIELDS:
                if not torch.equal(memory[key][:keep], previous[key]):
                    raise AssertionError(f"old prefix changed: {key}, stage {stage}")
            if memory["texts"][:keep] != previous["texts"]:
                raise AssertionError(f"old text prefix changed at stage {stage}")

        memory_path = out / f"append_fixed_s{stage}_memory.pt"
        torch.save(memory, memory_path)
        memory_hash = memory_digest(memory)
        memory_hashes.append(memory_hash)
        records.append({
            "arm": "append_fixed",
            "stage": int(stage),
            "keep_old_rows": int(keep),
            "new_rows": int(new_rows),
            "actual_new_rows": int(len(memory["x"]) - keep),
            "allocated_new_rows_by_class": {
                str(k): int(v) for k, v in block["allocation"].items()
            },
            "source_count": int(len(ids)),
            "classes": [int(v) for v in classes],
            "memory_file": memory_path.name,
            "memory_sha256": memory_hash,
            "historical_raw_reads": 0,
            "report_test_labels_used_for_construction": False,
        })
        previous = memory

    manifest = dict(canonical_manifest)
    manifest.update({
        "dataset": args.dataset,
        "data": str(args.data),
        "stage_classes": groups,
        "stage_split_source": f"p0_fresh_groups_seed_{int(args.order_seed)}",
        "stage_train_counts": stage_counts,
        "budgets": budgets,
        "results": records,
        "arms": ["append_fixed"],
        "n_stages": 5,
        "source_order_seed": int(args.order_seed),
        "construction_seed": int(args.construction_seed),
        "construction_source": "current-stage pool delta only",
        "historical_raw_reads": 0,
        "test_accessed": False,
        "stage_ids_sha256": sha256_json([ids.tolist() for ids in ids_by_stage]),
        "independent_from_development_orders": True,
    })
    manifest_path = out / "results.json"
    save_json(manifest_path, manifest)
    save_json(out / "p0_prepare_audit.json", {
        "dataset": args.dataset,
        "source_order_seed": int(args.order_seed),
        "construction_seed": int(args.construction_seed),
        "reference_stage_classes": reference_groups,
        "stage_classes": groups,
        "stage_train_counts": stage_counts,
        "budgets": budgets,
        "memory_sha256": memory_hashes,
        "historical_raw_reads": 0,
        "report_test_labels_used_for_construction": False,
        "manifest_sha256": sha256_file(manifest_path),
    })
    return {
        "dataset": args.dataset,
        "output": str(out),
        "order_seed": int(args.order_seed),
        "stage_classes": groups,
        "stage_train_counts": stage_counts,
        "budgets": budgets,
        "memory_sha256": memory_hashes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("arxiv", "bookchild", "sportsfit"), required=True)
    parser.add_argument("--canonical-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--data", default=None)
    parser.add_argument("--minilm", default="/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--order-seed", type=int, default=49013)
    parser.add_argument("--construction-seed", type=int, default=9001)
    args = parser.parse_args()
    if args.data is None:
        args.data = U.DATA_CFG[args.dataset]["path"]
    torch.set_num_threads(2)
    print(json.dumps(run(args), ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
