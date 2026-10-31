"""Run the recursive semantic-payload candidate on the frozen large-graph C0.

The existing P5 ``append_fixed`` artifacts are the only C0 source.  At each
incremental stage this runner reconstructs a deterministic source assignment
from the current delta, applies the already audited two-prototype proposal,
then performs class-local semantic payload assignment.  The resulting block
is appended to the previous *semantic* memory, while the current C0
``edge_index`` is reused exactly.

This is an evaluation runner, not a new selector.  It keeps the original
two-layer GCN student, bidirectional InfoNCE objective, fixed-rate budget,
batch and report seeds.  Historical raw nodes are never used to construct a
later block; only the current stage ids are read for construction.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import torch

import run_student_compatible_append as SCA
import run_tgc_role_aware_assignment_20260925 as Role
import run_tgc_two_prototype_smoke as Smoke
import run_unified_v3_continual as U
from semantic_payload_transport import apply_payload_permutation


PAYLOAD_KEYS = ("x", "text_emb", "texts", "mass", "depth", "stage")


def load(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True), encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def memory_sha256(memory: dict) -> str:
    digest = hashlib.sha256()
    for key in ("x", "text_emb", "mass", "depth", "stage", "edge_index"):
        value = memory[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(value.numpy().tobytes())
    for value in memory["texts"]:
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def clone_value(value):
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, list):
        return list(value)
    if isinstance(value, dict):
        return {key: clone_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(clone_value(item) for item in value)
    return copy.deepcopy(value)


def clone_memory(memory: dict) -> dict:
    return {key: clone_value(value) for key, value in memory.items()}


def row_digest(block: dict, index: int) -> str:
    digest = hashlib.sha256()
    for key in ("x", "text_emb", "mass", "depth", "stage"):
        value = block[key][index].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(value.numpy().tobytes())
    digest.update(str(block["texts"][index]).encode("utf-8"))
    return digest.hexdigest()


def set_pool_text_embeddings(data: dict, c0_root: Path, output: Path) -> dict:
    """Load the sealed pool-only embedding cache used by the C0 run."""
    cache = c0_root / "text_emb_pool_only.pt"
    if not cache.exists():
        cache = output / "text_emb_pool_only.pt"
    if cache.exists():
        text_emb = torch.load(cache, map_location="cpu", weights_only=False)
    else:
        pool_ids = torch.where(data["pool_mask"])[0]
        text_emb = torch.zeros((len(data["x"]), 384), dtype=torch.float32)
        text_emb[pool_ids] = data["encoder"].encode(
            [data["texts"][int(i)] for i in pool_ids]
        )
        torch.save(text_emb, cache)
    expected = (len(data["x"]), 384)
    if tuple(text_emb.shape) != expected:
        raise ValueError(f"invalid text cache shape {tuple(text_emb.shape)} != {expected}")
    excluded = ~data["pool_mask"]
    if bool(excluded.any()) and float(text_emb[excluded].abs().max()) != 0.0:
        raise AssertionError("construction text cache contains sealed rows")
    data["text_emb"] = text_emb.float()
    return data


def stage_ids(data: dict) -> list[torch.Tensor]:
    result = []
    for classes in data["stage_classes"]:
        class_tensor = torch.tensor(classes, dtype=torch.long)
        ids = torch.where(
            torch.isin(data["y"], class_tensor) & data["pool_mask"]
        )[0]
        result.append(torch.sort(ids).values)
    return result


def build_source_assignment(data: dict, ids: torch.Tensor,
                            allocation: dict[int, int], device: str
                            ) -> tuple[torch.Tensor, list[int]]:
    """Recreate the C0 class-local cluster slots from the current delta."""
    assignment = torch.full((len(ids),), -1, dtype=torch.long)
    row_classes: list[int] = []
    offset = 0
    for cls in sorted(int(v) for v in allocation):
        local_positions = torch.where(data["y"][ids] == int(cls))[0]
        rows = int(allocation[cls])
        if rows <= 0:
            continue
        if len(local_positions) == 0 or rows > len(local_positions):
            raise ValueError(
                f"invalid class allocation class={cls} rows={rows} "
                f"available={len(local_positions)}"
            )
        points = torch.cat((
            torch.nn.functional.normalize(data["x"][ids[local_positions]].float(), dim=1),
            data["text_emb"][ids[local_positions]].float(),
        ), dim=1)
        local_q = SCA.cluster_assign(
            points, torch.ones(len(points), dtype=torch.float32), rows,
            device=device,
        ).cpu()
        unique = sorted(set(int(v) for v in local_q.tolist()))
        if unique != list(range(rows)):
            raise AssertionError(
                f"non-exhaustive source assignment class={cls}: {unique} != {list(range(rows))}"
            )
        assignment[local_positions] = local_q + offset
        row_classes.extend([int(cls)] * rows)
        offset += rows
    if bool((assignment < 0).any()):
        raise AssertionError("source assignment left delta rows unassigned")
    if offset != len(row_classes):
        raise AssertionError("row class count mismatch")
    return assignment, row_classes


def permute_block(structural: dict, mapping: dict[int, int]) -> tuple[dict, list[int]]:
    block = clone_memory(structural)
    records = [None] * len(structural["x"])
    # Structural records are attached by the caller; this helper only moves
    # payload.  A separate audit stores the exact mapping.
    for source, target in mapping.items():
        for key in PAYLOAD_KEYS:
            block[key][target] = structural[key][source]
    return block, [int(v) for v in mapping.values()]


def candidate_from_parent(c0: dict, parent: dict, block: dict) -> dict:
    """Combine the current C0 topology with a recursive semantic prefix."""
    old_n = len(parent["x"])
    if len(c0["x"]) != old_n + len(block["x"]):
        raise AssertionError("C0 and parent/block row counts are incompatible")
    out = clone_memory(c0)
    for key in PAYLOAD_KEYS:
        if isinstance(out[key], torch.Tensor):
            out[key][:old_n] = parent[key]
            out[key][old_n:] = block[key]
        else:
            out[key] = list(parent[key]) + list(block[key])
    if not torch.equal(out["edge_index"], c0["edge_index"]):
        raise AssertionError("candidate changed the frozen C0 topology")
    return out


def semantic_block_from_mapping(structural: dict, mapping: dict[int, int]) -> dict:
    return apply_payload_permutation(structural, mapping)


def assert_candidate(c0: dict, parent: dict, block: dict,
                     candidate: dict, structural: dict,
                     mapping: dict[int, int]) -> None:
    old_n = len(parent["x"])
    if not torch.equal(candidate["edge_index"], c0["edge_index"]):
        raise AssertionError("topology mismatch")
    for key in ("x", "text_emb", "mass", "depth", "stage"):
        if not torch.equal(candidate[key][:old_n], parent[key]):
            raise AssertionError(f"recursive prefix changed: {key}")
    if candidate["texts"][:old_n] != parent["texts"]:
        raise AssertionError("recursive text prefix changed")
    if set(mapping) != set(range(len(block["x"]))) or len(set(mapping.values())) != len(block["x"]):
        raise AssertionError("semantic mapping is not a permutation")
    left = sorted(row_digest(structural, i) for i in range(len(structural["x"])))
    right = sorted(row_digest(candidate, old_n + i) for i in range(len(block["x"])))
    if left != right:
        raise AssertionError("semantic assignment did not conserve payload rows")


def load_manifest(c0_root: Path) -> dict:
    manifest_path = c0_root / "results.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("arms") != ["append_fixed"]:
        raise ValueError("large C0 source must be the frozen append_fixed arm")
    if int(manifest.get("n_stages", 0)) != 5:
        raise ValueError("this runner requires the frozen base+4 five-stage protocol")
    return manifest


def run_dataset(args: argparse.Namespace) -> dict:
    c0_root = Path(args.c0_root)
    out = Path(args.output_root) / f"{args.dataset}_base4"
    out.mkdir(parents=True, exist_ok=True)
    final_path = out / "results.json"
    if final_path.exists():
        raise FileExistsError(f"immutable output exists: {final_path}")

    source_manifest = load_manifest(c0_root)
    data = U.load_unified(
        args.dataset, args.data, args.minilm, args.device,
        n_stages=5, split_mode="base_incremental", base_fraction=0.4,
        report_dev_mod=int(source_manifest["report_dev_mod"]),
    )
    source_groups = [[int(v) for v in group]
                     for group in source_manifest["stage_classes"]]
    if data["stage_classes"] != source_groups:
        # P0 uses a pre-registered independent order.  The data loader still
        # supplies masks and prompts; only the legal class stream is replaced
        # by the frozen C0 manifest used for this run.
        data["stage_classes"] = source_groups
        U.G.STAGE_CLASSES = source_groups
    data = set_pool_text_embeddings(data, c0_root, out)
    data["prompt_emb"] = data["encoder"].encode(data["prompts"])
    data["real_adj"] = U.G.norm_adj(
        data["edge_index"], len(data["x"]), args.device
    )
    data["x_device"] = data["x"].to(args.device)
    ids_by_stage = stage_ids(data)
    expected_counts = [int(v) for v in source_manifest["stage_train_counts"]]
    if [len(v) for v in ids_by_stage] != expected_counts:
        raise AssertionError("base+4 source counts differ from C0 manifest")
    budgets = [int(v) for v in source_manifest["budgets"]]
    last_stage = 4 if args.max_stage is None else int(args.max_stage)
    if not 0 <= last_stage <= 4:
        raise ValueError("max-stage must be between 0 and 4")
    student_args = SimpleNamespace(
        student_seeds=[int(v) for v in args.student_seeds],
        student_batch=int(args.student_batch),
        student_steps=None,
    )

    protocol = {
        "name": "tgc-semantic-permutation-recursive-large-v1",
        "dataset": args.dataset,
        "source_c0_root": str(c0_root),
        "source_c0_protocol": source_manifest["protocol"],
        "stage_protocol": "base+4; same frozen C0 class split and fixed-rate budgets",
        "stage_classes": source_groups,
        "arms": ["c0", "semantic_recursive"],
        "candidate": "two-prototype structural proposal + class-local semantic conserved payload assignment",
        "parent_rule": "stage0 C0; stage t uses semantic_recursive stage t-1",
        "topology": "exact current-stage append_fixed C0 edge_index",
        "student": "original two-layer GCN + trainable text projection + bidirectional InfoNCE",
        "student_batch": int(args.student_batch),
        "student_seeds": [int(v) for v in args.student_seeds],
        "student_steps": "protocol_v3.student_steps(rows)",
        "memory_rate": source_manifest["rate"],
        "report_dev_mod": source_manifest["report_dev_mod"],
        "construction_source": "current stage pool delta only; historical raw rows inaccessible to block construction",
        "historical_raw_reads": 0,
        "report_test_labels_used_for_construction": False,
        "construction_seed": int(args.random_seed),
    }
    save_json(out / "protocol.json", protocol)

    previous_c0 = None
    parent = None
    rows = []
    for stage in range(last_stage + 1):
        c0_path = c0_root / f"append_fixed_s{stage}_memory.pt"
        c0 = load(c0_path)
        if int(len(c0["x"])) != budgets[stage]:
            raise AssertionError(f"C0 budget mismatch at stage {stage}")
        if stage == 0:
            parent = clone_memory(c0)
            semantic = clone_memory(c0)
            semantic_audit = {
                "stage": 0,
                "parent_arm": "c0",
                "same_c0_topology": True,
                "same_parent_prefix": True,
                "changed_rows": 0,
                "mapping_moves": 0,
                "historical_raw_reads": 0,
            }
            c0_metrics = U.run_students(c0, stage, data, student_args, args.device)
            semantic_metrics = c0_metrics
        else:
            if previous_c0 is None or parent is None:
                raise AssertionError("missing previous parent")
            old_n = len(parent["x"])
            if len(c0["x"]) <= old_n:
                raise AssertionError("large C0 budget did not grow")
            for key in PAYLOAD_KEYS:
                if key == "texts":
                    if c0[key][:len(previous_c0["x"])] != previous_c0[key]:
                        raise AssertionError("C0 old text prefix changed")
                elif not torch.equal(c0[key][:len(previous_c0["x"])], previous_c0[key]):
                    raise AssertionError(f"C0 old prefix changed at stage {stage}: {key}")
            allocation = {
                int(key): int(value)
                for key, value in source_manifest["results"][stage][
                    "allocated_new_rows_by_class"].items()
            }
            new_n = len(c0["x"]) - old_n
            if sum(allocation.values()) != new_n:
                raise AssertionError("C0 allocation does not match appended rows")
            ids = ids_by_stage[stage]
            q, row_classes = build_source_assignment(
                data, ids, allocation, args.device
            )
            c0_block = {
                key: (c0[key][old_n:].clone()
                      if isinstance(c0[key], torch.Tensor)
                      else list(c0[key][old_n:]))
                for key in PAYLOAD_KEYS
            }
            structural, structural_audit = Smoke.build_two_prototype_block(
                data, ids, q, row_classes, stage, args.device, c0_block
            )
            records = structural_audit["records"]
            mapping, mapping_audit = Role.complete_mapping(
                data, ids, q, row_classes, records,
                c0, old_n, "semantic_perm", args.random_seed,
            )
            semantic_block = semantic_block_from_mapping(structural, mapping)
            semantic = candidate_from_parent(c0, parent, semantic_block)
            assert_candidate(c0, parent, semantic_block, semantic,
                             structural, mapping)
            c0_metrics = U.run_students(c0, stage, data, student_args, args.device)
            semantic_metrics = U.run_students(
                semantic, stage, data, student_args, args.device
            )
            semantic_audit = {
                "stage": stage,
                "parent_arm": "semantic_recursive",
                "old_rows": int(old_n),
                "new_rows": int(new_n),
                "allocation": {str(k): int(v) for k, v in allocation.items()},
                "source_assignment_sha256": memory_sha256({
                    "x": q.float().unsqueeze(1),
                    "text_emb": torch.zeros((len(q), 1)),
                    "mass": q.float(),
                    "depth": q.clone(),
                    "stage": q.clone(),
                    "edge_index": torch.empty((2, 0), dtype=torch.long),
                    "texts": [str(v) for v in q.tolist()],
                }),
                "same_c0_topology": bool(torch.equal(
                    semantic["edge_index"], c0["edge_index"])),
                "same_parent_prefix": bool(torch.equal(
                    semantic["x"][:old_n], parent["x"])),
                "payload_multiset_preserved": True,
                "structural_audit": structural_audit,
                "mapping_audit": mapping_audit,
                "mapping_moves": int(sum(
                    int(source != target)
                    for source, target in mapping.items()
                )),
                "mapping": {str(k): int(v) for k, v in mapping.items()},
                "historical_raw_reads": 0,
                "report_test_labels_used_for_construction": False,
            }
        torch.save(semantic, out / f"semantic_recursive_s{stage}_memory.pt")
        save_json(out / f"stage_{stage}_audit.json", {
            "c0_memory_sha256": memory_sha256(c0),
            "semantic_memory_sha256": memory_sha256(semantic),
            **semantic_audit,
        })
        row = {
            "stage": int(stage),
            "budget": int(budgets[stage]),
            "c0": c0_metrics,
            "semantic_recursive": semantic_metrics,
            "delta": {
                "cumulative_f1_pp": float(
                    semantic_metrics["cum_dev_f1_mean"]
                    - c0_metrics["cum_dev_f1_mean"]),
                "old_f1_pp": (None if stage == 0 else float(
                    semantic_metrics["old_mean_f1_mean"]
                    - c0_metrics["old_mean_f1_mean"])),
                "new_f1_pp": (float(
                    semantic_metrics["current_f1_mean"]
                    - c0_metrics["current_f1_mean"])),
            },
            "audit_file": f"stage_{stage}_audit.json",
        }
        rows.append(row)
        print(json.dumps({
            "event": "stage_complete",
            "dataset": args.dataset,
            "stage": stage,
            "c0_f1": c0_metrics["cum_dev_f1_mean"],
            "semantic_f1": semantic_metrics["cum_dev_f1_mean"],
            "delta_pp": row["delta"]["cumulative_f1_pp"],
        }), flush=True)
        previous_c0 = c0
        parent = semantic

    summary = {
        "dataset": args.dataset,
        "stages": len(rows),
        "rows": rows,
        "cumulative_delta_mean_pp": float(
            sum(row["delta"]["cumulative_f1_pp"] for row in rows) / len(rows)),
        "new_delta_mean_pp": float(
            sum(row["delta"]["new_f1_pp"] for row in rows[1:]) / max(1, len(rows) - 1)),
        "positive_incremental_cells": int(sum(
            row["delta"]["cumulative_f1_pp"] > 0 for row in rows[1:])),
        "worst_incremental_cell_pp": float(min(
            row["delta"]["cumulative_f1_pp"] for row in rows[1:])),
    }
    save_json(final_path, {
        "protocol": protocol,
        "source_manifest_sha256": file_sha256(c0_root / "results.json"),
        "data_sha256": file_sha256(Path(args.data)),
        "summary": summary,
        "results": rows,
    })
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("arxiv", "bookchild", "sportsfit"), required=True)
    parser.add_argument("--c0-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--data", default=None)
    parser.add_argument("--minilm", default="/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--student-seeds", type=int, nargs="+", default=[100, 101, 102, 103, 104, 105])
    parser.add_argument("--student-batch", type=int, default=16)
    parser.add_argument("--random-seed", type=int, default=20260926)
    parser.add_argument("--max-stage", type=int, default=None,
                        help="stop after this base+4 stage; used only for smoke tests")
    args = parser.parse_args()
    if args.data is None:
        args.data = U.DATA_CFG[args.dataset]["path"]
    torch.set_num_threads(2)
    started = time.perf_counter()
    summary = run_dataset(args)
    print(json.dumps({"event": "complete", "elapsed_seconds": time.perf_counter() - started,
                      **summary}), flush=True)


if __name__ == "__main__":
    main()
