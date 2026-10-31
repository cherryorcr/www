"""Strict fixed-X text-only diagnostic for the CaT TAG adapter.

Each stage runs the CaT X optimizer exactly once.  The four text arms then
reuse the same optimized X and only change current-delta text attachment.
This isolates text construction from sparse propagation and optimizer
re-execution noise in the ordinary per-arm adapter runner.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

import torch

import external_continual_mechanisms as M
import run_external_continual_adapters_20260927 as E
import run_semantic_recursive_unified_baselines_20260926 as BASE


TEXT_ARM_MODES = {
    **E.CAT_TEXT_MODES,
    "cat_mmd_tag_text_center": "text_center",
    "cat_mmd_tag_random_text": "random",
    "cat_mmd_tag_wrong_class_text": "wrong_class",
    "cat_mmd_tag_cctp_top1": "text_center",
    "cat_mmd_tag_cctp_top3": "text_center",
    "cat_mmd_tag_cctp_top5": "text_center",
    "cat_mmd_tag_cctp_soft": "text_center_soft",
}
TEXT_ARM_SOURCES = {
    "cat_mmd_tag_cctp_top1": 1,
    "cat_mmd_tag_cctp_top3": 3,
    "cat_mmd_tag_cctp_top5": 5,
    "cat_mmd_tag_cctp_soft": 5,
}
ARMS = tuple(TEXT_ARM_MODES)


def save(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=True), encoding="utf-8")
    tmp.replace(path)


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def tensor_hash(value: torch.Tensor) -> str:
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(tuple(value.shape)).encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def protocol(args, cfg):
    out = E.protocol(args, cfg)
    out["name"] = "cat-tag-fixed-x-text-diagnostic-20261001"
    out["status"] = "strict fixed-X text-only diagnostic; mechanism adaptation, not official CaT reproduction"
    out["fixed_x"] = True
    out["fixed_x_description"] = (
        "one CaT X optimization per dataset/order/stage; all selected text arms "
        "reuse the same optimized X and differ only in current-delta text attachment"
    )
    out["cat_text_modes"] = TEXT_ARM_MODES
    return out


def prepare_cell(data, order, args, cfg):
    root = args.output_root / data["dataset"] / order
    root.mkdir(parents=True, exist_ok=True)
    if (root / "seal.json").exists():
        E.verify_cell(root)
        return

    ids, budgets = BASE.set_stream(data, order)
    groups = data["stage_classes"]
    flattened = [c for g in groups for c in g]
    if len(flattened) != len(set(flattened)):
        raise ValueError("fixed-X diagnostic requires disjoint class increments")

    parents = {arm: None for arm in args.arms}
    records = []
    for stage in range(3):
        E.sync(args.device)
        stage_start = time.perf_counter()
        delta = E.delta_view(data, ids[stage], stage)
        rows = budgets[stage] - len(next(iter(parents.values()))["x"]) if stage else budgets[stage]

        # This is the only CaT X optimization for this stage.  Text is
        # attached below from the shared X without rerunning the optimizer.
        x_start = time.perf_counter()
        x_block, x_audit = M.cat_block(
            delta, rows, data["encoder"], replace(cfg, text_mode="nearest"),
            args.device)
        E.sync(args.device)
        x_seconds = time.perf_counter() - x_start
        shared_x = x_block["x"].detach().clone()
        shared_x_hash = tensor_hash(shared_x)
        source_features = M.propagated_features(delta, (0.0, 1.0), args.device)
        labels = torch.tensor(x_audit["row_classes"], dtype=torch.long)

        for arm in args.arms:
            E.sync(args.device)
            text_start = time.perf_counter()
            arm_cfg = replace(
                cfg,
                text_mode=TEXT_ARM_MODES[arm],
                text_sources=TEXT_ARM_SOURCES.get(arm, cfg.text_sources),
            )
            block, audit = M.attach_text(
                delta, source_features, shared_x, labels, data["encoder"], arm_cfg)
            E.sync(args.device)
            text_seconds = time.perf_counter() - text_start
            memory = M.append_edge_free(parents[arm], block)
            E.check_memory(memory, parents[arm], budgets[stage], edge_free=True)
            audit.update(
                fixed_x=True,
                shared_x_hash=shared_x_hash,
                shared_x_source="single CaT X optimization shared by all text arms",
                x_optimizer_seconds=x_seconds,
                text_attachment_seconds=text_seconds,
                construction_seconds_accounted=x_seconds + text_seconds,
                source_x_audit=x_audit,
            )
            elapsed = time.perf_counter() - stage_start
            path = root / f"{arm}_s{stage}_memory.pt"
            torch.save(memory, path)
            record = {
                "arm": arm,
                "stage": stage,
                "rows": budgets[stage],
                "memory_file": path.name,
                "memory_sha256": E.memory_hash(memory),
                "file_sha256": E.file_hash(path),
                "delta_ids": ids[stage].tolist(),
                "parent_sha256": E.memory_hash(parents[arm]) if parents[arm] else None,
                "construction_seconds": x_seconds + text_seconds,
                "artifact_load_or_construct_seconds": elapsed,
                "audit": audit,
                "resources": E.resource_stats(memory, data["encoder"]),
            }
            records.append(record)
            parents[arm] = memory
            print(json.dumps({
                "event": "constructed",
                "dataset": data["dataset"],
                "order": order,
                "stage": stage,
                "arm": arm,
                "shared_x_hash": shared_x_hash,
                "seconds": elapsed,
            }), flush=True)

    save(root / "construction.json", {
        "stage_classes": groups,
        "budgets": budgets,
        "records": records,
        "fixed_x": True,
        "shared_x_hashes": {
            str(stage): next(
                r["audit"]["shared_x_hash"] for r in records if r["stage"] == stage
            ) for stage in range(3)
        },
    })
    save(root / "seal.json", {
        "construction_sha256": E.file_hash(root / "construction.json"),
        "protocol_sha256": E.file_hash(args.output_root / "protocol.json"),
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--recursive-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--minilm", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--datasets", nargs="+", default=["cora", "citeseer"])
    parser.add_argument("--orders", nargs="+", default=["order19611", "order19612"])
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--report-seeds", nargs="+", type=int, default=list(range(150, 156)))
    parser.add_argument("--seed", type=int, default=270927)
    parser.add_argument("--cat-steps", type=int, default=3000)
    parser.add_argument("--student-steps", type=int)
    parser.add_argument("--phase", choices=("all", "prepare", "evaluate", "audit"), default="all")
    args = parser.parse_args()
    torch.set_num_threads(2)
    cfg = M.Config(seed=args.seed, cat_steps=args.cat_steps)
    manifest = protocol(args, cfg)
    protocol_path = args.output_root / "protocol.json"
    if protocol_path.exists() and read(protocol_path) != manifest:
        raise ValueError("existing protocol differs; use a new output root")
    if not protocol_path.exists():
        save(protocol_path, manifest)

    data = {
        name: BASE.load_data(name, args.minilm, args.device, args.cache_root)
        for name in args.datasets
    }
    for value in data.values():
        value["prompt_emb"] = value["encoder"].encode(value["prompts"])

    phases = ("prepare", "evaluate", "audit") if args.phase == "all" else (args.phase,)
    for phase in phases:
        for name in args.datasets:
            for order in args.orders:
                if phase == "prepare":
                    prepare_cell(data[name], order, args, cfg)
                elif phase == "evaluate":
                    E.evaluate_cell(data[name], order, args)
                else:
                    E.audit_cell(data[name], order, args)

    if args.phase == "all":
        rows = [
            row
            for name in args.datasets
            for order in args.orders
            for row in read(args.output_root / name / order / "results.json")["rows"]
        ]
        save(args.output_root / "results.json", {"protocol": manifest, "rows": rows})
    print("CAT_TAG_FIXED_X_TEXT_DIAGNOSTIC_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
