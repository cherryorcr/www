"""Run XSP and XSP+ST under the strict small-graph external protocol.

The runner intentionally reuses the external-baseline data loader, stream
definition, canonical Student, and report-dev evaluator.  The only new
construction arms are X-conditioned Slot Partition (XSP) and its
straight-through Sinkhorn variant.  Each arm receives only the current delta
and its synthetic parent; historical raw rows are never passed to the
constructor.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

import external_continual_mechanisms as M
import protocol_v3
import run_semantic_recursive_unified_baselines_20260926 as BASE
import x_conditioned_payload_partition as XPP


ARMS = ("xsp", "xsp_st")


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=True),
                         encoding="utf-8")
    temporary.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def memory_hash(memory: dict) -> str:
    digest = hashlib.sha256()
    texts = json.dumps(memory["texts"], ensure_ascii=True).encode("utf-8")
    digest.update(texts)
    for key in ("x", "text_emb", "mass", "depth", "stage", "edge_index"):
        value = memory[key].detach().cpu().contiguous()
        digest.update(f"{key}:{value.dtype}:{list(value.shape)}".encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def sync(device: str) -> None:
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def delta_view(data: dict, ids: torch.Tensor, stage: int) -> dict:
    ids = ids.long().cpu()
    if len(ids.unique()) != len(ids):
        raise AssertionError("delta contains duplicate source ids")
    if not bool(data["pool_mask"][ids].all()):
        raise AssertionError("delta contains non-pool rows")
    return {
        "ids": ids,
        "stage": int(stage),
        "x": data["x"][ids].float(),
        "y": data["y"][ids].long(),
        "text_emb": data["text_emb"][ids].float(),
        "texts": [data["texts"][int(i)] for i in ids],
        "prompt_emb": data.get("prompt_emb"),
        "edge_index": BASE.induced_edges(
            data["edge_index"], ids, len(data["x"])),
    }


def build_xsp_block(data: dict, ids: torch.Tensor, rows: int, stage: int,
                    arm: str, device: str, construction_seed: int,
                    cat_steps: int, partition_steps: int) -> tuple[dict, dict]:
    delta = delta_view(data, ids, stage)
    source_features = M.propagated_features(delta, (0.0, 1.0), device)
    labels = M.row_labels(M.ratio_allocation(delta["y"], rows))

    # Match the large-graph XSP anchor: CaT class-wise MMD with five source
    # texts per row, followed by slot-conditioned payload transport.
    cat_cfg = M.Config(seed=int(construction_seed),
                       cat_steps=int(cat_steps),
                       text_sources=5,
                       text_mode="nearest")
    cctp_block, cctp_audit = M.cat_block(
        delta, rows, data["encoder"], cat_cfg, device)

    variant = "st" if arm == "xsp_st" else "current"
    partition_cfg = XPP.PartitionConfig(
        steps=int(partition_steps),
        hidden=256,
        lr_x=5e-4,
        text_chars=320,
        max_sources=6,
        straight_through_sinkhorn=(variant == "st"),
        lambda_coherence=0.0,
        lambda_hard=0.0,
        hard_refresh_steps=0,
        seed=int(construction_seed) + 152,
    )
    block, audit = XPP.build_x_conditioned_slot_partition(
        delta=delta,
        source_features=source_features,
        x0=cctp_block["x"],
        labels=labels,
        encoder=data["encoder"],
        cfg=partition_cfg,
        mass=cctp_block["mass"],
        device=device,
    )
    audit.update({
        "small_runner_arm": arm,
        "partition_variant": variant,
        "base_constructor": "CaT class-wise MMD X + XSP payload partition",
        "base_cat_audit": cctp_audit,
        "historical_raw_reads": 0,
        "current_delta_only": True,
        "old_memory_updated": False,
        "topology_updated": False,
        "budget_updated": False,
        "construction_seed": int(construction_seed),
        "partition_seed": int(construction_seed) + 152,
    })
    return block, audit


def check_memory(memory: dict, parent: dict | None, budget: int) -> None:
    if len(memory["x"]) != int(budget):
        raise AssertionError("memory row count mismatch")
    if len(memory["texts"]) != int(budget):
        raise AssertionError("memory/text count mismatch")
    if memory["edge_index"].numel():
        raise AssertionError("XSP small protocol must remain edge-free")
    for key in ("x", "text_emb", "mass", "depth", "stage"):
        if len(memory[key]) != int(budget):
            raise AssertionError(f"memory field length mismatch: {key}")
        if not bool(torch.isfinite(memory[key]).all()):
            raise AssertionError(f"nonfinite memory field: {key}")
    if parent is not None:
        old = len(parent["x"])
        for key in ("x", "text_emb", "mass", "depth", "stage"):
            if not torch.equal(memory[key][:old], parent[key]):
                raise AssertionError(f"parent prefix changed: {key}")
        if memory["texts"][:old] != parent["texts"]:
            raise AssertionError("parent text prefix changed")


def protocol_manifest(args, data: dict, order: str, cfg: M.Config) -> dict:
    return {
        "name": "xsp-small-external-strict-20261004",
        "dataset": data["dataset"],
        "order": order,
        "stage_protocol": "three-stage class-incremental",
        "arms": list(ARMS),
        "memory_rate": 0.05,
        "current_delta_only": True,
        "historical_raw_reads": 0,
        "report_test_labels_used_for_construction": False,
        "student": "same original 2-layer GCN + text projection + bidirectional InfoNCE",
        "student_seeds": [int(v) for v in args.report_seeds],
        "student_batch": int(protocol_v3.STUDENT.batch),
        "student_lr": float(protocol_v3.STUDENT.lr),
        "student_steps": "max(50,min(2048,16*memory_rows))",
        "evaluation": "full real graph, all-prompt report-dev node classification",
        "construction_seed": int(args.construction_seed),
        "cat_steps": int(args.cat_steps),
        "xsp_partition_steps": int(args.partition_steps),
        "xsp_config": {
            "text_sources": 5,
            "text_chars": 320,
            "max_sources": 6,
            "sinkhorn_temperature": 0.07,
            "sinkhorn_iterations": 8,
        },
        "cat_config": {
            "seed": int(cfg.seed),
            "cat_steps": int(cfg.cat_steps),
            "text_mode": "nearest",
        },
    }


def prepare_cell(data: dict, order: str, args, cfg: M.Config) -> None:
    root = args.output_root / data["dataset"] / order
    root.mkdir(parents=True, exist_ok=True)
    construction_path = root / "construction.json"
    if construction_path.exists():
        return

    ids, budgets = BASE.set_stream(data, order)
    records = []
    for arm in ARMS:
        parent = None
        for stage, (delta_ids, budget) in enumerate(zip(ids, budgets)):
            started = time.perf_counter()
            old_rows = 0 if parent is None else len(parent["x"])
            new_rows = int(budget) - old_rows
            if new_rows <= 0 or new_rows > len(delta_ids):
                raise AssertionError(
                    f"invalid rows for {data['dataset']}/{order}/s{stage}: "
                    f"new={new_rows}, delta={len(delta_ids)}")
            block, audit = build_xsp_block(
                data, delta_ids, new_rows, stage, arm, args.device,
                args.construction_seed, args.cat_steps, args.partition_steps)
            memory = M.append_edge_free(parent, block)
            check_memory(memory, parent, int(budget))
            memory_path = root / f"{arm}_s{stage}_memory.pt"
            torch.save(memory, memory_path)
            sync(args.device)
            elapsed = time.perf_counter() - started
            record = {
                "dataset": data["dataset"],
                "order": order,
                "arm": arm,
                "stage": int(stage),
                "rows": int(budget),
                "delta_ids": [int(v) for v in delta_ids.tolist()],
                "memory_file": memory_path.name,
                "memory_sha256": memory_hash(memory),
                "file_sha256": file_sha256(memory_path),
                "parent_sha256": memory_hash(parent) if parent else None,
                "construction_seconds": float(elapsed),
                "audit": audit,
            }
            records.append(record)
            parent = memory
            print(json.dumps({"event": "constructed",
                              "dataset": data["dataset"],
                              "order": order,
                              "arm": arm,
                              "stage": stage,
                              "seconds": elapsed}), flush=True)
    save(construction_path, {"protocol": protocol_manifest(args, data, order, cfg),
                             "stage_classes": data["stage_classes"],
                             "budgets": [int(v) for v in budgets],
                             "records": records})


def verify_cell(data: dict, order: str, args) -> dict:
    root = args.output_root / data["dataset"] / order
    construction = read(root / "construction.json")
    previous = {arm: None for arm in ARMS}
    ids, budgets = BASE.set_stream(data, order)
    for record in construction["records"]:
        arm = record["arm"]
        stage = int(record["stage"])
        if record["delta_ids"] != [int(v) for v in ids[stage].tolist()]:
            raise AssertionError("source delta changed after construction")
        if record["rows"] != int(budgets[stage]):
            raise AssertionError("budget changed after construction")
        memory_path = root / record["memory_file"]
        if file_sha256(memory_path) != record["file_sha256"]:
            raise AssertionError("memory file hash changed")
        memory = torch.load(memory_path, map_location="cpu", weights_only=False)
        if memory_hash(memory) != record["memory_sha256"]:
            raise AssertionError("memory content hash changed")
        check_memory(memory, previous[arm], int(budgets[stage]))
        previous[arm] = memory
    return construction


def evaluate_cell(data: dict, order: str, args) -> None:
    root = args.output_root / data["dataset"] / order
    construction = verify_cell(data, order, args)
    result_rows = []
    for record in construction["records"]:
        arm = record["arm"]
        stage = int(record["stage"])
        memory_path = root / record["memory_file"]
        memory = torch.load(memory_path, map_location="cpu", weights_only=False)
        steps = int(protocol_v3.student_steps(len(memory["x"]),
                                              override=args.student_steps))
        scores = []
        for seed in args.report_seeds:
            score_path = root / f"{arm}_s{stage}_seed{seed}.json"
            checkpoint_path = root / f"{arm}_s{stage}_seed{seed}.pt"
            if score_path.exists() and checkpoint_path.exists():
                result = read(score_path)
            else:
                sync(args.device)
                started = time.perf_counter()
                model = BASE.G.train_student(
                    memory, int(seed), args.device, steps,
                    lr=protocol_v3.STUDENT.lr,
                    batch=protocol_v3.STUDENT.batch)
                sync(args.device)
                result = {
                    "seed": int(seed),
                    "steps": steps,
                    **BASE.score_model(model, data, data["stage_classes"], stage),
                    "student_seconds": float(time.perf_counter() - started),
                    "memory_sha256": record["memory_sha256"],
                }
                torch.save({"state": {
                    k: v.detach().cpu() for k, v in model.state_dict().items()}},
                           checkpoint_path)
                result["checkpoint_sha256"] = file_sha256(checkpoint_path)
                save(score_path, result)
                del model
            if result.get("memory_sha256") != record["memory_sha256"]:
                raise AssertionError("score references a different memory")
            scores.append(result)
        result_rows.append({
            **record,
            "steps": steps,
            "aggregate": BASE.aggregate_stage_scores(scores),
            "per_student": scores,
        })
        save(root / "progress.json", {"rows": result_rows})
        print(json.dumps({"event": "evaluated",
                          "dataset": data["dataset"],
                          "order": order,
                          "arm": arm,
                          "stage": stage,
                          "f1": result_rows[-1]["aggregate"]["mean_f1"]}),
              flush=True)
    save(root / "results.json", {
        "protocol": construction["protocol"],
        "stage_classes": construction["stage_classes"],
        "budgets": construction["budgets"],
        "rows": result_rows,
    })


def audit_cell(data: dict, order: str, args) -> None:
    root = args.output_root / data["dataset"] / order
    construction = verify_cell(data, order, args)
    maximum = 0.0
    fits = 0
    for record in construction["records"]:
        arm, stage = record["arm"], int(record["stage"])
        memory = torch.load(root / record["memory_file"],
                            map_location="cpu", weights_only=False)
        for seed in args.report_seeds:
            score_path = root / f"{arm}_s{stage}_seed{seed}.json"
            checkpoint_path = root / f"{arm}_s{stage}_seed{seed}.pt"
            result = read(score_path)
            if result["checkpoint_sha256"] != file_sha256(checkpoint_path):
                raise AssertionError("checkpoint hash changed")
            model = BASE.G.CLIPStudent(memory["x"].shape[1]).to(args.device)
            checkpoint = torch.load(checkpoint_path, map_location="cpu",
                                    weights_only=False)
            model.load_state_dict(checkpoint["state"])
            model.eval()
            score = BASE.score_model(
                model, data, data["stage_classes"], stage)
            for key in ("cumulative_f1", "accuracy", "old_f1", "new_f1"):
                left, right = score.get(key), result.get(key)
                if left is not None and right is not None:
                    maximum = max(maximum, abs(float(left) - float(right)))
            fits += 1
            del model
    if maximum > 1e-7:
        raise AssertionError(f"checkpoint metric mismatch: {maximum}")
    save(root / "audit.json", {
        "passed": True,
        "fit_count": fits,
        "maximum_metric_error_pp": maximum,
        "scope": "source delta, budget, parent prefix, edge-free topology, checkpoint reload",
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--minilm", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--datasets", nargs="+", default=["cora", "citeseer"])
    parser.add_argument("--orders", nargs="+",
                        default=["order19611", "order19612", "order19613",
                                 "order19614", "order19615"])
    parser.add_argument("--report-seeds", nargs="+", type=int,
                        default=list(range(150, 156)))
    parser.add_argument("--construction-seed", type=int, default=270927)
    parser.add_argument("--cat-steps", type=int, default=3000)
    parser.add_argument("--partition-steps", type=int, default=30)
    parser.add_argument("--student-steps", type=int)
    parser.add_argument("--phase", choices=("all", "prepare", "evaluate", "audit"),
                        default="all")
    args = parser.parse_args()
    torch.set_num_threads(2)
    args.output_root.mkdir(parents=True, exist_ok=True)
    cfg = M.Config(seed=args.construction_seed, cat_steps=args.cat_steps)
    manifest = {
        "name": "xsp-small-external-strict-20261004",
        "datasets": args.datasets,
        "orders": args.orders,
        "arms": list(ARMS),
        "report_seeds": args.report_seeds,
        "construction_seed": args.construction_seed,
        "cat_steps": args.cat_steps,
        "partition_steps": args.partition_steps,
        "student_steps": "max(50,min(2048,16*memory_rows))",
        "historical_raw_reads": 0,
    }
    manifest_path = args.output_root / "protocol.json"
    if manifest_path.exists() and read(manifest_path) != manifest:
        raise ValueError("existing output root has a different protocol")
    if not manifest_path.exists():
        save(manifest_path, manifest)

    data = {name: BASE.load_data(name, args.minilm, args.device,
                                 args.cache_root)
            for name in args.datasets}
    for value in data.values():
        value["prompt_emb"] = value["encoder"].encode(value["prompts"])

    phases = ("prepare", "evaluate", "audit") if args.phase == "all" else (args.phase,)
    for phase in phases:
        for name in args.datasets:
            for order in args.orders:
                if phase == "prepare":
                    prepare_cell(data[name], order, args, cfg)
                elif phase == "evaluate":
                    evaluate_cell(data[name], order, args)
                else:
                    audit_cell(data[name], order, args)

    if args.phase == "all":
        rows = []
        for name in args.datasets:
            for order in args.orders:
                rows.extend(read(args.output_root / name / order / "results.json")["rows"])
        save(args.output_root / "results.json", {
            "protocol": manifest,
            "rows": rows,
        })
    print("XSP_SMALL_EXTERNAL_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
