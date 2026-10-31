"""CaT plus minimal differentiable Student-response bridge experiment.

The runner keeps the validated CaT protocol fixed and adds only a short
Student-compatible outer objective for the current appended block.  It is
intended to distinguish a useful response signal from the earlier standalone
MMD/text objectives before any P/A or topology changes are attempted.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import time

import torch

import cat_multimodal_minimal as MM
import external_continual_mechanisms as M
import run_external_continual_adapters_20260927 as E
import run_semantic_recursive_unified_baselines_20260926 as BASE
import student_response_bridge as RB


ARMS = ("cat_x", "cat_t_only", "response_t_only", "response_x_only",
        "response_xt", "mmd_response_xt", "support_t_only",
        "support_x_only", "support_xt", "mmd_support_x_only",
        "mmd_support_xt", "new_margin_x_only")


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=True),
                         encoding="utf-8")
    temporary.replace(path)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protocol(args, cat_cfg: M.Config, mm_cfg: MM.MMConfig,
             response_cfg: RB.ResponseConfig) -> dict:
    return {
        "name": "cat-student-response-bridge-20261002",
        "status": "minimal response feasibility; no P/A/topology revision",
        "datasets": args.datasets,
        "orders": args.orders,
        "arms": list(args.arms),
        "report_seeds": args.report_seeds,
        "config": {"cat": asdict(cat_cfg),
                    "mmd_text_control": asdict(mm_cfg),
                    "response_bridge": asdict(response_cfg)},
        "student": "same original two-layer GCN, frozen MiniLM, bidirectional InfoNCE",
        "memory": "same 5% cumulative legal-pool rows; append-only old prefix",
        "construction_access": "current delta plus already materialized synthetic parent only",
        "topology": "edge-free materialization and same implicit self-loop Student operator",
        "editable_payload": "current new block X/T only; old prefix detached",
        "evaluation": "all-prompt seen-node report-dev Macro-F1 and Accuracy",
        "report_driven_selection": False,
        "historical_raw_reads": 0,
    }


def prepare_cell(data: dict, order: str, args, cat_cfg: M.Config,
                 mm_cfg: MM.MMConfig, response_cfg: RB.ResponseConfig) -> None:
    root = args.output_root / data["dataset"] / order
    root.mkdir(parents=True, exist_ok=True)
    if (root / "seal.json").exists():
        return
    ids, budgets = BASE.set_stream(data, order)
    parents = {arm: None for arm in args.arms}
    records = []
    for stage in range(3):
        E.sync(args.device)
        delta = E.delta_view(data, ids[stage], stage)
        old_rows = 0 if stage == 0 else len(next(iter(parents.values()))["x"])
        new_rows = budgets[stage] - old_rows
        if new_rows < 1:
            raise ValueError("stage has no new rows")
        source_features = M.propagated_features(delta, (0.0, 1.0), args.device)
        labels = M.row_labels(M.ratio_allocation(delta["y"], new_rows))

        x_start = time.perf_counter()
        cat_cfg_stage = replace(cat_cfg, text_mode="text_center",
                                text_sources=3)
        base_block, x_audit = M.cat_block(
            delta, new_rows, data["encoder"], cat_cfg_stage, args.device)
        E.sync(args.device)
        x_seconds = time.perf_counter() - x_start
        shared_x = base_block["x"].detach().clone()
        cctp_block, cctp_audit = M.attach_text(
            delta, source_features, shared_x, labels, data["encoder"],
            cat_cfg_stage)
        shared_hash = hashlib.sha256(
            shared_x.contiguous().numpy().tobytes()).hexdigest()

        for arm in args.arms:
            started = time.perf_counter()
            if arm == "cat_x":
                block, audit = cctp_block, {
                    "constructor": "cat_x_plus_cctp_text",
                    "cat_audit": x_audit,
                    "text_audit": cctp_audit,
                    "text_source_ids": cctp_audit["text_source_ids"],
                    "row_classes": labels.tolist(),
                }
            elif arm == "cat_t_only":
                block, audit = MM.optimize_text_payload(
                    delta, source_features, shared_x,
                    cctp_block["text_emb"], cctp_block["texts"],
                    cctp_audit["text_source_ids"], labels, parents[arm],
                    mm_cfg, "text_only", cctp_block["mass"], args.device)
                audit["cat_audit"] = x_audit
                audit["cctp_audit"] = cctp_audit
            else:
                mode = arm
                cfg = response_cfg
                if arm in ("mmd_response_xt", "mmd_support_x_only",
                           "mmd_support_xt"):
                    cfg = replace(response_cfg,
                                  mmd_weight=float(args.response_mmd_weight))
                if arm in ("support_t_only", "support_x_only",
                           "support_xt", "mmd_support_x_only",
                           "mmd_support_xt"):
                    cfg = replace(cfg, new_weight=0.0, prompt_weight=0.0)
                if arm == "new_margin_x_only":
                    cfg = replace(
                        cfg,
                        new_weight=0.0,
                        prompt_weight=0.0,
                        support_weight=0.0,
                        support_prompt_weight=0.0,
                        new_margin_weight=float(
                            args.response_new_margin_weight),
                        new_margin=float(args.response_new_margin),
                        mmd_weight=0.0,
                    )
                block, audit = RB.optimize_response_block(
                    delta, source_features, shared_x,
                    cctp_block["text_emb"], cctp_block["texts"],
                    cctp_audit["text_source_ids"], labels, parents[arm],
                    cfg, mode, cctp_block["mass"], args.device,
                    prompt_emb=data["prompt_emb"],
                    prompt_ids=torch.arange(len(data["prompt_emb"])))
                audit["cat_audit"] = x_audit
                audit["cctp_audit"] = cctp_audit
            memory = M.append_edge_free(parents[arm], block)
            E.check_memory(memory, parents[arm], budgets[stage], edge_free=True)
            path = root / f"{arm}_s{stage}_memory.pt"
            torch.save(memory, path)
            audit.update(shared_x_hash=shared_hash,
                         x_optimizer_seconds=float(x_seconds),
                         construction_seconds_accounted=float(
                             time.perf_counter() - started + x_seconds))
            records.append({
                "arm": arm,
                "stage": stage,
                "rows": int(budgets[stage]),
                "memory_file": path.name,
                "memory_sha256": E.memory_hash(memory),
                "file_sha256": file_hash(path),
                "delta_ids": ids[stage].tolist(),
                "parent_sha256": E.memory_hash(parents[arm])
                if parents[arm] else None,
                "construction_seconds": float(
                    time.perf_counter() - started + x_seconds),
                "audit": audit,
                "resources": E.resource_stats(memory, data["encoder"]),
            })
            parents[arm] = memory
            print(json.dumps({"event": "constructed", "dataset": data["dataset"],
                              "order": order, "stage": stage, "arm": arm,
                              "seconds": time.perf_counter() - started}),
                  flush=True)
    save(root / "construction.json", {"budgets": budgets, "records": records})
    save(root / "seal.json", {
        "construction_sha256": file_hash(root / "construction.json"),
        "protocol_sha256": file_hash(args.output_root / "protocol.json"),
    })


def evaluate_cell(data: dict, order: str, args) -> None:
    BASE.set_stream(data, order)
    root = args.output_root / data["dataset"] / order
    construction = read(root / "construction.json")
    results = []
    for item in construction["records"]:
        memory = BASE.load(root / item["memory_file"])
        steps = BASE.protocol_v3.student_steps(
            len(memory["x"]), override=args.student_steps)
        scores = []
        for seed in args.report_seeds:
            path = root / f"{item['arm']}_s{item['stage']}_seed{seed}.json"
            if path.exists():
                result = read(path)
            else:
                E.sync(args.device)
                started = time.perf_counter()
                model = BASE.G.train_student(
                    memory, int(seed), args.device, steps, lr=0.001, batch=16)
                E.sync(args.device)
                result = {
                    "seed": int(seed),
                    "steps": int(steps),
                    **BASE.score_model(model, data, data["stage_classes"],
                                       item["stage"]),
                    "student_seconds": float(time.perf_counter() - started),
                    "memory_sha256": item["memory_sha256"],
                }
                checkpoint = path.with_suffix(".pt")
                torch.save({"state": {k: v.detach().cpu()
                                       for k, v in model.state_dict().items()}},
                           checkpoint)
                result["checkpoint_sha256"] = file_hash(checkpoint)
                save(path, result)
                del model
            scores.append(result)
        aggregate = BASE.aggregate_stage_scores(scores)
        results.append({
            "dataset": data["dataset"],
            "order": order,
            "arm": item["arm"],
            "stage": item["stage"],
            "memory_rows": item["rows"],
            "aggregate": aggregate,
            "per_student": scores,
            "construction_seconds": item["construction_seconds"],
            "student_wall_seconds": float(sum(
                s["student_seconds"] for s in scores)),
        })
        save(root / "progress.json", {"rows": results})
        print(json.dumps({"event": "evaluated", "dataset": data["dataset"],
                          "order": order, "stage": item["stage"],
                          "arm": item["arm"],
                          "f1": aggregate["mean_f1"]}), flush=True)
    save(root / "results.json", {
        "protocol": read(args.output_root / "protocol.json"),
        "rows": results,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--minilm", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--datasets", nargs="+", default=["cora", "citeseer"])
    parser.add_argument("--orders", nargs="+", default=["order19611"])
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--report-seeds", nargs="+", type=int,
                        default=[150, 151, 152])
    parser.add_argument("--cat-steps", type=int, default=3000)
    parser.add_argument("--mmd-steps", type=int, default=300)
    parser.add_argument("--mmd-hidden", type=int, default=256)
    parser.add_argument("--mmd-lr-text", type=float, default=5e-3)
    parser.add_argument("--mmd-lr-x", type=float, default=1e-3)
    parser.add_argument("--response-inner-steps", type=int, default=8)
    parser.add_argument("--response-inner-batch", type=int, default=16)
    parser.add_argument("--response-outer-steps", type=int, default=4)
    parser.add_argument("--response-outer-lr", type=float, default=2e-2)
    parser.add_argument("--response-seeds", nargs="+", type=int,
                        default=[9300, 9301])
    parser.add_argument("--response-old-weight", type=float, default=0.5)
    parser.add_argument("--response-new-weight", type=float, default=1.0)
    parser.add_argument("--response-prompt-weight", type=float, default=1.0)
    parser.add_argument("--response-new-margin-weight", type=float, default=0.0)
    parser.add_argument("--response-new-margin", type=float, default=0.20)
    parser.add_argument("--response-x-radius", type=float, default=0.20)
    parser.add_argument("--response-text-radius", type=float, default=0.25)
    parser.add_argument("--response-mmd-weight", type=float, default=1.0)
    parser.add_argument("--response-mmd-hidden", type=int, default=128)
    parser.add_argument("--student-steps", type=int)
    parser.add_argument("--phase", choices=("all", "prepare", "evaluate"),
                        default="all")
    args = parser.parse_args()
    torch.set_num_threads(2)
    cat_cfg = M.Config(seed=270927, cat_steps=args.cat_steps,
                       text_mode="text_center", text_sources=3)
    mm_cfg = MM.MMConfig(steps=args.mmd_steps, hidden=args.mmd_hidden,
                         lr_text=args.mmd_lr_text, lr_x=args.mmd_lr_x)
    response_cfg = RB.ResponseConfig(
        inner_steps=args.response_inner_steps,
        inner_batch=args.response_inner_batch,
        outer_steps=args.response_outer_steps,
        outer_lr=args.response_outer_lr,
        response_seeds=tuple(args.response_seeds),
        old_weight=args.response_old_weight,
        new_weight=args.response_new_weight,
        prompt_weight=args.response_prompt_weight,
        new_margin_weight=args.response_new_margin_weight,
        new_margin=args.response_new_margin,
        x_radius=args.response_x_radius,
        text_radius=args.response_text_radius,
        mmd_hidden=args.response_mmd_hidden,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = protocol(args, cat_cfg, mm_cfg, response_cfg)
    protocol_path = args.output_root / "protocol.json"
    if protocol_path.exists() and read(protocol_path) != manifest:
        raise ValueError("existing output root has a different protocol")
    if not protocol_path.exists():
        save(protocol_path, manifest)
    data = {name: BASE.load_data(name, args.minilm, args.device,
                                 args.cache_root)
            for name in args.datasets}
    for value in data.values():
        value["prompt_emb"] = value["encoder"].encode(value["prompts"])
    phases = ("prepare", "evaluate") if args.phase == "all" else (args.phase,)
    for phase in phases:
        for name in args.datasets:
            for order in args.orders:
                if phase == "prepare":
                    prepare_cell(data[name], order, args, cat_cfg, mm_cfg,
                                 response_cfg)
                else:
                    evaluate_cell(data[name], order, args)
    if args.phase == "all":
        rows = []
        for name in args.datasets:
            for order in args.orders:
                rows.extend(read(args.output_root / name / order /
                                 "results.json")["rows"])
        save(args.output_root / "results.json", {
            "protocol": manifest,
            "rows": rows,
        })
    print("CAT_STUDENT_RESPONSE_BRIDGE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()

