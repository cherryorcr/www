"""Minimal feasibility experiment for putting text inside the CaT objective."""
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


ARMS = ("cat_x", "cat_t_only", "cat_xt", "cat_textaware_x",
        "cat_textaware_soft", "cat_textaware_hard",
        "cat_joint_text_group", "cat_gated_joint_text_group", "random_text")
ARMS = ARMS + ("cat_payload_cctp", "cat_payload_text_mmr",
               "cat_payload_tagsam", "cat_payload_graph_role",
               "cat_payload_anchor_mix", "cat_payload_class_shared_tagsam",
               "cat_payload_class_shared_anchor_mix",
               "cat_payload_class_shared_two_tagsam")


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


def protocol(args, cat_cfg: M.Config, mm_cfg: MM.MMConfig) -> dict:
    return {
        "name": "cat-multimodal-minimal-20261002",
        "status": "feasibility; CaT X initialization plus X-conditioned text payload",
        "datasets": args.datasets,
        "orders": args.orders,
        "arms": list(args.arms),
        "report_seeds": args.report_seeds,
        "config": {"cat": asdict(cat_cfg), "multimodal": asdict(mm_cfg)},
        "student": "same original two-layer GCN, frozen text encoder, bidirectional InfoNCE",
        "memory": "same 5% cumulative legal-pool rows; append-only old prefix",
        "construction_access": "current delta plus already materialized synthetic parent only",
        "topology": "same edge-free materialization and Student implicit topology",
        "text": ("cat_t_only/cat_xt are free continuous payload controls; "
                 "cat_textaware_* derive a differentiable same-class soft text "
                 "payload from the current CaT X; cat_textaware_x retains the "
                 "CCTP text, while soft/hard expose the payload materialization; "
                 "cat_textaware_group_x assigns over a bounded bank of real "
                 "CCTP text groups and stores the selected group; "
                 "cat_joint_text_group jointly refines X and a bounded "
                 "X/prompt text query with a CaT trust-region anchor; "
                 "cat_gated_joint_text_group adds an explicit CCTP no-op "
                 "gate and margin-approved text replacement; cat_payload_* "
                 "uses fixed CaT X with class-wise local support pools and "
                 "real text chunks, with TAGSAM-style coverage treated as "
                 "a proxy rather than an official TAGSAM reproduction; "
                 "anchor_mix retains a bounded CCTP prefix and fills the "
                 "remaining budget with coverage-selected chunks; class_shared "
                 "variants reuse one selected payload per class to control "
                 "within-class text variance"),
        "evaluation": "all-prompt seen-node report-dev Macro-F1 and Accuracy",
        "report_driven_selection": False,
    }


def prepare_cell(data: dict, order: str, args, cat_cfg: M.Config,
                 mm_cfg: MM.MMConfig) -> None:
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
        # Keep the formal comparison's text-source budget in one place.  The
        # earlier feasibility run hard-coded three sources here, which made
        # its CCTP control incomparable with the five-source baseline matrix.
        cat_cfg_stage = replace(cat_cfg, text_mode="text_center",
                                text_sources=mm_cfg.text_sources)
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
            elif arm == "random_text":
                block, audit = M.attach_text(
                    delta, source_features, shared_x, labels,
                    data["encoder"], replace(cat_cfg_stage, text_mode="random"))
            elif arm in ("cat_textaware_x", "cat_textaware_soft",
                         "cat_textaware_hard"):
                block, audit = MM.optimize_text_aware_cat(
                    delta, source_features, shared_x,
                    cctp_block["text_emb"], cctp_block["texts"],
                    cctp_audit["text_source_ids"], labels, parents[arm],
                    data["encoder"], mm_cfg, cctp_block["mass"], args.device,
                    materialize=("base" if arm.endswith("_x")
                                 else ("soft" if arm.endswith("_soft")
                                       else "hard")))
                audit["cat_audit"] = x_audit
                audit["cctp_audit"] = cctp_audit
            elif arm == "cat_textaware_group_x":
                block, audit = MM.optimize_text_group_aware_cat(
                    delta, source_features, shared_x,
                    cctp_block["text_emb"], cctp_block["texts"],
                    cctp_audit["text_source_ids"], labels, parents[arm],
                    data["encoder"], mm_cfg, cctp_block["mass"], args.device)
                audit["cat_audit"] = x_audit
                audit["cctp_audit"] = cctp_audit
            elif arm == "cat_joint_text_group":
                block, audit = MM.optimize_joint_cat_text_groups(
                    delta, source_features, shared_x,
                    cctp_block["text_emb"], cctp_block["texts"],
                    cctp_audit["text_source_ids"], labels, parents[arm],
                    data["encoder"], mm_cfg, cctp_block["mass"], args.device)
                audit["cat_audit"] = x_audit
                audit["cctp_audit"] = cctp_audit
            elif arm == "cat_gated_joint_text_group":
                block, audit = MM.optimize_gated_joint_cat_text(
                    delta, source_features, shared_x,
                    cctp_block["text_emb"], cctp_block["texts"],
                    cctp_audit["text_source_ids"], labels, parents[arm],
                    data["encoder"], mm_cfg, cctp_block["mass"], args.device)
                audit["cat_audit"] = x_audit
                audit["cctp_audit"] = cctp_audit
            elif arm.startswith("cat_payload_"):
                mode = arm.removeprefix("cat_payload_")
                if mode == "tagsam":
                    mode = "tagsam_chunk"
                block, audit = MM.attach_chunk_payload(
                    delta, source_features, shared_x,
                    cctp_block["text_emb"], cctp_block["texts"],
                    cctp_audit["text_source_ids"], labels,
                    data["encoder"], mm_cfg, mode,
                    cctp_block["mass"], args.device)
                audit["cat_audit"] = x_audit
                audit["cctp_audit"] = cctp_audit
            else:
                block, audit = MM.optimize_text_payload(
                    delta, source_features, shared_x,
                    cctp_block["text_emb"], cctp_block["texts"],
                    cctp_audit["text_source_ids"], labels, parents[arm],
                    mm_cfg, "text_only" if arm == "cat_t_only" else "joint",
                    cctp_block["mass"], args.device)
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
    save(root / "seal.json", {"construction_sha256": file_hash(root / "construction.json"),
                               "protocol_sha256": file_hash(args.output_root / "protocol.json")})


def evaluate_cell(data: dict, order: str, args) -> None:
    # ``set_stream`` mutates the shared data object with the active class
    # groups.  Restore the order-specific stream before scoring; otherwise a
    # multi-order run would evaluate earlier cells with the last order's
    # report mask and prompt groups.
    BASE.set_stream(data, order)
    root = args.output_root / data["dataset"] / order
    construction = read(root / "construction.json")
    results = []
    for item in construction["records"]:
        memory = BASE.load(root / item["memory_file"])
        steps = BASE.protocol_v3.student_steps(len(memory["x"]),
                                               override=args.student_steps)
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
        results.append({"dataset": data["dataset"], "order": order,
                        "arm": item["arm"], "stage": item["stage"],
                        "memory_rows": item["rows"], "aggregate": aggregate,
                        "per_student": scores,
                        "construction_seconds": item["construction_seconds"],
                        "student_wall_seconds": float(sum(
                            s["student_seconds"] for s in scores))})
        save(root / "progress.json", {"rows": results})
        print(json.dumps({"event": "evaluated", "dataset": data["dataset"],
                          "order": order, "stage": item["stage"],
                          "arm": item["arm"],
                          "f1": aggregate["mean_f1"]}), flush=True)
    save(root / "results.json", {"protocol": read(args.output_root / "protocol.json"),
                                  "rows": results})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--minilm", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--datasets", nargs="+", default=["cora", "citeseer"])
    parser.add_argument("--orders", nargs="+", default=["order19611"])
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--report-seeds", nargs="+", type=int, default=[150, 151, 152])
    parser.add_argument("--cat-steps", type=int, default=3000)
    parser.add_argument("--mm-steps", type=int, default=300)
    parser.add_argument("--mm-hidden", type=int, default=256)
    parser.add_argument("--mm-lr-text", type=float, default=5e-3)
    parser.add_argument("--mm-lr-x", type=float, default=1e-3)
    parser.add_argument("--mm-lambda-text", type=float, default=1.0)
    parser.add_argument("--mm-lambda-cross", type=float, default=0.25)
    parser.add_argument("--mm-lambda-anchor-text", type=float, default=0.05)
    parser.add_argument("--mm-lambda-anchor-x", type=float, default=0.05)
    parser.add_argument("--mm-assignment-temperature", type=float, default=0.07)
    parser.add_argument("--mm-text-sources", type=int, default=3)
    parser.add_argument("--mm-text-chars", type=int, default=320)
    parser.add_argument("--mm-text-ridge", type=float, default=1e-2)
    parser.add_argument("--mm-text-group-count", type=int, default=4)
    parser.add_argument("--mm-lambda-group-entropy", type=float, default=0.02)
    parser.add_argument("--mm-lambda-prompt-margin", type=float, default=0.15)
    parser.add_argument("--mm-prompt-margin", type=float, default=0.05)
    parser.add_argument("--mm-lambda-query-anchor", type=float, default=0.5)
    parser.add_argument("--mm-x-trust-radius", type=float, default=0.05)
    parser.add_argument("--mm-lambda-cctp-anchor", type=float, default=0.5)
    parser.add_argument("--mm-joint-base-logit", type=float, default=1.5)
    parser.add_argument("--mm-joint-selection-margin", type=float, default=0.05)
    parser.add_argument("--mm-lambda-joint-base", type=float, default=0.5)
    parser.add_argument("--mm-joint-x-trust-radius", type=float, default=0.01)
    parser.add_argument("--mm-support-pool-size", type=int, default=32)
    parser.add_argument("--mm-chunk-chars", type=int, default=120)
    parser.add_argument("--mm-payload-max-chunks", type=int, default=8)
    parser.add_argument("--mm-payload-max-sources", type=int, default=6)
    parser.add_argument("--mm-payload-graph-weight", type=float, default=0.35)
    parser.add_argument("--mm-payload-text-weight", type=float, default=0.45)
    parser.add_argument("--mm-payload-prompt-weight", type=float, default=0.20)
    parser.add_argument("--mm-payload-redundancy", type=float, default=0.20)
    parser.add_argument("--mm-payload-contrastive-weight", type=float, default=0.35)
    parser.add_argument("--mm-payload-source-repeat-penalty", type=float, default=0.10)
    parser.add_argument("--student-steps", type=int)
    parser.add_argument("--phase", choices=("all", "prepare", "evaluate"), default="all")
    args = parser.parse_args()
    torch.set_num_threads(2)
    cat_cfg = M.Config(seed=270927, cat_steps=args.cat_steps,
                       text_mode="text_center",
                       text_sources=args.mm_text_sources)
    mm_cfg = MM.MMConfig(steps=args.mm_steps, hidden=args.mm_hidden,
                         lr_text=args.mm_lr_text, lr_x=args.mm_lr_x,
                         lambda_text=args.mm_lambda_text,
                         lambda_cross=args.mm_lambda_cross,
                         lambda_anchor_text=args.mm_lambda_anchor_text,
                         lambda_anchor_x=args.mm_lambda_anchor_x,
                         assignment_temperature=args.mm_assignment_temperature,
                         text_sources=args.mm_text_sources,
                         text_chars=args.mm_text_chars,
                         text_ridge=args.mm_text_ridge,
                         text_group_count=args.mm_text_group_count,
                         lambda_group_entropy=args.mm_lambda_group_entropy,
                         lambda_prompt_margin=args.mm_lambda_prompt_margin,
                         prompt_margin=args.mm_prompt_margin,
                         lambda_query_anchor=args.mm_lambda_query_anchor,
                         x_trust_radius=args.mm_x_trust_radius,
                         lambda_cctp_anchor=args.mm_lambda_cctp_anchor,
                         joint_base_logit=args.mm_joint_base_logit,
                         joint_selection_margin=args.mm_joint_selection_margin,
                         lambda_joint_base=args.mm_lambda_joint_base,
                         joint_x_trust_radius=args.mm_joint_x_trust_radius,
                         support_pool_size=args.mm_support_pool_size,
                         chunk_chars=args.mm_chunk_chars,
                         payload_max_chunks=args.mm_payload_max_chunks,
                         payload_max_sources=args.mm_payload_max_sources,
                         payload_graph_weight=args.mm_payload_graph_weight,
                         payload_text_weight=args.mm_payload_text_weight,
                         payload_prompt_weight=args.mm_payload_prompt_weight,
                         payload_redundancy=args.mm_payload_redundancy,
                         payload_contrastive_weight=args.mm_payload_contrastive_weight,
                         payload_source_repeat_penalty=args.mm_payload_source_repeat_penalty)
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = protocol(args, cat_cfg, mm_cfg)
    protocol_path = args.output_root / "protocol.json"
    if protocol_path.exists() and read(protocol_path) != manifest:
        raise ValueError("existing output root has a different protocol")
    if not protocol_path.exists():
        save(protocol_path, manifest)
    data = {name: BASE.load_data(name, args.minilm, args.device, args.cache_root)
            for name in args.datasets}
    for value in data.values():
        value["prompt_emb"] = value["encoder"].encode(value["prompts"])
    phases = ("prepare", "evaluate") if args.phase == "all" else (args.phase,)
    for phase in phases:
        for name in args.datasets:
            for order in args.orders:
                if phase == "prepare":
                    prepare_cell(data[name], order, args, cat_cfg, mm_cfg)
                else:
                    evaluate_cell(data[name], order, args)
    if args.phase == "all":
        rows = []
        for name in args.datasets:
            for order in args.orders:
                rows.extend(read(args.output_root / name / order / "results.json")["rows"])
        save(args.output_root / "results.json", {"protocol": manifest, "rows": rows})
    print("CAT_MULTIMODAL_MINIMAL_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
