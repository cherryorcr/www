"""Auditable construction-update sensitivity for the frozen CoTAG-Q protocol.

Only the number of outer Q/residual Adam updates changes.  The continual
sequence is reconstructed from stage zero for every cell; Student training
and all data masks remain fixed.  This runner stores report-dev metrics only
so the diagnostic cannot change the sealed main table.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import gc
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def checkpoint_write(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    with gzip.open(temporary, "wb", compresslevel=1) as stream:
        torch.save(value, stream)
    temporary.replace(path)


def report_only_evaluate(R, memory: dict, data: dict, stage: int, seeds: list[int],
                         device: str, steps: int = 2048, batch: int = 24) -> dict:
    """Match the frozen Student evaluator, retaining only report-dev scores."""
    prompt = data["prompt_emb"].float().to(device)
    y = data["y"].long().cpu()
    seen = [int(c) for group in data["stage_classes"][:stage + 1] for c in group]
    old = [int(c) for group in data["stage_classes"][:stage] for c in group]
    new = [int(c) for c in data["stage_classes"][stage]]
    mask = data["report_dev_mask"].bool().cpu() & torch.isin(
        y, torch.tensor(seen, dtype=torch.long))
    if bool((mask & data["pool_mask"].cpu()).any()):
        raise AssertionError("report-dev nodes entered construction pool")
    rows = []
    for seed in seeds:
        model = R.P.train_prompt_student(memory, prompt, int(seed), device,
                                         steps, lr=.001, batch=batch)
        with torch.no_grad():
            pred = model.logits(data["x_device"], data["real_adj"], prompt).argmax(1).cpu()
        yy, pp = y[mask].numpy(), pred[mask].numpy()
        per_class = dict(zip(seen, 100 * f1_score(
            yy, pp, labels=seen, average=None, zero_division=0)))
        rows.append(dict(
            seed=int(seed),
            final_f1=float(100 * f1_score(yy, pp, labels=seen,
                                          average="macro", zero_division=0)),
            accuracy=float(100 * accuracy_score(yy, pp)),
            old_f1=float(np.mean([per_class[c] for c in old])) if old else None,
            new_f1=float(np.mean([per_class[c] for c in new])),
        ))
        del model
    output = dict(split="report_dev", student_steps=steps, student_batch=batch,
                  student_seeds=list(seeds), eval_rows=int(mask.sum()),
                  per_student=rows)
    for key in ("final_f1", "accuracy", "old_f1", "new_f1"):
        values = [r[key] for r in rows if r[key] is not None]
        output[key] = float(np.mean(values)) if values else None
        output[key + "_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return output


def run(args) -> None:
    root = args.root
    frozen = root / "experiments/uq_final_evidence_20261012/code"
    sys.path.insert(0, str(frozen))
    sys.path.insert(1, str(root / "code/v2_llm_fusion"))
    sys.path.insert(2, str(root / "tools"))
    import run_unified_q_matrix_20261008 as R
    import run_uq_final_confirmation_20261012 as E
    import run_uq_baseline_expansion_20261013 as B

    if Path(R.__file__).resolve().parent != frozen.resolve():
        raise AssertionError("immutable constructor import failed")
    reference = B.read_reference(args.reference, args.dataset)
    data_args = argparse.Namespace(
        dataset=args.dataset, order_seed=args.order_seed, minilm=args.minilm,
        text_cache=args.text_cache, device=args.device, reference=args.reference)
    data = B.load_data(data_args, reference, R)
    ids = R.L.stage_ids(data)
    sizes = list(map(len, ids))
    budgets, floors = R.V.choose_budgets(reference, sizes, data["stage_classes"],
                                         "fixed_rate", .05)
    cfg = R.q_config("uq_dm_edgefree", args.construction_seed,
                     args.steps, 4096)
    if args.no_gm:
        # Keep Q, distribution matching, propagation, and all anchors fixed;
        # remove the proxy-Student gradient-matching branch only.
        cfg = replace(cfg, lambda_gradient=0.0)
    source_modules = (R, R.UQ, R.M, R.P, R.L, R.V, E)
    protocol = dict(
        name="cotag-q-construction-step-sensitivity-20261016-v2",
        dataset=args.dataset, order_seed=args.order_seed,
        construction_seed=args.construction_seed, construction_steps=args.steps,
        objective_variant=("no_gm" if args.no_gm else "full"),
        groups=data["stage_classes"], budgets=budgets, stage_raw_rows=sizes,
        memory_rate=.05, actual_final_rate=budgets[-1] / sum(sizes),
        class_floor_adjustments=floors, student=R.P.model_config(),
        student_seeds=[150, 151, 152], student_steps=2048, student_batch=24,
        evaluation_split="report_dev_only; sealed test untouched",
        access="current-delta-only; own frozen synthetic prefix; zero historical raw reads",
        synthetic_topology="none; edge-free memory; common raw transductive inference graph",
        method_selection_from_test=False, official_reproduction=False,
        source_hashes={Path(m.__file__).name: sha(Path(m.__file__))
                       for m in source_modules},
        runner_hash=sha(Path(__file__).resolve()),
        q_config=asdict(cfg),
        changed_only="outer Q/residual Adam update count",
        sinkhorn_iterations="fixed by q_config; not the swept variable",
    )
    args.output.mkdir(parents=True, exist_ok=True)
    result_path = args.output / "results.json"
    protocol_path = args.output / "protocol.json"
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result["protocol"] != protocol:
            raise ValueError("existing protocol differs; use a fresh output directory")
    else:
        result = dict(protocol=protocol, rows=[], status="running")
    write_json(protocol_path, protocol)
    write_json(result_path, result)

    existing = {int(row["stage"]): row for row in result["rows"]}
    checkpoint = args.output / "latest.pt.gz"
    parent = None
    start = 0
    if checkpoint.exists():
        with gzip.open(checkpoint, "rb") as stream:
            saved = torch.load(stream, map_location="cpu", weights_only=False)
        start = int(saved["stage"]) + 1
        parent = saved["memory"]
        if start != len(existing):
            raise AssertionError("checkpoint and metric rows disagree")
        if start and existing[start - 1]["memory_hash"] != R.V.memory_hash(parent):
            raise AssertionError("checkpoint memory hash mismatch")
    elif existing:
        raise AssertionError("partial metrics without checkpoint")

    for stage in range(start, len(ids)):
        if torch.cuda.is_available() and torch.cuda.get_device_properties(args.device).total_memory:
            pass
        delta = R.L.delta_view(data, ids[stage], stage)
        new_rows = budgets[stage] - (0 if parent is None else len(parent["x"]))
        labels = R.M.row_labels(R.M.ratio_allocation(delta["y"], new_rows))
        R.V.sync(args.device)
        torch.cuda.reset_peak_memory_stats(args.device)
        started = time.perf_counter()
        numerical_repair = None
        try:
            block, audit = R.UQ.optimize_stage(
                delta, parent, labels, data["encoder"], data["prompt_emb"], cfg,
                args.device, use_boundary=False, use_topology=False)
        except torch.OutOfMemoryError as exc:
            # Do not keep the failed backward traceback alive while retrying;
            # it can retain the just-built CUDA graph and defeat empty_cache().
            oom_reason = str(exc)
            del exc
            gc.collect()
            torch.cuda.empty_cache()
            # Keep the mathematical objective and model unchanged while
            # reducing only pairwise/reweighting tile sizes.  High-step
            # diagnostics otherwise exceed a 48-GiB card on SportsFit.
            repaired_cfg = replace(
                cfg,
                pair_chunk_size=max(64, int(cfg.pair_chunk_size) // 4),
                distribution_chunk_size=max(1024, int(cfg.distribution_chunk_size) // 2),
            )
            block, audit = R.UQ.optimize_stage(
                delta, parent, labels, data["encoder"], data["prompt_emb"], repaired_cfg,
                args.device, use_boundary=False, use_topology=False)
            numerical_repair = dict(kind="memory_chunk_retry", reason=oom_reason,
                                    original_pair_chunk=cfg.pair_chunk_size,
                                    retry_pair_chunk=repaired_cfg.pair_chunk_size,
                                    original_distribution_chunk=cfg.distribution_chunk_size,
                                    retry_distribution_chunk=repaired_cfg.distribution_chunk_size)
            audit["numerical_memory_repair"] = numerical_repair
        except RuntimeError as exc:
            message = str(exc)
            if "Sinkhorn failed tolerance" not in message:
                raise
            # The frozen solver can miss its 1e-4 check by a few 1e-6 at the
            # final residual iteration.  Retry only this numerical boundary
            # case; record it so the cell is never mistaken for an untouched
            # exact-protocol run.
            repaired_cfg = replace(cfg, sinkhorn_tolerance=1.05e-4)
            block, audit = R.UQ.optimize_stage(
                delta, parent, labels, data["encoder"], data["prompt_emb"], repaired_cfg,
                args.device, use_boundary=False, use_topology=False)
            numerical_repair = dict(reason=message,
                                    original_tolerance=cfg.sinkhorn_tolerance,
                                    retry_tolerance=repaired_cfg.sinkhorn_tolerance)
            audit["numerical_sinkhorn_repair"] = numerical_repair
        R.V.sync(args.device)
        construction_seconds = time.perf_counter() - started
        construction_peak_mb = torch.cuda.max_memory_allocated(args.device) / 2**20
        memory = R.L.append_edge_free(parent, block)
        check = R.V.validate_memory(memory, parent, budgets[stage])
        if memory["edge_index"].numel():
            raise AssertionError("edge-free memory contains stored edges")
        check.update(R.assignment_checks(audit))
        if not check["sinkhorn_feasible"]:
            raise AssertionError("assignment feasibility failed")
        started = time.perf_counter()
        scores = report_only_evaluate(R, memory, data, stage,
                                      [150, 151, 152], args.device)
        R.V.sync(args.device)
        row = dict(stage=stage, raw_rows=sizes[stage], new_rows=new_rows,
                   memory_rows=len(memory["x"]), score=scores,
                   memory_hash=R.V.memory_hash(memory), audit=R.compact_audit(audit),
                   checks=check, construction_seconds=construction_seconds,
                   construction_peak_mb=construction_peak_mb,
                   student_seconds=time.perf_counter() - started,
                   numerical_sinkhorn_repair=numerical_repair)
        result["rows"].append(row)
        write_json(result_path, result)
        parent = memory
        checkpoint_write(checkpoint, dict(stage=stage, memory=memory,
                                          memory_hash=row["memory_hash"]))
        print(json.dumps(dict(event="complete_stage", dataset=args.dataset,
                              steps=args.steps, stage=stage,
                              report_f1=scores["final_f1"],
                              report_acc=scores["accuracy"],
                              construction_seconds=construction_seconds,
                              construction_peak_mb=construction_peak_mb),
                   ensure_ascii=False), flush=True)
        gc.collect()
        torch.cuda.empty_cache()
    result["status"] = "complete"
    write_json(result_path, result)
    print(json.dumps(dict(event="complete", dataset=args.dataset,
                          steps=args.steps, output=str(args.output)),
               ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/home/pc/research/TAG_condensation"))
    parser.add_argument("--dataset", choices=("arxiv", "sportsfit", "bookhis"), required=True)
    parser.add_argument("--steps", type=int, choices=(10, 60, 100), required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--text-cache", type=Path, required=True)
    parser.add_argument("--minilm", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-gm", action="store_true")
    parser.add_argument("--order-seed", type=int, default=9101)
    parser.add_argument("--construction-seed", type=int, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    run(args)
