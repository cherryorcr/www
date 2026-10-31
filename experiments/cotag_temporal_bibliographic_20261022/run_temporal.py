"""Registered five-stage chronological evaluation on new citation subsets."""
import argparse
from dataclasses import asdict, replace
from pathlib import Path
import time
import traceback

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from runtime import C, R, E, S, torch, OUT, ARMS, HERE, load_data, snapshot_ids, cohort_masks, validate_delta


def evaluate(memory, data, arrival, stage, device, stage_dir):
    path = stage_dir / "scores.json"
    if path.exists():
        saved = C.read(path)
        if saved["memory_hash"] != R.V.memory_hash(memory) or saved["prediction_archive_sha256"] != C.sha(stage_dir / "predictions.pt.gz"):
            raise AssertionError("Saved evaluation provenance differs")
        return saved
    prompt = data["prompt_emb"].float().to(device)
    masks = {"report_dev": data["report_dev_mask"], "test": data["test_mask"]}
    classes = list(range(len(data["names"])))
    output = {mode: {split: [] for split in masks} for mode in ("arrived_snapshot", "current_snapshot")}
    resources, predictions = [], {}
    for seed in C.STUDENT_SEEDS:
        R.V.sync(device)
        torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        model = R.P.train_prompt_student(memory, prompt, seed, device, 2048, lr=.001, batch=24)
        R.V.sync(device)
        cost = dict(seed=seed, train_seconds=time.perf_counter()-start,
                    train_peak_mb=torch.cuda.max_memory_allocated(device)/2**20,
                    train_reserved_peak_mb=torch.cuda.max_memory_reserved(device)/2**20, inference={})
        predictions[seed] = {}
        for mode, splits in output.items():
            idx = snapshot_ids(arrival, stage, mode)
            if bool((arrival[idx] > stage).any()):
                raise AssertionError("Future node in inference")
            edge = R.L.induced_edges(data["edge_index"], idx, len(arrival))
            R.V.sync(device)
            torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            adj = R.P.normalized_adjacency(edge, len(idx), device)
            with torch.no_grad():
                pred = model.logits(data["x"][idx].to(device), adj, prompt).argmax(1).cpu()
            R.V.sync(device)
            cost["inference"][mode] = dict(seconds=time.perf_counter()-started, nodes=len(idx), edges=edge.shape[1],
                peak_mb=torch.cuda.max_memory_allocated(device)/2**20,
                reserved_peak_mb=torch.cuda.max_memory_reserved(device)/2**20)
            predictions[seed][mode] = dict(ids=idx, pred=pred)
            for split, mask in masks.items():
                metrics = dict(seed=seed)
                for cohort, allowed in cohort_masks(mask, arrival, stage).items():
                    local = allowed[idx]
                    count = int(local.sum())
                    if not count:
                        metrics[cohort] = None
                    else:
                        yy, pp = data["y"][idx][local].numpy(), pred[local].numpy()
                        metrics[cohort] = dict(nodes=count,
                            f1=float(100*f1_score(yy, pp, labels=classes, average="macro", zero_division=0)),
                            accuracy=float(100*accuracy_score(yy, pp)))
                splits[split].append(metrics)
            del adj, pred
        resources.append(cost)
        del model
        torch.cuda.empty_cache()
    for mode, splits in output.items():
        for split, items in list(splits.items()):
            aggregate = dict(per_student=items)
            for cohort in ("cumulative", "current", "historical"):
                values = [r[cohort] for r in items if r[cohort] is not None]
                aggregate[cohort] = None if not values else dict(nodes=values[0]["nodes"], **{
                    key: float(np.mean([v[key] for v in values])) for key in ("f1", "accuracy")}, **{
                    key+"_sd": float(np.std([v[key] for v in values], ddof=1)) for key in ("f1", "accuracy")})
            splits[split] = aggregate
    C.archive(stage_dir / "predictions.pt.gz", predictions)
    saved = dict(prediction=output, training_resources=resources, memory_hash=R.V.memory_hash(memory),
                 prediction_archive_sha256=C.sha(stage_dir / "predictions.pt.gz"))
    C.save(path, saved)
    return saved


def run(args):
    C.disk_check()
    directory = OUT / "runs" / args.dataset / f"{args.arm}_c{args.seed}"
    data, arrival, ids, budgets, floors, source, encoded, stats = load_data(args.dataset, args.device)
    cfg = replace(R.q_config("uq_dm_edgefree", args.seed, 30, 4096), sinkhorn_max_steps=16384)
    protocol = dict(name="bibliographic-publication-year-five-stage-v1", dataset=args.dataset, arm=args.arm,
        construction_seed=args.seed, student_seeds=C.STUDENT_SEEDS, student_steps=2048, student_batch=24,
        student=R.P.model_config(), q_config=asdict(cfg), memory_rate=.05, budgets=budgets,
        actual_final_rate=budgets[-1]/sum(map(len, ids)), class_floor_adjustments=floors,
        source=source, encoding=encoded, stages=stats, classes=data["names"],
        stage_id_hashes=[R.V.hash_tensor(v) for v in ids], pool_mask_hash=R.V.hash_tensor(data["pool_mask"]),
        report_mask_hash=R.V.hash_tensor(data["report_dev_mask"]), test_mask_hash=R.V.hash_tensor(data["test_mask"]),
        numerical_variant="verified float64 Sinkhorn normalization; cap16384; float32 Q; tolerance1e-4",
        precision_code_sha256=C.sha(S.__file__), source_hashes=C.source_hashes(),
        runner_hash=C.sha(__file__), runtime_hash=C.sha(HERE / "runtime.py"), registration_sha256=C.sha(HERE / "REGISTRATION.md"),
        access="only current-stage legal training nodes, their induced edges and own frozen synthetic prefix",
        inference_modes=["arrived_snapshot", "current_snapshot"],
        time_semantics="publication year, not measured platform ingestion time",
        edge_time_assumption="released reference list becomes available at citing publication year; impossible forward-year references excluded",
        taxonomy="fixed venue-derived research communities; not article-level expert topic labels",
        offline_cpu_harness=True, future_raw_on_gpu=False, method_selection_from_test=False,
        official_reproduction=False, evaluation="new held-out source-ID masks; protocol frozen before model outcomes")
    result = C.start_result(directory / "results.json", protocol)
    if result["status"] == "complete":
        for row in result["rows"]:
            folder = directory / f"s{row['stage']}"
            memory = C.load_memory(folder / "memory.pt.gz")
            if R.V.memory_hash(memory) != row["construction"]["memory_hash"]:
                raise AssertionError("Completed memory changed")
        C.emit(event="skip_verified_complete", dataset=args.dataset, arm=args.arm, seed=args.seed)
        return
    if args.arm == "uq_dm_edgefree":
        S.install(R.UQ)
    parent = None
    try:
        for stage, idx in enumerate(ids):
            stage_dir = directory / f"s{stage}"
            memory_path = stage_dir / "memory.pt.gz"
            audit_path = stage_dir / "construction.json"
            previous = next((r for r in result["rows"] if r["stage"] == stage), None)
            if memory_path.exists() and audit_path.exists():
                memory, construction = C.load_memory(memory_path), C.read(audit_path)
                R.V.validate_memory(memory, parent, budgets[stage])
                if R.V.memory_hash(memory) != construction["memory_hash"]:
                    raise AssertionError("Stage memory hash changed")
            else:
                if memory_path.exists() or audit_path.exists():
                    raise RuntimeError("Incomplete checkpoint pair; preserve for audit")
                legal = validate_delta(idx, arrival, data["pool_mask"], stage)
                delta = R.L.delta_view(data, idx, stage)
                labels = R.M.row_labels(R.M.ratio_allocation(delta["y"], budgets[stage]-(0 if parent is None else len(parent["x"]))))
                R.V.sync(args.device)
                torch.cuda.reset_peak_memory_stats(args.device)
                start = time.perf_counter()
                memory, audit = E.construct(args.arm, delta, parent, labels, cfg, data["encoder"], data["prompt_emb"], args.seed, args.device)
                R.V.sync(args.device)
                checks = R.V.validate_memory(memory, parent, budgets[stage])
                if memory["edge_index"].numel():
                    raise AssertionError("Edge-free memory unexpectedly has edges")
                if "q_row_sum_max_error" in audit:
                    checks.update(R.assignment_checks(audit))
                    if not checks["sinkhorn_feasible"]:
                        C.save(stage_dir / "failed_assignment.json", audit)
                        raise AssertionError("Sinkhorn feasibility failed")
                construction = dict(stage=stage, access=legal, checks=checks, seconds=time.perf_counter()-start,
                    peak_mb=torch.cuda.max_memory_allocated(args.device)/2**20,
                    reserved_peak_mb=torch.cuda.max_memory_reserved(args.device)/2**20,
                    memory_rows=len(memory["x"]), memory_hash=R.V.memory_hash(memory), audit=R.compact_audit(audit))
                C.archive(memory_path, memory)
                C.save(audit_path, construction)
                del delta
                C.emit(event="constructed", dataset=args.dataset, arm=args.arm, seed=args.seed, stage=stage,
                       seconds=construction["seconds"], peak_mb=construction["peak_mb"])
            if previous is None:
                scores = evaluate(memory, data, arrival, stage, args.device, stage_dir)
                result["rows"].append(dict(stage=stage, construction=construction, scores=scores))
                C.save(directory / "results.json", result)
                C.emit(event="evaluated", dataset=args.dataset, arm=args.arm, seed=args.seed, stage=stage,
                       report=scores["prediction"]["arrived_snapshot"]["report_dev"]["cumulative"])
            elif previous["construction"]["memory_hash"] != construction["memory_hash"]:
                raise AssertionError("Saved scores refer to a different memory")
            parent = memory
            torch.cuda.empty_cache()
        result["status"] = "complete"
    except Exception as error:
        result.update(status="failed", error=repr(error), traceback=traceback.format_exc())
        C.save(directory / "results.json", result)
        raise
    C.save(directory / "results.json", result)
    C.emit(event="complete", dataset=args.dataset, arm=args.arm, seed=args.seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("acm", "dblp"), required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--seed", type=int, default=271006)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.cuda.set_device(args.device)
    run(args)
