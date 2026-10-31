"""Frozen full-history and baseline resource comparisons; no sealed-test reads."""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("COTAG_ROOT", "/home/pc/research/TAG_condensation"))
BASE = ROOT / "experiments/cotag_memory_optimization_20261022"
spec = importlib.util.spec_from_file_location("memory_resource_base", BASE / "run.py")
A = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = A
spec.loader.exec_module(A)
sys.path.insert(0, str(HERE))
from protocol_utils import full_support, label_schedule, payload_bytes

torch, R = A.torch, A.R


def run(args):
    torch.set_num_threads(4)
    path = args.output / "results.json"
    if args.output.exists():
        raise FileExistsError(args.output)
    if shutil.disk_usage(ROOT).free < 8 * 2**30:
        raise RuntimeError("8 GiB disk floor")
    args.output.mkdir(parents=True)
    source = ROOT / "results/cotag_memory_optimization_20261022_v2" / f"{args.dataset}_combined_c{args.seed}"
    reference = A.read(source / "results.json")
    p = reference["protocol"]
    assert reference["status"] == "complete"
    for name, expected_hash in p["source_hashes"].items():
        assert A.digest(name) == expected_hash, name
    final_ref = reference["rows"][-1]
    ref_file = source / final_ref["memory_path"]
    assert A.digest(ref_file) == final_ref["memory_file_sha256"]
    final_labels = A.load_memory(ref_file)["labels"].long().cpu()
    data = R.L.load_large(args.dataset,
        "/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip",
        ROOT / f"results/public_baselines_text_cache_20261001/{args.dataset}", args.device,
        dict(groups=p["groups"], report_dev_mod=3))
    ids = R.L.stage_ids(data)
    assert [R.V.hash_tensor(v) for v in ids] == p["stage_hashes"]
    assert R.V.hash_tensor(data["pool_mask"]) == p["pool_hash"]
    assert R.V.hash_tensor(data["report_dev_mask"]) == p["report_hash"]
    assert not bool((data["pool_mask"].cpu() & data["report_dev_mask"].cpu()).any())
    full_ids = full_support(ids, data["pool_mask"])
    labels = label_schedule(data["y"], ids, p["budgets"], R.M.ratio_allocation, R.M.row_labels)
    assert torch.equal(torch.cat(labels), final_labels)
    cfg = A.U.Config(**p["config"])
    assert cfg.assignment_storage == "class_blocks" and cfg.lambda_gradient > 0 and cfg.steps == 30
    protocol = dict(base=p, arm=args.arm, full_history=args.arm == "full_history",
        source_result=str(source / "results.json"), source_result_sha256=A.digest(source / "results.json"),
        full_pool_hash=R.V.hash_tensor(full_ids), final_label_hash=R.V.hash_tensor(final_labels),
        runner_hash=A.digest(__file__), registration_hash=A.digest(HERE / "README.md"),
        protocol_utils_hash=A.digest(HERE / "protocol_utils.py"),
        baseline_config=A.asdict(R.M.Config(seed=args.seed, cat_steps=3000, text_sources=5, text_mode="nearest")),
        measurement="same resident encoder; original-graph GPU caches removed before construction",
        baseline_official_reproduction=False, full_history_rng_stage=0,
        config_updates_per_q_call=30, total_q_calls=1 if args.arm == "full_history" else None,
        same_total_compute=False, report_dev_only=True, cpu_threads=4,
        gpu_name=torch.cuda.get_device_name(args.device), torch_version=torch.__version__)
    result = dict(status="running", protocol=protocol, rows=[])
    A.save(path, result)
    parent = None
    plan = ([(len(ids)-1, full_ids, torch.cat(labels), 0)] if args.arm == "full_history" else
            [(s, raw, labels[s], s) for s, raw in enumerate(ids)])
    try:
        for stage, raw_ids, slot_labels, rng_stage in plan:
            if shutil.disk_usage(ROOT).free < 8 * 2**30:
                raise RuntimeError("8 GiB disk floor")
            delta = R.L.delta_view(data, raw_ids, rng_stage)
            assert torch.equal(delta["ids"], raw_ids)
            A.discard_eval_cache(data)
            R.V.sync(args.device)
            entry = torch.cuda.memory_allocated(args.device)
            torch.cuda.reset_peak_memory_stats(args.device)
            start, cpu_start = time.perf_counter(), time.process_time()
            with A.RSSMonitor() as rss:
                if args.arm == "full_history":
                    block, audit = A.U.optimize_stage(delta, None, slot_labels,
                        data["encoder"], data["prompt_emb"], cfg, args.device,
                        use_boundary=False, use_topology=False)
                else:
                    block, audit = R.make_baseline(delta, len(slot_labels), args.arm,
                                                   data["encoder"], args.seed, args.device)
                R.V.sync(args.device)
            seconds, cpu_seconds = time.perf_counter()-start, time.process_time()-cpu_start
            peak, reserved = torch.cuda.max_memory_allocated(args.device), torch.cuda.max_memory_reserved(args.device)
            assert torch.equal(block["labels"].cpu(), slot_labels)
            memory = R.L.append_edge_free(parent, block)
            checks = R.V.validate_memory(memory, parent, p["budgets"][stage])
            assert torch.equal(memory["labels"].cpu(), final_labels[:p["budgets"][stage]])
            assert not memory["edge_index"].numel()
            if args.arm == "full_history":
                checks.update(R.assignment_checks(audit))
                assert checks["sinkhorn_feasible"]
            file = args.output / f"stage{stage}_memory.pt.gz"
            A.archive(file, memory)
            row = dict(stage=stage, rng_stage=rng_stage, raw_rows=len(raw_ids), raw_edges=delta["edge_index"].shape[1],
                raw_id_hash=R.V.hash_tensor(raw_ids), new_rows=len(slot_labels), memory_rows=len(memory["x"]),
                construction_seconds=seconds, cpu_process_seconds=cpu_seconds, peak_bytes=peak,
                reserved_bytes=reserved, entry_bytes=entry, cpu_peak_rss_bytes=rss.peak,
                memory_path=file.name, memory_file_sha256=A.digest(file), memory_hash=R.V.memory_hash(memory),
                payload_bytes=payload_bytes(memory), checks=checks, audit=R.compact_audit(audit), scores=None)
            result["rows"].append(row)
            A.save(path, result)
            A.emit(event="constructed", dataset=args.dataset, arm=args.arm, seed=args.seed,
                   stage=stage, seconds=seconds, peak_gib=peak/2**30)
            if stage in (0, len(ids)-1):
                row["scores"] = A.evaluate(memory, data, stage, args.device, args.output)
            A.save(path, result)
            A.emit(event="stage_complete", dataset=args.dataset, arm=args.arm, seed=args.seed,
                   stage=stage, f1=None if row["scores"] is None else row["scores"]["f1"])
            parent = memory
            del block, delta
            gc.collect()
            torch.cuda.empty_cache()
        result["status"] = "complete"
        A.save(path, result)
    except Exception as error:
        result.update(status="failed", failure=repr(error), traceback=traceback.format_exc(),
                      failed_stage=stage, failure_peak_bytes=torch.cuda.max_memory_allocated(args.device),
                      failure_reserved_bytes=torch.cuda.max_memory_reserved(args.device))
        A.save(path, result)
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=("arxiv", "sportsfit"), required=True)
    p.add_argument("--arm", choices=("full_history", "cat_cctp", "gecc_hard"), required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args())
