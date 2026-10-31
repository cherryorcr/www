"""Four-way resource audit; continuation only from verified stage checkpoints."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import gc
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import time
import traceback

import numpy as np
import psutil
import torch
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(os.environ.get("COTAG_ROOT", "/home/pc/research/TAG_condensation"))
HERE = Path(__file__).resolve().parent
FROZEN = ROOT / "experiments/uq_final_evidence_20261012/code"
sys.path.insert(0, str(FROZEN))
sys.path.insert(1, str(ROOT / "code/v2_llm_fusion"))
import run_uq_final_confirmation_20261012 as E

R = E.R
sys.path.insert(0, str(HERE / "implementation"))
payload_name = "unified_q_payload_projection"
saved_payload = sys.modules.get(payload_name)
payload_spec = importlib.util.spec_from_file_location(payload_name, HERE / "implementation/unified_q_payload_projection.py")
payload_module = importlib.util.module_from_spec(payload_spec)
payload_spec.loader.exec_module(payload_module)
sys.modules[payload_name] = payload_module
spec = importlib.util.spec_from_file_location("memory_audit_constructor", HERE / "implementation/unified_q_multimodal.py")
U = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = U
spec.loader.exec_module(U)
if saved_payload is not None:
    sys.modules[payload_name] = saved_payload
VARIANTS = ("original", "cache_offload", "class_blocks", "combined")
STUDENT_SEEDS = (150, 151, 152)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    tmp.replace(path)


def emit(**item):
    print(json.dumps(item), flush=True)


def archive(path, memory):
    if path.exists():
        raise FileExistsError(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(tmp, "wb", compresslevel=3) as stream:
        torch.save(memory, stream)
    tmp.replace(path)


def load_memory(path):
    with gzip.open(path, "rb") as stream:
        return torch.load(stream, map_location="cpu", weights_only=False)


def discard_eval_cache(data):
    data.pop("x_device", None)
    data.pop("real_adj", None)
    gc.collect()
    torch.cuda.empty_cache()


def ensure_eval_cache(data, device):
    R.V.sync(device)
    start = time.perf_counter()
    if "real_adj" not in data:
        data["real_adj"] = R.P.normalized_adjacency(data["edge_index"], len(data["x"]), device)
    if "x_device" not in data:
        data["x_device"] = data["x"].float().to(device)
    R.V.sync(device)
    return time.perf_counter() - start


class RSSMonitor:
    def __enter__(self):
        self.stop = threading.Event()
        self.process = psutil.Process()
        self.peak = self.process.memory_info().rss

        def sample():
            while not self.stop.wait(.02):
                self.peak = max(self.peak, self.process.memory_info().rss)

        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.peak = max(self.peak, self.process.memory_info().rss)
        self.stop.set()
        self.thread.join()


def evaluate(memory, data, stage, device, output):
    seen = [int(c) for group in data["stage_classes"][:stage + 1] for c in group]
    mask = data["report_dev_mask"].cpu().bool() & torch.isin(data["y"].cpu(), torch.tensor(seen))
    if bool((mask & data["pool_mask"].cpu()).any()):
        raise AssertionError("pool/report overlap")
    truth = data["y"][mask].long().cpu().numpy()
    prompt = data["prompt_emb"].float().to(device)
    rows = []
    for seed in STUDENT_SEEDS:
        # Measure synthetic-memory training independently of original-graph inference.
        discard_eval_cache(data)
        R.V.sync(device)
        torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        model = R.P.train_prompt_student(memory, prompt, seed, device, 2048, lr=.001, batch=24)
        R.V.sync(device)
        train_seconds = time.perf_counter() - start
        train_peak = torch.cuda.max_memory_allocated(device)
        rebuild = ensure_eval_cache(data, device)
        torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        with torch.no_grad():
            prediction = model.logits(data["x_device"], data["real_adj"], prompt).argmax(1).cpu()
        R.V.sync(device)
        infer_seconds = time.perf_counter() - start
        infer_peak = torch.cuda.max_memory_allocated(device)
        pred = prediction[mask].numpy()
        p = output / f"stage{stage}_student{seed}.npz"
        if p.exists():
            raise FileExistsError(p)
        np.savez_compressed(p, ids=torch.where(mask)[0].numpy(), truth=truth, pred=pred, classes=seen)
        rows.append(dict(seed=seed, f1=100 * f1_score(truth, pred, labels=seen, average="macro", zero_division=0),
            accuracy=100 * accuracy_score(truth, pred), prediction_path=p.name, prediction_sha256=digest(p),
            train_seconds=train_seconds, train_peak_bytes=train_peak,
            inference_seconds=infer_seconds, inference_peak_bytes=infer_peak, cache_rebuild_seconds=rebuild))
        del model
        gc.collect()
        torch.cuda.empty_cache()
    return dict(split="report_dev", f1=float(np.mean([r["f1"] for r in rows])),
                accuracy=float(np.mean([r["accuracy"] for r in rows])), per_student=rows)


def run(args):
    torch.set_num_threads(4)
    path = args.output / "results.json"
    args.output.mkdir(parents=True, exist_ok=True)
    previous = read(path) if path.exists() else None
    if previous and previous["status"] == "failed":
        raise RuntimeError("failed attempt preserved; no automatic retry")
    if shutil.disk_usage(ROOT).free < 8 * 2**30:
        raise RuntimeError("8 GiB disk floor")
    ref = R.L.read(ROOT / f"experiments/uq_final_evidence_20261012/{args.dataset}_reference.json")
    groups = R.V.choose_groups(ref, 9101)
    data = R.L.load_large(args.dataset,
        "/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip",
        ROOT / f"results/public_baselines_text_cache_20261001/{args.dataset}", args.device,
        dict(groups=groups, report_dev_mod=ref.get("report_dev_mod", 3)))
    ids = R.L.stage_ids(data)
    budgets, floors = R.V.choose_budgets(ref, list(map(len, ids)), groups, "fixed_rate", .05)
    original_config = asdict(R.q_config("uq_dm_edgefree", args.seed, 30, 4096))
    cfg = U.Config(**original_config)
    blocked = args.variant in ("class_blocks", "combined")
    offload = args.variant in ("cache_offload", "combined")
    cfg = replace(cfg, assignment_storage="class_blocks" if blocked else "dense")
    source_paths = [Path(__file__), HERE / "README.md", Path(U.__file__),
                    HERE / "implementation/unified_q_blocks.py",
                    HERE / "implementation/unified_q_payload_projection.py"]
    source_paths += [Path(module.__file__) for module in (E, R, R.UQ, R.M, R.P, R.L, R.V, R.I, R.I.D)]
    protocol = dict(name="cotag-memory-optimization-v1", dataset=args.dataset, variant=args.variant,
        construction_seed=args.seed, order_seed=9101, groups=groups, budgets=budgets,
        class_floor_adjustments=floors, stage_raw_rows=list(map(len, ids)), config=asdict(cfg),
        student_steps=2048, student_batch=24, student_seeds=list(STUDENT_SEEDS),
        pool_hash=R.V.hash_tensor(data["pool_mask"]), report_hash=R.V.hash_tensor(data["report_dev_mask"]),
        stage_hashes=[R.V.hash_tensor(v) for v in ids],
        source_hashes={str(p): digest(p) for p in source_paths},
        evaluation="report-dev only, no method selection", cpu_threads=4,
        device_name=torch.cuda.get_device_name(args.device), torch_version=torch.__version__)
    protocol = json.loads(json.dumps(protocol))
    primary_path = ROOT / "results/uq_final_evidence_20261012/confirmation" / f"{args.dataset}_o9101_c{args.seed}/results.json"
    primary = read(primary_path)["protocol"]
    expected = dict(groups=groups, budgets=budgets, stage_raw_rows=list(map(len, ids)),
                    pool_mask_hash=protocol["pool_hash"], report_mask_hash=protocol["report_hash"],
                    stage_id_hashes=protocol["stage_hashes"], q_config=original_config)
    expected = json.loads(json.dumps(expected))
    checks = {k: value == primary[k] for k, value in expected.items()}
    if not all(checks.values()):
        raise AssertionError(dict(primary_protocol_mismatch=checks))
    protocol["primary_protocol_checks"] = checks
    protocol["primary_result_sha256"] = digest(primary_path)
    if previous and previous["protocol"] != protocol:
        raise RuntimeError("protocol changed; preserve previous run")
    result = previous or dict(status="running", protocol=protocol, rows=[])
    limit = min(args.stage_limit or len(ids), len(ids))
    if len(result["rows"]) >= limit:
        emit(event="skip_complete", dataset=args.dataset, variant=args.variant, seed=args.seed)
        return
    parent = None
    if result["rows"]:
        last = result["rows"][-1]
        if digest(args.output / last["memory_path"]) != last["memory_file_sha256"]:
            raise AssertionError("checkpoint file hash mismatch")
        parent = load_memory(args.output / last["memory_path"])
        if R.V.memory_hash(parent) != last["memory_hash"]:
            raise AssertionError("checkpoint payload mismatch")
    result["status"] = "running"
    save(path, result)
    try:
        for stage in range(len(result["rows"]), limit):
            if shutil.disk_usage(ROOT).free < 8 * 2**30:
                raise RuntimeError("8 GiB disk floor")
            delta = R.L.delta_view(data, ids[stage], stage)
            new_n = budgets[stage] - (0 if parent is None else len(parent["x"]))
            labels = R.M.row_labels(R.M.ratio_allocation(delta["y"], new_n))
            if offload:
                discard_eval_cache(data)
            else:
                ensure_eval_cache(data, args.device)
            gc.collect()
            torch.cuda.empty_cache()
            R.V.sync(args.device)
            entry_bytes = torch.cuda.memory_allocated(args.device)
            torch.cuda.reset_peak_memory_stats(args.device)
            start = time.perf_counter()
            with RSSMonitor() as rss:
                block, audit = U.optimize_stage(delta, parent, labels, data["encoder"],
                    data["prompt_emb"], cfg, args.device, use_boundary=False, use_topology=False)
                R.V.sync(args.device)
            elapsed = time.perf_counter() - start
            peak = torch.cuda.max_memory_allocated(args.device)
            reserved = torch.cuda.max_memory_reserved(args.device)
            exit_bytes = torch.cuda.memory_allocated(args.device)
            memory = R.L.append_edge_free(parent, block)
            checks = R.V.validate_memory(memory, parent, budgets[stage])
            checks.update(R.assignment_checks(audit))
            if not checks["sinkhorn_feasible"] or memory["edge_index"].numel():
                raise AssertionError(checks)
            mem_path = args.output / f"stage{stage}_memory.pt.gz"
            archive(mem_path, memory)
            emit(event="constructed", dataset=args.dataset, variant=args.variant, seed=args.seed,
                 stage=stage, seconds=elapsed, peak_gib=peak / 2**30)
            scores = evaluate(memory, data, stage, args.device, args.output) if stage in (0, len(ids)-1) else None
            row = dict(stage=stage, raw_rows=len(ids[stage]), new_rows=new_n, memory_rows=len(memory["x"]),
                construction_seconds=elapsed, peak_bytes=peak, reserved_bytes=reserved,
                entry_bytes=entry_bytes, exit_bytes=exit_bytes, cpu_peak_rss_bytes=rss.peak,
                memory_path=mem_path.name, memory_file_sha256=digest(mem_path),
                memory_hash=R.V.memory_hash(memory), checks=checks, scores=scores, audit=R.compact_audit(audit))
            result["rows"].append(row)
            save(path, result)
            emit(event="stage_complete", dataset=args.dataset, variant=args.variant, seed=args.seed,
                 stage=stage, f1=None if scores is None else scores["f1"])
            parent = memory
            del block, delta
            gc.collect()
            torch.cuda.empty_cache()
        result["status"] = "complete" if limit == len(ids) else "pilot_complete"
        save(path, result)
    except Exception as error:
        result.update(status="failed", failure=repr(error), traceback=traceback.format_exc())
        save(path, result)
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=("arxiv", "sportsfit"), required=True)
    p.add_argument("--variant", choices=VARIANTS, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--stage-limit", type=int, default=0)
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args())
