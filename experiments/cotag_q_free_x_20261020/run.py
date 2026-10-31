"""Paired primary-protocol feature-construction experiment, with immutable inputs."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(os.environ.get("COTAG_ROOT", "/home/pc/research/TAG_condensation"))
HERE = Path(__file__).resolve().parent
FROZEN = Path(os.environ.get("COTAG_FROZEN_CODE", str(ROOT / "experiments/uq_final_evidence_20261012/code")))
sys.path.insert(0, str(FROZEN))
sys.path.insert(1, str(ROOT / "code/v2_llm_fusion"))
import run_uq_final_confirmation_20261012 as E
import graphtext_sage_transfer as S

sys.path.insert(0, str(HERE))
import independent_x as C

R = E.R
FULL, FREE = "uq_dm_edgefree", "independent_x"
SEEDS = [150, 151, 152]


def plain(value):
    return json.loads(json.dumps(value))


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    tmp.replace(path)


def emit(**value):
    print(json.dumps(value), flush=True)


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


def sage_evaluate(memory, data, device):
    stage = len(data["stage_classes"]) - 1
    seen = [int(c) for group in data["stage_classes"] for c in group]
    old = [int(c) for group in data["stage_classes"][:stage] for c in group]
    new = list(map(int, data["stage_classes"][stage]))
    prompt = data["prompt_emb"].float().to(device)
    y = data["y"].long().cpu()
    masks = {name: data[key].bool().cpu() for name, key in (
        ("report_dev", "report_dev_mask"), ("sealed_test", "test_mask"))}
    if bool((masks["report_dev"] & masks["sealed_test"]).any()):
        raise AssertionError("report masks overlap")
    if any(bool((mask & data["pool_mask"].cpu()).any()) for mask in masks.values()):
        raise AssertionError("construction and reporting overlap")
    rows = {name: [] for name in masks}
    for seed in SEEDS:
        R.V.sync(device)
        torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        model = S.train_sage_student(memory, prompt, seed, device, 2048, lr=.001, batch=24)
        R.V.sync(device)
        seconds = time.perf_counter() - start
        peak = torch.cuda.max_memory_allocated(device) / 2**20
        with torch.no_grad():
            pred = model.logits(data["x_device"], data["real_adj"], prompt).argmax(1).cpu()
        for name, mask in masks.items():
            yy, pp = y[mask].numpy(), pred[mask].numpy()
            per_class = dict(zip(seen, 100 * f1_score(yy, pp, labels=seen, average=None, zero_division=0)))
            rows[name].append(dict(seed=seed,
                final_f1=float(100 * f1_score(yy, pp, labels=seen, average="macro", zero_division=0)),
                accuracy=float(100 * accuracy_score(yy, pp)),
                old_f1=float(np.mean([per_class[c] for c in old])),
                new_f1=float(np.mean([per_class[c] for c in new])),
                train_seconds=seconds, train_peak_mb=peak))
        del model
        torch.cuda.empty_cache()
    output = {}
    for name, items in rows.items():
        aggregate = dict(split=name, student_steps=2048, student_batch=24,
                         student_seeds=SEEDS, eval_rows=int(masks[name].sum()), per_student=items)
        for key in ("final_f1", "accuracy", "old_f1", "new_f1"):
            aggregate[key] = float(np.mean([r[key] for r in items]))
            aggregate[key + "_std"] = float(np.std([r[key] for r in items], ddof=1))
        output[name] = aggregate
    return output


def load(args):
    primary_path = ROOT / "results/uq_final_evidence_20261012/confirmation" / (
        f"{args.dataset}_o{args.order_seed}_c{args.construction_seed}/results.json")
    primary = read(primary_path)
    if primary["status"] != "complete":
        raise AssertionError("primary run not complete")
    sources = primary["protocol"]["source_hashes"]
    hashes = {Path(m.__file__).name: E.file_hash(m.__file__)
              for m in (R, R.UQ, R.M, R.P, R.L, R.V, R.I, R.I.D)}
    if hashes != sources or E.file_hash(E.__file__) != primary["protocol"]["runner_hash"]:
        raise AssertionError(dict(source_mismatch=hashes))
    if E.file_hash(S.__file__) != "139bb60ff99995f46556e6f878f21093cba65dc5dcc1d356770bd994f335c428":
        raise AssertionError("GraphSAGE source changed")
    ref = R.L.read(ROOT / f"experiments/uq_final_evidence_20261012/{args.dataset}_reference.json")
    groups = R.V.choose_groups(ref, args.order_seed)
    data = R.L.load_large(args.dataset,
        "/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip",
        ROOT / f"results/public_baselines_text_cache_20261001/{args.dataset}", args.device,
        dict(groups=groups, report_dev_mod=ref.get("report_dev_mod", 3)))
    ids = R.L.stage_ids(data)
    sizes = list(map(len, ids))
    budgets, floors = R.V.choose_budgets(ref, sizes, groups, "fixed_rate", .05)
    base = R.q_config(FULL, args.construction_seed, 30, 4096)
    protocol = dict(name="cotag-independent-x-20261020-v1", dataset=args.dataset,
        order_seed=args.order_seed, construction_seed=args.construction_seed,
        groups=groups, stage_raw_rows=sizes, budgets=budgets, class_floor_adjustments=floors,
        student=R.P.model_config(), student_steps=2048, student_batch=24, student_seeds=SEEDS,
        pool_mask_hash=R.V.hash_tensor(data["pool_mask"]),
        report_mask_hash=R.V.hash_tensor(data["report_dev_mask"]),
        test_mask_hash=R.V.hash_tensor(data["test_mask"]),
        stage_id_hashes=[R.V.hash_tensor(v) for v in ids], q_config=asdict(base),
        memory_rate=.05, actual_final_rate=budgets[-1] / sum(sizes),
        source_hashes=hashes, runner_hash=E.file_hash(__file__),
        control_hash=E.file_hash(C.__file__), sage_hash=E.file_hash(S.__file__),
        sage_student=S.model_config(), primary_path=str(primary_path),
        primary_file_hash=E.file_hash(primary_path), arms=[FULL, FREE],
        arm_changes={FREE: dict(lambda_x_anchor=0., feature_parameterization="independent X",
            feature_projection="none; trust_radius is not applied", propagation_target="still Q-derived")},
        evaluation="post-hoc report-dev and previously inspected test; not fresh blind selection",
        access="current-delta-only constructor; own frozen synthetic prefix; no historical raw reads",
        physical_loader="global transductive evaluation cache remains on device",
        method_selection_from_test=False, official_reproduction=False)
    protocol = plain(protocol)
    keys = ("groups", "stage_raw_rows", "budgets", "student", "student_steps", "student_batch",
            "student_seeds", "pool_mask_hash", "report_mask_hash", "test_mask_hash", "stage_id_hashes", "q_config")
    checks = {k: protocol[k] == primary["protocol"][k] for k in keys}
    if not all(checks.values()):
        raise AssertionError(dict(protocol_mismatch=checks))
    protocol["primary_protocol_checks"] = checks
    return primary_path, primary, data, ids, base, protocol


def run(args):
    path = args.output / "results.json"
    if path.exists():
        old = read(path)
        if old.get("status") == "complete":
            emit(event="skip_complete", path=str(path))
            return
        raise FileExistsError("Preserve incomplete results; use a fresh output path")
    if shutil.disk_usage(ROOT).free < 8 * 2**30:
        raise RuntimeError("less than 8 GiB disk available")
    primary_path, primary, data, ids, base, protocol = load(args)
    result = dict(protocol=protocol, status="running", rows=[], sage={})
    save(args.output / "protocol.json", protocol)
    save(path, result)
    original = [r for r in primary["rows"] if r["arm"] == FULL]
    if len(original) != len(ids):
        raise AssertionError("incomplete full-arm reference")
    try:
        for arm in (FULL, FREE):
            full_archive = primary_path.parent / f"final_{FULL}.pt.gz"
            if arm == FULL and full_archive.exists():
                memory = load_memory(full_archive)
                if R.V.memory_hash(memory) != original[-1]["memory_hash"]:
                    raise AssertionError("saved full memory hash mismatch")
                R.V.validate_memory(memory, None, protocol["budgets"][-1])
                for row in original:
                    result["rows"].append(dict(row, primary_metrics_reused=True,
                        memory_reconstructed=False, memory_archive=str(full_archive)))
                save(path, result)
                emit(event="reused_full_memory", dataset=args.dataset, seed=args.construction_seed)
            else:
                parent = None
                cfg = base if arm == FULL else replace(base, lambda_x_anchor=0.)
                for stage, raw_ids in enumerate(ids):
                    delta = R.L.delta_view(data, raw_ids, stage)
                    new_n = protocol["budgets"][stage] - (0 if parent is None else len(parent["x"]))
                    labels = R.M.row_labels(R.M.ratio_allocation(delta["y"], new_n))
                    diag, trajectory = {}, []

                    def observe(info):
                        raw_text = info["delta"]["text_emb"].float().to(args.device)
                        diag.update(C.diagnostics(info["q"], info["q0"], raw_text))

                    def observe_step(info):
                        trajectory.append(dict(step=info["step"] + 1,
                            **{k: float(info[k].detach()) for k in (
                                "loss", "gm", "distribution", "propagation")}))

                    R.V.sync(args.device)
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats(args.device)
                    start = time.perf_counter()
                    if arm == FULL:
                        block, audit = R.UQ.optimize_stage(delta, parent, labels, data["encoder"],
                            data["prompt_emb"], cfg, args.device, use_boundary=False, use_topology=False,
                            diagnostic_callback=observe, optimization_callback=observe_step)
                        audit["diagnostics"] = diag
                        audit["component_trajectory"] = trajectory
                    else:
                        block, audit = C.optimize(delta, parent, labels, data["encoder"],
                            data["prompt_emb"], cfg, args.device)
                    R.V.sync(args.device)
                    seconds = time.perf_counter() - start
                    peak = torch.cuda.max_memory_allocated(args.device) / 2**20
                    memory = R.L.append_edge_free(parent, block)
                    checks = R.V.validate_memory(memory, parent, protocol["budgets"][stage])
                    checks.update(R.assignment_checks(audit))
                    if not checks["sinkhorn_feasible"] or memory["edge_index"].numel():
                        raise AssertionError(checks)
                    h = R.V.memory_hash(memory)
                    reconstruction_hash_match = h == original[stage]["memory_hash"] if arm == FULL else None
                    emit(event="constructed", dataset=args.dataset, seed=args.construction_seed,
                         arm=arm, stage=stage, seconds=seconds, peak_mb=peak,
                         reconstruction_hash_match=reconstruction_hash_match)
                    if arm == FULL and reconstruction_hash_match:
                        scores = original[stage]["scores"]
                        student_seconds, reused = 0., True
                    else:
                        start = time.perf_counter()
                        scores = E.evaluate(memory, data, stage, SEEDS, args.device)
                        R.V.sync(args.device)
                        student_seconds, reused = time.perf_counter() - start, False
                    row = dict(arm=arm, stage=stage, raw_rows=len(raw_ids), new_rows=new_n,
                        memory_rows=len(memory["x"]), memory_hash=h, checks=checks,
                        construction_seconds=seconds, construction_peak_mb=peak,
                        scores=scores, score=scores["report_dev"], student_seconds=student_seconds,
                        primary_metrics_reused=reused, memory_reconstructed=arm == FULL,
                        reconstruction_hash_match=reconstruction_hash_match,
                        original_memory_hash=original[stage]["memory_hash"] if arm == FULL else None,
                        original_scores=original[stage]["scores"] if arm == FULL else None,
                        audit=R.compact_audit(audit))
                    result["rows"].append(row)
                    save(path, result)
                    emit(event="evaluated", dataset=args.dataset, seed=args.construction_seed,
                        arm=arm, stage=stage, report_f1=scores["report_dev"]["final_f1"],
                        test_f1=scores["sealed_test"]["final_f1"])
                    parent = memory
                    del block
                    torch.cuda.empty_cache()
                archive(args.output / f"final_{arm}.pt.gz", memory)
            result["sage"][arm] = dict(memory_hash=R.V.memory_hash(memory),
                scores=sage_evaluate(memory, data, args.device))
            save(path, result)
            emit(event="sage_complete", dataset=args.dataset, seed=args.construction_seed,
                arm=arm, scores={k: dict(f1=v["final_f1"], accuracy=v["accuracy"])
                                 for k, v in result["sage"][arm]["scores"].items()})
            del memory
            torch.cuda.empty_cache()
        result["status"] = "complete"
        save(path, result)
        emit(event="complete", dataset=args.dataset, seed=args.construction_seed)
    except BaseException as error:
        result["status"] = "failed"
        result["failure"] = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
        save(path, result)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("arxiv", "sportsfit", "bookhis"), required=True)
    parser.add_argument("--construction-seed", type=int, required=True)
    parser.add_argument("--order-seed", type=int, default=9101)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(2)
    run(args)
