"""Registered true-rate confirmation; no result-dependent configuration changes."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import gzip
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

import run_unified_q_matrix_20261008 as R


JOINT_REPLACEMENT_ARM = "uq_joint_replace"


def construction_config(arm, seed):
    base = R.q_config("uq_dm_edgefree", seed, 30, 4096)
    if arm == JOINT_REPLACEMENT_ARM:
        return replace(base, lambda_distribution=0.0, lambda_joint_distribution=1.0)
    return base


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evaluate(memory, data, stage, seeds, device, steps=2048, batch=24):
    """One frozen training run, two disjoint read-only reporting masks."""
    prompt = data["prompt_emb"].float().to(device)
    y = data["y"].long().cpu()
    seen = [int(c) for group in data["stage_classes"][:stage + 1] for c in group]
    old = [int(c) for group in data["stage_classes"][:stage] for c in group]
    new = list(map(int, data["stage_classes"][stage]))
    masks = {name: data[key].bool().cpu() & torch.isin(y, torch.tensor(seen))
             for name, key in (("report_dev", "report_dev_mask"), ("sealed_test", "test_mask"))}
    if bool((masks["report_dev"] & masks["sealed_test"]).any()):
        raise AssertionError("evaluation masks overlap")
    if any(bool((mask & data["pool_mask"].cpu()).any()) for mask in masks.values()):
        raise AssertionError("evaluation nodes entered construction pool")
    rows = {name: [] for name in masks}
    for seed in seeds:
        model = R.P.train_prompt_student(memory, prompt, int(seed), device, steps, lr=.001, batch=batch)
        with torch.no_grad():
            pred = model.logits(data["x_device"], data["real_adj"], prompt).argmax(1).cpu()
        for name, mask in masks.items():
            yy, pp = y[mask].numpy(), pred[mask].numpy()
            per_class = dict(zip(seen, 100 * f1_score(yy, pp, labels=seen, average=None, zero_division=0)))
            rows[name].append(dict(seed=int(seed),
                final_f1=float(100 * f1_score(yy, pp, labels=seen, average="macro", zero_division=0)),
                accuracy=float(100 * accuracy_score(yy, pp)),
                old_f1=float(np.mean([per_class[c] for c in old])) if old else None,
                new_f1=float(np.mean([per_class[c] for c in new]))))
        del model
    output = {}
    for name, items in rows.items():
        aggregate = {"split": name, "student_steps": steps, "student_batch": batch,
                     "student_seeds": list(seeds), "eval_rows": int(masks[name].sum()), "per_student": items}
        for key in ("final_f1", "accuracy", "old_f1", "new_f1"):
            values = [r[key] for r in items if r[key] is not None]
            aggregate[key] = float(np.mean(values)) if values else None
            aggregate[key + "_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.
        output[name] = aggregate
    return output


def construct(arm, delta, parent, labels, cfg, encoder, prompt, seed, device):
    if arm in ("uq_dm_edgefree", JOINT_REPLACEMENT_ARM):
        block, audit = R.UQ.optimize_stage(delta, parent, labels, encoder, prompt, cfg,
                                         device, use_boundary=False, use_topology=False)
    elif arm in ("cat_cctp", "gecc_hard"):
        block, audit = R.make_baseline(delta, len(labels), arm, encoder, seed, device)
    elif arm in ("gecc_soft", "tagsam_selector_response", "stad_core"):
        import uq_high_fidelity_baselines_20261012 as H
        block, audit = H.build(arm, delta, parent, labels, encoder, prompt, seed, device)
    else:
        import uq_controlled_assignment_20261012 as C
        block, audit = C.build(arm, delta, parent, labels, encoder, prompt, cfg, device)
    return R.L.append_edge_free(parent, block), audit


def run(args):
    reference = R.L.read(args.reference)
    groups = R.V.choose_groups(reference, args.order_seed)
    data = R.L.load_large(args.dataset, args.minilm, args.text_cache, args.device,
                         dict(groups=groups, report_dev_mod=reference.get("report_dev_mod", 3)))
    ids = R.L.stage_ids(data)
    sizes = list(map(len, ids))
    budgets, floors = R.V.choose_budgets(reference, sizes, groups, "fixed_rate", .05)
    cfg = construction_config("uq_dm_edgefree", args.construction_seed)
    joint_arm = JOINT_REPLACEMENT_ARM in args.arms
    if joint_arm:
        candidate = construction_config(JOINT_REPLACEMENT_ARM, args.construction_seed)
        differences = {key: [asdict(cfg)[key], asdict(candidate)[key]]
                       for key in asdict(cfg) if asdict(cfg)[key] != asdict(candidate)[key]}
        if differences != {"lambda_distribution": [1.0, 0.0],
                           "lambda_joint_distribution": [0.0, 1.0]}:
            raise AssertionError(f"unexpected joint replacement configuration: {differences}")
    protocol = dict(name="uq-final-frozen-confirmation-20261012-v1", dataset=args.dataset,
        groups=groups, order_seed=args.order_seed, construction_seed=args.construction_seed,
        student_seeds=[150, 151, 152], student_steps=2048, student_batch=24,
        student=R.P.model_config(), arms=args.arms, q_config=asdict(cfg),
        budgets=budgets, stage_raw_rows=sizes, memory_rate=.05,
        actual_final_rate=budgets[-1] / sum(sizes), class_floor_adjustments=floors,
        budget_mode="fixed_rate_current_legal_pool", reference_budgets=reference["budgets"],
        pool_mask_hash=R.V.hash_tensor(data["pool_mask"]),
        report_mask_hash=R.V.hash_tensor(data["report_dev_mask"]),
        test_mask_hash=R.V.hash_tensor(data["test_mask"]),
        stage_id_hashes=[R.V.hash_tensor(v) for v in ids],
        evaluation_split="report_dev and sealed_test stored separately; protocol frozen before either",
        access="current-delta-only; own frozen synthetic prefix; zero historical raw reads",
        synthetic_topology="none; common full raw transductive inference graph",
        source_hashes={Path(m.__file__).name: file_hash(m.__file__)
                       for m in (R, R.UQ, R.M, R.P, R.L, R.V, R.I, R.I.D)},
        runner_hash=file_hash(__file__), method_selection_from_test=False,
        official_reproduction=False,
        persistence="metrics/hashes; optional final compressed memory; no dense Q persisted")
    if joint_arm:
        protocol["joint_replacement"] = dict(
            arm=JOINT_REPLACEMENT_ARM, config=asdict(candidate), changes=differences,
            provenance="post-hoc candidate; additive paired MMD was negative on prior report-dev",
            inference="not an independent blind sealed selection")
        protocol["source_hashes"]["unified_q_pair_distribution.py"] = file_hash(R.UQ.J.__file__)
    if args.smoke_stage:
        if args.arms != [JOINT_REPLACEMENT_ARM]:
            raise ValueError("smoke stage is restricted to the joint replacement arm")
        protocol["smoke_stage"] = 0
    extensions = []
    if any(a in ("gecc_soft", "tagsam_selector_response", "stad_core") for a in args.arms):
        extensions.append("uq_high_fidelity_baselines_20261012.py")
    if any(a.startswith("control_") for a in args.arms):
        extensions.append("uq_controlled_assignment_20261012.py")
    for name in extensions:
        path = Path(__file__).parent / name
        if path.exists():
            protocol["source_hashes"][name] = file_hash(path)
    path = args.output / "results.json"
    if path.exists():
        result = json.loads(path.read_text(encoding="utf-8"))
        if result["protocol"] != protocol:
            raise ValueError("existing protocol differs; use a fresh directory")
    else:
        result = dict(protocol=protocol, rows=[], status="running")
    R.V.write_json(args.output / "protocol.json", protocol)
    R.V.write_json(path, result)
    for arm in args.arms:
        done = [r for r in result["rows"] if r["arm"] == arm]
        if len(done) == len(ids):
            continue
        if done:
            raise RuntimeError("partial arm: preserve evidence and retry in a fresh directory")
        parent = None
        for stage, raw_ids in enumerate(ids):
            delta = R.L.delta_view(data, raw_ids, stage)
            new_n = budgets[stage] - (0 if parent is None else len(parent["x"]))
            labels = R.M.row_labels(R.M.ratio_allocation(delta["y"], new_n))
            R.V.sync(args.device)
            torch.cuda.reset_peak_memory_stats(args.device)
            started = time.perf_counter()
            memory, audit = construct(arm, delta, parent, labels,
                                     construction_config(arm, args.construction_seed), data["encoder"],
                                     data["prompt_emb"], args.construction_seed, args.device)
            R.V.sync(args.device)
            seconds = time.perf_counter() - started
            peak = torch.cuda.max_memory_allocated(args.device) / 2**20
            checks = R.V.validate_memory(memory, parent, budgets[stage])
            if memory["edge_index"].numel():
                raise AssertionError("EF confirmation acquired synthetic edges")
            if "q_row_sum_max_error" in audit:
                checks.update(R.assignment_checks(audit))
                if not checks["sinkhorn_feasible"]:
                    R.V.write_json(args.output / f"failure_{arm}_s{stage}.json", dict(checks=checks, audit=audit))
                    raise AssertionError("assignment feasibility failed")
            if args.smoke_stage:
                joint = audit.get("joint_pair_distribution", {})
                if not joint.get("enabled") or not any(
                        (step.get("joint_distribution_q_gradient_norm") or 0) > 0
                        for step in audit.get("trace", [])):
                    raise AssertionError("joint objective did not reach Q")
                R.V.write_json(args.output / "smoke.json", dict(
                    dataset=args.dataset, arm=arm, stage=stage, seconds=seconds,
                    peak_mb=peak, checks=checks, audit=R.compact_audit(audit),
                    memory_hash=R.V.memory_hash(memory)))
                result["status"] = "smoke_complete"
                R.V.write_json(path, result)
                print(json.dumps(dict(event="smoke_complete", arm=arm, stage=stage,
                                      seconds=seconds, peak_mb=peak)), flush=True)
                return
            print(json.dumps(dict(event="constructed", dataset=args.dataset, arm=arm, stage=stage,
                                  seconds=seconds, peak_mb=peak)), flush=True)
            started = time.perf_counter()
            scores = evaluate(memory, data, stage, [150, 151, 152], args.device)
            R.V.sync(args.device)
            row = dict(arm=arm, stage=stage, raw_rows=len(raw_ids), new_rows=new_n,
                memory_rows=len(memory["x"]), score=scores["report_dev"], scores=scores,
                construction_seconds=seconds, construction_peak_mb=peak,
                student_seconds=time.perf_counter() - started,
                memory_hash=R.V.memory_hash(memory), checks=checks, audit=R.compact_audit(audit))
            result["rows"].append(row)
            R.V.write_json(path, result)
            print(json.dumps(dict(event="evaluated", arm=arm, stage=stage,
                report_f1=scores["report_dev"]["final_f1"], test_f1=scores["sealed_test"]["final_f1"])), flush=True)
            parent = memory
        if args.save_final:
            archive = args.output / f"final_{arm}.pt.gz"
            with gzip.open(archive.with_suffix(".gz.tmp"), "wb", compresslevel=3) as stream:
                torch.save(memory, stream)
            archive.with_suffix(".gz.tmp").replace(archive)
        del parent, memory
        torch.cuda.empty_cache()
    result["status"] = "complete"
    R.V.write_json(path, result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("arxiv", "sportsfit", "bookhis"), required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--text-cache", type=Path, required=True)
    parser.add_argument("--minilm", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--order-seed", type=int)
    parser.add_argument("--construction-seed", type=int, default=271006)
    parser.add_argument("--arms", nargs="+", default=["uq_dm_edgefree", "cat_cctp", "gecc_hard"])
    parser.add_argument("--save-final", action="store_true")
    parser.add_argument("--smoke-stage", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(2)
    run(args)
