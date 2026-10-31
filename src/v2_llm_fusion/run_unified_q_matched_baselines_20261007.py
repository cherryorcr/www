"""Public mechanism adapters using the frozen In-loop Q evaluator and inputs.

These are explicitly adapters, not official end-to-end reproductions.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F
import external_continual_mechanisms as M
import graphtext_prompt_student as P
import puma_tag_adapter as PU
import run_unified_q_large_smoke_20261007 as L
import run_unified_q_revision_20261007 as V


ARMS = ("random_coreset", "kcenter_coreset", "gecc_hard", "puma_tag")


def coreset(delta, rows, arm, encoder, seed, device):
    allocation = M.ratio_allocation(delta["y"], rows)
    labels = M.row_labels(allocation)
    selected = []
    for cls, count in allocation.items():
        ids = torch.where(delta["y"] == cls)[0]
        rng = torch.Generator().manual_seed(seed + 1009 * delta["stage"] + 37 * cls)
        if arm == "random_coreset":
            chosen = torch.randperm(len(ids), generator=rng)[:count]
        else:
            features = torch.cat((F.normalize(delta["x"][ids].float(), dim=1),
                                  F.normalize(delta["text_emb"][ids].float(), dim=1)), 1).to(device)
            chosen_list = [int(torch.randint(len(ids), (), generator=rng))]
            distance = (features - features[chosen_list[0]]).square().sum(1)
            for _ in range(1, count):
                distance[chosen_list[-1]] = -1
                index = int(distance.argmax())
                chosen_list.append(index)
                distance = torch.minimum(distance, (features - features[index]).square().sum(1))
            chosen = torch.tensor(chosen_list)
        selected.extend(ids[chosen].tolist())
    selected = torch.tensor(selected, dtype=torch.long)
    texts = [" ".join(str(delta["texts"][i]).split())[:320] for i in selected.tolist()]
    block = dict(x=delta["x"][selected].clone(), text_emb=torch.as_tensor(encoder.encode(texts)).cpu().float(),
        labels=labels, mass=torch.ones(rows), depth=torch.zeros(rows, dtype=torch.long),
        stage=torch.full((rows,), delta["stage"], dtype=torch.long), texts=texts,
        edge_index=torch.empty((2, 0), dtype=torch.long))
    audit = dict(method=arm, selection="class-ratio stratified; graph+text exact FPS" if arm == "kcenter_coreset" else "class-ratio stratified random",
        source_ids=delta["ids"][selected].tolist(), historical_raw_reads=0,
        text_rule="one actual selected node; whitespace cleanup; 320 chars; re-encode",
        topology="self-loop for matched common operator", official_reproduction=False)
    return block, audit


def construct(delta, rows, arm, encoder, seed, device):
    cfg = M.Config(seed=seed, cat_steps=3000, text_sources=5, text_mode="nearest")
    if arm in ("random_coreset", "kcenter_coreset"):
        return coreset(delta, rows, arm, encoder, seed, device)
    if arm == "gecc_hard":
        block, audit = M.gecc_block(delta, rows, encoder, cfg, device)
        block["labels"] = M.row_labels(audit["allocation"])
        audit.update(method="GECC-hard current-delta TAG adapter", official_reproduction=False,
                     config=asdict(cfg), historical_raw_reads=0)
        return block, audit
    if arm == "puma_tag":
        config = PU.PUMAConfig(seed=seed)
        block, audit = PU.build_puma_block(delta, rows, encoder, config, device, "puma_full")
        block["labels"] = M.row_labels(audit["cat_distribution_matching"]["allocation"])
        audit["config"] = asdict(config)
        audit["limitation"] = "PUMA-inspired pseudo-teacher on a deterministic withheld subset of labeled pool; not official PUMA"
        return block, audit
    raise ValueError(arm)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--text-cache", type=Path, required=True)
    p.add_argument("--minilm", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    p.add_argument("--order-seed", type=int)
    p.add_argument("--construction-seed", type=int, default=271006)
    p.add_argument("--budget-mode", choices=("fixed_schedule",), default="fixed_schedule")
    args = p.parse_args()
    torch.set_num_threads(2)
    ref = L.read(args.reference)
    groups = V.choose_groups(ref, args.order_seed)
    data = L.load_large(args.dataset, args.minilm, args.text_cache, args.device, dict(ref, groups=groups))
    ids = L.stage_ids(data)
    budgets = ref["budgets"]
    protocol = dict(dataset=args.dataset, groups=groups, budgets=budgets, budget_mode=args.budget_mode,
        order_seed=args.order_seed or ref["order_seed"], construction_seed=args.construction_seed,
        student=P.model_config(), student_seeds=[150, 151, 152], student_steps=2048, student_batch=24,
        report_dev_mod=ref["report_dev_mod"], report_mask_hash=V.hash_tensor(data["report_dev_mask"]),
        pool_mask_hash=V.hash_tensor(data["pool_mask"]), stage_id_hashes=[V.hash_tensor(i) for i in ids],
        evaluation_split="report-dev; not sealed test", access="current delta plus frozen synthetic prefix",
        historical_raw_reads_by_constructor=0, boundary_enabled=False, arms=args.arms,
        source_hashes={Path(m.__file__).name: hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
                       for m in (M, P, PU, L, V)},
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        baseline_status="mechanism adapters, NOT official reproductions")
    path = args.output / "results.json"
    result = json.loads(path.read_text()) if path.exists() else dict(protocol=protocol, rows=[], status="running")
    if result["protocol"] != protocol:
        raise ValueError("different protocol or code")
    V.write_json(args.output / "protocol.json", protocol)
    for arm in args.arms:
        done = [r for r in result["rows"] if r["arm"] == arm]
        checkpoint = args.output / f"latest_{arm}.pt"
        parent = None
        if done:
            saved = torch.load(checkpoint, weights_only=False, map_location="cpu")
            parent = saved["memory"]
            if saved["stage"] != len(done)-1 or V.memory_hash(parent) != done[-1]["memory_hash"]:
                raise AssertionError("checkpoint mismatch")
        for stage in range(len(done), len(ids)):
            delta = L.delta_view(data, ids[stage], stage)
            new_n = budgets[stage] - (0 if parent is None else len(parent["x"]))
            torch.cuda.reset_peak_memory_stats(args.device)
            V.sync(args.device)
            started = time.perf_counter()
            block, audit = construct(delta, new_n, arm, data["encoder"], args.construction_seed, args.device)
            memory = L.append_edge_free(parent, block)
            V.sync(args.device)
            construction = time.perf_counter()-started
            peak = torch.cuda.max_memory_allocated(args.device)/2**20
            checks = V.validate_memory(memory, parent, budgets[stage])
            print(json.dumps(dict(event="constructed", arm=arm, stage=stage, seconds=construction)), flush=True)
            started = time.perf_counter()
            score = L.score(memory, data, stage, [150, 151, 152], args.device, 2048, 24)
            V.sync(args.device)
            student_seconds = time.perf_counter()-started
            tmp = checkpoint.with_suffix(".tmp")
            torch.save(dict(stage=stage, memory=memory), tmp)
            tmp.replace(checkpoint)
            result["rows"].append(dict(arm=arm, stage=stage, raw_rows=len(ids[stage]), new_rows=new_n,
                memory_rows=budgets[stage], score=score, construction_seconds=construction,
                student_seconds=student_seconds, construction_peak_mb=peak,
                memory_hash=V.memory_hash(memory), audit=audit, checks=checks))
            V.write_json(path, result)
            print(json.dumps(dict(event="evaluated", arm=arm, stage=stage, f1=score["final_f1"])), flush=True)
            parent = memory
    result["status"] = "complete"
    V.write_json(path, result)


if __name__ == "__main__":
    main()
