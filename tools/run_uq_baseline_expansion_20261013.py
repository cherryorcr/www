"""Resumable baseline/data expansion using the immutable final-confirmation core."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans


ARMS = ("uq_dm_edgefree", "cat_cctp", "gecc_hard", "random_coreset",
        "kcenter_coreset", "cluster_medoid", "herding_coreset", "puma_tag",
        "gcond_x_core", "sfgc_core")
SMALL_GROUPS = {"cora": [[0, 1], [2, 3], [4, 5, 6]],
                "citeseer": [[0, 1], [2, 3], [4, 5]]}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_reference(path, dataset):
    if path is None:
        return dict(groups=SMALL_GROUPS[dataset], report_dev_mod=3, budgets=[])
    value = json.loads(path.read_text(encoding="utf-8"))
    return value.get("protocol", value)


def load_data(args, reference, R):
    groups = R.V.choose_groups(reference, args.order_seed)
    manifest = dict(groups=groups, report_dev_mod=reference.get("report_dev_mod", 3))
    if args.dataset not in SMALL_GROUPS:
        return R.L.load_large(args.dataset, args.minilm, args.text_cache, args.device, manifest)
    U = R.L.U
    data = U.load_unified(args.dataset, U.DATA_CFG[args.dataset]["path"], args.minilm,
                          args.device, n_stages=3, split_mode="canonical",
                          report_dev_mod=manifest["report_dev_mod"])
    cache = args.text_cache / "text_emb_pool_only.pt"
    if cache.exists():
        embedding = torch.load(cache, map_location="cpu", weights_only=False).float()
    else:
        ids = torch.where(data["pool_mask"])[0]
        embedding = torch.zeros((len(data["x"]), 384), dtype=torch.float32)
        embedding[ids] = torch.as_tensor(data["encoder"].encode(
            [data["texts"][int(i)] for i in ids])).cpu().float()
        cache.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache.with_suffix(".tmp")
        torch.save(embedding, temporary)
        temporary.replace(cache)
        R.V.write_json(cache.parent / "cache_manifest.json", dict(
            dataset=args.dataset, pool_mask_hash=R.V.hash_tensor(data["pool_mask"]),
            encoder=args.minilm, source="GraphCLIP raw_texts; legal construction pool only",
            rows=int(len(ids)), embedding_hash=R.V.hash_tensor(embedding)))
    if tuple(embedding.shape) != (len(data["x"]), 384):
        raise AssertionError("text cache shape differs")
    if bool((embedding[~data["pool_mask"]] != 0).any()):
        raise AssertionError("non-pool text entered the cache")
    data.update(text_emb=embedding, stage_classes=groups,
                prompt_emb=data["encoder"].encode(data["prompts"]),
                real_adj=R.P.normalized_adjacency(data["edge_index"], len(data["x"]), args.device),
                x_device=data["x"].float().to(args.device))
    return data


def representative_block(delta, rows, arm, encoder, seed, device, R):
    allocation = R.M.ratio_allocation(delta["y"], rows)
    selected, class_audit = [], []
    for cls, count in allocation.items():
        ids = torch.where(delta["y"] == cls)[0]
        class_seed = seed + 1009 * int(delta["stage"]) + 37 * int(cls)
        rng = torch.Generator().manual_seed(class_seed)
        if arm == "random_coreset":
            chosen = torch.randperm(len(ids), generator=rng)[:count]
        else:
            features = torch.cat((F.normalize(delta["x"][ids].float(), dim=1),
                                  F.normalize(delta["text_emb"][ids].float(), dim=1)), 1)
            if arm == "cluster_medoid":
                points = features.numpy()
                model = MiniBatchKMeans(n_clusters=count, batch_size=max(1024, 3 * count),
                    n_init=3, max_iter=100, random_state=class_seed,
                    reassignment_ratio=0.0)
                assignment = model.fit_predict(points)
                nearest = []
                for cluster in range(count):
                    members = np.flatnonzero(assignment == cluster)
                    if len(members):
                        distance = ((points[members] - model.cluster_centers_[cluster]) ** 2).sum(1)
                        nearest.append(int(members[distance.argmin()]))
                if len(nearest) < count:
                    remaining = np.setdiff1d(np.arange(len(ids)), np.asarray(nearest))
                    distance = ((points[remaining] - points.mean(0)) ** 2).sum(1)
                    nearest.extend(remaining[np.argsort(-distance)[:count-len(nearest)]].tolist())
                chosen = torch.tensor(nearest, dtype=torch.long)
                class_audit.append(dict(cls=int(cls), clusters=count,
                    occupied_clusters=int(len(np.unique(assignment))),
                    iterations=int(model.n_iter_), inertia=float(model.inertia_)))
            else:
                points = features.to(device)
                available = torch.ones(len(ids), dtype=torch.bool, device=device)
                chosen_list = []
                if arm == "kcenter_coreset":
                    index = int(torch.randint(len(ids), (), generator=rng))
                    distance = torch.full((len(ids),), float("inf"), device=device)
                elif arm == "herding_coreset":
                    target, running = points.mean(0), torch.zeros(points.shape[1], device=device)
                else:
                    raise ValueError(arm)
                for step in range(count):
                    if arm == "herding_coreset":
                        distance = ((running + points) / (step + 1) - target).square().sum(1)
                        distance[~available] = float("inf")
                        index = int(distance.argmin())
                        running += points[index]
                    chosen_list.append(index)
                    available[index] = False
                    if arm == "kcenter_coreset":
                        distance = torch.minimum(distance, (points - points[index]).square().sum(1))
                        distance[~available] = -1
                        index = int(distance.argmax())
                chosen = torch.tensor(chosen_list)
                del points
        if len(chosen) != count or len(torch.unique(chosen)) != count:
            raise AssertionError("representative selection did not fill its budget uniquely")
        selected.extend(ids[chosen].tolist())
    selected = torch.tensor(selected, dtype=torch.long)
    texts = [" ".join(str(delta["texts"][i]).split())[:320] for i in selected.tolist()]
    block = dict(x=delta["x"][selected].clone(),
        text_emb=torch.as_tensor(encoder.encode(texts)).cpu().float(),
        labels=R.M.row_labels(allocation), mass=torch.ones(rows),
        depth=torch.zeros(rows, dtype=torch.long),
        stage=torch.full((rows,), delta["stage"], dtype=torch.long), texts=texts,
        edge_index=torch.empty((2, 0), dtype=torch.long))
    rule = {"random_coreset": "class-stratified random without replacement",
            "kcenter_coreset": "exact greedy FPS in normalized X/text joint space",
            "cluster_medoid": "MiniBatchKMeans; nearest real member to each center; deterministic empty-cluster fill",
            "herding_coreset": "exact greedy class-mean herding in normalized X/text joint space"}[arm]
    return block, dict(method=arm, allocation=allocation, selection=rule,
        source_ids=delta["ids"][selected].tolist(), class_diagnostics=class_audit,
        text_rule="selected source text; whitespace cleanup; 320 characters; same MiniLM",
        historical_raw_reads=0, official_reproduction=False,
        identity="selection control; not an additional graph-condensation paper")


def construct(arm, delta, parent, labels, data, args, R, evaluator):
    if arm in ("uq_dm_edgefree", "cat_cctp", "gecc_hard"):
        cfg = R.q_config("uq_dm_edgefree", args.construction_seed, 30, 4096)
        return evaluator.construct(arm, delta, parent, labels, cfg, data["encoder"],
                                   data["prompt_emb"], args.construction_seed, args.device)
    if arm in ("random_coreset", "kcenter_coreset", "cluster_medoid", "herding_coreset"):
        block, audit = representative_block(delta, len(labels), arm, data["encoder"],
                                             args.construction_seed, args.device, R)
    elif arm == "puma_tag":
        import puma_tag_adapter as PU
        cfg = PU.PUMAConfig(seed=args.construction_seed)
        block, audit = PU.build_puma_block(delta, len(labels), data["encoder"], cfg,
                                           args.device, "puma_full")
        block["labels"] = R.M.row_labels(audit["cat_distribution_matching"]["allocation"])
        audit.update(config=asdict(cfg), official_reproduction=False,
                     identity="PUMA-inspired pseudo-teacher reference; not a native PUMA reproduction")
    else:
        import uq_native_graph_cores_20261013 as N
        block, audit = N.build(arm, delta, parent, labels, data["encoder"],
                              args.construction_seed, args.device, args.root)
    return R.L.append_edge_free(parent, block), audit


def run(args):
    frozen = args.root / "experiments/uq_final_evidence_20261012/code"
    sys.path.insert(0, str(frozen))
    sys.path.insert(1, str(args.root / "code/v2_llm_fusion"))
    import run_unified_q_matrix_20261008 as R
    import run_uq_final_confirmation_20261012 as evaluator
    if Path(R.__file__).resolve().parent != frozen.resolve():
        raise AssertionError("the immutable construction snapshot was not imported")
    reference = read_reference(args.reference, args.dataset)
    data = load_data(args, reference, R)
    ids = R.L.stage_ids(data)
    sizes = list(map(len, ids))
    budgets, floors = R.V.choose_budgets(reference, sizes, data["stage_classes"], "fixed_rate", .05)
    hashes = {Path(m.__file__).name: sha(m.__file__) for m in (R, R.L, R.M, R.V, R.P, R.UQ, evaluator)}
    protocol = dict(name="uq-baseline-expansion-20261013-v1", dataset=args.dataset,
        groups=data["stage_classes"], budgets=budgets, stage_raw_rows=sizes,
        order_seed=args.order_seed, construction_seed=args.construction_seed,
        student_seeds=[150, 151, 152], student_steps=2048, student_batch=24,
        student=R.P.model_config(), encoder=args.minilm, memory_rate=.05,
        actual_final_rate=budgets[-1] / sum(sizes), class_floor_adjustments=floors,
        budget_mode="fixed_rate_current_legal_pool", arms=args.arms,
        pool_mask_hash=R.V.hash_tensor(data["pool_mask"]),
        report_mask_hash=R.V.hash_tensor(data["report_dev_mask"]),
        test_mask_hash=R.V.hash_tensor(data["test_mask"]),
        stage_id_hashes=[R.V.hash_tensor(v) for v in ids],
        access="current-delta-only; own frozen synthetic prefix; zero historical raw reads",
        synthetic_topology="none; common raw transductive inference graph",
        source_hashes=hashes, runner_hash=sha(__file__),
        method_selection_from_test=False, official_reproduction=False,
        primary_split="sealed_test; report_dev stored separately; no result-driven tuning",
        new_dataset_status="additional registered confirmation; the historical sealed tables are unchanged")
    if "puma_tag" in args.arms:
        import puma_tag_adapter as PU
        protocol["source_hashes"][Path(PU.__file__).name] = sha(PU.__file__)
    if any(a in ("gcond_x_core", "sfgc_core") for a in args.arms):
        import uq_native_graph_cores_20261013 as N
        protocol["native_graph_core"] = N.registration(args.root)
        protocol["source_hashes"][Path(N.__file__).name] = sha(N.__file__)
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "results.json"
    if path.exists():
        result = json.loads(path.read_text())
        if result["protocol"] != protocol:
            raise ValueError("protocol differs; preserve the old output and choose a fresh path")
    else:
        result = dict(protocol=protocol, rows=[], status="running")
    R.V.write_json(args.output / "protocol.json", protocol)
    R.V.write_json(path, result)
    for arm in args.arms:
        done = [r for r in result["rows"] if r["arm"] == arm]
        if len(done) == len(ids):
            continue
        checkpoint = args.output / f"latest_{arm}.pt"
        parent = None
        if done:
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            parent = saved["memory"]
            if saved["stage"] != len(done)-1 or R.V.memory_hash(parent) != done[-1]["memory_hash"]:
                raise AssertionError("resume checkpoint disagrees with the metrics")
        for stage in range(len(done), len(ids)):
            delta = R.L.delta_view(data, ids[stage], stage)
            new_rows = budgets[stage] - (0 if parent is None else len(parent["x"]))
            labels = R.M.row_labels(R.M.ratio_allocation(delta["y"], new_rows))
            R.V.sync(args.device)
            torch.cuda.reset_peak_memory_stats(args.device)
            started = time.perf_counter()
            memory, audit = construct(arm, delta, parent, labels, data, args, R, evaluator)
            R.V.sync(args.device)
            seconds = time.perf_counter()-started
            peak_mb = torch.cuda.max_memory_allocated(args.device)/2**20
            checks = R.V.validate_memory(memory, parent, budgets[stage])
            if memory["edge_index"].numel():
                raise AssertionError("an edge-free baseline produced stored edges")
            if "q_row_sum_max_error" in audit:
                checks.update(R.assignment_checks(audit))
                if not checks["sinkhorn_feasible"]:
                    raise AssertionError("Sinkhorn feasibility failed")
            print(json.dumps(dict(event="constructed", dataset=args.dataset, arm=arm,
                                  stage=stage, seconds=seconds, peak_mb=peak_mb)), flush=True)
            started = time.perf_counter()
            scores = evaluator.evaluate(memory, data, stage, [150, 151, 152], args.device)
            R.V.sync(args.device)
            row = dict(arm=arm, stage=stage, raw_rows=len(ids[stage]), new_rows=new_rows,
                memory_rows=len(memory["x"]), scores=scores, score=scores["report_dev"],
                construction_seconds=seconds, construction_peak_mb=peak_mb,
                student_seconds=time.perf_counter()-started, memory_hash=R.V.memory_hash(memory),
                checks=checks, audit=R.compact_audit(audit) if arm == "uq_dm_edgefree" else audit)
            temporary = checkpoint.with_suffix(".tmp")
            torch.save(dict(stage=stage, memory=memory), temporary)
            temporary.replace(checkpoint)
            result["rows"].append(row)
            R.V.write_json(path, result)
            print(json.dumps(dict(event="evaluated", dataset=args.dataset, arm=arm, stage=stage,
                report_f1=scores["report_dev"]["final_f1"], test_f1=scores["sealed_test"]["final_f1"],
                test_acc=scores["sealed_test"]["accuracy"])), flush=True)
            parent = memory
        del memory, parent
        gc.collect()
        torch.cuda.empty_cache()
    result["status"] = "complete"
    R.V.write_json(path, result)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("/home/pc/research/TAG_condensation"))
    p.add_argument("--dataset", choices=("arxiv", "sportsfit", "bookhis", "cora", "citeseer"), required=True)
    p.add_argument("--reference", type=Path)
    p.add_argument("--text-cache", type=Path, required=True)
    p.add_argument("--minilm", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--order-seed", type=int, default=9101)
    p.add_argument("--construction-seed", type=int, default=271006)
    p.add_argument("--arms", nargs="+", choices=ARMS, required=True)
    a = p.parse_args()
    torch.set_num_threads(2)
    run(a)
