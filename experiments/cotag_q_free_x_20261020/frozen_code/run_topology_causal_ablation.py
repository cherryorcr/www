"""Frozen topology-causal ablation for CTAG-Append.

This runner keeps synthetic content, budgets, rows and pair counts fixed and
only swaps the internal-block and cross-block topology components. It is
resumable per memory and per student fit, and intentionally does not add new
budget, text, response or refresh modules.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import run_unified_v3_continual as U
import run_link_prediction_transfer as L
import large_lp_support as B
import safe_joint_kernel_append as S
import small_comparison_metrics as M


ARMS = ("c0", "cross_only", "internal_only", "t_degree",
        "random_ring", "degree_shuffle")
ORDERS = (49011, 49012)
SPLITS = (49101, 49102)
SEEDS = tuple(range(210, 216))
METHOD_SEED = 9001
PROTOCOL = "TOPOLOGY_CAUSAL_ABLATION_PROTOCOL_ZH_2026-09-21.md"
FIELDS = ("x", "text_emb", "mass", "depth", "stage")
T_WEIGHTS = (0.0, 1.0, 0.0)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def save_pt(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, tmp)
    tmp.replace(path)


def reset(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def fresh_groups(groups, seed):
    flat = sorted(c for group in groups for c in group)
    shuffled = np.random.default_rng(seed).permutation(flat).tolist()
    cuts = np.cumsum([0] + [len(group) for group in groups])
    return [sorted(shuffled[cuts[i]:cuts[i + 1]]) for i in range(len(groups))]


def unique_pairs(edge, n):
    return L.pairs(edge, n)


def components(edge, keep, total):
    pairs = unique_pairs(edge, total)
    old = pairs[:, (pairs[0] < keep) & (pairs[1] < keep)]
    internal = pairs[:, (pairs[0] >= keep) & (pairs[1] >= keep)]
    cross = pairs[:, (pairs[0] < keep) ^ (pairs[1] < keep)]
    return old, internal, cross


def undirected(pairs):
    return L.bidirectional(pairs) if pairs.numel() else torch.empty((2, 0), dtype=torch.long)


def apply_new_permutation(pairs, keep, new_n, permutation):
    out = pairs.clone()
    for row in range(2):
        mask = out[row] >= keep
        local = out[row, mask] - keep
        out[row, mask] = keep + permutation[local]
    return out


def build_combined(old, block, keep, internal_source, cross_source):
    total = keep + len(block["x"])
    old_pairs, _, _ = components(old["edge_index"], keep, keep) if old is not None else \
        (torch.empty((2, 0), dtype=torch.long),) * 3
    _, internal, cross = components(internal_source, keep, total)
    _, _, cross_selected = components(cross_source, keep, total)
    out = {key: (torch.cat((old[key], block[key])) if old is not None
                 else block[key].clone()) for key in FIELDS}
    out["texts"] = (old["texts"] + block["texts"]) if old is not None else list(block["texts"])
    old_edge = old["edge_index"] if old is not None else torch.empty((2, 0), dtype=torch.long)
    out["edge_index"] = torch.cat((old_edge, undirected(internal),
                                    undirected(cross_selected)), dim=1)
    return out


def degree_stats(memory):
    n = len(memory["x"])
    pairs = unique_pairs(memory["edge_index"], n)
    deg = torch.bincount(torch.cat((pairs[0], pairs[1])), minlength=n).float()
    mean = float(deg.mean()) if n else 0.0
    gini = float((torch.abs(deg[:, None] - deg[None, :]).mean() /
                  (2 * mean)).item()) if mean else 0.0
    return {"rows": n, "pairs": int(pairs.shape[1]),
            "isolated": int((deg == 0).sum()), "max_degree": int(deg.max()) if n else 0,
            "mean_degree": mean, "degree_gini": gini}


def norm_degree(pairs, n):
    deg = torch.ones(n, dtype=torch.float64)
    if pairs.numel():
        deg.scatter_add_(0, pairs[0], torch.ones(pairs.shape[1], dtype=torch.float64))
        deg.scatter_add_(0, pairs[1], torch.ones(pairs.shape[1], dtype=torch.float64))
    return deg


def propagation_drift(old, current, keep):
    if old is None or keep == 0:
        return {"old_block_row_l1": 0.0, "full_old_row_l1": 0.0,
                "cross_norm_mass": 0.0, "max_relative_load": 0.0}
    old_pairs = unique_pairs(old["edge_index"], keep)
    current_pairs = unique_pairs(current["edge_index"], len(current["x"]))
    old_deg = norm_degree(old_pairs, keep)
    new_deg = norm_degree(current_pairs, len(current["x"]))
    old_l1 = torch.zeros(keep, dtype=torch.float64)
    cross_mass = torch.zeros(keep, dtype=torch.float64)
    for u, v in old_pairs.T.tolist():
        old_w = 1.0 / torch.sqrt(old_deg[u] * old_deg[v])
        new_w = 1.0 / torch.sqrt(new_deg[u] * new_deg[v])
        old_l1[u] += abs(old_w - new_w)
        old_l1[v] += abs(old_w - new_w)
    for i in range(keep):
        old_l1[i] += abs(1.0 / old_deg[i] - 1.0 / new_deg[i])
    for u, v in current_pairs.T.tolist():
        if u < keep <= v:
            cross_mass[u] += 1.0 / torch.sqrt(new_deg[u] * new_deg[v])
        elif v < keep <= u:
            cross_mass[v] += 1.0 / torch.sqrt(new_deg[u] * new_deg[v])
    load = torch.bincount(
        current_pairs[0][(current_pairs[0] < keep) ^ (current_pairs[1] < keep)],
        minlength=keep).float()
    # The expression above counts the old endpoint only for canonical pairs.
    if current_pairs.numel():
        cross = current_pairs[:, (current_pairs[0] < keep) ^ (current_pairs[1] < keep)]
        endpoint = torch.where(cross[0] < keep, cross[0], cross[1])
        load = torch.bincount(endpoint, minlength=keep).float()
    return {"old_block_row_l1": float(old_l1.mean()),
            "full_old_row_l1": float((old_l1 + cross_mass).mean()),
            "cross_norm_mass": float(cross_mass.mean()),
            "max_relative_load": float((load / old_deg.float()).max())}


def topology_components(old, block, keep, order, stage, device):
    c0 = U.SCA.append_memory(old, block, keep, device=device)
    degree = B.append(old, block, T_WEIGHTS, "degree_balanced")
    c0_old, c0_internal, c0_cross = components(c0["edge_index"], keep, keep + len(block["x"]))
    d_old, d_internal, d_cross = components(degree["edge_index"], keep, keep + len(block["x"]))
    assert torch.equal(c0_old, d_old)
    rng = np.random.default_rng(METHOD_SEED + 1000 * int(order) + stage)
    permutation = torch.from_numpy(rng.permutation(len(block["x"]))).long()
    random_internal = apply_new_permutation(d_internal, keep, len(block["x"]), permutation)
    shuffled_all = apply_new_permutation(torch.cat((d_internal, d_cross), dim=1), keep,
                                         len(block["x"]), permutation)
    random_all_internal = shuffled_all[:, :d_internal.shape[1]]
    random_all_cross = shuffled_all[:, d_internal.shape[1]:]
    candidates = {
        "c0": build_combined(old, block, keep, c0["edge_index"], c0["edge_index"]),
        "cross_only": build_combined(old, block, keep,
                                      torch.cat((c0_old, c0_internal), dim=1),
                                      torch.cat((d_old, d_cross), dim=1)),
        "internal_only": build_combined(old, block, keep,
                                         torch.cat((d_old, d_internal), dim=1),
                                         torch.cat((c0_old, c0_cross), dim=1)),
        "t_degree": build_combined(old, block, keep,
                                    torch.cat((d_old, d_internal), dim=1),
                                    torch.cat((d_old, d_cross), dim=1)),
        "random_ring": build_combined(old, block, keep,
                                       torch.cat((d_old, random_internal), dim=1),
                                       torch.cat((d_old, d_cross), dim=1)),
        "degree_shuffle": build_combined(old, block, keep,
                                          torch.cat((d_old, random_all_internal), dim=1),
                                          torch.cat((d_old, random_all_cross), dim=1)),
    }
    reference_pairs = degree_stats(candidates["c0"])["pairs"]
    for name, memory in candidates.items():
        assert degree_stats(memory)["pairs"] == reference_pairs, (name, degree_stats(memory))
        assert len(memory["x"]) == keep + len(block["x"])
    return candidates


def frozen(old, current):
    if old is None:
        return
    keep = len(old["x"])
    for key in FIELDS:
        assert torch.equal(old[key], current[key][:keep]), key
    assert old["texts"] == current["texts"][:keep]
    prefix = current["edge_index"][:, (current["edge_index"] < keep).all(0)]
    assert torch.equal(old["edge_index"], prefix), "old-old edges changed"


def source_hashes():
    root = Path(__file__).resolve().parents[2]
    return {str(p.relative_to(root)).replace("\\", "/"): sha(p)
            for p in sorted((root / "code").rglob("*.py"))}


def fixed_eval(path, value):
    if path.exists():
        old = torch.load(path, map_location="cpu", weights_only=False)
        assert old.keys() == value.keys()
        for key in value:
            assert (torch.equal(old[key], value[key]) if isinstance(value[key], torch.Tensor)
                    else old[key] == value[key]), key
    else:
        save_pt(path, value)
    return sha(path)


def fit_record(stem, context, train):
    result = stem.with_suffix(".json")
    if result.exists():
        record = json.loads(result.read_text())
        assert all(record[key] == value for key, value in context.items())
        return record
    checkpoint, metrics, cost = train()
    save_pt(stem.with_suffix(".pt"), dict(checkpoint, context=context))
    record = dict(context, metrics=metrics, cost=cost,
                  checkpoint_sha256=sha(stem.with_suffix(".pt")))
    save_json(result, record)
    return record


def construct(D, root, device):
    path = root / "construction.json"
    if path.exists():
        return json.loads(path.read_text())
    previous = {arm: None for arm in ARMS}
    records = []
    source_only = {key: D[key] for key in ("x", "text_emb", "texts", "y")}
    for stage, classes in enumerate(D["stage_classes"]):
        ids = torch.where(D["pool_mask"] & torch.isin(D["y"], torch.tensor(classes)))[0]
        cumulative = sum(len(torch.where(D["pool_mask"] & torch.isin(
            D["y"], torch.tensor(c)))[0]) for c in D["stage_classes"][:stage + 1])
        budget = U.protocol_v3.growing_budget(cumulative, rate=.05)
        keep = 0 if stage == 0 else len(previous["c0"]["x"])
        reset(METHOD_SEED)
        block = U.build_stage_block(source_only, ids, budget - keep, D["encoder"], stage,
                                    "joint_g5_c320", "concat", device,
                                    "dispersion", classes)
        plain = {key: block[key] for key in FIELDS + ("texts",)}
        stage_record = {"stage": stage, "classes": classes, "source_ids": ids.tolist(),
                        "source_count": len(ids), "budget": budget, "keep": keep,
                        "arms": {}}
        candidates = topology_components(previous["c0"], plain, keep,
                                         int(root.name.split("_")[-1]), stage, device)
        # Build each arm from its own previous memory so later stages preserve the
        # exact topology history of that arm.
        for arm in ARMS:
            old = previous[arm]
            if arm == "c0" and stage == 0:
                memory = candidates[arm]
            else:
                memory = topology_components(old, plain, keep,
                                             int(root.name.split("_")[-1]), stage, device)[arm]
            frozen(old, memory)
            assert len(memory["x"]) == budget
            path_arm = root / "memories" / f"{arm}_s{stage}.pt"
            save_pt(path_arm, memory)
            previous[arm] = memory
            stage_record["arms"][arm] = {
                "path": str(path_arm.relative_to(root)), "sha256": sha(path_arm),
                "topology": {**degree_stats(memory), **propagation_drift(old, memory, keep)}}
        records.append(stage_record)
        print(json.dumps({"event": "constructed", "order": root.name,
                          "stage": stage, "rows": budget}), flush=True)
    save_json(path, records)
    return records


def classification(D, root, construction, device):
    records = []
    folder = root / "nc"
    folder.mkdir(exist_ok=True)
    report = torch.where(D["report_dev_mask"])[0]
    evaluation = {"ids": report, "y": D["y"][report], "prompts": D["prompt_emb"],
                  "stage_classes": D["stage_classes"]}
    evaluation_hash = fixed_eval(folder / "evaluation.pt", evaluation)
    for stage in range(5):
        for arm in ARMS:
            memory = torch.load(root / "memories" / f"{arm}_s{stage}.pt",
                                map_location="cpu", weights_only=False)
            for seed in SEEDS:
                stem = folder / f"{arm}_s{stage}_seed{seed}"
                context = {"task": "nc", "dataset": D["dataset"],
                           "order": int(root.name.split("_")[-1]), "stage": stage,
                           "arm": arm, "seed": seed,
                           "memory_sha256": sha(root / "memories" / f"{arm}_s{stage}.pt"),
                           "evaluation_sha256": evaluation_hash}
                def train():
                    reset(seed)
                    L.sync(device)
                    torch.cuda.reset_peak_memory_stats(device)
                    start = time.perf_counter()
                    model = U.G.train_student(U.G.memory_to_trainable(memory), seed,
                                              device, 2048, batch=16)
                    L.sync(device)
                    cost = {"train_seconds": time.perf_counter() - start,
                            "steps": 2048, "batch": 16,
                            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device)}
                    model.eval()
                    with torch.no_grad():
                        logits = model.logits(D["x_device"], D["real_adj"],
                                              D["prompt_emb"].to(device)).cpu()[report]
                    metrics = M.score_logits(logits, evaluation["y"].numpy(),
                                             D["stage_classes"], stage)
                    return {"state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                            "logits": logits}, metrics, cost
                records.append(fit_record(stem, context, train))
    return records


def cosine_scores(features, edge):
    return torch.cat([(features[part[0]] * features[part[1]]).sum(1)
                      for part in edge.split(8192, dim=1)])


def link_prediction(D, root, construction, device):
    records = []
    arrival = torch.full((len(D["x"]),), -1, dtype=torch.long)
    for stage, classes in enumerate(D["stage_classes"]):
        arrival[D["pool_mask"] & torch.isin(D["y"], torch.tensor(classes))] = stage
    for split_seed in SPLITS:
        folder = root / f"lp_split_{split_seed}"
        folder.mkdir(exist_ok=True)
        split = B.split_edges(D["edge_index"], arrival, split_seed)
        save_pt(folder / "edge_split.pt", split)
        for stage in range(5):
            ids = torch.where((arrival >= 0) & (arrival <= stage))[0]
            local = {k: L.localize(split[k], ids, len(arrival))
                     for k in ("train_pos", "train_neg", "test_pos", "test_neg")}
            real_x = L.features(D["x"][ids], D["text_emb"][ids])
            edge = L.bidirectional(local["train_pos"])
            evaluation = {"ids": ids, "arrival": arrival[ids], "x": real_x,
                          "edge_index": edge, "test_pos": local["test_pos"],
                          "test_neg": local["test_neg"]}
            evaluation_hash = fixed_eval(folder / f"evaluation_s{stage}.pt", evaluation)
            controls = {}
            for name, features in (("x", F.normalize(D["x"][ids], dim=1)),
                                   ("text", F.normalize(D["text_emb"][ids], dim=1)),
                                   ("xt", real_x)):
                controls[name] = L.metrics(cosine_scores(features, local["test_pos"]),
                                           cosine_scores(features, local["test_neg"]),
                                           local["test_pos"], local["test_neg"],
                                           arrival[ids], stage)
            save_json(folder / f"cosine_s{stage}.json", controls)
            for arm in ARMS:
                memory = torch.load(root / "memories" / f"{arm}_s{stage}.pt",
                                    map_location="cpu", weights_only=False)
                train_x = L.features(memory["x"], memory["text_emb"])
                train_pos = L.pairs(memory["edge_index"], len(train_x))
                train_neg = B.training_negatives(train_pos, len(train_x), split_seed + stage)
                for seed in SEEDS:
                    stem = folder / f"{arm}_s{stage}_seed{seed}"
                    context = {"task": "lp", "dataset": D["dataset"],
                               "order": int(root.name.split("_")[-1]), "split": split_seed,
                               "stage": stage, "arm": arm, "seed": seed,
                               "memory_sha256": sha(root / "memories" / f"{arm}_s{stage}.pt"),
                               "evaluation_sha256": evaluation_hash}
                    def train():
                        L.sync(device)
                        torch.cuda.reset_peak_memory_stats(device)
                        model, ps, ns, cost = L.fit(train_x, train_pos, train_neg,
                                                    real_x, edge, local["test_pos"],
                                                    local["test_neg"], seed, 256, device)
                        L.sync(device)
                        cost["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
                        metrics = L.metrics(ps, ns, local["test_pos"], local["test_neg"],
                                            arrival[ids], stage)
                        return {"state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                                "pos_scores": ps, "neg_scores": ns}, metrics, cost
                    records.append(fit_record(stem, context, train))
    return records


def main(args):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    p5 = json.loads((Path(args.c0_root) / "results.json").read_text())
    groups = {str(seed): fresh_groups(p5["stage_classes"], seed) for seed in ORDERS}
    manifest = {"dataset": args.dataset, "orders": groups, "seeds": list(SEEDS),
                "splits": list(SPLITS), "arms": list(ARMS), "rate": .05,
                "nc_steps": 2048, "nc_batch": 16, "lp_steps": 256,
                "method_seed": METHOD_SEED, "protocol": PROTOCOL,
                "data": args.data, "text": args.text,
                "data_sha256": sha(args.data), "text_sha256": sha(args.text),
                "original_class_groups": p5["stage_classes"],
                "sources": source_hashes(), "torch": torch.__version__}
    path = out / "manifest.json"
    if path.exists():
        assert json.loads(path.read_text()) == manifest, "frozen protocol changed"
    else:
        save_json(path, manifest)
    D = U.load_unified(args.dataset, args.data, args.minilm, args.device,
                       n_stages=5, split_mode="base_incremental")
    D["text_emb"] = torch.load(args.text, map_location="cpu", weights_only=False).float()
    D["prompt_emb"] = D["encoder"].encode(D["prompts"])
    D["x_device"] = D["x"].to(args.device)
    D["real_adj"] = U.G.norm_adj(D["edge_index"], len(D["x"]), args.device)
    all_records = []
    for order in ORDERS:
        root = out / f"order_{order}"
        root.mkdir(exist_ok=True)
        D["stage_classes"] = groups[str(order)]
        U.G.STAGE_CLASSES = D["stage_classes"]
        construction = construct(D, root, args.device)
        if args.construct_only:
            continue
        all_records.extend(classification(D, root, construction, args.device))
        all_records.extend(link_prediction(D, root, construction, args.device))
    save_json(out / "results.json", all_records)
    save_json(out / "complete.json", {"nc_fits": 360, "lp_fits": 720,
                                      "total_fits": len(all_records),
                                      "expected": 1080})
    print(json.dumps({"event": "DONE", "records": len(all_records)}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=("arxiv", "bookchild"), required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--text", required=True)
    p.add_argument("--minilm", required=True)
    p.add_argument("--c0-root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--construct-only", action="store_true")
    main(p.parse_args())
