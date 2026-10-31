"""Prospective large-graph confirmation using shared memories across tasks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

import run_unified_v3_continual as U
import run_link_prediction_transfer as L
import large_lp_support as B
import small_comparison_metrics as M
from run_large_lp_validation import sha, save, topology

ARMS = ("c0", "t_degree", "xt_degree")
WEIGHTS = {"t_degree": (0., 1., 0.), "xt_degree": (4 / 7, 3 / 7, 0.)}
FIELDS = ("x", "text_emb", "mass", "depth", "stage")
ORDERS = (49011, 49012)
SPLITS = (49101, 49102)
SEEDS = tuple(range(210, 216))
PROTOCOL = "DEGREE_CROSS_TASK_CONFIRMATION_PROTOCOL_ZH_2026-09-20.md"


def fresh_groups(groups, seed):
    flat = sorted(c for group in groups for c in group)
    if len(set(flat)) != len(flat) or any(not group for group in groups):
        raise ValueError("invalid class partition")
    shuffled = np.random.default_rng(seed).permutation(flat).tolist()
    edges = np.cumsum([0] + [len(group) for group in groups])
    return [sorted(shuffled[edges[i]:edges[i + 1]]) for i in range(len(groups))]


def frozen(old, current):
    if old is None:
        return
    keep = len(old["x"])
    for key in FIELDS:
        assert torch.equal(old[key], current[key][:keep]), key
    assert old["texts"] == current["texts"][:keep]
    edge = current["edge_index"]
    prefix = edge[:, (edge < keep).all(0)]
    assert torch.equal(prefix, old["edge_index"]), "old-old edges changed"


def save_pt(path, value):
    path = Path(path)
    tmp = path.with_suffix(".pt.tmp")
    torch.save(value, tmp)
    tmp.replace(path)


def fixed_evaluation(path, value):
    if path.exists():
        old = torch.load(path, map_location="cpu", weights_only=False)
        assert old.keys() == value.keys()
        for key in value:
            assert (torch.equal(old[key], value[key]) if isinstance(value[key], torch.Tensor)
                    else old[key] == value[key]), key
    else:
        save_pt(path, value)
    return sha(path)


def source_hashes():
    root = Path(__file__).resolve().parents[2]
    return {str(p.relative_to(root)).replace("\\", "/"): sha(p)
            for p in sorted((root / "code").rglob("*.py"))}


def reset(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def construct(D, root, device):
    complete = root / "construction.json"
    if complete.exists():
        records = json.loads(complete.read_text())
        for record in records:
            for arm in ARMS:
                assert sha(root / record["arms"][arm]["path"]) == record["arms"][arm]["sha256"]
        return records
    previous = dict.fromkeys(ARMS)
    records, cumulative = [], 0
    folder = root / "memories"
    folder.mkdir(exist_ok=True)
    # The constructor cannot inspect any real edge, inference graph or dev label.
    source_only = {key: D[key] for key in ("x", "text_emb", "texts", "y")}
    for stage, classes in enumerate(D["stage_classes"]):
        ids = torch.where(D["pool_mask"] & torch.isin(D["y"], torch.tensor(classes)))[0]
        cumulative += len(ids)
        budget = U.protocol_v3.growing_budget(cumulative, rate=.05)
        keep = 0 if stage == 0 else len(previous["c0"]["x"])
        reset(U.protocol_v3.METHOD_SEED)
        L.sync(device)
        start = time.perf_counter()
        block = U.build_stage_block(source_only, ids, budget - keep, D["encoder"],
                                    stage, "joint_g5_c320", "concat", device,
                                    "dispersion", classes)
        L.sync(device)
        shared_seconds = time.perf_counter() - start
        record = {"stage": stage, "classes": classes, "source_ids": ids.tolist(),
                  "source_count": len(ids), "cumulative_source_count": cumulative,
                  "budget": budget, "keep": keep, "allocation": block["allocation"],
                  "shared_block_seconds": shared_seconds, "arms": {}}
        plain = {key: block[key] for key in FIELDS + ("texts",)}
        for arm in ARMS:
            L.sync(device)
            start = time.perf_counter()
            if arm == "c0":
                memory = U.SCA.append_memory(previous[arm], plain, keep, device=device)
            else:
                memory = B.append(previous[arm], plain, WEIGHTS[arm], "degree_balanced")
            L.sync(device)
            seconds = time.perf_counter() - start
            frozen(previous[arm], memory)
            assert len(memory["x"]) == budget
            if arm != "c0":
                for key in FIELDS:
                    assert torch.equal(memory[key], previous["c0"][key]), key
                assert memory["texts"] == previous["c0"]["texts"]
                assert L.pairs(memory["edge_index"], budget).shape == L.pairs(previous["c0"]["edge_index"], budget).shape
            path = folder / f"{arm}_s{stage}.pt"
            save_pt(path, memory)
            previous[arm] = memory
            record["arms"][arm] = {"path": str(path.relative_to(root)), "sha256": sha(path),
                                    "topology_seconds": seconds, "bytes": path.stat().st_size,
                                    "topology": topology(memory)}
        records.append(record)
        print(json.dumps({"event": "constructed", "order": root.name, "stage": stage,
                          "sources": len(ids), "rows": budget}), flush=True)
    save(complete, records)
    return records


def fit_record(stem, context, train):
    result = stem.with_suffix(".json")
    if result.exists():
        record = json.loads(result.read_text())
        assert all(record[key] == value for key, value in context.items())
        assert sha(stem.with_suffix(".pt")) == record["checkpoint_sha256"]
    else:
        checkpoint, metrics, cost = train()
        save_pt(stem.with_suffix(".pt"), dict(checkpoint, context=context))
        record = dict(context, metrics=metrics, cost=cost,
                      checkpoint_sha256=sha(stem.with_suffix(".pt")))
        save(result, record)
    print(json.dumps({"task": context["task"], "order": context["order"],
                      "stage": context["stage"], "arm": context["arm"],
                      "seed": context["seed"], "split": context.get("split")}), flush=True)
    return record


def classification(D, root, construction, device):
    records = []
    folder = root / "nc"
    folder.mkdir(exist_ok=True)
    report = torch.where(D["report_dev_mask"])[0]
    evaluation = {"ids": report, "y": D["y"][report], "prompts": D["prompt_emb"],
                  "stage_classes": D["stage_classes"]}
    evaluation_hash = fixed_evaluation(folder / "evaluation.pt", evaluation)
    for stage in range(5):
        for arm in ARMS:
            memory_path = root / construction[stage]["arms"][arm]["path"]
            memory = torch.load(memory_path, map_location="cpu", weights_only=False)
            for seed in SEEDS:
                context = {"task": "nc", "dataset": D["dataset"], "order": int(root.name[6:]),
                           "stage": stage, "arm": arm, "seed": seed,
                           "memory_sha256": sha(memory_path), "evaluation_sha256": evaluation_hash}

                def train():
                    reset(seed)
                    L.sync(device)
                    torch.cuda.reset_peak_memory_stats(device)
                    start = time.perf_counter()
                    model = U.G.train_student(U.G.memory_to_trainable(memory), seed,
                                              device, 2048, batch=16)
                    L.sync(device)
                    train_seconds = time.perf_counter() - start
                    model.eval()
                    with torch.no_grad():
                        logits = model.logits(D["x_device"], D["real_adj"],
                                              D["prompt_emb"].to(device)).cpu()[report]
                    metrics = M.score_logits(logits, evaluation["y"].numpy(),
                                             D["stage_classes"], stage)
                    cost = {"train_seconds": train_seconds, "steps": 2048, "batch": 16,
                            "train_rows": len(memory["x"]),
                            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device)}
                    checkpoint = {"state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                                  "logits": logits}
                    return checkpoint, metrics, cost

                records.append(fit_record(folder / f"{arm}_s{stage}_seed{seed}", context, train))
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
            evaluation_hash = fixed_evaluation(folder / f"evaluation_s{stage}.pt", evaluation)
            controls = {}
            for name, features in (("x", F.normalize(D["x"][ids], dim=1)),
                                   ("text", F.normalize(D["text_emb"][ids], dim=1)),
                                   ("xt", real_x)):
                controls[name] = L.metrics(cosine_scores(features, local["test_pos"]),
                                          cosine_scores(features, local["test_neg"]),
                                          local["test_pos"], local["test_neg"], arrival[ids], stage)
            save(folder / f"cosine_s{stage}.json", controls)
            for arm in ARMS:
                memory_path = root / construction[stage]["arms"][arm]["path"]
                memory = torch.load(memory_path, map_location="cpu", weights_only=False)
                train_x = L.features(memory["x"], memory["text_emb"])
                train_pos = L.pairs(memory["edge_index"], len(train_x))
                train_neg = B.training_negatives(train_pos, len(train_x), split_seed + stage)
                for seed in SEEDS:
                    context = {"task": "lp", "dataset": D["dataset"], "order": int(root.name[6:]),
                               "split": split_seed, "stage": stage, "arm": arm, "seed": seed,
                               "memory_sha256": sha(memory_path), "evaluation_sha256": evaluation_hash}

                    def train():
                        torch.cuda.reset_peak_memory_stats(device)
                        model, ps, ns, cost = L.fit(train_x, train_pos, train_neg, real_x,
                                                    edge, local["test_pos"], local["test_neg"],
                                                    seed, 256, device)
                        cost["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
                        metrics = L.metrics(ps, ns, local["test_pos"], local["test_neg"], arrival[ids], stage)
                        checkpoint = {"state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                                      "pos_scores": ps, "neg_scores": ns}
                        return checkpoint, metrics, cost

                    records.append(fit_record(folder / f"{arm}_s{stage}_seed{seed}", context, train))
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
    assert all(g != p5["stage_classes"] for g in groups.values())
    manifest = {"dataset": args.dataset, "orders": groups, "seeds": list(SEEDS),
                "splits": list(SPLITS), "arms": list(ARMS), "rate": .05,
                "nc_steps": 2048, "nc_batch": 16, "lp_steps": 256,
                "data": args.data, "text": args.text, "minilm": args.minilm,
                "data_sha256": sha(args.data), "text_sha256": sha(args.text),
                "original_class_groups": p5["stage_classes"],
                "sources": source_hashes(), "torch": torch.__version__,
                "protocol_sha256": sha(Path(__file__).resolve().parents[2] / "docs/current" / PROTOCOL)}
    if (out / "manifest.json").exists():
        assert json.loads((out / "manifest.json").read_text()) == manifest, "frozen protocol changed"
    save(out / "manifest.json", manifest)
    started = time.perf_counter()
    D = U.load_unified(args.dataset, args.data, args.minilm, args.device,
                       n_stages=5, split_mode="base_incremental")
    D["text_emb"] = torch.load(args.text, map_location="cpu", weights_only=False).float()
    assert len(D["text_emb"]) == len(D["x"])
    assert not bool(D["text_emb"][~D["pool_mask"]].any())
    D["prompt_emb"] = D["encoder"].encode(D["prompts"])
    D["x_device"] = D["x"].to(args.device)
    D["real_adj"] = U.G.norm_adj(D["edge_index"], len(D["x"]), args.device)
    if not (out / "setup.json").exists():
        save(out / "setup.json", {"load_and_prompt_seconds": time.perf_counter() - started,
                                   "pool_text_encoding": "reused cached encoding; cost not measured here",
                                   "device": args.device})
    records = []
    for order_seed in ORDERS:
        root = out / f"order_{order_seed}"
        root.mkdir(exist_ok=True)
        D["stage_classes"] = groups[str(order_seed)]
        U.G.STAGE_CLASSES = D["stage_classes"]
        construction = construct(D, root, args.device)
        records.extend(classification(D, root, construction, args.device))
        records.extend(link_prediction(D, root, construction, args.device))
    save(out / "results.json", records)
    assert len(records) == 540
    save(out / "complete.json", {"nc_fits": 180, "lp_fits": 360, "total_fits": len(records)})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=("arxiv", "bookchild"), required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--text", required=True)
    p.add_argument("--c0-root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--minilm", default="/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip")
    main(p.parse_args())
