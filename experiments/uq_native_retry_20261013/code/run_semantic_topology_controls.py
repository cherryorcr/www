"""Matched semantic-neighbor controls for the frozen CTAG topology study.

This runner reuses audited C0 content/evaluation tensors and trains only the
four new topology arms. It never re-runs content construction and does not
modify the frozen six-arm experiment.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import large_lp_support as B
import run_link_prediction_transfer as L
import run_topology_causal_ablation as R
import run_unified_v3_continual as U
import small_comparison_metrics as M
from semantic_topology_controls import SELECTORS


ARMS = ("knn_nearest", "knn_capacity", "cycles_nearest", "cycles_capacity")
ORDERS = (49011, 49012)
SPLITS = (49101, 49102)
SEEDS = tuple(range(210, 216))
FIELDS = R.FIELDS
PROTOCOL = "SEMANTIC_TOPOLOGY_CONTROLS_PROTOCOL_ZH_2026-09-21.md"


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


def as_pairs(value):
    return torch.as_tensor(value, dtype=torch.long).contiguous()


def semantic_internal(block, operator):
    text = F.normalize(block["text_emb"].float(), dim=1).numpy()
    pairs, info = SELECTORS[operator](text @ text.T)
    return as_pairs(pairs), info


def combine(old, block, keep, internal_local, cross_global):
    old_edge = old["edge_index"] if old is not None else torch.empty((2, 0), dtype=torch.long)
    global_internal = internal_local + keep
    out = {key: (torch.cat((old[key], block[key])) if old is not None else block[key].clone())
           for key in FIELDS}
    out["texts"] = old["texts"] + block["texts"] if old is not None else list(block["texts"])
    out["edge_index"] = torch.cat((old_edge, R.undirected(global_internal),
                                    R.undirected(cross_global)), dim=1)
    return out


def topology_stats(memory):
    return R.degree_stats(memory)


def source_hashes(root):
    names = [
        "code/v2_llm_fusion/run_semantic_topology_controls.py",
        "code/v2_llm_fusion/semantic_topology_controls.py",
        "docs/current/SEMANTIC_TOPOLOGY_CONTROLS_PROTOCOL_ZH_2026-09-21.md",
    ]
    return {name: R.sha(root / name) for name in names}


def fit_record(stem, context, train):
    result = stem.with_suffix(".json")
    if result.exists():
        record = json.loads(result.read_text(encoding="utf-8"))
        assert all(record[key] == value for key, value in context.items())
        return record
    checkpoint, metrics, cost = train()
    save_pt(stem.with_suffix(".pt"), dict(checkpoint, context=context))
    record = dict(context, metrics=metrics, cost=cost,
                  checkpoint_sha256=R.sha(stem.with_suffix(".pt")))
    save_json(result, record)
    return record


def construction(baseline_root, output_root, order, groups, device):
    path = output_root / "construction.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    previous = {arm: None for arm in ARMS}
    records = []
    for stage, classes in enumerate(groups):
        base_meta = json.loads((baseline_root / f"order_{order}/construction.json").read_text()) [stage]
        base_path = baseline_root / f"order_{order}" / base_meta["arms"]["c0"]["path"]
        base = torch.load(base_path, map_location="cpu", weights_only=False)
        keep = 0 if stage == 0 else len(previous[ARMS[0]]["x"])
        budget = len(base["x"])
        assert base_meta["keep"] == keep and base_meta["budget"] == budget
        block = {key: base[key][keep:] for key in FIELDS}
        block["texts"] = base["texts"][keep:]

        # Cross endpoints are generated from the audited baseline content.
        # The selected old edge prefix is discarded; only the new cross pairs
        # are reused, so each new arm keeps its own historical topology.
        old_base_path = None if stage == 0 else baseline_root / f"order_{order}" / "memories" / f"c0_s{stage - 1}.pt"
        old_base = None if old_base_path is None else torch.load(
            old_base_path, map_location="cpu", weights_only=False)
        baseline_candidates = R.topology_components(old_base, block, keep, order, stage, device)
        _, _, nearest_cross = R.components(baseline_candidates["c0"]["edge_index"], keep, budget)
        _, _, capacity_cross = R.components(baseline_candidates["t_degree"]["edge_index"], keep, budget)
        if stage == 0:
            assert nearest_cross.numel() == 0 and capacity_cross.numel() == 0

        stage_record = {"stage": stage, "classes": classes, "budget": budget, "keep": keep,
                        "source_memory": str(base_path), "arms": {}}
        for arm in ARMS:
            operator = "knn" if arm.startswith("knn") else "cycles"
            cross = capacity_cross if arm.endswith("capacity") else nearest_cross
            internal, operator_info = semantic_internal(block, operator)
            memory = combine(previous[arm], block, keep, internal, cross)
            R.frozen(previous[arm], memory)
            assert len(memory["x"]) == budget
            assert R.degree_stats(memory)["pairs"] == R.degree_stats(base)["pairs"]
            path_arm = output_root / "memories" / f"{arm}_s{stage}.pt"
            save_pt(path_arm, memory)
            previous[arm] = memory
            stage_record["arms"][arm] = {
                "path": str(path_arm.relative_to(output_root)),
                "sha256": R.sha(path_arm),
                "operator": operator,
                "operator_info": operator_info,
                "cross_policy": "capacity" if arm.endswith("capacity") else "nearest",
                "topology": topology_stats(memory),
            }
        records.append(stage_record)
        print(json.dumps({"event": "constructed", "dataset_order": order,
                          "stage": stage, "rows": budget}), flush=True)
    save_json(path, records)
    return records


def classification(D, root, construction_records, device):
    records = []
    folder = root / "nc"
    folder.mkdir(exist_ok=True)
    report = torch.where(D["report_dev_mask"])[0]
    evaluation = {"ids": report, "y": D["y"][report], "prompts": D["prompt_emb"],
                  "stage_classes": D["stage_classes"]}
    evaluation_hash = R.fixed_eval(folder / "evaluation.pt", evaluation)
    order = int(root.name.split("_")[-1])
    for stage in range(len(D["stage_classes"])):
        for arm in ARMS:
            memory = torch.load(root / "memories" / f"{arm}_s{stage}.pt",
                                map_location="cpu", weights_only=False)
            for seed in SEEDS:
                stem = folder / f"{arm}_s{stage}_seed{seed}"
                context = {"task": "nc", "dataset": D["dataset"], "order": order,
                           "stage": stage, "arm": arm, "seed": seed,
                           "memory_sha256": R.sha(root / "memories" / f"{arm}_s{stage}.pt"),
                           "evaluation_sha256": evaluation_hash}
                def train():
                    R.reset(seed)
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


def link_prediction(D, root, construction_records, baseline_root, device):
    records = []
    order = int(root.name.split("_")[-1])
    arrival = torch.full((len(D["x"]),), -1, dtype=torch.long)
    for stage, classes in enumerate(D["stage_classes"]):
        arrival[D["pool_mask"] & torch.isin(D["y"], torch.tensor(classes))] = stage
    for split_seed in SPLITS:
        folder = root / f"lp_split_{split_seed}"
        folder.mkdir(exist_ok=True)
        baseline_folder = baseline_root / f"order_{order}/lp_split_{split_seed}"
        split = torch.load(baseline_folder / "edge_split.pt", map_location="cpu", weights_only=False)
        R.save_pt(folder / "edge_split.pt", split)
        for stage in range(len(D["stage_classes"])):
            ids = torch.where((arrival >= 0) & (arrival <= stage))[0]
            local = {k: L.localize(split[k], ids, len(arrival))
                     for k in ("train_pos", "train_neg", "test_pos", "test_neg")}
            real_x = L.features(D["x"][ids], D["text_emb"][ids])
            edge = L.bidirectional(local["train_pos"])
            baseline_evaluation = torch.load(
                baseline_folder / f"evaluation_s{stage}.pt",
                map_location="cpu", weights_only=False)
            for key, value in {"ids": ids, "arrival": arrival[ids], "x": real_x,
                               "edge_index": edge, "test_pos": local["test_pos"],
                               "test_neg": local["test_neg"]}.items():
                assert (torch.equal(value, baseline_evaluation[key])
                        if isinstance(value, torch.Tensor) else value == baseline_evaluation[key]), key
            evaluation = baseline_evaluation
            evaluation_hash = R.fixed_eval(folder / f"evaluation_s{stage}.pt", evaluation)
            controls = {}
            for name, features in (("x", F.normalize(D["x"][ids], dim=1)),
                                   ("text", F.normalize(D["text_emb"][ids], dim=1)),
                                   ("xt", real_x)):
                controls[name] = L.metrics(cosine_scores(features, local["test_pos"]),
                                           cosine_scores(features, local["test_neg"]),
                                           local["test_pos"], local["test_neg"],
                                           arrival[ids], stage)
            baseline_cosine = json.loads((baseline_folder / f"cosine_s{stage}.json").read_text())
            assert controls == baseline_cosine
            R.save_json(folder / f"cosine_s{stage}.json", baseline_cosine)
            for arm in ARMS:
                memory = torch.load(root / "memories" / f"{arm}_s{stage}.pt",
                                    map_location="cpu", weights_only=False)
                train_x = L.features(memory["x"], memory["text_emb"])
                train_pos = L.pairs(memory["edge_index"], len(train_x))
                train_neg = B.training_negatives(train_pos, len(train_x), split_seed + stage)
                for seed in SEEDS:
                    stem = folder / f"{arm}_s{stage}_seed{seed}"
                    context = {"task": "lp", "dataset": D["dataset"], "order": order,
                               "split": split_seed, "stage": stage, "arm": arm, "seed": seed,
                               "memory_sha256": R.sha(root / "memories" / f"{arm}_s{stage}.pt"),
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


def manifest_for(args, root, baseline_manifest, baseline_root):
    return {
        "dataset": args.dataset, "arms": list(ARMS), "orders": baseline_manifest["orders"],
        "seeds": list(SEEDS), "splits": list(SPLITS), "rate": baseline_manifest["rate"],
        "nc_steps": 2048, "nc_batch": 16, "lp_steps": 256,
        "protocol": PROTOCOL, "baseline_root": str(baseline_root),
        "baseline_manifest_sha256": R.sha(baseline_root / "manifest.json"),
        "data": baseline_manifest["data"], "text": baseline_manifest["text"],
        "data_sha256": baseline_manifest["data_sha256"], "text_sha256": baseline_manifest["text_sha256"],
        "minilm": args.minilm, "sources": source_hashes(root),
    }


def main(args):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    output = Path(args.output)
    baseline_root = Path(args.baseline_root)
    repo = Path(__file__).resolve().parents[2]
    baseline_manifest = json.loads((baseline_root / "manifest.json").read_text())
    manifest = manifest_for(args, repo, baseline_manifest, baseline_root)
    manifest_path = output / "manifest.json"
    output.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text()) == manifest, "frozen semantic protocol changed"
    else:
        save_json(manifest_path, manifest)
    D = U.load_unified(args.dataset, args.data, args.minilm, args.device,
                       n_stages=len(next(iter(manifest["orders"].values()))),
                       split_mode="base_incremental")
    D["text_emb"] = torch.load(args.text, map_location="cpu", weights_only=False).float()
    D["prompt_emb"] = D["encoder"].encode(D["prompts"])
    D["x_device"] = D["x"].to(args.device)
    D["real_adj"] = U.G.norm_adj(D["edge_index"], len(D["x"]), args.device)
    all_records = []
    for order in ORDERS:
        root = output / f"order_{order}"
        root.mkdir(exist_ok=True)
        groups = manifest["orders"][str(order)]
        D["stage_classes"] = groups
        U.G.STAGE_CLASSES = groups
        construction_records = construction(baseline_root, root, order, groups, args.device)
        if args.construct_only:
            continue
        all_records.extend(classification(D, root, construction_records, args.device))
        all_records.extend(link_prediction(D, root, construction_records, baseline_root, args.device))
    save_json(output / "results.json", all_records)
    save_json(output / "complete.json", {"nc_fits": 480, "lp_fits": 960,
                                         "total_fits": len(all_records), "expected": 1440})
    print(json.dumps({"event": "DONE", "records": len(all_records)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("arxiv", "bookchild"), required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--minilm", required=True)
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--construct-only", action="store_true")
    main(parser.parse_args())
