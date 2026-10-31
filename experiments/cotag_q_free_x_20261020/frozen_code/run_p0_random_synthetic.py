"""Phase-1 random-synthetic control under the frozen C0 protocol.

The control keeps the C0 5% growing budget, class allocation, append-only
memory management, topology degree, student, seeds, and report-dev metric.
Only the grouping within each incoming class changes: uniformly shuffled
balanced groups replace geometric clusters, then Mean-X and concat/re-encode
are applied exactly as for C0.
It is a control, not a proposed method.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import run_unified_v3_continual as U
P, G, SCA = U.protocol_v3, U.G, U.SCA


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def clone_memory(memory: dict) -> dict:
    return {key: value.clone() if isinstance(value, torch.Tensor)
            else list(value) if isinstance(value, list) else value
            for key, value in memory.items()}


def random_synthetic_block(data: dict, ids: torch.Tensor, rows: int,
                           stage: int, seed: int) -> tuple[dict, dict]:
    ids = torch.sort(ids.long().cpu()).values
    if not bool(data["pool_mask"][ids].all()):
        raise ValueError("random synthetic sources must be pool-only")
    classes = sorted(int(v) for v in data["stage_classes"][stage])
    allocation = U.allocate_rows_by_class(data, ids, classes, int(rows),
                                          "dispersion")
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    xs, texts, embeddings, masses = [], [], [], []
    per_class = {}
    for cls in classes:
        class_ids = torch.sort(ids[data["y"][ids] == int(cls)]).values
        count = min(int(allocation[int(cls)]), len(class_ids))
        if count <= 0:
            continue
        class_x = data["x"][class_ids].float()
        class_t = data["text_emb"][class_ids].float()
        joint = torch.cat((F.normalize(class_x, dim=1), class_t), dim=1)
        source_text = [data["texts"][int(i)] for i in class_ids]
        per_class[int(cls)] = count
        groups = torch.tensor_split(torch.randperm(len(class_ids), generator=gen), count)
        fused = []
        for members in groups:
            members = torch.sort(members).values
            xs.append(class_x[members].mean(0))
            text, _ = SCA.encode_cluster_text(joint, members, source_text,
                data["encoder"], "concat", P.FUSE_CHARS, P.FUSE_G,
                source_text_emb=class_t)
            fused.append(text)
            masses.append(float(len(members)))
        texts.extend(fused)
        embeddings.extend(data["encoder"].encode(fused))
    if not xs:
        raise ValueError("random synthetic block is empty")
    block = {
        "x": torch.stack(xs),
        "text_emb": torch.stack(embeddings),
        "texts": texts,
        "mass": torch.tensor(masses, dtype=torch.float32),
        "depth": torch.zeros(len(xs), dtype=torch.long),
        "stage": torch.full((len(xs),), int(stage), dtype=torch.long),
    }
    block["edge_index"] = SCA.text_knn_edges(block["text_emb"], P.DEGREE)
    return block, {"allocation": per_class, "seed": int(seed),
                   "rows": int(len(xs)), "operator": "random-class-partition-mean-concat-reencode"}


def load_data(dataset: str, minilm: str, device: str, cache: Path | None) -> dict:
    data_path = U.DATA_CFG[dataset]["path"]
    data = U.load_unified(dataset, data_path, minilm, device)
    if cache is not None and cache.exists():
        text_emb = torch.load(cache, map_location="cpu", weights_only=False)
    else:
        pool_ids = torch.where(data["pool_mask"])[0]
        pool_emb = data["encoder"].encode(
            [data["texts"][int(i)] for i in pool_ids])
        text_emb = torch.zeros((len(data["x"]), pool_emb.shape[1]),
                               dtype=pool_emb.dtype)
        text_emb[pool_ids] = pool_emb
    if text_emb.shape[0] != len(data["x"]):
        raise ValueError("text cache row count mismatch")
    if text_emb[~data["pool_mask"]].abs().max() != 0:
        raise ValueError("random control read excluded text embeddings")
    data["text_emb"] = text_emb
    data["prompt_emb"] = data["encoder"].encode(data["prompts"])
    data["real_adj"] = G.norm_adj(data["edge_index"], len(data["x"]), device)
    data["x_device"] = data["x"].float().to(device)
    return data


def fit_stage(data: dict, memory: dict, stage: int, seed: int,
              device: str) -> dict:
    steps = P.student_steps(len(memory["x"]))
    started = time.perf_counter()
    model = G.train_student(memory, int(seed), device, steps,
                            lr=P.STUDENT.lr, batch=P.STUDENT.batch)
    score = G.eval_stagewise(model, data["x_device"], data["real_adj"],
                             data["prompt_emb"], data["y"],
                             data["report_dev_mask"], stage)
    score.update({"seed": int(seed), "steps": int(steps),
                  "seconds": float(time.perf_counter() - started)})
    del model
    return score


def aggregate(rows: list[dict], stage: int) -> dict:
    keys = ("cumulative_macro_f1", "current_f1", "old_mean_f1")
    out = {"rows": len(rows), "stage": int(stage)}
    for key in keys:
        values = [float(r[key]) for r in rows if r.get(key) is not None]
        out[key + "_mean"] = float(np.mean(values)) if values else None
        out[key + "_std"] = float(np.std(values)) if values else None
    out["per_stage_mean"] = [
        float(np.mean([r["per_stage"][s]["macro_f1"] for r in rows]))
        for s in range(stage + 1)]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cora", "citeseer"], required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--minilm", default=(
        "/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip"))
    ap.add_argument("--text-cache", default=None)
    ap.add_argument("--student-seeds", nargs="+", type=int,
                    default=list(P.STUDENT_SEEDS))
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "results.json").exists():
        raise FileExistsError("existing random-synthetic run is immutable")
    cache = Path(args.text_cache) if args.text_cache else None
    data = load_data(args.dataset, args.minilm, args.device, cache)
    stage_ids = [torch.sort(torch.where(
        data["pool_mask"] & torch.isin(data["y"], torch.tensor(classes))
    )[0]).values for classes in data["stage_classes"]]
    budgets = [math.floor(0.05 * sum(len(v) for v in stage_ids[:stage + 1]))
               for stage in range(len(stage_ids))]
    manifest = {
        "protocol": P.PROTOCOL_NAME,
        "enumeration_protocol": P.ENUMERATION_PROTOCOL,
        "method": "random_synthetic_control",
        "dataset": args.dataset,
        "rate": 0.05,
        "budgets": budgets,
        "stage_classes": data["stage_classes"],
        "stage_pool_counts": [len(v) for v in stage_ids],
        "student_seeds": list(args.student_seeds),
        "student": {"batch": P.STUDENT.batch, "lr": P.STUDENT.lr,
                     "steps": "max(50,min(2048,16*rows))"},
        "construction_labels": "class protocol only; no report/test labels",
        "text_cache": str(cache) if cache else None,
        "test_labels_evaluated": False,
        "memory_management": "C0 fixed-rate append; no pruning",
        "random_operator": "balanced random class partition; Mean-X and concat/re-encode",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    memory = None
    results = []
    for stage, ids in enumerate(stage_ids):
        budget = budgets[stage]
        keep = 0 if memory is None else min(len(memory["x"]), budget)
        new_rows = budget - keep
        started = time.perf_counter()
        block, audit = random_synthetic_block(
            data, ids, new_rows, stage, 6400 + stage * 101)
        memory = SCA.append_memory(memory, block, keep,
                                   topology_mode="frozen_old",
                                   cross_degree=1, refresh_mode="frozen_old",
                                   device=args.device)
        torch.save(memory, out / f"random_synthetic_s{stage}_memory.pt")
        build_seconds = float(time.perf_counter() - started)
        scores = [fit_stage(data, memory, stage, seed, args.device)
                  for seed in args.student_seeds]
        row = {"arm": "random_synthetic", "stage": stage,
               "budget": budget, "keep": keep, "new_rows": new_rows,
               "build_seconds": build_seconds,
               "construction": audit, "metrics": aggregate(scores, stage),
               "per_seed": scores}
        results.append(row)
        (out / "progress.json").write_text(
            json.dumps({"manifest": manifest, "results": results}, indent=2))
        print(json.dumps({"dataset": args.dataset, "stage": stage,
                          "final_or_current_f1": row["metrics"][
                              "cumulative_macro_f1_mean"]}), flush=True)
    manifest["results"] = results
    (out / "results.json").write_text(json.dumps(manifest, indent=2))
    print("RANDOM_SYNTHETIC_DONE", flush=True)


if __name__ == "__main__":
    main()
