"""Phase-2 minimal block-preserving joint-kernel append experiment.

The candidate is intentionally narrow: it reuses C0's synthetic X and text
rows and changes only the internal/cross-block topology kernel.  The graph
component is computed from the current delta's induced source graph.  Old
synthetic rows, old text, old-old edges, total budget, student, and report
protocol are frozen.
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

import joint_kernel_append as J
import run_unified_v3_continual as U
P, G = U.protocol_v3, U.G


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def tensor_hash(memory: dict) -> str:
    h = hashlib.sha256()
    for key in ("x", "text_emb", "mass", "depth", "stage",
                "graph_signature", "edge_index"):
        value = memory.get(key)
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().contiguous()
            h.update(key.encode())
            h.update(str(value.dtype).encode())
            h.update(str(tuple(value.shape)).encode())
            h.update(value.numpy().tobytes())
    h.update("\0".join(memory.get("texts", [])).encode("utf-8"))
    return h.hexdigest()


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
        raise ValueError("joint-kernel construction read excluded text rows")
    data["text_emb"] = text_emb
    data["prompt_emb"] = data["encoder"].encode(data["prompts"])
    data["real_adj"] = G.norm_adj(data["edge_index"], len(data["x"]), device)
    data["x_device"] = data["x"].float().to(device)
    return data


def compare_c0(candidate: dict, c0: dict, stage: int) -> dict:
    if len(candidate["x"]) != len(c0["x"]):
        raise AssertionError(f"stage {stage}: total rows differ from C0")
    for key in ("x", "text_emb", "mass", "depth", "stage"):
        if not torch.allclose(candidate[key].float(), c0[key].float(),
                              atol=1e-6, rtol=0):
            raise AssertionError(f"stage {stage}: C0 field changed: {key}")
    if candidate["texts"] != c0["texts"]:
        raise AssertionError(f"stage {stage}: C0 text rows changed")
    return {"rows": int(len(candidate["x"])),
            "x_text_mass_match_c0": True,
            "c0_tensor_hash": tensor_hash(c0)}


def fit_stage(data: dict, memory: dict, stage: int, seed: int,
              device: str) -> dict:
    syn = G.memory_to_trainable(memory)
    steps = P.student_steps(len(syn["x"]))
    started = time.perf_counter()
    model = G.train_student(syn, int(seed), device, steps,
                            lr=P.STUDENT.lr, batch=P.STUDENT.batch)
    score = G.eval_stagewise(model, data["x_device"], data["real_adj"],
                             data["prompt_emb"], data["y"],
                             data["report_dev_mask"], stage)
    score.update({"seed": int(seed), "steps": int(steps),
                  "seconds": float(time.perf_counter() - started)})
    del model
    return score


def aggregate(rows: list[dict], stage: int) -> dict:
    out = {"rows": len(rows), "stage": int(stage)}
    for key in ("cumulative_macro_f1", "current_f1", "old_mean_f1"):
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
    ap.add_argument("--c0-root", default=None,
                    help="p5_rate_curve_small_20260919/<dataset>_r005")
    ap.add_argument("--student-seeds", nargs="+", type=int,
                    default=list(P.STUDENT_SEEDS))
    ap.add_argument("--degree", type=float, default=P.DEGREE)
    ap.add_argument("--cross-degree", type=int, default=1)
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "results.json").exists():
        raise FileExistsError("existing joint-kernel run is immutable")
    cache = Path(args.text_cache) if args.text_cache else None
    data = load_data(args.dataset, args.minilm, args.device, cache)
    stage_ids = [torch.sort(torch.where(
        data["pool_mask"] & torch.isin(data["y"], torch.tensor(classes))
    )[0]).values for classes in data["stage_classes"]]
    budgets = [math.floor(0.05 * sum(len(v) for v in stage_ids[:stage + 1]))
               for stage in range(len(stage_ids))]
    c0_root = Path(args.c0_root) if args.c0_root else None
    manifest = {
        "protocol": P.PROTOCOL_NAME,
        "enumeration_protocol": P.ENUMERATION_PROTOCOL,
        "method": "block_preserving_joint_kernel_append",
        "dataset": args.dataset,
        "rate": 0.05,
        "budgets": budgets,
        "stage_classes": data["stage_classes"],
        "stage_pool_counts": [len(v) for v in stage_ids],
        "student_seeds": list(args.student_seeds),
        "student": {"batch": P.STUDENT.batch, "lr": P.STUDENT.lr,
                     "steps": "max(50,min(2048,16*rows))"},
        "kernel_weights": list(J.KERNEL_WEIGHTS),
        "kernel_components": {"x": "synthetic X cosine",
                              "text": "frozen MiniLM embedding cosine",
                              "graph": "current-delta induced one-hop joint summary"},
        "changed_surface": "new-block internal edges and old-new cross edges only",
        "frozen_surface": ["X", "text", "rows", "mass", "old rows",
                           "old-old edges", "student", "steps", "seeds"],
        "construction_labels": "current-delta class protocol only; no report/test labels",
        "test_labels_evaluated": False,
        "c0_root": str(c0_root) if c0_root else None,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    memory = None
    results = []
    for stage, ids in enumerate(stage_ids):
        budget = budgets[stage]
        keep = 0 if memory is None else min(len(memory["x"]), budget)
        new_rows = budget - keep
        started = time.perf_counter()
        block = J.build_joint_block(U, data, ids, new_rows, data["encoder"],
                                    stage, args.device)
        memory, append_audit = J.append_joint(
            memory, block, keep, degree=args.degree,
            cross_degree=args.cross_degree)
        c0_audit = None
        if c0_root is not None:
            c0_path = c0_root / f"append_fixed_s{stage}_memory.pt"
            if not c0_path.exists():
                raise FileNotFoundError(c0_path)
            c0 = torch.load(c0_path, map_location="cpu", weights_only=False)
            c0_audit = compare_c0(memory, c0, stage)
        torch.save(memory, out / f"joint_kernel_s{stage}_memory.pt")
        build_seconds = float(time.perf_counter() - started)
        scores = [fit_stage(data, memory, stage, seed, args.device)
                  for seed in args.student_seeds]
        row = {"arm": "joint_kernel", "stage": stage,
               "budget": budget, "keep": keep, "new_rows": new_rows,
               "build_seconds": build_seconds,
               "append_audit": append_audit, "c0_audit": c0_audit,
               "tensor_sha256": tensor_hash(memory),
               "metrics": aggregate(scores, stage), "per_seed": scores}
        results.append(row)
        (out / "progress.json").write_text(
            json.dumps({"manifest": manifest, "results": results}, indent=2))
        print(json.dumps({"dataset": args.dataset, "stage": stage,
                          "cumulative_f1": row["metrics"][
                              "cumulative_macro_f1_mean"]}), flush=True)
    manifest["results"] = results
    (out / "results.json").write_text(json.dumps(manifest, indent=2))
    print("JOINT_KERNEL_APPEND_DONE", flush=True)


if __name__ == "__main__":
    main()
