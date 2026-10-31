"""Cora + Citeseer: is deterministic jitter HARMFUL on small graphs? (task ③)

If arxiv needs jitter (+1.5~2.0) but small graphs are hurt by it, the module must
be declared large-graph-only. Uses the same constructor as the main method.
Small graphs have very few rows (k=25/137 etc.), so we use each graph's own
row-norm scale (relative amplitude) rather than arxiv's absolute delta norm.

Amplitudes are RELATIVE to X0's mean row norm: {0, 0.25, 0.5, 1.0, 2.0}x.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

import protocol_v3

sys.path.insert(0, "/home/pc/research/TAG_condensation/code/v2_codex")
from run_multiscale_asm import (norm_adj, train_student, build_initial, load_dataset)


def bootstrap_ci(diffs, n_boot=10000, alpha=0.05, seed=0):
    d = np.asarray(diffs, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(d, size=len(d), replace=True).mean() for _ in range(n_boot)])
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(d.mean()), float(lo), float(hi)


def jitter_rows(x, rel, seed):
    g = torch.Generator().manual_seed(seed)
    noise = torch.randn(x.shape, generator=g)
    noise = noise / noise.norm(dim=1, keepdim=True).clamp_min(1e-9)
    return x + noise * rel * x.norm(dim=1, keepdim=True)


def run_graph(name, minilm, device, seeds, rels, jseed,
              student_batch, student_steps):
    D = load_dataset(name, minilm, device)
    x, texts, y, n = D["x"], D["texts"], D["y"], D["n"]
    pool = torch.where(D["pool_mask"])[0]
    k = math.floor(0.05 * len(pool))
    encoder = D["encoder"]
    prompt_emb = encoder.encode(D["prompts"])
    real_adj = norm_adj(D["edge_index"], n, device)
    xg = x.float().to(device)
    text_emb_pool = encoder.encode([texts[int(i)] for i in pool])
    initial = build_initial(x[pool], text_emb_pool, [texts[int(i)] for i in pool],
                            k, encoder, device)
    X0, T = initial["x"], initial["text_emb"]

    def mk(xnew):
        return {"x": xnew, "text_emb": T, "texts": initial["texts"],
                "edge_index": initial["edge_index"], "weights": initial["weights"]}

    def f1_for(syn, seed):
        steps = (student_steps if student_steps is not None else
                 protocol_v3.STUDENT.steps(k))
        model = train_student(syn, seed, device, steps, batch=student_batch)
        pred = model.logits(xg, real_adj, prompt_emb.to(device)).argmax(1).cpu()
        yy, pp = y[D["dev_mask"]].numpy(), pred[D["dev_mask"]].numpy()
        v = 100 * f1_score(yy, pp, average="macro", zero_division=0)
        del model
        return v

    print(f"\n[{name}] n={n} pool={len(pool)} k={k}", flush=True)
    res = {"k": k, "arms": {}}
    base_scores = [f1_for(mk(X0), s) for s in seeds]
    res["arms"]["0.0"] = {"scores": base_scores, "mean": float(np.mean(base_scores))}
    print(f"  rel=0.00 mean={np.mean(base_scores):.2f}", flush=True)
    for rel in rels:
        if rel == 0:
            continue
        scores = [f1_for(mk(jitter_rows(X0, rel, jseed)), s) for s in seeds]
        diffs = [scores[i] - base_scores[i] for i in range(len(seeds))]
        m, lo, hi = bootstrap_ci(diffs)
        res["arms"][str(rel)] = {"scores": scores, "mean": float(np.mean(scores)),
                                 "vs_frozen": m, "ci95": [lo, hi],
                                 "wins": int(sum(1 for v in diffs if v > 0))}
        print(f"  rel={rel:4.2f} mean={np.mean(scores):6.2f} vs_frozen={m:+.2f} "
              f"CI=[{lo:+.2f},{hi:+.2f}]", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=list(protocol_v3.STUDENT_SEEDS))
    ap.add_argument("--rels", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.0, 2.0])
    ap.add_argument("--jseed", type=int, default=protocol_v3.METHOD_SEED)
    ap.add_argument("--graph", default="both", choices=["cora", "citeseer", "both"])
    ap.add_argument("--minilm", default="/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip")
    ap.add_argument("--student-batch", type=int, default=protocol_v3.STUDENT.batch)
    ap.add_argument("--student-steps", type=int, default=None)
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "results.json").exists():
        raise FileExistsError("immutable: existing run")
    graphs = ["cora", "citeseer"] if args.graph == "both" else [args.graph]
    results = {}
    for g in graphs:
        results[g] = run_graph(g, args.minilm, args.device, args.seeds, args.rels,
                               args.jseed, args.student_batch, args.student_steps)
    json.dump({**protocol_v3.config_dict(student_batch=args.student_batch,
               student_steps=args.student_steps, student_seeds=args.seeds),
               "results": results, "rels": args.rels, "jseed": args.jseed,
               "test_accessed": False}, open(out / "results.json", "w"), indent=1)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
