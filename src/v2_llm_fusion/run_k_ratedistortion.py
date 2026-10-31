"""Check #2: does the rate-distortion ELBOW of the constructor align with the
start of the F1 plateau?  If yes, k can be set from a label-free closed form.

Per dataset: sweep the condensation rate (k/n_pool) over a wide grid; for each,
record
  distortion D(k) = mean squared distance of members to their assigned centre
                    (the k-means objective, normalized per node) -- label-free
  downstream dev Macro-F1 (student seeds)
Then define
  elbow  = smallest k where the marginal D reduction falls below 10% of the
           first-step reduction (a knee on the D-vs-k curve)
  plateau= smallest k whose F1 is within 1 std of the max F1 over the grid
and test whether elbow <= plateau (elbow lands at or before the plateau start).

Uses the frozen_v1 constructor; text tower per --dataset (arxiv uses MiniLM).
Distortion is computed in the SAME joint space the constructor clusters in.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score

sys.path.insert(0, "/home/pc/research/TAG_condensation/code/v2_codex")
from run_multiscale_asm import (norm_adj, train_student, build_initial, load_dataset,
                                maxmin_lloyd_gpu)


def distortion(joint, assign):
    """Normalized k-means objective in the clustering space."""
    d = 0.0
    for c in assign.unique():
        m = assign == c
        if m.sum() == 0:
            continue
        cen = joint[m].mean(0, keepdim=True)
        d += float((joint[m] - cen).pow(2).sum())
    return d / len(joint)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="arxiv", choices=["arxiv", "cora", "citeseer"])
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--rates", type=float, nargs="+",
                    default=[0.005, 0.01, 0.02, 0.05, 0.10, 0.20])
    ap.add_argument("--seeds", type=int, nargs="+", default=[200, 201, 202])
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "results.json").exists():
        raise FileExistsError("immutable")
    device = args.device

    minilm = "/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip"
    D = load_dataset(args.dataset, minilm, device)
    x, texts, y, n = D["x"], D["texts"], D["y"], D["n"]
    pool = torch.where(D["pool_mask"])[0]
    encoder = D["encoder"]
    prompt_emb = encoder.encode(D["prompts"])
    real_adj = norm_adj(D["edge_index"], n, device)
    xg = x.float().to(device)
    print(f"[{args.dataset}] n={n} pool={len(pool)}", flush=True)

    text_emb_pool = encoder.encode([texts[int(i)] for i in pool])
    joint = torch.cat((F.normalize(x[pool].float(), dim=1), text_emb_pool), dim=1)

    rows = []
    for rate in args.rates:
        k = max(2, math.floor(rate * len(pool)))
        assign = maxmin_lloyd_gpu(joint, k, device)
        dist = distortion(joint, assign)
        art = build_initial(x[pool], text_emb_pool, [texts[int(i)] for i in pool],
                            k, encoder, device)
        scores = []
        for seed in args.seeds:
            model = train_student(art, seed, device, min(2048, 16 * k))
            pred = model.logits(xg, real_adj, prompt_emb.to(device)).argmax(1).cpu()
            yy, pp = y[D["dev_mask"]].numpy(), pred[D["dev_mask"]].numpy()
            scores.append({"macro_f1": 100 * f1_score(yy, pp, average="macro", zero_division=0),
                           "accuracy": 100 * accuracy_score(yy, pp)})
            del model
        f1 = [s["macro_f1"] for s in scores]
        rows.append({"rate": rate, "k": int(k), "distortion": dist,
                     "f1_mean": float(np.mean(f1)), "f1_std": float(np.std(f1)),
                     "acc_mean": float(np.mean([s["accuracy"] for s in scores])),
                     "per_student": scores})
        print(f"rate={rate:6.3f} k={k:5d} D={dist:.6f} F1={np.mean(f1):.2f}"
              f"±{np.std(f1):.2f} Acc={rows[-1]['acc_mean']:.2f}", flush=True)

    # elbow: smallest k where marginal D reduction < 10% of the first-step reduction
    Ds = [r["distortion"] for r in rows]
    first_drop = Ds[0] - Ds[1] if len(Ds) > 1 else Ds[0]
    elbow_idx = len(rows) - 1
    for i in range(1, len(rows)):
        if (Ds[i - 1] - Ds[i]) < 0.10 * first_drop:
            elbow_idx = i
            break
    best_f1 = max(r["f1_mean"] for r in rows)
    best_std = rows[int(np.argmax([r["f1_mean"] for r in rows]))]["f1_std"]
    plateau_idx = len(rows) - 1
    for i, r in enumerate(rows):
        if best_f1 - r["f1_mean"] <= best_std:
            plateau_idx = i
            break

    summary = {
        "dataset": args.dataset, "n_pool": int(len(pool)),
        "elbow_rate": rows[elbow_idx]["rate"], "elbow_k": rows[elbow_idx]["k"],
        "elbow_distortion": rows[elbow_idx]["distortion"],
        "plateau_rate": rows[plateau_idx]["rate"], "plateau_k": rows[plateau_idx]["k"],
        "elbow_before_plateau": bool(rows[elbow_idx]["k"] <= rows[plateau_idx]["k"]),
        "first_step_drop": first_drop,
        "test_accessed": False,
    }
    json.dump({"rows": rows, "summary": summary}, open(out / "results.json", "w"), indent=1)

    print(f"\n=== {args.dataset} ===", flush=True)
    print(f"elbow  : rate={summary['elbow_rate']:.3f} k={summary['elbow_k']}", flush=True)
    print(f"plateau: rate={summary['plateau_rate']:.3f} k={summary['plateau_k']}", flush=True)
    print(f"elbow <= plateau : {summary['elbow_before_plateau']}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
