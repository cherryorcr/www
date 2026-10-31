"""Same-protocol external baselines for graph condensation.

The project's constraint: condensation is LABEL-FREE (no class labels used to
build the memory) and evaluation is ZERO-SHOT (class-prompt classification).
Supervised condensation methods (GCond, DosCond, SFGC, ...) do not fit this
setting -- they optimise synthetic features against a supervised loss, so any
number we produced for them would be either (a) their supervised variant under a
different protocol, or (b) a strawman. We therefore run:

  A. Label-free SELECTION baselines at the same row budget k (the standard
     coreset family), which are the honest same-setting competitors:
       - random_real       : k real training nodes, uniform
       - k_center          : greedy farthest-point selection in the joint space
       - herding           : greedy mean-matching (class-agnostic, label-free)
       - kmeans_medoid     : k-means on the joint space, take the real medoid of
                             each cluster (a "real-node" version of our method)
  B. Our constructor (frozen_v1) for reference.
  C. An ablation of our own constructor to isolate WHICH part matters:
       - ours_no_fuse      : mean X but a single nearest text (g=1)  -> fusion effect
       - ours_no_reencode  : mean of the 5 texts' embeddings instead of re-encoding
                             the concatenation -> tests the pairing/fusion design
       - ours_random_init  : random cluster seeding instead of max-min -> tests
                             determinism/quality of seeding

All arms are evaluated identically (same training-graph builder: text-kNN edges
for synthetic rows; for real-node selections we use the *induced real subgraph*
edges, which is the fair choice for a real-node baseline).

THIS IS THE "同一协议外部基线" REQUIRED BEFORE SUBMISSION.
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

import protocol_v3

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_multiscale_asm import (norm_adj, train_student, build_initial, load_dataset,
                                maxmin_lloyd_gpu, knn_topology_gpu)


def boot(diffs, n=10000, seed=0):
    d = np.asarray(diffs, float)
    g = np.random.default_rng(seed)
    m = np.array([g.choice(d, len(d), replace=True).mean() for _ in range(n)])
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--rate", type=float, default=0.05)
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=list(protocol_v3.STUDENT_SEEDS))
    ap.add_argument("--student-batch", type=int,
                    default=protocol_v3.STUDENT.batch)
    ap.add_argument("--student-steps", type=int, default=None)
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "results.json").exists():
        raise FileExistsError("immutable")
    device = args.device

    D = load_dataset(args.dataset, "/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip", device)
    x, texts, y, n = D["x"], D["texts"], D["y"], D["n"]
    pool = torch.where(D["pool_mask"])[0]
    k = max(2, math.floor(args.rate * len(pool)))
    encoder = D["encoder"]
    prompt_emb = encoder.encode(D["prompts"])
    real_adj = norm_adj(D["edge_index"], n, device)
    xg = x.float().to(device)
    print(f"[{args.dataset}] n={n} pool={len(pool)} k={k}", flush=True)

    te = encoder.encode([texts[int(i)] for i in pool])
    Xp = x[pool].float()
    joint = torch.cat((F.normalize(Xp, dim=1), te), dim=1)

    # ---------- arm builders ----------
    def synth_from_rows(X, T, texts_list, edges=None, degree=1.0):
        if edges is None:
            edges = knn_topology_gpu(T, degree, device)
        return {"x": X, "text_emb": T, "texts": texts_list,
                "edge_index": edges, "weights": torch.ones(len(X))}

    arms = {}

    # A1 random real nodes
    g = torch.Generator().manual_seed(11)
    sel = torch.randperm(len(pool), generator=g)[:k]
    rows = pool[sel]
    arms["random_real"] = synth_from_rows(Xp[sel], te[sel],
                                          [texts[int(i)] for i in rows])

    # A2 k-center (greedy farthest point) in joint space
    jj = joint.to(device)
    first = 0
    chosen = [first]
    dmin = (jj - jj[first]).pow(2).sum(1)
    for _ in range(k - 1):
        nx = int(dmin.argmax())
        chosen.append(nx)
        dmin = torch.minimum(dmin, (jj - jj[nx]).pow(2).sum(1))
    ci = torch.tensor(chosen)
    arms["k_center"] = synth_from_rows(Xp[ci], te[ci], [texts[int(i)] for i in pool[ci]])

    # A3 herding (mean matching, label-free): greedily pick the real row that
    # moves the running mean closest to the full-pool mean, in joint space.
    mu = jj.mean(0, keepdim=True)
    sel_idx, cur = [], torch.zeros_like(mu)
    avail = torch.ones(len(pool), dtype=torch.bool, device=device)
    for t in range(k):
        cand = (cur + jj) / (t + 1)
        dist = (cand - mu).pow(2).sum(1)
        dist[~avail] = float("inf")
        nx = int(dist.argmin())
        sel_idx.append(nx)
        cur = cur + jj[nx:nx + 1]
        avail[nx] = False
    hi = torch.tensor(sel_idx)
    arms["herding"] = synth_from_rows(Xp[hi], te[hi], [texts[int(i)] for i in pool[hi]])

    # A4 k-means medoids (real-node version of our method)
    assign = maxmin_lloyd_gpu(joint, k, device)
    med = []
    for c in assign.unique():
        m = torch.where(assign == c)[0]
        if len(m) == 0:
            continue
        cen = jj[m].mean(0, keepdim=True)
        med.append(int(m[(jj[m] - cen).pow(2).sum(1).argmin()]))
    mi = torch.tensor(med)
    arms["kmeans_medoid"] = synth_from_rows(Xp[mi], te[mi], [texts[int(i)] for i in pool[mi]])

    # B ours (frozen_v1)
    arms["ours"] = build_initial(x[pool], te, [texts[int(i)] for i in pool],
                                 k, encoder, device)

    # C ablations of our constructor
    #   no_fuse: single nearest text
    syn_x, fused, mass = [], [], []
    jg = joint.to(device)
    for c in assign.unique():
        m = torch.where(assign == c)[0]
        if len(m) == 0:
            continue
        syn_x.append(Xp[m].mean(0))
        cen = jg[m].mean(0, keepdim=True)
        near = m.cpu()[(jg[m] - cen).pow(2).sum(1).cpu().argmin()]
        fused.append(texts[int(pool[near])])
        mass.append(len(m))
    Xn = torch.stack(syn_x)
    Tn = encoder.encode(fused)
    arms["ours_no_fuse"] = synth_from_rows(Xn, Tn, fused)

    #   no_reencode: average the 5 nearest texts' embeddings instead of re-encoding
    embs, fused2 = [], []
    for c in assign.unique():
        m = torch.where(assign == c)[0]
        if len(m) == 0:
            continue
        cen = jg[m].mean(0, keepdim=True)
        order = m.cpu()[(jg[m] - cen).pow(2).sum(1).cpu().argsort()][:5]
        embs.append(te[order].mean(0))
        fused2.append(" ".join(texts[int(pool[i])][:64] for i in order))
    T2 = F.normalize(torch.stack(embs), dim=1)
    arms["ours_no_reencode"] = synth_from_rows(Xn, T2, fused2)

    #   random_init: random seeding then Lloyd (same k), everything else identical
    g2 = torch.Generator().manual_seed(123)
    seed_idx = torch.randperm(len(pool), generator=g2)[:k]
    C = jg[seed_idx.to(device)].clone()
    sidx = torch.arange(k, device=device)
    for _ in range(25):
        a = torch.cdist(jg, C).argmin(1)
        a[seed_idx.to(device)] = sidx
        for c in range(k):
            mm = a == c
            if mm.any():
                C[c] = jg[mm].mean(0)
    syn_x2, fused3 = [], []
    for c in range(k):
        mmg = torch.where(a == c)[0]
        if len(mmg) == 0:
            continue
        mm = mmg.cpu()
        syn_x2.append(Xp[mm].mean(0))
        cen = jg[mmg].mean(0, keepdim=True)
        order = mm[(jg[mmg] - cen).pow(2).sum(1).cpu().argsort()][:5]
        fused3.append(" ".join(texts[int(pool[i])][:64] for i in order))
    arms["ours_random_init"] = synth_from_rows(torch.stack(syn_x2),
                                               encoder.encode(fused3), fused3)

    print("  arms: " + ", ".join(f"{a}({len(arms[a]['x'])})" for a in arms), flush=True)

    # ---------- evaluate ----------
    res = {}
    for name, syn in arms.items():
        f1s, acs = [], []
        for seed in args.seeds:
            steps = (args.student_steps if args.student_steps is not None else
                     protocol_v3.STUDENT.steps(len(syn["x"])))
            model = train_student(syn, seed, device, steps,
                                  batch=args.student_batch)
            pred = model.logits(xg, real_adj, prompt_emb.to(device)).argmax(1).cpu()
            yy, pp = y[D["dev_mask"]].numpy(), pred[D["dev_mask"]].numpy()
            f1s.append(100 * f1_score(yy, pp, average="macro", zero_division=0))
            acs.append(100 * accuracy_score(yy, pp))
            del model
        res[name] = {"macro_f1": float(np.mean(f1s)), "macro_f1_std": float(np.std(f1s)),
                     "accuracy": float(np.mean(acs)), "accuracy_std": float(np.std(acs)),
                     "scores_f1": f1s}
        print(f"  {name:18s} F1={np.mean(f1s):6.2f}±{np.std(f1s):.2f} "
              f"Acc={np.mean(acs):6.2f}", flush=True)
        del syn

    base = res["ours"]["scores_f1"]
    for name in res:
        if name == "ours":
            continue
        d = [res[name]["scores_f1"][i] - base[i] for i in range(len(args.seeds))]
        m, lo, hi = boot(d)
        res[name]["vs_ours_f1"] = m
        res[name]["vs_ours_ci95"] = [lo, hi]
        res[name]["vs_ours_wins"] = int(sum(1 for v in d if v > 0))

    json.dump({**protocol_v3.config_dict(dataset=args.dataset, rate=args.rate,
               k=k, student_batch=args.student_batch,
               student_steps=(args.student_steps if args.student_steps is not None else
                              protocol_v3.STUDENT.steps(k)),
               student_seeds=args.seeds),
               "results": res,
               "test_accessed": False}, open(out / "results.json", "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
