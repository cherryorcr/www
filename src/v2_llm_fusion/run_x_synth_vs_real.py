"""Isolate the X design choice: synthetic cluster-mean X vs REAL node X.

Same clustering, same text fusion (frozen rule), same topology, same budget k --
ONLY the synthetic row's feature changes:
  mean_x      : x_j = mean of cluster members' X   (the frozen design)
  medoid_x    : x_j = X of the member nearest the cluster centre (real row)
  mean_x_norm : mean X, then L2-normalised (matches the scale used inside joint)

Also a control:
  medoid_x_medoid_text : real medoid X AND that node's FULL original text
                         (the pure "select real representatives" baseline)

If medoid_x >> mean_x, the frozen design's "all-synthetic mean features" choice
is the weak link on these datasets, not the text fusion.
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
from run_multiscale_asm import (norm_adj, train_student, load_dataset, maxmin_lloyd_gpu,
                                knn_topology_gpu)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--rate", type=float, default=0.05)
    ap.add_argument("--seeds", type=int, nargs="+", default=[700, 701, 702, 703, 704])
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
    te = encoder.encode([texts[int(i)] for i in pool])
    Xp = x[pool].float()
    joint = torch.cat((F.normalize(Xp, dim=1), te), dim=1)
    assign = maxmin_lloyd_gpu(joint, k, device)
    jg = joint.to(device)
    print(f"[{args.dataset}] pool={len(pool)} k={k}", flush=True)

    x_mean, x_med, x_mean_norm, texts_fused, texts_med, mass = [], [], [], [], [], []
    for c in assign.unique():
        mg = torch.where(assign == c)[0]
        if len(mg) == 0:
            continue
        m = mg.cpu()
        x_mean.append(Xp[m].mean(0))
        cen = jg[mg].mean(0, keepdim=True)
        near = int((jg[mg] - cen).pow(2).sum(1).argmin())
        med_local = mg[near].cpu()          # index within pool
        x_med.append(Xp[med_local])
        Xmn = Xp[m].mean(0)
        x_mean_norm.append(Xmn / Xmn.norm().clamp_min(1e-9))
        order = m[(jg[mg] - cen).pow(2).sum(1).cpu().argsort()][:5]
        per = max(48, 320 // max(1, len(order)))
        texts_fused.append(" ".join(texts[int(pool[i])][:per] for i in order))
        texts_med.append(texts[int(pool[med_local])])   # FULL original text
        mass.append(float(len(m)))

    def synth(X, T, tx):
        E = knn_topology_gpu(T, 1.0, device)
        return {"x": X, "text_emb": T, "texts": tx, "edge_index": E,
                "weights": torch.tensor(mass)}

    Xm = torch.stack(x_mean)
    Xmed = torch.stack(x_med)
    Xmn = torch.stack(x_mean_norm)
    Tf = encoder.encode(texts_fused)
    Tm = encoder.encode(texts_med)

    arms = {
        "mean_x + fused_text":   synth(Xm,   Tf, texts_fused),
        "medoid_x + fused_text": synth(Xmed, Tf, texts_fused),
        "mean_norm_x + fused":   synth(Xmn,  Tf, texts_fused),
        "medoid_x + real_text":  synth(Xmed, Tm, texts_med),
    }

    res = {}
    for nm, art in arms.items():
        f1s, acs = [], []
        for s in args.seeds:
            model = train_student(art, s, device, min(2048, 16 * k))
            pred = model.logits(xg, real_adj, prompt_emb.to(device)).argmax(1).cpu()
            yy, pp = y[D["dev_mask"]].numpy(), pred[D["dev_mask"]].numpy()
            f1s.append(100 * f1_score(yy, pp, average="macro", zero_division=0))
            acs.append(100 * accuracy_score(yy, pp))
            del model
        res[nm] = {"macro_f1": float(np.mean(f1s)), "macro_f1_std": float(np.std(f1s)),
                   "accuracy": float(np.mean(acs)), "scores": f1s}
        print(f"  {nm:24s} F1={np.mean(f1s):6.2f}±{np.std(f1s):.2f} Acc={np.mean(acs):6.2f}",
              flush=True)

    json.dump({"dataset": args.dataset, "k": k, "results": res, "test_accessed": False},
              open(out / "results.json", "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
