"""Re-run the ours-vs-baselines comparison with BATCH SWEPT per dataset.

Motivation: batch size turned out to be a 20+ point confound on small graphs
(see BATCH_SIZE_CONFOUND_ZH_2026-09-11.md). Every prior arm-vs-arm claim in this
project was measured at a single, arbitrarily-chosen batch. This script:

  1. sweeps batch for EACH arm independently (fair: each arm gets its best),
  2. reports both the per-arm optimum and the ranking at a COMMON batch,
  3. so we can tell whether k-center's advantage survives fair tuning.

Arms: ours (cluster-mean X + fused text), kcenter (farthest-point real nodes),
random_real, and raw-train reference for scale.
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

sys.path.insert(0, "/home/pc/research/TAG_condensation/code/v2_codex")
from run_multiscale_asm import (norm_adj, train_student, build_initial, load_dataset,
                                knn_topology_gpu)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--rate", type=float, default=0.05)
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=list(protocol_v3.STUDENT_SEEDS))
    ap.add_argument("--batches", type=int, nargs="+", default=[8, 16, 24, 32, 48, 64])
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
    steps = min(2048, 16 * k)
    print(f"[{args.dataset}] pool={len(pool)} k={k} steps={steps}", flush=True)

    ours = build_initial(x[pool], te, [texts[int(i)] for i in pool], k, encoder, device)

    joint = torch.cat((F.normalize(Xp, dim=1), te), dim=1).to(device)
    sel = [0]
    dmin = (joint - joint[0]).pow(2).sum(1)
    for _ in range(k - 1):
        nx = int(dmin.argmax()); sel.append(nx)
        dmin = torch.minimum(dmin, (joint - joint[nx]).pow(2).sum(1))
    ci = torch.tensor(sel)
    Tk = te[ci]
    kc = {"x": Xp[ci], "text_emb": Tk, "texts": [texts[int(i)] for i in pool[ci]],
          "edge_index": knn_topology_gpu(Tk, 1.0, device), "weights": torch.ones(len(ci))}

    g = torch.Generator().manual_seed(11)
    rs = torch.randperm(len(pool), generator=g)[:k]
    Tr = te[rs]
    rr = {"x": Xp[rs], "text_emb": Tr, "texts": [texts[int(i)] for i in pool[rs]],
          "edge_index": knn_topology_gpu(Tr, 1.0, device), "weights": torch.ones(len(rs))}

    arms = {"ours": ours, "kcenter": kc, "random_real": rr}
    res = {}
    for nm, art in arms.items():
        res[nm] = {}
        for b in args.batches:
            bb = min(b, len(art["x"]))
            f1s, acs = [], []
            for s in args.seeds:
                model = train_student(art, s, device, steps, batch=bb)
                pred = model.logits(xg, real_adj, prompt_emb.to(device)).argmax(1).cpu()
                yy, pp = y[D["dev_mask"]].numpy(), pred[D["dev_mask"]].numpy()
                f1s.append(100 * f1_score(yy, pp, average="macro", zero_division=0))
                acs.append(100 * accuracy_score(yy, pp))
                del model
            res[nm][f"b{bb}"] = {"macro_f1": float(np.mean(f1s)),
                                 "accuracy": float(np.mean(acs)), "scores": f1s}
            print(f"  {nm:12s} batch={bb:4d} F1={np.mean(f1s):6.2f} Acc={np.mean(acs):6.2f}",
                  flush=True)

    # summary: per-arm best, and ranking at each common batch
    best = {nm: max(res[nm].items(), key=lambda kv: kv[1]["macro_f1"]) for nm in res}
    print("\n  === per-arm BEST (fair tuning) ===", flush=True)
    for nm in res:
        bk, bv = best[nm]
        print(f"  {nm:12s} best {bk}: F1={bv['macro_f1']:.2f} Acc={bv['accuracy']:.2f}", flush=True)

    print("\n  === ranking at common batches ===", flush=True)
    for b in args.batches:
        vals = {nm: res[nm][f"b{min(b, len(arms[nm]['x']))}"]["macro_f1"] for nm in res}
        order = sorted(vals, key=lambda z: -vals[z])
        print(f"  batch={b:3d}: " + " > ".join(f"{nm}({vals[nm]:.2f})" for nm in order), flush=True)

    json.dump({**protocol_v3.config_dict(dataset=args.dataset, rate=args.rate,
               k=k, steps=steps, student_batch="sweep",
               student_seeds=args.seeds),
               "batches": args.batches,
               "results": res,
               "per_arm_best": {nm: {"batch": best[nm][0], **best[nm][1]} for nm in res},
               "test_accessed": False}, open(out / "results.json", "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
