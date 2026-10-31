"""Locate the scale gate S: at what graph size does deterministic jitter start to help?

Design: subsample the arxiv pool to a range of sizes (same data distribution, same
40 labels, same strategy, same rate), and at each size compare
  frozen  ->  jitter(amp)  for amp in a grid
with paired student seeds. This isolates SCALE as the only variable.

Subsampling: draw `frac` of the allowed pool (node_id%10 not in {1,2}); train the
teacher on the real induced subgraph of that subsample; construct on the same
subsample; evaluate always on the SAME full-size dev split (dev nodes are excluded
from pooling by construction, so no leakage).

Caveat to record: subsampled arxiv has lower intra-class redundancy per class than
the full graph (fewer nodes/class), so it is a proxy for scale, not identical to a
smaller dataset of the same type.
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


def jitter_rows(x, row_norm, seed):
    g = torch.Generator().manual_seed(seed)
    noise = torch.randn(x.shape, generator=g)
    noise = noise / noise.norm(dim=1, keepdim=True).clamp_min(1e-9) * row_norm
    return x + noise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--fracs", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.4, 1.0])
    ap.add_argument("--rate", type=float, default=0.05)
    ap.add_argument("--amps", type=float, nargs="+", default=[0.0, 0.9, 1.79])
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=list(protocol_v3.STUDENT_SEEDS))
    ap.add_argument("--jseed", type=int, default=protocol_v3.METHOD_SEED)
    ap.add_argument("--student-batch", type=int, default=protocol_v3.STUDENT.batch)
    ap.add_argument("--student-steps", type=int, default=None)
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "results.json").exists():
        raise FileExistsError("immutable: existing run")
    device = args.device

    D = load_dataset("arxiv", "/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip", device)
    x, texts, y, n = D["x"], D["texts"], D["y"], D["n"]
    full_pool = torch.where(D["pool_mask"])[0]
    encoder = D["encoder"]
    prompt_emb = encoder.encode(D["prompts"])
    real_adj = norm_adj(D["edge_index"], n, device)
    xg = x.float().to(device)
    e = D["edge_index"]
    print(f"[arxiv] n={n} full_pool={len(full_pool)}", flush=True)

    results = {"fracs": {}, "test_accessed": False}

    def f1_for(syn, seed, k):
        steps = (args.student_steps if args.student_steps is not None else
                 protocol_v3.STUDENT.steps(k))
        model = train_student(syn, seed, device, steps, batch=args.student_batch)
        pred = model.logits(xg, real_adj, prompt_emb.to(device)).argmax(1).cpu()
        yy, pp = y[D["dev_mask"]].numpy(), pred[D["dev_mask"]].numpy()
        v = 100 * f1_score(yy, pp, average="macro", zero_division=0)
        del model
        return v

    for frac in args.fracs:
        g = torch.Generator().manual_seed(7000)
        perm = torch.randperm(len(full_pool), generator=g)
        npool = max(1, int(frac * len(full_pool)))
        pool = full_pool[perm[:npool].sort().values]
        k = max(2, math.floor(args.rate * len(pool)))

        # induced subgraph on the subsampled pool
        remap = torch.full((n,), -1, dtype=torch.long)
        remap[pool] = torch.arange(len(pool))
        keep = (remap[e[0]] >= 0) & (remap[e[1]] >= 0)
        sub_edges = torch.stack((remap[e[0, keep]], remap[e[1, keep]]))

        text_emb_pool = encoder.encode([texts[int(i)] for i in pool])
        initial = build_initial(x[pool], text_emb_pool, [texts[int(i)] for i in pool],
                                k, encoder, device)
        X0 = initial["x"]
        tag = f"frac{frac}_n{len(pool)}_k{k}"
        print(f"\n[{tag}]", flush=True)
        res = {"n_pool": int(len(pool)), "k": int(k), "amps": {}}

        for amp in args.amps:
            X = X0 if amp == 0 else jitter_rows(X0, amp, args.jseed)
            syn = {"x": X, "text_emb": initial["text_emb"], "texts": initial["texts"],
                   "edge_index": initial["edge_index"], "weights": initial["weights"]}
            scores = [f1_for(syn, s, k) for s in args.seeds]
            res["amps"][str(amp)] = {"scores": scores, "mean": float(np.mean(scores)),
                                     "std": float(np.std(scores))}
            print(f"  amp={amp:4.2f} mean={np.mean(scores):6.2f} "
                  f"(n_pool={len(pool)}, k={k})", flush=True)

        # paired deltas vs amp=0
        base = res["amps"]["0.0"]["scores"]
        for amp in args.amps:
            if amp == 0:
                continue
            s = res["amps"][str(amp)]["scores"]
            diffs = [s[i] - base[i] for i in range(len(args.seeds))]
            m, lo, hi = bootstrap_ci(diffs)
            res["amps"][str(amp)]["vs_frozen"] = m
            res["amps"][str(amp)]["ci95"] = [lo, hi]
            res["amps"][str(amp)]["wins"] = int(sum(1 for v in diffs if v > 0))
            print(f"  amp={amp:4.2f} vs_frozen={m:+.2f} CI=[{lo:+.2f},{hi:+.2f}] "
                  f"wins={res['amps'][str(amp)]['wins']}/{len(args.seeds)}", flush=True)
        results["fracs"][tag] = res

    json.dump({**protocol_v3.config_dict(student_batch=args.student_batch,
               student_steps=args.student_steps, student_seeds=args.seeds),
               **results}, open(out / "results.json", "w"), indent=1)

    print("\n==== SCALE GATE ====", flush=True)
    print(f"{'n_pool':>8} {'k':>6} " + " ".join(f"amp{a}" for a in args.amps if a > 0))
    for tag, res in results["fracs"].items():
        line = f"{res['n_pool']:>8} {res['k']:>6} "
        for a in args.amps:
            if a == 0:
                continue
            v = res["amps"][str(a)]
            line += f"{v['vs_frozen']:+.2f}({v['wins']}) "
        print(line)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
