"""Check #1: does a SMOOTHNESS statistic beat n_pool at predicting jitter gain?

Mechanism under test (redundancy-smoothing hypothesis): jitter helps because the
cluster-mean X has been smoothed so much that rows are too similar to each other.
If true, a directly measured smoothness of the constructed X should track the
jitter gain BETTER than the pool size n_pool (whose ordering already failed: the
gain is non-monotonic in n_pool: +0.54 @13.5k, -0.60 @27k, +1.17 @54k).

Candidate statistics on the FROZEN constructed artifact (label-free, closed form):
  r_mean   mean pairwise cosine similarity of the constructed X rows   (higher = smoother)
  r_p95    95th pct of the pairwise cosine distribution
  spread   mean L2 distance of rows to their mean   (lower = smoother)
  spec     effective rank ratio: sum(eig)^2 / sum(eig^2) / k   (lower = more collapsed)
  tsim     same mean pairwise cosine but on text_emb (control: is it X-specific?)
  npool    baseline proxy (pool size)

We recompute the same subsample scales as run_scale_gate (frac 0.05..1.0, seed 7000)
so that the measured jitter gains can be paired with the statistics, then report
Spearman correlation of each statistic with the gain, using BOTH amplitudes.

The gain numbers are taken from the scale-gate run (same construction recipe), so
this script only needs to build the artifacts and compute statistics -- no student
training -- making it fast.
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
from scipy.stats import spearmanr, pearsonr

sys.path.insert(0, "/home/pc/research/TAG_condensation/code/v2_codex")
from run_multiscale_asm import build_initial, load_dataset

# jitter gains measured in the scale-gate run (frac -> {amp: (mean_delta, wins)})
GAINS_FROM_SCALE_GATE = {
    0.05:  {"0.9": (0.01, 3), "1.79": (-5.14, 0)},
    0.10:  {"0.9": (0.54, 4), "1.79": (-1.34, 2)},
    0.20:  {"0.9": (-0.60, 1), "1.79": (-2.02, 0)},
    0.40:  {"0.9": (1.17, 4), "1.79": (1.76, 4)},
    1.00:  {"0.9": (1.19, 4), "1.79": (1.97, 4)},
}


def pairwise_cos_stats(x, max_pairs=2_000_000, seed=0):
    """Mean / p95 of pairwise cosine similarity among rows of x."""
    xn = F.normalize(x.float(), dim=1)
    k = len(xn)
    idx = torch.triu_indices(k, k, offset=1)
    if idx.shape[1] > max_pairs:
        g = torch.Generator().manual_seed(seed)
        sel = torch.randperm(idx.shape[1], generator=g)[:max_pairs]
        idx = idx[:, sel]
    sims = (xn[idx[0]] * xn[idx[1]]).sum(1)
    return float(sims.mean()), float(torch.quantile(sims, 0.95)), float(sims.std())


def effective_rank(x):
    """Participation ratio of the spectrum, normalized by k. Lower = collapsed."""
    xc = x.float() - x.float().mean(0, keepdim=True)
    s = torch.linalg.svdvals(xc)
    ev = s.pow(2)
    pr = float(ev.sum().pow(2) / ev.pow(2).sum().clamp_min(1e-12))
    return pr / len(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--fracs", type=float, nargs="+", default=[0.05, 0.10, 0.20, 0.40, 1.00])
    ap.add_argument("--rate", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=7000)
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "results.json").exists():
        raise FileExistsError("immutable")
    device = args.device

    D = load_dataset("arxiv", "/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip", device)
    x, texts, n = D["x"], D["texts"], D["n"]
    full_pool = torch.where(D["pool_mask"])[0]
    encoder = D["encoder"]

    rows = []
    for frac in args.fracs:
        g = torch.Generator().manual_seed(args.seed)
        perm = torch.randperm(len(full_pool), generator=g)
        npool = max(1, int(frac * len(full_pool)))
        pool = full_pool[perm[:npool].sort().values]
        k = max(2, math.floor(args.rate * len(pool)))
        text_emb_pool = encoder.encode([texts[int(i)] for i in pool])
        art = build_initial(x[pool], text_emb_pool, [texts[int(i)] for i in pool],
                            k, encoder, device)
        X = art["x"]
        rm, rp95, rsd = pairwise_cos_stats(X)
        spread = float((X - X.mean(0, keepdim=True)).norm(dim=1).mean())
        er = effective_rank(X)
        tm, _, _ = pairwise_cos_stats(art["text_emb"])
        rec = {"frac": frac, "n_pool": int(len(pool)), "k": int(k),
               "r_mean": rm, "r_p95": rp95, "r_std": rsd, "spread": spread,
               "eff_rank": er, "text_sim": tm}
        rec["gain_amp0.9"] = GAINS_FROM_SCALE_GATE[frac]["0.9"][0]
        rec["gain_amp1.79"] = GAINS_FROM_SCALE_GATE[frac]["1.79"][0]
        rows.append(rec)
        print(f"n_pool={len(pool):7d} k={k:5d} r_mean={rm:.4f} spread={spread:.6f} "
              f"eff_rank={er:.4f} text_sim={tm:.4f} | gain .9={rec['gain_amp0.9']:+.2f} "
              f"1.79={rec['gain_amp1.79']:+.2f}", flush=True)

    print("\n=== Spearman correlation with jitter gain ===", flush=True)
    keys = ["r_mean", "r_p95", "spread", "eff_rank", "text_sim", "n_pool"]
    corr = {}
    for gk in ("gain_amp0.9", "gain_amp1.79"):
        gains = [r[gk] for r in rows]
        corr[gk] = {}
        print(f"\n[{gk}]", flush=True)
        for kk in keys:
            vals = [r[kk] for r in rows]
            rho, p = spearmanr(vals, gains)
            corr[gk][kk] = {"spearman": float(rho), "p": float(p)}
            print(f"  {kk:10s} rho={rho:+.3f} p={p:.3f}", flush=True)

    # monotonicity of the best statistic vs the non-monotonic gain
    json.dump({"rows": rows, "spearman": corr,
               "note": "gains taken from the scale-gate run (same constructor/seed)",
               "test_accessed": False}, open(out / "results.json", "w"), indent=1)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
