"""Test the two-factor mechanism criterion for enabling jitter.

Hypothesis (revised after the new-benchmark results): jitter helps iff BOTH
  (a) the constructed rows are highly redundant -- measured label-free as the mean
      pairwise cosine of the FUSED TEXT (T_paircos), and
  (b) there are enough rows for the smoothing to matter -- measured as k (rows)
      or equivalently n_pool/C.
Evidence so far (6 datasets):
  bookchild T=0.20 k=3074 -> -1.86 ; bookhis T=0.22 k=1662 -> -1.39
  wikics    T=0.55 k= 468 -> -32.85; sportsfit T=0.69 k=6922 -> +3.63
  arxiv     T~0.70 k=6773 -> +1.97
  -> high T with small k (wikics) still fails, so both factors are needed.

This script computes the label-free statistics for all datasets we have, so the
criterion can be stated on measured quantities rather than on class counts, and
checks the predicted sign against the measured jitter effect.
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

sys.path.insert(0, "/home/pc/research/TAG_condensation/code/v2_codex")
from run_multiscale_asm import build_initial, load_dataset

# measured jitter@1.79 effect (relative amplitude) from the completed runs
KNOWN_EFFECT = {
    "wikics":    (-32.85, 0),
    "bookhis":   (-1.39, 1),
    "bookchild": (-1.86, 0),
    "sportsfit": (+3.63, 4),
    "arxiv":     (+1.97, 4),
}


def paircos(vecs, cap=400_000, seed=0):
    vn = F.normalize(vecs.float(), dim=1)
    k = len(vn)
    idx = torch.triu_indices(k, k, offset=1)
    if idx.shape[1] > cap:
        g = torch.Generator().manual_seed(seed)
        sel = torch.randperm(idx.shape[1], generator=g)[:cap]
        idx = idx[:, sel]
    sims = (vn[idx[0]] * vn[idx[1]]).sum(1)
    return float(sims.mean()), float(torch.quantile(sims, 0.95))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["wikics", "bookhis", "bookchild", "sportsfit", "arxiv",
                             "cora", "citeseer"])
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "results.json").exists():
        raise FileExistsError("immutable")
    device = args.device

    rows = []
    for ds in args.datasets:
        try:
            D = load_dataset(ds, "/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip", device)
        except Exception as e:
            print(f"skip {ds}: {str(e)[:80]}", flush=True)
            continue
        x, texts = D["x"], D["texts"]
        pool = torch.where(D["pool_mask"])[0]
        k = max(2, math.floor(0.05 * len(pool)))
        te = D["encoder"].encode([texts[int(i)] for i in pool])
        art = build_initial(x[pool], te, [texts[int(i)] for i in pool], k, D["encoder"], device)
        tm, tp95 = paircos(art["text_emb"])
        xm, _ = paircos(art["x"])
        if ds == "arxiv" and not (out / f"arxiv_stats.json").exists():
            json.dump({"t_paircos": tm, "x_paircos": xm, "k": k}, open(out / f"arxiv_stats.json", "w"))
        rec = {"dataset": ds, "n_pool": int(len(pool)), "k": int(k),
               "T_paircos": tm, "T_p95": tp95, "X_paircos": xm,
               "k_over_1000": k / 1000.0}
        if ds in KNOWN_EFFECT:
            rec["jitter_effect"] = KNOWN_EFFECT[ds][0]
            rec["jitter_wins"] = KNOWN_EFFECT[ds][1]
        rows.append(rec)
        print(f"{ds:11s} k={k:5d} T_paircos={tm:.4f} X_paircos={xm:.4f} "
              f"effect={rec.get('jitter_effect','?')}", flush=True)
        del D, art
        torch.cuda.empty_cache()

    # criterion check: enable iff T_paircos >= tau_T AND k >= tau_K
    def score(tau_T, tau_K):
        ok = 0; tot = 0
        for r in rows:
            if "jitter_effect" not in r:
                continue
            pred = (r["T_paircos"] >= tau_T) and (r["k"] >= tau_K)
            actual = r["jitter_effect"] > 0
            ok += (pred == actual); tot += 1
        return ok, tot

    best = None
    for tT in np.arange(0.30, 0.80, 0.05):
        for tK in [200, 500, 1000, 2000, 3000, 5000]:
            ok, tot = score(float(tT), int(tK))
            if tot and (best is None or ok / tot > best[0]):
                best = (ok / tot, float(tT), int(tK), ok, tot)

    print(f"\nbest criterion: T_paircos>={best[1]:.2f} AND k>={best[2]} "
          f"-> {best[3]}/{best[4]} correct", flush=True)
    json.dump({"rows": rows, "best_criterion": {"T_paircos_min": best[1], "k_min": best[2],
                                                "accuracy": best[3] / best[4],
                                                "n_evaluated": best[4]},
               "test_accessed": False}, open(out / "results.json", "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
