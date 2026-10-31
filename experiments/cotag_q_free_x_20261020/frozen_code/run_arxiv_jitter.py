"""arxiv: deterministic-jitter amplitude sweep + robustness (tasks ①②).

Finding to formalize: on arxiv (169k) any isotropic X perturbation of ~ASM's
magnitude gives +1.5~2.0 over frozen; geometric debiasing does not work; ASM
is no better than random. So the defensible module is "constructor + fixed-seed
deterministic jitter". This script pins down its two free parameters.

  Task ①: amplitude sweep. ||delta X|| row-norm in a grid; find the optimum.
  Task ②: robustness. At the best amplitude, run K independent fixed seeds
          (deterministic given the seed) x 5 student seeds -> is the +1.5~2.0
          stable across jitter seeds, and what is the spread?

Same constructor artifact, same text/edges, only X perturbed. Paired bootstrap
over student seeds; also report the across-jitter-seed spread.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
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
    means = np.array([rng.choice(d, size=len(d), replace=True).mean()
                      for _ in range(n_boot)])
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(d.mean()), float(lo), float(hi)


def jitter_rows(x, row_norm, seed):
    """Deterministic per-row isotropic perturbation with given row norm."""
    g = torch.Generator().manual_seed(seed)
    noise = torch.randn(x.shape, generator=g)
    noise = noise / noise.norm(dim=1, keepdim=True).clamp_min(1e-9) * row_norm
    return x + noise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate", type=float, default=0.05)
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=list(protocol_v3.STUDENT_SEEDS))
    ap.add_argument("--student-batch", type=int, default=protocol_v3.STUDENT.batch)
    ap.add_argument("--student-steps", type=int, default=None)
    ap.add_argument("--amplitudes", type=float, nargs="+",
                    default=[0.0, 0.2, 0.45, 0.9, 1.79, 3.6])
    ap.add_argument("--robust-seeds", type=int, nargs="+",
                    default=list(protocol_v3.JITTER_SEEDS))
    ap.add_argument("--best-amp", type=float, default=-1.0,
                    help="amplitude for the robustness stage; <0 = pick argmax of sweep")
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "results.json").exists():
        raise FileExistsError("immutable: existing run")
    device = args.device

    D = load_dataset("arxiv", "/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip", device)
    x, texts, y, n = D["x"], D["texts"], D["y"], D["n"]
    pool = torch.where(D["pool_mask"])[0]
    k = math.floor(args.rate * len(pool))
    encoder = D["encoder"]
    prompt_emb = encoder.encode(D["prompts"])
    real_adj = norm_adj(D["edge_index"], n, device)
    xg = x.float().to(device)
    print(f"[arxiv] n={n} pool={len(pool)} k={k}", flush=True)

    text_emb_pool = encoder.encode([texts[int(i)] for i in pool])
    initial = build_initial(x[pool], text_emb_pool, [texts[int(i)] for i in pool],
                            k, encoder, device)
    torch.save(initial, out / "frozen_artifact.pt")
    X0, T = initial["x"], initial["text_emb"]

    def mk(xnew):
        return {"x": xnew, "text_emb": T, "texts": initial["texts"],
                "edge_index": initial["edge_index"], "weights": initial["weights"]}

    def f1_for(syn, seed):
        steps = protocol_v3.student_steps(k, override=args.student_steps)
        model = train_student(syn, seed, device, steps, batch=args.student_batch)
        pred = model.logits(xg, real_adj, prompt_emb.to(device)).argmax(1).cpu()
        yy, pp = y[D["dev_mask"]].numpy(), pred[D["dev_mask"]].numpy()
        v = 100 * f1_score(yy, pp, average="macro", zero_division=0)
        del model
        return v

    results = {"frozen_scores": [], "sweep": {}, "robust": {}}

    # frozen baseline
    for s in args.seeds:
        results["frozen_scores"].append(f1_for(mk(X0), s))
    frozen_mean = float(np.mean(results["frozen_scores"]))
    print(f"frozen: {frozen_mean:.2f}", flush=True)
    results["frozen_mean"] = frozen_mean

    # ---- Task ① amplitude sweep (fixed jitter seed 9001, all student seeds) ----
    print("\n== sweep ==", flush=True)
    for amp in args.amplitudes:
        if amp == 0.0:
            scores = results["frozen_scores"]
        else:
            syn = mk(jitter_rows(X0, amp, 9001))
            scores = [f1_for(syn, s) for s in args.seeds]
        diffs = [scores[i] - results["frozen_scores"][i] for i in range(len(args.seeds))]
        m, lo, hi = bootstrap_ci(diffs)
        results["sweep"][str(amp)] = {"scores": scores, "mean": float(np.mean(scores)),
                                      "vs_frozen": m, "ci95": [lo, hi],
                                      "wins": int(sum(1 for v in diffs if v > 0))}
        print(f"  amp={amp:5.2f} mean={np.mean(scores):6.2f} vs_frozen={m:+.2f} "
              f"CI=[{lo:+.2f},{hi:+.2f}]", flush=True)

    best_amp = args.best_amp
    if best_amp < 0:
        best_amp = max((a for a in args.amplitudes if a > 0),
                       key=lambda a: results["sweep"][str(a)]["mean"])
    results["best_amp"] = best_amp
    print(f"\nbest amp = {best_amp}", flush=True)

    # ---- Task ② robustness across jitter seeds at best amplitude ----
    print("\n== robustness ==", flush=True)
    jitter_means = []
    for js in args.robust_seeds:
        syn = mk(jitter_rows(X0, best_amp, js))
        scores = [f1_for(syn, s) for s in args.seeds]
        diffs = [scores[i] - results["frozen_scores"][i] for i in range(len(args.seeds))]
        m, lo, hi = bootstrap_ci(diffs)
        results["robust"][str(js)] = {"scores": scores, "mean": float(np.mean(scores)),
                                      "vs_frozen": m, "ci95": [lo, hi],
                                      "wins": int(sum(1 for v in diffs if v > 0))}
        jitter_means.append(float(np.mean(scores)))
        print(f"  jseed={js} mean={np.mean(scores):6.2f} vs_frozen={m:+.2f} "
              f"CI=[{lo:+.2f},{hi:+.2f}] wins={results['robust'][str(js)]['wins']}/5", flush=True)

    # aggregate across jitter seeds: every (jitter, student) pair vs frozen
    all_diffs = []
    for js in args.robust_seeds:
        all_diffs += [results["robust"][str(js)]["scores"][i] - results["frozen_scores"][i]
                      for i in range(len(args.seeds))]
    m, lo, hi = bootstrap_ci(all_diffs)
    results["robust_pooled"] = {"mean": m, "ci95": [lo, hi], "n": len(all_diffs),
                                "jitter_mean_of_means": float(np.mean(jitter_means)),
                                "jitter_std_of_means": float(np.std(jitter_means)),
                                "jitter_min": float(np.min(jitter_means)),
                                "jitter_max": float(np.max(jitter_means)),
                                "test_accessed": False}
    json.dump({**protocol_v3.config_dict(dataset="arxiv", rate=args.rate,
               student_batch=args.student_batch,
               student_steps=(args.student_steps if args.student_steps is not None else
                              protocol_v3.student_steps(k)),
               student_seeds=args.seeds), **results},
              open(out / "results.json", "w"), indent=1)
    print(f"\nrobust pooled vs_frozen={m:+.2f} CI=[{lo:+.2f},{hi:+.2f}] "
          f"n={len(all_diffs)}  jitter means {np.min(jitter_means):.2f}..{np.max(jitter_means):.2f} "
          f"(std {np.std(jitter_means):.2f})", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
