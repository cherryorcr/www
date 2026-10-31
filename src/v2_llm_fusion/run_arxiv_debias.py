"""arxiv: zero-cost DETERMINISTIC debiasing arms vs frozen / random / ASM.

Hypothesis from the 5-seed result: the constructor's mean-X is "under-adapted" —
it lives at a different scale/geometry from the frozen MiniLM fused text, so ANY
sufficient drift helps (matched random noise gives +2.0). If that is right, a
zero-cost deterministic transform of X should recover most of that +2.0 without
any optimization or randomness.

Arms (all: same constructor artifact, same text/edges/mass, only X transformed):
  frozen                X as constructed
  whiten                X whitened (PCA -> unit variance per component, keep dim)
  scale_align           per-dimension scale match of X to text_emb (zero-mean/unit-var
                        per dim on both, then rescale X to text_emb's per-dim std)
  procrustes            orthogonal W minimizing ||X W - text_emb||_F (fit on syn rows)
  whiten_scale          whiten then scale_align
  rand / asm            reference controls (rand = matched-norm noise, asm = 50-step)

Judgement: paired over student seeds, bootstrap 95% CI of arm - frozen, reported
next to the rand/asm references. If whiten ~ +2 and asm-whiten CI includes 0,
then the "optimization module" should be replaced by a deterministic debias.
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
from run_multiscale_asm import (norm_adj, train_student, build_initial,
                                load_dataset, asm_optimize)


def bootstrap_ci(diffs, n_boot=10000, alpha=0.05, seed=0):
    d = np.asarray(diffs, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(d, size=len(d), replace=True).mean()
                      for _ in range(n_boot)])
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(d.mean()), float(lo), float(hi)


def whiten(x):
    """PCA whitening: center, rotate to PCs, scale to unit variance, keep dim."""
    xc = x - x.mean(0, keepdim=True)
    # svd on centered data
    U, S, Vh = torch.linalg.svd(xc, full_matrices=False)
    z = xc @ Vh.T
    z = z / z.std(0, keepdim=True).clamp_min(1e-9)
    return z


def scale_align(x, t):
    """Match X's per-dim scale/offset to the text embedding's per-dim stats."""
    xc = (x - x.mean(0, keepdim=True)) / x.std(0, keepdim=True).clamp_min(1e-9)
    return xc * t.std(0, keepdim=True) + t.mean(0, keepdim=True)


def procrustes(x, t):
    """Orthogonal W minimizing ||X W - T||_F (no scaling, no translation)."""
    xc = x - x.mean(0, keepdim=True)
    tc = t - t.mean(0, keepdim=True)
    M = xc.T @ tc
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    W = U @ Vh
    return xc @ W + t.mean(0, keepdim=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate", type=float, default=0.05)
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=list(protocol_v3.STUDENT_SEEDS))
    ap.add_argument("--student-batch", type=int, default=protocol_v3.STUDENT.batch)
    ap.add_argument("--student-steps", type=int, default=None)
    ap.add_argument("--asm-steps", type=int, default=protocol_v3.ASM_STEPS)
    ap.add_argument("--rand-draws", type=int, default=5)
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
    print(f"[arxiv] n={n} pool={len(pool)} k={k} seeds={args.seeds}", flush=True)

    text_emb_pool = encoder.encode([texts[int(i)] for i in pool])
    initial = build_initial(x[pool], text_emb_pool, [texts[int(i)] for i in pool],
                            k, encoder, device)
    torch.save(initial, out / "frozen_artifact.pt")
    X0 = initial["x"]
    T = initial["text_emb"]
    print(f"X0 row norm={X0.norm(dim=1).mean():.3f}  T row norm={T.norm(dim=1).mean():.3f}", flush=True)

    # ---- deterministic debias arms (X only) ----
    xw = whiten(X0)
    xs = scale_align(X0, T)
    xp = procrustes(X0, T)
    xws = scale_align(whiten(X0), T)
    debias = {"whiten": xw, "scale_align": xs, "procrustes": xp, "whiten_scale": xws}
    for nm, xv in debias.items():
        print(f"  {nm:14s} row norm={xv.norm(dim=1).mean():.3f} "
              f"dist_to_X0={(xv-X0).norm(dim=1).mean():.3f}", flush=True)

    # teacher for ASM reference arm
    pool_remap = torch.full((n,), -1, dtype=torch.long)
    pool_remap[pool] = torch.arange(len(pool))
    e = D["edge_index"]
    keep = (pool_remap[e[0]] >= 0) & (pool_remap[e[1]] >= 0)
    real_pool_syn = {"x": x[pool].float(), "text_emb": text_emb_pool,
                     "texts": [texts[int(i)] for i in pool],
                     "edge_index": torch.stack((pool_remap[e[0, keep]], pool_remap[e[1, keep]])),
                     "weights": torch.ones(len(pool))}
    teacher = train_student(real_pool_syn, protocol_v3.TEACHER_SEED, device,
                            protocol_v3.TEACHER_STEPS, batch=args.student_batch)
    pool_adj = norm_adj(real_pool_syn["edge_index"], len(pool), device)
    ref = torch.randperm(len(pool), generator=torch.Generator().manual_seed(
        protocol_v3.REFERENCE_SEED))[:min(1024, len(pool))]
    t0 = time.time()
    asm_art, curve = asm_optimize(initial, teacher, x[pool], text_emb_pool,
                                  [texts[int(i)] for i in pool], ref, device,
                                  source_adj=pool_adj,
                                  steps=args.asm_steps,
                                  inner=protocol_v3.ASM_INNER_STEPS,
                                  batch=args.student_batch,
                                  outer_lr=protocol_v3.ASM_OUTER_LR)
    dx_norm = float((asm_art["x"] - X0).norm(dim=1).mean())
    print(f"asm done {time.time()-t0:.0f}s dx_norm={dx_norm:.3f}", flush=True)

    def mk(xnew):
        return {"x": xnew, "text_emb": T, "texts": initial["texts"],
                "edge_index": initial["edge_index"], "weights": initial["weights"]}

    arms = {"frozen": mk(X0), "asm": asm_art}
    for nm, xv in debias.items():
        arms[nm] = mk(xv)
    for d in range(args.rand_draws):
        gen = torch.Generator().manual_seed(5000 + d)
        noise = torch.randn(X0.shape, generator=gen)
        noise = noise / noise.norm(dim=1, keepdim=True).clamp_min(1e-9) * dx_norm
        arms[f"rand_{d}"] = mk(X0 + noise)

    def f1_for(syn, seed):
        steps = protocol_v3.student_steps(k, override=args.student_steps)
        model = train_student(syn, seed, device, steps, batch=args.student_batch)
        pred = model.logits(xg, real_adj, prompt_emb.to(device)).argmax(1).cpu()
        yy, pp = y[D["dev_mask"]].numpy(), pred[D["dev_mask"]].numpy()
        v = 100 * f1_score(yy, pp, average="macro", zero_division=0)
        del model
        return v

    per_seed = {a: {} for a in arms}
    for seed in args.seeds:
        for a, syn in arms.items():
            v = f1_for(syn, seed)
            per_seed[a][seed] = v
        print(f"  seed{seed}: " + " ".join(f"{a}={per_seed[a][seed]:.2f}" for a in arms), flush=True)

    def paired(base, other):
        diffs = [per_seed[other][s] - per_seed[base][s] for s in args.seeds]
        mean, lo, hi = bootstrap_ci(diffs)
        return {"diffs": diffs, "mean": mean, "ci95": [lo, hi],
                "wins": int(sum(1 for v in diffs if v > 0)), "n": len(diffs),
                "ci_excludes_zero": bool(lo > 0 or hi < 0)}

    summary = {"seeds": args.seeds, "k": k, "asm_steps": args.asm_steps,
               "dx_norm": dx_norm, "test_accessed": False,
               "arm_means": {a: float(np.mean([per_seed[a][s] for s in args.seeds])) for a in arms},
               "vs_frozen": {a: paired("frozen", a) for a in arms if a != "frozen"}}
    rand_pool = []
    for d in range(args.rand_draws):
        rand_pool += summary["vs_frozen"][f"rand_{d}"]["diffs"]
    m, lo, hi = bootstrap_ci(rand_pool)
    summary["rand_pooled"] = {"mean": m, "ci95": [lo, hi], "n": len(rand_pool),
                              "ci_excludes_zero": bool(lo > 0 or hi < 0)}

    json.dump({"summary": summary, "per_seed": {a: per_seed[a] for a in per_seed},
               **protocol_v3.config_dict(dataset="arxiv", rate=args.rate,
                                          student_batch=args.student_batch,
                                          student_steps=(args.student_steps if
                                                         args.student_steps is not None else
                                                         protocol_v3.student_steps(k)),
                                          student_seeds=args.seeds,
                                          asm_steps=args.asm_steps)},
              open(out / "results.json", "w"), indent=1)

    print("\n==== SUMMARY ====", flush=True)
    for a in sorted(arms, key=lambda z: -summary["arm_means"][z]):
        m = summary["arm_means"][a]
        if a in summary["vs_frozen"]:
            v = summary["vs_frozen"][a]
            print(f"{a:16s} mean={m:.2f}  vs_frozen={v['mean']:+.2f} "
                  f"CI95=[{v['ci95'][0]:+.2f},{v['ci95'][1]:+.2f}] wins={v['wins']}/{v['n']}")
        else:
            print(f"{a:16s} mean={m:.2f}  (baseline)")
    rp = summary["rand_pooled"]
    print(f"rand pooled vs_frozen={rp['mean']:+.2f} CI95=[{rp['ci95'][0]:+.2f},{rp['ci95'][1]:+.2f}]")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
