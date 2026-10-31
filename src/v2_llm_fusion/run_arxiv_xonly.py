"""arxiv follow-up: decisive X-only test + T1 no-proxy re-check at scale.

Key question after multiscale ASM (+3.31 at arxiv, loss flat):
is the gain (a) real optimization of X, or (b) weak perturbation of X?

Decisive test: the ASM artifact and the frozen artifact share IDENTICAL text /
topology / row count (k=6773); ONLY synthetic X differs. If F1 gain comes
purely from an X-space perturbation, then a RANDOM perturbation of the same
magnitude should give a similar gain.

Arms (all same text/k/topology):
  frozen            initial constructor X
  asm               ASM-optimized X (from the prior run's recipe)
  rand_perturb_x    frozen X + Gaussian noise with std matched to ||ΔX|| of ASM
  (2 draws)

Plus T1 re-check at scale: compute text-emb + X-emb shift/coverage/MMD stats
for a set of constructor variants (k in {1%,5%}, and noise arms), correlate
with their measured F1 -> does any proxy predict at 169k?
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
import torch.nn.functional as F
from sklearn.metrics import f1_score
from scipy.stats import spearmanr

import protocol_v3

sys.path.insert(0, "/home/pc/research/TAG_condensation/code/v2_codex")
from run_multiscale_asm import (FrozenMiniLM, CLIPStudent, norm_adj, train_student,
                                maxmin_lloyd_gpu, knn_topology_gpu, build_initial,
                                asm_optimize)
from run_multiscale_asm import load_dataset, DATA_CFG


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rate", type=float, default=0.05)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--student-seeds", type=int, nargs="+",
                        default=list(protocol_v3.STUDENT_SEEDS))
    parser.add_argument("--student-batch", type=int, default=protocol_v3.STUDENT.batch)
    parser.add_argument("--student-steps", type=int, default=None)
    parser.add_argument("--asm-steps", type=int, default=protocol_v3.ASM_STEPS)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "results.json").exists():
        raise FileExistsError("existing runs are immutable")
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

    # teacher for ASM
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

    def dev_f1(syn):
        scores = []
        for seed in args.student_seeds:
            steps = protocol_v3.student_steps(k, override=args.student_steps)
            model = train_student(syn, seed, device, steps, batch=args.student_batch)
            pred = model.logits(xg, real_adj, prompt_emb.to(device)).argmax(1).cpu()
            yy, pp = y[D["dev_mask"]].numpy(), pred[D["dev_mask"]].numpy()
            scores.append(100 * f1_score(yy, pp, average="macro", zero_division=0))
            del model
        return scores

    def mk(xnew):
        return {"x": xnew, "text_emb": initial["text_emb"], "texts": initial["texts"],
                "edge_index": initial["edge_index"], "weights": initial["weights"]}

    results = {}
    # frozen
    t0 = time.time()
    results["frozen"] = {"scores": dev_f1(mk(initial["x"])), "sec": time.time() - t0}
    print(f"frozen: {np.mean(results['frozen']['scores']):.2f}", flush=True)

    # ASM
    t0 = time.time()
    asm_art, curve = asm_optimize(initial, teacher, x[pool], text_emb_pool,
                                  [texts[int(i)] for i in pool], ref, device,
                                  source_adj=pool_adj,
                                  steps=args.asm_steps,
                                  inner=protocol_v3.ASM_INNER_STEPS,
                                  batch=args.student_batch,
                                  outer_lr=protocol_v3.ASM_OUTER_LR)
    asm_dx = (asm_art["x"] - initial["x"])
    dx_norm = float(asm_dx.norm(dim=1).mean())
    results["asm"] = {"scores": dev_f1(asm_art), "sec": time.time() - t0,
                      "dx_mean_row_norm": dx_norm}
    torch.save(asm_art, out / "asm_artifact.pt")
    print(f"asm: {np.mean(results['asm']['scores']):.2f} (dx row norm={dx_norm:.4f})", flush=True)

    # random perturbation with matched magnitude
    for draw in range(2):
        gen = torch.Generator().manual_seed(3000 + draw)
        noise = torch.randn(initial["x"].shape, generator=gen)
        noise = noise / noise.norm(dim=1, keepdim=True).clamp_min(1e-9) * dx_norm
        # random direction, but keep same per-row norm as ASM's delta magnitude
        xpert = initial["x"] + noise
        r = dev_f1(mk(xpert))
        results[f"rand_perturb_{draw}"] = {"scores": r, "sec": 0}
        print(f"rand_perturb_{draw}: {np.mean(r):.2f}", flush=True)

    # T1 stats at scale
    re_emb = encoder.encode([texts[int(i)] for i in pool])
    stats = {}
    for arm in results:
        xa = {"frozen": initial["x"],
              "asm": asm_art["x"]}.get(arm)
        if xa is None:
            continue
        te = F.normalize(initial["text_emb"], dim=1)
        sims = te @ re_emb.T
        stats[arm] = {
            "shift_nn": float((1 - sims.max(1).values).mean()),
            "cover_mean": float((1 - sims.max(0).values).mean()),
            "pair_cos": float(_pair_cos(te)),
            "x_norm_mean": float(xa.norm(dim=1).mean()),
        }
    f1s = [np.mean(results[a]["scores"]) for a in stats]
    print("\nT1-at-scale (arxiv):", flush=True)
    for key in ("shift_nn", "cover_mean", "pair_cos", "x_norm_mean"):
        vals = [stats[a][key] for a in stats]
        if len(set(vals)) > 1:
            sp = spearmanr(vals, f1s)
            print(f"  {key:12s} rho={sp.statistic:+.3f} p={sp.pvalue:.3f}", flush=True)

    json.dump({"results": {a: {"scores": results[a]["scores"],
                               "mean": float(np.mean(results[a]["scores"]))}
                           for a in results},
               "stats": stats, "k": k, "asm_loss_curve_tail": curve[-3:],
               "student_batch": args.student_batch,
               "student_steps": (args.student_steps if args.student_steps is not None else
                                  protocol_v3.student_steps(k)),
               "test_accessed": False, **protocol_v3.config_dict()},
              open(out / "results.json", "w"), indent=1)
    print("DONE", flush=True)


def _pair_cos(e):
    k = len(e)
    pairs = torch.triu_indices(k, k, offset=1)
    step = 20000
    tot, cnt = 0.0, 0
    for s in range(0, pairs.shape[1], step):
        p = pairs[:, s:s + step]
        tot += float((e[p[0]] * e[p[1]]).sum(1).sum())
        cnt += p.shape[1]
    return tot / max(cnt, 1)


if __name__ == "__main__":
    main()
