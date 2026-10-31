"""arxiv: 5-seed confirmation of the ASM gain on the main method's own constructor,
plus multi-draw matched random-perturbation controls.

Purpose (task ①): is the +3.5 ASM gain on arxiv robust enough to admit the
conditional optimization module into the main method?

Protocol judgement (PROTOCOL v2 sec.6): paired per-seed differences, bootstrap
95% CI of the mean paired difference must exclude 0, AND report win rate.
Both frozen and ASM artifacts are fixed once (deterministic constructor, fixed
ASM seed); only the student eval seed varies — so this is the paired design.

Arms (all share frozen_v1's constructor: maxmin+Lloyd, mean X, top-5 fused text
re-encoded by MiniLM, text-kNN degree=1.0):
  frozen              constructor X
  asm                 ~50-step ASM on X  (the sweet spot from the anchored sweep)
  rand_perturb_d      frozen X + noise, per-row norm matched to ASM's ||dX||, d in {0..4}

Judgement targets:
  asm - frozen : paired, 5 seeds -> CI + win rate
  rand - frozen: if ~0 while asm > 0, the gain is a real optimization direction
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

import protocol_v3

sys.path.insert(0, "/home/pc/research/TAG_condensation/code/v2_codex")
from run_multiscale_asm import (CLIPStudent, norm_adj, train_student, build_initial,
                                load_dataset, asm_optimize)


def bootstrap_ci(diffs, n_boot=10000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    d = np.asarray(diffs, dtype=float)
    means = np.array([rng.choice(d, size=len(d), replace=True).mean()
                      for _ in range(n_boot)])
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(d.mean()), float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate", type=float, default=0.05)
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=list(protocol_v3.STUDENT_SEEDS))
    ap.add_argument("--draws", type=int, default=5)
    ap.add_argument("--asm-steps", type=int, default=protocol_v3.ASM_STEPS)
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

    # teacher on the full real pool (same recipe as the prior runs)
    pool_remap = torch.full((n,), -1, dtype=torch.long)
    pool_remap[pool] = torch.arange(len(pool))
    e = D["edge_index"]
    keep = (pool_remap[e[0]] >= 0) & (pool_remap[e[1]] >= 0)
    real_pool_syn = {"x": x[pool].float(), "text_emb": text_emb_pool,
                     "texts": [texts[int(i)] for i in pool],
                     "edge_index": torch.stack((pool_remap[e[0, keep]], pool_remap[e[1, keep]])),
                     "weights": torch.ones(len(pool))}
    teacher = train_student(real_pool_syn, protocol_v3.TEACHER_SEED, device,
                            protocol_v3.TEACHER_STEPS,
                            batch=args.student_batch)

    pool_adj = norm_adj(real_pool_syn["edge_index"], len(pool), device)
    ref = torch.randperm(len(pool), generator=torch.Generator().manual_seed(
        protocol_v3.REFERENCE_SEED))[:min(1024, len(pool))]

    def f1_for(syn, seed):
        steps = (args.student_steps if args.student_steps is not None else
                 protocol_v3.STUDENT.steps(k))
        model = train_student(syn, seed, device, steps, batch=args.student_batch)
        pred = model.logits(xg, real_adj, prompt_emb.to(device)).argmax(1).cpu()
        yy, pp = y[D["dev_mask"]].numpy(), pred[D["dev_mask"]].numpy()
        out = 100 * f1_score(yy, pp, average="macro", zero_division=0)
        del model
        return out

    def mk(xnew):
        return {"x": xnew, "text_emb": initial["text_emb"], "texts": initial["texts"],
                "edge_index": initial["edge_index"], "weights": initial["weights"]}

    # ASM (50 steps, unanchored)
    t0 = time.time()
    asm_art, curve = asm_optimize(initial, teacher, x[pool], text_emb_pool,
                                  [texts[int(i)] for i in pool], ref, device,
                                  source_adj=pool_adj,
                                  steps=args.asm_steps,
                                  inner=protocol_v3.ASM_INNER_STEPS,
                                  batch=args.student_batch,
                                  outer_lr=protocol_v3.ASM_OUTER_LR)
    dx = asm_art["x"] - initial["x"]
    dx_norm = float(dx.norm(dim=1).mean())
    torch.save(asm_art, out / "asm_artifact.pt")
    print(f"asm done in {time.time()-t0:.0f}s  loss {curve[0]:.4f}->{curve[-1]:.4f}  "
          f"dx_row_norm={dx_norm:.4f}", flush=True)

    # matched random perturbations
    rand_arts = {}
    for d in range(args.draws):
        gen = torch.Generator().manual_seed(5000 + d)
        noise = torch.randn(initial["x"].shape, generator=gen)
        noise = noise / noise.norm(dim=1, keepdim=True).clamp_min(1e-9) * dx_norm
        rand_arts[d] = mk(initial["x"] + noise)

    arms = {"frozen": mk(initial["x"]), "asm": asm_art}
    arms.update({f"rand_perturb_{d}": rand_arts[d] for d in rand_arts})

    per_seed = {a: {} for a in arms}
    for seed in args.seeds:
        for a, syn in arms.items():
            v = f1_for(syn, seed)
            per_seed[a][seed] = v
            print(f"  seed{seed} {a}: {v:.2f}", flush=True)

    def paired(base, other):
        diffs = [per_seed[other][s] - per_seed[base][s] for s in args.seeds]
        mean, lo, hi = bootstrap_ci(diffs)
        wins = int(sum(1 for x in diffs if x > 0))
        return {"diffs": diffs, "mean": mean, "ci95": [lo, hi], "wins": f"{wins}/{len(diffs)}",
                "ci_excludes_zero": bool(lo > 0 or hi < 0)}

    summary = {
        "seeds": args.seeds, "k": k, "asm_steps": args.asm_steps, "draws": args.draws,
        "dx_row_norm": dx_norm,
        "asm_loss_first": float(curve[0]), "asm_loss_last": float(curve[-1]),
        "arm_means": {a: float(np.mean(list(per_seed[a].values()))) for a in arms},
        "arm_std": {a: float(np.std(list(per_seed[a].values()))) for a in arms},
        "asm_minus_frozen": paired("frozen", "asm"),
        "rand_minus_frozen": {f"draw{d}": paired("frozen", f"rand_perturb_{d}")
                              for d in range(args.draws)},
        "test_accessed": False,
               **protocol_v3.config_dict(dataset="arxiv", rate=args.rate,
                                  student_batch=args.student_batch,
                                  student_steps=(args.student_steps if
                                                 args.student_steps is not None else
                                                 protocol_v3.STUDENT.steps(k)),
                                  student_seeds=args.seeds,
                                  asm_steps=args.asm_steps),
    }
    # aggregate over random draws
    all_rand_diffs = []
    for d in range(args.draws):
        all_rand_diffs.extend(summary["rand_minus_frozen"][f"draw{d}"]["diffs"])
    m, lo, hi = bootstrap_ci(all_rand_diffs)
    summary["rand_pooled"] = {"mean": m, "ci95": [lo, hi], "n": len(all_rand_diffs),
                              "ci_excludes_zero": bool(lo > 0 or hi < 0)}

    json.dump({"summary": summary, "per_seed": {a: per_seed[a] for a in per_seed}},
              open(out / "results.json", "w"), indent=1)

    print("\n==== SUMMARY ====", flush=True)
    for a in arms:
        print(f"{a:16s} mean={summary['arm_means'][a]:.2f} std={summary['arm_std'][a]:.2f}")
    af = summary["asm_minus_frozen"]
    print(f"\nasm - frozen : {af['mean']:+.2f}  CI95=[{af['ci95'][0]:+.2f},{af['ci95'][1]:+.2f}] "
          f"wins={af['wins']} excludes0={af['ci_excludes_zero']}")
    rp = summary["rand_pooled"]
    print(f"rand - frozen: {rp['mean']:+.2f}  CI95=[{rp['ci95'][0]:+.2f},{rp['ci95'][1]:+.2f}] "
          f"n={rp['n']} excludes0={rp['ci_excludes_zero']}")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
