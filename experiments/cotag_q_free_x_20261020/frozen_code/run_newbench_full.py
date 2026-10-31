"""Full comparison on the NEW (never-touched) TAG benchmarks:
frozen vs deterministic jitter vs geometric debias, with paired student seeds.

Purpose: does the arxiv finding (jitter helps on large graphs) transfer to
independent datasets, and where does the scale gate S sit now that we have
four new sizes (wikics 9.4k, bookhis 33k, sportsfit 138k, bookchild 54k pool)?

For each dataset:
  frozen                constructor X
  jitter(eta)           row-wise noise, row-norm = eta * (mean row norm of X)
                        -- RELATIVE amplitude, so it is comparable across datasets
  debias(scale_align)   zero-cost deterministic control (expected to fail, as on arxiv)
paired over student seeds; bootstrap CI of each arm minus frozen.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score

import protocol_v3

sys.path.insert(0, "/home/pc/research/TAG_condensation/code/v2_codex")
from run_multiscale_asm import (norm_adj, train_student, build_initial, load_dataset)


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
    ap.add_argument("--etas", type=float, nargs="+", default=[0.9, 1.79])
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=list(protocol_v3.STUDENT_SEEDS))
    ap.add_argument("--student-batch", type=int, default=protocol_v3.STUDENT.batch)
    ap.add_argument("--student-steps", type=int, default=None)
    ap.add_argument("--jseed", type=int, default=protocol_v3.METHOD_SEED)
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "results.json").exists():
        raise FileExistsError("immutable")
    device = args.device

    D = load_dataset(args.dataset, "/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip", device)
    x, texts, y, n = D["x"], D["texts"], D["y"], D["n"]
    pool = torch.where(D["pool_mask"])[0]
    k = math.floor(args.rate * len(pool))
    encoder = D["encoder"]
    prompt_emb = encoder.encode(D["prompts"])
    real_adj = norm_adj(D["edge_index"], n, device)
    xg = x.float().to(device)
    print(f"[{args.dataset}] n={n} pool={len(pool)} k={k} classes={y.max().item()+1}", flush=True)

    text_emb_pool = encoder.encode([texts[int(i)] for i in pool])
    art = build_initial(x[pool], text_emb_pool, [texts[int(i)] for i in pool],
                        k, encoder, device)
    X0, T = art["x"], art["text_emb"]
    row_norm = float(X0.norm(dim=1).mean())
    print(f"  X row_norm={row_norm:.4f}", flush=True)

    def mk(xnew):
        return {"x": xnew, "text_emb": T, "texts": art["texts"],
                "edge_index": art["edge_index"], "weights": art["weights"]}

    def f1_for(syn, seed):
        steps = protocol_v3.student_steps(k, override=args.student_steps)
        model = train_student(syn, seed, device, steps, batch=args.student_batch)
        pred = model.logits(xg, real_adj, prompt_emb.to(device)).argmax(1).cpu()
        yy, pp = y[D["dev_mask"]].numpy(), pred[D["dev_mask"]].numpy()
        r = {"macro_f1": 100 * f1_score(yy, pp, average="macro", zero_division=0),
             "accuracy": 100 * accuracy_score(yy, pp)}
        del model
        return r

    def jitter(eta):
        g = torch.Generator().manual_seed(args.jseed)
        nz = torch.randn(X0.shape, generator=g)
        nz = nz / nz.norm(dim=1, keepdim=True).clamp_min(1e-9) * (eta * row_norm)
        return X0 + nz

    def scale_align(x, t):
        """Deterministic debias control.

        Per-dimension matching is only defined when dim(X) == dim(T) (e.g. arxiv,
        384==384). Some benchmarks (wikics) have raw X of a different width
        (300) than the text embedding (384); there we fall back to a GLOBAL
        scale match (match overall std of X to overall std of T), which is the
        only well-defined zero-cost control in that case.
        """
        xc = x - x.mean(0, keepdim=True)
        xs = xc.std(0, keepdim=True).clamp_min(1e-9)
        xc = xc / xs
        if x.shape[1] == t.shape[1]:
            return xc * t.std(0, keepdim=True) + t.mean(0, keepdim=True)
        # global fallback: match the overall scale of X to T's overall scale
        return xc * (t.std().clamp_min(1e-9) / xc.std().clamp_min(1e-9))

    arms = {"frozen": mk(X0), "scale_align": mk(scale_align(X0, T))}
    for e in args.etas:
        arms[f"jitter_{e}"] = mk(jitter(e))

    per_seed = {a: {} for a in arms}
    for s in args.seeds:
        for a, syn in arms.items():
            per_seed[a][s] = f1_for(syn, s)
        print("  seed%d: " % s + " ".join(f"{a}={per_seed[a][s]['macro_f1']:.2f}" for a in arms),
              flush=True)

    res = {}
    base_f1 = [per_seed["frozen"][s]["macro_f1"] for s in args.seeds]
    base_ac = [per_seed["frozen"][s]["accuracy"] for s in args.seeds]
    res["frozen"] = {"macro_f1": float(np.mean(base_f1)), "accuracy": float(np.mean(base_ac)),
                     "macro_f1_std": float(np.std(base_f1))}
    for a in arms:
        if a == "frozen":
            continue
        d_f1 = [per_seed[a][s]["macro_f1"] - per_seed["frozen"][s]["macro_f1"] for s in args.seeds]
        d_ac = [per_seed[a][s]["accuracy"] - per_seed["frozen"][s]["accuracy"] for s in args.seeds]
        m, lo, hi = boot(d_f1)
        ma, la, ha = boot(d_ac)
        res[a] = {"macro_f1": float(np.mean([per_seed[a][s]["macro_f1"] for s in args.seeds])),
                  "accuracy": float(np.mean([per_seed[a][s]["accuracy"] for s in args.seeds])),
                  "vs_frozen_f1": m, "ci95_f1": [lo, hi],
                  "vs_frozen_acc": ma, "ci95_acc": [la, ha],
                  "wins": int(sum(1 for v in d_f1 if v > 0))}
        print(f"  {a:14s} F1={res[a]['macro_f1']:.2f} vs_frozen={m:+.2f} "
              f"CI=[{lo:+.2f},{hi:+.2f}] wins={res[a]['wins']}/5", flush=True)

    json.dump({**protocol_v3.config_dict(dataset=args.dataset, rate=args.rate,
               student_batch=args.student_batch,
               student_steps=(args.student_steps if args.student_steps is not None else
                              protocol_v3.student_steps(k)),
               student_seeds=args.seeds),
               "dataset": args.dataset, "n": int(n), "n_pool": int(len(pool)), "k": int(k),
               "row_norm": row_norm, "results": res,
               "per_seed": {a: {str(s): per_seed[a][s] for s in args.seeds} for a in arms},
               "test_accessed": False},
              open(out / "results.json", "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
