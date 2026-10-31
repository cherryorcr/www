"""Scan batch size for large graphs to verify batch=64 is near-optimal.

From BATCH_SIZE_CONFOUND: Cora optimal=16, sportsfit optimal≤138. Need to verify
whether the default batch=64 used in run_multiscale_asm is actually good for
BookChild/SportsFit/arXiv, or whether it's another arbitrary choice.

Sweep range: 16, 24, 32, 48, 64, 96, 128 (cover both small-graph optimum and
larger values).
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--rate", type=float, default=0.05)
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=list(protocol_v3.STUDENT_SEEDS))
    ap.add_argument("--batches", type=int, nargs="+",
                    default=[16, 24, 32, 48, 64, 96, 128])
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "results.json").exists():
        raise FileExistsError("immutable")
    device = args.device

    D = load_dataset(args.dataset,
                     "/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip",
                     device)
    x, texts, y, n = D["x"], D["texts"], D["y"], D["n"]
    pool = torch.where(D["pool_mask"])[0]
    k = max(2, math.floor(args.rate * len(pool)))
    encoder = D["encoder"]
    prompt_emb = encoder.encode(D["prompts"])
    real_adj = norm_adj(D["edge_index"], n, device)
    xg = x.float().to(device)
    te = encoder.encode([texts[int(i)] for i in pool])
    steps = protocol_v3.student_steps(k)
    print(f"[{args.dataset}] pool={len(pool)} k={k} steps={steps}", flush=True)

    ours = build_initial(x[pool], te, [texts[int(i)] for i in pool], k, encoder, device)

    res = {}
    for b in args.batches:
        bb = min(b, len(ours["x"]))
        f1s, acs = [], []
        for s in args.seeds:
            model = train_student(ours, s, device, steps, batch=bb)
            pred = model.logits(xg, real_adj, prompt_emb.to(device)).argmax(1).cpu()
            yy, pp = y[D["dev_mask"]].numpy(), pred[D["dev_mask"]].numpy()
            f1s.append(100 * f1_score(yy, pp, average="macro", zero_division=0))
            acs.append(100 * accuracy_score(yy, pp))
            del model
        res[f"b{bb}"] = {"macro_f1": float(np.mean(f1s)),
                         "macro_f1_std": float(np.std(f1s)),
                         "accuracy": float(np.mean(acs)),
                         "scores": f1s}
        print(f"  batch={bb:4d} F1={np.mean(f1s):6.2f}±{np.std(f1s):.2f} "
              f"Acc={np.mean(acs):6.2f}", flush=True)

    best_b = max(res.items(), key=lambda kv: kv[1]["macro_f1"])
    print(f"\n  BEST: {best_b[0]} F1={best_b[1]['macro_f1']:.2f}", flush=True)

    json.dump({**protocol_v3.config_dict(dataset=args.dataset, rate=args.rate,
               k=k, steps=steps, student_batch="sweep", student_seeds=args.seeds),
               "batches": args.batches,
               "results": res,
               "best_batch": best_b[0],
               "test_accessed": False},
              open(out / "results.json", "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
