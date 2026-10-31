"""Is k-center's advantage a STUDENT-TRAINING artifact?

Our student is trained for min(2048, 16*rows) steps. k-center rows are real
nodes (natural feature distribution); our rows are cluster means (smoothed).
If the smoothed rows simply need MORE optimisation, then sweeping the student
step count should close the gap. If the gap persists at every budget, the
difference is structural.

Arms (same budget k, same evaluator):
  ours    : cluster-mean X + top-5 fused text re-encoded, text-kNN edges
  kcenter : farthest-point real nodes, their X and full text, text-kNN edges

Step budgets swept: 0.5x, 1x, 2x, 4x of the standard min(2048, 16k), plus a
fixed-4096 and fixed-16384 condition for the larger graphs.

Also reports the training LOSS curve for both, to see whether ours is
under-optimised (train loss still falling) or has already converged.
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


def train_with_curve(syn, seed, device, steps, lr=protocol_v3.STUDENT.lr,
                     batch=protocol_v3.STUDENT.batch):
    """Same as train_student but records the loss trajectory."""
    from run_multiscale_asm import CLIPStudent
    torch.manual_seed(seed)
    model = CLIPStudent(syn["x"].shape[1]).to(device)
    x, t = syn["x"].to(device), syn["text_emb"].to(device)
    adj = norm_adj(syn["edge_index"], len(x), device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed * 7 + 1)
    n = len(x)
    curve = []
    for i in range(steps):
        rows = torch.randperm(n, generator=gen)[:min(batch, n)].to(device)
        g = model.encode_graph(x, adj)[rows]
        tt = model.encode_text(t[rows])
        logits = model.logit_scale.exp().clamp(max=100) * g @ tt.T
        target = torch.arange(len(rows), device=device)
        loss = (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target)) / 2
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if i % max(1, steps // 20) == 0:
            curve.append(round(float(loss), 5))
    return model.eval(), curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--rate", type=float, default=0.05)
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=list(protocol_v3.STUDENT_SEEDS))
    ap.add_argument("--student-batch", type=int,
                    default=protocol_v3.STUDENT.batch)
    ap.add_argument("--student-steps", type=int, default=None,
                    help="base steps before multiplier; otherwise protocol-scaled")
    ap.add_argument("--mult", type=float, nargs="+", default=[0.5, 1.0, 2.0, 4.0])
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
    base_steps = (args.student_steps if args.student_steps is not None else
                  protocol_v3.STUDENT.steps(k))
    print(f"[{args.dataset}] pool={len(pool)} k={k} base_steps={base_steps}", flush=True)

    # arm A: ours
    ours = build_initial(x[pool], te, [texts[int(i)] for i in pool], k, encoder, device)

    # arm B: k-center on real nodes
    joint = torch.cat((F.normalize(Xp, dim=1), te), dim=1).to(device)
    sel = [0]
    dmin = (joint - joint[0]).pow(2).sum(1)
    for _ in range(k - 1):
        nx = int(dmin.argmax())
        sel.append(nx)
        dmin = torch.minimum(dmin, (joint - joint[nx]).pow(2).sum(1))
    ci = torch.tensor(sel)
    Tk = te[ci]
    kc = {"x": Xp[ci], "text_emb": Tk, "texts": [texts[int(i)] for i in pool[ci]],
          "edge_index": knn_topology_gpu(Tk, 1.0, device), "weights": torch.ones(len(ci))}
    print(f"  ours rows={len(ours['x'])} kc rows={len(kc['x'])}", flush=True)

    res = {"ours": {}, "kcenter": {}}
    for mult in args.mult:
        steps = max(50, int(base_steps * mult))
        for nm, art in (("ours", ours), ("kcenter", kc)):
            f1s, acs, curves = [], [], []
            for s in args.seeds:
                model, curve = train_with_curve(art, s, device, steps,
                                                batch=args.student_batch)
                pred = model.logits(xg, real_adj, prompt_emb.to(device)).argmax(1).cpu()
                yy, pp = y[D["dev_mask"]].numpy(), pred[D["dev_mask"]].numpy()
                f1s.append(100 * f1_score(yy, pp, average="macro", zero_division=0))
                acs.append(100 * accuracy_score(yy, pp))
                curves.append(curve)
                del model
            res[nm][f"steps{steps}"] = {"macro_f1": float(np.mean(f1s)),
                                        "accuracy": float(np.mean(acs)),
                                        "loss_curve": curves[0]}
            print(f"  mult={mult:4.1f} steps={steps:5d} {nm:8s} "
                  f"F1={np.mean(f1s):6.2f} Acc={np.mean(acs):6.2f} "
                  f"loss0={curves[0][0]:.4f} lossEnd={curves[0][-1]:.4f}", flush=True)

    json.dump({**protocol_v3.config_dict(dataset=args.dataset, rate=args.rate,
               k=k, base_steps=base_steps,
               student_batch=args.student_batch, student_seeds=args.seeds),
               "results": res, "test_accessed": False},
              open(out / "results.json", "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
