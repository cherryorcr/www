"""arxiv anchored optimization: constructor init + few ASM steps (+ optional
proximity anchor ||X - X_init||^2), the natural form of the revised method.

Reuses the EXACT asm_optimize inner/outer loop from run_multiscale_asm (the one
that produced asm +3.31 and the X-only 21.86) so results are comparable; the
only addition is an optional proximity anchor to the initialized X.

Sweep: steps in {5,20,50,100}, anchor lambda in {0, 1e-2}; track X drift from
init -> does a few steps (or an anchor) capture most of the gain while staying
near the constructor?
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
import run_multiscale_asm as M
from run_multiscale_asm import (CLIPStudent, norm_adj, train_student, build_initial,
                                load_dataset, _func_graph, _func_text, _asm_loss)


def anchored_asm(initial, teacher, ref_rows_local, device, steps, inner, batch,
                 outer_lr, inner_lr, anchor_lambda):
    """asm_optimize + proximity anchor. Keeps the original loop verbatim."""
    k = len(initial["x"])
    syn_x = initial["x"].clone().to(device)
    x_init = syn_x.detach().clone()
    text_emb = initial["text_emb"].to(device)
    edges = initial["edge_index"]
    adj = norm_adj(edges, k, device)
    ref = ref_rows_local.to(device)

    teacher.eval()
    with torch.no_grad():
        teacher_logits = teacher.response(syn_x, adj, text_emb)[ref][:, ref]

    var = torch.nn.Parameter(syn_x.clone())
    opt = torch.optim.Adam([var], lr=outer_lr)
    curve = []
    for step in range(1, steps + 1):
        torch.manual_seed(1000 * step + 7)
        student = CLIPStudent(syn_x.shape[1]).to(device)
        params = dict(student.named_parameters())
        gen = torch.Generator().manual_seed(step * 13 + 1)
        rows_all = torch.randperm(k, generator=gen)
        inner_params = {n: p for n, p in params.items()}
        for it in range(inner):
            ridx = rows_all[it * batch:(it + 1) * batch]
            if len(ridx) < 2:
                continue
            g = _func_graph(inner_params, var, adj)
            tt = _func_text(inner_params, text_emb)
            logits = student.logit_scale.exp().clamp(max=100) * (g[ridx] @ tt[ridx].T) / 0.07
            target = torch.arange(len(ridx), device=device)
            loss = (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target)) / 2
            grads = torch.autograd.grad(loss, tuple(inner_params.values()),
                                        create_graph=True, allow_unused=True)
            grads = [gd if gd is not None else torch.zeros_like(v)
                     for gd, v in zip(grads, inner_params.values())]
            inner_params = {n: v - inner_lr * gd
                            for (n, v), gd in zip(inner_params.items(), grads)}
        g = _func_graph(inner_params, var, adj)
        tt = _func_text(inner_params, text_emb)
        student_logits = (g[ref] @ tt[ref].T) / 0.07
        asm = _asm_loss(student_logits, teacher_logits)
        opt.zero_grad(set_to_none=True)
        asm.backward()
        if anchor_lambda > 0:
            var.grad = var.grad + 2 * anchor_lambda * (var.detach() - x_init)
        opt.step()
        with torch.no_grad():
            var.clamp_(-10, 10)
        curve.append(float(asm.detach()))
        if step % 10 == 0 or step == steps:
            print(f"    step{step}: asm_loss={curve[-1]:.4f}", flush=True)

    out = {kk: initial[kk] for kk in initial}
    out["x"] = var.detach().cpu()
    return out, curve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rate", type=float, default=0.05)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--student-seeds", type=int, nargs="+",
                        default=list(protocol_v3.STUDENT_SEEDS))
    parser.add_argument("--student-batch", type=int, default=protocol_v3.STUDENT.batch)
    parser.add_argument("--student-steps", type=int, default=None)
    parser.add_argument("--steps-grid", type=int, nargs="+", default=[5, 20, 50, 100])
    parser.add_argument("--anchor-grid", type=float, nargs="+", default=[0.0, 1e-2])
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

    ref = torch.randperm(k, generator=torch.Generator().manual_seed(777))[:1024]

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

    def drift(xa):
        return float((xa - initial["x"]).norm(dim=1).mean())

    results = {}
    t0 = time.time()
    results["frozen"] = {"scores": dev_f1({**initial, "x": initial["x"]}),
                         "sec": time.time() - t0, "drift": 0.0}
    print(f"frozen: {np.mean(results['frozen']['scores']):.2f}", flush=True)

    for lam in args.anchor_grid:
        for steps in args.steps_grid:
            key = f"steps{steps}_lam{lam}"
            t0 = time.time()
            art, curve = anchored_asm(initial, teacher, ref, device, steps,
                                      inner=protocol_v3.ASM_INNER_STEPS,
                                      batch=args.student_batch,
                                      outer_lr=protocol_v3.ASM_OUTER_LR,
                                      inner_lr=5e-3, anchor_lambda=lam)
            r = dev_f1(art)
            results[key] = {"scores": r, "sec": time.time() - t0, "drift": drift(art["x"]),
                            "curve_tail": curve[-1]}
            torch.save(art, out / f"artifact_{key}.pt")
            print(f"{key}: F1={np.mean(r):.2f} drift={results[key]['drift']:.2f} "
                  f"loss={curve[-1]:.4f}", flush=True)

    json.dump({"results": {a: {"scores": results[a]["scores"],
                               "mean": float(np.mean(results[a]["scores"])),
                               "drift": results[a]["drift"],
                               "curve_tail": results[a].get("curve_tail")}
                           for a in results},
               "k": k, "student_batch": args.student_batch,
               "student_steps": (args.student_steps if args.student_steps is not None else
                                  protocol_v3.student_steps(k)),
               "test_accessed": False, **protocol_v3.config_dict()},
              open(out / "results.json", "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
