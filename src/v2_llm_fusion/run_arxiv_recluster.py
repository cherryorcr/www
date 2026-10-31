"""arxiv class-incremental: recluster vs append (the ② question).

append    : keep old rows byte-identical; construct only the new delta.
recluster : re-cluster old memory (weighted by mass) TOGETHER with the current
            delta, allowing old-new mixed clusters (implicit capacity
            reallocation).  This is the operator that, on Cora/Citeseer, traded
            history retention for current-class plasticity.

Weights: old rows participate with weight=mass (a row representing 20 real nodes
counts 20x), new raw rows with weight=1 -- same rule as the frozen_v1 study.
Text: mixed clusters blend old and new selected texts; pure-old/pure-new keep
their own. X: mean over members (weighted). Eval identical to the continual run.

Purpose: on the class-incremental arxiv split, does mix-in-reclustering help the
current stages (plasticity) at the cost of old stages (retention) -- i.e. does
the same trade-off as Cora/Citeseer reproduce at 45k nodes / 13 classes/stage?
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
from sklearn.metrics import f1_score, accuracy_score
import protocol_v3

sys.path.insert(0, "/home/pc/research/TAG_condensation/code/v2_codex")
from run_multiscale_asm import FrozenMiniLM, norm_adj, train_student, load_dataset

DATA_DIR = Path("/home/pc/research/TAGSAM/data")


def maxmin_lloyd_weighted(joint, w, k, device, rounds=25):
    """Deterministic max-min seeding + weighted Lloyd. w: (n,) positive weights."""
    n = len(joint)
    j = joint.to(device)
    wg = w.to(device)
    centers = [j[0].clone()]
    seed_indices = [0]
    # farthest-point by weighted distance to nearest center
    dmin = (j - centers[0]).pow(2).sum(1)
    for _ in range(k - 1):
        idx = int(dmin.argmax())
        centers.append(j[idx].clone())
        seed_indices.append(idx)
        dmin = torch.minimum(dmin, (j - centers[-1]).pow(2).sum(1))
    C = torch.stack(centers)
    assign = None
    for _ in range(rounds):
        d = torch.cdist(j, C)
        new = d.argmin(1)
        # keep seeds in their own cluster
        new[torch.tensor(seed_indices, device=device)] = torch.arange(k, device=device)
        if assign is not None and torch.equal(new, assign):
            break
        assign = new
        for c in range(k):
            m = assign == c
            if m.any():
                ww = wg[m]
                C[c] = (j[m] * ww[:, None]).sum(0) / ww.sum().clamp_min(1e-9)
    return assign.cpu()


def construct_weighted(joint, x_raw, texts, w, k, encoder, device,
                       fuse_g=protocol_v3.FUSE_G,
                       fuse_chars=protocol_v3.FUSE_CHARS):
    """Weighted clustering -> synthetic rows; text fused from members' texts."""
    assign = maxmin_lloyd_weighted(joint, w, k, device)
    syn_x, fused, mass = [], [], []
    joint_gpu = joint.to(device)
    for c in range(k):
        mem = torch.where(assign == c)[0]
        if len(mem) == 0:
            continue
        ww = w[mem]
        syn_x.append((x_raw[mem] * ww[:, None]).sum(0) / ww.sum().clamp_min(1e-9))
        mg = joint_gpu[mem.to(device)]
        d = torch.cdist(mg, mg.mean(0, keepdim=True)).flatten()
        order = mem[d.argsort().cpu()][:fuse_g]
        per = max(48, fuse_chars // max(1, len(order)))
        fused.append(" ".join(texts[int(i)][:per] for i in order))
        mass.append(float(ww.sum()))
    syn_x = torch.stack(syn_x)
    fused_emb = encoder.encode(fused)
    return {"x": syn_x, "text_emb": fused_emb, "texts": fused,
            "edge_index": None, "weights": torch.tensor(mass)}


def rebuild_edges(mem, encoder, device, degree=protocol_v3.DEGREE):
    from run_multiscale_asm import knn_topology_gpu
    mem["edge_index"] = knn_topology_gpu(mem["text_emb"], degree, device)
    return mem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-prefix", default="ogbn_arxiv_full_classincr3")
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--rate", type=float, default=0.05)
    ap.add_argument("--student-seeds", type=int, nargs="+",
                    default=list(protocol_v3.STUDENT_SEEDS))
    ap.add_argument("--student-batch", type=int, default=protocol_v3.STUDENT.batch)
    ap.add_argument("--student-steps", type=int, default=None)
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "results.json").exists():
        raise FileExistsError("immutable")
    device = args.device

    D = load_dataset("arxiv", "/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip", device)
    x_full, texts_full, y_full, n = D["x"], D["texts"], D["y"], D["n"]
    encoder = D["encoder"]
    prompt_emb = encoder.encode(D["prompts"])
    text_emb_full = encoder.encode(texts_full)

    deltas, cums = [], []
    for t in (1, 2, 3):
        deltas.append(torch.load(DATA_DIR / f"{args.split_prefix}_delta_t{t}.pt",
                                 map_location="cpu", weights_only=False))
        cums.append(torch.load(DATA_DIR / f"{args.split_prefix}_cumulative_t{t}.pt",
                               map_location="cpu", weights_only=False))

    def pool_rows(dd):
        oid = dd["original_node_ids"]
        return torch.where(((oid % 10) != 1) & ((oid % 10) != 2))[0]

    pools = [pool_rows(dd) for dd in deltas]
    budgets = [math.floor(args.rate * c) for c in np.cumsum([len(p) for p in pools])]
    print(f"pools={[len(p) for p in pools]} budgets={budgets}", flush=True)

    def delta_view(i):
        dd = deltas[i]
        rows = pools[i]
        oid = dd["original_node_ids"][rows]
        return dd["node_f"][rows], text_emb_full[oid], [texts_full[int(v)] for v in oid]

    eval_cache = []
    for i in range(3):
        cc = cums[i]
        oid = cc["original_node_ids"]
        eval_cache.append((norm_adj(cc["edge_index"], len(oid), device),
                           cc["node_f"].float().to(device), y_full[oid],
                           protocol_v3.split_masks(oid)[0], cc["arrival_stage"].clone()))

    def dev_eval(mem, i):
        adj, xg, y_e, dev_m, st = eval_cache[i]
        scores = []
        for seed in args.student_seeds:
            model = train_student(mem, seed, device,
                                  protocol_v3.student_steps(len(mem["x"]),
                                                            override=args.student_steps),
                                  batch=args.student_batch)
            pred = model.logits(xg, adj, prompt_emb.to(device)).argmax(1).cpu()
            res = {"per_stage": []}
            ay, ap = [], []
            for s in range(i + 1):
                m = dev_m & (st == s)
                yy, pp = y_e[m].numpy(), pred[m].numpy()
                res["per_stage"].append({"macro_f1": 100 * f1_score(yy, pp, average="macro",
                                                                    labels=np.unique(yy), zero_division=0),
                                         "n": int(m.sum())})
                ay.append(yy); ap.append(pp)
            yy, pp = np.concatenate(ay), np.concatenate(ap)
            res["cumulative_macro_f1"] = 100 * f1_score(yy, pp, average="macro", labels=np.unique(yy), zero_division=0)
            res["cumulative_accuracy"] = 100 * accuracy_score(yy, pp)
            res["current_f1"] = res["per_stage"][-1]["macro_f1"]
            if i >= 1:
                res["old_mean_f1"] = float(np.mean([r["macro_f1"] for r in res["per_stage"][:-1]]))
            scores.append(res)
            del model
        return scores

    mem = None
    results = []
    for t in range(3):
        xd, ted, txt = delta_view(t)
        pool = torch.where(D["pool_mask"])[0] if False else None  # unused
        # current memory: if None -> construct from delta alone; else re-cluster
        if mem is None:
            joint_new = torch.cat((torch.nn.functional.normalize(xd.float(), dim=1), ted), dim=1)
            w_new = torch.ones(len(xd))
            sub = construct_weighted(joint_new, xd.float(), txt, w_new, budgets[t], encoder, device)
            mem = rebuild_edges(sub, encoder, device)
        else:
            # rebuild old members' joint repr from the memory's own rows:
            # (approximation: use memory rows as the pool, weighted by mass)
            joint_old = torch.cat((torch.nn.functional.normalize(mem["x"].float(), dim=1), mem["text_emb"]), dim=1)
            w_old = mem["weights"].clone()
            joint_new = torch.cat((torch.nn.functional.normalize(xd.float(), dim=1), ted), dim=1)
            w_new = torch.ones(len(xd))
            J = torch.cat((joint_old, joint_new), 0)
            X = torch.cat((mem["x"].float(), xd.float()), 0)
            T = list(mem["texts"]) + list(txt)
            W = torch.cat((w_old, w_new), 0)
            k = budgets[t]
            newmem = construct_weighted(J, X, T, W, k, encoder, device)
            newmem["arrival_stage"] = torch.zeros(len(newmem["x"]), dtype=torch.long)
            mem = rebuild_edges(newmem, encoder, device)
        scores = dev_eval(mem, t)
        cum = float(np.mean([s["cumulative_macro_f1"] for s in scores]))
        cur = float(np.mean([s["current_f1"] for s in scores]))
        old = float(np.mean([s["old_mean_f1"] for s in scores])) if t >= 1 else None
        results.append({"arm_stage": f"recluster_s{t}", "roots": int(len(mem["x"])),
                        "cum_mean": cum, "cur_mean": cur, "old_mean": old,
                        "per_student": scores})
        print(json.dumps({"arm_stage": f"recluster_s{t}", "roots": int(len(mem["x"])),
                          "cum_mean": cum, "cur_mean": cur, "old_mean": old}), flush=True)

    json.dump({"arming": "recluster", "split_prefix": args.split_prefix,
               "continual": results, "test_accessed": False,
               **protocol_v3.config_dict(dataset="arxiv", rate=args.rate,
                                         student_batch=args.student_batch,
                                         student_steps_override=args.student_steps,
                                         student_seeds=args.student_seeds)},
              open(out / "results.json", "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
