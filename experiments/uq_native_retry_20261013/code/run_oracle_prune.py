"""Oracle-cluster-pruning diagnostic to close direction F/G.

Question: is there ANY per-cluster signal by which to adaptively shrink old
memory (F) or selectively forget clusters (G)? Test the ceiling with an
oracle using TRUE labels: if pruning by oracle cluster purity does not beat
random pruning, then no label-free feature can (G is closed, F has no signal
to allocate by). If oracle purity DOES help, a label-free proxy may exist.

Uses Cora static k=135 memory (stage-0, all 7 classes present), the frozen
protocol, and measures per-cluster label purity as the oracle signal.
Arms (each keeps 95 clusters, drops 40):
  pruned_best_purity  drop the 40 lowest-purity clusters (oracle-guided)
  pruned_worst_purity drop the 40 highest-purity clusters (anti-oracle)
  pruned_random      drop 40 random clusters (3 draws)
  full_95            keep first 95 (control: budget effect only)
  no_shrink          full 135 (baseline)

Also reports per-cluster feature correlations with oracle purity to see which
cheap features (if any) track the oracle signal.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import protocol_v3

import sys
sys.path.insert(0, "/home/pc/research/TAGSAM/graphclip_native_work")
from run_gate1_continual import (
    FrozenMiniLM, CLIPStudent, norm_adj, train_student, zero_shot_eval,
    CLASS_NAMES, PROMPTS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/home/pc/research/GraphCLIP/processed_data/cora.pt")
    parser.add_argument("--minilm", default="/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip")
    parser.add_argument("--output", required=True)
    parser.add_argument("--rate", type=float, default=0.05)
    parser.add_argument("--keep", type=int, default=95)
    parser.add_argument("--student-seeds", type=int, nargs="+",
                        default=list(protocol_v3.STUDENT_SEEDS))
    parser.add_argument("--student-batch", type=int, default=protocol_v3.STUDENT.batch)
    parser.add_argument("--student-steps", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "results.json").exists():
        raise FileExistsError("existing runs are immutable")
    device = args.device

    data = torch.load(args.data, map_location="cpu", weights_only=False)
    x, y, texts = data.x, data.y, list(data.raw_texts)
    n = len(x)
    dev_mask = (torch.arange(n) % 10) == 1
    k = math.floor(args.rate * n)
    encoder = FrozenMiniLM(args.minilm, device)

    # ---- build the SAME deterministic cluster assignment as the method ----
    text_emb = encoder.encode(texts)
    joint = torch.cat((F.normalize(x.float(), dim=1), text_emb), dim=1)
    assign = None
    pts = joint
    centers = [0]
    d = torch.cdist(pts, pts[0:1]).flatten().square()
    for _ in range(1, k):
        d[torch.tensor(centers)] = -1.0
        nxt = int(torch.argmax(d))
        centers.append(nxt)
        d = torch.minimum(d, torch.cdist(pts, pts[nxt:nxt + 1]).flatten().square())
    centers_idx = torch.tensor(centers)
    c = pts[centers_idx].clone()
    for _ in range(25):
        a2 = torch.cdist(pts, c).argmin(1)
        a2[centers_idx] = torch.arange(k)
        newc = torch.stack([pts[a2 == j].mean(0) for j in range(k)])
        if torch.allclose(newc, c, atol=1e-6, rtol=0):
            break
        c = newc
    a2 = torch.cdist(pts, c).argmin(1)
    a2[centers_idx] = torch.arange(k)
    assign = a2

    # ---- build full artifact, get cluster means / purity / features ----
    syn_x, fused, weights = [], [], []
    cluster_purity = []
    cluster_feat = {"mass": [], "radius": [], "coherence": []}
    joint_g = joint.to(device)
    for j in range(k):
        members = torch.where(assign == j)[0]
        syn_x.append(x[members].float().mean(0))
        order = members[torch.argsort(
            torch.cdist(joint_g[members.to(device)],
                        joint_g[members.to(device)].mean(0, keepdim=True)).flatten())][:5]
        per = max(48, 320 // max(1, len(order)))
        fused.append(" ".join(texts[int(i)][:per] for i in order))
        weights.append(float(len(members)))
        # features
        cluster_purity.append(
            float(torch.bincount(y[members]).max() / len(members)))
        r = torch.cdist(joint_g[members.to(device)],
                        joint_g[members.to(device)].mean(0, keepdim=True)).flatten()
        cluster_feat["radius"].append(float(r.max()))
        cluster_feat["mass"].append(float(len(members)))
    syn_x = torch.stack(syn_x)
    fused_emb = encoder.encode(fused)
    cluster_purity = torch.tensor(cluster_purity)
    # coherence: distance of cluster mean X to its members' X mean (proxy)
    cluster_feat["coherence"] = [float(torch.norm(x[members].mean(0) - x[members].mean(0)))
                                 for members in [torch.where(assign == j)[0] for j in range(k)]]

    # correlation of cheap features with oracle purity
    print("feature vs oracle-purity Spearman:", flush=True)
    from scipy.stats import spearmanr
    for fname, vals in cluster_feat.items():
        sp = spearmanr(vals, cluster_purity.tolist())
        print(f"  {fname:10s} rho={sp.statistic:+.3f} p={sp.pvalue:.3f}", flush=True)

    # ---- train the no_shrink baseline (full 135) ----
    full_syn = {"x": syn_x, "text_emb": fused_emb, "texts": fused,
                "edge_index": make_edges(fused_emb, 1.0), "weights": torch.tensor(weights)}
    prompt_emb = encoder.encode(PROMPTS)
    real_adj = norm_adj(data.edge_index, n, device)
    xg = x.float().to(device)

    def run_arm(name, mask):
        s = {kk: (vv[mask] if torch.is_tensor(vv) else [vv[i] for i in mask.tolist()])
             for kk, vv in full_syn.items()}
        s["edge_index"] = make_edges(s["text_emb"], 1.0)
        s["x"] = s["x"].clone()
        s["text_emb"] = s["text_emb"].clone()
        scores = []
        for seed in args.student_seeds:
            steps = protocol_v3.student_steps(len(s["x"]), override=args.student_steps)
            model = train_student(s, seed, device, steps, batch=args.student_batch)
            scores.append(zero_shot_eval(model, xg, real_adj, prompt_emb, y, dev_mask))
        f1s = [sc["macro_f1"] for sc in scores]
        row = {"arm": name, "roots": len(s["x"]),
               "dev_macro_f1_mean": float(np.mean(f1s)),
               "dev_macro_f1_std": float(np.std(f1s))}
        print(json.dumps(row), flush=True)
        return row

    def make_edges(emb, degree):
        kk = len(emb)
        pairs = torch.triu_indices(kk, kk, offset=1)
        sc = (emb[pairs[0]] * emb[pairs[1]]).sum(1)
        cnt = min(int(kk * degree), pairs.shape[1])
        ch = pairs[:, torch.topk(sc, cnt).indices]
        return torch.cat((ch, ch.flip(0)), 1)

    results = []
    results.append(run_arm("no_shrink_k135", torch.arange(k)))

    keep = args.keep
    drop = k - keep
    # oracle purity: keep the `keep` highest-purity clusters
    order_by_purity = torch.argsort(cluster_purity, descending=True)
    best_mask = torch.zeros(k, dtype=torch.bool)
    best_mask[order_by_purity[:keep]] = True
    results.append(run_arm("pruned_best_purity", best_mask))
    worst_mask = torch.zeros(k, dtype=torch.bool)
    worst_mask[order_by_purity[-keep:]] = True
    results.append(run_arm("pruned_worst_purity", worst_mask))
    for dr in range(3):
        gen = torch.Generator().manual_seed(500 + dr)
        rmask = torch.zeros(k, dtype=torch.bool)
        rmask[torch.randperm(k, generator=gen)[:keep]] = True
        results.append(run_arm(f"pruned_random_{dr}", rmask))

    json.dump({**protocol_v3.config_dict(dataset="cora", rate=args.rate,
               student_batch=args.student_batch,
               student_steps=(args.student_steps if args.student_steps is not None else
                              protocol_v3.student_steps(k)),
               student_seeds=args.student_seeds),
               "results": results, "cluster_purity": cluster_purity.tolist(),
               "features": cluster_feat, "k": k, "keep": keep,
               "test_accessed": False,
               "note": "purity used as ORACLE diagnostic only; never usable legally"},
              open(out / "results.json", "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
