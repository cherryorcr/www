"""Direction E: ogbn-arxiv (169k nodes, 40 classes) scalability gate.

Static condensation gate at rate in {1%, 5%} with the frozen_v1 constructor,
plus wall-clock breakdown of every phase (encode / cluster / fuse / topology /
student train) to decide whether larger datasets are feasible.

Data: /home/pc/research/TAGSAM/data/ogbn_arxiv_full_minilm_semantic.pt
  node_f (169343,384 minilm), edge_index, text_list, label_list (40 classes),
  labels_desc (semantic names), node_year.
Splits: node_id%10==1 dev, %10==2 test (untouched), rest = condensation pool.
Prompts: "a computer science paper about {semantic label}".

Scalability adaptations (documented, minimal):
  - maxmin seeding on GPU (cdist per step on 169k x 768 is fine on 4090)
  - Lloyd on GPU
  - kNN topology: k up to 8467 rows at 5% -> pairwise is 8467^2 = 72M pairs,
    chunked on GPU
  - student: full-graph GCN on 169k nodes with sparse adj (fits easily)
Arms: constructor vs random-real (3 draws) vs class-balanced random.
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
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModel, AutoTokenizer


class FrozenMiniLM:
    def __init__(self, path, device):
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModel.from_pretrained(path).to(device).eval()
        self.device = device

    @torch.no_grad()
    def encode(self, texts, batch_size=128, max_length=256):
        parts = []
        for s in range(0, len(texts), batch_size):
            b = self.tokenizer(texts[s:s + batch_size], padding=True, truncation=True,
                               max_length=max_length, return_tensors="pt").to(self.device)
            h = self.model(**b).last_hidden_state
            m = b["attention_mask"].unsqueeze(-1).to(h.dtype)
            parts.append(F.normalize(((h * m).sum(1) / m.sum(1).clamp_min(1e-9)).float(), dim=1).cpu())
        return torch.cat(parts)


class GCNLayer(torch.nn.Module):
    def __init__(self, din, dout):
        super().__init__()
        self.lin = torch.nn.Linear(din, dout)

    def forward(self, x, adj):
        return adj @ self.lin(x)


class CLIPStudent(torch.nn.Module):
    def __init__(self, in_dim, hidden=384, out=384, text_dim=384):
        super().__init__()
        self.g1, self.g2 = GCNLayer(in_dim, hidden), GCNLayer(hidden, out)
        self.text_proj = torch.nn.Linear(text_dim, out)
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * math.log(1 / 0.07))

    def encode_graph(self, x, adj):
        return F.normalize(self.g2(F.relu(self.g1(x, adj)), adj), dim=1)

    def encode_text(self, t):
        return F.normalize(self.text_proj(t), dim=1)

    def logits(self, x, adj, t):
        return self.logit_scale.exp().clamp(max=100) * (
            self.encode_graph(x, adj) @ self.encode_text(t).T)


def norm_adj(edge_index, n, device):
    idx = torch.cat((edge_index, torch.arange(n).repeat(2, 1)), 1).to(device)
    val = torch.ones(idx.shape[1], device=device)
    deg = torch.zeros(n, device=device).scatter_add_(0, idx[0], val)
    dinv = deg.clamp_min(1e-9).pow(-0.5)
    return torch.sparse_coo_tensor(idx, dinv[idx[0]] * val * dinv[idx[1]], (n, n)).coalesce()


def maxmin_lloyd_gpu(points_cpu, k, device, iters=15, chunk=20000):
    """GPU maxmin + Lloyd for large n. points: (n,d) cpu float."""
    pts = points_cpu.to(device)
    n = len(pts)
    centers = [0]
    d = torch.cdist(pts, pts[0:1]).flatten().square()
    for _ in range(1, k):
        d[torch.tensor(centers, device=device)] = -1.0
        nxt = int(torch.argmax(d))
        centers.append(nxt)
        d = torch.minimum(d, torch.cdist(pts, pts[nxt:nxt + 1]).flatten().square())
    centers_idx = torch.tensor(centers, device=device)
    c = pts[centers_idx].clone()
    for _ in range(iters):
        assigns = []
        for s in range(0, n, chunk):
            assigns.append(torch.cdist(pts[s:s + chunk], c).argmin(1))
        assign = torch.cat(assigns)
        assign[centers_idx] = torch.arange(k, device=device)
        newc = torch.zeros_like(c)
        cnt = torch.zeros(k, device=device)
        newc.index_add_(0, assign, pts)
        cnt.index_add_(0, assign, torch.ones(n, device=device))
        newc = newc / cnt.clamp_min(1).unsqueeze(1)
        if torch.allclose(newc, c, atol=1e-5, rtol=0):
            c = newc
            break
        c = newc
    assigns = []
    for s in range(0, n, chunk):
        assigns.append(torch.cdist(pts[s:s + chunk], c).argmin(1))
    assign = torch.cat(assigns)
    assign[centers_idx] = torch.arange(k, device=device)
    return assign.cpu()


def knn_topology_gpu(emb_cpu, degree, device, chunk=2000):
    """global top-E symmetric text-kNN for large k, chunked."""
    e = emb_cpu.to(device)
    k = len(e)
    count = int(k * degree)
    # collect top candidates per row then global top-E
    scores, rows, cols = [], [], []
    per_row = max(4, int(degree * 3))
    for s in range(0, k, chunk):
        sim = e[s:s + chunk] @ e.T
        for local in range(sim.shape[0]):
            sim[local, s + local] = -2.0
        top_v, top_i = sim.topk(per_row, dim=1)
        r = torch.arange(s, s + sim.shape[0], device=device).unsqueeze(1).expand_as(top_i)
        scores.append(top_v.flatten())
        rows.append(r.flatten())
        cols.append(top_i.flatten())
    scores = torch.cat(scores)
    rows = torch.cat(rows)
    cols = torch.cat(cols)
    # dedupe unordered pairs
    a = torch.minimum(rows, cols)
    b = torch.maximum(rows, cols)
    key = a * k + b
    uniq, inv = torch.unique(key, return_inverse=True)
    best = torch.full((len(uniq),), -2.0, device=device)
    best.scatter_reduce_(0, inv, scores, reduce="amax")
    order = best.argsort(descending=True)[:count]
    ua = (uniq[order] // k)
    ub = (uniq[order] % k)
    edges = torch.stack([torch.cat([ua, ub]), torch.cat([ub, ua])])
    return edges.cpu()


def build_artifact(x, joint, texts, assign, k, encoder, device, degree=1.0, timings=None):
    t0 = time.time()
    syn_x, fused, weights = [], [], []
    joint_gpu = joint.to(device)
    for j in range(k):
        members = torch.where(assign == j)[0]
        if len(members) == 0:
            continue
        syn_x.append(x[members].float().mean(0))
        mg = joint_gpu[members.to(device)]
        d = torch.cdist(mg, mg.mean(0, keepdim=True)).flatten()
        order = members[d.argsort().cpu()][:5]
        per = max(48, 320 // max(1, len(order)))
        fused.append(" ".join(texts[int(i)][:per] for i in order))
        weights.append(float(len(members)))
    syn_x = torch.stack(syn_x)
    if timings is not None:
        timings["fuse_select"] = time.time() - t0
    t0 = time.time()
    fused_emb = encoder.encode(fused)
    if timings is not None:
        timings["fuse_encode"] = time.time() - t0
    t0 = time.time()
    edges = knn_topology_gpu(fused_emb, degree, device)
    if timings is not None:
        timings["topology"] = time.time() - t0
    return {"x": syn_x, "text_emb": fused_emb, "texts": fused,
            "edge_index": edges, "weights": torch.tensor(weights)}


def train_student(syn, seed, device, steps, lr=1e-3, batch=64):
    torch.manual_seed(seed)
    model = CLIPStudent(syn["x"].shape[1]).to(device)
    x, t = syn["x"].to(device), syn["text_emb"].to(device)
    adj = norm_adj(syn["edge_index"], len(x), device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed * 7 + 1)
    n = len(x)
    for _ in range(steps):
        rows = torch.randperm(n, generator=gen)[:min(batch, n)].to(device)
        g = model.encode_graph(x, adj)[rows]
        tt = model.encode_text(t[rows])
        logits = model.logit_scale.exp().clamp(max=100) * g @ tt.T
        target = torch.arange(len(rows), device=device)
        loss = (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target)) / 2
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return model.eval()


@torch.no_grad()
def zero_shot_eval(model, x, adj, prompt_emb, y, mask):
    pred = model.logits(x, adj, prompt_emb.to(x.device)).argmax(1).cpu()
    yy, pp = y[mask].numpy(), pred[mask].numpy()
    return {"macro_f1": 100 * f1_score(yy, pp, average="macro", zero_division=0),
            "accuracy": 100 * accuracy_score(yy, pp)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/home/pc/research/TAGSAM/data/ogbn_arxiv_full_minilm_semantic.pt")
    parser.add_argument("--minilm", default="/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rates", type=float, nargs="+", default=[0.01, 0.05])
    parser.add_argument("--student-seeds", type=int, nargs="+", default=[100, 101, 102])
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "results.json").exists():
        raise FileExistsError("existing runs are immutable")

    device = args.device
    t_total = time.time()
    timings = {}

    t0 = time.time()
    d = torch.load(args.data, map_location="cpu", weights_only=False)
    x = d["node_f"]
    texts = list(d["text_list"])
    # label_list contains string labels like 'arxiv cs cr'; map to indices via all_labels
    all_labels = [str(v) for v in np.asarray(d["all_labels"]).tolist()]
    lab2idx = {name: i for i, name in enumerate(all_labels)}
    y = torch.tensor([lab2idx[str(v)] for v in np.asarray(d["label_list"]).tolist()],
                     dtype=torch.long)
    labels_desc = list(d["labels_desc"])
    n = len(x)
    timings["load"] = time.time() - t0
    print(f"loaded: n={n}, classes={y.max().item()+1}, load={timings['load']:.0f}s", flush=True)

    dev_mask = (torch.arange(n) % 10) == 1
    pool_mask = ~dev_mask & ((torch.arange(n) % 10) != 2)  # exclude dev & sealed test
    pool = torch.where(pool_mask)[0]

    encoder = FrozenMiniLM(args.minilm, device)
    # semantic prompts
    def clean(name):
        return name.replace("arXiv computer science category: ", "").replace(
            "arXiv computer science category ", "")
    prompts = [f"a computer science paper about {clean(nm).lower()}" for nm in labels_desc]
    prompt_emb = encoder.encode(prompts)

    t0 = time.time()
    text_emb = encoder.encode(texts)   # 169k texts
    timings["encode_all_texts"] = time.time() - t0
    print(f"encoded {n} texts in {timings['encode_all_texts']:.0f}s", flush=True)

    t0 = time.time()
    real_adj = norm_adj(d["edge_index"], n, device)
    xg = x.float().to(device)
    timings["real_graph_setup"] = time.time() - t0

    joint = torch.cat((F.normalize(x.float(), dim=1), text_emb), dim=1)
    joint_pool = joint[pool]
    x_pool = x[pool]
    texts_pool = [texts[int(i)] for i in pool]
    y_pool = y[pool]

    results = []
    for rate in args.rates:
        k = math.floor(rate * len(pool))
        steps = min(2048, 16 * k)
        print(f"\n===== rate={rate} k={k} steps={steps} =====", flush=True)
        arm_t = {}

        # ---- constructor ----
        t0 = time.time()
        assign = maxmin_lloyd_gpu(joint_pool, k, device)
        arm_t["cluster"] = time.time() - t0
        print(f"  cluster: {arm_t['cluster']:.0f}s", flush=True)
        syn = build_artifact(x_pool, joint_pool, texts_pool, assign, k, encoder,
                             device, timings=arm_t)
        print(f"  fuse_select={arm_t['fuse_select']:.0f}s fuse_encode={arm_t['fuse_encode']:.0f}s "
              f"topology={arm_t['topology']:.0f}s", flush=True)
        torch.save({key: syn[key] for key in syn}, out / f"syn_construct_r{rate}.pt")

        arms = {"construct": syn}

        # ---- random real (single draw x3 handled as separate arms) ----
        for draw in range(2):
            gen = torch.Generator().manual_seed(1000 + draw)
            idx = pool[torch.randperm(len(pool), generator=gen)[:k]]
            remap = torch.full((n,), -1, dtype=torch.long)
            remap[idx] = torch.arange(k)
            e = d["edge_index"]
            keep = (remap[e[0]] >= 0) & (remap[e[1]] >= 0)
            arms[f"random_real_{draw}"] = {
                "x": x[idx].float(), "text_emb": text_emb[idx],
                "texts": [texts[int(i)] for i in idx],
                "edge_index": torch.stack((remap[e[0, keep]], remap[e[1, keep]])),
                "weights": torch.ones(k)}

        for arm, syn_a in arms.items():
            t0 = time.time()
            scores = []
            for seed in args.student_seeds:
                model = train_student(syn_a, seed, device, steps)
                scores.append(zero_shot_eval(model, xg, real_adj, prompt_emb, y, dev_mask))
                del model
            train_s = time.time() - t0
            f1s = [s["macro_f1"] for s in scores]
            row = {"rate": rate, "arm": arm, "k": k, "steps": steps,
                   "dev_macro_f1_mean": float(np.mean(f1s)),
                   "dev_macro_f1_std": float(np.std(f1s)),
                   "dev_accuracy_mean": float(np.mean([s["accuracy"] for s in scores])),
                   "train_eval_seconds": train_s,
                   "condense_timings": arm_t if arm == "construct" else None,
                   "per_student": scores}
            results.append(row)
            print(json.dumps({kk: row[kk] for kk in ("rate", "arm", "dev_macro_f1_mean",
                                                     "dev_macro_f1_std", "train_eval_seconds")}), flush=True)

    timings["total"] = time.time() - t_total
    json.dump({"results": results, "global_timings": timings, "n_nodes": n,
               "dev_rule": "node_id%10==1", "test_rule": "node_id%10==2 SEALED",
               "test_accessed": False},
              open(out / "results.json", "w"), indent=1)
    print(f"\nDONE total={timings['total']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
