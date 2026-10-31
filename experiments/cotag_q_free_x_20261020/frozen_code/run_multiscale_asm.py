"""Multi-scale ASM probe: does the teacher-response proxy become usable at scale?

TAGSAM authors suggest our "ASM fails" finding is a small-data artifact.
Directly test it: run the SAME ASM bilevel optimization at three scales and
measure (a) whether the outer loss actually decreases, (b) whether the
optimized artifact beats frozen initialization under independent students.

Scales:
  cora      2,708 nodes    k = 5%  pool
  citeseer  3,312 nodes    k = 5%  pool
  arxiv     169,343 nodes  k = 1% and 5% pool  (GPU-adapted ASM)

ASM core (from frozen inference ASM, simplified, GPU-adapted):
  outer: for step in 1..T
    seed a fresh student; train inner_steps on the synthetic graph (inner)
    evaluate student response on a reference set of REAL nodes
    match student's graph-text similarity matrix to the teacher's
      (row/col softmax BCE)  -> gradient into synthetic X   (outer)
  Compare asm_loss at step 1 vs step T, and dev F1:
    frozen       = the initial synthetic artifact (no ASM)
    asm_best     = best-by-dev ASM snapshot (with a small held-out selection)
    asm_last     = last ASM snapshot

Teacher: a CLIP student trained on the FULL legal pool (real graph), frozen.
Reference set: a fixed random sample of real pool nodes (128 Cora/Citeseer,
1024 arxiv).

If asm_loss is flat and asm_best <= frozen at ALL scales -> proxy stays
unusable with scale (response to TAGSAM authors). If loss drops and ASM wins
at arxiv -> our claim narrows to small-data and we say so.
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

import protocol_v3

CLASS_NAMES_CORA = ["case based", "genetic algorithms", "neural networks",
                    "probabilistic methods", "reinforcement learning",
                    "rule learning", "theory"]
CLASS_NAMES_CITESEER = ["agents", "machine learning", "information retrieval",
                        "databases", "human computer interaction",
                        "artificial intelligence"]
DATA_CFG = {
    "cora": {
        "path": "/home/pc/research/GraphCLIP/processed_data/cora.pt",
        "kind": "graphclip", "names": CLASS_NAMES_CORA},
    "citeseer": {
        "path": "/home/pc/research/GraphCLIP/processed_data/citeseer.pt",
        "kind": "graphclip", "names": CLASS_NAMES_CITESEER},
    "arxiv": {
        "path": "/home/pc/research/TAGSAM/data/ogbn_arxiv_full_minilm_semantic.pt",
        "kind": "arxiv", "names": None},  # prompts from labels_desc
    "wikics": {
        "path": "/home/pc/research/TAGSAM/data/tagbench/wikics.pt",
        "kind": "tagbench", "names": None},
    "bookhis": {
        "path": "/home/pc/research/TAGSAM/data/tagbench/bookhis.pt",
        "kind": "tagbench", "names": None},
    "bookchild": {
        "path": "/home/pc/research/TAGSAM/data/tagbench/bookchild.pt",
        "kind": "tagbench", "names": None},
    "sportsfit": {
        "path": "/home/pc/research/TAGSAM/data/tagbench/sportsfit.pt",
        "kind": "tagbench", "names": None},
}


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

    def response(self, x, adj, t):
        """graph-text similarity matrix on given rows (temperature-scaled)."""
        return self.encode_graph(x, adj) @ self.encode_text(t).T / 0.07

    def logits(self, x, adj, t):
        """zero-shot class-prompt logits (temperature-scaled)."""
        return self.logit_scale.exp().clamp(max=100) * (
            self.encode_graph(x, adj) @ self.encode_text(t).T)


def norm_adj(edge_index, n, device):
    idx = torch.cat((edge_index, torch.arange(n).repeat(2, 1)), 1).to(device)
    val = torch.ones(idx.shape[1], device=device)
    deg = torch.zeros(n, device=device).scatter_add_(0, idx[0], val)
    dinv = deg.clamp_min(1e-9).pow(-0.5)
    return torch.sparse_coo_tensor(idx, dinv[idx[0]] * val * dinv[idx[1]], (n, n)).coalesce()


def maxmin_lloyd_gpu(points_cpu, k, device, iters=15, chunk=20000):
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
    e = emb_cpu.to(device)
    k = len(e)
    count = int(k * degree)
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
    scores, rows, cols = torch.cat(scores), torch.cat(rows), torch.cat(cols)
    a = torch.minimum(rows, cols)
    b = torch.maximum(rows, cols)
    key = a * k + b
    uniq, inv = torch.unique(key, return_inverse=True)
    best = torch.full((len(uniq),), -2.0, device=device)
    best.scatter_reduce_(0, inv, scores, reduce="amax")
    order = best.argsort(descending=True)[:count]
    ua, ub = uniq[order] // k, uniq[order] % k
    return torch.stack([torch.cat([ua, ub]), torch.cat([ub, ua])]).cpu()


def build_initial(x, text_emb, texts, k, encoder, device,
                  degree=protocol_v3.DEGREE,
                  fuse_g=protocol_v3.FUSE_G,
                  fuse_chars=protocol_v3.FUSE_CHARS):
    joint = torch.cat((F.normalize(x.float(), dim=1), text_emb), dim=1)
    assign = maxmin_lloyd_gpu(joint, k, device)
    joint_gpu = joint.to(device)
    syn_x, fused, weights = [], [], []
    for j in range(k):
        members = torch.where(assign == j)[0]
        if len(members) == 0:
            continue
        syn_x.append(x[members].float().mean(0))
        mg = joint_gpu[members.to(device)]
        d = torch.cdist(mg, mg.mean(0, keepdim=True)).flatten()
        order = members[d.argsort().cpu()][:fuse_g]
        per = max(48, fuse_chars // max(1, len(order)))
        fused.append(" ".join(texts[int(i)][:per] for i in order))
        weights.append(float(len(members)))
    syn_x = torch.stack(syn_x)
    fused_emb = encoder.encode(fused)
    edges = knn_topology_gpu(fused_emb, degree, device)
    return {"x": syn_x, "text_emb": fused_emb, "texts": fused,
            "edge_index": edges, "weights": torch.tensor(weights)}


def train_student(syn, seed, device, steps, lr=protocol_v3.STUDENT.lr,
                  batch=protocol_v3.STUDENT.batch):
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


# ------------------------------- ASM (bilevel) ------------------------------

def asm_optimize(initial, teacher, source_x, source_emb, source_texts,
                 ref_rows_local, device, source_adj=None,
                 steps=protocol_v3.ASM_STEPS,
                 inner=protocol_v3.ASM_INNER_STEPS,
                 batch=protocol_v3.ASM_BATCH,
                 outer_lr=protocol_v3.ASM_OUTER_LR, inner_lr=5e-3, eval_every=20,
                 dev_eval=None):
    """Optimize synthetic X by matching teacher response on reference rows.

    reference rows are indices into the SOURCE (real pool). Both teacher and
    differentiable student responses are evaluated on that same induced real
    graph; the synthetic graph is used only for the inner training updates.
    teacher is a frozen CLIPStudent trained on the full real pool.
    """
    k = len(initial["x"])
    syn_x = initial["x"].clone().to(device)
    text_emb = initial["text_emb"].to(device)
    edges = initial["edge_index"]
    adj = norm_adj(edges, k, device)

    # Reference rows index the real source pool.  Keeping the reference graph
    # separate from the synthetic graph makes the teacher target meaningful.
    ref = ref_rows_local.to(device)
    source_x = source_x.float().to(device)
    source_emb = source_emb.to(device)
    if source_adj is None:
        raise ValueError("ASM requires the induced real reference adjacency")

    # teacher response on the reference rows (teacher sees its own real graph)
    teacher.eval()
    with torch.no_grad():
        teacher_logits = teacher.response(source_x, source_adj, source_emb)[ref][:, ref]

    var = torch.nn.Parameter(syn_x.clone())
    opt = torch.optim.Adam([var], lr=outer_lr)
    curve = []
    for step in range(1, steps + 1):
        torch.manual_seed(1000 * step + 7)
        student = CLIPStudent(syn_x.shape[1]).to(device)
        params = dict(student.named_parameters())
        gen = torch.Generator().manual_seed(step * 13 + 1)
        rows_all = torch.randperm(k, generator=gen)
        # inner: train student on synthetic graph (functional)
        inner_params = {n: p for n, p in params.items()}
        for it in range(inner):
            ridx = rows_all[it * batch:(it + 1) * batch]
            if len(ridx) < 2:
                continue
            adj_s = adj
            g = _func_graph(inner_params, var, adj_s)
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
        # outer: student response on reference rows vs teacher
        g = _func_graph(inner_params, source_x, source_adj)
        tt = _func_text(inner_params, source_emb)
        student_logits = (g[ref] @ tt[ref].T) / 0.07
        asm = _asm_loss(student_logits, teacher_logits)
        opt.zero_grad(set_to_none=True)
        asm.backward()
        opt.step()
        with torch.no_grad():
            var.clamp_(-10, 10)
        curve.append(float(asm.detach()))
        if step % eval_every == 0 or step == 1 or step == steps:
            tag = f"step{step}"
            if dev_eval is not None:
                m = dev_eval(var.detach(), adj)
                print(f"    ASM {tag}: asm_loss={curve[-1]:.4f} dev_f1={m:.2f}", flush=True)
            else:
                print(f"    ASM {tag}: asm_loss={curve[-1]:.4f}", flush=True)

    # build artifact from optimized X
    out = {kk: (initial[kk] if not torch.is_tensor(initial[kk]) else initial[kk])
           for kk in initial}
    out["x"] = var.detach().cpu()
    return out, curve


def _func_graph(params, x, adj):
    h = adj @ F.linear(x, params["g1.lin.weight"], params["g1.lin.bias"])
    h = F.relu(h)
    h = adj @ F.linear(h, params["g2.lin.weight"], params["g2.lin.bias"])
    return F.normalize(h, dim=1)


def _func_text(params, t):
    return F.normalize(F.linear(t, params["text_proj.weight"], params["text_proj.bias"]), dim=1)


def _asm_loss(s, t):
    t = t.detach()
    row = F.binary_cross_entropy(s.softmax(1), t.softmax(1))
    col = F.binary_cross_entropy(s.softmax(0), t.softmax(0))
    return (row + col) / 2


# ---------------------------------- driver ----------------------------------

def load_dataset(name, minilm, device):
    cfg = DATA_CFG[name]
    d = torch.load(cfg["path"], map_location="cpu", weights_only=False)
    if cfg["kind"] == "graphclip":
        x = d.x
        texts = list(d.raw_texts)
        y = d.y
        edge_index = d.edge_index
        n = len(x)
        dev_mask, _, pool_mask = protocol_v3.split_masks(torch.arange(n))
        names = cfg["names"]
        prompts = [f"a research paper about {nm}" for nm in names]
    else:  # arxiv or tagbench dictionary schema
        x = d["node_f"]
        texts = list(d["text_list"])
        all_labels = [str(v) for v in np.asarray(d["all_labels"]).tolist()]
        lab2idx = {nm: i for i, nm in enumerate(all_labels)}
        y = torch.tensor([lab2idx[str(v)] for v in np.asarray(d["label_list"]).tolist()],
                         dtype=torch.long)
        edge_index = d["edge_index"]
        n = len(x)
        dev_mask, _, pool_mask = protocol_v3.split_masks(torch.arange(n))
        desc = list(d.get("labels_desc", all_labels))
        if cfg["kind"] == "arxiv":
            clean = lambda s: s.replace("arXiv computer science category: ", "").replace(
                "arXiv computer science category ", "")
            prompts = [f"a computer science paper about {clean(nm).lower()}" for nm in desc]
        else:
            prompts = [f"a document about {nm}" for nm in desc]
    encoder = FrozenMiniLM(minilm, device)
    return dict(x=x, texts=texts, y=y, edge_index=edge_index, n=n,
                dev_mask=dev_mask, pool_mask=pool_mask, prompts=prompts, encoder=encoder)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(DATA_CFG), required=True)
    parser.add_argument("--rate", type=float, default=0.05)
    parser.add_argument("--minilm", default="/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--asm-steps", type=int, default=protocol_v3.ASM_STEPS)
    parser.add_argument("--inner", type=int, default=protocol_v3.ASM_INNER_STEPS)
    parser.add_argument("--outer-lr", type=float, default=protocol_v3.ASM_OUTER_LR)
    parser.add_argument("--ref-size", type=int, default=128)
    parser.add_argument("--student-seeds", type=int, nargs="+",
                        default=list(protocol_v3.STUDENT_SEEDS))
    parser.add_argument("--student-batch", type=int,
                        default=protocol_v3.STUDENT.batch)
    parser.add_argument("--student-steps", type=int, default=None,
                        help="fixed evaluation steps; otherwise protocol-scaled")
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "results.json").exists():
        raise FileExistsError("existing runs are immutable")
    device = args.device
    D = load_dataset(args.dataset, args.minilm, device)
    x, texts, y = D["x"], D["texts"], D["y"]
    n = D["n"]
    pool = torch.where(D["pool_mask"])[0]
    k = math.floor(args.rate * len(pool))
    encoder = D["encoder"]
    prompt_emb = encoder.encode(D["prompts"])
    real_adj = norm_adj(D["edge_index"], n, device)
    xg = x.float().to(device)
    print(f"[{args.dataset}] n={n} pool={len(pool)} k={k}", flush=True)

    # text embeddings for the pool
    text_emb_pool = encoder.encode([texts[int(i)] for i in pool])

    # initial synthetic artifact (frozen constructor)
    t0 = time.time()
    initial = build_initial(x[pool], text_emb_pool, [texts[int(i)] for i in pool],
                            k, encoder, device)
    print(f"  constructor: {time.time()-t0:.0f}s", flush=True)

    # teacher: CLIP student trained on the FULL real pool graph
    pool_remap = torch.full((n,), -1, dtype=torch.long)
    pool_remap[pool] = torch.arange(len(pool))
    e = D["edge_index"]
    keep = (pool_remap[e[0]] >= 0) & (pool_remap[e[1]] >= 0)
    real_pool_syn = {"x": x[pool].float(), "text_emb": text_emb_pool,
                     "texts": [texts[int(i)] for i in pool],
                     "edge_index": torch.stack((pool_remap[e[0, keep]], pool_remap[e[1, keep]])),
                     "weights": torch.ones(len(pool))}
    t0 = time.time()
    teacher = train_student(real_pool_syn, protocol_v3.TEACHER_SEED, device,
                            protocol_v3.TEACHER_STEPS, batch=args.student_batch)
    print(f"  teacher trained: {time.time()-t0:.0f}s", flush=True)

    # dev evaluator: retrain student on a candidate synthetic artifact
    def dev_eval(cand_x, adj):
        tmp = {kk: initial[kk] for kk in initial}
        tmp = {"x": cand_x.cpu(), "text_emb": initial["text_emb"],
               "texts": initial["texts"], "edge_index": initial["edge_index"],
               "weights": initial["weights"]}
        steps = (args.student_steps if args.student_steps is not None else
                 protocol_v3.STUDENT.steps(len(tmp["x"])))
        model = train_student(tmp, protocol_v3.METHOD_SEED, device, steps,
                              batch=args.student_batch)
        pred = model.logits(xg, real_adj, prompt_emb.to(device)).argmax(1).cpu()
        yy, pp = y[D["dev_mask"]].numpy(), pred[D["dev_mask"]].numpy()
        return 100 * f1_score(yy, pp, average="macro", zero_division=0)

    # frozen baseline
    t0 = time.time()
    frozen_scores = []
    for seed in args.student_seeds:
        steps = (args.student_steps if args.student_steps is not None else
                 protocol_v3.STUDENT.steps(k))
        model = train_student(initial, seed, device, steps,
                              batch=args.student_batch)
        pred = model.logits(xg, real_adj, prompt_emb.to(device)).argmax(1).cpu()
        yy, pp = y[D["dev_mask"]].numpy(), pred[D["dev_mask"]].numpy()
        frozen_scores.append(100 * f1_score(yy, pp, average="macro", zero_division=0))
        del model
    print(f"  frozen dev_f1 = {np.mean(frozen_scores):.2f} ± {np.std(frozen_scores):.2f} "
          f"({time.time()-t0:.0f}s)", flush=True)

    # ASM optimization
    pool_adj = norm_adj(real_pool_syn["edge_index"], len(pool), device)
    ref_size = min(args.ref_size, len(pool))
    gen = torch.Generator().manual_seed(protocol_v3.REFERENCE_SEED)
    ref_rows = torch.randperm(len(pool), generator=gen)[:ref_size]
    t0 = time.time()
    asm_art, curve = asm_optimize(initial, teacher, x[pool], text_emb_pool,
                                  [texts[int(i)] for i in pool], ref_rows, device,
                                  source_adj=pool_adj,
                                  steps=args.asm_steps, inner=args.inner,
                                  batch=args.student_batch,
                                  outer_lr=args.outer_lr, dev_eval=dev_eval)
    asm_seconds = time.time() - t0

    # evaluate ASM artifact
    asm_scores = []
    for seed in args.student_seeds:
        steps = (args.student_steps if args.student_steps is not None else
                 protocol_v3.STUDENT.steps(k))
        model = train_student(asm_art, seed, device, steps,
                              batch=args.student_batch)
        pred = model.logits(xg, real_adj, prompt_emb.to(device)).argmax(1).cpu()
        yy, pp = y[D["dev_mask"]].numpy(), pred[D["dev_mask"]].numpy()
        asm_scores.append(100 * f1_score(yy, pp, average="macro", zero_division=0))
        del model

    loss_drop_pct = 100 * (curve[0] - curve[-1]) / max(curve[0], 1e-9)
    result = {
        "dataset": args.dataset, "n": int(n), "pool": int(len(pool)), "k": int(k),
        "rate": args.rate,
        "frozen_dev_f1_mean": float(np.mean(frozen_scores)),
        "frozen_dev_f1_std": float(np.std(frozen_scores)),
        "asm_dev_f1_mean": float(np.mean(asm_scores)),
        "asm_dev_f1_std": float(np.std(asm_scores)),
        "asm_minus_frozen": float(np.mean(asm_scores) - np.mean(frozen_scores)),
        "asm_loss_first": curve[0], "asm_loss_last": curve[-1],
        "asm_loss_drop_pct": loss_drop_pct,
        "asm_seconds": asm_seconds,
        "frozen_scores": frozen_scores, "asm_scores": asm_scores,
        "curve": curve,
        "test_accessed": False,
        **protocol_v3.config_dict(dataset=args.dataset, rate=args.rate,
                                  student_batch=args.student_batch,
                                  student_steps=(args.student_steps if
                                                 args.student_steps is not None else
                                                 protocol_v3.STUDENT.steps(k)),
                                  student_seeds=args.student_seeds,
                                  asm_steps=args.asm_steps, inner=args.inner),
    }
    json.dump(result, open(out / "results.json", "w"), indent=1)
    print(json.dumps({kk: result[kk] for kk in (
        "dataset", "n", "k", "frozen_dev_f1_mean", "asm_dev_f1_mean",
        "asm_minus_frozen", "asm_loss_drop_pct", "asm_seconds")}), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
