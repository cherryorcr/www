"""Direction-B headroom gate: how much can BETTER CLUSTERING buy?

Before building learnable contrastive clustering (direction B), measure the
ceiling: if an oracle partition barely beats maxmin+Lloyd, B is capped and
not worth building; if the gap is large, B has real headroom.

Arms (artifact construction identical everywhere: mean synthetic X, top-5
truncated concat text, text-kNN deg=1.0 — only the PARTITION changes):
  baseline_maxmin     current frozen_v1 clustering (joint space)
  kmeans_multi        20-restart k-means++ best-inertia in joint space
                      (pure geometric quality, label-free)
  text_only_cluster   maxmin+Lloyd in text-embedding space only
                      (is the joint [X||text] space the right metric?)
  oracle_label        DIAGNOSTIC ONLY: partition by true labels first,
                      budget proportional to class size, maxmin+Lloyd within
                      each class. Upper bound for class-aligned clustering
                      (labels never used in any legal arm).

Protocol: Cora k=135, 6 students x 2048 steps, dev-only, node_id%10==1.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import protocol_v3
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModel, AutoTokenizer

CLASS_NAMES = ["case based", "genetic algorithms", "neural networks",
               "probabilistic methods", "reinforcement learning",
               "rule learning", "theory"]
PROMPTS = [f"a research paper about {n}" for n in CLASS_NAMES]


class FrozenMiniLM:
    def __init__(self, path, device):
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModel.from_pretrained(path).to(device).eval()
        self.device = device

    @torch.no_grad()
    def encode(self, texts, batch_size=64, max_length=256):
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


def maxmin_lloyd(points, k, iters=25):
    """Deterministic maxmin seeding + Lloyd. Returns assignment tensor."""
    n = len(points)
    centers = [0]
    dist = torch.cdist(points, points[0:1]).flatten().square()
    for _ in range(1, k):
        dist[torch.tensor(centers)] = -1.0
        nxt = int(torch.argmax(dist))
        centers.append(nxt)
        dist = torch.minimum(dist, torch.cdist(points, points[nxt:nxt + 1]).flatten().square())
    centers_idx = torch.tensor(centers)
    c = points[centers_idx].clone()
    for _ in range(iters):
        assign = torch.cdist(points, c).argmin(1)
        assign[centers_idx] = torch.arange(k)
        newc = torch.stack([points[assign == j].mean(0) if (assign == j).any() else c[j]
                            for j in range(k)])
        if torch.allclose(newc, c, atol=1e-6, rtol=0):
            break
        c = newc
    assign = torch.cdist(points, c).argmin(1)
    assign[centers_idx] = torch.arange(k)
    return assign


def kmeans_multi(points, k, restarts=20):
    """k-means++ with multiple restarts, keep best inertia. Seeded (reproducible)."""
    best_assign, best_inertia = None, float("inf")
    n = len(points)
    for r in range(restarts):
        gen = torch.Generator().manual_seed(1000 + r)
        # k-means++ seeding
        centers = [int(torch.randint(n, (1,), generator=gen))]
        dist = torch.cdist(points, points[centers[0]:centers[0] + 1]).flatten().square()
        for _ in range(1, k):
            dist_c = dist.clone()
            dist_c[torch.tensor(centers)] = 0
            if dist_c.sum() <= 0:
                cand = int(torch.randint(n, (1,), generator=gen))
            else:
                cand = int(torch.multinomial(dist_c, 1, generator=gen))
            centers.append(cand)
            dist = torch.minimum(dist, torch.cdist(points, points[cand:cand + 1]).flatten().square())
        c = points[torch.tensor(centers)].clone()
        for _ in range(25):
            assign = torch.cdist(points, c).argmin(1)
            newc = torch.stack([points[assign == j].mean(0) if (assign == j).any() else c[j]
                                for j in range(k)])
            if torch.allclose(newc, c, atol=1e-6, rtol=0):
                break
            c = newc
        assign = torch.cdist(points, c).argmin(1)
        inertia = float(((points - c[assign]) ** 2).sum())
        if inertia < best_inertia:
            best_inertia, best_assign = inertia, assign.clone()
    return best_assign


def oracle_label_partition(joint, y, k):
    """DIAGNOSTIC: stratify by true label, proportional budget, maxmin+Lloyd inside."""
    n = len(y)
    classes = sorted(set(int(v) for v in y.tolist()))
    counts = {c: int((y == c).sum()) for c in classes}
    quotas, rem = {}, k
    for i, c in enumerate(classes):
        q = max(1, round(k * counts[c] / n)) if i < len(classes) - 1 else rem
        q = min(q, counts[c], rem - (len(classes) - 1 - i))
        quotas[c] = q
        rem -= q
    assign = torch.full((n,), -1, dtype=torch.long)
    offset = 0
    for c in classes:
        rows = torch.where(y == c)[0]
        sub = maxmin_lloyd(joint[rows], quotas[c]) if quotas[c] > 1 else torch.zeros(len(rows), dtype=torch.long)
        assign[rows] = sub + offset
        offset += quotas[c]
    return assign, offset


def build_artifact(x, joint, texts, assign, k, encoder, degree=1.0):
    syn_x, fused, weights = [], [], []
    for j in range(k):
        members = torch.where(assign == j)[0]
        if len(members) == 0:
            continue
        syn_x.append(x[members].float().mean(0))
        centroid = joint[members].mean(0, keepdim=True)
        d = torch.cdist(joint[members], centroid).flatten()
        order = members[torch.argsort(d)][:5]
        per = max(48, 320 // max(1, len(order)))
        fused.append(" ".join(texts[int(i)][:per] for i in order))
        weights.append(float(len(members)))
    syn_x = torch.stack(syn_x)
    kk = len(syn_x)
    fused_emb = encoder.encode(fused)
    pairs = torch.triu_indices(kk, kk, offset=1)
    score = (fused_emb[pairs[0]] * fused_emb[pairs[1]]).sum(1)
    count = min(int(kk * degree), pairs.shape[1])
    chosen = pairs[:, torch.topk(score, count).indices]
    edges = torch.cat((chosen, chosen.flip(0)), 1)
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


@torch.no_grad()
def zero_shot_eval(model, x, adj, prompt_emb, y, mask):
    pred = model.logits(x, adj, prompt_emb.to(x.device)).argmax(1).cpu()
    yy, pp = y[mask].numpy(), pred[mask].numpy()
    return {"macro_f1": 100 * f1_score(yy, pp, average="macro", zero_division=0),
            "accuracy": 100 * accuracy_score(yy, pp)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/home/pc/research/GraphCLIP/processed_data/cora.pt")
    parser.add_argument("--minilm", default="/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rate", type=float, default=0.05)
    parser.add_argument("--student-seeds", type=int, nargs="+",
                        default=list(protocol_v3.STUDENT_SEEDS))
    parser.add_argument("--student-batch", type=int, default=protocol_v3.STUDENT.batch)
    parser.add_argument("--student-steps", type=int, default=None)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "results.json").exists():
        raise FileExistsError("existing runs are immutable")

    data = torch.load(args.data, map_location="cpu", weights_only=False)
    x, y, texts = data.x, data.y, list(data.raw_texts)
    n = len(x)
    dev_mask = (torch.arange(n) % 10) == 1
    k = math.floor(args.rate * n)
    device = args.device
    encoder = FrozenMiniLM(args.minilm, device)
    text_emb = encoder.encode(texts)
    prompt_emb = encoder.encode(PROMPTS)
    real_adj = norm_adj(data.edge_index, n, device)
    xg = x.float().to(device)
    joint = torch.cat((F.normalize(x.float(), dim=1), text_emb), dim=1)

    # build partitions
    partitions = {}
    print("building partitions...", flush=True)
    partitions["baseline_maxmin"] = (maxmin_lloyd(joint, k), k)
    partitions["kmeans_multi"] = (kmeans_multi(joint, k), k)
    partitions["text_only_cluster"] = (maxmin_lloyd(text_emb, k), k)
    oa, ok = oracle_label_partition(joint, y, k)
    partitions["oracle_label"] = (oa, ok)

    results = []
    for arm, (assign, kk) in partitions.items():
        print(f"\n=== {arm} (k={kk}) ===", flush=True)
        # cluster purity audit (diagnostic, uses labels for measurement only)
        purity = 0.0
        for j in range(kk):
            members = torch.where(assign == j)[0]
            if len(members):
                purity += int(torch.bincount(y[members]).max())
        purity = 100 * purity / n
        syn = build_artifact(x, joint, texts, assign, kk, encoder)
        torch.save({key: syn[key] for key in syn}, out / f"syn_{arm}.pt")
        scores = []
        steps = protocol_v3.student_steps(len(syn["x"]), override=args.student_steps)
        for seed in args.student_seeds:
            model = train_student(syn, seed, device, steps=steps,
                                  batch=args.student_batch)
            scores.append(zero_shot_eval(model, xg, real_adj, prompt_emb, y, dev_mask))
            del model
        f1s = [s["macro_f1"] for s in scores]
        row = {"arm": arm, "roots": len(syn["x"]),
               "student_steps": steps, "student_batch": args.student_batch,
               "cluster_purity": purity,
               "dev_macro_f1_mean": float(np.mean(f1s)),
               "dev_macro_f1_std": float(np.std(f1s)),
               "per_student": scores}
        results.append(row)
        print(json.dumps({key: row[key] for key in ("arm", "cluster_purity",
                                                    "dev_macro_f1_mean", "dev_macro_f1_std")}), flush=True)

    json.dump({**protocol_v3.config_dict(dataset="cora", rate=args.rate,
               student_batch=args.student_batch,
               student_steps=args.student_steps,
               student_seeds=args.student_seeds),
               "results": results, "k": k, "rate": args.rate,
               "dev_rule": "node_id % 10 == 1", "test_accessed": False,
               "note": "oracle_label uses labels: DIAGNOSTIC upper bound only"},
              open(out / "results.json", "w"), indent=1)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
