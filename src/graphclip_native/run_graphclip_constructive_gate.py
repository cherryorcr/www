"""GraphCLIP-native constructive static TAG condensation gate.

User constraints (2026-09-08):
1. downstream = GraphCLIP-style interface, simplified where the full stack is
   unavailable: frozen MiniLM (all-MiniLM-L6-v2, the 'tiny' text tower of the
   released GraphCLIP) as text encoder, a trainable GCN as graph encoder
   (replacing the GPS graph transformer), symmetric CLIP InfoNCE with learned
   logit scale, and zero-shot class-prompt node classification;
2. the condensed artifact must contain NO real node: synthetic X only
   (mass-weighted cluster centroids), no real-node X copies;
3. text must be FUSED, not selected: per cluster, the g member texts closest
   to the center are concatenated (budgeted) and re-encoded with the frozen
   MiniLM; the fused readable text ships in the artifact.

Everything before student training is deterministic (maxmin + Lloyd + fixed
truncation), so the constructor has zero condensation-seed variance.

Evaluation: students train ONLY on the condensed graph; zero-shot 7-class
Macro-F1 on a deterministic dev split of real Cora (transductive propagation).
Test indices are never read. Dev split: node_id % 10 == 1 (10%).
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

CLASS_NAMES = ["Case Based", "Genetic Algorithms", "Neural Networks",
               "Probabilistic Methods", "Reinforcement Learning",
               "Rule Learning", "Theory"]
PROMPTS = [f"a research paper about {name.lower()}" for name in CLASS_NAMES]


# --------------------------- frozen text encoder ---------------------------

class FrozenMiniLM:
    def __init__(self, path: str, device: str):
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModel.from_pretrained(path).to(device).eval()
        self.device = device

    @torch.no_grad()
    def encode(self, texts: list[str], batch_size: int = 64, max_length: int = 256) -> torch.Tensor:
        parts = []
        for start in range(0, len(texts), batch_size):
            batch = self.tokenizer(texts[start:start + batch_size], padding=True,
                                   truncation=True, max_length=max_length,
                                   return_tensors="pt").to(self.device)
            hidden = self.model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1e-9)
            parts.append(F.normalize(pooled.float(), dim=1).cpu())
        return torch.cat(parts)


# ----------------------------- student model ------------------------------

class GCNLayer(torch.nn.Module):
    def __init__(self, din, dout):
        super().__init__()
        self.lin = torch.nn.Linear(din, dout)

    def forward(self, x, adj):
        return adj @ self.lin(x)


class CLIPStudent(torch.nn.Module):
    """Simplified GraphCLIP student: GCN graph tower + linear text projector."""

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


def norm_adj(edge_index: torch.Tensor, n: int, device) -> torch.Tensor:
    idx = torch.cat((edge_index, torch.arange(n).repeat(2, 1)), 1).to(device)
    val = torch.ones(idx.shape[1], device=device)
    deg = torch.zeros(n, device=device).scatter_add_(0, idx[0], val)
    dinv = deg.clamp_min(1e-9).pow(-0.5)
    return torch.sparse_coo_tensor(idx, dinv[idx[0]] * val * dinv[idx[1]],
                                   (n, n)).coalesce()


# ----------------------- deterministic condensation ------------------------

def condense(x: torch.Tensor, text_emb: torch.Tensor, texts: list[str], k: int,
             fuse_g: int, encoder: FrozenMiniLM, degree: float,
             fuse_chars: int = 320) -> dict:
    n = len(x)
    joint = torch.cat((F.normalize(x.float(), dim=1), text_emb), dim=1)
    # deterministic maxmin seeding (uniform mass)
    first = 0
    centers = [first]
    dist = torch.cdist(joint, joint[first:first + 1]).flatten().square()
    for _ in range(1, k):
        dist[torch.tensor(centers)] = -1.0
        nxt = int(torch.argmax(dist))
        centers.append(nxt)
        dist = torch.minimum(dist, torch.cdist(joint, joint[nxt:nxt + 1]).flatten().square())
    centers_idx = torch.tensor(centers)
    c = joint[centers_idx].clone()
    for _ in range(25):
        assign = torch.cdist(joint, c).argmin(1)
        assign[centers_idx] = torch.arange(k)
        newc = torch.stack([joint[assign == j].mean(0) for j in range(k)])
        if torch.allclose(newc, c, atol=1e-6, rtol=0):
            break
        c = newc
    assign = torch.cdist(joint, c).argmin(1)
    assign[centers_idx] = torch.arange(k)

    syn_x, fused_texts, weights = [], [], []
    for j in range(k):
        members = torch.where(assign == j)[0]
        syn_x.append(x[members].float().mean(0))          # synthetic X: no real row
        d = torch.cdist(joint[members], c[j:j + 1]).flatten()
        order = members[torch.argsort(d)][:fuse_g]
        budget = max(48, fuse_chars // max(1, len(order)))
        fused = " ".join(texts[int(i)][:budget] for i in order)  # text fusion
        fused_texts.append(fused)
        weights.append(float(len(members)))
    syn_x = torch.stack(syn_x)
    fused_emb = encoder.encode(fused_texts)

    # deterministic symmetric kNN topology in fused-text space
    pairs = torch.triu_indices(k, k, offset=1)
    score = (fused_emb[pairs[0]] * fused_emb[pairs[1]]).sum(1)
    count = min(int(k * degree), pairs.shape[1])
    chosen = pairs[:, torch.topk(score, count).indices]
    edges = torch.cat((chosen, chosen.flip(0)), 1)
    return {"x": syn_x, "text_emb": fused_emb, "texts": fused_texts,
            "edge_index": edges, "weights": torch.tensor(weights),
            "assign": assign, "meta": {"k": k, "fuse_g": fuse_g, "degree": degree,
                                       "x": "cluster_centroid_synthetic",
                                       "text": "top-g_member_concat_reencoded",
                                       "deterministic": True}}


# ------------------------------ train / eval -------------------------------

def train_student(syn: dict, seed: int, device: str, steps: int = 512,
                  lr: float = 1e-3, batch: int = 24) -> CLIPStudent:
    torch.manual_seed(seed)
    model = CLIPStudent(syn["x"].shape[1]).to(device)
    x = syn["x"].to(device)
    t = syn["text_emb"].to(device)
    adj = norm_adj(syn["edge_index"], len(x), device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(x)
    gen = torch.Generator().manual_seed(seed * 7 + 1)
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
def zero_shot_eval(model: CLIPStudent, x, adj, prompt_emb, y, mask) -> dict:
    pred = model.logits(x, adj, prompt_emb.to(x.device)).argmax(1).cpu()
    yy, pp = y[mask].numpy(), pred[mask].numpy()
    return {"macro_f1": 100 * f1_score(yy, pp, average="macro", zero_division=0),
            "accuracy": 100 * accuracy_score(yy, pp)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/home/pc/research/GraphCLIP/processed_data/cora.pt")
    parser.add_argument("--minilm", default="/home/pc/models/all-MiniLM-L6-v2")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rate", type=float, default=0.05)
    parser.add_argument("--fuse-g", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--degree", type=float, default=2.0)
    parser.add_argument("--student-seeds", type=int, nargs="+",
                        default=[100, 101, 102, 103, 104, 105])
    parser.add_argument("--student-steps", type=int, default=512)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "results.json").exists():
        raise FileExistsError("existing runs are immutable")

    data = torch.load(args.data, map_location="cpu", weights_only=False)
    x, y, texts = data.x, data.y, data.raw_texts
    n = len(x)
    dev_mask = (torch.arange(n) % 10) == 1          # deterministic 10% dev
    k = math.floor(args.rate * n)
    device = args.device
    encoder = FrozenMiniLM(args.minilm, device)
    text_emb = encoder.encode(list(texts))
    prompt_emb = encoder.encode(PROMPTS)
    real_adj = norm_adj(data.edge_index, n, device)
    xg = x.float().to(device)

    def run_arm(name: str, syn: dict) -> dict:
        scores = []
        for seed in args.student_seeds:
            model = train_student(syn, seed, device, steps=args.student_steps)
            scores.append(zero_shot_eval(model, xg, real_adj, prompt_emb, y, dev_mask))
            del model
        f1s = [s["macro_f1"] for s in scores]
        row = {"arm": name, "roots": len(syn["x"]),
               "dev_macro_f1_mean": float(np.mean(f1s)),
               "dev_macro_f1_std": float(np.std(f1s)),
               "dev_accuracy_mean": float(np.mean([s["accuracy"] for s in scores])),
               "per_student": scores, "meta": syn.get("meta", {})}
        print(json.dumps({key: row[key] for key in ("arm", "roots", "dev_macro_f1_mean",
                                                    "dev_macro_f1_std")}), flush=True)
        return row

    results = []
    # ---- ours: synthetic X + fused text, sweep fuse_g -------------------
    for g in args.fuse_g:
        syn = condense(x, text_emb, list(texts), k, g, encoder, args.degree)
        again = condense(x, text_emb, list(texts), k, g, encoder, args.degree)
        for key in ("x", "text_emb", "edge_index", "weights"):
            if not torch.equal(syn[key], again[key]):
                raise RuntimeError(f"nondeterministic field {key}")
        torch.save({key: syn[key] for key in ("x", "text_emb", "texts", "edge_index", "weights", "meta")},
                   out / f"syn_fuseg{g}.pt")
        results.append(run_arm(f"synthetic_fused_g{g}", syn))

    # ---- baseline: random real nodes (3 draws) --------------------------
    for draw in range(3):
        gen = torch.Generator().manual_seed(1000 + draw)
        idx = torch.randperm(n, generator=gen)[:k]
        remap = torch.full((n,), -1, dtype=torch.long)
        remap[idx] = torch.arange(k)
        e = data.edge_index
        keep = (remap[e[0]] >= 0) & (remap[e[1]] >= 0)
        syn = {"x": x[idx].float(), "text_emb": text_emb[idx],
               "texts": [texts[int(i)] for i in idx],
               "edge_index": torch.stack((remap[e[0, keep]], remap[e[1, keep]])),
               "weights": torch.ones(k), "meta": {"draw": draw}}
        results.append(run_arm(f"random_real_draw{draw}", syn))

    # ---- reference: full real graph (non-compressed) ---------------------
    syn_full = {"x": x.float(), "text_emb": text_emb, "texts": list(texts),
                "edge_index": data.edge_index, "weights": torch.ones(n), "meta": {}}
    results.append(run_arm("full_real_reference", syn_full))

    json.dump({"results": results, "k": k, "rate": args.rate, "degree": args.degree,
               "dev_rule": "node_id % 10 == 1", "test_accessed": False,
               "prompts": PROMPTS, "student_steps": args.student_steps},
              open(out / "results.json", "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
