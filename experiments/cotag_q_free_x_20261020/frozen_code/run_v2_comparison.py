"""GraphCLIP constructive condensation v2 with LLM text fusion.

This is the v2 upgrade of the frozen_v1 constructive method. The key difference:
- v1: top-g texts by distance → truncated concatenation → re-encode
- v2: top-g texts by distance → LLM-generated summary → re-encode

Everything else (clustering, synthetic X, topology) remains the same to isolate
the effect of LLM text fusion.
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

# Import LLM fusion module
import sys
sys.path.append(str(Path(__file__).parent.parent / "v2_llm_fusion"))
from llm_text_fusion import LLMTextFusion

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

def condense_v1(x: torch.Tensor, text_emb: torch.Tensor, texts: list[str], k: int,
                fuse_g: int, encoder: FrozenMiniLM, degree: float,
                fuse_chars: int = 320) -> dict:
    """V1 baseline: top-g concatenation."""
    n = len(x)
    joint = torch.cat((F.normalize(x.float(), dim=1), text_emb), dim=1)

    # Deterministic maxmin + Lloyd clustering
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
        syn_x.append(x[members].float().mean(0))  # Synthetic X: cluster mean

        # V1: select top-g by distance, concatenate with truncation
        d = torch.cdist(joint[members], c[j:j + 1]).flatten()
        order = members[torch.argsort(d)][:fuse_g]
        budget = max(48, fuse_chars // max(1, len(order)))
        fused = " ".join(texts[int(i)][:budget] for i in order)
        fused_texts.append(fused)
        weights.append(float(len(members)))

    syn_x = torch.stack(syn_x)
    fused_emb = encoder.encode(fused_texts)

    # Deterministic symmetric kNN topology
    pairs = torch.triu_indices(k, k, offset=1)
    score = (fused_emb[pairs[0]] * fused_emb[pairs[1]]).sum(1)
    count = min(int(k * degree), pairs.shape[1])
    chosen = pairs[:, torch.topk(score, count).indices]
    edges = torch.cat((chosen, chosen.flip(0)), 1)

    return {"x": syn_x, "text_emb": fused_emb, "texts": fused_texts,
            "edge_index": edges, "weights": torch.tensor(weights),
            "assign": assign, "meta": {"k": k, "fuse_g": fuse_g, "degree": degree,
                                       "method": "v1_concat",
                                       "x": "cluster_centroid_synthetic",
                                       "text": "top-g_concat_reencoded",
                                       "deterministic": True}}


def condense_v2_llm(x: torch.Tensor, text_emb: torch.Tensor, texts: list[str], k: int,
                     fuse_g: int, encoder: FrozenMiniLM, degree: float,
                     llm_backend: str = "gpt4o", target_words: int = 200,
                     **llm_kwargs) -> dict:
    """V2: LLM-driven text fusion."""
    n = len(x)
    joint = torch.cat((F.normalize(x.float(), dim=1), text_emb), dim=1)

    # Same clustering as v1
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

    # Collect texts per cluster for LLM fusion
    texts_per_cluster = []
    syn_x, weights = [], []
    for j in range(k):
        members = torch.where(assign == j)[0]
        syn_x.append(x[members].float().mean(0))

        # Select top-g by distance
        d = torch.cdist(joint[members], c[j:j + 1]).flatten()
        order = members[torch.argsort(d)][:fuse_g]
        cluster_texts = [texts[int(i)] for i in order]
        texts_per_cluster.append(cluster_texts)
        weights.append(float(len(members)))

    syn_x = torch.stack(syn_x)

    # V2: LLM batch fusion
    print(f"Fusing {k} clusters with LLM ({llm_backend})...")
    try:
        fusioner = LLMTextFusion(backend=llm_backend, **llm_kwargs)
        fused_texts = []
        for i, cluster_texts in enumerate(texts_per_cluster):
            print(f"  Cluster {i+1}/{k}...")
            fused = fusioner.fuse_texts(cluster_texts, cluster_id=i,
                                        target_words=target_words,
                                        domain="machine learning research")
            fused_texts.append(fused)
            time.sleep(0.5)  # Rate limiting
    except Exception as e:
        print(f"LLM fusion failed: {e}")
        print("Falling back to v1 concatenation")
        # Fallback to v1
        fused_texts = []
        for cluster_texts in texts_per_cluster:
            fused = " ".join(cluster_texts)[:800]
            fused_texts.append(fused)

    fused_emb = encoder.encode(fused_texts)

    # Same topology as v1
    pairs = torch.triu_indices(k, k, offset=1)
    score = (fused_emb[pairs[0]] * fused_emb[pairs[1]]).sum(1)
    count = min(int(k * degree), pairs.shape[1])
    chosen = pairs[:, torch.topk(score, count).indices]
    edges = torch.cat((chosen, chosen.flip(0)), 1)

    return {"x": syn_x, "text_emb": fused_emb, "texts": fused_texts,
            "edge_index": edges, "weights": torch.tensor(weights),
            "assign": assign, "meta": {"k": k, "fuse_g": fuse_g, "degree": degree,
                                       "method": "v2_llm_fusion",
                                       "llm_backend": llm_backend,
                                       "target_words": target_words,
                                       "x": "cluster_centroid_synthetic",
                                       "text": "llm_generated_summary_reencoded",
                                       "deterministic": False}}  # LLM may have variance


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
    parser.add_argument("--fuse-g", type=int, default=5)
    parser.add_argument("--degree", type=float, default=1.0)
    parser.add_argument("--student-seeds", type=int, nargs="+",
                        default=[100, 101, 102, 103, 104, 105])
    parser.add_argument("--student-steps", type=int, default=2048)

    # V2-specific args
    parser.add_argument("--method", choices=["v1_concat", "v2_llm"], required=True,
                        help="Condensation method: v1 (concat) or v2 (LLM)")
    parser.add_argument("--llm-backend", default="gpt4o",
                        choices=["gpt4o", "gpt4-turbo", "claude-sonnet", "claude-opus", "llama70b"])
    parser.add_argument("--target-words", type=int, default=200,
                        help="Target length for LLM-generated summaries")

    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    # Load data
    data = torch.load(args.data, map_location="cpu", weights_only=False)
    x, y, texts = data.x, data.y, data.raw_texts
    n = len(x)
    dev_mask = (torch.arange(n) % 10) == 1  # 10% dev
    k = math.floor(args.rate * n)
    device = args.device

    # Initialize encoder
    encoder = FrozenMiniLM(args.minilm, device)
    text_emb = encoder.encode(list(texts))
    prompt_emb = encoder.encode(PROMPTS)
    real_adj = norm_adj(data.edge_index, n, device)
    xg = x.float().to(device)

    # Condense
    print(f"\n=== Condensing with {args.method} (k={k}, g={args.fuse_g}, deg={args.degree}) ===")
    if args.method == "v1_concat":
        syn = condense_v1(x, text_emb, list(texts), k, args.fuse_g, encoder, args.degree)
    elif args.method == "v2_llm":
        syn = condense_v2_llm(x, text_emb, list(texts), k, args.fuse_g, encoder, args.degree,
                              llm_backend=args.llm_backend, target_words=args.target_words)

    # Save synthetic artifact
    torch.save(syn, out / f"syn_{args.method}.pt")

    # Train students
    print(f"\n=== Training {len(args.student_seeds)} students ({args.student_steps} steps) ===")
    scores = []
    for seed in args.student_seeds:
        print(f"  Student seed {seed}...")
        model = train_student(syn, seed, device, steps=args.student_steps)
        score = zero_shot_eval(model, xg, real_adj, prompt_emb, y, dev_mask)
        scores.append(score)
        print(f"    Dev Macro-F1: {score['macro_f1']:.2f}%")
        del model

    # Aggregate results
    f1s = [s["macro_f1"] for s in scores]
    result = {
        "method": args.method,
        "llm_backend": args.llm_backend if args.method == "v2_llm" else None,
        "target_words": args.target_words if args.method == "v2_llm" else None,
        "k": k,
        "fuse_g": args.fuse_g,
        "degree": args.degree,
        "student_steps": args.student_steps,
        "dev_macro_f1_mean": float(np.mean(f1s)),
        "dev_macro_f1_std": float(np.std(f1s)),
        "per_student": scores,
        "meta": syn["meta"]
    }

    with open(out / "results.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n=== Final Results ===")
    print(f"Method: {args.method}")
    if args.method == "v2_llm":
        print(f"LLM: {args.llm_backend} (target: {args.target_words} words)")
    print(f"Dev Macro-F1: {result['dev_macro_f1_mean']:.2f} ± {result['dev_macro_f1_std']:.2f}%")
    print(f"\nResults saved to {out}/")


if __name__ == "__main__":
    main()
