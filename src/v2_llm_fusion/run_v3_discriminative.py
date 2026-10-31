"""Discriminability-aware LLM fusion: round 2 of direction A.

Round-1 finding: budget-matched LLM abstracts lose to concat by -9.1 (0/6)
because smooth academic phrasing collapses embedding discriminability, while
same-length comparison shows LLM organization helps (+4.7, 5/6). Round 2
re-targets the LLM at DISCRIMINABILITY instead of fluency.

Arms (all budget-matched to ~320-360 chars, vs v1 baseline):
  v1_concat        frozen_v1 baseline (top-5 truncated concat, 320 chars)
  disc_prompt      LLM description forced to reuse verbatim technical terms,
                   generic phrasing banned
  keyword_bag      LLM extracts 15-20 most distinctive verbatim terms/phrases
                   from up to 12 members -> comma list (max literal sharpness,
                   LLM's cross-cluster selection replaces top-5 truncation)
  concat_kw        240-char concat + ~100-char LLM keywords appended (floor
                   protection + coverage)
  best_of_n        4 disc-prompt variants per cluster; pick candidate whose
                   MiniLM embedding is farthest from other clusters' mean
                   (label-free training-free proxy for RL objective)

Protocol identical to prior gates: Cora k=135, deterministic clustering,
mean synthetic X, text-kNN deg=1.0, 6 students x 2048 steps, dev-only.
LLM: local Qwen3-32B-AWQ vLLM, t=0 (t=0.7 for best_of_n variants), no-think.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModel, AutoTokenizer

CLASS_NAMES = ["case based", "genetic algorithms", "neural networks",
               "probabilistic methods", "reinforcement learning",
               "rule learning", "theory"]
PROMPTS = [f"a research paper about {n}" for n in CLASS_NAMES]
VLLM_URL = "http://localhost:8000/v1/chat/completions"
BUDGET = 340  # chars, matched to v1's ~320 measured average


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


def cluster(x, text_emb, k):
    joint = torch.cat((F.normalize(x.float(), dim=1), text_emb), dim=1)
    centers = [0]
    dist = torch.cdist(joint, joint[0:1]).flatten().square()
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
    return joint, assign, c


# ------------------------------ LLM plumbing --------------------------------

def call_llm(prompt, max_tokens=200, temperature=0.0):
    for _ in range(3):
        try:
            r = requests.post(VLLM_URL, json={
                "model": "qwen3-32b",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "chat_template_kwargs": {"enable_thinking": False},
            }, timeout=180)
            if r.status_code == 200:
                text = r.json()["choices"][0]["message"]["content"].strip()
                if "</think>" in text:
                    text = text.split("</think>")[-1].strip()
                if text:
                    return text
        except Exception:
            pass
        time.sleep(2)
    return None


def excerpts(member_texts, n=12, each=120):
    return "\n".join(f"{i+1}. {t[:each]}" for i, t in enumerate(member_texts[:n]))


def disc_prompt_text(member_texts, variant=0):
    variants = [
        "Describe what is DISTINCTIVE about this cluster",
        "List this cluster's unique research focus",
        "State what separates this cluster from other machine-learning topics",
        "Summarize the specific technical niche of this cluster",
    ]
    return (f"Below are research paper excerpts from ONE topic cluster (a larger corpus "
            f"has many other clusters on different topics):\n\n{excerpts(member_texts)}\n\n"
            f"{variants[variant % 4]} in about 45 words. STRICT rules:\n"
            f"- Reuse exact technical terms and phrases from the excerpts VERBATIM\n"
            f"- NO generic academic phrasing (never 'this paper', 'various approaches', "
            f"'explores', 'presents')\n"
            f"- Prefer rare, specific terminology over broad descriptions\n"
            f"Output only the description. /no_think")


def keyword_prompt_text(member_texts):
    return (f"Below are research paper excerpts from ONE topic cluster:\n\n"
            f"{excerpts(member_texts)}\n\n"
            f"Extract the 15-20 most distinctive technical terms and short phrases "
            f"(copy them VERBATIM from the excerpts; prefer rare, specific terms; "
            f"no generic words like 'learning' or 'algorithm' alone). "
            f"Output ONLY a comma-separated list. /no_think")


def concat_fusion(member_texts, budget):
    per = max(48, budget // max(1, len(member_texts)))
    return " ".join(t[:per] for t in member_texts)


def fit_budget(text, budget=BUDGET):
    if len(text) <= budget:
        return text
    return text[:budget].rsplit(" ", 1)[0]


def run_parallel(fn, arg_lists, workers=8):
    results = [None] * len(arg_lists)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for idx, val in pool.map(fn, list(enumerate(arg_lists))):
            results[idx] = val
    return results


# ------------------------------- condensation -------------------------------

def build_arm_texts(arm, member_lists, encoder):
    k = len(member_lists)
    if arm == "v1_concat":
        return [concat_fusion(m[:5], 320) for m in member_lists]

    if arm == "disc_prompt":
        def job(a):
            i, m = a
            out = call_llm(disc_prompt_text(m), max_tokens=120)
            return i, fit_budget(out) if out else concat_fusion(m[:5], 320)
        return run_parallel(job, member_lists)

    if arm == "keyword_bag":
        def job(a):
            i, m = a
            out = call_llm(keyword_prompt_text(m), max_tokens=150)
            return i, fit_budget(out) if out else concat_fusion(m[:5], 320)
        return run_parallel(job, member_lists)

    if arm == "concat_kw":
        def job(a):
            i, m = a
            out = call_llm(keyword_prompt_text(m), max_tokens=150)
            base = concat_fusion(m[:5], 240)
            kw = fit_budget(out, 100) if out else ""
            return i, (base + " " + kw).strip()
        return run_parallel(job, member_lists)

    if arm == "best_of_n":
        # generate 4 disc-prompt variants per cluster (t=0.7 for diversity)
        def job(a):
            i, m = a
            cands = []
            for v in range(4):
                out = call_llm(disc_prompt_text(m, variant=v), max_tokens=120,
                               temperature=0.7)
                if out:
                    cands.append(fit_budget(out))
            if not cands:
                cands = [concat_fusion(m[:5], 320)]
            return i, cands
        all_cands = run_parallel(job, member_lists)
        # embed all candidates; pick per cluster the one farthest from OTHER
        # clusters' candidate-mean embedding (label-free discriminability)
        flat, owner = [], []
        for ci, cands in enumerate(all_cands):
            for t in cands:
                flat.append(t)
                owner.append(ci)
        emb = encoder.encode(flat)
        owner = torch.tensor(owner)
        cluster_means = torch.stack([emb[owner == ci].mean(0) for ci in range(k)])
        cluster_means = F.normalize(cluster_means, dim=1)
        chosen = []
        for ci in range(k):
            idxs = (owner == ci).nonzero(as_tuple=True)[0]
            others = F.normalize(
                cluster_means[torch.arange(k) != ci].mean(0, keepdim=True), dim=1)
            sims = (F.normalize(emb[idxs], dim=1) @ others.T).flatten()
            best_local = idxs[int(torch.argmin(sims))]
            chosen.append(flat[int(best_local)])
        return chosen

    raise ValueError(arm)


def condense(data, text_emb, texts, k, encoder, arm, degree=1.0):
    x = data.x
    joint, assign, c = cluster(x, text_emb, k)
    syn_x, member_lists, weights = [], [], []
    for j in range(k):
        members = torch.where(assign == j)[0]
        syn_x.append(x[members].float().mean(0))
        d = torch.cdist(joint[members], c[j:j + 1]).flatten()
        order = members[torch.argsort(d)][:12]
        member_lists.append([texts[int(i)] for i in order])
        weights.append(float(len(members)))
    syn_x = torch.stack(syn_x)

    started = time.time()
    fused = build_arm_texts(arm, member_lists, encoder)
    fuse_seconds = time.time() - started

    fused_emb = encoder.encode(fused)
    pairs = torch.triu_indices(k, k, offset=1)
    score = (fused_emb[pairs[0]] * fused_emb[pairs[1]]).sum(1)
    count = min(int(k * degree), pairs.shape[1])
    chosen = pairs[:, torch.topk(score, count).indices]
    edges = torch.cat((chosen, chosen.flip(0)), 1)
    # discriminability audit: mean pairwise cosine of fused embeddings
    mean_pairwise_cos = float(score.mean())
    return {"x": syn_x, "text_emb": fused_emb, "texts": fused, "edge_index": edges,
            "weights": torch.tensor(weights),
            "meta": {"arm": arm, "k": k,
                     "text_utf8_bytes": sum(len(t.encode()) for t in fused),
                     "mean_pairwise_cos": mean_pairwise_cos,
                     "fuse_seconds": fuse_seconds}}


def train_student(syn, seed, device, steps=2048, lr=1e-3, batch=24):
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
    parser.add_argument("--arms", nargs="+",
                        default=["v1_concat", "disc_prompt", "keyword_bag", "concat_kw", "best_of_n"])
    parser.add_argument("--student-seeds", type=int, nargs="+",
                        default=[100, 101, 102, 103, 104, 105])
    parser.add_argument("--student-steps", type=int, default=2048)
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

    results = []
    for arm in args.arms:
        print(f"\n=== {arm} ===", flush=True)
        syn = condense(data, text_emb, texts, k, encoder, arm)
        torch.save({kk: syn[kk] for kk in ("x", "text_emb", "texts", "edge_index", "weights", "meta")},
                   out / f"syn_{arm}.pt")
        print(f"  bytes={syn['meta']['text_utf8_bytes']} "
              f"pairwise_cos={syn['meta']['mean_pairwise_cos']:.4f} "
              f"fuse={syn['meta']['fuse_seconds']:.0f}s", flush=True)
        scores = []
        for seed in args.student_seeds:
            model = train_student(syn, seed, device, steps=args.student_steps)
            scores.append(zero_shot_eval(model, xg, real_adj, prompt_emb, y, dev_mask))
            del model
        f1s = [s["macro_f1"] for s in scores]
        row = {"arm": arm, "roots": k,
               "dev_macro_f1_mean": float(np.mean(f1s)),
               "dev_macro_f1_std": float(np.std(f1s)),
               "text_utf8_bytes": syn["meta"]["text_utf8_bytes"],
               "mean_pairwise_cos": syn["meta"]["mean_pairwise_cos"],
               "fuse_seconds": syn["meta"]["fuse_seconds"],
               "per_student": scores}
        results.append(row)
        print(json.dumps({kk: row[kk] for kk in ("arm", "dev_macro_f1_mean", "dev_macro_f1_std",
                                                 "mean_pairwise_cos")}), flush=True)

    json.dump({"results": results, "k": k, "rate": args.rate,
               "dev_rule": "node_id % 10 == 1", "test_accessed": False,
               "budget_chars": BUDGET,
               "llm": "local qwen3-32b-awq vllm no-think"},
              open(out / "results.json", "w"), indent=1)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
