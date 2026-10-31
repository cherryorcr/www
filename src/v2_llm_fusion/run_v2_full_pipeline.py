"""Full pipeline: v1 concat vs v2 local-LLM fusion, with student training.

Static Cora gate (identical protocol to the frozen_v1 graphclip gate):
deterministic maxmin+Lloyd clustering in [X||text] space, synthetic X =
cluster mean, text per arm, MiniLM re-encode, text-kNN topology deg=1.0,
CLIP GCN students x 6 seeds x 2048 steps, dev Macro-F1 (node_id%10==1).

Arms (pre-registered):
  v1_concat_320   frozen_v1 baseline (top-5 truncated concat, 320 chars)
  v1_concat_700   longer concat control (top-8, ~700 chars) - isolates
                  "longer text" from "LLM quality"
  v2_llm_50w      LLM fusion, ~50 words (~320 chars, budget-matched to v1)
  v2_llm_100w     LLM fusion, ~100 words (~700 chars)

LLM: local Qwen3-32B-AWQ via vLLM (localhost:8000), temperature=0,
thinking disabled, English prompts. Parallel requests via thread pool.
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


class FrozenMiniLM:
    def __init__(self, path: str, device: str):
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


# --------------------- deterministic clustering (frozen_v1) -----------------

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


# ------------------------------- text fusion --------------------------------

def concat_fusion(member_texts, budget_chars):
    per = max(48, budget_chars // max(1, len(member_texts)))
    return " ".join(t[:per] for t in member_texts)


def llm_fuse_one(args):
    idx, member_texts, target_words = args
    texts_str = "\n".join(f"{i+1}. {t[:150]}" for i, t in enumerate(member_texts[:8]))
    prompt = (f"Below are {min(len(member_texts), 8)} research paper excerpts from one "
              f"topic cluster:\n\n{texts_str}\n\n"
              f"Write ONE fused abstract of about {target_words} words that covers the "
              f"common research theme and preserves key technical terms (algorithms, "
              f"models, methods). Output ONLY the abstract. /no_think")
    for attempt in range(3):
        try:
            r = requests.post(VLLM_URL, json={
                "model": "qwen3-32b",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": int(target_words * 2.5),
                "temperature": 0,
                "chat_template_kwargs": {"enable_thinking": False},
            }, timeout=180)
            if r.status_code == 200:
                text = r.json()["choices"][0]["message"]["content"].strip()
                if "</think>" in text:
                    text = text.split("</think>")[-1].strip()
                if text:
                    return idx, text
        except Exception:
            pass
        time.sleep(2)
    return idx, concat_fusion(member_texts[:5], 320)  # fallback = v1


def llm_fusion_batch(cluster_texts, target_words, workers=8):
    """cluster_texts: list of member-text lists, one per cluster."""
    results = [None] * len(cluster_texts)
    fallback = 0
    jobs = [(i, t, target_words) for i, t in enumerate(cluster_texts)]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for idx, text in pool.map(llm_fuse_one, jobs):
            results[idx] = text
    return results


# ------------------------------- condensation -------------------------------

def condense(data, text_emb, texts, k, encoder, arm, degree=1.0, fuse_g=5):
    x = data.x
    joint, assign, c = cluster(x, text_emb, k)
    syn_x, member_lists, weights = [], [], []
    for j in range(k):
        members = torch.where(assign == j)[0]
        syn_x.append(x[members].float().mean(0))
        d = torch.cdist(joint[members], c[j:j + 1]).flatten()
        n_pick = 8 if "llm" in arm or "700" in arm else fuse_g
        order = members[torch.argsort(d)][:n_pick]
        member_lists.append([texts[int(i)] for i in order])
        weights.append(float(len(members)))
    syn_x = torch.stack(syn_x)

    started = time.time()
    if arm == "v1_concat_320":
        fused = [concat_fusion(m[:5], 320) for m in member_lists]
    elif arm == "v1_concat_700":
        fused = [concat_fusion(m[:8], 700) for m in member_lists]
    elif arm == "v2_llm_50w":
        fused = llm_fusion_batch(member_lists, target_words=50)
    elif arm == "v2_llm_100w":
        fused = llm_fusion_batch(member_lists, target_words=100)
    else:
        raise ValueError(arm)
    fuse_seconds = time.time() - started

    fused_emb = encoder.encode(fused)
    pairs = torch.triu_indices(k, k, offset=1)
    score = (fused_emb[pairs[0]] * fused_emb[pairs[1]]).sum(1)
    count = min(int(k * degree), pairs.shape[1])
    chosen = pairs[:, torch.topk(score, count).indices]
    edges = torch.cat((chosen, chosen.flip(0)), 1)
    text_bytes = sum(len(t.encode()) for t in fused)
    return {"x": syn_x, "text_emb": fused_emb, "texts": fused, "edge_index": edges,
            "weights": torch.tensor(weights),
            "meta": {"arm": arm, "k": k, "degree": degree,
                     "text_utf8_bytes": text_bytes, "fuse_seconds": fuse_seconds}}


# ------------------------------ train / eval --------------------------------

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
                        default=["v1_concat_320", "v1_concat_700", "v2_llm_50w", "v2_llm_100w"])
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
        print(f"  condensed: k={k}, text_bytes={syn['meta']['text_utf8_bytes']}, "
              f"fuse_time={syn['meta']['fuse_seconds']:.0f}s", flush=True)
        scores = []
        for seed in args.student_seeds:
            model = train_student(syn, seed, device, steps=args.student_steps)
            scores.append(zero_shot_eval(model, xg, real_adj, prompt_emb, y, dev_mask))
            del model
        f1s = [s["macro_f1"] for s in scores]
        row = {"arm": arm, "roots": k,
               "dev_macro_f1_mean": float(np.mean(f1s)),
               "dev_macro_f1_std": float(np.std(f1s)),
               "dev_accuracy_mean": float(np.mean([s["accuracy"] for s in scores])),
               "text_utf8_bytes": syn["meta"]["text_utf8_bytes"],
               "fuse_seconds": syn["meta"]["fuse_seconds"],
               "per_student": scores}
        results.append(row)
        print(json.dumps({kk: row[kk] for kk in ("arm", "dev_macro_f1_mean", "dev_macro_f1_std",
                                                 "text_utf8_bytes")}), flush=True)

    json.dump({"results": results, "k": k, "rate": args.rate,
               "dev_rule": "node_id % 10 == 1", "test_accessed": False,
               "student_steps": args.student_steps,
               "llm": "local qwen3-32b-awq vllm t=0 no-think"},
              open(out / "results.json", "w"), indent=1)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
