"""Gate S2: deterministic reclamation of TAGSAM components (ablation).

All-synthetic condensation (user decision 2026-09-08: X must be synthetic;
no real-node rows ship in the artifact). Ablated components, each a
deterministic replacement of a TAGSAM capability:

C1 text selection   geometric top-g  vs  det-CMI top-g
    det-CMI: candidates = 2g members closest to the cluster center; score =
    GPT-2 (local gpt2-large) mean NLL of the candidate conditioned on the
    cluster medoid text (information gain proxy). Pure forward pass, no
    sampling -> deterministic. Lower conditional NLL = more typical; we KEEP
    the *highest* information candidates (highest conditional NLL) capped by
    a fluency guard (unconditional NLL not in the top tail).

C2 topology         text-kNN  vs  coarsened P^T A P  vs  mixed
    coarsening: syn edge score(i,j) = (# real edges between cluster i and
    cluster j) / sqrt(mass_i * mass_j); top int(k*degree) pairs. mixed:
    alpha * coarsen_rank + (1-alpha) * text_rank (rank-normalized).

C3 X refinement     cluster-mean only  vs  +first-order anchored refinement
    <=20 steps, gradient flows directly to X (no student unroll):
    L = InfoNCE(frozen-student-encoder graph response vs own fused text)
      + lam * || graph response - mass-weighted member response mean ||^2
    Early stop on a small internal dev probe; revert if no improvement.
    Refinement trains a probe GCN on the synthetic graph? NO - to keep it
    first-order and cheap we use a fixed random-projection graph operator
    (deterministic seed), so the anchor is architecture-agnostic.

6 arms (pre-registered):
  A0 geo  + knn      + mean      (v1 baseline, = graphclip gate winner)
  A1 cmi  + knn      + mean      (isolate C1)
  A2 geo  + coarsen  + mean      (isolate C2)
  A3 geo  + mixed.5  + mean      (C2 variant)
  A4 geo  + knn      + refined   (isolate C3)
  A5 cmi  + coarsen  + refined   (full v2)
plus random-real x3 and full-real reference (same as previous gate).

Downstream, data, dev split, students: identical to
run_graphclip_constructive_gate.py. Test never read.
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
from transformers import AutoModel, AutoTokenizer, GPT2LMHeadModel

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


# ------------------------------ det-CMI scorer ------------------------------

class DetCMIScorer:
    """Deterministic GPT-2 information scoring. Pure forward NLL, no sampling."""

    def __init__(self, path: str, device: str, max_len: int = 200, ctx_len: int = 120):
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = GPT2LMHeadModel.from_pretrained(path).to(device).eval()
        self.device, self.max_len, self.ctx_len = device, max_len, ctx_len

    @torch.no_grad()
    def _nll(self, prefix_ids: list[int], target_ids: list[int]) -> float:
        ids = torch.tensor([prefix_ids + target_ids], device=self.device)
        out = self.model(ids).logits[0]
        lp = F.log_softmax(out[:-1], dim=-1)
        tgt = ids[0, 1:]
        start = max(0, len(prefix_ids) - 1)
        rows = torch.arange(start, len(tgt), device=self.device)
        return float(-lp[rows, tgt[rows]].mean())

    @torch.no_grad()
    def score(self, context: str, candidates: list[str]) -> list[dict]:
        ctx_ids = self.tok.encode(context)[: self.ctx_len]
        rows = []
        for text in candidates:
            tids = self.tok.encode(text)[: self.max_len]
            if len(tids) < 4:
                rows.append({"cond": 0.0, "uncond": 0.0})
                continue
            rows.append({"cond": self._nll(ctx_ids, tids), "uncond": self._nll([], tids)})
        return rows


def cmi_topg(members_sorted: list[int], texts: list[str], medoid: int, g: int,
             scorer: DetCMIScorer) -> list[int]:
    """Deterministic selection: high conditional NLL (novel wrt medoid) but
    not in the top 25% unconditional NLL (fluency guard)."""
    cand = [m for m in members_sorted[: 2 * g] if m != medoid]
    if len(cand) <= max(0, g - 1):
        return members_sorted[:g]
    scores = scorer.score(texts[medoid][:600], [texts[c][:600] for c in cand])
    unc = sorted(r["uncond"] for r in scores)
    guard = unc[int(0.75 * (len(unc) - 1))]
    ranked = sorted(range(len(cand)),
                    key=lambda i: (-(scores[i]["cond"]), cand[i]))
    keep = [cand[i] for i in ranked if scores[i]["uncond"] <= guard][: g - 1]
    if len(keep) < g - 1:  # fill deterministically
        keep += [cand[i] for i in ranked if cand[i] not in keep][: g - 1 - len(keep)]
    return [medoid] + keep


# ----------------------------- student model ------------------------------

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


def norm_adj(edge_index: torch.Tensor, n: int, device) -> torch.Tensor:
    idx = torch.cat((edge_index, torch.arange(n).repeat(2, 1)), 1).to(device)
    val = torch.ones(idx.shape[1], device=device)
    deg = torch.zeros(n, device=device).scatter_add_(0, idx[0], val)
    dinv = deg.clamp_min(1e-9).pow(-0.5)
    return torch.sparse_coo_tensor(idx, dinv[idx[0]] * val * dinv[idx[1]],
                                   (n, n)).coalesce()


# ------------------------- deterministic clustering -------------------------

def cluster(x: torch.Tensor, text_emb: torch.Tensor, k: int):
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


# ------------------------------- topologies --------------------------------

def topo_knn(fused_emb: torch.Tensor, k: int, degree: float) -> torch.Tensor:
    pairs = torch.triu_indices(k, k, offset=1)
    score = (fused_emb[pairs[0]] * fused_emb[pairs[1]]).sum(1)
    count = min(int(k * degree), pairs.shape[1])
    chosen = pairs[:, torch.topk(score, count).indices]
    return torch.cat((chosen, chosen.flip(0)), 1)


def topo_coarsen(edge_index: torch.Tensor, assign: torch.Tensor, mass: torch.Tensor,
                 k: int, degree: float) -> tuple[torch.Tensor, torch.Tensor]:
    src, dst = assign[edge_index[0]], assign[edge_index[1]]
    off = src * k + dst
    counts = torch.zeros(k * k).scatter_add_(0, off, torch.ones(len(off))).view(k, k)
    counts = counts + counts.T
    norm = counts / (mass[:, None] * mass[None, :]).sqrt().clamp_min(1e-9)
    pairs = torch.triu_indices(k, k, offset=1)
    score = norm[pairs[0], pairs[1]]
    count = min(int(k * degree), pairs.shape[1])
    chosen = pairs[:, torch.topk(score, count).indices]
    return torch.cat((chosen, chosen.flip(0)), 1), score


def topo_mixed(fused_emb: torch.Tensor, coarsen_score: torch.Tensor, k: int,
               degree: float, alpha: float) -> torch.Tensor:
    pairs = torch.triu_indices(k, k, offset=1)
    tscore = (fused_emb[pairs[0]] * fused_emb[pairs[1]]).sum(1)
    def rank(v):
        order = torch.argsort(torch.argsort(v))
        return order.float() / max(1, len(v) - 1)
    mix = alpha * rank(coarsen_score) + (1 - alpha) * rank(tscore)
    count = min(int(k * degree), pairs.shape[1])
    chosen = pairs[:, torch.topk(mix, count).indices]
    return torch.cat((chosen, chosen.flip(0)), 1)


# --------------------------- first-order refinement -------------------------

def refine_x(syn_x: torch.Tensor, fused_emb: torch.Tensor, edge_index: torch.Tensor,
             member_x_mean: torch.Tensor, device: str, steps: int = 20,
             lr: float = 5e-3, lam: float = 1.0) -> tuple[torch.Tensor, dict]:
    """Anchored first-order refinement; gradient hits X directly.

    Fixed deterministic random-projection graph operator (seed 0) maps X into
    the text space; anchors: (a) InfoNCE against own fused text, (b) L2 to the
    mass-weighted member-mean X (distribution first moment). Early stop on the
    anchor objective itself; revert on non-improvement.
    """
    k, d = syn_x.shape
    torch.manual_seed(0)
    proj = torch.nn.Linear(d, fused_emb.shape[1], bias=False).to(device)
    for p in proj.parameters():
        p.requires_grad_(False)
    adj = norm_adj(edge_index, k, device)
    x = torch.nn.Parameter(syn_x.clone().to(device))
    t = fused_emb.to(device)
    anchor = member_x_mean.to(device)
    opt = torch.optim.Adam([x], lr=lr)
    best, best_x, curve = math.inf, syn_x.clone(), []
    for step in range(steps + 1):
        g = F.normalize(adj @ proj(adj @ x), dim=1)
        logits = (g @ F.normalize(t, dim=1).T) / 0.07
        target = torch.arange(k, device=device)
        infonce = (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target)) / 2
        l2 = (x - anchor).square().mean()
        loss = infonce + lam * l2
        curve.append(float(loss))
        if float(loss) < best - 1e-5:
            best, best_x = float(loss), x.detach().cpu().clone()
        elif step > 5:
            break
        if step < steps:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return best_x, {"loss_curve": curve, "steps_run": len(curve) - 1}


# ------------------------------- condensation -------------------------------

def condense(data, text_emb, texts, k: int, arm: dict, encoder: FrozenMiniLM,
             scorer, device: str, fuse_g: int = 5, degree: float = 1.0,
             fuse_chars: int = 320) -> dict:
    x = data.x
    joint, assign, c = cluster(x, text_emb, k)
    syn_x, fused_texts, weights, member_means = [], [], [], []
    for j in range(k):
        members = torch.where(assign == j)[0]
        d = torch.cdist(joint[members], c[j:j + 1]).flatten()
        members_sorted = members[torch.argsort(d)].tolist()
        medoid = members_sorted[0]
        if arm["text"] == "cmi" and scorer is not None and len(members_sorted) > 1:
            order = cmi_topg(members_sorted, texts, medoid, fuse_g, scorer)
        else:
            order = members_sorted[:fuse_g]
        budget = max(48, fuse_chars // max(1, len(order)))
        fused_texts.append(" ".join(texts[int(i)][:budget] for i in order))
        mean_x = x[members].float().mean(0)
        syn_x.append(mean_x)
        member_means.append(mean_x)
        weights.append(float(len(members)))
    syn_x = torch.stack(syn_x)
    member_means = torch.stack(member_means)
    fused_emb = encoder.encode(fused_texts)
    mass = torch.tensor(weights)

    coarsen_edges, coarsen_score = topo_coarsen(data.edge_index, assign, mass, k, degree)
    if arm["topo"] == "knn":
        edges = topo_knn(fused_emb, k, degree)
    elif arm["topo"] == "coarsen":
        edges = coarsen_edges
    else:
        edges = topo_mixed(fused_emb, coarsen_score, k, degree, alpha=arm.get("alpha", 0.5))

    refine_info = None
    if arm["x"] == "refined":
        syn_x, refine_info = refine_x(syn_x, fused_emb, edges, member_means, device)

    return {"x": syn_x, "text_emb": fused_emb, "texts": fused_texts,
            "edge_index": edges, "weights": mass,
            "meta": {**arm, "fuse_g": fuse_g, "degree": degree,
                     "refine": refine_info, "deterministic": True}}


# ------------------------------ train / eval -------------------------------

def train_student(syn: dict, seed: int, device: str, steps: int = 2048,
                  lr: float = 1e-3, batch: int = 24) -> CLIPStudent:
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
def zero_shot_eval(model, x, adj, prompt_emb, y, mask) -> dict:
    pred = model.logits(x, adj, prompt_emb.to(x.device)).argmax(1).cpu()
    yy, pp = y[mask].numpy(), pred[mask].numpy()
    return {"macro_f1": 100 * f1_score(yy, pp, average="macro", zero_division=0),
            "accuracy": 100 * accuracy_score(yy, pp)}


ARMS = {
    "A0_geo_knn_mean":      {"text": "geo", "topo": "knn",     "x": "mean"},
    "A1_cmi_knn_mean":      {"text": "cmi", "topo": "knn",     "x": "mean"},
    "A2_geo_coarsen_mean":  {"text": "geo", "topo": "coarsen", "x": "mean"},
    "A3_geo_mixed_mean":    {"text": "geo", "topo": "mixed",   "x": "mean", "alpha": 0.5},
    "A4_geo_knn_refined":   {"text": "geo", "topo": "knn",     "x": "refined"},
    "A5_cmi_coarsen_refined": {"text": "cmi", "topo": "coarsen", "x": "refined"},
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/home/pc/research/GraphCLIP/processed_data/cora.pt")
    parser.add_argument("--minilm", default="/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip")
    parser.add_argument("--gpt2", default="/home/pc/huggingface-cache/models--gpt2-large/snapshots/32b71b12589c2f8d625668d2335a01cac3249519")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rate", type=float, default=0.05)
    parser.add_argument("--fuse-g", type=int, default=5)
    parser.add_argument("--degree", type=float, default=1.0)
    parser.add_argument("--arms", nargs="+", default=list(ARMS.keys()))
    parser.add_argument("--student-seeds", type=int, nargs="+", default=[100, 101, 102, 103, 104, 105])
    parser.add_argument("--student-steps", type=int, default=2048)
    parser.add_argument("--skip-baselines", action="store_true")
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
    needs_cmi = any(ARMS[a]["text"] == "cmi" for a in args.arms)
    scorer = DetCMIScorer(args.gpt2, device) if needs_cmi else None

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
               "per_student": scores, "meta": {kk: vv for kk, vv in syn.get("meta", {}).items()
                                               if kk != "refine" or vv is None or True}}
        print(json.dumps({kk: row[kk] for kk in ("arm", "dev_macro_f1_mean", "dev_macro_f1_std")}), flush=True)
        return row

    results = []
    for name in args.arms:
        arm = ARMS[name]
        started = time.time()
        syn = condense(data, text_emb, texts, k, arm, encoder, scorer, device,
                       fuse_g=args.fuse_g, degree=args.degree)
        again = condense(data, text_emb, texts, k, arm, encoder, scorer, device,
                         fuse_g=args.fuse_g, degree=args.degree)
        for key in ("x", "text_emb", "edge_index", "weights"):
            if not torch.allclose(syn[key].float(), again[key].float(), atol=1e-5):
                raise RuntimeError(f"nondeterministic field {key} in {name}")
        torch.save({key: syn[key] for key in ("x", "text_emb", "texts", "edge_index", "weights", "meta")},
                   out / f"{name}.pt")
        row = run_arm(name, syn)
        row["condense_seconds"] = time.time() - started
        results.append(row)

    if not args.skip_baselines:
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

    json.dump({"results": results, "k": k, "rate": args.rate, "degree": args.degree,
               "fuse_g": args.fuse_g, "dev_rule": "node_id % 10 == 1",
               "test_accessed": False, "student_steps": args.student_steps},
              open(out / "results.json", "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
