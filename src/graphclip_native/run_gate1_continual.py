"""Gate 1'': constructive continual TAG condensation, stage 0 -> 1 -> 2.

Class-incremental Cora (official GraphCLIP cora.pt):
  stage 0 classes [0,1]; stage 1 adds [2,3]; stage 2 adds [4,5,6].
After a stage closes, its raw training graph is unreadable. Only the exported
synthetic memory crosses the boundary. Memory rows are fully synthetic
(cluster-mean X, fused readable text re-encoded by the frozen MiniLM tower,
mass, fusion_depth). Budget k_t = floor(rate * cumulative train nodes).

Arms
  append_fixed      old memory frozen; delta condensed alone into the budget
                    remainder (constructive analogue of the old Append arm)
  recluster         mass-weighted joint re-clustering of old synthetic rows
                    (mass = represented real nodes) with new delta rows
                    (mass = 1); keep/merge/new emerge from one operator.
                    Layered text re-fusion with fusion_depth dilution guard.
  full_history      condense from all raw seen so far (upper reference,
                    violates the continual constraint; diagnostic only)

Downstream/eval identical to Gate S2: frozen MiniLM text tower extracted from
the GraphCLIP checkpoint, GCN CLIP student trained ONLY on the artifact,
zero-shot 7-prompt classification on the real graph (fixed label space),
deterministic dev split node_id%10==1, per-stage masks over seen classes.
Test never read. Constructor deterministic (audited by rebuild equality).
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
STAGE_CLASSES = [[0, 1], [2, 3], [4, 5, 6]]
FUSE_G = 5
DEGREE = 1.0
FUSE_CHARS = 320


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


# ------------------- mass-weighted deterministic clustering -----------------

def weighted_cluster(points: torch.Tensor, mass: torch.Tensor, k: int):
    """Deterministic maxmin seeding + mass-weighted Lloyd. Returns assignment."""
    n = len(points)
    if k >= n:
        return torch.arange(n)
    first = int(torch.argmax(mass))
    centers = [first]
    dist = torch.cdist(points, points[first:first + 1]).flatten().square() * mass
    for _ in range(1, k):
        dist[torch.tensor(centers)] = -1.0
        nxt = int(torch.argmax(dist))
        centers.append(nxt)
        dist = torch.minimum(dist, torch.cdist(points, points[nxt:nxt + 1]).flatten().square() * mass)
    centers_idx = torch.tensor(centers)
    c = points[centers_idx].clone()
    for _ in range(25):
        assign = torch.cdist(points, c).argmin(1)
        assign[centers_idx] = torch.arange(k)
        newc = []
        for j in range(k):
            members = torch.where(assign == j)[0]
            m = mass[members]
            newc.append((points[members] * m[:, None]).sum(0) / m.sum())
        newc = torch.stack(newc)
        if torch.allclose(newc, c, atol=1e-6, rtol=0):
            break
        c = newc
    assign = torch.cdist(points, c).argmin(1)
    assign[centers_idx] = torch.arange(k)
    return assign


# ----------------------------- memory structure -----------------------------
# memory dict: x (k,384) synthetic, text_emb (k,384), texts list, mass (k,),
#              depth (k,) long, stage (k,) long (birth stage)


def fuse_new_only(joint, members, texts, budget_chars=FUSE_CHARS, g=FUSE_G):
    center = joint[members].mean(0, keepdim=True)
    d = torch.cdist(joint[members], center).flatten()
    order = members[torch.argsort(d)][:g]
    per = max(48, budget_chars // max(1, len(order)))
    return " ".join(texts[int(i)][:per] for i in order)


def condense_static(x, text_emb, texts, k, encoder):
    """Stage-0 / full-history constructor (v1, all-synthetic)."""
    joint = torch.cat((F.normalize(x.float(), dim=1), text_emb), dim=1)
    assign = weighted_cluster(joint, torch.ones(len(x)), k)
    syn_x, fused, mass = [], [], []
    for j in range(int(assign.max()) + 1):
        members = torch.where(assign == j)[0]
        syn_x.append(x[members].float().mean(0))
        fused.append(fuse_new_only(joint, members, texts))
        mass.append(float(len(members)))
    syn_x = torch.stack(syn_x)
    emb = encoder.encode(fused)
    return {"x": syn_x, "text_emb": emb, "texts": fused,
            "mass": torch.tensor(mass), "depth": torch.zeros(len(syn_x), dtype=torch.long),
            "stage": torch.zeros(len(syn_x), dtype=torch.long)}


def continual_recluster(memory, delta_x, delta_emb, delta_texts, k_new, stage,
                        encoder):
    """Mass-weighted joint re-clustering with layered text re-fusion."""
    old_n = len(memory["x"])
    pts = torch.cat((
        torch.cat((F.normalize(memory["x"].float(), dim=1), memory["text_emb"]), dim=1),
        torch.cat((F.normalize(delta_x.float(), dim=1), delta_emb), dim=1)))
    mass = torch.cat((memory["mass"], torch.ones(len(delta_x))))
    joint_delta = torch.cat((F.normalize(delta_x.float(), dim=1), delta_emb), dim=1)
    assign = weighted_cluster(pts, mass, k_new)
    syn_x, fused, out_mass, out_depth, out_stage = [], [], [], [], []
    audit = {"pure_old": 0, "pure_new": 0, "mixed": 0, "old_merged_away": 0}
    for j in range(int(assign.max()) + 1):
        members = torch.where(assign == j)[0]
        olds = members[members < old_n]
        news = members[members >= old_n] - old_n
        m_old = memory["mass"][olds] if len(olds) else torch.zeros(0)
        total = float(m_old.sum() + len(news))
        # synthetic X: mass-weighted mean over old synthetic rows + new real rows
        parts, weights = [], []
        if len(olds):
            parts.append(memory["x"][olds].float())
            weights.append(m_old)
        if len(news):
            parts.append(delta_x[news].float())
            weights.append(torch.ones(len(news)))
        allx = torch.cat(parts)
        allw = torch.cat(weights)
        syn_x.append((allx * allw[:, None]).sum(0) / allw.sum())
        # layered text fusion with depth guard
        if len(olds) and not len(news):
            audit["pure_old"] += 1
            order = olds[torch.argsort(m_old, descending=True)]
            if len(order) == 1:
                fused.append(memory["texts"][int(order[0])])
                depth = int(memory["depth"][order[0]])
            else:  # old clusters merged together
                per = max(64, FUSE_CHARS // len(order))
                fused.append(" ".join(memory["texts"][int(i)][:per] for i in order))
                depth = int(memory["depth"][order].max()) + 1
                audit["old_merged_away"] += len(order) - 1
        elif len(news) and not len(olds):
            audit["pure_new"] += 1
            fused.append(fuse_new_only(joint_delta, news, delta_texts))
            depth = 0
        else:
            audit["mixed"] += 1
            main_old = olds[int(torch.argmax(m_old))]
            old_text = memory["texts"][int(main_old)]
            old_depth = int(memory["depth"][main_old])
            if old_depth >= 2:   # dilution guard: old segment budget locked
                old_seg = old_text[:FUSE_CHARS]
                new_budget = FUSE_CHARS // 2
            else:
                old_seg = old_text[:FUSE_CHARS * 2 // 3]
                new_budget = FUSE_CHARS
            new_seg = fuse_new_only(joint_delta, news, delta_texts,
                                    budget_chars=new_budget, g=FUSE_G)
            fused.append(old_seg + " " + new_seg)
            depth = old_depth + 1
        out_mass.append(total)
        out_depth.append(depth)
        out_stage.append(stage if len(news) else int(memory["stage"][olds].min()))
    syn_x = torch.stack(syn_x)
    emb = encoder.encode(fused)
    return {"x": syn_x, "text_emb": emb, "texts": fused,
            "mass": torch.tensor(out_mass),
            "depth": torch.tensor(out_depth, dtype=torch.long),
            "stage": torch.tensor(out_stage, dtype=torch.long)}, audit


def memory_to_trainable(memory):
    k = len(memory["x"])
    pairs = torch.triu_indices(k, k, offset=1)
    score = (memory["text_emb"][pairs[0]] * memory["text_emb"][pairs[1]]).sum(1)
    count = min(int(k * DEGREE), pairs.shape[1])
    chosen = pairs[:, torch.topk(score, count).indices]
    edges = torch.cat((chosen, chosen.flip(0)), 1)
    return {"x": memory["x"], "text_emb": memory["text_emb"],
            "texts": memory["texts"], "edge_index": edges, "weights": memory["mass"]}


# ------------------------------ train / eval -------------------------------

def train_student(syn: dict, seed: int, device: str, steps: int, lr: float = 1e-3,
                  batch: int = 24) -> CLIPStudent:
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
def eval_stagewise(model, xg, adj, prompt_emb, y, dev_mask, seen_stages) -> dict:
    pred = model.logits(xg, adj, prompt_emb.to(xg.device)).argmax(1).cpu()
    out = {"per_stage": []}
    all_y, all_p = [], []
    for s in range(seen_stages + 1):
        cls = torch.tensor(STAGE_CLASSES[s])
        mask = dev_mask & torch.isin(y, cls)
        yy, pp = y[mask].numpy(), pred[mask].numpy()
        out["per_stage"].append({
            "macro_f1": 100 * f1_score(yy, pp, average="macro",
                                       labels=np.unique(yy), zero_division=0),
            "accuracy": 100 * accuracy_score(yy, pp), "n": int(mask.sum())})
        all_y.append(yy)
        all_p.append(pp)
    yy, pp = np.concatenate(all_y), np.concatenate(all_p)
    out["cumulative_macro_f1"] = 100 * f1_score(yy, pp, average="macro",
                                                labels=np.unique(yy), zero_division=0)
    out["cumulative_accuracy"] = 100 * accuracy_score(yy, pp)
    if seen_stages >= 1:
        out["old_mean_f1"] = float(np.mean([r["macro_f1"] for r in out["per_stage"][:-1]]))
    out["current_f1"] = out["per_stage"][-1]["macro_f1"]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/home/pc/research/GraphCLIP/processed_data/cora.pt")
    parser.add_argument("--minilm", default="/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rate", type=float, default=0.05)
    parser.add_argument("--arms", nargs="+", default=["recluster", "append_fixed", "full_history"])
    parser.add_argument("--student-seeds", type=int, nargs="+", default=[100, 101, 102, 103, 104, 105])
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
    device = args.device
    encoder = FrozenMiniLM(args.minilm, device)
    text_emb = encoder.encode(texts)
    prompt_emb = encoder.encode(PROMPTS)
    real_adj = norm_adj(data.edge_index, n, device)
    xg = x.float().to(device)

    # per-stage train pools (dev rows excluded from condensation sources)
    stage_train = []
    for cls in STAGE_CLASSES:
        m = torch.isin(y, torch.tensor(cls)) & ~dev_mask
        stage_train.append(torch.where(m)[0])
    cum_counts = np.cumsum([len(s) for s in stage_train])
    budgets = [math.floor(args.rate * c) for c in cum_counts]
    print(json.dumps({"stage_train_counts": [len(s) for s in stage_train],
                      "budgets": budgets}), flush=True)

    def run_students(memory, seen_stages, tag) -> dict:
        syn = memory_to_trainable(memory)
        rows = []
        for seed in args.student_seeds:
            model = train_student(syn, seed, device, args.student_steps)
            rows.append(eval_stagewise(model, xg, real_adj, prompt_emb, y, dev_mask, seen_stages))
            del model
        agg = {"arm_stage": tag, "roots": len(memory["x"]),
               "depth_hist": torch.bincount(memory["depth"]).tolist(),
               "cum_dev_f1_mean": float(np.mean([r["cumulative_macro_f1"] for r in rows])),
               "cum_dev_f1_std": float(np.std([r["cumulative_macro_f1"] for r in rows])),
               "current_f1_mean": float(np.mean([r["current_f1"] for r in rows])),
               "old_mean_f1_mean": (float(np.mean([r["old_mean_f1"] for r in rows]))
                                     if seen_stages >= 1 else None),
               "per_stage_mean": [float(np.mean([r["per_stage"][s]["macro_f1"] for r in rows]))
                                  for s in range(seen_stages + 1)],
               "per_student": rows}
        print(json.dumps({kk: agg[kk] for kk in ("arm_stage", "roots", "cum_dev_f1_mean",
                                                 "current_f1_mean", "old_mean_f1_mean")}), flush=True)
        return agg

    results, audits = [], {}
    for arm in args.arms:
        memory = None
        for stage in range(3):
            ids = stage_train[stage]
            dx, de = x[ids], text_emb[ids]
            dt = [texts[int(i)] for i in ids]
            k_t = budgets[stage]
            if arm == "full_history" :
                pool = torch.cat(stage_train[: stage + 1])
                memory = condense_static(x[pool], text_emb[pool],
                                         [texts[int(i)] for i in pool], k_t, encoder)
            elif stage == 0:
                memory = condense_static(dx, de, dt, k_t, encoder)
            elif arm == "append_fixed":
                k_new = k_t - len(memory["x"])
                block = condense_static(dx, de, dt, k_new, encoder)
                block["stage"] = torch.full((len(block["x"]),), stage, dtype=torch.long)
                memory = {key: (memory[key] + block[key] if key == "texts"
                                 else torch.cat((memory[key], block[key])))
                          for key in ("x", "text_emb", "texts", "mass", "depth", "stage")}
            else:  # recluster
                memory, audit = continual_recluster(memory, dx, de, dt, k_t, stage, encoder)
                audits[f"{arm}_s{stage}"] = audit
            torch.save(memory, out / f"{arm}_s{stage}_memory.pt")
            results.append(run_students(memory, stage, f"{arm}_s{stage}"))

    json.dump({"results": results, "recluster_audits": audits, "budgets": budgets,
               "rate": args.rate, "dev_rule": "node_id % 10 == 1",
               "stage_classes": STAGE_CLASSES, "test_accessed": False,
               "student_steps": args.student_steps},
              open(out / "results.json", "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
