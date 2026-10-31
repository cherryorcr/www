"""v2 standalone validation: LLM summarization vs v1 concatenation.

Self-contained runner with all necessary components inlined.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM

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


class ClusterSummarizer:
    def __init__(self, model_path: str, device: str = "cuda:0"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map=device, torch_dtype=torch.float16, trust_remote_code=True
        ).eval()
        self.device = device

    @torch.no_grad()
    def summarize_cluster(self, member_texts: list[str], cluster_size: int,
                          budget_chars: int = 320, max_input_texts: int = 7) -> str:
        selected = member_texts[:max_input_texts]
        texts_block = "\n".join(f"{i+1}. {t[:200]}" for i, t in enumerate(selected))
        prompt = f"""You are summarizing research paper abstracts for graph condensation. Given {len(selected)} texts from a cluster of {cluster_size} papers, write ONE concise paragraph (~{budget_chars} characters) that:
1. Covers the common research theme across all texts
2. Preserves technical terms and distinguishing details
3. Reads as a coherent standalone abstract (no meta-commentary like "these papers discuss...")

INPUT TEXTS:
{texts_block}

OUTPUT (one paragraph, {budget_chars} chars):"""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(self.device)
        outputs = self.model.generate(
            **inputs, max_new_tokens=budget_chars // 3, do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        generated = self.tokenizer.decode(outputs[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        summary = generated.strip().split("\n\n")[0]
        if len(summary) > budget_chars:
            summary = summary[:budget_chars].rsplit(" ", 1)[0]
        return summary


class GCNLayer(nn.Module):
    def __init__(self, din, dout):
        super().__init__()
        self.lin = nn.Linear(din, dout)

    def forward(self, x, adj):
        return adj @ self.lin(x)


class CLIPStudent(nn.Module):
    def __init__(self, in_dim, hidden=384, out=384, text_dim=384):
        super().__init__()
        self.g1, self.g2 = GCNLayer(in_dim, hidden), GCNLayer(hidden, out)
        self.text_proj = nn.Linear(text_dim, out)
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))

    def encode_graph(self, x, adj):
        return F.normalize(self.g2(F.relu(self.g1(x, adj)), adj), dim=1)

    def encode_text(self, t):
        return F.normalize(self.text_proj(t), dim=1)

    def logits(self, x, adj, t):
        return self.logit_scale.exp().clamp(max=100) * self.encode_graph(x, adj) @ self.encode_text(t).T


def norm_adj(edge_index, n, device):
    idx = torch.cat((edge_index, torch.arange(n, device=device).repeat(2, 1)), 1)
    val = torch.ones(idx.shape[1], device=device)
    deg = torch.zeros(n, device=device).scatter_add_(0, idx[0], val)
    dinv = deg.clamp_min(1e-9).pow(-0.5)
    return torch.sparse_coo_tensor(idx, dinv[idx[0]] * val * dinv[idx[1]], (n, n)).coalesce()


def weighted_cluster(points: torch.Tensor, mass: torch.Tensor, k: int):
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


def condense_v1(x, text_emb, texts, k, encoder):
    joint = torch.cat((F.normalize(x.float(), dim=1), text_emb), dim=1)
    assign = weighted_cluster(joint, torch.ones(len(x)), k)
    syn_x, fused, mass = [], [], []
    for j in range(int(assign.max()) + 1):
        members = torch.where(assign == j)[0]
        syn_x.append(x[members].float().mean(0))
        mass.append(float(len(members)))
        center = joint[members].mean(0, keepdim=True)
        d = torch.cdist(joint[members], center).flatten()
        order = members[torch.argsort(d)][:FUSE_G]
        per = max(48, FUSE_CHARS // max(1, len(order)))
        fused.append(" ".join(texts[int(i)][:per] for i in order))
    emb = encoder.encode(fused)
    return {"x": torch.stack(syn_x), "text_emb": emb, "texts": fused,
            "mass": torch.tensor(mass)}


def condense_v2(x, text_emb, texts, k, encoder, llm_summarizer):
    joint = torch.cat((F.normalize(x.float(), dim=1), text_emb), dim=1)
    assign = weighted_cluster(joint, torch.ones(len(x)), k)
    syn_x, fused, mass = [], [], []
    for j in range(int(assign.max()) + 1):
        members = torch.where(assign == j)[0]
        syn_x.append(x[members].float().mean(0))
        mass.append(float(len(members)))
        center = joint[members].mean(0, keepdim=True)
        d = torch.cdist(joint[members], center).flatten()
        order = members[torch.argsort(d)][:FUSE_G]
        member_texts = [texts[int(i)] for i in order]
        summary = llm_summarizer.summarize_cluster(
            member_texts, cluster_size=len(members),
            budget_chars=FUSE_CHARS, max_input_texts=FUSE_G
        )
        fused.append(summary)
    emb = encoder.encode(fused)
    return {"x": torch.stack(syn_x), "text_emb": emb, "texts": fused,
            "mass": torch.tensor(mass)}


def memory_to_trainable(memory):
    k = len(memory["x"])
    pairs = torch.triu_indices(k, k, offset=1)
    score = (memory["text_emb"][pairs[0]] * memory["text_emb"][pairs[1]]).sum(1)
    count = min(int(k * DEGREE), pairs.shape[1])
    chosen = pairs[:, torch.topk(score, count).indices]
    edges = torch.cat((chosen, chosen.flip(0)), 1)
    return {"x": memory["x"], "text_emb": memory["text_emb"],
            "texts": memory["texts"], "edge_index": edges}


def train_student(syn: dict, seed: int, device: str, steps: int, lr: float = 1e-3, batch: int = 24):
    torch.manual_seed(seed)
    model = CLIPStudent(syn["x"].shape[1]).to(device)
    x, t = syn["x"].to(device), syn["text_emb"].to(device)
    adj = norm_adj(syn["edge_index"], len(x), device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed * 7 + 1)
    n = len(x)
    for _ in range(steps):
        rows = torch.randperm(n, generator=gen)[:min(batch, n)].to(device)
        logits = model.logits(x, adj, t[rows])
        target = torch.arange(len(rows), device=device)
        loss = (F.cross_entropy(logits[rows], target) + F.cross_entropy(logits.T, target)) / 2
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/home/pc/research/GraphCLIP/processed_data/cora.pt")
    parser.add_argument("--minilm", default="/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip")
    parser.add_argument("--qwen", default="/home/pc/models/Qwen3-32B-AWQ")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--student-seeds", type=int, nargs="+", default=[100, 101, 102])
    parser.add_argument("--student-steps", type=int, default=2048)
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "results.json").exists():
        raise FileExistsError(f"{out}/results.json exists")

    data = torch.load(args.data, map_location="cpu", weights_only=False)
    x, y, texts = data.x, data.y, list(data.raw_texts)
    n = len(x)
    dev_mask = (torch.arange(n) % 10) == 1
    device = args.device

    print("Loading MiniLM...", flush=True)
    encoder = FrozenMiniLM(args.minilm, device)
    text_emb = encoder.encode(texts)
    prompt_emb = encoder.encode(PROMPTS)

    print(f"Loading Qwen3...", flush=True)
    llm_summarizer = ClusterSummarizer(args.qwen, device=device)

    real_adj = norm_adj(data.edge_index, n, device)
    xg = x.float().to(device)

    stage2_cls = torch.tensor(STAGE_CLASSES[2])
    stage2_train = torch.where(torch.isin(y, stage2_cls) & ~dev_mask)[0]
    k = 121
    print(json.dumps({"stage2_nodes": len(stage2_train), "budget": k}), flush=True)

    dx, de, dt = x[stage2_train], text_emb[stage2_train], [texts[int(i)] for i in stage2_train]

    results = []
    for variant, fn in [("v1_concat", condense_v1),
                        ("v2_llm", lambda *a: condense_v2(*a, llm_summarizer))]:
        print(f"\n=== {variant} ===", flush=True)
        memory = fn(dx, de, dt, k, encoder)
        torch.save(memory, out / f"{variant}_memory.pt")

        memory2 = fn(dx, de, dt, k, encoder)
        for key in ("x", "text_emb", "mass"):
            if not torch.allclose(memory[key], memory2[key], atol=1e-6):
                raise RuntimeError(f"{variant} non-deterministic")
        print(f"  Determinism: PASS", flush=True)

        syn = memory_to_trainable(memory)
        per_student = []
        for seed in args.student_seeds:
            model = train_student(syn, seed, device, args.student_steps)
            metrics = eval_stagewise(model, xg, real_adj, prompt_emb, y, dev_mask, 2)
            per_student.append(metrics)
            print(f"  seed={seed} cum={metrics['cumulative_macro_f1']:.2f}", flush=True)
            del model

        agg = {
            "variant": variant, "roots": len(memory["x"]),
            "cum_f1_mean": float(np.mean([r["cumulative_macro_f1"] for r in per_student])),
            "cum_f1_std": float(np.std([r["cumulative_macro_f1"] for r in per_student])),
            "per_student": per_student
        }
        results.append(agg)
        print(json.dumps({k: agg[k] for k in ("variant", "cum_f1_mean", "cum_f1_std")}), flush=True)

    json.dump({"results": results, "budget": k, "steps": args.student_steps},
              open(out / "results.json", "w"), indent=1)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
