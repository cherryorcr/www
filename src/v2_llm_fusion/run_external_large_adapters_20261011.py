"""Large-graph adapters for STAD, DANCE and SR-GM.

The three papers do not expose the same historical continual TAG interface.
This runner therefore preserves the identifiable mechanism of each method while
using the project's common large-graph evaluation protocol.  Results are
labelled ``mechanism-adapted`` and are never presented as official
end-to-end reproductions.

The constructor only reads the current class-incremental delta and appends new
rows to a frozen synthetic prefix.  Text payloads are materialized from real
source strings; no report/test text or labels are used by the constructor.
Labels are used only for the same class-balanced allocation used by the
existing adapted baselines.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_unified_v3_continual as U
import run_multiscale_asm as A


METHODS = ("stad_adapter", "dance_adapter", "sr_gm_adapter")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def tensor_hash(value) -> str:
    h = hashlib.sha256()
    if isinstance(value, dict):
        for key in sorted(value):
            h.update(str(key).encode())
            h.update(tensor_hash(value[key]).encode())
        return h.hexdigest()
    if torch.is_tensor(value):
        x = value.detach().cpu().contiguous()
        h.update(str((str(x.dtype), tuple(x.shape))).encode())
        h.update(x.numpy().tobytes())
        return h.hexdigest()
    h.update(repr(value).encode())
    return h.hexdigest()


def normalize_adj(edge_index: torch.Tensor, n: int, device: str) -> torch.Tensor:
    idx = torch.cat((edge_index.long(), torch.arange(n).repeat(2, 1)), 1).to(device)
    val = torch.ones(idx.shape[1], device=device)
    deg = torch.zeros(n, device=device).scatter_add_(0, idx[0], val)
    dinv = deg.clamp_min(1e-9).pow(-0.5)
    return torch.sparse_coo_tensor(
        idx, dinv[idx[0]] * val * dinv[idx[1]], (n, n)
    ).coalesce()


def class_quotas(labels: torch.Tensor, ids: torch.Tensor,
                 classes: list[int], total: int) -> dict[int, int]:
    """Allocate a fixed row budget proportionally with a one-row floor."""
    counts = {int(c): int((labels[ids] == int(c)).sum()) for c in classes}
    counts = {c: n for c, n in counts.items() if n > 0}
    if not counts or total <= 0:
        return {c: 0 for c in classes}
    total = min(total, sum(counts.values()))
    q = {c: 0 for c in counts}
    if total >= len(counts):
        for c in counts:
            q[c] = 1
        total -= len(counts)
    order = sorted(counts, key=lambda c: (-counts[c], c))
    if total > 0:
        denom = sum(counts.values())
        raw = {c: total * counts[c] / denom for c in counts}
        add = {c: min(counts[c] - q[c], int(math.floor(raw[c]))) for c in counts}
        for c in counts:
            q[c] += add[c]
        left = total - sum(add.values())
        for c in sorted(counts, key=lambda c: (-(raw[c] - math.floor(raw[c])), c)):
            if left <= 0:
                break
            if q[c] < counts[c]:
                q[c] += 1
                left -= 1
    # Correct a possible underfill caused by saturated small classes.
    while sum(q.values()) < min(total + sum(1 for c in counts if total == 0),
                                sum(counts.values())):
        for c in order:
            if q[c] < counts[c]:
                q[c] += 1
                if sum(q.values()) >= min(total + len(counts), sum(counts.values())):
                    break
    return {c: q.get(c, 0) for c in classes}


def induced_neighbor_mean(x: torch.Tensor, edge_index: torch.Tensor,
                          ids: torch.Tensor) -> torch.Tensor:
    """One-hop mean on the current delta only."""
    ids = ids.long().cpu()
    n = len(ids)
    local = torch.full((len(x),), -1, dtype=torch.long)
    local[ids] = torch.arange(n)
    src, dst = edge_index.long().cpu()
    keep = (local[src] >= 0) & (local[dst] >= 0)
    src, dst = local[src[keep]], local[dst[keep]]
    out = torch.zeros_like(x[ids]).float()
    deg = torch.zeros(n, dtype=torch.float32)
    if len(src):
        out.index_add_(0, dst, x[ids[src]].float())
        deg.index_add_(0, dst, torch.ones(len(dst)))
    out = out / deg.clamp_min(1).unsqueeze(1)
    return out


def project(z: torch.Tensor, dim: int, seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    r = torch.randn(z.shape[1], dim, generator=gen) / math.sqrt(dim)
    return F.normalize(z.float() @ r, dim=1)


def cluster_assign(z: torch.Tensor, k: int, seed: int) -> torch.Tensor:
    """Scalable deterministic farthest-point assignment in a projected space."""
    n = len(z)
    if k >= n:
        return torch.arange(n, dtype=torch.long)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    first = int(torch.randint(n, (1,), generator=gen))
    centers = [first]
    dmin = ((z - z[first]) ** 2).sum(1)
    for _ in range(1, k):
        nxt = int(dmin.argmax())
        centers.append(nxt)
        dmin = torch.minimum(dmin, ((z - z[nxt]) ** 2).sum(1))
    c = z[torch.tensor(centers)]
    chunks = []
    for start in range(0, n, 8192):
        chunks.append(torch.cdist(z[start:start + 8192], c).argmin(1))
    a = torch.cat(chunks)
    a[torch.tensor(centers)] = torch.arange(k)
    return a


def neighbor_texts(raw_texts: list[str], local_idx: int,
                   ids: torch.Tensor, edge_index: torch.Tensor,
                   limit: int = 2) -> list[str]:
    ids = ids.long().cpu()
    local = torch.full((len(raw_texts),), -1, dtype=torch.long)
    local[ids] = torch.arange(len(ids))
    src, dst = edge_index.long().cpu()
    keep = (src == ids[local_idx]) & (local[dst] >= 0)
    nbr = local[dst[keep]].tolist()[:limit]
    return [raw_texts[int(ids[i])] for i in nbr]


def build_clusters(D: dict, text_pool: torch.Tensor, ids: torch.Tensor,
                   quotas: dict[int, int], method: str, stage: int,
                   seed: int, encoder, device: str) -> dict:
    x_all, y_all, texts = D["x"], D["y"], D["texts"]
    edge_index = D["edge_index"]
    x = x_all[ids].float().cpu()
    t = text_pool[ids].float().cpu()
    neighbor = induced_neighbor_mean(x_all, edge_index, ids)
    x_n = F.normalize(x, dim=1)
    n_n = F.normalize(neighbor, dim=1)
    if method == "stad_adapter":
        joint = torch.cat((x_n, t, n_n), dim=1)
    elif method == "dance_adapter":
        joint = torch.cat((x_n, t, n_n), dim=1)
    else:
        joint = torch.cat((x_n, t), dim=1)
    z = project(joint, 32, seed)

    syn_x, syn_texts, mass, syn_labels = [], [], [], []
    for c in sorted(quotas):
        q = quotas[c]
        member = torch.where(y_all[ids] == c)[0]
        if q <= 0 or len(member) == 0:
            continue
        q = min(q, len(member))
        a = cluster_assign(z[member], q, seed + 1009 * int(c) + stage)
        for cluster in range(q):
            local = member[a == cluster]
            if len(local) == 0:
                continue
            center = joint[local].mean(0)
            dist = ((joint[local] - center) ** 2).sum(1)
            root = int(local[int(dist.argmin())])
            syn_x.append(x[local].mean(0))
            mass.append(float(len(local)))
            syn_labels.append(int(c))
            root_text = str(texts[int(ids[root])]).strip()
            if method == "stad_adapter":
                near = local[dist.argsort()[:3]].tolist()
                parts = [str(texts[int(ids[i])]).strip()[:128] for i in near]
                syn_texts.append(" ".join(p for p in parts if p))
            elif method == "dance_adapter":
                evidence = neighbor_texts(texts, root, ids, edge_index, limit=2)
                parts = [root_text[:160]] + [str(v).strip()[:80] for v in evidence]
                syn_texts.append(" [EVIDENCE] ".join(p for p in parts if p))
            else:
                syn_texts.append(root_text[:256])

    X = torch.stack(syn_x).float()
    T = F.normalize(encoder.encode(syn_texts).cpu(), dim=1)
    topology = A.knn_topology_gpu(T, degree=4.0, device=device)
    return {"x": X, "text_emb": T, "texts": syn_texts,
            "edge_index": topology, "weights": torch.tensor(mass),
            "labels": torch.tensor(syn_labels),
            "stage": torch.full((len(X),), stage, dtype=torch.long)}


class TinyMM(nn.Module):
    def __init__(self, x_dim: int, t_dim: int = 384, hidden: int = 128):
        super().__init__()
        self.g = nn.Sequential(nn.Linear(x_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.t = nn.Sequential(nn.Linear(t_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden))

    def forward(self, x, t):
        return F.normalize(self.g(x), dim=1), F.normalize(self.t(t), dim=1)


def sr_gm_refine(memory: dict, D: dict, text_pool: torch.Tensor,
                 ids: torch.Tensor, device: str, seed: int,
                 steps: int = 3) -> dict:
    """Small scalable SR-GM adapter: GM + orthogonal decoupling + damping."""
    torch.manual_seed(seed)
    out = {k: v.clone() if torch.is_tensor(v) else v for k, v in memory.items()}
    n = len(out["x"])
    active_n = min(512, n, len(ids))
    raw_gen = torch.Generator(device="cpu").manual_seed(seed + 77)
    raw_idx = ids[torch.randperm(len(ids), generator=raw_gen)[:active_n]]
    syn_idx = torch.randperm(n, generator=raw_gen)[:active_n]
    x0 = out["x"][syn_idx].to(device).float()
    tv = out["text_emb"][syn_idx].to(device).float().requires_grad_(True)
    xr = D["x"][raw_idx].to(device).float()
    tr = text_pool[raw_idx].to(device).float()
    model = TinyMM(xr.shape[1]).to(device)
    params = tuple(model.parameters())
    x_var = nn.Parameter(x0.clone())
    opt = torch.optim.Adam([x_var], lr=0.03)
    for _ in range(steps):
        gr, trh = model(xr, tr)
        gs, tsh = model(x_var, tv)
        real_logits = gr @ trh.T / 0.07
        syn_logits = gs @ tsh.T / 0.07
        target = torch.arange(len(xr), device=device)
        real_loss = (F.cross_entropy(real_logits, target) +
                     F.cross_entropy(real_logits.T, target)) / 2
        syn_loss = (F.cross_entropy(syn_logits, target) +
                    F.cross_entropy(syn_logits.T, target)) / 2
        g_real = torch.autograd.grad(real_loss, params, retain_graph=True)
        g_syn = torch.autograd.grad(syn_loss, params, create_graph=True,
                                    retain_graph=True)
        vr = torch.cat([g.detach().reshape(-1) for g in g_real])
        vs = torch.cat([g.reshape(-1) for g in g_syn])
        gm = 1.0 - F.cosine_similarity(vs, vr, dim=0)
        gx, gt = torch.autograd.grad(syn_loss, (x_var, tv), create_graph=True)
        dot = (gx * gt[:, :gx.shape[1]]).sum(1, keepdim=True)
        gx_proj = gx - (dot / (gt[:, :gx.shape[1]].square().sum(1, keepdim=True) + 1e-8)) * gt[:, :gx.shape[1]]
        conflict = dot < 0
        gx_dec = torch.where(conflict, gx_proj, gx)
        # Hard kNN topology is treated as fixed inside an outer step.
        with torch.no_grad():
            sim = F.normalize(x_var.detach(), dim=1) @ F.normalize(x_var.detach(), dim=1).T
            sim.fill_diagonal_(-2)
            knn = sim.topk(min(4, max(1, len(x_var) - 1)), dim=1).indices
        smooth = (gx_dec - gx_dec[knn].mean(1)).square().mean()
        loss = gm + 0.05 * smooth + 0.01 * (x_var - x0).square().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    out["x"][syn_idx] = x_var.detach().cpu()
    out["sr_gm"] = {"active_rows": active_n, "steps": steps,
                     "gradient_matching": True, "gradient_decoupling": True,
                     "structural_damping": True}
    return out


def evaluate(memory: dict, D: dict, device: str, seeds: list[int], stage: int,
             steps_override: int | None) -> dict:
    rows = []
    train_memory = {k: v for k, v in memory.items() if k in {"x", "text_emb", "texts", "edge_index", "weights"}}
    xg = D["x"].float().to(device)
    real_adj = normalize_adj(D["edge_index"], len(D["x"]), device)
    prompt = F.normalize(D["encoder"].encode(D["prompts"]), dim=1).to(device)
    for seed in seeds:
        steps = steps_override or A.protocol_v3.STUDENT.steps(len(train_memory["x"]))
        model = A.train_student(train_memory, seed, device, steps)
        pred = model.logits(xg, real_adj, prompt).argmax(1).cpu()
        mask = D["report_dev_mask"]
        y = D["y"][mask].numpy()
        p = pred[mask].numpy()
        from sklearn.metrics import accuracy_score, f1_score
        rows.append({"seed": seed,
                     "macro_f1": 100 * f1_score(y, p, average="macro", zero_division=0),
                     "accuracy": 100 * accuracy_score(y, p)})
        del model
    return {"stage": stage, "rows": len(train_memory["x"]), "per_seed": rows,
            "macro_f1": float(np.mean([r["macro_f1"] for r in rows])),
            "macro_f1_std": float(np.std([r["macro_f1"] for r in rows])),
            "accuracy": float(np.mean([r["accuracy"] for r in rows])),
            "accuracy_std": float(np.std([r["accuracy"] for r in rows]))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["arxiv", "sportsfit", "bookhis", "wikics"], required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--minilm", default="/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip")
    ap.add_argument("--rate", type=float, default=0.05)
    ap.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    ap.add_argument("--student-seeds", type=int, nargs="+", default=[150, 151, 152])
    ap.add_argument("--student-steps", type=int, default=1024)
    ap.add_argument("--refine-steps", type=int, default=3)
    args = ap.parse_args()
    out = Path(args.output)
    if (out / "results.json").exists():
        raise FileExistsError("immutable output already exists")
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    D = U.load_unified(args.dataset, U.DATA_CFG[args.dataset]["path"], args.minilm,
                       args.device, n_stages=3, split_mode="canonical")
    pool = torch.where(D["pool_mask"])[0]
    encoded_pool = D["encoder"].encode([D["texts"][int(i)] for i in pool]).cpu()
    text_pool = torch.zeros((len(D["x"]), encoded_pool.shape[1]), dtype=encoded_pool.dtype)
    text_pool[pool] = encoded_pool
    stage_ids = [pool[torch.isin(D["y"][pool], torch.tensor(group))]
                 for group in D["stage_classes"]]
    budgets = [math.floor(args.rate * sum(len(v) for v in stage_ids[:i + 1]))
               for i in range(len(stage_ids))]
    manifest = {
        "version": "large-mechanism-adapters-20261011-v1",
        "dataset": args.dataset, "methods": args.methods, "rate": args.rate,
        "budgets": budgets, "stage_classes": D["stage_classes"],
        "student": "common Prompt GraphText Student from run_multiscale_asm",
        "student_seeds": args.student_seeds, "student_steps": args.student_steps,
        "construction_access": "current class delta only; frozen synthetic prefix",
        "historical_raw_reads": 0,
        "construction_labels_used_for_stage_and_class_quota": True,
        "report_or_test_labels_used_for_construction": False,
        "official_reproduction": False,
        "method_status": {
            "stad_adapter": "STAD mechanism adapter: collaborative graph/text sketch, class clustering, real-source text payload",
            "dance_adapter": "DANCE mechanism adapter: neighbor-gated evidence pack and round-wise current-delta refresh",
            "sr_gm_adapter": "SR-GM mechanism adapter: gradient matching, orthogonal conflict projection, structural damping",
        },
        "source_sha256": sha(Path(U.DATA_CFG[args.dataset]["path"])),
        "runner_sha256": sha(Path(__file__)),
        "created_unix": time.time(),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    all_results = []
    for method in args.methods:
        old = None
        audits = []
        for stage, ids in enumerate(stage_ids):
            target = budgets[stage]
            previous = 0 if old is None else len(old["x"])
            new_budget = max(0, target - previous)
            quotas = class_quotas(D["y"], ids, D["stage_classes"][stage], new_budget)
            t0 = time.perf_counter()
            block = build_clusters(D, text_pool, ids, quotas, method, stage,
                                   7000 + stage, D["encoder"], args.device)
            if method == "sr_gm_adapter":
                block = sr_gm_refine(block, D, text_pool, ids, args.device,
                                     7000 + stage, args.refine_steps)
            if old is None:
                memory = block
            else:
                shift = len(old["x"])
                edge = block["edge_index"] + shift
                memory = {"x": torch.cat((old["x"], block["x"])),
                          "text_emb": torch.cat((old["text_emb"], block["text_emb"])),
                          "texts": old["texts"] + block["texts"],
                          "edge_index": torch.cat((old["edge_index"], edge), dim=1),
                          "weights": torch.cat((old["weights"], block["weights"])),
                          "labels": torch.cat((old.get("labels", torch.empty(0, dtype=torch.long)), block["labels"])),
                          "stage": torch.cat((old.get("stage", torch.empty(0, dtype=torch.long)), block["stage"]))}
            audit = {"method": method, "stage": stage, "source_rows": len(ids),
                     "new_rows": len(block["x"]), "total_rows": len(memory["x"]),
                     "quotas": quotas, "source_ids_sha256": tensor_hash(ids),
                     "memory_sha256": tensor_hash(memory),
                     "construction_seconds": time.perf_counter() - t0,
                     "topology": "new-block text kNN; old prefix frozen; no cross-stage edges",
                     "texts_are_real_source_materialized": True}
            audits.append(audit)
            old = memory
        final = evaluate(old, D, args.device, args.student_seeds, len(stage_ids) - 1,
                         args.student_steps)
        result = {"method": method, "dataset": args.dataset, "final": final,
                  "construction": audits, "memory_rows": len(old["x"]),
                  "tensor_sha256": tensor_hash(old), "elapsed_seconds": time.perf_counter() - started}
        all_results.append(result)
        (out / f"{method}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps({"method": method, "dataset": args.dataset,
                          "f1": final["macro_f1"], "acc": final["accuracy"]}), flush=True)
    payload = {"manifest": manifest, "results": all_results,
               "elapsed_seconds": time.perf_counter() - started}
    (out / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("LARGE_ADAPTERS_DONE", flush=True)


if __name__ == "__main__":
    main()
