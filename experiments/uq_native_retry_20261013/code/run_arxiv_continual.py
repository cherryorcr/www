"""ogbn-arxiv part 2: raw upper reference, 10% rate, and CONTINUAL (temporal3).

Static additions (full-graph pool ~135k rows, dev/test rows excluded):
  raw_full        all pool rows real X/text/edges, direct training (steps capped 8192)
  construct_10pct frozen_v1 constructor at rate=0.10

Continual (temporal 3-stage by publication year, pre-built splits):
  delta_t1/t2/t3 (~56k nodes each), cumulative_t1/t2/t3 as evaluation graphs.
  Budgets k_t = floor(rate * cumulative pool size), rate=0.05.
  Arms:
    append        M_t = M_{t-1} (+) construct(delta_t pool, k_t - |M_{t-1}|)   [ours]
    full_history  re-condense cumulative pool to k_t each stage (violating ref)
    raw           cumulative pool direct training each stage (violating ref)
  Eval at stage t: transductive on cumulative_t graph, dev = orig_id%10==1,
  per-stage split by arrival_stage; cumulative/old/current Macro-F1.
  NOTE: temporal setting = all 40 classes present in every stage (unlike Cora
  class-incremental); "old/current" measures stage-arrival cohorts.

Text embeddings: encode all 169k texts once, index by original_node_ids.
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

DATA_DIR = Path("/home/pc/research/TAGSAM/data")
FULL = DATA_DIR / "ogbn_arxiv_full_minilm_semantic.pt"


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

    def logits(self, x, adj, t):
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


def construct(x, text_emb, texts, k, encoder, device,
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
    mass = torch.tensor(weights)
    return {"x": syn_x, "text_emb": fused_emb, "texts": fused,
            "edge_index": edges, "weights": mass, "mass": mass,
            "depth": torch.zeros(len(syn_x), dtype=torch.long),
            "stage": torch.zeros(len(syn_x), dtype=torch.long)}


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
def eval_stagewise(model, eval_x, eval_adj, prompt_emb, y_eval, dev_mask, stage_arr, seen):
    pred = model.logits(eval_x, eval_adj, prompt_emb.to(eval_x.device)).argmax(1).cpu()
    out = {"per_stage": []}
    all_y, all_p = [], []
    for s in range(seen + 1):
        mask = dev_mask & (stage_arr == s)
        yy, pp = y_eval[mask].numpy(), pred[mask].numpy()
        out["per_stage"].append(
            {"macro_f1": 100 * f1_score(yy, pp, average="macro", labels=np.unique(yy),
                                        zero_division=0),
             "n": int(mask.sum())})
        all_y.append(yy)
        all_p.append(pp)
    yy, pp = np.concatenate(all_y), np.concatenate(all_p)
    out["cumulative_macro_f1"] = 100 * f1_score(yy, pp, average="macro",
                                                labels=np.unique(yy), zero_division=0)
    out["cumulative_accuracy"] = 100 * accuracy_score(yy, pp)
    if seen >= 1:
        out["old_mean_f1"] = float(np.mean([r["macro_f1"] for r in out["per_stage"][:-1]]))
    out["current_f1"] = out["per_stage"][-1]["macro_f1"]
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--minilm", default="/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rate", type=float, default=0.05)
    parser.add_argument("--student-seeds", type=int, nargs="+",
                        default=list(protocol_v3.STUDENT_SEEDS))
    parser.add_argument("--student-batch", type=int,
                        default=protocol_v3.STUDENT.batch)
    parser.add_argument("--student-steps", type=int, default=None,
                        help="fixed steps; otherwise protocol-scaled")
    parser.add_argument("--raw-steps-cap", type=int, default=protocol_v3.RAW_STEPS_CAP)
    parser.add_argument("--split-prefix", default="ogbn_arxiv_full_temporal3",
                        help="prepared split prefix under DATA_DIR")
    parser.add_argument("--skip-static", action="store_true")
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "results.json").exists():
        raise FileExistsError("existing runs are immutable")
    device = args.device

    # ---------- load full graph, encode all texts once ----------
    d = torch.load(FULL, map_location="cpu", weights_only=False)
    x_full = d["node_f"]
    texts_full = list(d["text_list"])
    all_labels = [str(v) for v in np.asarray(d["all_labels"]).tolist()]
    lab2idx = {name: i for i, name in enumerate(all_labels)}
    y_full = torch.tensor([lab2idx[str(v)] for v in np.asarray(d["label_list"]).tolist()],
                          dtype=torch.long)
    labels_desc = list(d["labels_desc"])
    n = len(x_full)

    encoder = FrozenMiniLM(args.minilm, device)

    def clean(name):
        return name.replace("arXiv computer science category: ", "").replace(
            "arXiv computer science category ", "")
    prompts = [f"a computer science paper about {clean(nm).lower()}" for nm in labels_desc]
    prompt_emb = encoder.encode(prompts)

    emb_cache = out / "text_emb_full.pt"
    t0 = time.time()
    text_emb_full = encoder.encode(texts_full)
    torch.save(text_emb_full, emb_cache)
    print(f"encoded {n} texts in {time.time()-t0:.0f}s", flush=True)

    dev_full, _, pool_full = protocol_v3.split_masks(torch.arange(n))

    results = {"static": [], "continual": []}

    # ================= static additions =================
    if not args.skip_static:
        real_adj = norm_adj(d["edge_index"], n, device)
        xg = x_full.float().to(device)

        # raw upper: all pool rows, real everything
        pool_idx = torch.where(pool_full)[0]
        remap = torch.full((n,), -1, dtype=torch.long)
        remap[pool_idx] = torch.arange(len(pool_idx))
        e = d["edge_index"]
        keep = (remap[e[0]] >= 0) & (remap[e[1]] >= 0)
        raw_syn = {"x": x_full[pool_idx].float(), "text_emb": text_emb_full[pool_idx],
                   "texts": [texts_full[int(i)] for i in pool_idx],
                   "edge_index": torch.stack((remap[e[0, keep]], remap[e[1, keep]])),
                   "weights": torch.ones(len(pool_idx))}
        steps = args.raw_steps_cap
        t0 = time.time()
        scores = []
        for seed in args.student_seeds:
            model = train_student(raw_syn, seed, device, steps,
                                  batch=args.student_batch)
            pred = model.logits(xg, real_adj, prompt_emb.to(device)).argmax(1).cpu()
            yy, pp = y_full[dev_full].numpy(), pred[dev_full].numpy()
            scores.append({"macro_f1": 100 * f1_score(yy, pp, average="macro", zero_division=0),
                           "accuracy": 100 * accuracy_score(yy, pp)})
            del model
        f1s = [s["macro_f1"] for s in scores]
        results["static"].append({"arm": "raw_full_pool", "k": len(pool_idx), "steps": steps,
                                  "dev_macro_f1_mean": float(np.mean(f1s)),
                                  "dev_macro_f1_std": float(np.std(f1s)),
                                  "seconds": time.time() - t0, "per_student": scores})
        print(json.dumps(results["static"][-1] | {"per_student": None}), flush=True)

        # 10% construct
        k10 = math.floor(0.10 * len(pool_idx))
        t0 = time.time()
        syn10 = construct(x_full[pool_idx], text_emb_full[pool_idx],
                          [texts_full[int(i)] for i in pool_idx], k10, encoder, device)
        cond_s = time.time() - t0
        scores = []
        for seed in args.student_seeds:
            model = train_student(syn10, seed, device,
                                  protocol_v3.student_steps(k10, override=args.student_steps),
                                  batch=args.student_batch)
            pred = model.logits(xg, real_adj, prompt_emb.to(device)).argmax(1).cpu()
            yy, pp = y_full[dev_full].numpy(), pred[dev_full].numpy()
            scores.append({"macro_f1": 100 * f1_score(yy, pp, average="macro", zero_division=0),
                           "accuracy": 100 * accuracy_score(yy, pp)})
            del model
        f1s = [s["macro_f1"] for s in scores]
        results["static"].append({"arm": "construct_10pct", "k": k10,
                                  "dev_macro_f1_mean": float(np.mean(f1s)),
                                  "dev_macro_f1_std": float(np.std(f1s)),
                                  "condense_seconds": cond_s, "per_student": scores})
        print(json.dumps(results["static"][-1] | {"per_student": None}), flush=True)
        del real_adj, xg
        torch.cuda.empty_cache()

    # ================= continual temporal3 =================
    deltas, cums = [], []
    for t in (1, 2, 3):
        deltas.append(torch.load(DATA_DIR / f"{args.split_prefix}_delta_t{t}.pt",
                                 map_location="cpu", weights_only=False))
        cums.append(torch.load(DATA_DIR / f"{args.split_prefix}_cumulative_t{t}.pt",
                                map_location="cpu", weights_only=False))

    def pool_rows(dd):
        oid = dd["original_node_ids"]
        _, _, m = protocol_v3.split_masks(oid)
        return torch.where(m)[0]

    pools = [pool_rows(dd) for dd in deltas]
    cum_pool_sizes = np.cumsum([len(p) for p in pools])
    budgets = [math.floor(args.rate * c) for c in cum_pool_sizes]
    print(f"pools={[len(p) for p in pools]} budgets={budgets}", flush=True)

    def delta_view(i):
        dd = deltas[i]
        rows = pools[i]
        oid = dd["original_node_ids"][rows]
        return (dd["node_f"][rows], text_emb_full[oid],
                [texts_full[int(v)] for v in oid])

    def eval_setup(i):
        cc = cums[i]
        oid = cc["original_node_ids"]
        y_eval = y_full[oid]
        dev_mask, _, _ = protocol_v3.split_masks(oid)
        stage_arr = cc["arrival_stage"].clone()
        stage_arr = stage_arr - stage_arr.min()  # normalize to 0-based
        adj = norm_adj(cc["edge_index"], len(oid), device)
        xg = cc["node_f"].float().to(device)
        return xg, adj, y_eval, dev_mask, stage_arr

    eval_cache = [eval_setup(i) for i in range(3)]

    def run_students(syn, stage_i, tag):
        steps = protocol_v3.student_steps(len(syn["x"]), override=args.student_steps)
        xg, adj, y_eval, dev_mask, stage_arr = eval_cache[stage_i]
        rows = []
        for seed in args.student_seeds:
            model = train_student(syn, seed, device, steps,
                                  batch=args.student_batch)
            rows.append(eval_stagewise(model, xg, adj, prompt_emb, y_eval,
                                       dev_mask, stage_arr, stage_i))
            del model
        agg = {"arm_stage": tag, "roots": len(syn["x"]), "steps": steps,
               "student_steps": steps, "student_batch": args.student_batch,
               "student_seeds": list(args.student_seeds),
               "cum_mean": float(np.mean([r["cumulative_macro_f1"] for r in rows])),
               "cum_std": float(np.std([r["cumulative_macro_f1"] for r in rows])),
               "cur_mean": float(np.mean([r["current_f1"] for r in rows])),
               "old_mean": (float(np.mean([r["old_mean_f1"] for r in rows]))
                            if stage_i >= 1 else None),
               "per_student": rows}
        results["continual"].append(agg)
        print(json.dumps({k: agg[k] for k in ("arm_stage", "roots", "cum_mean",
                                              "old_mean", "cur_mean")}), flush=True)

    # ---- append (ours) ----
    memory = None
    for i in range(3):
        dx, de, dt = delta_view(i)
        k_t = budgets[i]
        old_rows = 0 if memory is None else len(memory["x"])
        seen_labels = set()
        for prior in range(i):
            seen_labels.update(str(v) for v in deltas[prior]["label_list"])
        new_class_count = len({str(v) for v in deltas[i]["label_list"]} - seen_labels)
        keep_rows, k_new = protocol_v3.append_quota(k_t, old_rows, new_class_count)
        if memory is None:
            keep_rows = 0
            k_new = k_t
        block = construct(dx, de, dt, k_new, encoder, device)
        block["stage"] = torch.full((len(block["x"]),), i, dtype=torch.long)
        if memory is None:
            memory = block
        else:
            order = torch.argsort(memory["weights"], descending=True, stable=True)[:keep_rows]
            order = torch.sort(order).values
            old_part = {"x": memory["x"][order],
                        "text_emb": memory["text_emb"][order],
                        "texts": [memory["texts"][int(i)] for i in order],
                        "weights": memory["weights"][order],
                        "mass": memory["mass"][order],
                        "depth": memory["depth"][order],
                        "stage": memory["stage"][order]}
            merged = {key: (old_part[key] + block[key] if key == "texts"
                             else torch.cat((old_part[key], block[key])))
                      for key in ("x", "text_emb", "texts", "weights", "mass", "depth", "stage")}
            merged["edge_index"] = knn_topology_gpu(
                merged["text_emb"], protocol_v3.DEGREE, device)
            memory = merged
        torch.save({k: memory[k] for k in memory}, out / f"append_s{i}_memory.pt")
        run_students(memory, i, f"append_s{i}")

    # ---- full_history recondense ----
    for i in range(3):
        rows_all, embs_all, txts_all = [], [], []
        for j in range(i + 1):
            dx, de, dt = delta_view(j)
            rows_all.append(dx)
            embs_all.append(de)
            txts_all.extend(dt)
        syn = construct(torch.cat(rows_all), torch.cat(embs_all), txts_all,
                        budgets[i], encoder, device)
        run_students(syn, i, f"full_history_s{i}")

    # ---- raw per stage ----
    for i in range(3):
        cc = cums[i]
        oid = cc["original_node_ids"]
        _, _, m = protocol_v3.split_masks(oid)
        rows = torch.where(m)[0]
        remap = torch.full((len(oid),), -1, dtype=torch.long)
        remap[rows] = torch.arange(len(rows))
        e = cc["edge_index"]
        keep = (remap[e[0]] >= 0) & (remap[e[1]] >= 0)
        syn = {"x": cc["node_f"][rows].float(), "text_emb": text_emb_full[oid[rows]],
               "texts": [texts_full[int(v)] for v in oid[rows]],
               "edge_index": torch.stack((remap[e[0, keep]], remap[e[1, keep]])),
               "weights": torch.ones(len(rows))}
        steps = protocol_v3.raw_steps(override=args.raw_steps_cap)
        xg, adj, y_eval, dev_mask, stage_arr = eval_cache[i]
        rws = []
        for seed in args.student_seeds:
            model = train_student(syn, seed, device, steps,
                                  batch=args.student_batch)
            rws.append(eval_stagewise(model, xg, adj, prompt_emb, y_eval,
                                      dev_mask, stage_arr, i))
            del model
        agg = {"arm_stage": f"raw_s{i}", "roots": len(rows), "steps": steps,
               "cum_mean": float(np.mean([r["cumulative_macro_f1"] for r in rws])),
               "cum_std": float(np.std([r["cumulative_macro_f1"] for r in rws])),
               "cur_mean": float(np.mean([r["current_f1"] for r in rws])),
               "old_mean": (float(np.mean([r["old_mean_f1"] for r in rws])) if i >= 1 else None),
               "per_student": rws}
        results["continual"].append(agg)
        print(json.dumps({k: agg[k] for k in ("arm_stage", "roots", "cum_mean",
                                              "old_mean", "cur_mean")}), flush=True)

    json.dump({**results, "rate": args.rate, "budgets": budgets,
               **protocol_v3.config_dict(),
               "protocol": protocol_v3.PROTOCOL_NAME,
               "student_batch": args.student_batch,
               "student_steps_override": args.student_steps,
               "student_seeds": args.student_seeds,
               "dev_rule": "orig_id%10==1", "test_rule": "orig_id%10==2 SEALED",
               "test_accessed": False, "setting": args.split_prefix},
              open(out / "results.json", "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
