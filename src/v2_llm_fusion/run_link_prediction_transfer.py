"""Leakage-controlled downstream GAE transfer from frozen synthetic memories."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as TF
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.nn import GAE, GCNConv

import protocol_v3 as P
import safe_joint_kernel_append as S

ARMS = ("c0", "joint_kernel", "degree_balanced", "coverage", "full_raw_equal")
SPLITS = (29001, 29002, 29003)
SEEDS = tuple(range(100, 106))
ROOT = Path("/home/pc/research/TAG_condensation")
PROTOCOL = "LINK_PREDICTION_FROZEN_PROTOCOL_ZH_2026-09-20.md"


def save_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pairs(edge, n):
    edge = edge.long().cpu()
    lo, hi = edge.min(0).values, edge.max(0).values
    keys = torch.unique(lo[lo != hi] * n + hi[lo != hi], sorted=True)
    return torch.stack((keys // n, keys % n))


def bidirectional(edge):
    return torch.cat((edge, edge.flip(0)), dim=1)


def pair_keys(edge, n):
    return set((edge[0] * n + edge[1]).tolist())


def split_edges(edge, arrival, seed):
    """Stratify by endpoint arrival; fixed negatives preserve slice prevalence."""
    n = len(arrival)
    positive = pairs(edge, n)
    positive = positive[:, (arrival[positive] >= 0).all(0)]
    rng = np.random.default_rng(seed)
    u, v = np.triu_indices(n, 1)
    admissible = (arrival[u].numpy() >= 0) & (arrival[v].numpy() >= 0)
    keys = positive[0].numpy() * n + positive[1].numpy()
    admissible &= ~np.isin(u * n + v, keys)
    negative = torch.from_numpy(np.stack((u[admissible], v[admissible]))).long()
    ap, an = arrival[positive].sort(dim=0).values, arrival[negative].sort(dim=0).values
    buckets = {key: [] for key in ("train_pos", "val_pos", "test_pos", "val_neg", "test_neg", "train_neg")}
    for first in range(3):
        for last in range(first, 3):
            pidx = torch.where((ap[0] == first) & (ap[1] == last))[0].numpy()
            nidx = torch.where((an[0] == first) & (an[1] == last))[0].numpy()
            rng.shuffle(pidx)
            # Permuting each negative stratum also freezes train-negative sampling.
            rng.shuffle(nidx)
            nv = nt = int(len(pidx) * .1)
            if nv + nt > len(nidx):
                raise ValueError("not enough genuine nonedges")
            buckets["val_pos"].append(positive[:, pidx[:nv]])
            buckets["test_pos"].append(positive[:, pidx[nv:nv + nt]])
            buckets["train_pos"].append(positive[:, pidx[nv + nt:]])
            buckets["val_neg"].append(negative[:, nidx[:nv]])
            buckets["test_neg"].append(negative[:, nidx[nv:nv + nt]])
            buckets["train_neg"].append(negative[:, nidx[nv + nt:]])
    result = {key: torch.cat(value, dim=1) for key, value in buckets.items()}
    result.update(arrival=arrival, seed=seed, full_pos=positive)
    validate_split(result)
    return result


def validate_split(split):
    n = len(split["arrival"])
    groups = {k: pair_keys(split[k], n) for k in ("train_pos", "val_pos", "test_pos", "train_neg", "val_neg", "test_neg")}
    all_keys = set()
    for name, group in groups.items():
        edge = split[name]
        assert len(group) == edge.shape[1]
        assert bool((edge[0] < edge[1]).all())
        assert bool((split["arrival"][edge] >= 0).all())
        assert not (group & all_keys), name
        all_keys |= group
    assert groups["train_pos"] | groups["val_pos"] | groups["test_pos"] == pair_keys(split["full_pos"], n)
    for name in ("val", "test"):
        pa = split["arrival"][split[name + "_pos"]].sort(dim=0).values
        na = split["arrival"][split[name + "_neg"]].sort(dim=0).values
        assert torch.equal(pa, na), "negative endpoint strata differ"


def localize(edge, ids, total):
    remap = torch.full((total,), -1, dtype=torch.long)
    remap[ids] = torch.arange(len(ids))
    local = remap[edge]
    return local[:, (local >= 0).all(0)]


def features(x, text):
    return torch.cat((TF.normalize(x.float(), dim=1), TF.normalize(text.float(), dim=1)), dim=1) / math.sqrt(2)


def synthetic_negatives(edge, n):
    allowed = torch.ones((n, n), dtype=torch.bool).triu(1)
    allowed[edge[0], edge[1]] = False
    return allowed.nonzero().T.contiguous()


def masks(edge, arrival, stage):
    ends = arrival[edge]
    return {"all": torch.ones(edge.shape[1], dtype=torch.bool),
            "old": (ends < stage).all(0), "new": (ends == stage).any(0),
            "cross": (ends[0] == stage) != (ends[1] == stage),
            "within_new": (ends == stage).all(0)}


def metrics(pos, neg, positive, negative, arrival, stage):
    pm, nm = masks(positive, arrival, stage), masks(negative, arrival, stage)
    result = {}
    for name in pm:
        p, n = pos[pm[name]], neg[nm[name]]
        item = {"positives": len(p), "negatives": len(n), "auc": None, "ap": None}
        if len(p) and len(n):
            y = np.r_[np.ones(len(p)), np.zeros(len(n))]
            values = np.r_[p.numpy(), n.numpy()]
            item.update(auc=float(roc_auc_score(y, values)), ap=float(average_precision_score(y, values)))
        result[name] = item
    return result


class Encoder(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.first = GCNConv(input_dim, 384, cached=False)
        self.second = GCNConv(384, 128, cached=False)

    def forward(self, x, edge):
        return self.second(self.first(x, edge).relu(), edge)


def sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def fit(train_x, train_pos, train_neg, real_x, real_edge, test_pos, test_neg,
        seed, steps, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = GAE(Encoder(train_x.shape[1])).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    x, pe, ne = train_x.to(device), train_pos.to(device), train_neg.to(device)
    adjacency = bidirectional(pe)
    if not pe.shape[1] or not ne.shape[1]:
        raise ValueError("empty training positive/negative pool")
    rng = np.random.default_rng(seed)
    batch = min(256, pe.shape[1], ne.shape[1])
    sync(device)
    start = time.perf_counter()
    model.train()
    for _ in range(steps):
        # A separate generator makes edge sampling independent of initialization/device RNG.
        pi = torch.as_tensor(rng.choice(pe.shape[1], batch, replace=False), device=device)
        ni = torch.as_tensor(rng.choice(ne.shape[1], batch, replace=False), device=device)
        optimizer.zero_grad(set_to_none=True)
        z = model.encode(x, adjacency)
        loss = model.recon_loss(z, pe[:, pi], ne[:, ni])
        loss.backward()
        optimizer.step()
    sync(device)
    seconds = time.perf_counter() - start
    model.eval()
    with torch.no_grad():
        z = model.encode(real_x.to(device), real_edge.to(device))
        # Logits retain ordering even when sigmoid saturates.
        ps = model.decoder(z, test_pos.to(device), sigmoid=False).cpu()
        ns = model.decoder(z, test_neg.to(device), sigmoid=False).cpu()
    return model, ps, ns, {"train_seconds": seconds, "last_loss": float(loss.detach()),
                          "train_rows": len(x), "train_positive_pairs": pe.shape[1],
                          "negative_pool_size": ne.shape[1], "edge_batch": batch,
                          "steps": steps, "seed": seed}


def rebuild(data, split, archive, output):
    # Import the original contracts only here; unit tests do not need their full stack.
    import run_safe_joint_kernel_experiments as E
    start = time.perf_counter()
    memories = {arm: [] for arm in ARMS[:-1]}
    provenance = []
    masked = dict(data, edge_index=bidirectional(split["train_pos"]))
    for stage in range(3):
        base_path = archive / f"c0_s{stage}_memory.pt"
        base = torch.load(base_path, map_location="cpu", weights_only=False)
        meta_path = archive / f"c0_s{stage}_construction.json"
        meta = json.loads(meta_path.read_text())
        assert E.F.J_hash(base) == meta["content_sha256"]
        old_n = 0 if stage == 0 else len(memories["c0"][-1]["x"])
        if stage:
            E.C.assert_frozen(memories["c0"][-1], base)
        block = E.copy_block(base, old_n)
        sources_path = archive / f"joint_kernel_s{stage}_sources.json"
        groups = json.loads(sources_path.read_text())["source_groups"]
        ids = torch.where(split["arrival"] == stage)[0]
        flat = torch.tensor([i for group in groups for i in group])
        assert torch.equal(flat.sort().values, ids), "source partition differs from current pool delta"
        joint = torch.cat((TF.normalize(data["x"][ids], dim=1), data["text_emb"][ids]), dim=1)
        graph = E.J.source_local_graph_summary(masked, ids, joint)
        remap = torch.full((len(data["x"]),), -1, dtype=torch.long)
        remap[ids] = torch.arange(len(ids))
        signatures = []
        assert len(groups) == len(block["x"])
        for row, group in enumerate(groups):
            assert torch.equal(data["x"][group].mean(0), block["x"][row])
            assert len(group) == int(block["mass"][row])
            signatures.append(graph[remap[group]].mean(0))
        block["graph_signature"] = TF.normalize(torch.stack(signatures), dim=1)
        block = S.compress_signature(block, "fp32")
        memories["c0"].append(base)
        record = {"stage": stage, "c0_file_sha256": sha(base_path), "sources_sha256": sha(sources_path),
                  "cached_c0_build_seconds": meta["build_seconds"], "arms": {}}
        for arm in ARMS[:-1]:
            build_start = time.perf_counter()
            if arm != "c0":
                old = memories[arm][-1] if stage else None
                mem, audit = S.append_variant(old, block, old_n, arm)
                if old is not None:
                    E.C.assert_frozen(old, mem)
                for key in ("x", "text_emb", "mass", "depth", "stage"):
                    assert torch.equal(mem[key], base[key]), key
                assert mem["texts"] == base["texts"]
                assert pairs(mem["edge_index"], len(mem["x"])).shape == pairs(base["edge_index"], len(base["x"])).shape
                memories[arm].append(mem)
            mem = memories[arm][-1]
            torch.save(mem, output / f"{arm}_s{stage}_memory.pt")
            record["arms"][arm] = {"payload": E.payload(mem), "content_sha256": E.F.J_hash(mem),
                                    "append_save_seconds": time.perf_counter() - build_start}
        provenance.append(record)
    save_json(output / "construction.json", {"masked_edges_only": True, "stage_records": provenance,
                                              "shared_rebuild_seconds": time.perf_counter() - start})
    return memories


def run(args):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    raw_path = Path("/home/pc/research/GraphCLIP/processed_data") / (args.dataset + ".pt")
    text_path = ROOT / f"results/p5_rate_curve_small_20260919/{args.dataset}_r005/text_emb_pool_only.pt"
    raw = torch.load(raw_path, map_location="cpu", weights_only=False)
    text = torch.load(text_path, map_location="cpu", weights_only=False)
    assert isinstance(text, torch.Tensor)
    pool = P.split_masks_v4(torch.arange(len(raw.x)))[-1]
    assert torch.count_nonzero(text[~pool]) == 0
    arrival = torch.full((len(raw.x),), -1, dtype=torch.long)
    classes = [[0, 1], [2, 3], list(range(4, 7 if args.dataset == "cora" else 6))]
    for stage, labels in enumerate(classes):
        arrival[pool & torch.isin(raw.y, torch.tensor(labels))] = stage
    data = {"x": raw.x.float(), "text_emb": text.float(), "pool_mask": pool}
    archive = ROOT / f"results/safe_joint_kernel_v2_{args.dataset}_20260920/s3_canonical_r05"
    source_paths = [Path(__file__), Path(S.__file__), Path(P.__file__),
                    Path(__file__).with_name("joint_kernel_append.py"),
                    Path(__file__).resolve().parents[2] / "docs/current" / PROTOCOL]
    config = {"dataset": args.dataset, "steps": args.steps, "split_seeds": args.splits, "student_seeds": args.seeds,
              "arms": list(ARMS), "rate": .05, "classes": classes, "raw_sha256": sha(raw_path),
              "text_sha256": sha(text_path), "source_hashes": {p.name: sha(p) for p in source_paths},
              "torch_version": torch.__version__, "device": args.device,
              "evaluation_scope": "seen construction-pool nodes; held-out undirected edges"}
    manifest = out / "manifest.json"
    if manifest.exists():
        assert json.loads(manifest.read_text()) == config, "frozen LP config/source changed"
    save_json(manifest, config)
    results = []
    for split_seed in args.splits:
        folder = out / f"split_{split_seed}"
        folder.mkdir(exist_ok=True)
        split = split_edges(raw.edge_index, arrival, split_seed)
        torch.save(split, folder / "edge_split.pt")
        save_json(folder / "split_counts.json", {k: v.shape[1] for k, v in split.items()
                                                  if isinstance(v, torch.Tensor) and v.ndim == 2})
        memories = rebuild(data, split, archive, folder)
        for stage in range(3):
            ids = torch.where((arrival >= 0) & (arrival <= stage))[0]
            localized = {k: localize(split[k], ids, len(arrival)) for k in
                         ("train_pos", "train_neg", "test_pos", "test_neg")}
            local_arrival = arrival[ids]
            real_x = features(data["x"][ids], text[ids])
            real_edge = bidirectional(localized["train_pos"])
            eval_data = {"ids": ids, "arrival": local_arrival, "x": real_x,
                         "edge_index": real_edge, "test_pos": localized["test_pos"], "test_neg": localized["test_neg"]}
            torch.save(eval_data, folder / f"evaluation_s{stage}.pt")
            diagnostics = {}
            for name, values in (("x_cosine", TF.normalize(data["x"][ids], dim=1)),
                                 ("text_cosine", TF.normalize(text[ids], dim=1)), ("joint_cosine", real_x)):
                ps = (values[localized["test_pos"][0]] * values[localized["test_pos"][1]]).sum(1)
                ns = (values[localized["test_neg"][0]] * values[localized["test_neg"][1]]).sum(1)
                diagnostics[name] = metrics(ps, ns, localized["test_pos"], localized["test_neg"], local_arrival, stage)
            save_json(folder / f"cosine_s{stage}.json", diagnostics)
            for arm in ARMS:
                if arm == "full_raw_equal":
                    tx, tp, tn = real_x, localized["train_pos"], localized["train_neg"]
                else:
                    memory = memories[arm][stage]
                    tx = features(memory["x"], memory["text_emb"])
                    tp = pairs(memory["edge_index"], len(tx))
                    tn = synthetic_negatives(tp, len(tx))
                for seed in args.seeds:
                    stem = folder / f"{arm}_s{stage}_seed{seed}"
                    result_path = stem.with_suffix(".json")
                    if result_path.exists():
                        result = json.loads(result_path.read_text())
                        assert stem.with_suffix(".pt").exists()
                    else:
                        model, ps, ns, costs = fit(tx, tp, tn, real_x, real_edge,
                            localized["test_pos"], localized["test_neg"], seed, args.steps, args.device)
                        result = {"dataset": args.dataset, "split": split_seed, "stage": stage,
                                  "arm": arm, "seed": seed, "cost": costs,
                                  "metrics": metrics(ps, ns, localized["test_pos"], localized["test_neg"], local_arrival, stage)}
                        torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                                    "pos_scores": ps, "neg_scores": ns, "metadata": result,
                                    "evaluation_sha256": sha(folder / f"evaluation_s{stage}.pt")}, stem.with_suffix(".pt"))
                        result["checkpoint_sha256"] = sha(stem.with_suffix(".pt"))
                        save_json(result_path, result)
                    results.append(result)
                    print(json.dumps({k: result[k] for k in ("dataset", "split", "stage", "arm", "seed")}
                                     | result["metrics"]["all"]), flush=True)
    save_json(out / "results.json", results)
    save_json(out / "complete.json", {"fits": len(results), "expected": len(args.splits) * 3 * len(ARMS) * len(args.seeds)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("cora", "citeseer"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--splits", nargs="+", type=int, default=list(SPLITS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    run(parser.parse_args())
