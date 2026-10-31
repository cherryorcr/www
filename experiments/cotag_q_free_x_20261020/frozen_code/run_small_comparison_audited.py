"""Reproducible small-graph comparisons; report-dev only, resumable per fit.

Archived v4 artifacts contain the actual selected class allocations. Loading
them here avoids confusing a single-recipe SCA arm with the v4 mainline.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import run_unified_v3_continual as U
from small_comparison_metrics import evaluate

G, SCA, P = U.G, U.SCA, U.protocol_v3
RESULTS = Path("/home/pc/research/TAG_condensation/results")
CORE_ARMS = ["dispersion", "v4", "concat64", "random_real", "k_center",
             "k_center_class", "herding", "kmeans_medoid", "recluster",
             "full_history_syn", "raw_equal", "raw_long", "raw_finetune",
             "v4_init_finetune"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_hash(tensors):
    digest = hashlib.sha256()
    for key, value in sorted(tensors.items()):
        if isinstance(value, torch.Tensor):
            v = value.detach().cpu().contiguous()
            digest.update(str((key, str(v.dtype), tuple(v.shape))).encode())
            digest.update(v.numpy().tobytes())
    return digest.hexdigest()


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def induced_edges(edges, ids, total):
    ids = ids.long().cpu()
    if len(torch.unique(ids)) != len(ids) or ids.min() < 0 or ids.max() >= total:
        raise ValueError("invalid source node IDs")
    remap = torch.full((total,), -1, dtype=torch.long)
    remap[ids] = torch.arange(len(ids))
    local = remap[edges.long().cpu()]
    return local[:, (local >= 0).all(0)]


def raw_support(data, ids, stage):
    if not data["pool_mask"][ids].all():
        raise ValueError("holdout source access")
    return {"x": data["x"][ids].float(), "text_emb": data["text_emb"][ids],
            "texts": [data["texts"][int(i)] for i in ids],
            "edge_index": induced_edges(data["edge_index"], ids, len(data["x"])),
            "mass": torch.ones(len(ids)), "depth": torch.zeros(len(ids), dtype=torch.long),
            "stage": torch.full((len(ids),), stage, dtype=torch.long), "source_ids": ids}


def assert_frozen(old, new):
    if old is None:
        return
    n = len(old["x"])
    for key in ("x", "text_emb", "mass", "stage"):
        if not torch.equal(old[key], new[key][:n]):
            raise AssertionError(f"old {key} changed")
    if old["texts"] != new["texts"][:n]:
        raise AssertionError("old text changed")
    edges = new["edge_index"]
    old_edges = edges[:, (edges < n).all(0)]
    if not torch.equal(old["edge_index"], old_edges):
        raise AssertionError("old-old topology changed")


def artifact_dir(dataset, arm):
    if arm == "concat64":
        return RESULTS / "p2_token_controls_2026_09_16" / dataset / "concat_t64_g5_r0.5_a0.5"
    name = "v4_mainline_safe" if arm == "v4" else "v4_baseline_current"
    return RESULTS / f"{name}_2026_09_15_{dataset}"


def real_block(data, ids, rows, method, stage, device):
    # Resolve original IDs once and retain them in the artifact for audits.
    block, audit = U.build_real_selection_block(data, ids, rows, method, stage, device)
    selected, remaining = [], ids.tolist()
    for x, text in zip(block["x"], block["texts"]):
        found = next(i for i in remaining if data["texts"][i] == text
                     and torch.equal(data["x"][i].float(), x))
        selected.append(found)
        remaining.remove(found)
    block["source_ids"] = torch.tensor(selected)
    return block, audit


def construct(data, arm, stages, budgets, out, device):
    memories, audits, old = [], [], None
    for stage, ids in enumerate(stages):
        target = out / f"{arm}_s{stage}_memory.pt"
        audit_path = out / f"{arm}_s{stage}_construction.json"
        if target.exists() and audit_path.exists():
            mem = torch.load(target, map_location="cpu", weights_only=False)
            audit = json.loads(audit_path.read_text())
            if tensor_hash(mem) != audit["tensor_sha256"]:
                raise ValueError("cached memory hash mismatch")
        else:
            started = time.perf_counter()
            keep = 0 if old is None else len(old["x"])
            new_rows = budgets[stage] - keep
            audit = {"arm": arm, "stage": stage, "keep": keep, "new": new_rows}
            if arm in ("dispersion", "v4", "concat64"):
                root = artifact_dir(data["dataset"], arm)
                source = root / f"append_fixed_s{stage}_memory.pt"
                manifest = json.loads((root / "results.json").read_text())
                if arm == "v4" and not manifest.get("response_marginal"):
                    raise ValueError("v4 artifact lacks response marginal selection")
                mem = torch.load(source, map_location="cpu", weights_only=False)
                audit.update(source=str(source), source_sha256=sha(source),
                             manifest_sha256=sha(root / "results.json"),
                             selected_allocation=next(r for r in manifest["results"]
                                 if r["stage"] == stage))
            elif arm.startswith("raw"):
                source_ids = ids if arm == "raw_finetune" else torch.cat(stages[:stage + 1])
                mem = raw_support(data, source_ids, stage)
            elif arm == "full_history_syn":
                source_ids = torch.cat(stages[:stage + 1])
                mem = U.build_stage_block(data, source_ids, budgets[stage], data["encoder"],
                    stage, "joint_g5_c320", "concat", device, "dispersion",
                    [c for group in data["stage_classes"][:stage + 1] for c in group], .5, .5)
                mem = G.memory_to_trainable(mem) | {k: mem[k] for k in ("mass", "stage", "depth")}
            elif arm == "recluster":
                if old is None:
                    mem = G.condense_static(data["x"][ids], data["text_emb"][ids],
                        [data["texts"][int(i)] for i in ids], budgets[stage], data["encoder"], device)
                else:
                    mem, extra = G.continual_recluster(old, data["x"][ids], data["text_emb"][ids],
                        [data["texts"][int(i)] for i in ids], budgets[stage], stage, data["encoder"], device)
                    audit["recluster"] = extra
                mem = G.memory_to_trainable(mem) | {k: mem[k] for k in ("mass", "stage", "depth")}
            else:
                if arm == "k_center_class":
                    allocations = U.allocate_rows_by_class(data, ids, data["stage_classes"][stage],
                                                           new_rows, "dispersion")
                    # allocate_rows_by_class also returns class summary records.
                    if isinstance(allocations, tuple):
                        allocations = allocations[0]
                    parts, subaudits = [], []
                    for cls in data["stage_classes"][stage]:
                        count = allocations[cls]
                        part, subaudit = real_block(data, ids[data["y"][ids] == cls], count,
                                                   "k_center", stage, device)
                        parts.append(part)
                        subaudits.append(subaudit)
                    block = {key: torch.cat([p[key] for p in parts])
                             for key in ("x", "text_emb", "mass", "depth", "stage", "source_ids")}
                    block["texts"] = [t for p in parts for t in p["texts"]]
                    audit["class_allocation"] = allocations
                else:
                    block, extra = real_block(data, ids, new_rows, arm, stage, device)
                    audit.update(extra)
                mem = SCA.append_memory(old, block, keep, topology_mode="frozen_old",
                    cross_degree=1, refresh_mode="frozen_old", device=device)
                mem["source_ids"] = torch.cat([old["source_ids"], block["source_ids"]]) if old else block["source_ids"]
            audit.update(build_seconds=time.perf_counter() - started, rows=len(mem["x"]),
                         tensor_sha256=tensor_hash(mem))
            torch.save(mem, target)
            save_json(audit_path, audit)
        if not arm.startswith("raw") and len(mem["x"]) != budgets[stage]:
            raise AssertionError(f"budget mismatch: {arm} stage {stage}")
        if arm not in ("full_history_syn", "recluster") and not arm.startswith("raw"):
            assert_frozen(old, mem)
        memories.append(mem)
        audits.append(audit)
        old = mem
    return memories, audits


def training_adjacency(memory, device):
    """Preserve externally condensed normalized edge weights without renormalizing."""
    if "adj_values" not in memory:
        return G.norm_adj(memory["edge_index"], len(memory["x"]), device)
    if memory.get("adj_normalized") is not True:
        raise ValueError("weighted support must declare normalized adjacency")
    values, indices = memory["adj_values"], memory["edge_index"]
    n = len(memory["x"])
    if (values.ndim != 1 or indices.shape != (2, len(values))
            or not torch.isfinite(values).all() or (values < 0).any()
            or (indices < 0).any() or (indices >= n).any()):
        raise ValueError("invalid normalized weighted adjacency")
    return torch.sparse_coo_tensor(indices.to(device), values.to(device),
                                   (n, n), device=device).coalesce()


def fit(memory, seed, steps, device, data, stage, initial=None, select=False, checkpoint=None):
    torch.manual_seed(seed)
    model = G.CLIPStudent(memory["x"].shape[1]).to(device)
    if initial is not None:
        model.load_state_dict(initial)
    start_hash = tensor_hash(model.state_dict())
    x, t = memory["x"].to(device), memory["text_emb"].to(device)
    adj = training_adjacency(memory, device)
    opt = torch.optim.Adam(model.parameters(), lr=P.STUDENT.lr)
    gen = torch.Generator().manual_seed(seed * 7 + 1)
    best, best_state, best_step, curve = -1., None, None, []
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    for step in range(1, steps + 1):
        model.train()
        rows = torch.randperm(len(x), generator=gen)[:min(P.STUDENT.batch, len(x))].to(device)
        g, tt = model.encode_graph(x, adj)[rows], model.encode_text(t[rows])
        # Preserve canonical floating-point operation order for regression.
        logits = model.logit_scale.exp().clamp(max=100) * g @ tt.T
        target = torch.arange(len(rows), device=device)
        loss = (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target)) / 2
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite loss at {step}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if select and (step % 512 == 0 or step == steps):
            value = evaluate(model, data, stage, "selector_dev_mask")["all"]["cumulative_f1"]
            curve.append({"step": step, "selector_all_f1": value})
            if value > best:
                best, best_step = value, step
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    row = {"seed": seed, "steps": steps, "rows": len(x), "seconds": elapsed,
           "peak_allocated_mb": torch.cuda.max_memory_allocated(device) / 2**20,
           "start_state_sha256": start_hash, "end_state_sha256": tensor_hash(state),
           "support_sha256": tensor_hash(memory), "metrics": evaluate(model, data, stage)}
    if select:
        model.load_state_dict(best_state)
        row.update(selected_steps=best_step, selection_curve=curve,
                   selected_metrics=evaluate(model, data, stage))
    if checkpoint:
        torch.save({"state": state, "selected_state": best_state, "result": row}, checkpoint)
    del model, opt
    return row


def aggregate(rows):
    result = {}
    for scope in ("all", "seen"):
        result[scope] = {}
        for key in ("cumulative_f1", "old_f1", "new_f1", "accuracy"):
            values = [r["metrics"][scope][key] for r in rows if r["metrics"][scope][key] is not None]
            result[scope][key] = {"mean": float(np.mean(values)), "std": float(np.std(values))} if values else None
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cora", "citeseer"], required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--minilm", default=str(Path(U.__file__).parent / "minilm_from_graphclip"))
    ap.add_argument("--arms", nargs="+", default=CORE_ARMS)
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(100, 106)))
    ap.add_argument("--raw-steps", type=int, default=8192)
    args = ap.parse_args()
    torch.set_num_threads(2)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    data_path = U.DATA_CFG[args.dataset]["path"]
    data = U.load_unified(args.dataset, data_path, args.minilm, args.device)
    cache = artifact_dir(args.dataset, "v4") / "text_emb_pool_only.pt"
    data["text_emb"] = torch.load(cache, map_location="cpu", weights_only=False)
    if data["text_emb"][~data["pool_mask"]].abs().max() != 0:
        raise ValueError("source embedding cache contains holdout rows")
    data["prompt_emb"] = data["encoder"].encode(data["prompts"])
    data["real_adj"] = G.norm_adj(data["edge_index"], len(data["x"]), args.device)
    data["x_device"] = data["x"].to(args.device)
    stages = [torch.where(data["pool_mask"] & torch.isin(data["y"], torch.tensor(cs)))[0]
              for cs in data["stage_classes"]]
    budgets = [math.floor(.05 * sum(len(i) for i in stages[:s + 1])) for s in range(len(stages))]
    manifest = {"version": "small-comparison-audited-v1", "dataset": args.dataset,
        "data_sha256": sha(data_path), "text_cache_sha256": sha(cache),
        "code_sha256": {Path(p).name: sha(p) for p in
            [__file__, U.__file__, G.__file__, SCA.__file__, P.__file__,
             Path(__file__).with_name("small_comparison_metrics.py")]},
        "stage_classes": data["stage_classes"], "budgets": budgets,
        "stage_pool_counts": [len(ids) for ids in stages], "seeds": args.seeds,
        "arms": args.arms, "raw_steps": args.raw_steps, "batch": P.STUDENT.batch,
        "rate": .05, "lr": P.STUDENT.lr, "test_labels_evaluated": False,
        "evaluation_graph": "transductive full graph including unlabeled holdout/future inputs",
        "construction_labels": "v4/dispersion/full_history_syn/k_center_class use pool class labels",
        "training_labels": "student InfoNCE does not use class labels",
        "raw_long_selection": "all-prompt selector-dev every 512 steps; final and selected reported separately",
        "fine_tune": "inherit model parameters, reset Adam, train current raw delta only",
        "coreset_topology": "same semantic append rule as synthetic memory, not induced raw edges",
        "prompt_scopes": ["all classes (historical primary)", "seen classes"],
        "token64": "P2 text-only control with frozen legacy feature/topology; not newly selected v4"}
    manifest_path = out / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("run configuration changed; choose a new output directory")
    save_json(manifest_path, manifest)
    results, audits, supports = [], {}, {}
    for arm in args.arms:
        if arm in ("v4_init_finetune", "raw_finetune"):
            init_arm = "v4" if arm == "v4_init_finetune" else "raw_equal"
            if init_arm not in supports:
                supports[init_arm], audits[init_arm] = construct(data, init_arm, stages, budgets, out, args.device)
            memories = [supports[init_arm][0]] + [raw_support(data, ids, s) for s, ids in enumerate(stages) if s]
        else:
            memories, audits[arm] = construct(data, arm, stages, budgets, out, args.device)
            supports[arm] = memories
        for stage, memory in enumerate(memories):
            scores = []
            steps = args.raw_steps if arm == "raw_long" else P.student_steps(budgets[stage])
            for seed in args.seeds:
                key = f"{arm}_s{stage}_seed{seed}"
                rowpath, checkpoint = out / f"{key}.json", out / f"{key}.pt"
                if rowpath.exists() and checkpoint.exists():
                    row = json.loads(rowpath.read_text())
                    if row["support_sha256"] != tensor_hash(memory) or row["steps"] != steps:
                        raise ValueError("stale fit checkpoint")
                else:
                    initial = None
                    if arm.endswith("finetune") and stage > 0:
                        previous = torch.load(out / f"{arm}_s{stage - 1}_seed{seed}.pt",
                                              map_location="cpu", weights_only=False)
                        initial = previous["state"]
                    row = fit(memory, seed, steps, args.device, data, stage, initial,
                              arm == "raw_long", checkpoint)
                    if initial is not None and row["start_state_sha256"] != tensor_hash(initial):
                        raise AssertionError("fine-tuning did not inherit prior state")
                    save_json(rowpath, row)
                scores.append(row)
            summary = {"arm": arm, "stage": stage, "steps": steps, "rows": len(memory["x"]),
                       "metrics": aggregate(scores), "per_seed": scores}
            if arm == "raw_long":
                summary["selected_metrics"] = aggregate([dict(r, metrics=r["selected_metrics"]) for r in scores])
            results.append(summary)
            save_json(out / "progress.json", {"manifest": manifest, "results": results, "audits": audits})
            print(json.dumps({"arm": arm, "stage": stage, "all": summary["metrics"]["all"],
                              "seen": summary["metrics"]["seen"]["cumulative_f1"]}), flush=True)
    save_json(out / "results.json", {"manifest": manifest, "results": results, "audits": audits})
    print("AUDITED_COMPARISON_DONE", flush=True)


if __name__ == "__main__":
    main()
