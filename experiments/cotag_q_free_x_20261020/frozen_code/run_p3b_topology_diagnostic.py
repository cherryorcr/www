"""P3-B controlled topology diagnostic for continual synthetic memory.

The runner fixes Mean-X, dispersion allocation, text, rows, stage order and
student training.  It varies only the new-to-old cross edges:

* current: text-similarity nearest old endpoints;
* none: no cross edges;
* random: the same number of deterministic random endpoints;
* quality: the same number of endpoints selected by a joint X/text score.

The experiment is intentionally a small-graph mechanism diagnostic.  It does
not change old-old edges, old rows, or the synthetic node construction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import run_small_comparison_audited as C
import run_small_order_cost_confirmation as O
import run_unified_v3_continual as U

G, SCA, P = U.G, U.SCA, U.protocol_v3

ARMS = ("current", "none", "random", "quality")
ORDER_SEEDS = (1701, 1702)
STUDENT_SEEDS = tuple(range(100, 106))
METHOD_SEED = 9001


def sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def save_json(path: Path, value) -> None:
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _edge_parts(edge_index: torch.Tensor, old_n: int, new_n: int) -> dict:
    """Count directed edge entries by old/new block, excluding self loops."""
    edges = edge_index.long().cpu()
    src, dst = edges[0], edges[1]
    src_old, dst_old = src < old_n, dst < old_n
    cross = src_old ^ dst_old
    new_new = (~src_old) & (~dst_old)
    old_old = src_old & dst_old
    return {
        "old_old_directed": int(old_old.sum()),
        "new_new_directed": int(new_new.sum()),
        "cross_directed": int(cross.sum()),
        "cross_undirected": int(cross.sum() // 2),
        "self_loops": int(((src == dst) & old_old).sum()),
    }


def _undirected_cross(edge_index: torch.Tensor, old_n: int) -> torch.Tensor:
    edges = edge_index.long().cpu()
    src, dst = edges[0], edges[1]
    cross = (src < old_n) ^ (dst < old_n)
    pairs = edges[:, cross]
    if pairs.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long)
    low = torch.minimum(pairs[0], pairs[1])
    high = torch.maximum(pairs[0], pairs[1])
    pairs = torch.stack((low, high), dim=0).t().unique(dim=0)
    return pairs.t().contiguous()


def _make_cross_edges(old: dict, block: dict, variant: str, stage: int,
                      order_name: str, cross_degree: int = 1) -> torch.Tensor:
    """Return directed cross edges with an identical cardinality per variant."""
    if variant not in ARMS:
        raise ValueError(f"unknown topology variant: {variant}")
    old_n, new_n = len(old["x"]), len(block["x"])
    if (variant == "none" or old_n == 0 or new_n == 0
            or cross_degree <= 0):
        return torch.empty((2, 0), dtype=torch.long)
    width = min(int(cross_degree), old_n)
    if variant == "current":
        old_t = F.normalize(old["text_emb"].float(), dim=1)
        new_t = F.normalize(block["text_emb"].float(), dim=1)
        nearest = torch.topk(new_t @ old_t.T, width, dim=1,
                             sorted=True).indices
    elif variant == "random":
        seed = METHOD_SEED + stage * 1009 + sum(ord(c) for c in order_name)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        nearest = torch.stack([torch.randperm(old_n, generator=generator)[:width]
                               for _ in range(new_n)])
    elif variant == "quality":
        old_t = F.normalize(old["text_emb"].float(), dim=1)
        new_t = F.normalize(block["text_emb"].float(), dim=1)
        old_x = F.normalize(old["x"].float(), dim=1)
        new_x = F.normalize(block["x"].float(), dim=1)
        score = 0.5 * (new_t @ old_t.T) + 0.5 * (new_x @ old_x.T)
        nearest = torch.topk(score, width, dim=1, sorted=True).indices
    else:
        raise ValueError(f"unknown topology variant: {variant}")
    new_ids = torch.arange(new_n, dtype=torch.long).unsqueeze(1)
    new_ids = new_ids.expand_as(nearest)
    forward = torch.stack((nearest, old_n + new_ids), dim=0).reshape(2, -1)
    return torch.cat((forward, forward.flip(0)), dim=1).contiguous()


def append_variant(old: dict | None, block: dict, keep: int, variant: str,
                   stage: int, order_name: str, device: str) -> dict:
    if variant not in ARMS:
        raise ValueError(f"unknown topology variant: {variant}")
    if old is None:
        out = dict(block)
        out["edge_index"] = SCA.memory_edges(block, device=device)
        out["topology_variant"] = variant
        return out
    if keep != len(old["x"]):
        raise ValueError("P3-B forbids pruning/reordering old rows")
    old_n = len(old["x"])
    old_edges = SCA.memory_edges(old, device=device)
    new_edges = SCA.memory_edges(block, device=device) + old_n
    cross = _make_cross_edges(old, block, variant, stage, order_name)
    if cross.numel():
        cross = cross.to(torch.long)
    out = {
        key: (old[key] + block[key] if key == "texts"
              else torch.cat((old[key], block[key])))
        for key in ("x", "text_emb", "texts", "mass", "depth", "stage")
    }
    out["edge_index"] = torch.cat((old_edges, new_edges, cross), dim=1).long()
    out["topology_variant"] = variant
    out["cross_degree"] = 1
    out["cross_edge_count"] = int(cross.shape[1] // 2)
    return out


def _dense_adj(edge_index: torch.Tensor, n: int, device: str) -> torch.Tensor:
    return G.norm_adj(edge_index, n, device).to_dense().detach().cpu()


def topology_metrics(previous: dict | None, current: dict, device: str,
                     variant: str) -> dict:
    """Measure normalized propagation changes caused by the append."""
    n = len(current["x"])
    if previous is None:
        return {
            "old_rows": 0,
            "new_rows": n,
            "old_old_normalized_fro_drift": 0.0,
            "old_old_normalized_relative_drift": 0.0,
            "old_propagation_relative_drift": 0.0,
            "cross_normalized_l1": 0.0,
            "cross_normalized_fro": 0.0,
            "cross_x_message_relative": 0.0,
            "old_total_x_propagation_relative_drift": 0.0,
            "edge_counts": _edge_parts(current["edge_index"], 0, n),
        }
    old_n = len(previous["x"])
    full_adj = _dense_adj(current["edge_index"], n, device)
    old_adj = _dense_adj(previous["edge_index"], old_n, device)
    old_block = full_adj[:old_n, :old_n]
    diff = old_block - old_adj
    x_old = previous["x"].float().cpu()
    old_prop = old_block @ x_old
    base_prop = old_adj @ x_old
    denom = float(base_prop.norm().clamp_min(1e-12))
    cross_block = full_adj[:old_n, old_n:]
    cross_message = cross_block @ current["x"][old_n:].float().cpu()
    return {
        "old_rows": old_n,
        "new_rows": n - old_n,
        "old_old_normalized_fro_drift": float(diff.norm()),
        "old_old_normalized_relative_drift": float(diff.norm()
                                                     / old_adj.norm().clamp_min(1e-12)),
        "old_propagation_relative_drift": float((old_prop - base_prop).norm() / denom),
        "cross_normalized_l1": float(cross_block.abs().sum()),
        "cross_normalized_fro": float(cross_block.norm()),
        "cross_x_message_relative": float(cross_message.norm() / denom),
        "old_total_x_propagation_relative_drift": float(
            (old_prop + cross_message - base_prop).norm() / denom),
        "edge_counts": _edge_parts(current["edge_index"], old_n, n - old_n),
        "variant": variant,
    }


@torch.no_grad()
def trained_old_embedding_drift(memory: dict, previous: dict | None,
                                checkpoint: Path, data: dict, device: str) -> float:
    if previous is None:
        return 0.0
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = G.CLIPStudent(memory["x"].shape[1]).to(device)
    model.load_state_dict(saved["state"])
    model.eval()
    full_adj = C.training_adjacency(memory, device)
    old_adj = C.training_adjacency(previous, device)
    x_old = previous["x"].to(device)
    full_z = model.encode_graph(memory["x"].to(device), full_adj)[:len(x_old)]
    old_z = model.encode_graph(x_old, old_adj)
    return float((full_z - old_z).norm()
                 / old_z.norm().clamp_min(1e-12))


def stage_orders(dataset: str) -> dict[str, list[list[int]]]:
    sizes = [2, 2, 3] if dataset == "cora" else [2, 2, 2]
    orders = {}
    for name, seed in [("canonical", None)] + [(f"order{s}", s) for s in ORDER_SEEDS]:
        classes = list(range(sum(sizes)))
        if seed is not None:
            random.Random(seed).shuffle(classes)
        offset, groups = 0, []
        for size in sizes:
            groups.append(classes[offset:offset + size])
            offset += size
        orders[name] = groups
    return orders


def prepare(dataset: str, minilm: str, device: str, out: Path) -> dict:
    sync(device)
    started = time.perf_counter()
    data = U.load_unified(dataset, U.DATA_CFG[dataset]["path"], minilm, device)
    cache = C.artifact_dir(dataset, "v4") / "text_emb_pool_only.pt"
    data["text_emb"] = torch.load(cache, map_location="cpu", weights_only=False)
    if data["text_emb"][~data["pool_mask"]].abs().max() != 0:
        raise AssertionError("source encoding cache contains holdout rows")
    data["prompt_emb"] = data["encoder"].encode(data["prompts"])
    data["real_adj"] = G.norm_adj(data["edge_index"], len(data["x"]), device)
    data["x_device"] = data["x"].to(device)
    sync(device)
    save_json(out / "preprocessing.json", {
        "dataset": dataset,
        "source_sha256": sha(Path(U.DATA_CFG[dataset]["path"])),
        "text_cache_sha256": sha(cache),
        "test_labels_evaluated": False,
        "shared_seconds": time.perf_counter() - started,
        "scope": "dataset load and cached pool encoding; no fresh source encoding cost",
    })
    return data


def build_blocks(data, groups, order_name, out, device):
    """Build current-delta-only blocks once and share them across all arms."""
    groups = P.canonicalize_class_groups(groups)
    stages, budgets = O.set_stages(data, groups)
    O.lock_json(out / "manifest.json", {
        "order": order_name, "stage_classes": groups, "budgets": budgets,
        "stage_pool_counts": list(map(len, stages)),
        "stage_ids_sha256": C.tensor_hash({str(s): ids for s, ids in enumerate(stages)}),
        "enumeration_protocol": P.ENUMERATION_PROTOCOL,
        "student_steps": [P.student_steps(k) for k in budgets]})
    blocks, audits = [], []
    for stage, ids in enumerate(stages):
        if not data["pool_mask"][ids].all():
            raise AssertionError("holdout construction access")
        keep = budgets[stage - 1] if stage else 0
        new_rows = budgets[stage] - keep
        path, audit_path = out / f"block_s{stage}.pt", out / f"block_s{stage}.json"
        if path.exists() and audit_path.exists():
            block = torch.load(path, map_location="cpu", weights_only=False)
            audit = json.loads(audit_path.read_text())
            if C.sha(path) != audit["file_sha256"]:
                raise AssertionError("cached shared block changed")
        else:
            sync(device)
            started = time.perf_counter()
            block = U.build_stage_block(
                data, ids, new_rows, data["encoder"], stage,
                "joint_g5_c320", "concat", device, "dispersion",
                data["stage_classes"][stage], .5, .5, x_operator="mean")
            block["edge_index"] = SCA.memory_edges(block, device=device)
            sync(device)
            seconds = time.perf_counter() - started
            torch.save(block, path)
            audit = {"stage": stage, "rows": new_rows, "build_seconds": seconds,
                     "file_sha256": C.sha(path), "tensor_sha256": C.tensor_hash(block),
                     "source_ids_sha256": C.tensor_hash({"ids": ids})}
            save_json(audit_path, audit)
        if len(block["x"]) != new_rows:
            raise AssertionError("shared block budget mismatch")
        blocks.append(block)
        audits.append(audit)
    return blocks, audits


def assert_control(previous, memory, reference, block):
    C.assert_frozen(previous, memory)
    for key in ("x", "text_emb", "mass", "depth", "stage"):
        if not torch.equal(memory[key], reference[key]):
            raise AssertionError(f"between-arm control failed: {key}")
    if memory["texts"] != reference["texts"]:
        raise AssertionError("between-arm text mismatch")
    keep = len(memory["x"]) - len(block["x"])
    edges = memory["edge_index"]
    new_edges = edges[:, (edges >= keep).all(0)] - keep
    if not torch.equal(new_edges, block["edge_index"]):
        raise AssertionError("new-new edges changed")
    if len(torch.unique(edges.T, dim=0)) != edges.shape[1]:
        raise AssertionError("duplicate edges")


def fit_or_load(memory: dict, previous: dict | None, data: dict, stage: int,
                arm: str, order_name: str, seed: int, out: Path,
                device: str) -> tuple[dict, Path]:
    key = f"{arm}_{order_name}_s{stage}_seed{seed}"
    row_path, checkpoint = out / f"{key}.json", out / f"{key}.pt"
    steps = P.student_steps(len(memory["x"]))
    if row_path.exists() and checkpoint.exists():
        row = json.loads(row_path.read_text())
        if row["support_sha256"] != C.tensor_hash(memory) or row["steps"] != steps:
            raise ValueError(f"stale cached fit: {key}")
    else:
        sync(device)
        start = time.perf_counter()
        row = C.fit(memory, seed, steps, device, data, stage, None, False, checkpoint)
        sync(device)
        row["fit_wall_seconds_including_report_io"] = time.perf_counter() - start
        row["topology_variant"] = arm
        row["order"] = order_name
        row["stage"] = stage
        row["old_embedding_drift"] = trained_old_embedding_drift(
            memory, previous, checkpoint, data, device)
        save_json(row_path, row)
    O.verify_fit(memory, row, checkpoint, data, stage, device)
    return row, checkpoint


def aggregate(rows: list[dict]) -> dict:
    def stats(key: str):
        values = [r["metrics"]["all"][key] for r in rows
                  if r["metrics"]["all"][key] is not None]
        if not values:
            return None
        return {"mean": float(np.mean(values)), "std": float(np.std(values)),
                "n": len(values)}
    return {
        "final_f1": stats("cumulative_f1"),
        "old_f1": stats("old_f1"),
        "new_f1": stats("new_f1"),
        "accuracy": stats("accuracy"),
        "old_embedding_drift": {
            "mean": float(np.mean([r["old_embedding_drift"] for r in rows])),
            "std": float(np.std([r["old_embedding_drift"] for r in rows])),
        },
    }


def run_order(data: dict, order_name: str, groups: list[list[int]], arms: list[str],
              seeds: list[int], out: Path, device: str, reference_root: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    all_results, all_topology = [], {}
    supports, costs, regression, counterfactual = {}, {}, [], {}
    blocks, block_audits = build_blocks(data, groups, order_name, out, device)
    reference_memories = []
    for stage in range(len(blocks)):
        path = reference_root / data["dataset"] / order_name / f"mean_x_s{stage}_memory.pt"
        reference_memories.append(torch.load(path, map_location="cpu", weights_only=False))
    fresh_fit_count = 0
    for arm in arms:
        memories, topology, topology_costs, old = [], [], [], None
        for stage, block in enumerate(blocks):
            sync(device)
            started = time.perf_counter()
            memory = append_variant(old, block, len(old["x"]) if old else 0,
                                    arm, stage, order_name, device)
            sync(device)
            seconds = time.perf_counter() - started
            assert_control(old, memory, reference_memories[stage], block)
            if arm == "current" and C.tensor_hash(memory) != C.tensor_hash(reference_memories[stage]):
                raise AssertionError("current memory regression vs P1-X")
            expected = 0 if stage == 0 or arm == "none" else len(block["x"])
            cross_count = _edge_parts(memory["edge_index"], len(old["x"]) if old else 0,
                                      len(block["x"]))["cross_undirected"]
            if cross_count != expected:
                raise AssertionError("cross edge count mismatch")
            torch.save(memory, out / f"{arm}_s{stage}_memory.pt")
            metric = topology_metrics(old, memory, device, arm)
            metric.update(tensor_sha256=C.tensor_hash(memory),
                          non_edge_sha256=C.tensor_hash({k: v for k, v in memory.items()
                                                        if k != "edge_index"}))
            topology.append(metric)
            topology_costs.append(seconds)
            memories.append(memory)
            old = memory
        audit_cost_path = out / f"{arm}_build_cost.json"
        if audit_cost_path.exists():
            build_cost = json.loads(audit_cost_path.read_text())
        else:
            build_cost = {"shared_block_build_seconds": sum(a["build_seconds"] for a in block_audits),
                          "topology_seconds": sum(topology_costs),
                          "scope": "shared block construction plus arm topology; diagnostics and IO excluded"}
            save_json(audit_cost_path, build_cost)
        supports[arm] = memories
        all_topology[arm] = topology
        save_json(out / f"{arm}_topology.json", topology)
        stage_rows = []
        for stage, memory in enumerate(memories):
            seed_rows = []
            for seed in seeds:
                if not (out / f"{arm}_{order_name}_s{stage}_seed{seed}.json").exists():
                    fresh_fit_count += 1
                row, checkpoint = fit_or_load(
                    memory, memories[stage - 1] if stage else None, data, stage,
                    arm, order_name, seed, out, device)
                seed_rows.append(row)
                if arm == "current":
                    ref = json.loads((reference_root / data["dataset"] / order_name /
                                      f"mean_x_s{stage}_seed{seed}.json").read_text())
                    check = O.check_archived_fit(row, ref, "mean_x", stage)
                    regression.append(check)
                if stage:
                    # A common history/model isolates only this stage's cross edges.
                    prev = reference_memories[stage - 1]
                    cf_mem = append_variant(prev, blocks[stage], len(prev["x"]), arm,
                                            stage, order_name, device)
                    cp = reference_root / data["dataset"] / order_name / f"mean_x_s{stage-1}_seed{seed}.pt"
                    entry = counterfactual.setdefault(arm, {}).setdefault(str(stage), {
                        "topology": topology_metrics(prev, cf_mem, device, arm), "per_seed": []})
                    entry["per_seed"].append({"seed": seed, "frozen_prior_model_drift":
                        trained_old_embedding_drift(cf_mem, prev, cp, data, device)})
            result = {"arm": arm, "order": order_name, "stage": stage,
                      "rows": len(memory["x"]), "metrics": aggregate(seed_rows),
                      "per_seed": seed_rows}
            stage_rows.append(result)
            all_results.append(result)
            save_json(out / "progress.json", {"results": all_results,
                                               "topology": all_topology})
            print(json.dumps({"event": "stage_done", "dataset": data["dataset"],
                              "order": order_name, "arm": arm, "stage": stage,
                              "f1": result["metrics"]["final_f1"]["mean"]}), flush=True)
        save_json(out / f"{arm}_results.json", stage_rows)
        flat = [r for s in stage_rows for r in s["per_seed"]]
        costs[arm] = dict(build_cost,
                         student_seconds=sum(r["seconds"] for r in flat),
                         student_wall_seconds=sum(r["fit_wall_seconds_including_report_io"] for r in flat),
                         student_steps=sum(r["steps"] for r in flat),
                         student_fits=len(flat),
                         payload=[O.payload_stats(m) for m in memories])
    save_json(out / "canonical_regression.json", regression)
    save_json(out / "common_history_diagnostics.json", counterfactual)
    save_json(out / "results.json", {"results": all_results,
                                      "topology": all_topology, "costs": costs})
    return len(all_results) * len(seeds), fresh_fit_count


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cora", "citeseer"], required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    ap.add_argument("--reference-root", type=Path,
                    default=C.RESULTS / "p1x_x_operator_20260918")
    ap.add_argument("--minilm", default=str(Path(U.__file__).parent / "minilm_from_graphclip"))
    args = ap.parse_args()
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    plan = {
        "version": "p3b-topology-diagnostic-v1",
        "dataset": args.dataset,
        "orders": stage_orders(args.dataset),
        "arms": args.arms,
        "student_seeds": list(STUDENT_SEEDS),
        "rate": 0.05,
        "student_batch": P.STUDENT.batch,
        "student_lr": P.STUDENT.lr,
        "student_steps": "protocol_v3.student_steps(rows)",
        "x_operator": "mean",
        "allocation": "dispersion",
        "text_recipe": "joint_g5_c320",
        "text_operator": "concat",
        "cross_degree": 1,
        "quality_score": "0.5*cos(text)+0.5*cos(x)",
        "random_seed": METHOD_SEED,
        "test_labels_evaluated": False,
        "reference_root": str(args.reference_root),
        "reference_plan_sha256": sha(args.reference_root / args.dataset / "plan.json"),
        "classification": "known-order diagnostic, not independent generalization or promotion",
        "quality_policy": "fixed heuristic, no label/response selection or threshold tuning",
        "primary": "report-dev all-class prompts; final/stage-mean F1, old/new and forgetting",
        "evaluation_graph": "unchanged transductive real graph including unlabeled future/holdout inputs",
        "cost_scope": "cache preprocessing separate; block+topology build and six-seed student separated",
        "promotion": "none from this diagnostic; independent orders/seeds required",
        "python": platform.python_version(), "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(args.device), "num_threads": 2,
        "code_sha256": {str(Path(p)): sha(Path(p)) for p in
                         [__file__, C.__file__, O.__file__, G.__file__, U.__file__, SCA.__file__, P.__file__,
                          Path(C.__file__).with_name("small_comparison_metrics.py")]},
    }
    plan_path = args.output / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("run configuration changed; use a new output directory")
    save_json(plan_path, plan)
    save_json(args.output / "hardware_start.json", {"nvidia_smi": subprocess.check_output(
        ["nvidia-smi"], text=True), "unix_time": time.time()})
    data = prepare(args.dataset, args.minilm, args.device, args.output)
    checked, fresh = 0, 0
    for name, groups in plan["orders"].items():
        count, count_fresh = run_order(data, name, groups, list(args.arms), list(STUDENT_SEEDS),
                  args.output / name, args.device, args.reference_root)
        checked += count
        fresh += count_fresh
    save_json(args.output / "acceptance.json", {
        "complete": True,
        "orders": list(plan["orders"]),
        "arms": list(args.arms),
        "fits": len(plan["orders"]) * len(args.arms) * 3 * len(STUDENT_SEEDS),
        "checkpoint_reloads": checked, "fresh_fits_this_invocation": fresh,
        "test_labels_evaluated": False,
        "control": "fixed Mean-X, dispersion, text, rows, order, student; topology only",
    })
    print("P3B_DONE", flush=True)


if __name__ == "__main__":
    main()
