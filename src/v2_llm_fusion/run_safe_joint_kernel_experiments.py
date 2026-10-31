"""Frozen small-graph follow-up: topology, components, and byte-fair signatures."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import torch

import run_frozen_baseline_joint_kernel as F
import safe_joint_kernel_append as S

C, G, U, P, J, R = F.C, F.G, F.U, F.P, F.J, F.R
ARMS = ("c0", "k_center", "joint_kernel", "degree_balanced", "coverage",
        "operator_stable", "drift_gated")
EXTRA_ARMS = ("kernel_x", "kernel_t", "kernel_g", "kernel_xt", "node_fp16",
              "node_sketch64", "byte_fp32", "byte_fp16", "byte_sketch64")
PROTOCOL = "SAFE_JOINT_KERNEL_V2_FROZEN_PROTOCOL_ZH_2026-09-20.md"


def sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def source_hashes():
    paths = [__file__, S.__file__, F.__file__, J.__file__, C.__file__, U.__file__,
             P.__file__, G.__file__, R.__file__, F.M.__file__, F.SCA.__file__,
             Path(__file__).with_name("run_multiscale_asm.py"),
             Path(__file__).resolve().parents[2] / "docs/current" / PROTOCOL]
    return {Path(p).name: C.sha(p) for p in paths}


def payload(memory):
    value = F.payload(memory)
    value["signature_metadata_bytes"] = S.signature_metadata_bytes(memory)
    value["logical_bytes"] += value["signature_metadata_bytes"]
    return value


def gini(values):
    values = torch.sort(values.float().flatten()).values
    if not len(values) or float(values.sum()) <= 0:
        return 0.
    index = torch.arange(1, len(values) + 1, dtype=torch.float32)
    return float(((2 * index - len(values) - 1) * values).sum() / (len(values) * values.sum()))


def topology_metrics(old, current):
    n = 0 if old is None else len(old["x"])
    total = len(current["x"])
    edge = current["edge_index"].long().cpu()
    internal = edge[:, (edge >= n).all(0)] - n
    internal_degree = torch.bincount(internal[0], minlength=total - n).float()
    degree = torch.bincount(edge[0], minlength=total).float()
    cross = edge[:, ((edge[0] < n) != (edge[1] < n))]
    # Count unique endpoint pairs once, independently of direction/multiplicity.
    pairs = {(min(i, j), max(i, j)) for i, j in cross.T.tolist()}
    old_counts = torch.bincount(torch.tensor([i for i, _ in pairs], dtype=torch.long), minlength=n).float()
    old_norm = old_prop = effective_prop = 0.
    if old is not None:
        previous_adj = C.training_adjacency(old, "cpu").to_dense()
        current_adj = C.training_adjacency(current, "cpu").to_dense()
        old_block = current_adj[:n, :n]
        previous_message = previous_adj @ old["x"].float()
        old_norm = float((old_block - previous_adj).norm() / previous_adj.norm().clamp_min(1e-12))
        old_prop = float(((old_block - previous_adj) @ old["x"].float()).norm() /
                         previous_message.norm().clamp_min(1e-12))
        effective_prop = float((current_adj[:n] @ current["x"].float() - previous_message).norm() /
                               previous_message.norm().clamp_min(1e-12))
    return {
        "old_rows": n, "new_rows": total - n, "old_normalized_drift": old_norm,
        "old_propagation_drift": old_prop, "old_effective_message_drift": effective_prop,
        "cross_edge_entries": cross.shape[1], "cross_unique_pairs": len(pairs),
        "cross_old_endpoint_gini": gini(old_counts),
        "max_old_cross_degree": int(old_counts.max()) if n else 0,
        "new_internal_degree_gini": gini(internal_degree),
        "max_new_internal_degree": int(internal_degree.max()) if len(internal_degree) else 0,
        "new_total_degree_gini": gini(degree[n:]),
        "max_new_degree": int(degree[n:].max()) if total > n else 0,
    }


def copy_block(memory, keep):
    result = {k: memory[k][keep:].clone() for k in ("x", "text_emb", "mass", "depth", "stage")}
    result["texts"] = list(memory["texts"][keep:])
    return result


def c0_append(old, block, device):
    out = F.SCA.append_memory(old, block, 0 if old is None else len(old["x"]),
        topology_mode="frozen_old", cross_degree=1, refresh_mode="frozen_old", device=device)
    if "graph_signature" in block:
        out["graph_signature"] = block["graph_signature"] if old is None else torch.cat((old["graph_signature"], block["graph_signature"]))
    if "signature_config" in block:
        out["signature_config"] = dict(block["signature_config"])
    return out


def gated_append(old, block, device):
    keep = 0 if old is None else len(old["x"])
    candidate, audit = S.append_variant(old, block, keep, "joint_kernel")
    if old is None:
        return candidate, dict(audit, gate_fallback=False, gate_scope="no old block")
    reference = c0_append(old, block, device)
    cand, ref = topology_metrics(old, candidate), topology_metrics(old, reference)
    # Both alternatives extend the SAME previous memory, so history is not a confound.
    fallback = any(cand[k] > ref[k] + 1e-8 for k in
                   ("old_normalized_drift", "old_effective_message_drift"))
    audit.update(gate_fallback=fallback, gate_candidate=cand, gate_reference=ref,
                 gate_scope="same old memory, C0 append fallback, no student/report labels")
    return (reference if fallback else candidate), audit


def signature_mode(arm):
    return arm.split("_", 1)[1] if arm.startswith(("node_", "byte_")) else "fp32"


def build_memories(data, stages, budgets, arm, out, device, c0_memories, c0_audits):
    if arm == "c0":
        memories, audits = c0_memories, c0_audits
    elif arm == "k_center":
        memories, audits = F.construction(data, stages, budgets, arm, out, device)
    else:
        memories, audits, old = [], [], None
        for stage, ids in enumerate(stages):
            path = out / f"{arm}_s{stage}_memory.pt"
            meta = out / f"{arm}_s{stage}_construction.json"
            keep = 0 if old is None else len(old["x"])
            if path.exists() and meta.exists():
                memory = torch.load(path, map_location="cpu", weights_only=False)
                audit = json.loads(meta.read_text())
                if (F.J_hash(memory) != audit["content_sha256"] or
                        memory.get("signature_config") != audit["signature_config"]):
                    raise AssertionError("cached memory mismatch")
            else:
                sync(device)
                started = time.perf_counter()
                byte_limit = payload(c0_memories[stage])["logical_bytes"]
                byte_mode = arm.startswith("byte_")
                attempts = []
                maximum = min(len(ids), budgets[stage] - keep)
                minimum = len(data["stage_classes"][stage])
                for nnew in range(maximum, minimum - 1, -1):
                    if byte_mode:
                        base = U.build_stage_block(data, ids, nnew, data["encoder"], stage,
                            "joint_g5_c320", "concat", device, "dispersion", data["stage_classes"][stage])
                    else:
                        base = copy_block(c0_memories[stage], keep)
                    block = J.build_joint_block(U, data, ids, nnew, data["encoder"], stage, device, base=base)
                    source_groups = block.pop("audit_source_groups")
                    block = S.compress_signature(block, signature_mode(arm))
                    if arm == "drift_gated":
                        memory, extra = gated_append(old, block, device)
                    else:
                        variant = "joint_kernel" if arm.startswith(("node_", "byte_")) else arm
                        memory, extra = S.append_variant(old, block, keep, variant)
                    attempts.append({"new_rows": nnew, "bytes": payload(memory)["logical_bytes"]})
                    if not byte_mode or attempts[-1]["bytes"] <= byte_limit:
                        break
                else:
                    raise ValueError("byte budget cannot cover all arriving classes")
                sync(device)
                exclusive = time.perf_counter() - started
                shared = c0_audits[stage]["build_seconds"]
                audit = {"arm": arm, "stage": stage, "keep": keep, "new": len(memory["x"]) - keep,
                    "build_seconds": exclusive + shared, "exclusive_build_seconds": exclusive,
                    "shared_c0_seconds": shared, "byte_limit": byte_limit if byte_mode else None,
                    "byte_search_attempts": attempts, "payload": payload(memory),
                    "content_sha256": F.J_hash(memory), "tensor_sha256": C.tensor_hash(memory),
                    "signature_config": memory.get("signature_config"),
                    "topology": topology_metrics(old, memory), "variant_audit": extra}
                torch.save(memory, path)
                audit["serialized_file_bytes"] = path.stat().st_size
                C.save_json(meta, audit)
                C.save_json(out / f"{arm}_s{stage}_sources.json", {
                    "source_groups": source_groups, "current_delta": ids.tolist(),
                    "scope": "offline audit only, not persistent memory or replay"})
            if not arm.startswith("byte_"):
                control = c0_memories[stage]
                for field in ("x", "text_emb", "mass", "depth", "stage"):
                    if not torch.equal(memory[field], control[field]):
                        raise AssertionError(f"candidate changed C0 {field}")
                if memory["texts"] != control["texts"] or memory["edge_index"].shape != control["edge_index"].shape:
                    raise AssertionError("candidate changed C0 text/edge budget")
            C.assert_frozen(old, memory)
            if old is not None:
                for field in ("depth", "graph_signature"):
                    if not torch.equal(old[field], memory[field][:keep]):
                        raise AssertionError(f"old {field} changed")
            memories.append(memory)
            audits.append(audit)
            old = memory
    previous = None
    for stage, (memory, audit) in enumerate(zip(memories, audits)):
        audit.update(payload=payload(memory), topology=topology_metrics(previous, memory),
                     content_sha256=F.J_hash(memory), tensor_sha256=C.tensor_hash(memory))
        C.save_json(out / f"{arm}_s{stage}_construction.json", audit)
        previous = memory
    return memories, audits


def fit_or_load(memory, data, stage, seed, arm, out, device, steps):
    key = f"{arm}_s{stage}_seed{seed}"
    row_path, checkpoint = out / f"{key}.json", out / f"{key}.pt"
    if row_path.exists() and checkpoint.exists():
        row = json.loads(row_path.read_text())
        if (row["support_sha256"] != C.tensor_hash(memory) or row["steps"] != steps
                or row["checkpoint_sha256"] != C.sha(checkpoint)):
            raise ValueError(f"stale fit: {key}")
        return row
    started = time.perf_counter()
    row = C.fit(memory, seed, steps, device, data, stage, checkpoint=checkpoint)
    row["metrics"] = json.loads(json.dumps(row["metrics"]))
    row["checkpoint_sha256"] = F.reload_prediction(checkpoint, row, data, stage, device)
    row.update(arm=arm, stage=stage, dataset=data["dataset"],
        fit_wall_seconds_including_report_io=time.perf_counter() - started,
        effective_batch=min(P.STUDENT.batch, len(memory["x"])))
    C.save_json(row_path, row)
    return row


def run_cell(data, groups, rate, nstages, order, arms, seeds, out, device, frozen):
    stages, budgets = F.set_stream(data, groups, rate)
    out.mkdir(parents=True, exist_ok=True)
    config = dict(frozen, order=order, stage_count=nstages, rate=rate, arms=arms,
        stage_classes=groups, budgets=budgets, pool_counts=[len(v) for v in stages],
        source_ids_sha256=U.sha256_json([v.tolist() for v in stages]),
        steps=[P.student_steps(b) for b in budgets],
        byte_control_steps="same as full-row C0, not reduced-row schedule")
    F.lock(out / "manifest.json", config)
    c0_memories, c0_audits = F.construction(data, stages, budgets, "c0", out, device)
    results, audits_by_arm = [], {}
    for arm in arms:
        memories, audits = build_memories(data, stages, budgets, arm, out, device, c0_memories, c0_audits)
        audits_by_arm[arm] = audits
        for stage, memory in enumerate(memories):
            rows = [fit_or_load(memory, data, stage, seed, arm, out, device, config["steps"][stage])
                    for seed in seeds]
            summary = {"arm": arm, "stage": stage, "rows": len(memory["x"]),
                "steps": config["steps"][stage], "metrics": C.aggregate(rows), "per_seed": rows,
                "topology": audits[stage]["topology"]}
            results.append(summary)
            C.save_json(out / "progress.json", {"manifest": config, "results": results, "audits": audits_by_arm})
            print(json.dumps({"event": "stage_done", "dataset": data["dataset"], "cell": out.name,
                "arm": arm, "stage": stage, "f1": summary["metrics"]["all"]["cumulative_f1"]["mean"]}), flush=True)
    C.save_json(out / "results.json", {"manifest": config, "results": results, "audits": audits_by_arm})
    return len(arms) * nstages * len(seeds)


def load_data(dataset, minilm, device, cache):
    data = R.load_data(dataset, minilm, device, cache)
    data["encoder"].encode(data["prompts"])
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["cora", "citeseer"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rates", type=float, nargs="+", default=[.01, .05])
    parser.add_argument("--stages", type=int, nargs="+", default=[3, 5])
    parser.add_argument("--orders", nargs="+", default=["canonical", "order19102"])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(P.STUDENT_SEEDS))
    parser.add_argument("--minilm", default="/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip")
    parser.add_argument("--cache-root", type=Path, default=Path("/home/pc/research/TAG_condensation/results"))
    args = parser.parse_args()
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    cache = args.cache_root / "p5_rate_curve_small_20260919" / f"{args.dataset}_r005" / "text_emb_pool_only.pt"
    cells = []
    for count in args.stages:
        for order in args.orders:
            for rate in args.rates:
                extra = count == 3 and order == "canonical" and rate == .05
                cells.append({"count": count, "order": order, "rate": rate,
                    "name": f"s{count}_{order}_r{int(round(rate * 100)):02d}",
                    "arms": list(ARMS + EXTRA_ARMS if extra else ARMS)})
    freeze = {"version": "safe-joint-kernel-v2", "dataset": args.dataset, "cells": cells,
        "seeds": args.seeds, "source_hashes": source_hashes(), "torch": torch.__version__,
        "device": args.device, "data_sha256": C.sha(U.DATA_CFG[args.dataset]["path"]),
        "cache_sha256": C.sha(cache), "student_batch": P.STUDENT.batch,
        "report_metric": "pooled all-prompt report-dev Macro-F1", "test_labels_evaluated": False,
        "no_report_based_construction": True, "development_conditions_not_independent_confirmation": True,
        "expected_fits": sum(c["count"] * len(c["arms"]) * len(args.seeds) for c in cells)}
    F.lock(args.output / "freeze.json", freeze)
    started = time.perf_counter()
    data = load_data(args.dataset, args.minilm, args.device, cache)
    sync(args.device)
    C.save_json(args.output / "preprocessing.json", {"seconds": time.perf_counter() - started,
        "scope": "shared data/prompt/cache load; initial pool text encoding not charged"})
    count = 0
    for cell in cells:
        count += run_cell(data, F.stage_groups(args.dataset, cell["count"], cell["order"]),
            cell["rate"], cell["count"], cell["order"], cell["arms"], args.seeds,
            args.output / cell["name"], args.device, freeze)
    if count != freeze["expected_fits"]:
        raise AssertionError("fit count mismatch")
    C.save_json(args.output / "completion.json", {"complete": True, "fits": count,
        "cells": len(cells), "freeze_sha256": C.sha(args.output / "freeze.json"),
        "elapsed_seconds": time.perf_counter() - started,
        "completed_at": datetime.now(timezone.utc).isoformat()})
    print("SAFE_JOINT_KERNEL_DONE", flush=True)


if __name__ == "__main__":
    main()
