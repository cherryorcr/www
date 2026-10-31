"""Frozen Phase-1 baseline and full Phase-2 joint-kernel experiment matrix."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import time

import torch

import run_small_comparison_audited as C
import small_comparison_metrics as M
import joint_kernel_append as J
import run_p0_random_synthetic as R

U, G, SCA, P = C.U, C.G, C.SCA, C.P
MATRIX_ARMS = ["c0", "k_center", "recluster", "joint_kernel"]
BASELINE_ARMS = ["random_synthetic"]


def sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def lock(path, value):
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError(f"frozen configuration changed: {path}")
    C.save_json(path, value)


def source_hashes():
    files = (
        __file__, C.__file__, M.__file__, J.__file__, R.__file__, U.__file__,
        G.__file__, SCA.__file__, P.__file__,
        Path(__file__).with_name("run_multiscale_asm.py"),
        Path(__file__).resolve().parents[2] / "docs/current/PHASE12_C0_JOINT_KERNEL_FROZEN_PROTOCOL_ZH_2026-09-20.md")
    return {Path(p).name: C.sha(p) for p in files}


def stage_groups(dataset, count, order):
    n = 7 if dataset == "cora" else 6
    sizes = ([2, 2, n - 4] if count == 3 else [n - 4, 1, 1, 1, 1])
    if count not in (3, 5):
        raise ValueError("only preregistered 3/5-stage chains are allowed")
    classes = list(range(n))
    if order != "canonical":
        random.Random(int(order.removeprefix("order"))).shuffle(classes)
    groups, at = [], 0
    for width in sizes:
        groups.append(sorted(classes[at:at + width]))
        at += width
    return P.canonicalize_class_groups(groups)


def set_stream(data, groups, rate):
    data["stage_classes"] = groups
    G.STAGE_CLASSES = groups
    ids = [torch.where(data["pool_mask"] & torch.isin(
        data["y"], torch.tensor(group)))[0].sort().values for group in groups]
    budgets = [math.floor(rate * sum(map(len, ids[:i + 1]))) for i in range(len(ids))]
    if any(b <= (budgets[i-1] if i else 0) for i, b in enumerate(budgets)):
        raise ValueError("the registered rate must add rows at every stage")
    return ids, budgets


def payload(memory):
    fields = {key: v.numel() * v.element_size() for key, v in memory.items()
              if isinstance(v, torch.Tensor)}
    text_bytes = sum(len(t.encode("utf-8")) for t in memory["texts"])
    return {"tensor_bytes": fields, "text_utf8_bytes": text_bytes,
            "logical_bytes": sum(fields.values()) + text_bytes,
            "rows": len(memory["x"]), "edge_entries": memory["edge_index"].shape[1]}


def adjacency_drift(old, memory):
    if old is None:
        return None
    n = len(old["x"])
    before = C.training_adjacency(old, "cpu").to_dense()
    after = C.training_adjacency(memory, "cpu").to_dense()[:n, :n]
    return float((after - before).norm() / before.norm().clamp_min(1e-12))


def construction(data, stages, budgets, arm, out, device, c0_root=None):
    if arm in ("k_center", "recluster"):
        memories, audits = C.construct(data, arm, stages, budgets, out, device)
        for memory, audit in zip(memories, audits):
            audit["payload"] = payload(memory)
        return memories, audits
    memories, audits, old = [], [], None
    for stage, ids in enumerate(stages):
        path = out / f"{arm}_s{stage}_memory.pt"
        meta = out / f"{arm}_s{stage}_construction.json"
        keep = 0 if old is None else len(old["x"])
        nnew = budgets[stage] - keep
        if path.exists() and meta.exists():
            memory = torch.load(path, map_location="cpu", weights_only=False)
            audit = json.loads(meta.read_text())
            if J_hash(memory) != audit["content_sha256"]:
                raise AssertionError("cached support changed")
        else:
            sync(device)
            started = time.perf_counter()
            extra = {}
            if arm == "random_synthetic":
                block, extra = R.random_synthetic_block(data, ids, nnew, stage, 6400 + 101 * stage)
            elif arm == "c0":
                block = U.build_stage_block(data, ids, nnew, data["encoder"], stage,
                    "joint_g5_c320", "concat", device, "dispersion", data["stage_classes"][stage])
            else:
                c0 = torch.load(out / f"c0_s{stage}_memory.pt", map_location="cpu", weights_only=False)
                base = {k: c0[k][keep:] for k in ("x", "text_emb", "mass", "depth", "stage")}
                base["texts"] = c0["texts"][keep:]
                block = J.build_joint_block(U, data, ids, nnew, data["encoder"], stage, device, base=base)
                groups = block.pop("audit_source_groups")
                C.save_json(out / f"joint_kernel_s{stage}_sources.json", {
                    "source_groups": groups, "current_delta": ids.tolist(),
                    "scope": "offline audit only; never persistent synthetic memory"})
            if arm == "joint_kernel":
                memory, extra = J.append_joint(old, block, keep)
                for key in ("x", "text_emb", "mass", "depth", "stage"):
                    if not torch.equal(memory[key], c0[key]):
                        raise AssertionError(f"topology-only candidate changed {key}")
                if memory["texts"] != c0["texts"]:
                    raise AssertionError("topology-only candidate changed text")
                if memory["edge_index"].shape != c0["edge_index"].shape:
                    raise AssertionError("edge budget differs from C0")
            else:
                memory = SCA.append_memory(old, block, keep, topology_mode="frozen_old",
                    cross_degree=1, refresh_mode="frozen_old", device=device)
            sync(device)
            exclusive_build = time.perf_counter() - started
            # Candidate reuses C0 rows; charge the full shared construction to
            # both methods rather than making the candidate's base rows free.
            shared = 0.0
            if arm == "joint_kernel":
                shared = json.loads((out / f"c0_s{stage}_construction.json").read_text())["build_seconds"]
            audit = {"arm": arm, "stage": stage, "keep": keep, "new": nnew,
                "build_seconds": exclusive_build + shared,
                "exclusive_build_seconds": exclusive_build, "shared_c0_seconds": shared,
                "payload": payload(memory), "content_sha256": J_hash(memory),
                "tensor_sha256": C.tensor_hash(memory), "construction": extra,
                "normalized_old_block_relative_drift": adjacency_drift(old, memory)}
            if arm == "c0" and c0_root is not None:
                archive = torch.load(c0_root / f"append_fixed_s{stage}_memory.pt",
                                     map_location="cpu", weights_only=False)
                if J_hash(archive) != J_hash(memory):
                    raise AssertionError("C0 canonical artifact regression")
                audit["canonical_archive_equal"] = True
            torch.save(memory, path)
            C.save_json(meta, audit)
        if len(memory["x"]) != budgets[stage]:
            raise AssertionError("row budget mismatch")
        C.assert_frozen(old, memory)
        memories.append(memory)
        audits.append(audit)
        old = memory
    return memories, audits


def J_hash(memory):
    import hashlib
    return hashlib.sha256((C.tensor_hash(memory) + json.dumps(memory["texts"],
        ensure_ascii=True)).encode()).hexdigest()


@torch.no_grad()
def reload_prediction(checkpoint, row, data, stage, device):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if C.tensor_hash(saved["state"]) != row["end_state_sha256"]:
        raise AssertionError("checkpoint hash mismatch")
    model = G.CLIPStudent(data["x"].shape[1]).to(device)
    model.load_state_dict(saved["state"])
    model.eval()
    mask = data["report_dev_mask"]
    logits = model.logits(data["x_device"], data["real_adj"], data["prompt_emb"].to(device)).cpu()[mask]
    target = data["y"][mask]
    metrics = M.score_logits(logits, target.numpy(), data["stage_classes"], stage)
    if json.loads(json.dumps(metrics)) != row["metrics"]:
        raise AssertionError("checkpoint reload metrics differ")
    saved["prediction"] = {"logits": logits, "target": target,
        "original_ids": data["original_ids"][mask], "stage_classes": data["stage_classes"],
        "stage": stage, "split": "report_dev"}
    torch.save(saved, checkpoint)
    return C.sha(checkpoint)


def run_cell(data, stages, budgets, out, device, arms, seeds, config, c0_root):
    out.mkdir(parents=True, exist_ok=True)
    lock(out / "manifest.json", config)
    results, construction_audits = [], {}
    for arm in arms:
        memories, audits = construction(data, stages, budgets, arm, out, device, c0_root)
        construction_audits[arm] = audits
        for stage, memory in enumerate(memories):
            scores = []
            for seed in seeds:
                key = f"{arm}_s{stage}_seed{seed}"
                rowpath, checkpoint = out / f"{key}.json", out / f"{key}.pt"
                steps = P.student_steps(budgets[stage])
                if rowpath.exists() and checkpoint.exists():
                    row = json.loads(rowpath.read_text())
                    if (row["support_sha256"] != C.tensor_hash(memory)
                            or row["checkpoint_sha256"] != C.sha(checkpoint)
                            or row["steps"] != steps):
                        raise AssertionError("stale fit")
                else:
                    started = time.perf_counter()
                    row = C.fit(memory, seed, steps, device, data, stage, checkpoint=checkpoint)
                    row["metrics"] = json.loads(json.dumps(row["metrics"]))
                    row["checkpoint_sha256"] = reload_prediction(checkpoint, row, data, stage, device)
                    row.update(arm=arm, stage=stage, dataset=data["dataset"],
                        fit_reload_io_seconds=time.perf_counter() - started,
                        effective_batch=min(P.STUDENT.batch, len(memory["x"])))
                    C.save_json(rowpath, row)
                scores.append(row)
            summary = {"arm": arm, "stage": stage, "steps": steps,
                "rows": len(memory["x"]), "metrics": C.aggregate(scores), "per_seed": scores}
            results.append(summary)
            C.save_json(out / "progress.json", {"manifest": config, "results": results, "audits": construction_audits})
            print(json.dumps({"cell": out.name, "dataset": data["dataset"], "arm": arm,
                "stage": stage, "f1": summary["metrics"]["all"]["cumulative_f1"]["mean"]}), flush=True)
    C.save_json(out / "results.json", {"manifest": config, "results": results, "audits": construction_audits})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cora", "citeseer"], required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--phase", choices=["baseline", "matrix"], default="matrix")
    ap.add_argument("--rates", type=float, nargs="+", default=[.01, .05, .10])
    ap.add_argument("--stages", type=int, nargs="+", default=[3, 5])
    ap.add_argument("--orders", nargs="+", default=["canonical", "order19101", "order19102"])
    ap.add_argument("--seeds", type=int, nargs="+", default=list(P.STUDENT_SEEDS))
    ap.add_argument("--minilm", default="/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip")
    ap.add_argument("--archive-root", type=Path, default=Path("/home/pc/research/TAG_condensation/results"))
    args = ap.parse_args()
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.phase == "baseline":
        args.rates, args.stages, args.orders = [.05], [3], ["canonical"]
    arms = BASELINE_ARMS if args.phase == "baseline" else MATRIX_ARMS
    freeze = {"name": "phase12-c0-joint-kernel-v1", "dataset": args.dataset,
        "phase": args.phase, "rates": args.rates, "stages": args.stages,
        "orders": args.orders, "seeds": args.seeds, "arms": arms,
        "source_hashes": source_hashes(), "torch": torch.__version__,
        "device": args.device, "student_batch": P.STUDENT.batch,
        "kernel_weights": list(J.KERNEL_WEIGHTS), "report_metric": "pooled all-prompt Macro-F1",
        "test_labels_evaluated": False, "no_report_based_selection": True}
    lock(args.output / "freeze.json", freeze)
    started = time.perf_counter()
    cache = args.archive_root / "p5_rate_curve_small_20260919" / f"{args.dataset}_r005" / "text_emb_pool_only.pt"
    data = R.load_data(args.dataset, args.minilm, args.device, cache)
    sync(args.device)
    # Warm the same encoder kernels for both C0 and candidate constructions.
    data["encoder"].encode(data["prompts"])
    sync(args.device)
    preprocessing = {"load_cache_prompt_seconds": time.perf_counter() - started,
        "data_sha256": C.sha(U.DATA_CFG[args.dataset]["path"]), "cache_sha256": C.sha(cache),
        "scope": "shared preprocessing; pool text encoding is cached and excluded from method build costs"}
    C.save_json(args.output / "preprocessing.json", preprocessing)
    expected = 0
    for nstages in args.stages:
        for order in args.orders:
            groups = stage_groups(args.dataset, nstages, order)
            for rate in args.rates:
                stage_ids, budgets = set_stream(data, groups, rate)
                name = f"s{nstages}_{order}_r{int(round(rate * 100)):02d}"
                config = dict(freeze, rate=rate, stage_classes=groups, budgets=budgets,
                    pool_counts=[len(ids) for ids in stage_ids],
                    source_ids_sha256=U.sha256_json([v.tolist() for v in stage_ids]))
                c0_root = (args.archive_root / "p5_rate_curve_small_20260919" / f"{args.dataset}_r005"
                           if nstages == 3 and order == "canonical" and rate == .05 else None)
                run_cell(data, stage_ids, budgets, args.output / name, args.device,
                         arms, args.seeds, config, c0_root)
                expected += nstages * len(arms) * len(args.seeds)
    C.save_json(args.output / "completion.json", {"expected_and_completed_fits": expected,
        "completed_at": datetime.now(timezone.utc).isoformat(), "freeze_sha256": C.sha(args.output / "freeze.json")})
    print("FROZEN_PHASE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
