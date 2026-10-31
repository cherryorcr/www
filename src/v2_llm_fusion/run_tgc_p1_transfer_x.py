"""P1: transfer-aware bounded optimization of the incoming synthetic X.

The experiment starts from sealed C0 memories.  It changes only the current
new rows' X, while keeping text, row budget, mass, row order and C0 topology
fixed.  A differentiable short student rollout is optimized against an
unlabeled current-delta graph/text InfoNCE objective.  The current delta is
split deterministically into construction and probe subgraphs; report labels
are never used to choose X.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import run_tgc_r15_r11_cross as R15
import run_tgc_r4_relational as R4
import tgc_revision_gate as RG
import tgc_student_revision as SR


R, T, D, B = R15.R, R15.T, R15.D, R15.B
INNER_SEEDS = (7800, 7801)
FIELDS_FIXED = ("text_emb", "texts", "mass", "depth", "stage", "edge_index")


def load(path: Path):
    return torch.load(path, map_location="cpu", weights_only=False)


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True), encoding="utf-8")


def scalar(value) -> float:
    return float(value.detach().cpu()) if isinstance(value, torch.Tensor) else float(value)


def transfer_loss(params: dict, x: torch.Tensor, adj: torch.Tensor,
                  text: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    graph = SR.graph(params, x, adj)[rows]
    encoded = F.normalize(F.linear(text[rows], params["text_proj.weight"],
                                   params["text_proj.bias"]), dim=1)
    logits = params["logit_scale"].exp().clamp(max=100) * graph @ encoded.T
    target = torch.arange(len(rows), device=x.device)
    return (F.cross_entropy(logits, target)
            + F.cross_entropy(logits.T, target)) / 2


def batch_plan(n: int, seed: int, batch: int = 16, count: int = 4) -> list[torch.Tensor]:
    if n < 2:
        raise ValueError("transfer split needs at least two raw rows")
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    result = []
    for _ in range(count):
        result.append(torch.randperm(n, generator=gen)[:min(batch, n)])
    return result


def mean_transfer(params: dict, raw: dict, plans: list[torch.Tensor], device: str) -> torch.Tensor:
    x = raw["x"].to(device)
    text = raw["text_emb"].to(device)
    adj = R.C.training_adjacency(raw, device)
    values = [transfer_loss(params, x, adj, text, ids.to(device)) for ids in plans]
    return torch.stack(values).mean()


def split_delta(data: dict, ids: torch.Tensor, stage: int, seed: int) -> tuple[dict, dict]:
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    perm = torch.randperm(len(ids), generator=gen)
    cut = max(2, min(len(ids) - 2, int(round(.8 * len(ids)))))
    train_ids = ids[perm[:cut]]
    probe_ids = ids[perm[cut:]]
    return RG.raw_delta_block(data, train_ids, stage), RG.raw_delta_block(data, probe_ids, stage)


def project_x(value: torch.Tensor, anchor: torch.Tensor, radius: float) -> None:
    with torch.no_grad():
        displacement = value - anchor
        scale = anchor.norm(dim=1, keepdim=True).clamp_min(1e-6) * float(radius)
        norm = displacement.norm(dim=1, keepdim=True).clamp_min(1e-12)
        value.copy_(anchor + displacement * (scale / norm).clamp(max=1.0))


def synthetic_batch_loss(params: dict, x: torch.Tensor, adj: torch.Tensor,
                         text: torch.Tensor, rows: list[torch.Tensor],
                         device: str) -> torch.Tensor:
    values = [transfer_loss(params, x, adj, text, ids.to(device)) for ids in rows]
    return torch.stack(values).mean()


def optimize_x(memory: dict, train_raw: dict, probe_raw: dict, old_n: int,
               device: str, outer_steps: int, inner_steps: int,
               outer_lr: float, radius: float, split_seed: int,
               old_weight: float, old_tolerance: float) -> tuple[dict, dict]:
    started = time.perf_counter()
    base_x = memory["x"].float().to(device)
    new_rows = torch.arange(old_n, len(memory["x"]), device=device)
    if len(new_rows) < 2:
        return memory, {"status": "noop", "new_rows": int(len(new_rows))}
    candidate_x = torch.nn.Parameter(base_x[new_rows].detach().clone())
    optimizer = torch.optim.Adam([candidate_x], lr=float(outer_lr))
    syn_adj = R.C.training_adjacency(memory, device)
    text = memory["text_emb"].float().to(device)
    train_plans = batch_plan(len(train_raw["x"]), split_seed + 11)
    probe_plans = batch_plan(len(probe_raw["x"]), split_seed + 29)
    old_plans = batch_plan(old_n, split_seed + 47)
    old_indices = torch.arange(old_n, device=device)
    baseline_old = []
    base_params = {}
    for seed in INNER_SEEDS:
        base_params[seed] = SR.unroll(base_x, syn_adj, text, seed,
                                      int(inner_steps), differentiable=False)
        baseline_old.append(synthetic_batch_loss(
            base_params[seed], base_x, syn_adj, text, old_plans, device).detach())
    baseline_old = dict(zip(INNER_SEEDS, baseline_old))
    curve = []
    for outer in range(1, int(outer_steps) + 1):
        full_x = base_x.clone()
        full_x[new_rows] = candidate_x
        objectives = []
        old_values = []
        preserve_values = []
        for seed in INNER_SEEDS:
            params = SR.unroll(full_x, syn_adj, text, seed, int(inner_steps),
                               differentiable=True)
            new_value = mean_transfer(params, train_raw, train_plans, device)
            old_value = synthetic_batch_loss(params, full_x, syn_adj, text,
                                             old_plans, device)
            preserve = F.relu(old_value - baseline_old[seed] - float(old_tolerance))
            objectives.append(new_value + float(old_weight) * preserve)
            old_values.append(old_value)
            preserve_values.append(preserve)
        loss = torch.stack(objectives).mean()
        displacement = (candidate_x - base_x[new_rows]).square().mean()
        total = loss + .01 * displacement
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        optimizer.step()
        project_x(candidate_x, base_x[new_rows], radius)
        post_x = base_x.clone()
        post_x[new_rows] = candidate_x
        probe_values = []
        post_old_values = []
        for seed in INNER_SEEDS:
            params = SR.unroll(post_x.detach(), syn_adj, text, seed,
                               int(inner_steps), differentiable=False)
            probe_values.append(mean_transfer(params, probe_raw, probe_plans, device).detach())
            post_old_values.append(synthetic_batch_loss(
                params, post_x.detach(), syn_adj, text, old_plans, device).detach())
        curve.append({
            "outer_step": outer,
            "train_transfer_loss": scalar(loss),
            "probe_transfer_loss": scalar(torch.stack(probe_values).mean()),
            "train_old_synthetic_loss": scalar(torch.stack(old_values).mean()),
            "probe_old_synthetic_loss": scalar(torch.stack(post_old_values).mean()),
            "old_preserve_penalty": scalar(torch.stack(preserve_values).mean()),
            "x_displacement_rms": scalar((candidate_x - base_x[new_rows]).norm(dim=1).mean()),
        })
    result = {key: (value.clone() if isinstance(value, torch.Tensor)
                    else list(value) if isinstance(value, list) else value)
              for key, value in memory.items()}
    result["x"] = base_x.detach().cpu().clone()
    result["x"][old_n:] = candidate_x.detach().cpu()
    audit = {
        "status": "optimized", "old_rows": int(old_n),
        "new_rows": int(len(new_rows)), "inner_seeds": list(INNER_SEEDS),
        "inner_steps": int(inner_steps), "outer_steps": int(outer_steps),
        "outer_lr": float(outer_lr), "radius": float(radius),
        "old_weight": float(old_weight), "old_tolerance": float(old_tolerance),
        "baseline_old_synthetic_loss": {str(k): scalar(v) for k, v in baseline_old.items()},
        "split_seed": int(split_seed), "curve": curve,
        "seconds": float(time.perf_counter() - started),
        "train_raw_rows": int(len(train_raw["x"])),
        "probe_raw_rows": int(len(probe_raw["x"])),
    }
    return result, audit


def assert_candidate(base: dict, candidate: dict, old_n: int) -> None:
    for key in FIELDS_FIXED:
        if key == "texts":
            assert candidate[key] == base[key]
        else:
            assert torch.equal(candidate[key], base[key]), key
    if old_n:
        assert torch.equal(candidate["x"][:old_n], base["x"][:old_n])
    assert len(candidate["x"]) == len(base["x"])
    assert torch.isfinite(candidate["x"]).all()


def evaluate_one(memory: dict, data: dict, stage: int, seed: int,
                 out: Path, device: str) -> dict:
    steps = R.P.student_steps(len(memory["x"]))
    key = f"transfer_x_s{stage}_seed{seed}"
    checkpoint = out / f"{key}.pt"
    row = R.C.fit(memory, int(seed), steps, device, data, stage,
                  checkpoint=checkpoint)
    row.update({"arm": "transfer_x", "stage": int(stage), "seed": int(seed),
                "support_memory_sha256": T.memory_hash(memory),
                "checkpoint_sha256": R.C.sha(checkpoint)})
    save_json(out / f"{key}.json", row)
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--c0-root", required=True, type=Path)
    ap.add_argument("--dataset", choices=("cora", "citeseer"), nargs="+",
                    default=["cora", "citeseer"])
    ap.add_argument("--order", nargs="+", default=["order19601", "order19602", "order19603"])
    ap.add_argument("--stage", nargs="+", type=int, default=[1, 2])
    ap.add_argument("--seed", nargs="+", type=int, default=list(range(150, 156)))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--minilm", default="/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip")
    ap.add_argument("--text-cache-root", type=Path,
                    default=Path("/home/pc/research/TAG_condensation/results/p5_rate_curve_small_20260919"))
    ap.add_argument("--outer-steps", type=int, default=8)
    ap.add_argument("--inner-steps", type=int, default=16)
    ap.add_argument("--outer-lr", type=float, default=.01)
    ap.add_argument("--radius", type=float, default=.05)
    ap.add_argument("--old-weight", type=float, default=1.0)
    ap.add_argument("--old-tolerance", type=float, default=.01)
    args = ap.parse_args()
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "name": "tgc-p1-transfer-aware-x-v1",
        "datasets": args.dataset, "orders": args.order, "stages": args.stage,
        "student_seeds": args.seed, "inner_seeds": list(INNER_SEEDS),
        "outer_steps": args.outer_steps, "inner_steps": args.inner_steps,
        "outer_lr": args.outer_lr, "radius": args.radius,
        "old_weight": args.old_weight, "old_tolerance": args.old_tolerance,
        "base": "sealed C0 memory from R15",
        "changed": "incoming synthetic X only",
        "fixed": ["old X", "text", "mass", "depth", "stage", "C0 topology",
                  "rows", "student protocol"],
        "objective": "unlabeled current-delta graph/text InfoNCE after differentiable student rollout",
        "current_delta_split": "deterministic 80/20 train/probe; no report labels",
        "historical_raw_reads_for_construction": 0,
        "report_labels_used_for_construction": False,
    }
    save_json(args.output / "manifest.json", manifest)
    all_rows = []
    for dataset in args.dataset:
        cache = args.text_cache_root / f"{dataset}_r005" / "text_emb_pool_only.pt"
        data = R.R.load_data(dataset, args.minilm, args.device, cache)
        for order in args.order:
            ids, budgets = D.set_stream(data, order)
            for stage in args.stage:
                base = load(args.c0_root / dataset / order / f"c0_s{stage}_memory.pt")
                old_n = 0 if stage == 0 else len(load(
                    args.c0_root / dataset / order / f"c0_s{stage-1}_memory.pt")["x"])
                train_raw, probe_raw = split_delta(data, ids[stage], stage,
                                                   260924 + 100 * stage + sum(ord(c) for c in order))
                candidate, audit = optimize_x(
                    base, train_raw, probe_raw, old_n, args.device,
                    args.outer_steps, args.inner_steps, args.outer_lr,
                    args.radius, 260924 + stage, args.old_weight,
                    args.old_tolerance)
                assert_candidate(base, candidate, old_n)
                cell = args.output / dataset / order
                cell.mkdir(parents=True, exist_ok=True)
                torch.save(candidate, cell / f"transfer_x_s{stage}_memory.pt")
                save_json(cell / f"transfer_x_s{stage}_construction.json", audit)
                save_json(cell / f"c0_s{stage}_construction_reference.json", {
                    "source": str(args.c0_root / dataset / order / f"c0_s{stage}_memory.pt"),
                    "memory_sha256": T.memory_hash(base), "old_rows": old_n,
                    "rows": len(base["x"]), "budgets": budgets,
                })
                base_rows = []
                for seed in args.seed:
                    candidate_row = evaluate_one(candidate, data, stage, seed,
                                                 cell, args.device)
                    base_row = json.loads((args.c0_root / dataset / order /
                                           f"c0_s{stage}_seed{seed}.json").read_text())
                    cmet, bmet = candidate_row["metrics"]["all"], base_row["metrics"]["all"]
                    merged = {
                        "dataset": dataset, "order": order, "stage": stage,
                        "seed": seed, "candidate_final_f1": cmet["cumulative_f1"],
                        "base_final_f1": bmet["cumulative_f1"],
                        "delta_final_f1": cmet["cumulative_f1"] - bmet["cumulative_f1"],
                        "candidate_new_f1": cmet["new_f1"], "base_new_f1": bmet["new_f1"],
                        "delta_new_f1": cmet["new_f1"] - bmet["new_f1"],
                        "candidate_old_f1": cmet["old_f1"], "base_old_f1": bmet["old_f1"],
                        "delta_old_f1": cmet["old_f1"] - bmet["old_f1"],
                        "construction_seconds": audit["seconds"],
                        "probe_transfer_loss_final": audit["curve"][-1]["probe_transfer_loss"],
                    }
                    base_rows.append(merged)
                    all_rows.append(merged)
                    print(json.dumps(merged), flush=True)
                save_json(cell / f"summary_s{stage}.json", {
                    "dataset": dataset, "order": order, "stage": stage,
                    "construction": audit, "rows": base_rows,
                    "mean_delta_final_f1": float(np.mean([r["delta_final_f1"] for r in base_rows])),
                    "mean_delta_new_f1": float(np.mean([r["delta_new_f1"] for r in base_rows])),
                })
    fields = sorted({key for row in all_rows for key in row})
    with (args.output / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)
    save_json(args.output / "completion.json", {"status": "complete", "rows": len(all_rows)})
    print(json.dumps({"status": "complete", "rows": len(all_rows)}), flush=True)


if __name__ == "__main__":
    main()
