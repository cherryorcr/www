"""Shared-P plus bounded-X joint condensation.

This runner is the follow-up to message-conditioned X.  A single legal soft
assignment P materializes the incoming text and binary topology.  The
P-weighted X is the first ablation; a second arm jointly optimizes only the
incoming X through a short differentiable Student rollout while keeping the
same P-derived text/topology.  Construction uses only the old synthetic
memory and the current raw delta; report labels are never used.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import run_tgc_joint_student_response as V1
import run_tgc_r11_c0_anchor as R11
import run_tgc_two_prototype_smoke as B
import tgc_shared_assignment as J
import tgc_relational_response as M
import tgc_student_revision as SR
import run_tgc_two_prototype_role_permutation as Base


ARMS = ("c0", "shared_p", "shared_px")
OUTER_SEEDS = (8200, 8201)


def load(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True), encoding="utf-8")


def dense_normalized(edge_index: torch.Tensor, n: int, device: str,
                     edge_weight: torch.Tensor | None = None) -> torch.Tensor:
    edge = edge_index.long().cpu()
    adj = torch.zeros((n, n), dtype=torch.float32, device=device)
    if edge.numel():
        values = (torch.ones(edge.shape[1], dtype=torch.float32)
                  if edge_weight is None else edge_weight.float().cpu())
        adj[edge[0].to(device), edge[1].to(device)] = values.to(device)
    adj.fill_diagonal_(1.0)
    degree = adj.sum(1).clamp_min(1.0)
    scale = degree.rsqrt()
    return scale[:, None] * adj * scale[None, :]


def batches(n: int, seed: int, count: int = 4,
           batch: int = 16) -> list[torch.Tensor]:
    if n < 2:
        raise ValueError("need at least two rows for a transfer batch")
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    return [torch.randperm(n, generator=gen)[:min(batch, n)]
            for _ in range(count)]


def transfer_loss(params: dict, x: torch.Tensor, adj: torch.Tensor,
                  text: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    graph = SR.graph(params, x, adj)[rows]
    encoded = F.normalize(F.linear(text[rows], params["text_proj.weight"],
                                   params["text_proj.bias"]), dim=1)
    logits = params["logit_scale"].exp().clamp(max=100) * graph @ encoded.T
    target = torch.arange(len(rows), device=x.device)
    return (F.cross_entropy(logits, target)
            + F.cross_entropy(logits.T, target)) / 2


def mean_transfer(params: dict, x: torch.Tensor, adj: torch.Tensor,
                  text: torch.Tensor, plans: list[torch.Tensor]) -> torch.Tensor:
    values = [transfer_loss(params, x, adj, text, ids.to(x.device))
              for ids in plans]
    return torch.stack(values).mean()


def mean_synthetic(params: dict, x: torch.Tensor, adj: torch.Tensor,
                   text: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    return transfer_loss(params, x, adj, text, rows)


def p_from_logits(logits: torch.Tensor, allowed: torch.Tensor,
                  uniform: torch.Tensor, floor: float) -> torch.Tensor:
    soft = logits.masked_fill(~allowed, -1e9).softmax(1)
    return (1.0 - floor) * soft + floor * uniform


def material_base(parent: dict, c0: dict, old_n: int) -> dict:
    block = {
        key: c0[key][old_n:].clone()
        if isinstance(c0[key], torch.Tensor) else list(c0[key][old_n:])
        for key in Base.PAYLOAD_KEYS
    }
    return R11.append_parent(parent, block, "cpu")


def raw_support(parent: dict, delta: dict, base: dict, q: torch.Tensor) -> dict:
    return M.lifted_support(parent, delta, base, q)


def compact_from_p(old_x: torch.Tensor, old_text: torch.Tensor,
                   delta_x: torch.Tensor, delta_text: torch.Tensor,
                   support: dict, p: torch.Tensor, z: torch.Tensor,
                   old_n: int, device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mass = p.sum(0).clamp_min(1e-8)
    compact_text_new = p.T @ delta_text / mass[:, None]
    compact_x = torch.cat((old_x, z), 0)
    compact_text = torch.cat((old_text, compact_text_new), 0)
    raw_adj = J.raw_adjacency(support, device).to_dense()
    eye = torch.eye(old_n, device=device)
    upper = torch.cat((eye, torch.zeros((old_n, p.shape[1]), device=device)), 1)
    lower = torch.cat((torch.zeros((p.shape[0], old_n), device=device), p), 1)
    full_p = torch.cat((upper, lower), 0)
    coarse = full_p.T @ raw_adj @ full_p
    compact_adj = dense_normalized_from_dense(coarse)
    return compact_x, compact_text, compact_adj


def dense_normalized_from_dense(adj: torch.Tensor) -> torch.Tensor:
    value = adj.clamp_min(0.0)
    value = value.clone()
    indices = torch.arange(len(value), device=value.device)
    value[indices, indices] += 1.0
    degree = value.sum(1).clamp_min(1e-8)
    scale = degree.rsqrt()
    return scale[:, None] * value * scale[None, :]


def optimize_p_x(parent: dict, c0: dict, delta: dict, q: torch.Tensor,
                 labels: torch.Tensor, data: dict, old_n: int,
                 device: str, outer_steps: int = 8,
                 inner_steps: int = 8,
                 freeze_topology: bool = False) -> tuple[dict, dict, dict]:
    started = time.perf_counter()
    base = material_base(parent, c0, old_n)
    support = raw_support(parent, delta, base, q)
    m, k = len(delta["x"]), len(base["x"]) - old_n
    row_classes = B.class_rows_from_c0(q, labels)
    class_tensor = torch.tensor(row_classes, dtype=torch.long)
    allowed = labels.cpu()[:, None] == class_tensor[None, :]
    uniform = allowed.float() / allowed.sum(1, keepdim=True).clamp_min(1.0)
    p_initial = R11.initial_assignment(q, labels, J.Config(), device).to(device)
    p_initial = p_initial.clamp_min(1e-6)
    logits = torch.nn.Parameter(p_initial.log())
    z_initial = c0["x"][old_n:].float().to(device)
    z = torch.nn.Parameter(z_initial.clone())
    raw_x = delta["x"].float().to(device)
    raw_text = delta["text_emb"].float().to(device)
    raw_adj = dense_normalized(delta["edge_index"], len(raw_x), device,
                               delta.get("edge_weight"))
    old_x = parent["x"].float().to(device)
    old_text = parent["text_emb"].float().to(device)
    old_adj = dense_normalized(parent["edge_index"], old_n, device,
                               parent.get("edge_weight"))
    frozen_adj = None
    if freeze_topology:
        # The topology constraint must hold during the differentiable Student
        # rollout, not only when the final memory is serialized.
        frozen_adj = dense_normalized(
            c0["edge_index"], len(c0["x"]), device, c0.get("edge_weight")
        )
    train_plans = batches(m, 2609261)
    probe_plans = batches(m, 2609262)
    old_plan = torch.randperm(old_n, generator=torch.Generator(device="cpu").manual_seed(2609263))
    old_plan = old_plan[:min(16, old_n)]
    opt = torch.optim.Adam([logits, z], lr=0.01)

    def current(frozen: bool = False):
        p = p_from_logits(logits, allowed.to(device), uniform.to(device), 0.02)
        compact_x, compact_text, compact_adj = compact_from_p(
            old_x, old_text, raw_x, raw_text, support, p, z, old_n, device)
        if frozen:
            compact_adj = frozen_adj
        return p, compact_x, compact_text, compact_adj

    baseline_old = []
    with torch.no_grad():
        p0 = p_initial.detach()
        x0, t0, a0 = compact_from_p(old_x, old_text, raw_x, raw_text,
                                    support, p0, z_initial, old_n, device)
        if freeze_topology:
            a0 = frozen_adj
    for seed in OUTER_SEEDS:
        params = SR.unroll(x0.detach(), a0.detach(), t0.detach(), seed,
                           inner_steps, differentiable=False)
        baseline_old.append(float(mean_synthetic(
            params, x0.detach(), a0.detach(), t0.detach(), old_plan.to(device)).detach()))
    baseline_old_value = sum(baseline_old) / len(baseline_old)
    best_score = math.inf
    best_p, best_z, best_audit = None, None, None
    curve = []
    for step in range(1, outer_steps + 1):
        p, compact_x, compact_text, compact_adj = current(freeze_topology)
        train_values, old_values = [], []
        for seed, old_base in zip(OUTER_SEEDS, baseline_old):
            params = SR.unroll(compact_x, compact_adj, compact_text, seed,
                               inner_steps, differentiable=True)
            train_values.append(mean_transfer(params, raw_x, raw_adj, raw_text,
                                              train_plans))
            old_values.append(mean_synthetic(params, compact_x, compact_adj,
                                             compact_text, old_plan.to(device)))
        transfer = torch.stack(train_values).mean()
        old_loss = torch.stack(old_values).mean()
        preserve = F.relu(old_loss - baseline_old_value - 0.01)
        recon = (p @ z - raw_x).square().mean()
        p_anchor = (p * (p.clamp_min(1e-8).log()
                         - p_initial.log())).sum(1).mean()
        x_anchor = (z - z_initial).square().mean()
        objective = (transfer + 0.50 * preserve + 0.10 * recon
                     + 0.05 * p_anchor + 0.01 * x_anchor)
        opt.zero_grad(set_to_none=True)
        objective.backward()
        if logits.grad is None or z.grad is None:
            raise FloatingPointError("joint P+X gradient missing")
        if not torch.isfinite(logits.grad).all() or not torch.isfinite(z.grad).all():
            raise FloatingPointError("joint P+X gradient nonfinite")
        grad_norm = float(torch.cat((logits.grad.flatten(), z.grad.flatten())).norm())
        opt.step()
        with torch.no_grad():
            # Keep P and X close enough to the legal C0-centered proposal.
            z_delta = z - z_initial
            radius = z_initial.norm(dim=1, keepdim=True).clamp_min(1e-6) * 0.05
            z.copy_(z_initial + z_delta *
                    (radius / z_delta.norm(dim=1, keepdim=True).clamp_min(1e-12)).clamp(max=1.0))
        with torch.no_grad():
            p_eval, x_eval, t_eval, a_eval = current(freeze_topology)
        probe_values, probe_old = [], []
        for seed in OUTER_SEEDS:
            # Even a non-differentiable inner rollout needs autograd to compute
            # parameter gradients; only its connection to P/X is detached.
            with torch.enable_grad():
                params = SR.unroll(x_eval.detach(), a_eval.detach(), t_eval.detach(),
                                   seed, inner_steps, differentiable=False)
                probe_values.append(mean_transfer(params, raw_x, raw_adj, raw_text,
                                                  probe_plans).detach())
                probe_old.append(mean_synthetic(params, x_eval.detach(), a_eval.detach(),
                                                t_eval.detach(), old_plan.to(device)).detach())
        with torch.no_grad():
            probe_transfer = float(torch.stack(probe_values).mean())
            probe_old_loss = float(torch.stack(probe_old).mean())
            score = probe_transfer + 0.50 * max(0.0, probe_old_loss - baseline_old_value - 0.01)
            record = {
                "step": step, "train_objective": float(objective.detach()),
                "train_transfer": float(transfer.detach()),
                "train_old_loss": float(old_loss.detach()),
                "probe_transfer": probe_transfer, "probe_old_loss": probe_old_loss,
                "score": score, "p_kl": float(p_anchor.detach()),
                "x_anchor": float(x_anchor.detach()),
                "x_displacement": float((z - z_initial).norm(dim=1).mean()),
                "gradient_norm": grad_norm,
            }
            curve.append(record)
            if score < best_score:
                best_score = score
                best_p = p_eval.detach().clone()
                best_z = z.detach().clone()
                best_audit = record.copy()
    if best_p is None or best_z is None:
        raise AssertionError("joint P+X did not produce a checkpoint")
    coarse = R11.coarse_from_assignment(support, best_p.detach().cpu(), old_n, device)
    candidates, material_audit = J.materialize(
        base, delta, best_p.detach().cpu(), coarse, data["encoder"], J.Config())
    shared_p = candidates["joint"]
    shared_px = J.T.clone_memory(shared_p)
    shared_px["x"][old_n:] = best_z.detach().cpu()
    if freeze_topology:
        shared_p["edge_index"] = c0["edge_index"].clone()
        shared_px["edge_index"] = c0["edge_index"].clone()
    for key in ("edge_index", "texts", "text_emb", "mass", "depth", "stage"):
        if key == "texts":
            assert shared_px[key] == shared_p[key]
        else:
            assert torch.equal(shared_px[key], shared_p[key]), key
    assert torch.equal(shared_px["edge_index"][:,
                         (shared_px["edge_index"] < old_n).all(0)],
                       parent["edge_index"])
    audit = {
        "method": "tgc-shared-p-bounded-x-v1",
        "old_rows": old_n, "delta_rows": m, "new_rows": k,
        "outer_seeds": list(OUTER_SEEDS), "outer_steps": outer_steps,
        "inner_steps": inner_steps, "best_score": best_score,
        "baseline_old_loss": baseline_old_value,
        "best_checkpoint": best_audit, "curve": curve,
        "p_initial_sha256": V1.tensor_hash(p_initial.detach().cpu()),
        "p_final_sha256": V1.tensor_hash(best_p.detach().cpu()),
        "p_mean_kl": float((best_p * (best_p.clamp_min(1e-8).log()
                                      - p_initial.log())).sum(1).mean()),
        "x_displacement_mean": float((best_z - z_initial).norm(dim=1).mean()),
        "materialization": material_audit,
        "topology_mode": "frozen_c0" if freeze_topology else "projected_from_p",
        "optimization_topology_mode": (
            "frozen_c0" if freeze_topology else "projected_from_p"
        ),
        "historical_raw_reads": 0,
        "report_labels_used": False,
        "seconds": time.perf_counter() - started,
    }
    return shared_p, shared_px, audit


def run(args: argparse.Namespace) -> None:
    source = Path(args.source_root)
    cell = source / args.dataset / args.order
    parent = load(cell / f"c0_s{args.stage - 1}_memory.pt")
    c0 = load(cell / f"c0_s{args.stage}_memory.pt")
    data = V1.load_data(args.dataset, args.minilm, args.device,
                        Path(args.cache_root))
    ids, budgets = R11.D.set_stream(data, args.order)
    old_n = len(parent["x"])
    q = R11.D.current_assignment(data, ids[args.stage], c0, old_n,
                                 args.stage, args.device)
    labels = data["y"][ids[args.stage]].cpu()
    delta = J.T.raw_delta_block(data, ids[args.stage], args.stage)
    shared_p, shared_px, audit = optimize_p_x(
        parent, c0, delta, q, labels, data, old_n, args.device,
        args.outer_steps, args.inner_steps, args.freeze_topology)
    output = Path(args.output) / f"{args.dataset}_{args.order}_s{args.stage}"
    if (output / "results.json").exists():
        raise FileExistsError(f"immutable output exists: {output}")
    output.mkdir(parents=True, exist_ok=True)
    torch.save(c0, output / "c0_memory.pt")
    torch.save(shared_p, output / "shared_p_memory.pt")
    torch.save(shared_px, output / "shared_px_memory.pt")
    protocol = {
        "name": "tgc-shared-p-bounded-x-v1",
        "dataset": args.dataset, "order": args.order, "stage": int(args.stage),
        "arms": ["c0", "shared_p", "shared_px"],
        "assignment": (
            "one legal class-masked P shared by X initialization and text; "
            + ("final topology is frozen C0" if args.freeze_topology
               else "P also materializes projected topology")
        ),
        "x": "bounded differentiable Student-facing update from the same P-derived compact graph",
        "student_inner_seeds": list(OUTER_SEEDS), "report_f1_selection": False,
        "historical_raw_reads": 0, "source_root": str(source),
        "topology_mode": "frozen_c0" if args.freeze_topology else "projected_from_p",
        "optimization_topology_mode": (
            "frozen_c0" if args.freeze_topology else "projected_from_p"
        ),
    }
    save_json(output / "protocol.json", protocol)
    save_json(output / "build_audit.json", {"protocol": protocol,
                                              "budgets": budgets,
                                              "audit": audit,
                                              "same_p_text_assignment": True,
                                              "same_p_text_topology": not args.freeze_topology,
                                              "topology_matches_c0": args.freeze_topology,
                                              "same_old_prefix": True})
    save_json(output / "results.json", {"protocol": protocol, "audit": audit})
    print(json.dumps({"event": "complete", "dataset": args.dataset,
                      "order": args.order, "stage": int(args.stage),
                      "output": str(output)}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("cora", "citeseer"), required=True)
    parser.add_argument("--order", required=True)
    parser.add_argument("--stage", type=int, choices=(1, 2), required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--minilm", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--outer-steps", type=int, default=8)
    parser.add_argument("--inner-steps", type=int, default=8)
    parser.add_argument("--freeze-topology", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    run(args)


if __name__ == "__main__":
    main()
