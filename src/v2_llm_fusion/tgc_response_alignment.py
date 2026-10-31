"""Historical-X meta updates at a fixed canonical Adam trajectory prefix."""
from __future__ import annotations

from dataclasses import asdict
import math
import time

import torch
import torch.nn.functional as F

import tgc_student_revision as S

R, T = S.R, S.T


def prefix_hash(prefix):
    tensors = {f"{group}/{name}": value for group in ("params", "first", "second")
               for name, value in prefix[group].items()}
    tensors["rng"] = prefix["rng"]
    tensors["steps_seed"] = torch.tensor([prefix["steps"], prefix["seed"]])
    return R.C.tensor_hash(tensors)


def make_prefix(x, adjacency, text, seed, steps):
    if steps < 0:
        raise ValueError("negative prefix length")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(int(seed))
        model = R.G.CLIPStudent(x.shape[1], text_dim=text.shape[1]).to(x.device, dtype=x.dtype)
    optimizer = torch.optim.Adam(model.parameters(), lr=R.P.STUDENT.lr)
    gen = torch.Generator().manual_seed(int(seed) * 7 + 1)
    for _ in range(steps):
        ids = torch.randperm(len(x), generator=gen)[:min(R.P.STUDENT.batch, len(x))].to(x.device)
        loss = T._info_nce(model, x.detach(), text.detach(), adjacency, ids)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    params, first, second = {}, {}, {}
    for name, value in model.named_parameters():
        state = optimizer.state[value]
        params[name] = value.detach().clone()
        first[name] = state.get("exp_avg", torch.zeros_like(value)).detach().clone()
        second[name] = state.get("exp_avg_sq", torch.zeros_like(value)).detach().clone()
    return {"params": params, "first": first, "second": second,
            "rng": gen.get_state(), "steps": steps, "seed": int(seed)}


def prepare_prefixes(memory, warmup, device, cfg=S.Config()):
    x, text = memory["x"].to(device), memory["text_emb"].to(device)
    adjacency = R.C.training_adjacency(memory, device)
    R.sync(device)
    started = time.perf_counter()
    prefixes = {seed: make_prefix(x, adjacency, text, seed, warmup)
                for seed in (*cfg.train_seeds, *cfg.validation_seeds)}
    R.sync(device)
    audit = {"warmup_steps": warmup, "seconds": time.perf_counter() - started,
             "student_steps": warmup * len(prefixes),
             "prefix_hashes": {str(s): prefix_hash(p) for s, p in prefixes.items()},
             "input_memory_sha256": T.memory_hash(memory),
             "labels_used": False, "historical_raw_reads": 0}
    return prefixes, audit


def suffix(prefix, x, adjacency, text, steps, differentiable=True):
    params = {k: v.to(x.device).detach().clone().requires_grad_() for k, v in prefix["params"].items()}
    first = {k: v.to(x.device).detach().clone() for k, v in prefix["first"].items()}
    second = {k: v.to(x.device).detach().clone() for k, v in prefix["second"].items()}
    gen = torch.Generator().set_state(prefix["rng"].cpu())
    for offset in range(1, steps + 1):
        rows = torch.randperm(len(x), generator=gen)[:min(R.P.STUDENT.batch, len(x))].to(x.device)
        loss = S.contrastive(params, x, adjacency, text, rows)
        gradients = torch.autograd.grad(loss, tuple(params.values()), create_graph=differentiable)
        params, first, second = S.adam_update(params, gradients, first, second,
                                              prefix["steps"] + offset, R.P.STUDENT.lr)
        if not differentiable:
            params = {k: v.detach().requires_grad_() for k, v in params.items()}
            first = {k: v.detach() for k, v in first.items()}
            second = {k: v.detach() for k, v in second.items()}
    return params


def optimize(memory, support, rows, stage, horizon, prefixes, device,
             cfg=S.Config(), stop=False):
    if horizon not in (1, 4) or not rows or len(set(rows)) != len(rows):
        raise ValueError("invalid horizon or editable rows")
    old_n = int(support["old_rows"])
    if min(rows) < 0 or max(rows) >= old_n:
        raise ValueError("nonhistorical editable row")
    seeds = (*cfg.train_seeds, *cfg.validation_seeds)
    if set(prefixes) != set(seeds) or len({p["steps"] for p in prefixes.values()}) != 1:
        raise ValueError("invalid shared prefix bundle")
    before = {str(s): prefix_hash(prefixes[s]) for s in seeds}
    memory_hash, support_hash = T.memory_hash(memory), T.memory_hash(support)
    base, text = memory["x"].to(device), memory["text_emb"].to(device)
    adjacency = R.C.training_adjacency(memory, device)
    px, pt = support["x"].to(device).detach(), support["text_emb"].to(device).detach()
    pa = R.C.training_adjacency(support, device)
    splits, batches = S.probe_plan(old_n, len(px), stage, cfg)
    selected = torch.tensor(rows, device=device)
    relative = torch.nn.Parameter(torch.zeros_like(base[selected]))
    optimizer = torch.optim.Adam([relative], lr=cfg.outer_lr)
    R.sync(device)
    started = time.perf_counter()
    baseline = {}
    for split, split_seeds in (("train", cfg.train_seeds), ("validation", cfg.validation_seeds)):
        baseline[split] = []
        for seed in split_seeds:
            params = suffix(prefixes[seed], base, adjacency, text, horizon, False)
            with torch.no_grad():
                losses = S.probe_losses(params, px, pa, pt, batches[split])
            baseline[split].append({k: float(v) for k, v in losses.items()})

    def objective(split, differentiable):
        split_seeds = cfg.train_seeds if split == "train" else cfg.validation_seeds
        values, observations = [], []
        for i, seed in enumerate(split_seeds):
            x = S.changed_x(base, selected, relative)
            params = suffix(prefixes[seed], x, adjacency, text, horizon, differentiable)
            losses = S.probe_losses(params, px, pa, pt, batches[split])
            preserve = F.relu(losses["old"] - baseline[split][i]["old"] - cfg.old_tolerance)
            values.append(losses["new"] + preserve)
            observations.append({k: float(v.detach()) for k, v in losses.items()})
        regularizer = cfg.move_weight * relative.square().sum(1).mean()
        return torch.stack(values).mean() + regularizer, observations

    curve, best_score, best_relative, best_step, best_probe = [], math.inf, None, None, None
    for step in range(1, cfg.outer_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        loss, observations = objective("train", not stop)
        if stop:
            loss = loss + relative.sum() * 0
        loss.backward()
        grad = relative.grad
        if grad is None or not torch.isfinite(grad).all() or not torch.isfinite(loss):
            raise FloatingPointError("invalid suffix hypergradient")
        norm = float(grad.norm())
        optimizer.step()
        S.project(relative, cfg.radius)
        record = {"step": step, "train_objective": float(loss.detach()),
                  "gradient_norm": norm, "train_probe": observations}
        if step % 4 == 0:
            val, observations = objective("validation", False)
            score = float(val.detach())
            if not math.isfinite(score):
                raise FloatingPointError("invalid validation score")
            record.update(validation_objective=score, validation_probe=observations)
            if score < best_score:
                best_score, best_step, best_probe = score, step, observations
                best_relative = relative.detach().clone()
        curve.append(record)
    if best_relative is None:
        raise AssertionError("missing validation checkpoint")
    result = T.clone_memory(memory)
    result["x"] = S.changed_x(base, selected, best_relative).detach().cpu()
    if stop and not torch.equal(result["x"], memory["x"]):
        raise AssertionError("stop must be an exact no-op")
    if T.memory_hash(memory) != memory_hash or T.memory_hash(support) != support_hash:
        raise AssertionError("input memory/probe mutated")
    if before != {str(s): prefix_hash(prefixes[s]) for s in seeds}:
        raise AssertionError("shared prefix mutated")
    R.sync(device)
    audit = {"config": asdict(cfg), "horizon": horizon,
             "warmup_steps": next(iter(prefixes.values()))["steps"],
             "stop_gradient": stop, "rows": rows, "stage": stage,
             "prefix_hashes": before, "selected_outer_step": best_step,
             "baseline_probe": baseline, "selected_validation": best_probe,
             "selected_objective": best_score,
             "accepted": S.acceptance(best_probe, baseline["validation"], cfg),
             "curve": curve, "nonzero_gradient_steps": sum(r["gradient_norm"] > 0 for r in curve),
             "seconds": time.perf_counter() - started,
             "input_memory_sha256": memory_hash, "probe_sha256": support_hash,
             "splits": {k: {s: v.tolist() for s, v in part.items()} for k, part in splits.items()},
             "probe_batches": {s: {k: [v.tolist() for v in groups] for k, groups in part.items()}
                               for s, part in batches.items()},
             "report_labels_used": False, "historical_raw_reads": 0,
             "suffix_student_steps": (cfg.outer_steps * len(cfg.train_seeds)
                 + cfg.outer_steps // 4 * len(cfg.validation_seeds) + len(seeds)) * horizon,
             "surrogate": "fixed C0 prefix, no gradient through prefix; not full-trajectory differentiation"}
    return result, audit


def numerical_check(memory, seed, warmup, horizon, device):
    x, text = memory["x"].to(device), memory["text_emb"].to(device)
    adjacency = R.C.training_adjacency(memory, device)
    prefix = make_prefix(x, adjacency, text, seed, warmup)
    result = suffix(prefix, x, adjacency, text, horizon, False)
    reference = make_prefix(x, adjacency, text, seed, warmup + horizon)
    worst = 0.
    for name, value in result.items():
        target = reference["params"][name]
        torch.testing.assert_close(value, target, atol=2e-5, rtol=2e-4)
        worst = max(worst, float((value.detach() - target).abs().max()))
    return {"warmup": warmup, "horizon": horizon, "max_parameter_difference": worst}
