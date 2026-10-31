"""Bounded old-X optimization through scratch-student Adam updates."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time

import torch
import torch.nn.functional as F

import run_tgc_revision_gate as R
import tgc_revision_gate as T


def graph(params, x, adjacency):
    h = F.relu(adjacency @ F.linear(x, params["g1.lin.weight"],
                                    params["g1.lin.bias"]))
    return F.normalize(adjacency @ F.linear(h, params["g2.lin.weight"],
                                             params["g2.lin.bias"]), dim=1)


def contrastive(params, x, adjacency, text, rows):
    g = graph(params, x, adjacency)[rows]
    t = F.normalize(F.linear(text[rows], params["text_proj.weight"],
                             params["text_proj.bias"]), dim=1)
    logits = params["logit_scale"].exp().clamp(max=100) * g @ t.T
    labels = torch.arange(len(rows), device=x.device)
    return (F.cross_entropy(logits, labels)
            + F.cross_entropy(logits.T, labels)) / 2


def sqrt_zero_safe(value):
    return torch.where(value > 0, value.clamp_min(1e-30).sqrt(),
                       torch.zeros_like(value))


def adam_update(params, gradients, first, second, step, lr=.001):
    new, m, v = {}, {}, {}
    for (name, param), grad in zip(params.items(), gradients):
        m[name] = torch.lerp(first[name], grad, .1)
        v[name] = second[name] * .999 + grad.square() * .001
        denom = sqrt_zero_safe(v[name]) / math.sqrt(1 - .999 ** step) + 1e-8
        new[name] = torch.addcdiv(param, m[name], denom,
                                  value=-lr / (1 - .9 ** step))
    return new, m, v


@dataclass(frozen=True)
class Config:
    radius: float = .1
    outer_steps: int = 16
    outer_lr: float = .01
    old_tolerance: float = .01
    move_weight: float = .01
    gain_min: float = 1e-4
    train_seeds: tuple = (7500, 7501)
    validation_seeds: tuple = (7510, 7511)
    batch: int = 16
    probe_batches: int = 4

    def __post_init__(self):
        if (self.radius < 0 or not math.isfinite(self.radius)
                or self.outer_steps < 4 or self.outer_steps % 4
                or self.outer_lr <= 0 or self.batch < 2 or self.probe_batches < 1):
            raise ValueError("invalid meta-revision configuration")
        if set(self.train_seeds) & set(self.validation_seeds):
            raise ValueError("optimization/validation seeds overlap")


def unroll(x, adjacency, text, seed, steps, differentiable=True):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(int(seed))
        model = R.G.CLIPStudent(x.shape[1], text_dim=text.shape[1]).to(
            device=x.device, dtype=x.dtype)
    params = dict(model.named_parameters())
    first = {k: torch.zeros_like(v) for k, v in params.items()}
    second = {k: torch.zeros_like(v) for k, v in params.items()}
    gen = torch.Generator().manual_seed(seed * 7 + 1)
    for step in range(1, steps + 1):
        rows = torch.randperm(len(x), generator=gen)[:min(16, len(x))].to(x.device)
        loss = contrastive(params, x, adjacency, text, rows)
        grads = torch.autograd.grad(loss, tuple(params.values()),
                                    create_graph=differentiable)
        params, first, second = adam_update(params, grads, first, second, step)
        if not differentiable:
            params = {k: v.detach().requires_grad_() for k, v in params.items()}
            first = {k: v.detach() for k, v in first.items()}
            second = {k: v.detach() for k, v in second.items()}
    return params


def probe_plan(old_n, total_n, stage, cfg):
    if min(old_n, total_n - old_n) < 4:
        raise ValueError("need at least four old and four new probe rows")
    gen = torch.Generator().manual_seed(7520 + stage)
    splits, batches = {}, {}
    for name, start, count in (("old", 0, old_n), ("new", old_n, total_n - old_n)):
        order = torch.randperm(count, generator=gen) + start
        size = max(2, count // 5)
        splits[name] = {"train": order[size:], "validation": order[:size]}
    for split in ("train", "validation"):
        batches[split] = {}
        for name in ("old", "new"):
            ids = splits[name][split]
            batches[split][name] = [ids[torch.randperm(len(ids), generator=gen)
                [:min(cfg.batch, len(ids))]] for _ in range(cfg.probe_batches)]
    return splits, batches


def probe_losses(params, x, adjacency, text, batches):
    graph_emb = graph(params, x, adjacency)
    encoded = F.normalize(F.linear(text, params["text_proj.weight"],
                                    params["text_proj.bias"]), dim=1)
    losses = {}
    for name, groups in batches.items():
        values = []
        for ids in groups:
            ids = ids.to(x.device)
            logits = params["logit_scale"].exp().clamp(max=100) * graph_emb[ids] @ encoded[ids].T
            labels = torch.arange(len(ids), device=x.device)
            values.append((F.cross_entropy(logits, labels)
                           + F.cross_entropy(logits.T, labels)) / 2)
        losses[name] = torch.stack(values).mean()
    return losses


def changed_x(base, rows, relative):
    scale = base[rows].norm(dim=1, keepdim=True).clamp_min(1e-6)
    return base.index_add(0, rows, scale * relative)


def project(relative, radius):
    with torch.no_grad():
        relative.mul_((radius / relative.norm(dim=1, keepdim=True).clamp_min(1e-12)).clamp(max=1))


def acceptance(metrics, baseline, cfg):
    return all(m["new"] <= b["new"] - cfg.gain_min
               and m["old"] <= b["old"] + cfg.old_tolerance
               for m, b in zip(metrics, baseline))


def optimize(memory, support, rows, stage, horizon, device, cfg=Config(), stop=False):
    if horizon not in (1, 4, 8):
        raise ValueError("unregistered inner horizon")
    old_n = int(support["old_rows"])
    if not rows or len(set(rows)) != len(rows) or min(rows) < 0 or max(rows) >= old_n:
        raise ValueError("invalid editable historical rows")
    memory_hash, support_hash = T.memory_hash(memory), T.memory_hash(support)
    R.sync(device)
    started = time.perf_counter()
    base, text = memory["x"].to(device), memory["text_emb"].to(device)
    adjacency = R.C.training_adjacency(memory, device)
    px, pt = support["x"].to(device).detach(), support["text_emb"].to(device).detach()
    pa = R.C.training_adjacency(support, device)
    splits, batches = probe_plan(old_n, len(px), stage, cfg)
    selected = torch.tensor(rows, device=device)
    relative = torch.nn.Parameter(torch.zeros_like(base[selected]))
    opt = torch.optim.Adam([relative], lr=cfg.outer_lr)
    baseline = {}
    for split, seeds in (("train", cfg.train_seeds), ("validation", cfg.validation_seeds)):
        baseline[split] = []
        for seed in seeds:
            params = unroll(base, adjacency, text, seed, horizon, False)
            with torch.no_grad():
                losses = probe_losses(params, px, pa, pt, batches[split])
            baseline[split].append({k: float(v) for k, v in losses.items()})

    def objective(split, differentiable):
        seeds = cfg.train_seeds if split == "train" else cfg.validation_seeds
        objectives, observations = [], []
        for index, seed in enumerate(seeds):
            x = changed_x(base, selected, relative)
            params = unroll(x, adjacency, text, seed, horizon, differentiable)
            losses = probe_losses(params, px, pa, pt, batches[split])
            preserve = F.relu(losses["old"] - baseline[split][index]["old"] - cfg.old_tolerance)
            value = losses["new"] + preserve
            observations.append({k: float(v.detach()) for k, v in losses.items()})
            objectives.append(value)
        regularizer = cfg.move_weight * relative.square().sum(1).mean()
        return torch.stack(objectives).mean() + regularizer, observations

    trajectory, best_value, best_relative, best_step = [], math.inf, None, None
    best_observations = None
    for step in range(1, cfg.outer_steps + 1):
        opt.zero_grad(set_to_none=True)
        loss, observations = objective("train", not stop)
        # Keep the negative-control objective connected to the editable tensor
        # while blocking the student-to-X path; this should yield zero X update.
        if stop:
            loss = loss + relative.sum() * 0.0
        loss.backward()
        grad = relative.grad
        if grad is None or not torch.isfinite(grad).all() or not torch.isfinite(loss):
            raise FloatingPointError("missing/nonfinite historical-X hypergradient")
        grad_norm = float(grad.norm())
        opt.step()
        project(relative, cfg.radius)
        record = {"step": step, "train_objective": float(loss.detach()),
                  "gradient_norm": grad_norm, "train_probe": observations}
        if step % 4 == 0:
            value, observations = objective("validation", False)
            score = float(value.detach())
            if not math.isfinite(score):
                raise FloatingPointError("nonfinite validation objective")
            record.update(validation_objective=score, validation_probe=observations)
            if score < best_value:
                best_value, best_step = score, step
                best_relative = relative.detach().clone()
                best_observations = observations
        trajectory.append(record)
    if best_relative is None:
        raise AssertionError("no validation checkpoint")
    result = T.clone_memory(memory)
    result["x"] = changed_x(base, selected, best_relative).detach().cpu()
    accepted = acceptance(best_observations, baseline["validation"], cfg)
    if stop and not torch.equal(result["x"], memory["x"]):
        raise AssertionError("stop-gradient must leave X unchanged")
    if T.memory_hash(memory) != memory_hash or T.memory_hash(support) != support_hash:
        raise AssertionError("optimizer mutated its frozen inputs")
    R.sync(device)
    audit = {"config": asdict(cfg), "horizon": horizon, "stop_gradient": stop,
             "rows": rows, "stage": stage, "selected_outer_step": best_step,
             "baseline_probe": baseline, "selected_validation": best_observations,
             "selected_objective": best_value, "accepted": accepted,
             "curve": trajectory, "nonzero_gradient_steps": sum(r["gradient_norm"] > 0 for r in trajectory),
             "seconds": time.perf_counter() - started,
             "input_memory_sha256": memory_hash, "probe_sha256": support_hash,
             "splits": {k: {s: v.tolist() for s, v in part.items()} for k, part in splits.items()},
             "probe_batches": {s: {k: [v.tolist() for v in groups] for k, groups in part.items()}
                               for s, part in batches.items()},
             "report_labels_used": False, "historical_raw_reads": 0,
             "outer_train_student_steps": cfg.outer_steps * len(cfg.train_seeds) * horizon,
             "validation_student_steps": cfg.outer_steps // 4 * len(cfg.validation_seeds) * horizon,
             "baseline_student_steps": (len(cfg.train_seeds) + len(cfg.validation_seeds)) * horizon}
    return result, audit


def matched_random(memory, candidate, rows, stage):
    result = T.clone_memory(memory)
    gen = torch.Generator().manual_seed(7600 + stage)
    ids = torch.tensor(rows)
    direction = torch.randn(memory["x"][ids].shape, generator=gen)
    lengths = (candidate["x"][ids] - memory["x"][ids]).norm(dim=1, keepdim=True)
    result["x"][ids] += F.normalize(direction, dim=1) * lengths
    return result


def invariants(base, candidate, rows, old_n, protected, radius):
    if base.keys() != candidate.keys():
        raise AssertionError("memory schema changed")
    for key, value in base.items():
        if key == "x":
            continue
        equal = torch.equal(value, candidate[key]) if isinstance(value, torch.Tensor) else value == candidate[key]
        if not equal:
            raise AssertionError(f"immutable field changed: {key}")
    if candidate["x"].shape != base["x"].shape or not torch.isfinite(candidate["x"]).all():
        raise AssertionError("invalid X")
    changed = torch.where((base["x"] != candidate["x"]).any(1))[0].tolist()
    if not set(changed) <= set(rows) or set(changed) & set(protected) or any(i >= old_n for i in changed):
        raise AssertionError("changed forbidden row")
    relative = (candidate["x"] - base["x"]).norm(dim=1) / base["x"].norm(dim=1).clamp_min(1e-6)
    if float(relative.max()) > radius + 1e-5:
        raise AssertionError("radius exceeded")
    return {"changed_rows": changed, "max_relative_displacement": float(relative.max()),
            "relative_displacement": relative.tolist()}


def numerical_check(memory, seed, device, horizons=(1, 4, 8)):
    x, t = memory["x"].to(device), memory["text_emb"].to(device)
    adj = R.C.training_adjacency(memory, device)
    records = []
    for steps in horizons:
        params = unroll(x, adj, t, seed, steps, False)
        torch.manual_seed(seed)
        model = R.G.CLIPStudent(x.shape[1], text_dim=t.shape[1]).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=.001)
        gen = torch.Generator().manual_seed(seed * 7 + 1)
        for _ in range(steps):
            rows = torch.randperm(len(x), generator=gen)[:min(16, len(x))].to(device)
            loss = T._info_nce(model, x, t, adj, rows)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        worst = 0.
        for name, value in model.named_parameters():
            torch.testing.assert_close(params[name], value, atol=2e-5, rtol=2e-4)
            worst = max(worst, float((params[name] - value).detach().abs().max()))
        records.append({"steps": steps, "max_parameter_abs_difference": worst})
    return records
