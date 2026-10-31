"""R4 legal-reference relational condensation; no report data enters this module."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time

import torch
import torch.nn.functional as F

import tgc_student_revision as S
import tgc_revision_gate as T
from run_multiscale_asm import _asm_loss

R = S.R
SUPPORTS = ("legacy_uniform", "legacy_compact", "lift_uniform", "lift_compact")


def adam_update(params, grads, first, second, step):
    """Out-of-place CUDA foreach Adam, with a finite derivative at zero RMS."""
    names = tuple(params)
    # PyTorch 2.4 foreach backward cannot index a zero-dimensional logit_scale.
    # Flattening preserves elementwise arithmetic and gives its gradient a axis.
    flat = lambda values: [v.reshape(-1) for v in values]
    gradients = flat(grads)
    m = torch._foreach_lerp(flat(first.values()), gradients, 1 - .9)
    v = torch._foreach_addcmul(torch._foreach_mul(flat(second.values()), .999),
                              gradients, gradients, 1 - .999)
    denom = torch._foreach_div([S.sqrt_zero_safe(value) for value in v],
                               [math.sqrt(1 - .999 ** step)] * len(names))
    denom = torch._foreach_add(denom, 1e-8)
    updated = torch._foreach_addcdiv(flat(params.values()), m, denom,
                                    [-.001 / (1 - .9 ** step)] * len(names))
    return tuple({name: value.reshape(params[name].shape) for name, value in zip(names, values)}
                 for values in (updated, m, v))


def unroll(x, adj, text, seed, steps, differentiable=True):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model = R.G.CLIPStudent(x.shape[1], text_dim=text.shape[1]).to(device=x.device, dtype=x.dtype)
    params = dict(model.named_parameters())
    first, second = ({k: torch.zeros_like(v) for k, v in params.items()} for _ in range(2))
    generator = torch.Generator().manual_seed(seed * 7 + 1)
    for step in range(1, steps + 1):
        rows = torch.randperm(len(x), generator=generator)[:min(16, len(x))].to(x.device)
        loss = S.contrastive(params, x, adj, text, rows)
        grads = torch.autograd.grad(loss, tuple(params.values()), create_graph=differentiable)
        update = adam_update if x.device.type == "cuda" else S.adam_update
        params, first, second = update(params, grads, first, second, step)
        if not differentiable:
            params = {k: v.detach().requires_grad_() for k, v in params.items()}
            first, second = ({k: v.detach() for k, v in state.items()} for state in (first, second))
    return params


def numerical_check(memory, seed, device, horizons=(1, 4, 32)):
    x, text, adj = memory["x"].to(device), memory["text_emb"].to(device), R.C.training_adjacency(memory, device)
    result = []
    for steps in horizons:
        actual = unroll(x, adj, text, seed, steps, False)
        torch.manual_seed(seed)
        model = R.G.CLIPStudent(x.shape[1], text_dim=text.shape[1]).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=.001)
        generator = torch.Generator().manual_seed(seed * 7 + 1)
        for _ in range(steps):
            rows = torch.randperm(len(x), generator=generator)[:min(16, len(x))].to(device)
            loss = T._info_nce(model, x, text, adj, rows)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        maximum = 0.
        for name, value in model.named_parameters():
            torch.testing.assert_close(actual[name], value, atol=2e-5, rtol=2e-4)
            maximum = max(maximum, float((actual[name] - value).detach().abs().max()))
        result.append({"steps": steps, "max_parameter_abs_difference": maximum,
                       "atol": 2e-5, "rtol": 2e-4, "cuda_foreach_matched": True})
    return result


@dataclass(frozen=True)
class Config:
    horizon: int = 32
    outer_steps: int = 16
    outer_lr: float = .01
    radius: float = .1
    old_epsilon: float = .001
    dual_lr: float = 10.
    initial_dual: float = 1.
    temperature: float = .07
    train_seeds: tuple = (8100, 8101)
    validation_seeds: tuple = (8110, 8111)
    teacher_seeds: tuple = (8000, 8001)
    teacher_steps: int = 128
    teacher_batch: int = 64
    direct_steps: int = 128
    probe_seed: int = 8090
    new_train_queries: int = 128
    new_validation_queries: int = 32

    def __post_init__(self):
        if min(self.horizon, self.outer_steps, self.teacher_steps, self.direct_steps) <= 0:
            raise ValueError("positive training lengths required")
        if not (self.radius > 0 and self.old_epsilon >= 0 and self.temperature > 0):
            raise ValueError("invalid optimization bounds")
        groups = (set(self.train_seeds), set(self.validation_seeds), set(self.teacher_seeds))
        if any(not group for group in groups):
            raise ValueError("empty seed group")
        if any(groups[i] & groups[j] for i in range(3) for j in range(i)):
            raise ValueError("construction/validation/teacher seeds overlap")


def weighted_adjacency(edge_index, n, device, weights=None, dtype=torch.float32):
    if (edge_index.ndim != 2 or edge_index.shape[0] != 2 or n <= 0
            or (edge_index < 0).any() or (edge_index >= n).any()):
        raise ValueError("invalid reference edge indices")
    indices = torch.cat((edge_index.cpu(), torch.arange(n).repeat(2, 1)), 1).to(device)
    w = torch.ones(edge_index.shape[1], dtype=dtype) if weights is None else weights.to(dtype).cpu()
    if len(w) != edge_index.shape[1] or not torch.isfinite(w).all() or (w < 0).any():
        raise ValueError("invalid reference edge weights")
    values = torch.cat((w, torch.ones(n, dtype=dtype))).to(device)
    degree = torch.zeros(n, device=device, dtype=dtype).scatter_add_(0, indices[0], values)
    inv = degree.clamp_min(1e-9).rsqrt()
    return torch.sparse_coo_tensor(indices, inv[indices[0]] * values * inv[indices[1]],
                                  (n, n)).coalesce()


def lifted_support(old, delta, compact, assignment):
    """Spread each compact cross-edge across its current raw cluster members."""
    old_n, new_n = len(old["x"]), len(delta["x"])
    k = len(compact["x"]) - old_n
    assignment = assignment.long().cpu()
    if (assignment.ndim != 1 or new_n == 0 or k <= 0 or len(assignment) != new_n
            or assignment.min() < 0 or assignment.max() >= k):
        raise ValueError("assignment shape or range mismatch")
    sizes = torch.bincount(assignment, minlength=k)
    if (sizes == 0).any():
        raise ValueError("empty compact cluster")
    edges = compact["edge_index"].long().cpu()
    oo = edges[:, (edges[0] < old_n) & (edges[1] < old_n)]
    cross = edges[:, (edges[0] < old_n) & (edges[1] >= old_n)]
    reverse = edges[:, (edges[0] >= old_n) & (edges[1] < old_n)].flip(0)
    if sorted(cross.T.tolist()) != sorted(reverse.T.tolist()):
        raise ValueError("compact cross-edges must be symmetric")
    if sorted(oo.T.tolist()) != sorted(old["edge_index"].T.tolist()):
        raise ValueError("compact old topology is not frozen")
    parts = [oo, delta["edge_index"].long().cpu() + old_n]
    weights = [torch.ones(parts[0].shape[1]), torch.ones(parts[1].shape[1])]
    for i, j in cross.T.tolist():
        members = torch.where(assignment == j - old_n)[0] + old_n
        forward = torch.stack((torch.full_like(members, i), members))
        parts.extend((forward, forward.flip(0)))
        w = torch.full((len(members),), 1. / len(members))
        weights.extend((w, w))
    result = T._concat_support(old, delta)
    result["edge_index"] = torch.cat(parts, 1)
    result["edge_weight"] = torch.cat(weights)
    return result


def adjacency(support, device):
    return weighted_adjacency(support["edge_index"], len(support["x"]), device,
                              support.get("edge_weight"), support["x"].dtype)


def sample_rows(n, old_n, batch, generator, mode, compact_old_fraction):
    if not (0 < old_n < n and batch > 1 and 0 <= compact_old_fraction <= 1):
        raise ValueError("invalid sampler bounds")
    batch = min(batch, n)
    if mode == "uniform":
        return torch.randperm(n, generator=generator)[:batch]
    if mode != "compact":
        raise ValueError(mode)
    want_old = int(round(batch * compact_old_fraction))
    old_count = min(old_n, want_old)
    new_count = min(n - old_n, batch - old_count)
    old_count = min(old_n, batch - new_count)
    old_ids = torch.randperm(old_n, generator=generator)[:old_count]
    new_ids = torch.randperm(n - old_n, generator=generator)[:new_count] + old_n
    rows = torch.cat((old_ids, new_ids))
    return rows[torch.randperm(len(rows), generator=generator)]


def probe_plan(old_n, new_n, stage, cfg):
    if min(old_n, new_n) < 4:
        raise ValueError("insufficient legal old/new probe rows")
    gen = torch.Generator().manual_seed(cfg.probe_seed + stage)
    old = torch.randperm(old_n, generator=gen)
    new = torch.randperm(new_n, generator=gen) + old_n
    hold_old = max(2, old_n // 5)
    hold_new = min(cfg.new_validation_queries, max(2, new_n // 5))
    return {
        "train": torch.cat((old[hold_old:], new[hold_new:hold_new + cfg.new_train_queries])),
        "validation": torch.cat((old[:hold_old], new[:hold_new])),
    }


def logits(params, x, adj, text, ids, temperature):
    g = S.graph(params, x, adj)[ids]
    t = F.normalize(F.linear(text[ids], params["text_proj.weight"], params["text_proj.bias"]), dim=1)
    return g @ t.T / temperature


def probabilities(logit):
    return logit.softmax(1), logit.T.softmax(1)


def divergence(logit, targets, mask):
    if not mask.any():
        return logit.sum() * 0.
    values = [F.kl_div(z[mask].log_softmax(1), p[mask], reduction="batchmean")
              for z, p in zip((logit, logit.T), (p.to(logit) for p in targets))]
    return torch.stack(values).mean().clamp_min(0)


def hard_loss(logit, mask):
    if not mask.any():
        return logit.sum() * 0.
    targets = torch.arange(len(logit), device=logit.device)[mask]
    return (F.cross_entropy(logit[mask], targets) + F.cross_entropy(logit.T[mask], targets)) / 2


def stats(logit, old_mask):
    new_mask = ~old_mask
    diagonal = torch.arange(len(logit), device=logit.device)
    result = {}
    for name, mask in (("old", old_mask), ("new", new_mask), ("mixed", torch.ones_like(old_mask))):
        result[name + "_nce"] = float(hard_loss(logit, mask))
        result[name + "_retrieval"] = float(((logit.argmax(1)[mask] == diagonal[mask]).float().mean()
                                             + (logit.argmax(0)[mask] == diagonal[mask]).float().mean()) / 2)
    result["old_to_new_mass"] = float(logit.softmax(1)[old_mask][:, new_mask].sum(1).mean())
    result["new_to_old_mass"] = float(logit.softmax(1)[new_mask][:, old_mask].sum(1).mean())
    return result


def fit_teacher(support, base, seed, mode, cfg, device):
    R.sync(device)
    started = time.perf_counter()
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model = R.G.CLIPStudent(support["x"].shape[1], text_dim=support["text_emb"].shape[1]).to(device)
    x, text, adj = support["x"].to(device), support["text_emb"].to(device), adjacency(support, device)
    opt = torch.optim.Adam(model.parameters(), lr=.001)
    gen = torch.Generator().manual_seed(seed * 7 + 1)
    old_n, sampled_old, sampled_total, losses = support["old_rows"], 0, 0, []
    sequence = []
    for _ in range(cfg.teacher_steps):
        ids = sample_rows(len(x), old_n, cfg.teacher_batch, gen, mode, old_n / len(base["x"]))
        if len(ids) != len(ids.unique()):
            raise AssertionError("duplicate contrastive IDs")
        sampled_old += int((ids < old_n).sum())
        sampled_total += len(ids)
        sequence.append(ids)
        loss = T._info_nce(model, x, text, adj, ids.to(device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    model.eval()
    R.sync(device)
    return model, {"seed": seed, "sampler": mode, "steps": cfg.teacher_steps,
                   "sampled_old_fraction": sampled_old / sampled_total,
                   "batch_sequence_hash": R.C.tensor_hash({"ids": torch.stack(sequence)}),
                   "loss_first": losses[0], "loss_last": losses[-1],
                   "seconds": time.perf_counter() - started,
                   "state_sha256": R.C.tensor_hash(model.state_dict())}


def fit_direct_distillation(support, targets, plan, cfg, device):
    """Uncompressed distillability diagnostic, not a condensation result."""
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(8120)
        model = R.G.CLIPStudent(support["x"].shape[1], text_dim=support["text_emb"].shape[1]).to(device)
    x, text, adj = support["x"].to(device), support["text_emb"].to(device), adjacency(support, device)
    opt = torch.optim.Adam(model.parameters(), lr=.001)
    ids = plan["train"].to(device)
    all_queries = torch.ones(len(ids), dtype=torch.bool, device=device)
    curve = []
    for step in range(1, cfg.direct_steps + 1):
        value = logits(dict(model.named_parameters()), x, adj, text, ids, cfg.temperature)
        loss = divergence(value, targets["train"], all_queries)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step in (1, min(32, cfg.direct_steps), cfg.direct_steps):
            with torch.no_grad():
                v = plan["validation"].to(device)
                score = logits(dict(model.named_parameters()), x, adj, text, v, cfg.temperature)
                val = divergence(score, targets["validation"], torch.ones(len(v), dtype=torch.bool, device=device))
            curve.append({"step": step, "train_kl": float(loss.detach()), "validation_kl": float(val)})
    return {"curve": curve, "compressed": False, "report_labels_used": False}


def build_context(base, old, delta, lifted, stage, cfg, device, diagnostics=True):
    R.sync(device)
    started = time.perf_counter()
    plan = probe_plan(len(old["x"]), len(delta["x"]), stage, cfg)
    contexts, audits = {}, {}
    base_adj = R.C.training_adjacency(base, device).to_dense()
    old_n = len(old["x"])
    for name in SUPPORTS:
        support = T._concat_support(old, delta) if name.startswith("legacy") else lifted
        adj = adjacency(support, device)
        targets, raw_logits, teacher_rows = {}, {}, []
        for seed in cfg.teacher_seeds:
            model, audit = fit_teacher(support, base, seed, name.split("_")[1], cfg, device)
            audit["probes"] = {}
            with torch.no_grad():
                for split, ids in plan.items():
                    value = logits(dict(model.named_parameters()), support["x"].to(device), adj,
                                   support["text_emb"].to(device), ids.to(device), cfg.temperature)
                    audit["probes"][split] = stats(value, ids.to(device) < old_n)
                    raw_logits.setdefault(split, []).append(value.detach())
            if name == "legacy_uniform" and seed == cfg.teacher_seeds[0]:
                observation = T._teacher_observation(model, support, adj, device)
                utility = observation["gradient_norm"][:old_n] * (2 - observation["alignment"][:old_n]).clamp_min(1.)
                utility *= 1 + T._delta_affinity(old, delta).clamp_min(0.)
            teacher_rows.append(audit)
            del model
        for split in plan:
            pairs = [probabilities(v) for v in raw_logits[split]]
            targets[split] = tuple(torch.stack([p[i] for p in pairs]).mean(0).detach() for i in (0, 1))
        old_error = float((adj.to_dense()[:old_n, :old_n] - base_adj[:old_n, :old_n]).abs().max())
        contexts[name] = {"support": support, "adj": adj, "plan": plan, "targets": targets,
                          "teacher_logits": {s: torch.stack(v).mean(0).detach() for s, v in raw_logits.items()}}
        ensemble_stats = {split: stats(torch.stack(values).mean(0), plan[split].to(device) < old_n)
                          for split, values in raw_logits.items()}
        disagreement = {split: float(torch.stack([
            divergence(v, targets[split], torch.ones(len(v), dtype=torch.bool, device=device))
            for v in values]).mean()) for split, values in raw_logits.items()}
        audits[name] = {"teachers": teacher_rows, "support_sha256": T.memory_hash(support),
                        "ensemble_probes": ensemble_stats, "teacher_disagreement_kl": disagreement,
                        "old_normalized_block_max_error": old_error,
                        "plan": {s: v.tolist() for s, v in plan.items()}}
        if diagnostics:
            audits[name]["direct_distillation"] = fit_direct_distillation(support, targets, plan, cfg, device)
        if name.startswith("lift") and old_error > 2e-6:
            raise AssertionError("lift failed to preserve old normalized block")
    protected = set(T.protected_rows(old, .8).tolist())
    quota = min(old_n - len(protected), max(1, math.floor(.1 * old_n)))
    editable = sorted((i for i in range(old_n) if i not in protected), key=lambda i: (-float(utility[i]), i))[:quota]
    R.sync(device)
    return contexts, {"supports": audits, "editable_old": editable, "protected": sorted(protected),
                      "seconds": time.perf_counter() - started, "config": asdict(cfg)}


def validate_memory(base, candidate, rows, radius):
    if base.keys() != candidate.keys():
        raise AssertionError("schema changed")
    for key, value in base.items():
        if key == "x":
            continue
        equal = torch.equal(value, candidate[key]) if isinstance(value, torch.Tensor) else value == candidate[key]
        if not equal:
            raise AssertionError(f"immutable field changed: {key}")
    if candidate["x"].shape != base["x"].shape or not torch.isfinite(candidate["x"]).all():
        raise AssertionError("nonfinite/wrong shape X")
    changed = torch.where((base["x"] != candidate["x"]).any(1))[0].tolist()
    if not set(changed) <= set(rows):
        raise AssertionError("forbidden edit")
    drift = (candidate["x"] - base["x"]).norm(dim=1) / base["x"].norm(dim=1).clamp_min(1e-6)
    if float(drift.max()) > radius + 1e-5:
        raise AssertionError("displacement bound violated")
    return {"changed_rows": changed, "max_relative_displacement": float(drift.max())}


def optimize(base, context, editable_old, stage, location, objective, cfg, device,
             preserve=True, stop=False, shuffled=False):
    R.sync(device)
    started = time.perf_counter()
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
    support, plan = context["support"], context["plan"]
    old_n = support["old_rows"]
    if location not in ("old", "new", "joint"):
        raise ValueError("unknown editable location")
    if len(editable_old) != len(set(editable_old)) or any(i < 0 or i >= old_n for i in editable_old):
        raise ValueError("invalid editable old rows")
    rows = (list(editable_old) if location == "old" else list(range(old_n, len(base["x"]))))
    if location == "joint":
        rows = sorted(set(rows + list(editable_old)))
    if not rows:
        raise ValueError("empty optimization mask")
    base_hash, support_hash = T.memory_hash(base), T.memory_hash(support)
    x, text = base["x"].to(device), base["text_emb"].to(device)
    adj = R.C.training_adjacency(base, device)
    px, pt, pa = support["x"].to(device).detach(), support["text_emb"].to(device).detach(), context["adj"]
    selected = torch.tensor(rows, device=device)
    relative = torch.nn.Parameter(torch.zeros_like(x[selected]))
    optimizer = torch.optim.Adam([relative], lr=cfg.outer_lr)
    baseline = {}
    targets = {split: tuple(p.to(device).detach() for p in pairs)
               for split, pairs in context["targets"].items()}
    if shuffled:
        generator = torch.Generator().manual_seed(8140 + stage)
        targets = {}
        for split, pairs in context["targets"].items():
            pairs = tuple(p.to(device).detach() for p in pairs)
            perm = torch.randperm(len(pairs[0]), generator=generator).to(device)
            targets[split] = (pairs[0][:, perm], pairs[1][perm, :])
    for split, seeds in (("train", cfg.train_seeds), ("validation", cfg.validation_seeds)):
        baseline[split] = []
        for seed in seeds:
            params = unroll(x, adj, text, seed, cfg.horizon, False)
            with torch.no_grad():
                ids = plan[split].to(device)
                value = logits(params, px, pa, pt, ids, cfg.temperature)
                baseline[split].append(tuple(v.detach() for v in probabilities(value)))

    def response(split, differentiable):
        seeds = cfg.train_seeds if split == "train" else cfg.validation_seeds
        new_values, old_values = [], []
        ids = plan[split].to(device)
        old_mask = ids < old_n
        for index, seed in enumerate(seeds):
            candidate_x = S.changed_x(x, selected, relative)
            params = unroll(candidate_x, adj, text, seed, cfg.horizon, differentiable)
            value = logits(params, px, pa, pt, ids, cfg.temperature)
            if objective == "relation":
                new_loss = divergence(value, targets[split], ~old_mask)
            elif objective == "infonce":
                new_loss = hard_loss(value, ~old_mask)
            elif objective == "asm":
                teacher = context["teacher_logits"][split].to(device)
                new_loss = _asm_loss(value[~old_mask][:, ~old_mask], teacher[~old_mask][:, ~old_mask])
            else:
                raise ValueError(objective)
            old_loss = divergence(value, baseline[split][index], old_mask)
            new_values.append(new_loss)
            old_values.append(old_loss)
        return torch.stack(new_values).mean(), torch.stack(old_values).mean()

    initial = {}
    for split in plan:
        new_loss, old_loss = response(split, False)
        initial[split] = {"new": float(new_loss.detach()), "old_kl": float(old_loss.detach())}
    curve, dual = [], cfg.initial_dual
    for step in range(1, cfg.outer_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        new_loss, old_loss = response("train", not stop)
        value = new_loss + (dual * (old_loss - cfg.old_epsilon) if preserve else 0.)
        if stop:
            value = value + relative.sum() * 0.
        value.backward()
        if relative.grad is None or not torch.isfinite(relative.grad).all() or not torch.isfinite(value):
            raise FloatingPointError("nonfinite/missing R4 hypergradient")
        grad_norm = float(relative.grad.norm())
        torch.nn.utils.clip_grad_norm_([relative], 5.)
        optimizer.step()
        S.project(relative, cfg.radius)
        measured_old = float(old_loss.detach())
        curve.append({"step": step, "new_objective": float(new_loss.detach()), "old_kl": measured_old,
                      "dual": dual, "constraint_active": measured_old > cfg.old_epsilon,
                      "gradient_norm": grad_norm})
        if preserve:
            dual = max(0., dual + cfg.dual_lr * (measured_old - cfg.old_epsilon))
    unprojected = relative.detach().clone()
    feasibility = []
    scales = (1., .5, .25, .125, .0625, .03125, .015625, 0.) if preserve else (1.,)
    for scale in scales:
        with torch.no_grad():
            relative.copy_(unprojected * scale)
        new_loss, old_loss = response("train", False)
        feasibility.append({"scale": scale, "new": float(new_loss.detach()), "old_kl": float(old_loss.detach())})
        if not preserve or float(old_loss.detach()) <= cfg.old_epsilon + 1e-7:
            break
    final = {}
    for split in plan:
        n, o = response(split, False)
        final[split] = {"new": float(n.detach()), "old_kl": float(o.detach())}
    if preserve and final["train"]["old_kl"] > cfg.old_epsilon + 1e-7:
        raise AssertionError("old relation constraint infeasible even at zero edit")
    result = T.clone_memory(base)
    result["x"] = S.changed_x(x, selected, relative.detach()).detach().cpu()
    inv = validate_memory(base, result, rows, cfg.radius)
    if stop and T.memory_hash(result) != base_hash:
        raise AssertionError("stop control changed memory")
    if T.memory_hash(base) != base_hash or T.memory_hash(support) != support_hash:
        raise AssertionError("optimizer changed frozen inputs")
    R.sync(device)
    return result, {"config": asdict(cfg), "location": location, "objective": objective,
                    "preserve": preserve, "stop": stop, "shuffled": shuffled,
                    "rows": rows, "invariants": inv, "curve": curve,
                    "initial": initial, "final": final, "feasibility": feasibility,
                    "nonzero_gradient_steps": sum(v["gradient_norm"] > 0 for v in curve),
                    "input_memory_sha256": base_hash, "support_sha256": support_hash,
                    "report_labels_used": False, "historical_raw_reads": 0,
                    "outer_train_student_steps": cfg.outer_steps * len(cfg.train_seeds) * cfg.horizon,
                    "diagnostic_student_steps": (3 * (len(cfg.train_seeds) + len(cfg.validation_seeds))
                                                 + len(feasibility) * len(cfg.train_seeds)) * cfg.horizon,
                    "selected_outer_step": cfg.outer_steps,
                    "seconds": time.perf_counter() - started,
                    "peak_cuda_bytes": torch.cuda.max_memory_allocated(device) if str(device).startswith("cuda") else 0}
