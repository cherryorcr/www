"""Frozen, label-free scratch-student selectors for TGC-R3."""
from __future__ import annotations

from dataclasses import asdict
import math
from pathlib import Path
from statistics import mean
import time

import torch

import tgc_response_alignment as A
import tgc_student_revision as S

R, T = S.R, S.T
WARMUP = 64
CHECKPOINTS = (32, 64, 128)
SELECTOR_SEEDS = (7820, 7821, 7822)
REPORT_SEEDS = tuple(range(9600, 9606))
DEFAULT_CFG = S.Config(train_seeds=(7800, 7801), validation_seeds=(7810, 7811))
GROUPS = {
    "cora": {"order19221": [[0, 5], [3, 4], [1, 2, 6]],
             "order19222": [[5, 6], [3, 4], [0, 1, 2]]},
    "citeseer": {"order19221": [[0, 3], [2, 4], [1, 5]],
                 "order19222": [[3, 4], [2, 5], [0, 1]]},
}
ORDERS = ("order19221", "order19222")
POLICIES = ("selected_gate", "selected_new64", "selected_mixed64", "selected_curve")
ARMS = ("c0", "teacher", "candidate", "random", "stop") + POLICIES


def probe_batches(old_n, total_n, stage):
    if min(old_n, total_n - old_n) < 8:
        raise ValueError("selector requires at least eight rows per probe stratum")
    generator = torch.Generator().manual_seed(7830 + stage)
    old, new = torch.arange(old_n), torch.arange(old_n, total_n)
    result = {"old": [], "new": [], "mixed": []}
    for _ in range(4):
        for name, ids in (("old", old), ("new", new)):
            result[name].append(ids[torch.randperm(len(ids), generator=generator)[:min(16, len(ids))]])
        result["mixed"].append(torch.cat((
            old[torch.randperm(len(old), generator=generator)[:8]],
            new[torch.randperm(len(new), generator=generator)[:8]])))
    return result


def scratch_trajectory(memory, support, batches, seed, device, checkpoints=CHECKPOINTS):
    """Reuse a prefix only *within* one scratch trajectory, never across memories."""
    if tuple(sorted(set(checkpoints))) != tuple(checkpoints) or min(checkpoints) < 1:
        raise ValueError("checkpoints must be unique increasing positive steps")
    x, text = memory["x"].to(device), memory["text_emb"].to(device)
    adj = R.C.training_adjacency(memory, device)
    px, pt = support["x"].to(device), support["text_emb"].to(device)
    pa = R.C.training_adjacency(support, device)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model = R.G.CLIPStudent(x.shape[1], text_dim=text.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=R.P.STUDENT.lr)
    generator = torch.Generator().manual_seed(seed * 7 + 1)
    start_hash = R.C.tensor_hash(model.state_dict())
    curve, states, sequence = [], {}, []
    for step in range(1, max(checkpoints) + 1):
        model.train()
        ids = torch.randperm(len(x), generator=generator)[:min(R.P.STUDENT.batch, len(x))]
        sequence.append(ids.tolist())
        loss = T._info_nce(model, x, text, adj, ids.to(device))
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite selector training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step not in checkpoints:
            continue
        # Evaluation is pure: no optimizer update, random sampling, or mode drift.
        rng_before = generator.get_state().clone()
        cpu_rng = torch.random.get_rng_state().clone()
        cuda_rng = torch.cuda.get_rng_state(device).clone() if str(device).startswith("cuda") else None
        model.eval()
        with torch.no_grad():
            scores = S.probe_losses(dict(model.named_parameters()), px, pa, pt, batches)
        model.train()
        if not torch.equal(rng_before, generator.get_state()) or not torch.equal(cpu_rng, torch.random.get_rng_state()):
            raise AssertionError("probe evaluation changed RNG")
        if cuda_rng is not None and not torch.equal(cuda_rng, torch.cuda.get_rng_state(device)):
            raise AssertionError("probe evaluation changed CUDA RNG")
        params, first, second = {}, {}, {}
        for name, value in model.named_parameters():
            state = optimizer.state[value]
            params[name] = value.detach().cpu().clone()
            first[name] = state["exp_avg"].detach().cpu().clone()
            second[name] = state["exp_avg_sq"].detach().cpu().clone()
        prefix = {"params": params, "first": first, "second": second,
                  "rng": rng_before, "steps": step, "seed": seed}
        states[step] = {"prefix": prefix, "optimizer": T.clone_value(optimizer.state_dict()),
                        "cpu_rng": cpu_rng, "cuda_rng": cuda_rng}
        record = {"step": step, **{k: float(v) for k, v in scores.items()},
                  "prefix_hash": A.prefix_hash(prefix)}
        if any(not math.isfinite(record[k]) for k in ("old", "new", "mixed")):
            raise FloatingPointError("nonfinite selector probe")
        curve.append(record)
    return {"seed": seed, "start_hash": start_hash, "curve": curve,
            "batch_sequence": sequence, "states": states}


def policy_scores(per_seed, cfg=DEFAULT_CFG):
    if len(per_seed) != len(SELECTOR_SEEDS) or {r["seed"] for r in per_seed} != set(SELECTOR_SEEDS):
        raise ValueError("incomplete/duplicate selector seeds")
    details = []
    for row in per_seed:
        curves = {arm: {r["step"]: r for r in row[arm]} for arm in ("c0", "candidate")}
        if any(len(row[a]) != len(CHECKPOINTS) or set(curves[a]) != set(CHECKPOINTS) for a in curves):
            raise ValueError("incomplete/duplicate response checkpoints")
        gains = {str(step): {key: curves["c0"][step][key] - curves["candidate"][step][key]
                            for key in ("old", "new", "mixed")} for step in CHECKPOINTS}
        if not all(math.isfinite(v) for point in gains.values() for v in point.values()):
            raise ValueError("nonfinite response gain")
        details.append({"seed": row["seed"], "gains": gains,
                        "curve_score": mean(gains[str(s)]["mixed"] for s in CHECKPOINTS)})
    policies = {}
    for name, key in (("selected_new64", "new"), ("selected_mixed64", "mixed"),
                      ("selected_curve", None)):
        scores = [r["curve_score"] if key is None else r["gains"]["64"][key] for r in details]
        times = CHECKPOINTS if key is None else (64,)
        old_safe = all(r["gains"][str(step)]["old"] >= -cfg.old_tolerance
                       for r in details for step in times)
        positive = sum(v >= cfg.gain_min for v in scores)
        accepted = mean(scores) >= cfg.gain_min and positive >= 2 and old_safe
        policies[name] = {"mean_gain": mean(scores), "positive_seeds": positive,
                          "old_safe": old_safe, "accepted": accepted,
                          "selected_arm": "candidate" if accepted else "c0"}
    return {"per_seed_gains": details, "policies": policies}


def curve_scores(base, candidate, support, stage, device, out: Path):
    batches = probe_batches(support["old_rows"], len(support["x"]), stage)
    R.sync(device)
    started = time.perf_counter()
    records = []
    for seed in SELECTOR_SEEDS:
        trajectories = {name: scratch_trajectory(memory, support, batches, seed, device)
                        for name, memory in (("c0", base), ("candidate", candidate))}
        left, right = trajectories["c0"], trajectories["candidate"]
        if left["start_hash"] != right["start_hash"] or left["batch_sequence"] != right["batch_sequence"]:
            raise AssertionError("unpaired selector scratch trajectories")
        path = out / f"s{stage}_selector_seed{seed}.pt"
        torch.save(trajectories, path)
        records.append({"seed": seed, "start_hash": left["start_hash"],
                        "c0": left["curve"], "candidate": right["curve"],
                        "checkpoint_file": path.name, "checkpoint_sha256": R.C.sha(path)})
    R.sync(device)
    return {"per_seed": records, **policy_scores(records),
            "seconds": time.perf_counter() - started,
            "student_steps": 2 * len(SELECTOR_SEEDS) * max(CHECKPOINTS),
            "probe_batches": {k: [v.tolist() for v in values] for k, values in batches.items()},
            "labels_used": False, "historical_raw_reads": 0,
            "probe_scope": "old synthetic and current raw delta; unlabeled reuse, not held-out data"}


def decision_summary(optimization, curve):
    policies = policy_scores(curve["per_seed"])["policies"]
    policies["selected_gate"] = {"accepted": bool(optimization["accepted"]),
                                 "selected_arm": "candidate" if optimization["accepted"] else "c0"}
    return {"primary_policy": "selected_curve", "policies": policies,
            "selector_labels_used": False, "historical_raw_reads": 0}


def frozen_protocol():
    groups = [set(DEFAULT_CFG.train_seeds), set(DEFAULT_CFG.validation_seeds),
              set(SELECTOR_SEEDS), set(REPORT_SEEDS)]
    if any(a & b for i, a in enumerate(groups) for b in groups[i + 1:]):
        raise ValueError("overlapping seed roles")
    return {"name": "tgc-r3-scratch-selector-v2", "groups": GROUPS, "rate": .05,
            "stages": [0, 1, 2], "arms": list(ARMS), "report_seeds": list(REPORT_SEEDS),
            "selector_seeds": list(SELECTOR_SEEDS), "checkpoints": list(CHECKPOINTS),
            "meta_config": asdict(DEFAULT_CFG), "warmup": WARMUP, "suffix": 1,
            "primary_policy": "selected_curve", "probe_seed": "7830+stage",
            "student": asdict(R.P.STUDENT), "teacher": T.RevisionConfig().dict(),
            "scope": "independent orders/seeds, shared C0 parents, not recursive rollout",
            "selector_rule": "mean gain>=1e-4; >=2/3 positive seeds; every old loss rise<=.01",
            "curve_rule": "equal mean of mixed 8-old/8-new InfoNCE gains at 32/64/128",
            "selector_labels_used": False, "test_labels_evaluated": False,
            "promotion": {"final_mean_min_pp": .5, "final_cells_ge_minus1": 3,
                          "final_worst_min_pp": -2, "s1_mean_min_pp": 0,
                          "mean_new_min_pp": 0, "worst_old_min_pp": -2,
                          "vs_random_min_pp_strict": 0, "vs_teacher_min_pp": 0}}
