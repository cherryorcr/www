"""Label-free response trajectories on a fixed legal probe support.

This is an offline ranking diagnostic, not a deployed sequential selector.
Selection scores never receive report labels or canonical student scores.
"""
from __future__ import annotations

import time

import torch
import torch.nn.functional as F

import run_tgc_revision_gate as R
import tgc_revision_gate as T


def probe_rows(old_n: int, n: int, batch: int = 16, repeats: int = 8):
    if not 0 < old_n < n:
        raise ValueError("probe needs old synthetic and current raw rows")
    generator = torch.Generator().manual_seed(9700)
    return {name: [offset + torch.randperm(count, generator=generator)[:min(batch, count)]
                   for _ in range(repeats)]
            for name, offset, count in (("old", 0, old_n), ("new", old_n, n - old_n))}


def loss_from_embeddings(model, graph, text, rows):
    logits = model.logit_scale.exp().clamp(max=100) * graph[rows] @ text[rows].T
    labels = torch.arange(len(rows), device=graph.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


def model_hash(model):
    return R.C.tensor_hash({k: v.detach().cpu() for k, v in model.state_dict().items()})


def response_curve(memory: dict, support: dict, seed: int, device: str,
                   horizons=(32, 64, 128), probes: bool = True) -> dict:
    started = time.perf_counter()
    if not horizons or sorted(set(horizons)) != list(horizons) or horizons[0] <= 0:
        raise ValueError("horizons must be unique positive increasing steps")
    torch.manual_seed(seed)
    model = R.G.CLIPStudent(memory["x"].shape[1]).to(device)
    start_hash = model_hash(model)
    x, text = memory["x"].to(device), memory["text_emb"].to(device)
    adj = R.C.training_adjacency(memory, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=R.P.STUDENT.lr)
    generator = torch.Generator().manual_seed(seed * 7 + 1)
    px, pt = support["x"].to(device), support["text_emb"].to(device)
    pa = R.C.training_adjacency(support, device)
    batches = probe_rows(support["old_rows"], len(px))
    train_probe = torch.Generator().manual_seed(9701)
    train_rows = [torch.randperm(len(x), generator=train_probe)[:min(16, len(x))]
                  for _ in range(8)]
    trajectory = []
    training_seconds = scoring_seconds = 0.0
    R.sync(device)
    segment = time.perf_counter()
    for step in range(1, max(horizons) + 1):
        model.train()
        rows = torch.randperm(len(x), generator=generator)[:min(R.P.STUDENT.batch, len(x))].to(device)
        graph, encoded_text = model.encode_graph(x, adj)[rows], model.encode_text(text[rows])
        logits = model.logit_scale.exp().clamp(max=100) * graph @ encoded_text.T
        labels = torch.arange(len(rows), device=device)
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite selection-student loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step not in horizons:
            continue
        R.sync(device)
        training_seconds += time.perf_counter() - segment
        before = model_hash(model)
        cpu_rng = torch.random.get_rng_state().clone()
        cuda_rng = torch.cuda.get_rng_state(device).clone() if device.startswith("cuda") else None
        generator_state = generator.get_state().clone()
        row = {"step": step, "state_sha256": before}
        scoring_started = time.perf_counter()
        if probes:
            model.eval()
            with torch.no_grad():
                graph, encoded_text = model.encode_graph(px, pa), model.encode_text(pt)
                for name, sets in batches.items():
                    row[name + "_loss"] = sum(float(loss_from_embeddings(
                        model, graph, encoded_text, idx.to(device))) for idx in sets) / len(sets)
                graph, encoded_text = model.encode_graph(x, adj), model.encode_text(text)
                row["train_loss"] = sum(float(loss_from_embeddings(
                    model, graph, encoded_text, idx.to(device))) for idx in train_rows) / len(train_rows)
            row["balanced_loss"] = (row["old_loss"] + row["new_loss"]) / 2
            model.train()
        if (before != model_hash(model) or not torch.equal(cpu_rng, torch.random.get_rng_state())
                or not torch.equal(generator_state, generator.get_state())
                or (cuda_rng is not None and not torch.equal(cuda_rng, torch.cuda.get_rng_state(device)))):
            raise AssertionError("intermediate scoring changed model or RNG")
        R.sync(device)
        scoring_seconds += time.perf_counter() - scoring_started
        trajectory.append(row)
        segment = time.perf_counter()
    R.sync(device)
    return {"seed": seed, "horizons": list(horizons), "start_state_sha256": start_hash,
            "end_state_sha256": model_hash(model), "trajectory": trajectory,
            "training_seconds": training_seconds, "scoring_seconds": scoring_seconds,
            "wall_seconds": time.perf_counter() - started,
            "memory_sha256": T.memory_hash(memory), "probe_sha256": T.memory_hash(support),
            "report_labels_used": False, "training_state_unchanged_by_probe": True}


def select_indices(scores: list[dict], horizon: int | None, old_tolerance=0.02):
    """Return balanced-response, guarded-response, and own-training-loss choices."""
    measures = []
    for candidate in scores:
        observations = [r for run in candidate["responses"] for r in run["trajectory"]
                        if horizon is None or r["step"] == horizon]
        measures.append({name: sum(r[name] for r in observations) / len(observations)
                         for name in ("balanced_loss", "old_loss", "new_loss", "train_loss")})
    eligible = [i for i, m in enumerate(measures)
                if m["old_loss"] <= measures[0]["old_loss"] + old_tolerance]
    return {"balanced": min(range(len(measures)), key=lambda i: (measures[i]["balanced_loss"], i)),
            "guarded": min(eligible, key=lambda i: (measures[i]["balanced_loss"], i)),
            "own_training_loss": min(range(len(measures)), key=lambda i: (measures[i]["train_loss"], i)),
            "measures": measures}
