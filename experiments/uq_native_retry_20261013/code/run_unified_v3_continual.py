"""Unified v3 continual runner for GraphCLIP/TAGBench/arXiv schemas.

This runner is deliberately conservative: it keeps the existing student and
construction implementations, but gives every arm the same data split, budget,
student seeds, output manifest, and stagewise metrics.  New mechanisms are
explicit CLI switches, so the frozen-old append result remains the reference.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
LOCAL_SCA = HERE / "run_student_compatible_append.py"
if not LOCAL_SCA.exists():
    sys.path.insert(0, str(HERE.parent / "frozen_v1_2026_09_09"))
elif not (HERE / "run_gate1_continual.py").exists():
    sys.path.insert(1, str(HERE.parent / "frozen_v1_2026_09_09"))
import protocol_v3
from run_multiscale_asm import DATA_CFG, FrozenMiniLM
import run_gate1_continual as G
import run_student_compatible_append as SCA


GRAPHCLIP_CLASSES = {
    "cora": ["case based", "genetic algorithms", "neural networks",
             "probabilistic methods", "reinforcement learning",
             "rule learning", "theory"],
    "citeseer": ["agents", "machine learning", "information retrieval",
                 "databases", "human computer interaction",
                 "artificial intelligence"],
}


def derive_stage_classes(y: torch.Tensor, pool_mask: torch.Tensor,
                         n_stages: int = 3, split_mode: str = "round_robin",
                         base_fraction: float = 0.4) -> list[list[int]]:
    """Greedily balance class counts without using test rows."""
    counts = torch.bincount(y[pool_mask], minlength=int(y.max()) + 1)
    totals = [0] * n_stages
    stages = [[] for _ in range(n_stages)]
    # Round-robin classes by descending pool size.  Pure size balancing can
    # leave the largest class alone in stage 0 on tagbench, making the
    # continual protocol depend on one dataset's class imbalance.  A class
    # count balanced split is easier to compare across datasets; node-count
    # imbalance is recorded in the manifest instead of hidden by the split.
    order = torch.argsort(counts, descending=True).tolist()
    if split_mode == "frequency":
        order = sorted(order)
    if split_mode == "base_incremental":
        base_count = max(1, min(len(order) - 1,
                                int(round(len(order) * float(base_fraction)))))
        stages[0] = [int(v) for v in order[:base_count]]
        totals[0] = int(counts[order[:base_count]].sum())
        for index, cls in enumerate(order[base_count:]):
            target = 1 + (index % max(1, n_stages - 1))
            stages[target].append(int(cls))
            totals[target] += int(counts[cls])
        return [sorted(s) for s in stages if s]
    for index, cls in enumerate(order):
        target = index % n_stages
        stages[target].append(int(cls))
        totals[target] += int(counts[cls])
    return [sorted(s) for s in stages if s]


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(value) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_unified(dataset: str, path: str, minilm: str, device: str,
                 n_stages: int | None = None, split_mode: str = "canonical",
                 base_fraction: float = 0.4,
                 report_dev_mod: int | None = protocol_v3.REPORT_DEV_MOD) -> dict:
    d = torch.load(path, map_location="cpu", weights_only=False)
    if hasattr(d, "raw_texts"):
        x = d.x.float()
        y = d.y.long()
        texts = list(d.raw_texts)
        edge_index = d.edge_index.long()
        original_ids = torch.arange(len(x), dtype=torch.long)
        names = GRAPHCLIP_CLASSES[dataset]
        prompts = [f"a research paper about {name}" for name in names]
    else:
        x = d["node_f"].float()
        texts = list(d["text_list"])
        names = [str(v) for v in np.asarray(d["all_labels"]).tolist()]
        name_to_id = {name: i for i, name in enumerate(names)}
        y = torch.tensor([name_to_id[str(v)]
                          for v in np.asarray(d["label_list"]).tolist()],
                         dtype=torch.long)
        edge_index = d["edge_index"].long()
        original_ids = d.get("original_node_ids",
                             torch.arange(len(x), dtype=torch.long)).long()
        desc = [str(v) for v in d.get("labels_desc", names)]
        if dataset == "arxiv":
            clean = lambda s: s.replace("arXiv computer science category: ", "").replace(
                "arXiv computer science category ", "")
            prompts = [f"a computer science paper about {clean(v).lower()}"
                       for v in desc]
        else:
            prompts = [f"a document about {v}" for v in desc]

    selector_dev_mask, report_dev_mask, test_mask, pool_mask = \
        protocol_v3.split_masks_v4(original_ids, report_dev_mod)
    split_source = "graphclip_fixed"
    requested_stages = 3 if n_stages is None else int(n_stages)
    if requested_stages < 2:
        raise ValueError("n_stages must be at least 2")
    if dataset in GRAPHCLIP_CLASSES and requested_stages == 3 and split_mode == "canonical":
        if dataset == "cora":
            stage_classes = [[0, 1], [2, 3], [4, 5, 6]]
        else:
            stage_classes = [[0, 1], [2, 3], [4, 5]]
        split_source = "graphclip_fixed"
    else:
        stage_classes = None
        split_source = None
        if dataset == "arxiv" and requested_stages == 3 and split_mode == "canonical":
            # Reuse the frozen class-incremental split when it is available.
            # This keeps the new unified runner comparable to the earlier
            # arXiv continual evidence instead of silently inventing a new
            # stream partition.
            data_path = Path(path)
            split_prefix = data_path.parent / "ogbn_arxiv_full_classincr3"
            split_classes = []
            for stage in (1, 2, 3):
                split_file = split_prefix.parent / f"{split_prefix.name}_delta_t{stage}.pt"
                if not split_file.exists():
                    split_classes = []
                    break
                delta = torch.load(split_file, map_location="cpu", weights_only=False)
                names_in_stage = {str(v) for v in np.asarray(delta["label_list"]).tolist()}
                split_classes.append(sorted(name_to_id[name] for name in names_in_stage))
            if split_classes:
                stage_classes = split_classes
                split_source = str(split_prefix)
        if stage_classes is None:
            actual_split_mode = ("round_robin" if split_mode == "canonical"
                                 else split_mode)
            stage_classes = derive_stage_classes(
                y, pool_mask, requested_stages, actual_split_mode,
                base_fraction)
            split_source = f"deterministic_{actual_split_mode}_by_pool_count"
    stage_classes = protocol_v3.canonicalize_class_groups(stage_classes)
    G.CLASS_NAMES = names
    G.PROMPTS = prompts
    G.STAGE_CLASSES = stage_classes
    return {
        "dataset": dataset,
        "x": x,
        "y": y,
        "texts": texts,
        "edge_index": edge_index,
        "original_ids": original_ids,
        # Keep the old key as the reporting mask for downstream helpers;
        # selector-dev is passed explicitly to SCA below.
        "dev_mask": report_dev_mask,
        "selector_dev_mask": selector_dev_mask,
        "report_dev_mask": report_dev_mask,
        "test_mask": test_mask,
        "pool_mask": pool_mask,
        "stage_classes": stage_classes,
        "stage_split_source": split_source,
        "prompts": prompts,
        "encoder": FrozenMiniLM(minilm, device),
    }


def response_steps(total_rows: int) -> list[int]:
    final = protocol_v3.student_steps(total_rows)
    base = max(128, final // 4)
    return sorted(set(max(50, min(final, int(round(base * m))))
                   for m in (0.5, 1.0, 2.0)))


def information_utility(D: dict, ids: torch.Tensor,
                        prior_ids: torch.Tensor | None) -> float:
    """Estimate delta utility from source-only novelty and dispersion."""
    difficulty = SCA.delta_difficulty(D["x"][ids], D["text_emb"][ids])
    if prior_ids is None or len(prior_ids) == 0 or len(ids) == 0:
        novelty = difficulty
    else:
        current = torch.cat((F.normalize(D["x"][ids].float(), dim=1),
                             D["text_emb"][ids].float()), dim=1).mean(0)
        prior = torch.cat((F.normalize(D["x"][prior_ids].float(), dim=1),
                           D["text_emb"][prior_ids].float()), dim=1).mean(0)
        novelty = float((1.0 - F.cosine_similarity(
            current.unsqueeze(0), prior.unsqueeze(0)).squeeze()).clamp(0.0, 1.0))
    return float(max(0.0, min(1.0, 0.5 * difficulty + 0.5 * novelty)))


def _build_compact_block(*args, x_operator: str = "mean", **kwargs) -> dict:
    """Call the SCA constructor while preserving old test doubles/API users."""
    if x_operator != "mean":
        kwargs["x_operator"] = x_operator
    return SCA.build_compact_block(*args, **kwargs)


def run_students(memory: dict, stage: int, D: dict, args, device: str) -> dict:
    syn = G.memory_to_trainable(memory)
    real_adj = D["real_adj"]
    xg = D["x_device"]
    prompt_emb = D["prompt_emb"]
    rows = []
    started = time.perf_counter()
    for seed in args.student_seeds:
        steps = protocol_v3.student_steps(len(syn["x"]),
                                          override=args.student_steps)
        model = G.train_student(syn, int(seed), device, steps,
                                batch=args.student_batch)
        score = G.eval_stagewise(model, xg, real_adj, prompt_emb, D["y"],
                                 D["report_dev_mask"], stage)
        rows.append(score)
        del model
    return {
        "roots": int(len(syn["x"])),
        "student_steps": int(steps),
        "student_batch": int(args.student_batch),
        "student_seeds": list(args.student_seeds),
        "student_training_runs": int(len(args.student_seeds)),
        "student_optimizer_steps_total": int(steps * len(args.student_seeds)),
        "student_train_seconds": float(time.perf_counter() - started),
        "cum_dev_f1_mean": float(np.mean([r["cumulative_macro_f1"] for r in rows])),
        "cum_dev_f1_std": float(np.std([r["cumulative_macro_f1"] for r in rows])),
        "current_f1_mean": float(np.mean([r["current_f1"] for r in rows])),
        "old_mean_f1_mean": (float(np.mean([r["old_mean_f1"] for r in rows]))
                             if stage > 0 else None),
        "per_stage_mean": [float(np.mean([r["per_stage"][s]["macro_f1"]
                                           for r in rows]))
                           for s in range(stage + 1)],
        "per_student": rows,
    }


def _allocate_integer_budget(weights: torch.Tensor, budget: int,
                             minimum: int = 0) -> list[int]:
    """Largest-remainder allocation with a per-class floor."""
    budget, count = int(budget), int(len(weights))
    if count == 0 or budget <= 0:
        return [0] * count
    minimum = min(int(minimum), budget // count)
    remaining = budget - minimum * count
    weights = weights.float().clamp_min(1e-8)
    raw = remaining * weights / weights.sum()
    alloc = torch.floor(raw).long() + minimum
    left = budget - int(alloc.sum())
    if left > 0:
        order = torch.argsort(raw - torch.floor(raw), descending=True).tolist()
        for i in order[:left]:
            alloc[i] += 1
    return [int(v) for v in alloc.tolist()]


def allocate_rows_by_class(D: dict, ids: torch.Tensor, classes: list[int],
                           budget: int, mode: str) -> dict[int, int]:
    """Allocate a new block by class without using dev/test labels."""
    classes = sorted(int(c) for c in classes)
    counts, dispersions = [], []
    for cls in classes:
        local = ids[D["y"][ids] == cls]
        counts.append(max(1, len(local)))
        if len(local) <= 1:
            dispersions.append(0.0)
        else:
            x = F.normalize(D["x"][local].float(), dim=1)
            t = F.normalize(D["text_emb"][local].float(), dim=1)
            dx = 1.0 - (x @ x.mean(0, keepdim=True).T).mean()
            dt = 1.0 - (t @ t.mean(0, keepdim=True).T).mean()
            dispersions.append(float(torch.clamp((dx + dt) * 0.5, 0.0, 1.0)))
    count_t = torch.tensor(counts, dtype=torch.float32)
    if mode == "legacy_15":
        required = 15 * len(classes)
        if int(budget) >= required:
            weights = torch.ones_like(count_t)
            minimum = 15
        else:
            # Preserve a runnable low-rate control: when the requested total
            # budget cannot satisfy the historical floor, allocate equally
            # with one row per class whenever possible and record the actual
            # allocation in the stage result.
            weights = torch.ones_like(count_t)
            minimum = 1 if int(budget) >= len(classes) else 0
    elif mode == "equal":
        weights = torch.ones_like(count_t)
    elif mode == "linear":
        weights = count_t
    elif mode == "sqrt":
        weights = count_t.sqrt()
    elif mode == "dispersion":
        weights = count_t.sqrt() * (1.0 + torch.tensor(dispersions))
    else:
        raise ValueError(f"unknown allocation mode: {mode}")
    if mode != "legacy_15":
        minimum = 1 if int(budget) >= len(classes) else 0
    return dict(zip(classes, _allocate_integer_budget(weights, int(budget), minimum)))


def _subcluster_count(n: int, max_subclusters: int = 8) -> int:
    """Choose a small deterministic number of within-class subclusters."""
    n = int(n)
    if n <= 8:
        return 1
    return max(2, min(int(max_subclusters), n, int(math.sqrt(n / 64.0)) + 1))


def _subcluster_records(D: dict, ids: torch.Tensor, classes: list[int],
                        device: str | None = None,
                        max_subclusters: int = 8) -> list[dict]:
    """Build synthetic-only allocation units inside each class.

    Labels are used only to define the continual class protocol.  The actual
    units are discovered in the joint graph-text space and contain source
    indices only as construction inputs; no source node is copied to memory.
    """
    records = []
    for cls in sorted(int(c) for c in classes):
        class_ids = ids[D["y"][ids] == cls]
        if len(class_ids) == 0:
            continue
        points = torch.cat((F.normalize(D["x"][class_ids].float(), dim=1),
                            D["text_emb"][class_ids].float()), dim=1)
        q = _subcluster_count(len(class_ids), max_subclusters)
        assign = SCA.cluster_assign(points, torch.ones(len(points)), q,
                                    device=device)
        for sub_id in range(int(assign.max().item()) + 1):
            local = torch.where(assign == sub_id)[0]
            member_ids = class_ids[local]
            member_points = points[local]
            center = member_points.mean(0, keepdim=True)
            dispersion = float((1.0 - F.cosine_similarity(
                F.normalize(member_points, dim=1),
                F.normalize(center, dim=1))).mean().clamp(0.0, 1.0))
            records.append({
                "class": cls,
                "subcluster": int(sub_id),
                "ids": member_ids,
                "mass": int(len(member_ids)),
                "dispersion": dispersion,
                "complexity": float(math.sqrt(max(1, len(member_ids)))
                                     * (1.0 + dispersion)),
            })
    return records


def build_subcluster_complexity_block(
        D: dict, ids: torch.Tensor, k: int, encoder, stage: int,
        recipe: str, text_operator: str, device: str | None = None,
        classes: list[int] | None = None,
        max_subclusters: int = 8,
        diversity_lambda: float = 0.5,
        dual_alpha: float = 0.5,
        x_operator: str = "mean") -> dict:
    """Allocate rows to within-class joint graph-text subclusters."""
    if classes is None:
        classes = sorted(set(int(v) for v in D["y"][ids].tolist()))
    records = _subcluster_records(D, ids, classes, device, max_subclusters)
    if not records:
        raise ValueError("subcluster allocation found no source records")
    budget = min(int(k), sum(int(r["mass"]) for r in records))
    weights = torch.tensor([r["complexity"] for r in records],
                           dtype=torch.float32)
    minimum = 1 if budget >= len(records) else 0
    rows = _allocate_integer_budget(weights, budget, minimum)
    blocks = []
    allocation = []
    for record, row_count in zip(records, rows):
        row_count = min(int(row_count), int(record["mass"]))
        allocation.append({
            "class": int(record["class"]),
            "subcluster": int(record["subcluster"]),
            "source_mass": int(record["mass"]),
            "dispersion": float(record["dispersion"]),
            "complexity": float(record["complexity"]),
            "rows": int(row_count),
        })
        if row_count <= 0:
            continue
        member_ids = record["ids"]
        blocks.append(_build_compact_block(
            D["x"][member_ids], D["text_emb"][member_ids],
            [D["texts"][int(i)] for i in member_ids], row_count, encoder,
            recipe, stage, text_operator=text_operator, device=device,
            diversity_lambda=diversity_lambda, dual_alpha=dual_alpha,
            x_operator=x_operator))
    if not blocks:
        raise ValueError("subcluster allocation produced no rows")
    return {
        "x": torch.cat([b["x"] for b in blocks]),
        "text_emb": torch.cat([b["text_emb"] for b in blocks]),
        "texts": sum((b["texts"] for b in blocks), []),
        "mass": torch.cat([b["mass"] for b in blocks]),
        "depth": torch.cat([b["depth"] for b in blocks]),
        "stage": torch.cat([b["stage"] for b in blocks]),
        "allocation": allocation,
        "requested_rows": int(k),
        "actual_rows": int(sum(len(b["x"]) for b in blocks)),
        "subcluster_count": int(len(records)),
    }


def build_stage_block(D: dict, ids: torch.Tensor, k: int, encoder,
                      stage: int, recipe: str, text_operator: str,
                      device: str | None = None,
                      allocation_mode: str = "legacy_15",
                      classes: list[int] | None = None,
                      diversity_lambda: float = 0.5,
                      dual_alpha: float = 0.5,
                      x_operator: str = "mean") -> dict:
    if classes is None:
        classes = sorted(set(int(v) for v in D["y"][ids].tolist()))
    else:
        classes = sorted(int(c) for c in classes)
    available = sum(int((D["y"][ids] == int(cls)).sum()) for cls in classes)
    effective_k = min(int(k), available)
    if effective_k <= 0:
        raise ValueError("stage block has no available source rows")
    if allocation_mode == "subcluster_complexity":
        return build_subcluster_complexity_block(
            D, ids, effective_k, encoder, stage, recipe, text_operator,
            device=device, classes=classes,
            diversity_lambda=diversity_lambda, dual_alpha=dual_alpha,
            x_operator=x_operator)
    allocation = allocate_rows_by_class(D, ids, classes, effective_k, allocation_mode)
    blocks = []
    for cls in classes:
        class_ids = ids[D["y"][ids] == int(cls)]
        rows = allocation[int(cls)]
        if rows <= 0 or len(class_ids) == 0:
            continue
        rows = min(rows, len(class_ids))
        blocks.append(_build_compact_block(
            D["x"][class_ids], D["text_emb"][class_ids],
            [D["texts"][int(i)] for i in class_ids], rows, encoder,
            recipe, stage, text_operator=text_operator, device=device,
            diversity_lambda=diversity_lambda, dual_alpha=dual_alpha,
            x_operator=x_operator))
    if not blocks:
        raise ValueError("stage block allocation produced no rows")
    return {
        "x": torch.cat([b["x"] for b in blocks]),
        "text_emb": torch.cat([b["text_emb"] for b in blocks]),
        "texts": sum((b["texts"] for b in blocks), []),
        "mass": torch.cat([b["mass"] for b in blocks]),
        "depth": torch.cat([b["depth"] for b in blocks]),
        "stage": torch.cat([b["stage"] for b in blocks]),
        "allocation": allocation,
        "requested_rows": int(k),
        "actual_rows": int(sum(len(b["x"]) for b in blocks)),
    }


def build_stage_block_from_class_rows(
        D: dict, ids: torch.Tensor, class_rows: dict[int, int], encoder,
        stage: int, recipe: str, text_operator: str,
        device: str | None = None,
        inner_mode: str = "dispersion",
        diversity_lambda: float = 0.5,
        dual_alpha: float = 0.5,
        x_operator: str = "mean") -> dict:
    """Build a synthetic block from an explicit class-row allocation."""
    blocks = []
    for cls in sorted(int(c) for c in class_rows):
        rows = int(class_rows[cls])
        class_ids = ids[D["y"][ids] == cls]
        if rows <= 0 or len(class_ids) == 0:
            continue
        rows = min(rows, len(class_ids))
        if inner_mode == "subcluster_complexity":
            block = build_subcluster_complexity_block(
                D, class_ids, rows, encoder, stage, recipe, text_operator,
                device=device, classes=[cls],
                diversity_lambda=diversity_lambda, dual_alpha=dual_alpha,
                x_operator=x_operator)
        else:
            block = _build_compact_block(
                D["x"][class_ids], D["text_emb"][class_ids],
                [D["texts"][int(i)] for i in class_ids], rows, encoder,
                recipe, stage, text_operator=text_operator, device=device,
                diversity_lambda=diversity_lambda, dual_alpha=dual_alpha,
                x_operator=x_operator)
        blocks.append(block)
    if not blocks:
        raise ValueError("explicit class allocation produced no rows")
    return {
        "x": torch.cat([b["x"] for b in blocks]),
        "text_emb": torch.cat([b["text_emb"] for b in blocks]),
        "texts": sum((b["texts"] for b in blocks), []),
        "mass": torch.cat([b["mass"] for b in blocks]),
        "depth": torch.cat([b["depth"] for b in blocks]),
        "stage": torch.cat([b["stage"] for b in blocks]),
        "allocation": class_rows,
        "requested_rows": int(sum(class_rows.values())),
        "actual_rows": int(sum(len(b["x"]) for b in blocks)),
    }


def concat_synthetic_blocks(first: dict, second: dict) -> dict:
    """Concatenate synthetic blocks while preserving their provenance fields."""
    return {
        "x": torch.cat((first["x"], second["x"])),
        "text_emb": torch.cat((first["text_emb"], second["text_emb"])),
        "texts": list(first["texts"]) + list(second["texts"]),
        "mass": torch.cat((first["mass"], second["mass"])),
        "depth": torch.cat((first["depth"], second["depth"])),
        "stage": torch.cat((first["stage"], second["stage"])),
        "allocation": {
            "base": first.get("allocation"),
            "extra": second.get("allocation"),
        },
        "requested_rows": int(len(first["x"]) + len(second["x"])),
        "actual_rows": int(len(first["x"]) + len(second["x"])),
    }


def _base_class_allocation(D: dict, ids: torch.Tensor, classes: list[int],
                           budget: int, mode: str) -> dict[int, int]:
    """Return a class-level starting point for response allocation."""
    if mode == "subcluster_complexity":
        records = _subcluster_records(D, ids, classes)
        class_weights = {}
        for record in records:
            cls = int(record["class"])
            class_weights[cls] = class_weights.get(cls, 0.0) + float(record["complexity"])
        weights = torch.tensor([class_weights.get(int(c), 1.0) for c in classes])
        rows = _allocate_integer_budget(
            weights, int(budget), 1 if budget >= len(classes) else 0)
        return dict(zip([int(c) for c in classes], rows))
    return allocate_rows_by_class(D, ids, classes, budget, mode)


def _response_score(response: dict, old_weight: float = 0.25) -> float:
    old = response.get("old_tail_mean")
    return float(response["cumulative_tail_mean"]
                 + float(old_weight) * (0.0 if old is None else old))


def _response_uncertainty(response: dict) -> float:
    """Conservative uncertainty proxy across response horizons and grids."""
    return float(max(
        response.get("cumulative_tail_std", 0.0),
        response.get("tail_std", 0.0),
    ))


def _response_lcb(response: dict, old_weight: float = 0.25,
                  risk_z: float = 1.0) -> float:
    """Risk-adjusted response score used by the v4 selector."""
    score = _response_score(response, old_weight)
    return float(score - float(risk_z) * _response_uncertainty(response))


def _response_horizon_scores(response: dict,
                             old_weight: float = 0.25) -> list[float]:
    """Return one utility score per response horizon."""
    scores = []
    for grid in response.get("grids", []):
        old = grid.get("old_mean_f1")
        old_value = 0.0 if old is None else float(old["mean"])
        scores.append(float(grid["cumulative_f1"]["mean"])
                      + float(old_weight) * old_value)
    return scores


def _response_horizon_gain(candidate: dict, reference: dict,
                           old_weight: float = 0.25) -> float:
    """Worst-case gain across horizons; negative means ranking disagreement."""
    left = _response_horizon_scores(candidate, old_weight)
    right = _response_horizon_scores(reference, old_weight)
    if not left or len(left) != len(right):
        return float("-inf")
    return float(min(a - b for a, b in zip(left, right)))


def _selection_cost(selection: dict | None) -> tuple[int, int]:
    """Collect selector cost from flat or composed v4 selection audits."""
    if not selection:
        return 0, 0
    if selection.get("mode") == "response_marginal_plus_reserve":
        left = _selection_cost(selection.get("marginal"))
        right = _selection_cost(selection.get("response_gain"))
        return left[0] + right[0], left[1] + right[1]
    return (int(selection.get("selection_training_runs", 0)),
            int(selection.get("selection_optimizer_steps_total", 0)))


def build_response_marginal_block(
        old_memory: dict | None, D: dict, ids: torch.Tensor, k_new: int,
        keep: int, stage: int, args, device: str) -> tuple[dict, dict]:
    """Allocate fixed new rows using selector-dev student marginal gains."""
    classes = sorted(int(c) for c in D["stage_classes"][stage])
    base_mode = args.response_base_allocation
    allocation = _base_class_allocation(D, ids, classes, int(k_new), base_mode)
    curve = response_steps(int(k_new) + int(keep))
    real_adj = D["real_adj"]
    xg = D["x_device"]
    prompt_emb = D["prompt_emb"]
    evaluation_count = 0

    def evaluate(candidate_rows: dict[int, int]) -> tuple[float, float, dict, dict]:
        nonlocal evaluation_count
        evaluation_count += 1
        block = build_stage_block_from_class_rows(
            D, ids, candidate_rows, D["encoder"], stage,
            args.text_recipe, args.text_operator, device,
            inner_mode=base_mode,
            diversity_lambda=args.text_diversity_lambda,
            dual_alpha=args.dual_prototype_alpha,
            x_operator=args.x_operator)
        memory = SCA.append_memory(
            old_memory, block, keep, topology_mode=args.topology_mode,
            cross_degree=args.cross_degree, refresh_mode=args.refresh_mode,
            refresh_ratio=args.refresh_ratio, refresh_alpha=args.refresh_alpha,
            device=device)
        response = SCA.evaluate_response_curve(
            memory, stage, args.selection_seeds, args.selection_batches,
            curve, device, xg, real_adj, prompt_emb, D["y"],
            D["selector_dev_mask"])
        score = _response_score(response, args.response_old_weight)
        lcb = _response_lcb(response, args.response_old_weight,
                            args.response_risk_z)
        return score, lcb, block, response

    current_score, current_lcb, current_block, current_response = evaluate(allocation)
    decisions = []
    for round_id in range(int(args.response_marginal_rounds)):
        candidates = []
        for recipient in classes:
            for donor in classes:
                if donor == recipient or int(allocation.get(donor, 0)) <= 1:
                    continue
                proposal = dict(allocation)
                proposal[donor] -= 1
                proposal[recipient] = proposal.get(recipient, 0) + 1
                score, lcb, block, response = evaluate(proposal)
                horizon_gain = _response_horizon_gain(
                    response, current_response, args.response_old_weight)
                candidates.append({"recipient": recipient, "donor": donor,
                                   "score": score, "lcb": lcb,
                                   "gain": score - current_score,
                                   "lcb_gain": lcb - current_lcb,
                                   "horizon_gain": horizon_gain,
                                   "allocation": proposal, "block": block,
                                   "response": response})
        if not candidates:
            break
        best = max(candidates, key=lambda item: (item["lcb_gain"],
                                                  -item["recipient"],
                                                  -item["donor"]))
        decision = {k: v for k, v in best.items()
                    if k not in {"block", "response"}}
        decision["round"] = round_id
        decisions.append(decision)
        if (float(best["lcb_gain"]) <= float(args.response_marginal_threshold)
                or float(best["horizon_gain"])
                < float(args.response_min_horizon_gain)
                or float(best["lcb"]) < float(args.response_min_lcb)):
            break
        allocation = dict(best["allocation"])
        current_score = float(best["score"])
        current_lcb = float(best["lcb"])
        current_block = best["block"]
        current_response = best["response"]

    return current_block, {
        "mode": "response_marginal",
        "base_allocation": base_mode,
        "selected_class_rows": allocation,
        "response_score": current_score,
        "response_lcb": current_lcb,
        "response": current_response,
        "decisions": decisions,
        "response_steps": curve,
        "response_evaluations": int(evaluation_count),
        "selection_training_runs": int(
            evaluation_count * len(args.selection_seeds)
            * len(args.selection_batches) * len(curve)),
        "selection_optimizer_steps_total": int(
            evaluation_count * len(args.selection_seeds)
            * len(args.selection_batches) * sum(curve)),
    }


def build_matched_random_transfer_block(
        D: dict, ids: torch.Tensor, k_new: int, stage: int,
        args, device: str) -> tuple[dict, dict]:
    """Build a fixed-budget random one-row transfer control.

    This control starts from the same dispersion allocation as P1-B and moves
    at most one row between two incoming classes.  It is synthetic-only and
    keeps the total new rows, text recipe, X operator, and topology unchanged.
    """
    classes = sorted(int(c) for c in D["stage_classes"][stage])
    base = _base_class_allocation(D, ids, classes, int(k_new), "dispersion")
    proposal = dict(base)
    eligible_donors = [c for c in classes if int(proposal.get(c, 0)) > 1]
    if eligible_donors and len(classes) > 1:
        generator = torch.Generator().manual_seed(
            int(protocol_v3.METHOD_SEED) + 7001 + int(stage))
        donor_index = int(torch.randint(
            len(eligible_donors), (1,), generator=generator).item())
        donor = int(eligible_donors[donor_index])
        recipients = [c for c in classes if c != donor]
        recipient_index = int(torch.randint(
            len(recipients), (1,), generator=generator).item())
        recipient = int(recipients[recipient_index])
        proposal[donor] -= 1
        proposal[recipient] = int(proposal.get(recipient, 0)) + 1
        transfer = {"donor": donor, "recipient": recipient, "rows": 1}
    else:
        transfer = None
    block = build_stage_block_from_class_rows(
        D, ids, proposal, D["encoder"], stage,
        args.text_recipe, args.text_operator, device,
        inner_mode="dispersion",
        diversity_lambda=args.text_diversity_lambda,
        dual_alpha=args.dual_prototype_alpha,
        x_operator=args.x_operator)
    return block, {
        "mode": "matched_random_transfer",
        "base_allocation": base,
        "selected_class_rows": proposal,
        "transfer": transfer,
        "selection_training_runs": 0,
        "selection_optimizer_steps_total": 0,
    }


def select_response_gain_expansion(
        old_memory: dict | None, D: dict, ids: torch.Tensor, k_t: int,
        keep: int, k_new: int, stage: int, args,
        device: str, base_block: dict | None = None) -> tuple[dict, int, int, dict]:
    """Select reserve rows with LCB and full-step terminal confirmation."""
    capacity = max(0, int(len(ids)) - int(k_new))
    if base_block is None:
        base_block = build_stage_block(
            D, ids, k_new, D["encoder"], stage, args.text_recipe,
            args.text_operator, device, args.allocation_mode,
            D["stage_classes"][stage], args.text_diversity_lambda,
            args.dual_prototype_alpha,
            x_operator=args.x_operator)
    if capacity <= 0:
        return base_block, keep, k_new, {"accepted": False, "extra_rows": 0,
                                         "reason": "no_extra_capacity"}

    requested = [0, 2, 4, 8]
    extras = sorted({min(int(v), capacity) for v in requested if int(v) > 0})
    curve = response_steps(int(k_t) + max(extras))
    real_adj = D["real_adj"]
    xg = D["x_device"]
    prompt_emb = D["prompt_emb"]
    evaluation_count = 0

    def score(block: dict, block_keep: int) -> tuple[float, float, dict, dict]:
        nonlocal evaluation_count
        evaluation_count += 1
        candidate = SCA.append_memory(
            old_memory, block, block_keep,
            topology_mode=args.topology_mode, cross_degree=args.cross_degree,
            refresh_mode=args.refresh_mode, refresh_ratio=args.refresh_ratio,
            refresh_alpha=args.refresh_alpha, device=device)
        response = SCA.evaluate_response_curve(
            candidate, stage, args.selection_seeds, args.selection_batches,
            curve, device, xg, real_adj, prompt_emb, D["y"],
            D["selector_dev_mask"])
        return (_response_score(response, args.response_old_weight),
                _response_lcb(response, args.response_old_weight,
                              args.response_gain_risk_z), response, candidate)

    base_score, base_lcb, base_response, base_memory = score(base_block, keep)

    def terminal(memory: dict) -> dict:
        steps = protocol_v3.student_steps(
            len(memory["x"]), override=args.student_steps)
        grid = SCA.evaluate_grid(
            memory, stage, args.selection_seeds, args.selection_batches,
            int(steps), device, xg, real_adj, prompt_emb, D["y"],
            D["selector_dev_mask"])
        old = grid["old_mean_f1"]
        return {"steps": int(steps),
                "cumulative_mean": float(grid["cumulative_f1"]["mean"]),
                "old_mean": None if old is None else float(old["mean"])}

    base_terminal = terminal(base_memory)
    candidates = [{"extra_rows": 0, "score": float(base_score),
                   "lcb": float(base_lcb), "gain": 0.0,
                   "lcb_gain": 0.0, "terminal": base_terminal}]
    expanded_payload = {0: (base_block, base_memory, base_response)}
    for extra in extras:
        extra_block = build_stage_block(
            D, ids, extra, D["encoder"], stage, args.text_recipe,
            args.text_operator, device, args.response_base_allocation,
            D["stage_classes"][stage], args.text_diversity_lambda,
            args.dual_prototype_alpha,
            x_operator=args.x_operator)
        expanded_block = concat_synthetic_blocks(base_block, extra_block)
        score_v, lcb_v, response_v, memory_v = score(expanded_block, keep)
        terminal_v = terminal(memory_v)
        expanded_payload[extra] = (expanded_block, memory_v, response_v)
        candidates.append({
            "extra_rows": int(extra), "score": float(score_v),
            "lcb": float(lcb_v), "gain": float(score_v - base_score),
            "lcb_gain": float(lcb_v - base_lcb),
            "terminal": terminal_v,
        })

    best = max(candidates, key=lambda item: (item["lcb"], -item["extra_rows"]))
    terminal_gain = (float(best["terminal"]["cumulative_mean"])
                     - float(base_terminal["cumulative_mean"]))
    accepted = bool(
        best["extra_rows"] > 0
        and best["lcb_gain"] > float(args.response_gain_threshold)
        and best["lcb"] >= float(args.response_gain_min_lcb)
        and terminal_gain >= -float(args.response_gain_retention_tolerance))
    chosen_extra = int(best["extra_rows"] if accepted else 0)
    chosen_block, _, chosen_response = expanded_payload[chosen_extra]
    return (chosen_block,
            int(keep),
            int(k_new + chosen_extra),
            {"accepted": accepted, "extra_rows": chosen_extra,
             "requested_extra_rows": extras, "gain": float(best["gain"]),
             "lcb_gain": float(best["lcb_gain"]),
             "terminal_gain": terminal_gain,
             "base_score": float(base_score), "base_lcb": float(base_lcb),
             "selected_score": float(best["score"]),
             "selected_lcb": float(best["lcb"]),
             "base_response": base_response,
             "selected_response": chosen_response,
             "base_terminal": base_terminal,
             "selected_terminal": best["terminal"],
             "candidates": candidates,
             "response_steps": curve,
             "response_evaluations": int(evaluation_count),
             "selection_training_runs": int(
                 evaluation_count * len(args.selection_seeds)
                 * len(args.selection_batches) * len(curve)),
             "selection_optimizer_steps_total": int(
                 evaluation_count * len(args.selection_seeds)
                 * len(args.selection_batches) * sum(curve))})


def select_sca(old_memory, block_args, D: dict, stage: int, k_t: int,
               keep: int, k_new: int, args, device: str) -> tuple[dict, dict]:
    reports, memories = [], {}
    curve = response_steps(k_t)
    real_adj = D["real_adj"]
    xg = D["x_device"]
    prompt_emb = D["prompt_emb"]
    for recipe in args.candidate_recipes:
        block = build_stage_block(D, block_args, k_new, D["encoder"], stage,
                                  recipe, args.text_operator, device,
                                  args.allocation_mode, D["stage_classes"][stage],
                                  args.text_diversity_lambda,
                                  args.dual_prototype_alpha,
                                  x_operator=args.x_operator)
        memory = SCA.append_memory(
            old_memory, block, keep, topology_mode=args.topology_mode,
            cross_degree=args.cross_degree, refresh_mode=args.refresh_mode,
            refresh_ratio=args.refresh_ratio, refresh_alpha=args.refresh_alpha,
            device=device)
        memories[recipe] = memory
        response = SCA.evaluate_response_curve(
            memory, stage, args.selection_seeds, args.selection_batches, curve,
            device, xg, real_adj, prompt_emb, D["y"],
            D["selector_dev_mask"])
        reports.append({"candidate": recipe,
                        "response_auc": response["response_auc"],
                        "tail_mean": response["tail_mean"],
                        "tail_std": response["tail_std"],
                        "cumulative_tail_mean": response["cumulative_tail_mean"],
                        "response": response})
    selected = sorted(reports, key=lambda r: (r["cumulative_tail_mean"],
                                               r["response_auc"],
                                               -r["tail_std"], r["candidate"]),
                      reverse=True)[0]
    selection_runs = len(args.candidate_recipes) * len(args.selection_seeds) \
        * len(args.selection_batches) * len(curve)
    return memories[selected["candidate"]], {
        "selected": selected["candidate"],
        "selection_steps": curve,
        "selection_training_runs": int(selection_runs),
        "selection_optimizer_steps_total": int(
            selection_runs / max(1, len(curve)) * sum(curve)),
        "candidates": reports,
    }


BASELINE_SEED_OFFSETS = {
    "random_real": 101,
    "k_center": 211,
    "k_center_subsampled": 521,
    "herding": 307,
    "kmeans_medoid": 401,
    "kmeans_medoid_subsampled": 611,
}


def build_real_selection_block(D: dict, ids: torch.Tensor, k: int,
                               method: str, stage: int,
                               device: str,
                               candidate_cap: int = 0,
                               selection_batch: int = 64,
                               projection_dim: int = 32) -> tuple[dict, dict]:
    """Select real delta rows with the same label-free v3 budget.

    The stage membership comes from the stream protocol, but individual row
    selection never reads labels.  The returned block is consumed by the same
    append operator as synthetic candidates, so only construction changes.
    """
    if method not in BASELINE_SEED_OFFSETS:
        raise ValueError(f"unknown real-node baseline: {method}")
    if k <= 0 or len(ids) == 0:
        raise ValueError(f"invalid selection size {k} for {method}")
    k = min(int(k), len(ids))
    x = D["x"][ids].float()
    text_emb = D["text_emb"][ids].float()
    points = torch.cat((F.normalize(x, dim=1), text_emb), dim=1).to(device)
    n = len(ids)
    seed = protocol_v3.METHOD_SEED + stage * 1009 + BASELINE_SEED_OFFSETS[method]

    approximate = method.endswith("_subsampled")
    if approximate and int(candidate_cap) <= 0:
        raise ValueError(f"{method} requires --selection-candidate-cap > 0")
    candidate_cap = int(candidate_cap)
    candidate_rows = n
    candidate_index = torch.arange(n, dtype=torch.long)
    if approximate and n > candidate_cap:
        candidate_cap = max(candidate_cap, k)
        generator = torch.Generator().manual_seed(seed + 17)
        candidate_index = torch.randperm(n, generator=generator)[:candidate_cap]
        candidate_rows = int(len(candidate_index))
    candidate_device = candidate_index.to(device)
    work_points = points.index_select(0, candidate_device)

    def batched_fps(points_to_select: torch.Tensor, count: int,
                    seed_value: int) -> torch.Tensor:
        """Scalable approximate FPS used only by explicit subsampled arms.

        A low-dimensional fixed random projection and batched farthest-point
        updates avoid the O(n*k) Python loop of exact FPS on large graphs.
        The selected rows remain real source rows and the approximation is
        recorded in the selection audit.
        """
        dim = int(points_to_select.shape[1])
        p_dim = max(4, min(int(projection_dim), dim))
        generator = torch.Generator(device=device).manual_seed(int(seed_value) + 29)
        projection = torch.randn((dim, p_dim), generator=generator,
                                 device=device, dtype=points_to_select.dtype)
        projection = F.normalize(projection, dim=0)
        projected = points_to_select @ projection
        take_batch = max(1, int(selection_batch))
        selected_parts = []
        dmin = (projected - projected[0]).pow(2).sum(1)
        selected_parts.append(torch.zeros(1, dtype=torch.long, device=device))
        dmin[0] = -1.0
        while sum(int(v.numel()) for v in selected_parts) < count:
            remaining = count - sum(int(v.numel()) for v in selected_parts)
            take = min(take_batch, remaining)
            local = torch.topk(dmin, take, sorted=False).indices
            selected_parts.append(local)
            distances = torch.cdist(projected, projected[local]).square().amin(1)
            dmin = torch.minimum(dmin, distances)
            dmin[local] = -1.0
        return torch.cat(selected_parts)[:count].cpu()

    if k == n or method == "random_real":
        generator = torch.Generator().manual_seed(seed)
        selected = torch.randperm(n, generator=generator)[:k]
    elif method == "k_center":
        if k > len(work_points):
            raise ValueError(f"candidate set {len(work_points)} is smaller than k={k}")
        selected_list = [0]
        dmin = (work_points - work_points[0]).pow(2).sum(1)
        for _ in range(1, k):
            dmin[selected_list[-1]] = -1.0
            nxt = int(torch.argmax(dmin).item())
            selected_list.append(nxt)
            dmin = torch.minimum(dmin, (work_points - work_points[nxt]).pow(2).sum(1))
        selected = candidate_index[torch.tensor(selected_list, dtype=torch.long)]
    elif method == "k_center_subsampled":
        if k > len(work_points):
            raise ValueError(f"candidate set {len(work_points)} is smaller than k={k}")
        local = batched_fps(work_points, k, seed)
        selected = candidate_index[local]
    elif method == "herding":
        target = points.mean(0, keepdim=True)
        selected_list, running = [], torch.zeros_like(target)
        available = torch.ones(n, dtype=torch.bool, device=device)
        for step in range(k):
            candidate_mean = (running + points) / (step + 1)
            distance = (candidate_mean - target).pow(2).sum(1)
            distance[~available] = float("inf")
            nxt = int(torch.argmin(distance).item())
            selected_list.append(nxt)
            running = running + points[nxt:nxt + 1]
            available[nxt] = False
        selected = torch.tensor(selected_list, dtype=torch.long)
    elif method == "kmeans_medoid":
        if k > len(work_points):
            raise ValueError(f"candidate set {len(work_points)} is smaller than k={k}")
        # Deterministic max-min Lloyd on either all candidates or the sealed
        # candidate subset; the selected medoids are always real source rows.
        assign = SCA.cluster_assign(work_points.cpu(),
                                    torch.ones(len(work_points)), k,
                                    device=device)
        selected_list = []
        for cluster in range(k):
            members = torch.where(assign == cluster)[0]
            if len(members) == 0:
                continue
            member_points = work_points[members.to(device)]
            center = member_points.mean(0, keepdim=True)
            medoid = members[(member_points - center).pow(2).sum(1).argmin()]
            selected_list.append(int(candidate_index[int(medoid)].item()))
        selected = torch.tensor(selected_list, dtype=torch.long)
    elif method == "kmeans_medoid_subsampled":
        if k > len(work_points):
            raise ValueError(f"candidate set {len(work_points)} is smaller than k={k}")
        # The scalable control uses batched FPS centers as real medoids.  It
        # preserves the real-node/memory budget contract while making clear
        # that it is not an exact Lloyd run on the full source pool.
        local = batched_fps(work_points, k, seed)
        selected = candidate_index[local]
    else:
        raise ValueError(f"unknown real-node baseline: {method}")

    chosen_ids = ids[selected]
    block = {
        "x": D["x"][chosen_ids].float(),
        "text_emb": D["text_emb"][chosen_ids].float(),
        "texts": [D["texts"][int(i)] for i in chosen_ids],
        "mass": torch.ones(len(chosen_ids)),
        "depth": torch.zeros(len(chosen_ids), dtype=torch.long),
        "stage": torch.full((len(chosen_ids),), stage, dtype=torch.long),
    }
    return block, {"method": method, "selected_rows": int(len(chosen_ids)),
                   "selected_source_ids": [int(v) for v in chosen_ids.tolist()],
                   "selection_seed": int(seed),
                   "approximate": bool(approximate),
                   "candidate_cap": (int(candidate_cap) if approximate else None),
                   "candidate_rows": int(candidate_rows),
                   "selection_batch": (int(selection_batch) if approximate else None),
                   "projection_dim": (int(projection_dim) if approximate else None),
                   "approximation": ("batched_fps_random_projection"
                                     if approximate else "exact")}


def condense_static_gpu(x: torch.Tensor, text_emb: torch.Tensor,
                        texts: list[str], k: int, encoder,
                        device: str, stage: int = 0) -> dict:
    """Full-history re-condensation with the existing GPU cluster backend.

    Unlike append arms, this constructor receives the complete cumulative raw
    pool at every stage and may rewrite every synthetic row.  The only change
    from the legacy full-history control is replacing the CPU MiniBatchKMeans
    prediction pass with the already audited chunked GPU assignment used by
    the large-graph C0 constructor.
    """
    if k <= 0 or len(x) < k:
        raise ValueError(f"invalid full-history condensation size k={k} n={len(x)}")
    joint = torch.cat((F.normalize(x.float(), dim=1), text_emb.float()), dim=1)
    assign = SCA.cluster_assign(
        joint, torch.ones(len(x), dtype=torch.float32), int(k), device=device
    )
    syn_x, fused, mass = [], [], []
    for j in range(int(assign.max()) + 1):
        members = torch.where(assign == j)[0]
        if len(members) == 0:
            continue
        syn_x.append(x[members].float().mean(0))
        fused.append(G.fuse_new_only(joint, members, texts))
        mass.append(float(len(members)))
    syn_x = torch.stack(syn_x)
    emb = encoder.encode(fused)
    topology_builder = getattr(G, "text_topology", None)
    if topology_builder is None:
        # The current Student helper owns the scalable text-kNN builder;
        # older frozen Gate modules exposed the same operation on G.
        topology_builder = lambda value, device=None: SCA.text_knn_edges(value)
    else:
        # Frozen Gate versions disagree on whether the optional device
        # argument is accepted.  Inspect once so large static runs do not
        # build the expensive text graph twice just to probe the signature.
        try:
            accepts_device = "device" in inspect.signature(
                topology_builder).parameters
        except (TypeError, ValueError):
            accepts_device = False
        if accepts_device:
            topology_builder = lambda value, device=None: G.text_topology(
                value, device=device)
        else:
            topology_builder = lambda value, device=None: G.text_topology(value)
    return {
        "x": syn_x,
        "text_emb": emb,
        "texts": fused,
        "mass": torch.tensor(mass),
        "depth": torch.zeros(len(syn_x), dtype=torch.long),
        "stage": torch.full((len(syn_x),), int(stage), dtype=torch.long),
        "edge_index": topology_builder(emb, device=device),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(DATA_CFG), required=True)
    ap.add_argument("--data", default=None)
    ap.add_argument("--minilm", default="/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip")
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--rate", type=float, default=0.05)
    ap.add_argument("--budget-mode", choices=["fixed-rate", "fixed-total", "hybrid", "adaptive-rate"], default="fixed-rate")
    ap.add_argument("--total-budget", type=int, default=None)
    ap.add_argument("--min-increment", type=int, default=0)
    ap.add_argument("--adaptive-strength", type=float, default=0.75)
    ap.add_argument("--min-rate", type=float, default=0.01)
    ap.add_argument("--max-rate", type=float, default=0.10)
    ap.add_argument("--n-stages", type=int, default=3)
    ap.add_argument("--split-mode", choices=["canonical", "round_robin", "frequency", "base_incremental"], default="canonical")
    ap.add_argument("--base-fraction", type=float, default=0.4)
    ap.add_argument("--report-dev-mod", type=int, default=protocol_v3.REPORT_DEV_MOD)
    ap.add_argument("--allocation-mode", choices=["legacy_15", "equal", "linear", "sqrt", "dispersion", "subcluster_complexity"], default="legacy_15")
    ap.add_argument("--legacy-floor-quota", action="store_true",
                    help="reproduce the historical legacy_15 append quota; "
                         "otherwise keep/new rows are matched across allocation modes")
    ap.add_argument("--arms", nargs="+",
                    default=["append_fixed", "recluster", "full_history", "sca"])
    ap.add_argument("--candidate-recipes", nargs="+", default=["joint_g5_c320"])
    ap.add_argument("--selection-seeds", type=int, nargs="+", default=[200, 201, 202])
    ap.add_argument("--selection-batches", type=int, nargs="+", default=[8, 16, 32])
    ap.add_argument("--student-seeds", type=int, nargs="+",
                    default=list(protocol_v3.STUDENT_SEEDS))
    ap.add_argument("--student-batch", type=int, default=protocol_v3.STUDENT.batch)
    ap.add_argument("--student-steps", type=int, default=None)
    ap.add_argument("--topology-mode", choices=["frozen_old", "global"], default="frozen_old")
    ap.add_argument("--topology-reference", default=None,
                    help="directory containing reference arm_s{stage}_memory.pt files")
    ap.add_argument("--cross-degree", type=int, default=1)
    ap.add_argument("--selection-candidate-cap", type=int, default=0,
                    help="candidate rows for the explicit *_subsampled coreset arms")
    ap.add_argument("--selection-batch", type=int, default=64,
                    help="batched FPS width for the explicit *_subsampled arms")
    ap.add_argument("--selection-projection-dim", type=int, default=32,
                    help="random projection dimension for scalable coreset arms")
    ap.add_argument("--text-operator", choices=["concat", "diverse_concat",
                                                 "dual_proto", "mean_pool",
                                                 "weighted_pool", "set_pool",
                                                 "no_reencode", "no_fuse", "no_text"],
                    default="concat")
    ap.add_argument("--text-recipe", choices=sorted(SCA.RECIPE_SPECS),
                    default="joint_g5_c320")
    ap.add_argument("--text-diversity-lambda", type=float, default=0.5)
    ap.add_argument("--dual-prototype-alpha", type=float, default=0.5)
    ap.add_argument("--x-operator", choices=list(SCA.X_OPERATORS),
                    default="mean",
                    help="synthetic feature operator; P1-X varies only X")
    ap.add_argument("--refresh-mode", choices=["frozen_old", "rescore_only",
                                                "bounded_refresh"],
                    default="frozen_old")
    ap.add_argument("--refresh-ratio", type=float, default=0.10)
    ap.add_argument("--refresh-alpha", type=float, default=0.25)
    ap.add_argument("--dynamic-rows", action="store_true")
    ap.add_argument("--response-marginal", action="store_true",
                    help="use selector-dev student response to transfer fixed rows between classes")
    ap.add_argument("--response-base-allocation",
                    choices=["dispersion", "sqrt", "subcluster_complexity"],
                    default="dispersion")
    ap.add_argument("--response-marginal-rounds", type=int, default=1)
    ap.add_argument("--response-marginal-threshold", type=float, default=0.0)
    ap.add_argument("--response-old-weight", type=float, default=0.25)
    ap.add_argument("--response-risk-z", type=float, default=1.0)
    ap.add_argument("--response-min-lcb", type=float, default=-1e9)
    ap.add_argument("--response-min-horizon-gain", type=float, default=-1e9)
    ap.add_argument("--response-gain-extra", action="store_true",
                    help="accept extra rows only when selector-dev response improves")
    ap.add_argument("--response-gain-max-extra", type=int, default=8)
    ap.add_argument("--response-gain-threshold", type=float, default=0.0)
    ap.add_argument("--response-gain-risk-z", type=float, default=1.0)
    ap.add_argument("--response-gain-min-lcb", type=float, default=0.0)
    ap.add_argument("--response-gain-retention-tolerance", type=float, default=0.0)
    ap.add_argument("--p1b-guarded", action="store_true",
                    help="run the P1-B fixed-budget guarded response protocol")
    ap.add_argument("--v4-mainline", action="store_true",
                    help="use the auditable v4 fixed-rate response-calibrated preset")
    ap.add_argument("--v4-enable-reserve", action="store_true",
                    help="enable conservative v4 reserve-budget confirmation")
    args = ap.parse_args()

    if args.p1b_guarded:
        # P1-B is deliberately separate from the frozen v4 response arm:
        # response can transfer at most one fixed-budget row, but cannot grow
        # the memory or change the student training protocol.
        args.arms = ["append_fixed", "response_guarded",
                     "matched_random_transfer"]
        args.budget_mode = "fixed-rate"
        args.allocation_mode = "dispersion"
        args.response_base_allocation = "dispersion"
        args.response_marginal = True
        args.response_marginal_rounds = 1
        args.response_marginal_threshold = max(
            0.5, float(args.response_marginal_threshold))
        args.response_old_weight = max(0.25, float(args.response_old_weight))
        args.response_risk_z = max(1.0, float(args.response_risk_z))
        args.response_min_lcb = max(0.0, float(args.response_min_lcb))
        args.response_min_horizon_gain = max(
            0.0, float(args.response_min_horizon_gain))
        # These seeds are independent from both the report seeds (100--105)
        # and the P3-A selector seeds (300--302).
        args.selection_seeds = [401, 402]
        args.selection_batches = [16]

    if args.v4_mainline:
        # Keep explicit non-default overrides intact while making the v4
        # preset reproducible from one switch.
        if args.arms == ["append_fixed", "recluster", "full_history", "sca"]:
            args.arms = ["append_fixed"]
        if args.allocation_mode == "legacy_15":
            args.allocation_mode = "dispersion"
        if args.selection_seeds == [200, 201, 202]:
            args.selection_seeds = [300, 301, 302]
        args.response_marginal = True
        args.response_base_allocation = "dispersion"
        args.response_marginal_rounds = max(2, int(args.response_marginal_rounds))
        args.response_marginal_threshold = max(
            0.0, float(args.response_marginal_threshold))
        args.response_risk_z = max(0.0, float(args.response_risk_z))
        args.response_min_lcb = max(0.0, float(args.response_min_lcb))
        args.response_min_horizon_gain = max(
            0.0, float(args.response_min_horizon_gain))
        args.response_gain_extra = bool(args.v4_enable_reserve)
        if args.v4_enable_reserve:
            args.response_gain_threshold = max(
                0.0, float(args.response_gain_threshold))
            args.response_gain_min_lcb = max(
                0.0, float(args.response_gain_min_lcb))

    if args.response_marginal and args.response_gain_extra and not args.v4_mainline:
        raise ValueError("response-marginal and response-gain-extra are separate ablations")

    if args.data is None:
        args.data = DATA_CFG[args.dataset]["path"]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "results.json").exists():
        raise FileExistsError("existing runs are immutable")
    unknown = sorted(set(args.candidate_recipes) - set(SCA.RECIPE_SPECS))
    if unknown:
        raise ValueError(f"unknown candidate recipes: {unknown}")

    D = load_unified(args.dataset, args.data, args.minilm, args.device,
                     n_stages=args.n_stages, split_mode=args.split_mode,
                     base_fraction=args.base_fraction,
                     report_dev_mod=args.report_dev_mod)
    pool_ids = torch.where(D["pool_mask"])[0]
    text_cache = out / "text_emb_pool_only.pt"
    text_embedding_started = time.perf_counter()
    text_cache_hit = text_cache.exists()
    if text_cache.exists():
        text_emb = torch.load(text_cache, map_location="cpu", weights_only=False)
        if tuple(text_emb.shape) != (len(D["x"]), 384):
            raise ValueError(f"invalid text embedding cache shape: {tuple(text_emb.shape)}")
        excluded = ~D["pool_mask"]
        if excluded.any() and float(text_emb[excluded].abs().max()) != 0.0:
            raise ValueError("pool-only text cache contains excluded-source embeddings")
        print(f"loaded text embeddings: {text_cache}", flush=True)
    else:
        # Construction text is encoded only for the sealed source pool.  The
        # report-dev and test text fields remain present in the raw container
        # for provenance, but their embeddings never enter construction.
        pool_texts = [D["texts"][int(i)] for i in pool_ids]
        pool_emb = D["encoder"].encode(pool_texts)
        text_emb = torch.zeros((len(D["x"]), 384), dtype=pool_emb.dtype)
        text_emb[pool_ids] = pool_emb
        torch.save(text_emb, text_cache)
        print(f"encoded and cached {len(pool_emb)} pool text embeddings", flush=True)
    text_embedding_seconds = float(time.perf_counter() - text_embedding_started)
    D["text_emb"] = text_emb
    D["prompt_emb"] = D["encoder"].encode(D["prompts"])
    D["real_adj"] = G.norm_adj(D["edge_index"], len(D["x"]), args.device)
    D["x_device"] = D["x"].to(args.device)
    stage_train = []
    for classes in D["stage_classes"]:
        stage_train.append(torch.sort(torch.where(
            torch.isin(D["y"], torch.tensor(classes))
            & D["pool_mask"])[0]).values)
    budgets, stage_utilities, effective_rates = [], [], []
    prior_ids = None
    for s in range(len(stage_train)):
        cumulative = int(sum(len(v) for v in stage_train[:s + 1]))
        previous = budgets[-1] if budgets else 0
        if args.budget_mode == "adaptive-rate":
            utility = information_utility(D, stage_train[s], prior_ids)
            budget, effective_rate = protocol_v3.adaptive_rate_budget(
                cumulative, previous, utility, base_rate=args.rate,
                strength=args.adaptive_strength, min_rate=args.min_rate,
                max_rate=args.max_rate, min_increment=args.min_increment)
        else:
            utility, effective_rate = None, args.rate
            mode = args.budget_mode.replace("-", "_")
            budget = protocol_v3.growing_budget(
                cumulative, rate=args.rate, mode=mode,
                total_budget=args.total_budget, previous_budget=previous,
                min_increment=args.min_increment)
        budgets.append(int(budget))
        stage_utilities.append(utility)
        effective_rates.append(float(effective_rate))
        prior_ids = stage_train[s] if prior_ids is None else torch.cat((prior_ids, stage_train[s]))
    print(json.dumps({"dataset": args.dataset,
                      "stage_train_counts": [len(v) for v in stage_train],
                      "budgets": budgets,
                      "stage_classes": D["stage_classes"],
                      "stage_utilities": stage_utilities,
                      "effective_rates": effective_rates}), flush=True)

    all_results = []
    for arm in args.arms:
        memory = None
        arm_rows = []
        for stage, ids in enumerate(stage_train):
            k_t = budgets[stage]
            stage_build_started = time.perf_counter()
            difficulty = None
            if stage == 0:
                keep, k_new = 0, k_t
            else:
                if args.budget_mode == "fixed-total":
                    keep, k_new = protocol_v3.replacement_quota(
                        k_t, len(memory["x"]), len(ids),
                        sum(len(v) for v in stage_train[:stage + 1]),
                        len(D["stage_classes"][stage]))
                elif args.dynamic_rows:
                    local = torch.searchsorted(pool_ids, ids)
                    de = D["text_emb"][ids]
                    difficulty = SCA.delta_difficulty(D["x"][ids], de)
                    keep, k_new = protocol_v3.dynamic_append_quota(
                        k_t, len(memory["x"]), difficulty,
                        len(D["stage_classes"][stage]))
                elif args.legacy_floor_quota and args.allocation_mode == "legacy_15":
                    base_new = max(0, int(k_t) - len(memory["x"]))
                    floor_new = 15 * len(D["stage_classes"][stage])
                    if int(k_t) >= floor_new:
                        keep, k_new = protocol_v3.append_quota(
                            k_t, len(memory["x"]),
                            len(D["stage_classes"][stage]))
                    else:
                        keep, k_new = min(len(memory["x"]), int(k_t)), base_new
                else:
                    k_new = max(0, int(k_t) - len(memory["x"]))
                    keep = min(len(memory["x"]), int(k_t))

            selection = None
            allocation = None
            response_gain_audit = None
            if (k_new > 0 and arm not in BASELINE_SEED_OFFSETS
                    and arm not in {"full_history", "full_history_gpu"}):
                if args.allocation_mode == "subcluster_complexity":
                    allocation = {"mode": "subcluster_complexity"}
                else:
                    allocation = allocate_rows_by_class(
                        D, ids, D["stage_classes"][stage], k_new,
                        args.allocation_mode)
            if k_new <= 0 and arm not in {"full_history", "full_history_gpu"}:
                if memory is None:
                    raise ValueError("stage-0 budget must allocate at least one row")
                memory = G.shrink_by_mass(memory, min(len(memory["x"]), int(k_t)))
                selection = {"skipped": True, "reason": "budget_not_growing"}
            elif arm in {"full_history", "full_history_gpu"}:
                source = torch.cat(stage_train[:stage + 1])
                if arm == "full_history_gpu":
                    memory = condense_static_gpu(
                        D["x"][source], D["text_emb"][source],
                        [D["texts"][int(i)] for i in source], k_t,
                        D["encoder"], args.device, stage=stage)
                else:
                    memory = G.condense_static(
                        D["x"][source], D["text_emb"][source],
                        [D["texts"][int(i)] for i in source], k_t,
                        D["encoder"], device=args.device)
            elif arm == "sca":
                memory, selection = select_sca(
                    memory, ids, D, stage, k_t, keep, k_new, args, args.device)
            elif arm in BASELINE_SEED_OFFSETS:
                block, selection = build_real_selection_block(
                    D, ids, k_new, arm, stage, args.device,
                    candidate_cap=args.selection_candidate_cap,
                    selection_batch=args.selection_batch,
                    projection_dim=args.selection_projection_dim)
                memory = SCA.append_memory(
                    memory, block, keep, topology_mode=args.topology_mode,
                    cross_degree=args.cross_degree, refresh_mode=args.refresh_mode,
                    refresh_ratio=args.refresh_ratio, refresh_alpha=args.refresh_alpha,
                    device=args.device)
            elif stage == 0 or arm in {"append_fixed", "response_guarded",
                                       "matched_random_transfer"}:
                response_enabled = bool(
                    args.response_marginal
                    and (arm == "response_guarded"
                         or (arm == "append_fixed"
                             and not args.p1b_guarded)))
                if response_enabled:
                    block, selection = build_response_marginal_block(
                        memory, D, ids, k_new, keep, stage, args, args.device)
                    allocation = selection["selected_class_rows"]
                    if args.response_gain_extra:
                        block, keep, k_new, response_gain_audit = \
                            select_response_gain_expansion(
                                memory, D, ids, k_t, keep, k_new, stage,
                                args, args.device, base_block=block)
                        selection = {
                            "mode": "response_marginal_plus_reserve",
                            "marginal": selection,
                            "response_gain": response_gain_audit,
                        }
                elif arm == "matched_random_transfer":
                    block, selection = build_matched_random_transfer_block(
                        D, ids, k_new, stage, args, args.device)
                    allocation = selection["selected_class_rows"]
                elif args.response_gain_extra:
                    block, keep, k_new, response_gain_audit = \
                        select_response_gain_expansion(
                            memory, D, ids, k_t, keep, k_new, stage,
                            args, args.device)
                    selection = {"mode": "response_gain",
                                 **response_gain_audit}
                else:
                    block = build_stage_block(
                        D, ids, k_new, D["encoder"], stage,
                        args.text_recipe, args.text_operator,
                        args.device, args.allocation_mode,
                        D["stage_classes"][stage],
                        args.text_diversity_lambda,
                        args.dual_prototype_alpha,
                        x_operator=args.x_operator)
                    if args.allocation_mode == "subcluster_complexity":
                        allocation = block.get("allocation")
                memory = SCA.append_memory(
                    memory, block, keep, topology_mode=args.topology_mode,
                    cross_degree=args.cross_degree, refresh_mode=args.refresh_mode,
                    refresh_ratio=args.refresh_ratio, refresh_alpha=args.refresh_alpha,
                    device=args.device)
            elif arm == "recluster":
                dx = D["x"][ids]
                de = D["text_emb"][ids]
                memory, audit = G.continual_recluster(
                    memory, dx, de, [D["texts"][int(i)] for i in ids],
                    k_t, stage, D["encoder"], device=args.device)
                selection = {"recluster_audit": audit}
            else:
                raise ValueError(f"unknown arm: {arm}")

            topology_reference_used = False
            if args.topology_reference is not None:
                reference_path = Path(args.topology_reference) / \
                    f"{arm}_s{stage}_memory.pt"
                if not reference_path.exists():
                    raise FileNotFoundError(
                        f"missing topology reference: {reference_path}")
                reference = torch.load(reference_path, map_location="cpu",
                                       weights_only=False)
                for key in ("x", "mass", "depth", "stage"):
                    if key not in reference or key not in memory:
                        raise ValueError(f"topology reference lacks {key}")
                    if reference[key].shape != memory[key].shape or not torch.allclose(
                            reference[key].float(), memory[key].float(),
                            atol=1e-6, rtol=0):
                        raise ValueError(
                            f"topology reference mismatch at stage {stage}: {key}")
                memory["edge_index"] = reference["edge_index"].clone().long()
                topology_reference_used = True

            stage_build_seconds = float(time.perf_counter() - stage_build_started)

            torch.save(memory, out / f"{arm}_s{stage}_memory.pt")
            metrics = run_students(memory, stage, D, args, args.device)
            metrics["text_audit"] = SCA.text_audit(
                memory["texts"], D["encoder"].tokenizer,
                operator=args.text_operator, recipe=args.text_recipe)
            selector_runs, selector_steps = _selection_cost(selection)
            actual_new_rows = (int(len(memory["x"]) - keep)
                               if arm not in {"full_history", "full_history_gpu",
                                              "recluster"}
                               else int(len(memory["x"])))
            row = {"arm": arm, "stage": stage, "keep_old_rows": int(keep),
                   "new_rows": int(k_new), "selection": selection,
                   "allocation_mode": args.allocation_mode,
                   "allocated_new_rows_by_class": allocation,
                   "legacy_15_floor_relaxed": bool(
                       args.allocation_mode == "legacy_15"
                       and int(k_new) < 15 * len(D["stage_classes"][stage])),
                   "actual_new_rows": actual_new_rows,
                   "delta_difficulty": difficulty,
                   "effective_budget": int(len(memory["x"])),
                   "response_gain_extra_rows": int(
                       (response_gain_audit or {}).get("extra_rows", 0)),
                   "selection_training_runs": int(selector_runs),
                   "selection_optimizer_steps_total": int(selector_steps),
                   "stage_build_seconds": stage_build_seconds,
                   "topology_reference_used": topology_reference_used,
                   "refresh_metadata": memory.get("refresh_metadata"),
                   **metrics}
            arm_rows.append(row)
            print(json.dumps({"arm": arm, "stage": stage,
                              "cum_dev_f1_mean": row["cum_dev_f1_mean"],
                              "current_f1_mean": row["current_f1_mean"]}),
                  flush=True)
        all_results.extend(arm_rows)

    manifest = {
        **protocol_v3.config_dict(dataset=args.dataset, data=args.data,
                                  rate=args.rate,
                                  student_batch=args.student_batch,
                                  student_steps=(args.student_steps
                                                 if args.student_steps is not None
                                                 else "protocol-scaled"),
                                  student_seeds=args.student_seeds),
        "runner": "run_unified_v3_continual.py",
        "arms": args.arms,
        "candidate_recipes": args.candidate_recipes,
        "selection_seeds": args.selection_seeds,
        "selection_batches": args.selection_batches,
        "topology_mode": args.topology_mode,
        "topology_reference": args.topology_reference,
        "cross_degree": args.cross_degree,
        "selection_candidate_cap": args.selection_candidate_cap,
        "selection_batch": args.selection_batch,
        "selection_projection_dim": args.selection_projection_dim,
        "text_operator": args.text_operator,
        "text_recipe": args.text_recipe,
        "x_operator": args.x_operator,
        "text_diversity_lambda": args.text_diversity_lambda,
        "dual_prototype_alpha": args.dual_prototype_alpha,
        "text_budget_policy": (
            "legacy_v4_chars_per_snippet_with_join_separators; tokenizer_audit"
            if args.text_operator == "concat"
            else "legacy_character_experiment; use_run_p2_token_controls_for_fair_comparison"),
        "refresh_mode": args.refresh_mode,
        "refresh_ratio": args.refresh_ratio,
        "refresh_alpha": args.refresh_alpha,
        "dynamic_rows": args.dynamic_rows,
        "response_marginal": args.response_marginal,
        "response_base_allocation": args.response_base_allocation,
        "response_marginal_rounds": args.response_marginal_rounds,
        "response_marginal_threshold": args.response_marginal_threshold,
        "response_old_weight": args.response_old_weight,
        "response_risk_z": args.response_risk_z,
        "response_min_lcb": args.response_min_lcb,
        "response_min_horizon_gain": args.response_min_horizon_gain,
        "response_gain_extra": args.response_gain_extra,
        "response_gain_max_extra": args.response_gain_max_extra,
        "response_gain_threshold": args.response_gain_threshold,
        "response_gain_risk_z": args.response_gain_risk_z,
        "response_gain_min_lcb": args.response_gain_min_lcb,
        "response_gain_retention_tolerance": args.response_gain_retention_tolerance,
        "p1b_guarded": args.p1b_guarded,
        "p1b_protocol": ({
            "name": "fixed-budget-conservative-response-transfer-v1",
            "base_allocation": "dispersion",
            "max_transfer_rows_per_stage": 1,
            "total_budget_change": 0,
            "selection_seeds": [401, 402],
            "selection_batches": [16],
            "selection_horizons": "canonical response_steps(total_rows)",
            "threshold_lcb_gain": float(args.response_marginal_threshold),
            "minimum_horizon_gain": float(args.response_min_horizon_gain),
            "fallback": "dispersion allocation",
            "random_control": "matched_random_transfer",
        } if args.p1b_guarded else None),
        "v4_mainline": args.v4_mainline,
        "v4_enable_reserve": args.v4_enable_reserve,
        "budget_mode": args.budget_mode,
        "total_budget": args.total_budget,
        "min_increment": args.min_increment,
        "adaptive_strength": args.adaptive_strength,
        "min_rate": args.min_rate,
        "max_rate": args.max_rate,
        "n_stages": args.n_stages,
        "split_mode": args.split_mode,
        "base_fraction": args.base_fraction,
        "report_dev_mod": args.report_dev_mod,
        "allocation_mode": args.allocation_mode,
        "legacy_floor_quota": args.legacy_floor_quota,
        "n": int(len(D["x"])),
        "pool": int(pool_ids.numel()),
        "stage_classes": D["stage_classes"],
        "stage_split_source": D["stage_split_source"],
        "enumeration_protocol": protocol_v3.ENUMERATION_PROTOCOL,
        "class_order_policy": "sorted-within-stage; stage-order-preserved",
        "stage_ids_sha256": sha256_json([v.tolist() for v in stage_train]),
        "seed_policy": {
            "student": list(args.student_seeds),
            "method": protocol_v3.METHOD_SEED,
            "selection": list(args.selection_seeds),
        },
        "stage_train_counts": [int(len(v)) for v in stage_train],
        "budgets": budgets,
        "stage_utilities": stage_utilities,
        "effective_rates": effective_rates,
        "text_cache_hit": bool(text_cache_hit),
        "text_embedding_seconds": text_embedding_seconds,
        "source_hash_sha256": sha256_file(args.data),
        "stage_manifest_sha256": sha256_json({
            "stage_classes": D["stage_classes"],
            "stage_train_counts": [int(len(v)) for v in stage_train],
            "budgets": budgets,
        }),
        "mask_counts": {
            "selector_dev": int(D["selector_dev_mask"].sum()),
            "report_dev": int(D["report_dev_mask"].sum()),
            "test": int(D["test_mask"].sum()),
            "pool": int(D["pool_mask"].sum()),
        },
        "synthetic_only_main_arms": True,
        "embedding_cache_policy": "pool_only; selector_dev/report_dev/test embeddings excluded",
        "prompt_sha256": sha256_json(D["prompts"]),
        "code_sha256": {
            "protocol_v3.py": sha256_file(protocol_v3.__file__),
            "run_unified_v3_continual.py": sha256_file(__file__),
            "run_student_compatible_append.py": sha256_file(SCA.__file__),
            "run_gate1_continual.py": sha256_file(G.__file__),
        },
        "test_accessed": False,
        "test_source_sealed": True,
        "total_student_training_runs": int(sum(
            r.get("student_training_runs", 0) for r in all_results)),
        "total_student_optimizer_steps": int(sum(
            r.get("student_optimizer_steps_total", 0) for r in all_results)),
        "total_selector_training_runs": int(sum(
            r.get("selection_training_runs", 0)
            for r in all_results)),
        "total_selector_optimizer_steps": int(sum(
            r.get("selection_optimizer_steps_total", 0)
            for r in all_results)),
        "results": all_results,
    }
    json.dump(manifest, (out / "results.json").open("w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
