"""Unified small-graph comparison for the v4 continual TAG protocol.

This runner separates three questions that were previously mixed together:

1. How well can a student do with the full legal raw source pool (offline
   upper reference, retrained from scratch at every stage)?
2. How well do synthetic append and label-free real-node coreset baselines do
   when the student is retrained from the current memory?
3. How much can a stage-0 student recover by fine-tuning only on the newly
   arrived raw delta?

All construction sources exclude selector-dev, report-dev, and test. The
comparison reports report-dev only; test remains sealed. Published TAG methods
whose objective/task is supervised or non-continual are not reimplemented here
as if they were same-protocol competitors. The included external arms are
label-free coreset/condensation controls that can honestly be run under the
same GraphCLIP continual protocol.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

import protocol_v3
import run_unified_v3_continual as U
import run_student_compatible_append as SCA
import run_gate1_continual as G


REAL_ARMS = {"random_real", "k_center", "herding", "kmeans_medoid"}
SYN_ARMS = {"append_fixed", "sca", "recluster", "full_history_syn"}
ALL_ARMS = sorted(REAL_ARMS | SYN_ARMS)


def induced_edges(edge_index: torch.Tensor, ids: torch.Tensor,
                  total_nodes: int) -> torch.Tensor:
    """Return the raw graph induced by ids with compact local indices."""
    ids = ids.long().cpu()
    remap = torch.full((int(total_nodes),), -1, dtype=torch.long)
    remap[ids] = torch.arange(len(ids), dtype=torch.long)
    edges = edge_index.long().cpu()
    valid = (remap[edges[0]] >= 0) & (remap[edges[1]] >= 0)
    return torch.stack((remap[edges[0, valid]], remap[edges[1, valid]]))


def raw_support(D: dict, ids: torch.Tensor, stage: int) -> dict:
    """Build a raw training support using only legal pool nodes."""
    ids = ids.long().cpu()
    return {
        "x": D["x"][ids].float(),
        "text_emb": D["text_emb"][ids].float(),
        "texts": [D["texts"][int(i)] for i in ids],
        "edge_index": induced_edges(D["edge_index"], ids, len(D["x"])),
        "weights": torch.ones(len(ids)),
        "mass": torch.ones(len(ids)),
        "stage": torch.full((len(ids),), int(stage), dtype=torch.long),
        "depth": torch.zeros(len(ids), dtype=torch.long),
    }


def make_sca_args() -> SimpleNamespace:
    """Use the audited v4 selector settings for the comparison arm."""
    return SimpleNamespace(
        candidate_recipes=["joint_g5_c320"],
        text_operator="concat",
        allocation_mode="dispersion",
        text_diversity_lambda=0.5,
        dual_prototype_alpha=0.5,
        selection_seeds=[300, 301, 302],
        selection_batches=[8, 16, 32],
        topology_mode="frozen_old",
        cross_degree=1,
        refresh_mode="frozen_old",
        refresh_ratio=0.10,
        refresh_alpha=0.25,
    )


def _selected_ids_for_block(D: dict, ids: torch.Tensor, block: dict) -> torch.Tensor:
    """Recover selected source ids for a real-node block deterministically."""
    remaining = list(ids.long().cpu().tolist())
    selected = []
    for row, text in zip(block["x"], block["texts"]):
        candidates = [i for i in remaining
                      if D["texts"][i] == text
                      and torch.equal(D["x"][i].float(), row.float())]
        if not candidates:
            raise RuntimeError("real-node baseline block lost source provenance")
        chosen = candidates[0]
        selected.append(chosen)
        remaining.remove(chosen)
    return torch.tensor(selected, dtype=torch.long)


def build_trajectories(D: dict, stage_train: list[torch.Tensor],
                       budgets: list[int], arms: list[str], device: str,
                       include_sca: bool) -> tuple[dict, dict]:
    """Construct every continual memory trajectory and its audit metadata."""
    trajectories = {}
    audits = {}
    sca_args = make_sca_args()
    for arm in arms:
        memory = None
        memories, blocks, rows = [], [], []
        for stage, ids in enumerate(stage_train):
            k_t = int(budgets[stage])
            if arm == "full_history_syn":
                source = torch.cat(stage_train[: stage + 1])
                memory = G.condense_static(
                    D["x"][source], D["text_emb"][source],
                    [D["texts"][int(i)] for i in source], k_t,
                    D["encoder"], device=device)
                block = raw_support(D, ids, stage)
                keep, k_new = 0, len(memory["x"])
                audit = {"mode": "offline_full_history_recondense",
                         "source_rows": int(len(source))}
            else:
                keep = 0 if memory is None else min(len(memory["x"]), k_t)
                k_new = k_t if memory is None else max(0, k_t - keep)
                if arm in REAL_ARMS:
                    block, audit = U.build_real_selection_block(
                        D, ids, k_new, arm, stage, device)
                    selected_ids = _selected_ids_for_block(D, ids, block)
                    block["edge_index"] = induced_edges(
                        D["edge_index"], selected_ids, len(D["x"]))
                elif arm == "recluster":
                    if memory is None:
                        memory = G.condense_static(
                            D["x"][ids], D["text_emb"][ids],
                            [D["texts"][int(i)] for i in ids], k_t,
                            D["encoder"], device=device)
                        block = raw_support(D, ids, stage)
                        audit = {"mode": "stage0_condense"}
                        keep, k_new = 0, len(memory["x"])
                    else:
                        memory, recluster_audit = G.continual_recluster(
                            memory, D["x"][ids], D["text_emb"][ids],
                            [D["texts"][int(i)] for i in ids], k_t, stage,
                            D["encoder"], device=device)
                        block = raw_support(D, ids, stage)
                        audit = {"mode": "history_recluster",
                                 "recluster": recluster_audit}
                        keep, k_new = len(memory["x"]), 0
                else:
                    block = U.build_stage_block(
                        D, ids, k_new, D["encoder"], stage,
                        "joint_g5_c320", "concat", device,
                        "dispersion", D["stage_classes"][stage], 0.5, 0.5)
                    audit = {"mode": "dispersion_synthetic_block"}
                    if arm == "sca" and include_sca and memory is not None:
                        memory, audit = U.select_sca(
                            memory, ids, D, stage, k_t, keep, k_new,
                            sca_args, device)
                        block = None
                if arm != "recluster" or memory is None:
                    if arm != "sca" or block is not None:
                        memory = SCA.append_memory(
                            memory, block, keep,
                            topology_mode="frozen_old", cross_degree=1,
                            refresh_mode="frozen_old", device=device)

            memories.append(memory)
            blocks.append(block)
            rows.append({"stage": int(stage), "budget": k_t,
                         "keep_old_rows": int(keep),
                         "new_rows": int(k_new), "audit": audit,
                         "memory_rows": int(len(memory["x"]))})
        trajectories[arm] = {"memories": memories, "blocks": blocks}
        audits[arm] = rows
    return trajectories, audits


def eval_seen_prompt(model, D: dict, stage: int) -> dict:
    """Report strict seen-prompt F1 beside open-vocabulary F1."""
    logits = model.logits(D["x_device"], D["real_adj"],
                          D["prompt_emb"].to(D["x_device"].device))
    allowed = sorted({int(c) for cls in D["stage_classes"][:stage + 1]
                      for c in cls})
    pred = logits[:, allowed].argmax(1).cpu()
    target = D["y"].cpu()
    values = []
    for classes in D["stage_classes"][:stage + 1]:
        mask = D["report_dev_mask"] & torch.isin(
            target, torch.tensor(classes, dtype=target.dtype))
        yy, pp = target[mask].numpy(), pred[mask].numpy()
        values.append(100 * f1_score(yy, pp, average="macro",
                                     labels=np.unique(yy), zero_division=0))
    return {"seen_prompt_per_stage": values,
            "seen_prompt_cumulative": float(np.mean(values))}


def aggregate_scores(scores: list[dict], seeds: list[int], support: dict,
                     steps: int, tag: str, elapsed: float) -> dict:
    return {
        "tag": tag,
        "roots": int(len(support["x"])),
        "steps": int(steps),
        "student_seeds": [int(v) for v in seeds],
        "cum_dev_f1_mean": float(np.mean([s["cumulative_macro_f1"]
                                           for s in scores])),
        "cum_dev_f1_std": float(np.std([s["cumulative_macro_f1"]
                                         for s in scores])),
        "current_f1_mean": float(np.mean([s["current_f1"] for s in scores])),
        "old_mean_f1_mean": (float(np.mean([s["old_mean_f1"] for s in scores
                                             if s.get("old_mean_f1") is not None]))
                              if any(s.get("old_mean_f1") is not None
                                     for s in scores) else None),
        "per_stage_mean": [float(np.mean([s["per_stage"][i]["macro_f1"]
                                           for s in scores]))
                           for i in range(len(scores[0]["per_stage"]))],
        "seen_prompt_cumulative_mean": float(np.mean([
            s["seen_prompt_cumulative"] for s in scores])),
        "elapsed_seconds": float(elapsed),
        "per_student": scores,
    }


def fit_scratch(support: dict, D: dict, stage: int, seeds: list[int],
                steps: int, batch: int, device: str, tag: str) -> dict:
    scores = []
    started = time.perf_counter()
    for seed in seeds:
        model = G.train_student(support, int(seed), device, int(steps),
                                batch=int(batch))
        score = G.eval_stagewise(model, D["x_device"], D["real_adj"],
                                 D["prompt_emb"], D["y"],
                                 D["report_dev_mask"], stage)
        score.update(eval_seen_prompt(model, D, stage))
        scores.append(score)
        del model
    return aggregate_scores(scores, seeds, support, steps, tag,
                            time.perf_counter() - started)


def run_scratch(trajectories: dict, D: dict, stage_train: list[torch.Tensor],
                arms: list[str], seeds: list[int], batch: int,
                raw_steps: int, device: str) -> list[dict]:
    rows = []
    for arm in arms:
        for stage, memory in enumerate(trajectories[arm]["memories"]):
            steps = protocol_v3.student_steps(len(memory["x"]))
            row = fit_scratch(memory, D, stage, seeds, steps, batch, device,
                              f"{arm}:scratch:s{stage}")
            row.update({"arm": arm, "stage": int(stage), "mode": "scratch"})
            rows.append(row)
    # Offline raw upper reference: all legal raw nodes seen so far and an
    # extended cap, so it is not confused with a compressed-memory score.
    for stage in range(len(stage_train)):
        source = torch.cat(stage_train[:stage + 1])
        support = raw_support(D, source, stage)
        row = fit_scratch(support, D, stage, seeds, int(raw_steps), batch,
                          device, f"raw_full_retrain:s{stage}")
        row.update({"arm": "raw_full_retrain", "stage": int(stage),
                    "mode": "scratch", "upper_reference": True})
        rows.append(row)
    return rows


def fine_tune(model: torch.nn.Module, support: dict, seed: int,
              steps: int, batch: int, device: str) -> torch.nn.Module:
    """Continue one student using only the newly arrived raw support."""
    model.train()
    x = support["x"].to(device)
    t = support["text_emb"].to(device)
    adj = G.norm_adj(support["edge_index"], len(x), device)
    opt = torch.optim.Adam(model.parameters(), lr=protocol_v3.STUDENT.lr)
    gen = torch.Generator().manual_seed(int(seed) * 17 + 5)
    for _ in range(int(steps)):
        rows = torch.randperm(len(x), generator=gen)[:min(int(batch), len(x))]
        rows = rows.to(device)
        g = model.encode_graph(x, adj)[rows]
        tt = model.encode_text(t[rows])
        logits = model.logit_scale.exp().clamp(max=100) * g @ tt.T
        target = torch.arange(len(rows), device=device)
        loss = (F.cross_entropy(logits, target)
                + F.cross_entropy(logits.T, target)) / 2
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return model.eval()


def run_finetune(trajectories: dict, D: dict,
                 stage_train: list[torch.Tensor], arms: list[str],
                 seeds: list[int], batch: int, device: str) -> list[dict]:
    rows = []
    for arm in arms:
        memories = trajectories[arm]["memories"]
        models = []
        base_steps = protocol_v3.student_steps(len(memories[0]["x"]))
        for seed in seeds:
            models.append(G.train_student(memories[0], int(seed), device,
                                          base_steps, batch=batch))
        for stage in range(len(stage_train)):
            if stage > 0:
                support = raw_support(D, stage_train[stage], stage)
                ft_steps = protocol_v3.student_steps(len(support["x"]))
                started = time.perf_counter()
                for index, seed in enumerate(seeds):
                    models[index] = fine_tune(
                        models[index], support, int(seed), ft_steps, batch, device)
                elapsed = time.perf_counter() - started
            else:
                support = memories[0]
                ft_steps, elapsed = base_steps, 0.0
            scores = []
            for model in models:
                score = G.eval_stagewise(model, D["x_device"], D["real_adj"],
                                         D["prompt_emb"], D["y"],
                                         D["report_dev_mask"], stage)
                score.update(eval_seen_prompt(model, D, stage))
                scores.append(score)
            row = aggregate_scores(
                scores, seeds, memories[stage], ft_steps,
                f"{arm}:delta_finetune:s{stage}", elapsed)
            row.update({"arm": arm, "stage": int(stage),
                        "mode": "delta_finetune",
                        "support_rows": int(len(support["x"])),
                        "fine_tune_steps": int(ft_steps),
                        "pretrain_rows": int(len(memories[stage - 1]["x"]))
                        if stage > 0 else int(len(memories[0]["x"]))})
            rows.append(row)
        for model in models:
            del model
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(U.DATA_CFG), required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--rate", type=float, default=0.05)
    ap.add_argument("--arms", nargs="+", default=[
        "append_fixed", "sca", "random_real", "k_center", "herding",
        "kmeans_medoid", "recluster", "full_history_syn"])
    ap.add_argument("--student-seeds", type=int, nargs="+",
                    default=list(protocol_v3.STUDENT_SEEDS))
    ap.add_argument("--student-batch", type=int,
                    default=protocol_v3.STUDENT.batch)
    ap.add_argument("--raw-steps-cap", type=int,
                    default=protocol_v3.RAW_STEPS_CAP)
    ap.add_argument("--no-sca", action="store_true")
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "results.json").exists():
        raise FileExistsError(f"immutable output exists: {out}")
    unknown = sorted(set(args.arms) - set(ALL_ARMS))
    if unknown:
        raise ValueError(f"unknown arms: {unknown}")

    D = U.load_unified(
        args.dataset, U.DATA_CFG[args.dataset]["path"],
        "/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip",
        args.device, n_stages=3, split_mode="canonical",
        base_fraction=0.4, report_dev_mod=protocol_v3.REPORT_DEV_MOD)
    pool_ids = torch.where(D["pool_mask"])[0]
    pool_emb = D["encoder"].encode([D["texts"][int(i)] for i in pool_ids])
    text_emb = torch.zeros((len(D["x"]), 384), dtype=pool_emb.dtype)
    text_emb[pool_ids] = pool_emb
    D["text_emb"] = text_emb
    D["prompt_emb"] = D["encoder"].encode(D["prompts"])
    D["real_adj"] = G.norm_adj(D["edge_index"], len(D["x"]), args.device)
    D["x_device"] = D["x"].to(args.device)
    stage_train = [torch.where(torch.isin(
        D["y"], torch.tensor(classes)) & D["pool_mask"])[0]
                   for classes in D["stage_classes"]]
    budgets = [math.floor(args.rate * sum(len(v)
                                          for v in stage_train[:s + 1]))
               for s in range(len(stage_train))]
    print(json.dumps({"dataset": args.dataset,
                      "stage_classes": D["stage_classes"],
                      "stage_train_counts": [len(v) for v in stage_train],
                      "budgets": budgets}), flush=True)

    trajectories, audits = build_trajectories(
        D, stage_train, budgets, args.arms, args.device, not args.no_sca)
    scratch = run_scratch(trajectories, D, stage_train, args.arms,
                          args.student_seeds, args.student_batch,
                          args.raw_steps_cap, args.device)
    finetune = run_finetune(trajectories, D, stage_train, args.arms,
                            args.student_seeds, args.student_batch,
                            args.device)
    manifest = {
        **protocol_v3.config_dict(dataset=args.dataset, rate=args.rate,
                                  student_batch=args.student_batch,
                                  student_seeds=args.student_seeds,
                                  raw_steps_cap=args.raw_steps_cap),
        "runner": "run_small_protocol_comparison.py",
        "arms": args.arms,
        "stage_classes": D["stage_classes"],
        "stage_split_source": D["stage_split_source"],
        "stage_train_counts": [int(len(v)) for v in stage_train],
        "budgets": budgets,
        "mask_counts": {
            "selector_dev": int(D["selector_dev_mask"].sum()),
            "report_dev": int(D["report_dev_mask"].sum()),
            "test": int(D["test_mask"].sum()),
            "pool": int(D["pool_mask"].sum()),
        },
        "test_accessed": False,
        "test_source_sealed": True,
        "construction_label_free": True,
        "raw_upper_reference": (
            "all legal pool nodes seen so far; retrained from scratch; "
            "extended steps cap"),
        "fine_tune_protocol": (
            "stage0 trained on each arm memory; later stages fine-tune only "
            "on current raw pool delta"),
        "external_baseline_scope": (
            "label-free real-node coreset and continual condensation controls; "
            "supervised TAG methods are not silently treated as same-task baselines"),
        "trajectory_audits": audits,
        "scratch_results": scratch,
        "finetune_results": finetune,
    }
    json.dump(manifest, (out / "results.json").open("w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
