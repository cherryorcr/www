"""Supervised full-data upper reference for the canonical small TAG task.

This arm is intentionally separate from zero-shot GraphCLIP results.  It uses
all legal pool labels available through a stage and a full transductive graph
for message passing, selects a checkpoint on selector-dev, and reports only on
the untouched report-dev split.  It is a task upper reference, not a
label-free condensation method.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

import run_small_comparison_audited as C
import run_unified_v3_continual as U
from small_comparison_metrics import score_predictions


P = C.P
G = C.G


class SupervisedGCN(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.g1 = nn.Linear(in_dim, hidden)
        self.g2 = nn.Linear(hidden, out_dim)

    def logits(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = F.relu(adj @ self.g1(x))
        return adj @ self.g2(h)


def load_data(dataset: str, minilm: str, device: str) -> dict:
    data = U.load_unified(dataset, U.DATA_CFG[dataset]["path"], minilm, device,
                           n_stages=3, split_mode="canonical")
    data["real_adj"] = G.norm_adj(data["edge_index"], len(data["x"]), device)
    data["x_device"] = data["x"].to(device)
    return data


def score_model(model, data, stage: int, split: str, device: str) -> dict:
    model.eval()
    with torch.no_grad():
        logits = model.logits(data["x_device"], data["real_adj"])
    mask = data[split]
    seen = sorted({int(c) for group in data["stage_classes"][:stage + 1]
                   for c in group})
    target = data["y"][mask].cpu().numpy()
    all_pred = logits[mask.to(device)].argmax(1).cpu().numpy()
    seen_logits = logits[mask.to(device)][:, seen]
    seen_pred = torch.tensor(seen)[seen_logits.argmax(1).cpu()].numpy()
    return {"all": score_predictions(target, all_pred, data["stage_classes"], stage),
            "seen": score_predictions(target, seen_pred, data["stage_classes"], stage)}


def validate_train_ids(data, train_ids, stage):
    if not len(train_ids) or len(train_ids.unique()) != len(train_ids):
        raise ValueError("empty or duplicated supervised source IDs")
    if train_ids.min() < 0 or train_ids.max() >= len(data["x"]):
        raise ValueError("invalid supervised source IDs")
    seen = torch.tensor([c for group in data["stage_classes"][:stage + 1] for c in group])
    expected = torch.where(data["pool_mask"] & torch.isin(data["y"], seen))[0]
    if not torch.equal(train_ids.sort().values, expected):
        raise ValueError("supervised source is not exactly the legal seen pool")
    for split in ("selector_dev_mask", "report_dev_mask", "test_mask"):
        if data[split][train_ids].any():
            raise ValueError(f"held-out labels in training: {split}")


def fit(data: dict, train_ids: torch.Tensor, stage: int, seed: int,
        steps: int, device: str, lr: float = .001, weight_decay: float = 0.,
        eval_every: int = 25, patience: int = 20, min_steps: int = 512,
        checkpoint: Path | None = None, evaluate_report: bool = True) -> dict:
    validate_train_ids(data, train_ids, stage)
    if steps < 1 or eval_every < 1 or patience < 1:
        raise ValueError("steps, evaluation interval and patience must be positive")
    torch.manual_seed(seed)
    model = SupervisedGCN(data["x"].shape[1], 384, int(data["y"].max()) + 1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    start_hash = C.tensor_hash(model.state_dict())
    best = -1.0
    best_step = None
    best_state = None
    curve = []
    stale = 0
    cuda = torch.device(device).type == "cuda"
    if cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    device_ids = train_ids.to(device)
    targets = data["y"][train_ids].to(device)
    for step in range(1, steps + 1):
        model.train()
        logits = model.logits(data["x_device"], data["real_adj"])
        loss = F.cross_entropy(logits[device_ids], targets)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite supervised loss at {step}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % eval_every == 0 or step == steps:
            m = score_model(model, data, stage, "selector_dev_mask", device)
            value = m["all"]["cumulative_f1"]
            curve.append({"step": step, "selector_all_f1": value,
                          "train_loss": float(loss.detach())})
            if value > best:
                best, best_step = value, step
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
            if step >= min_steps and stale >= patience:
                break
    if cuda:
        torch.cuda.synchronize(device)
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    row = {"seed": seed, "stage": stage, "steps": step, "max_steps": steps,
           "lr": lr, "weight_decay": weight_decay,
           "eval_every": eval_every, "patience": patience, "min_steps": min_steps,
           "start_state_sha256": start_hash,
           "train_ids_sha256": C.tensor_hash({"source_ids": train_ids}),
           "train_rows": len(train_ids), "seconds": time.perf_counter() - started,
           "peak_allocated_mb": torch.cuda.max_memory_allocated(device) / 2**20 if cuda else None,
           "end_state_sha256": C.tensor_hash(state)}
    if best_state is None:
        raise AssertionError("selector did not produce a checkpoint")
    row.update(selected_steps=best_step, selection_curve=curve,
               selector_best_f1=best, selected_state_sha256=C.tensor_hash(best_state))
    if evaluate_report:
        row["metrics"] = score_model(model, data, stage, "report_dev_mask", device)
    model.load_state_dict(best_state)
    if evaluate_report:
        row["selected_metrics"] = score_model(model, data, stage,
                                               "report_dev_mask", device)
    if checkpoint is not None:
        torch.save({"state": state, "selected_state": best_state,
                    "train_ids": train_ids, "result": row}, checkpoint)
    return row


def load_cached_fit(path: Path, checkpoint: Path, ids: torch.Tensor) -> dict:
    row = json.loads(path.read_text())
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    for key, hash_key in (("state", "end_state_sha256"),
                          ("selected_state", "selected_state_sha256")):
        if C.tensor_hash(saved[key]) != row[hash_key]:
            raise ValueError(f"corrupt supervised checkpoint: {checkpoint}")
    if (json.loads(json.dumps(saved["result"])) != row or not torch.equal(saved["train_ids"], ids)
            or C.tensor_hash({"source_ids": ids}) != row["train_ids_sha256"]):
        raise ValueError("stale supervised provenance")
    return row


def aggregate_selected(rows: list[dict]) -> dict:
    return C.aggregate([dict(r, metrics=r["selected_metrics"]) for r in rows])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cora", "citeseer"], required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--minilm", default="/home/pc/research/TAG_condensation/code/graphclip_unified_v3_2026_09_13/minilm_from_graphclip")
    ap.add_argument("--steps", type=int, default=P.RAW_STEPS_CAP)
    ap.add_argument("--seeds", nargs="+", type=int, default=list(P.STUDENT_SEEDS))
    ap.add_argument("--selection-seeds", nargs="+", type=int, default=[7001, 7002])
    ap.add_argument("--learning-rates", nargs="+", type=float, default=[.001, .01])
    ap.add_argument("--weight-decays", nargs="+", type=float, default=[0., .0005])
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--min-steps", type=int, default=512)
    args = ap.parse_args()
    if set(args.seeds) & set(args.selection_seeds):
        raise ValueError("selection and report seeds must be disjoint")
    torch.set_num_threads(2)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    data = load_data(args.dataset, args.minilm, args.device)
    stages = [torch.where(data["pool_mask"] & torch.isin(
        data["y"], torch.tensor(classes)))[0]
              for classes in data["stage_classes"]]
    train_ids = [torch.cat(stages[:s + 1]) for s in range(len(stages))]
    manifest = {
        "version": "supervised-full-data-searched-reference-v2",
        "dataset": args.dataset,
        "source_sha256": C.sha(U.DATA_CFG[args.dataset]["path"]),
        "runner_sha256": C.sha(__file__),
        "stage_classes": data["stage_classes"],
        "stage_train_counts": [len(v) for v in train_ids],
        "steps": args.steps, "seeds": args.seeds,
        "model": "two-layer full-graph GCN classifier, hidden=384, output=all classes",
        "optimizer": "Adam; fixed finite grid selected on selector-dev only",
        "learning_rates": args.learning_rates, "weight_decays": args.weight_decays,
        "selection_seeds": args.selection_seeds,
        "eval_every": args.eval_every, "patience": args.patience,
        "min_steps": args.min_steps,
        "training": "cross-entropy on all legal pool labels seen through current stage",
        "selection": "per-stage grid by mean selector-dev all-prompt cumulative Macro-F1 over independent seeds; then checkpoint selection per report seed",
        "evaluation": "report-dev Macro-F1; transductive full real graph",
        "test_labels_evaluated": False,
        "report_dev_used_for_selection": False,
        "upper_bound_notice": "finite-search supervised reference, not a theoretical or exhaustively tuned upper bound",
        "dependencies": {Path(m.__file__).name: C.sha(m.__file__) for m in (C, U, G, P)},
        "comparison_boundary": "separate from zero-shot GraphCLIP and condensation arms",
    }
    manifest_path = out / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("supervised output already has a different manifest")
    C.save_json(manifest_path, manifest)
    results = []
    for stage, ids in enumerate(train_ids):
        grid = []
        for lr in args.learning_rates:
            for wd in args.weight_decays:
                trials = []
                for seed in args.selection_seeds:
                    key = f"search_s{stage}_lr{lr}_wd{wd}_seed{seed}"
                    path, ckpt = out / f"{key}.json", out / f"{key}.pt"
                    if path.exists() and ckpt.exists():
                        trial = load_cached_fit(path, ckpt, ids)
                    else:
                        trial = fit(data, ids, stage, seed, args.steps, args.device,
                                    lr, wd, args.eval_every, args.patience, args.min_steps,
                                    checkpoint=ckpt, evaluate_report=False)
                        C.save_json(path, trial)
                    trials.append(trial)
                grid.append({"lr": lr, "weight_decay": wd, "trials": trials,
                             "selector_mean": sum(t["selector_best_f1"] for t in trials) / len(trials)})
        chosen = max(grid, key=lambda r: r["selector_mean"])
        C.save_json(out / f"search_s{stage}.json", {"grid": grid,
                    "chosen": {k: chosen[k] for k in ("lr", "weight_decay", "selector_mean")}})
        rows = []
        for seed in args.seeds:
            key = f"supervised_s{stage}_seed{seed}"
            row_path = out / f"{key}.json"
            checkpoint = out / f"{key}.pt"
            if row_path.exists() and checkpoint.exists():
                row = load_cached_fit(row_path, checkpoint, ids)
            else:
                row = fit(data, ids, stage, seed, args.steps, args.device,
                          chosen["lr"], chosen["weight_decay"], args.eval_every,
                          args.patience, args.min_steps, checkpoint=checkpoint)
                C.save_json(row_path, row)
            rows.append(row)
        result = {"arm": "supervised_full_data_upper_bound", "stage": stage,
                  "rows": len(ids), "steps": args.steps, "per_seed": rows,
                  "metrics": aggregate_selected(rows),
                  "selected_config": {k: chosen[k] for k in ("lr", "weight_decay")},
                  "search_seconds": sum(t["seconds"] for g in grid for t in g["trials"]),
                  "terminal_metrics": C.aggregate(rows)}
        results.append(result)
        C.save_json(out / "progress.json", {"manifest": manifest, "results": results})
        print(json.dumps({"stage": stage, "selected": result["metrics"]["all"]}),
              flush=True)
    C.save_json(out / "results.json", {"manifest": manifest, "results": results})
    print("SUPERVISED_UPPER_BOUND_DONE", flush=True)


if __name__ == "__main__":
    main()
