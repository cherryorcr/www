"""Fixed supervised sanity check of frozen STAD artifacts, not score reproduction."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import run_stad_graphclip_static as B
from small_comparison_metrics import score_predictions

C = B.C
STAGE = 2
SEEDS = list(range(100, 106))


def new_model(models, data, device, seed):
    torch.manual_seed(seed)
    return models.GCN(data["x"].shape[1], 64, int(data["y"].max()) + 1,
                      dropout=.5, device=device).to(device)


@torch.no_grad()
def score(model, data, split, device):
    if split not in {"selector_dev_mask", "report_dev_mask"}:
        raise ValueError("sanity check cannot evaluate test labels")
    model.eval()
    mask = data[split]
    pred = model(data["x_device"], data["real_adj"])[mask.to(device)].argmax(1)
    return score_predictions(data["y"][mask].numpy(), pred.cpu().numpy(),
                             data["stage_classes"], STAGE)


def fit(models, data, support, labels, seed, device, steps=600):
    model = new_model(models, data, device, seed)
    start_hash = C.tensor_hash(model.state_dict())
    x = support["x"].to(device)
    adj = C.training_adjacency(support, device)
    labels = labels.to(device)
    if labels.shape != (len(x),) or labels.dtype != torch.long:
        raise ValueError("invalid supervised support labels")
    opt = torch.optim.Adam(model.parameters(), lr=.01, weight_decay=.0005)
    best, best_step, best_state, curve = -1., None, None, []
    cuda = torch.device(device).type == "cuda"
    if cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for step in range(1, steps + 1):
        model.train()
        loss = F.cross_entropy(model(x, adj), labels)
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite supervised sanity loss")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        selector = score(model, data, "selector_dev_mask", device)
        value = selector["accuracy"]
        curve.append({"step": step, "selector_accuracy": value,
                      "train_loss": float(loss.detach())})
        if value > best:
            best, best_step = value, step
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is None:
        raise ValueError("no valid selector checkpoint")
    if cuda:
        torch.cuda.synchronize(device)
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    row = {"seed": seed, "stage": STAGE, "steps": steps,
           "rows": len(x), "start_state_sha256": start_hash,
           "end_state_sha256": C.tensor_hash(state),
           "selected_state_sha256": C.tensor_hash(best_state),
           "support_sha256": C.tensor_hash(support),
           "labels_sha256": C.tensor_hash({"labels": labels.cpu()}),
           "seconds": time.perf_counter() - started,
           "peak_allocated_mb": torch.cuda.max_memory_allocated(device) / 2**20 if cuda else None,
           "selected_steps": best_step, "selector_best_accuracy": best,
           "selection_curve": curve,
           "metrics": score(model, data, "report_dev_mask", device)}
    model.load_state_dict(best_state)
    row["selected_metrics"] = score(model, data, "report_dev_mask", device)
    return row, {"state": state, "selected_state": best_state, "result": row}


def audit_fit(models, data, support, labels, row, saved, device):
    if B.normalized_metrics(saved["result"]) != B.normalized_metrics(row):
        raise ValueError("checkpoint and JSON disagree")
    if (row["support_sha256"] != C.tensor_hash(support)
            or row["labels_sha256"] != C.tensor_hash({"labels": labels})):
        raise ValueError("stale support or labels")
    model = new_model(models, data, device, row["seed"])
    if C.tensor_hash(model.state_dict()) != row["start_state_sha256"]:
        raise ValueError("initialization mismatch")
    curve = row["selection_curve"]
    if [v["step"] for v in curve] != list(range(1, row["steps"] + 1)):
        raise ValueError("incomplete selection trajectory")
    best = max(curve, key=lambda v: v["selector_accuracy"])
    if (best["step"] != row["selected_steps"]
            or best["selector_accuracy"] != row["selector_best_accuracy"]):
        raise ValueError("incorrect selector checkpoint")
    for state_key, hash_key, metric_key in (
            ("state", "end_state_sha256", "metrics"),
            ("selected_state", "selected_state_sha256", "selected_metrics")):
        if C.tensor_hash(saved[state_key]) != row[hash_key]:
            raise ValueError("corrupt checkpoint")
        model.load_state_dict(saved[state_key])
        actual = score(model, data, "report_dev_mask", device)
        if B.normalized_metrics(actual) != B.normalized_metrics(row[metric_key]):
            raise ValueError("restored metrics mismatch")
    if score(model, data, "selector_dev_mask", device)["accuracy"] != row["selector_best_accuracy"]:
        raise ValueError("restored selector mismatch")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cora", "citeseer"], required=True)
    ap.add_argument("--stad-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--official", type=Path, default=Path("/home/pc/research/STAD_official"))
    ap.add_argument("--minilm", default=str(Path(B.U.__file__).parent / "minilm_from_graphclip"))
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    torch.set_num_threads(2)
    if torch.device(args.device).type == "cuda":
        torch.cuda.set_device(args.device)
    models, _, _, _, official = B.official_components(args.official)
    data = B.S.load_data(args.dataset, args.minilm, args.device)
    stages, budgets = B.S.stage_inputs(data)
    ids = torch.cat(stages)
    B.validate_sources(data, ids, STAGE)
    base = args.stad_root / args.dataset / "s2"
    memory = torch.load(base / "memory.pt", map_location="cpu", weights_only=False)
    core = torch.load(base / "core.pt", map_location="cpu", weights_only=False)
    construction = json.loads((base / "construction.json").read_text())
    if (C.tensor_hash(memory) != construction["tensor_sha256"]
            or C.sha(base / "memory.pt") != construction["memory_file_sha256"]
            or construction["source_ids"] != ids.tolist()
            or len(memory["x"]) != budgets[-1]):
        raise ValueError("STAD artifact provenance mismatch")
    raw = C.raw_support(data, ids, STAGE)
    if C.tensor_hash(raw) != construction["source_support_sha256"]:
        raise ValueError("STAD source pool mismatch")
    seen = torch.tensor([c for group in data["stage_classes"] for c in group])
    supports = {"stad_supervised": (memory, seen[core["labels"].long()]),
                "raw_supervised_matched": (raw, data["y"][ids].long())}
    manifest = {"version": "stad-supervised-sanity-v1", "dataset": args.dataset,
                "stage": STAGE, "seeds": SEEDS, "steps": 600,
                "model": "official GCN64, dropout0.5, Adam0.01, weight_decay0.0005",
                "selection": "selector accuracy every epoch; earliest tie",
                "runner_sha256": C.sha(__file__), "official": official,
                "source_sha256": C.sha(B.U.DATA_CFG[args.dataset]["path"]),
                "memory_sha256": C.sha(base / "memory.pt"),
                "core_sha256": C.sha(base / "core.pt"),
                "dependencies": {Path(m.__file__).name: C.sha(m.__file__)
                                 for m in (B, B.S, C, B.U, B.G, B.P)},
                "test_labels_evaluated": False, "report_dev_used_for_selection": False,
                "notice": "post-hoc fixed-setting diagnostic, not official reproduction; not the GraphCLIP retention denominator"}
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("output has a different manifest")
    C.save_json(path, manifest)
    results = []
    for arm, (support, labels) in supports.items():
        rows = []
        for seed in SEEDS:
            path = args.output / f"{arm}_seed{seed}.json"
            ckpt = path.with_suffix(".pt")
            if path.exists() and ckpt.exists():
                row = json.loads(path.read_text())
            else:
                row, saved = fit(models, data, support, labels, seed, args.device)
                torch.save(saved, ckpt)
                C.save_json(path, row)
            saved = torch.load(ckpt, map_location="cpu", weights_only=False)
            audit_fit(models, data, support, labels, row, saved, args.device)
            rows.append(row)
        values = [r["selected_metrics"]["cumulative_f1"] for r in rows]
        result = {"arm": arm, "per_seed": rows, "rows": len(support["x"]),
                  "final_f1": float(np.mean(values)), "std_population": float(np.std(values)),
                  "accuracy": float(np.mean([r["selected_metrics"]["accuracy"] for r in rows])),
                  "student_seconds": sum(r["seconds"] for r in rows)}
        results.append(result)
        print(json.dumps({k: v for k, v in result.items() if k != "per_seed"}), flush=True)
    C.save_json(args.output / "results.json", {"manifest": manifest, "results": results})
    C.save_json(args.output / "acceptance.json", {"errors": [], "fits_restored": 12,
                "report_metrics_recomputed": 24, "test_labels_evaluated": False})


if __name__ == "__main__":
    main()
