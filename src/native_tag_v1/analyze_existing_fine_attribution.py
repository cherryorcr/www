"""Analyze already exported native TAG recursive artifacts.

This read-only utility computes row drift and structural fingerprints for the
paired append-joint/protected-joint runs.  It deliberately does not load the
confirmation partition or retrain a model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def sha(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().contiguous()
        payload = str((tuple(value.shape), str(value.dtype))).encode() + value.numpy().tobytes()
    else:
        payload = repr(value).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def row_drift(before, after, rows):
    mask = torch.as_tensor(rows, dtype=torch.bool)
    d = (after["x"][mask] - before["x"][mask]).double()
    b = before["x"][mask].double()
    norms = d.norm(dim=1)
    base = b.norm(dim=1).clamp_min(1e-12)
    return {
        "rows": int(mask.sum()),
        "mean_l2": float(norms.mean()) if mask.any() else 0.0,
        "median_l2": float(norms.median()) if mask.any() else 0.0,
        "max_l2": float(norms.max()) if mask.any() else 0.0,
        "mean_relative_l2": float((norms / base).mean()) if mask.any() else 0.0,
        "x_norm_before_mean": float(b.norm(dim=1).mean()) if mask.any() else 0.0,
        "x_norm_after_mean": float(after["x"][mask].double().norm(dim=1).mean()) if mask.any() else 0.0,
    }


def compare_variant(aj0, pj0, aj, pj, stage, label):
    if aj["origin_ids"].tolist() != pj["origin_ids"].tolist():
        raise ValueError(f"paired {label} rows differ")
    old = aj0["stage"] != stage
    new = aj0["stage"] == stage
    fixed = ("edge_index", "text_emb", "texts", "weights", "origin_ids", "stage")
    return {
        "label": label,
        "initial_pair_identical_x": bool(torch.equal(aj0["x"], pj0["x"])),
        "initial_pair_identical_fixed": {key: sha(aj0[key]) == sha(pj0[key]) for key in fixed},
        "final_pair_identical_fixed": {key: sha(aj[key]) == sha(pj[key]) for key in fixed},
        "joint_old_drift": row_drift(aj0, aj, old),
        "protected_old_drift": row_drift(pj0, pj, old),
        "joint_new_drift": row_drift(aj0, aj, new),
        "protected_new_drift": row_drift(pj0, pj, new),
        "joint_vs_protected_final_old_x_l2": float((aj["x"][old] - pj["x"][old]).double().norm(dim=1).mean()),
        "joint_vs_protected_final_new_x_l2": float((aj["x"][new] - pj["x"][new]).double().norm(dim=1).mean()),
        "joint_norms": aj.get("metadata", {}).get("last_norms", {}),
        "protected_norms": pj.get("metadata", {}).get("last_norms", {}),
        "joint_selected_step": aj.get("metadata", {}).get("selected_outer_step"),
        "protected_selected_step": pj.get("metadata", {}).get("selected_outer_step"),
    }


def compare_pair(root: Path, stage: int):
    aj0 = load(root / f"append_joint_s{stage}_initial.pt")
    pj0 = load(root / f"protected_joint_s{stage}_initial.pt")
    aj = load(root / f"append_joint_s{stage}_artifact.pt")
    pj = load(root / f"protected_joint_s{stage}_artifact.pt")
    if aj0["origin_ids"].tolist() != pj0["origin_ids"].tolist():
        raise ValueError("paired initial rows differ")
    last_aj = load(root / f"append_joint_s{stage}_last.pt")
    last_pj = load(root / f"protected_joint_s{stage}_last.pt")
    best_aj = load(root / f"append_joint_s{stage}_best.pt")
    best_pj = load(root / f"protected_joint_s{stage}_best.pt")
    result = {
        "stage": stage,
        "roots": len(aj0["x"]),
        "old_rows": int((aj0["stage"] != stage).sum()),
        "new_rows": int((aj0["stage"] == stage).sum()),
        "selected": compare_variant(aj0, pj0, aj, pj, stage, "selected"),
        "best": compare_variant(aj0, pj0, best_aj, best_pj, stage, "best_snapshot"),
        "last": compare_variant(aj0, pj0, last_aj, last_pj, stage, "last_snapshot"),
        "metadata": {
            "joint": aj.get("metadata", {}),
            "protected": pj.get("metadata", {}),
        },
    }
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--stages", type=int, nargs="+", default=[1, 2])
    p.add_argument("--output", required=True)
    args = p.parse_args()
    root = Path(args.root)
    out = {str(stage): compare_pair(root, stage) for stage in args.stages}
    Path(args.output).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
