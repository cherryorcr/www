"""Measure label-free synthetic graph/text correspondence along an ASM path."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import torch
import torch.nn.functional as F

from native_core import DATA_KEYS, clean_data, create_model


STEP_RE = re.compile(r".*_outer_(\d+)\.pt$")


def tensor_sha(value) -> str:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().contiguous()
        raw = str((tuple(value.shape), str(value.dtype))).encode() + value.numpy().tobytes()
    else:
        raw = repr(value).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def correspondence(logits: torch.Tensor) -> dict:
    n = len(logits)
    if n < 2:
        raise ValueError("correspondence metric needs at least two pairs")
    target = torch.arange(n, device=logits.device)
    cosine = logits * 0.07
    positive = cosine.diag()
    mask = torch.eye(n, dtype=torch.bool, device=logits.device)
    row_negative = cosine.masked_fill(mask, -torch.inf).max(1).values
    col_negative = cosine.masked_fill(mask, -torch.inf).max(0).values
    return {
        "rows": n,
        "symmetric_infonce": float((F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target)) / 2),
        "positive_cosine_mean": float(positive.mean()),
        "row_margin_mean": float((positive - row_negative).mean()),
        "column_margin_mean": float((positive - col_negative).mean()),
        "graph_to_text_recall_at_1": float((logits.argmax(1) == target).float().mean()),
        "text_to_graph_recall_at_1": float((logits.argmax(0) == target).float().mean()),
    }


@torch.no_grad()
def analyze(model, artifact: dict, current_stage: int) -> dict:
    parameter = next(model.parameters())
    data = clean_data(artifact, parameter.device, with_adjacency=True)
    data["x"] = data["x"].to(parameter.dtype)
    data["text_emb"] = data["text_emb"].to(parameter.dtype)
    data["_adj"] = data["_adj"].to(parameter.dtype)
    model.eval()
    logits = model(data)
    old = torch.where(data["stage"] < current_stage)[0]
    new = torch.where(data["stage"] == current_stage)[0]
    result = {"all": correspondence(logits)}
    if len(old) >= 2:
        result["old_within"] = correspondence(logits[old][:, old])
    if len(new) >= 2:
        result["current_within"] = correspondence(logits[new][:, new])
    cosine = logits * 0.07
    if len(old) and len(new):
        result["cross_stage"] = {
            "old_graph_to_current_text_max_mean": float(cosine[old][:, new].max(1).values.mean()),
            "current_graph_to_old_text_max_mean": float(cosine[new][:, old].max(1).values.mean()),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-dir", type=Path, required=True)
    parser.add_argument("--teacher-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", type=int, default=2)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    saved = torch.load(args.teacher_model, map_location="cpu", weights_only=True)
    if saved.get("training_labels_used"):
        raise ValueError("teacher model claims labels were used")
    model = create_model(**saved["model_config"]).to(args.device)
    model.load_state_dict(saved["state_dict"])
    paths = []
    for path in args.trajectory_dir.glob("*_outer_*.pt"):
        match = STEP_RE.match(path.name)
        if match:
            paths.append((int(match.group(1)), path))
    paths.sort()
    if not paths or paths[0][0] != 0:
        raise FileNotFoundError("trajectory must contain an outer-step 0 artifact")

    initial = torch.load(paths[0][1], map_location="cpu", weights_only=True)
    fixed = {key: tensor_sha(initial[key]) for key in DATA_KEYS if key != "x"}
    rows = []
    for step, path in paths:
        artifact = torch.load(path, map_location="cpu", weights_only=True)
        if any(tensor_sha(artifact[key]) != digest for key, digest in fixed.items()):
            raise RuntimeError(f"fixed field changed at outer step {step}")
        rows.append({"outer_step": step, "artifact": path.name,
                     "x_sha256": tensor_sha(artifact["x"]),
                     "metrics": analyze(model, artifact, args.stage)})
    output = {
        "protocol": "label_free_text_alignment_trajectory_v1",
        "teacher_model": str(args.teacher_model),
        "stage": args.stage,
        "fixed_field_sha256": fixed,
        "labels_loaded": False,
        "rows": rows,
    }
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
