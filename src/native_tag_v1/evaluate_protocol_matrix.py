"""Evaluate frozen native TAG students on the full stage/partition matrix.

This is a read-only development evaluator.  It never selects a checkpoint and
never passes labels back to training or condensation.  Saved selected student
models are evaluated on:

* each original stage-train induced graph;
* each arrived stage's checkpoint and test masks;
* each not-yet-arrived stage's test mask as a global-ontology future probe.

Checkpoint/test/future evaluation uses the prepared whole-stage graph, matching
the current transductive TAGSAM protocol.  The train diagnostic uses only the
stage-train induced graph, so held-out nodes are absent from its message graph.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

from native_core import create_model, evaluate_logits


MODEL_RE = re.compile(r"^(?P<arm>.+)_s(?P<stage>\d+)_m(?P<seed>\d+)_model\.pt$")


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def score(y: np.ndarray, pred: np.ndarray, labels: list[int]) -> dict:
    if len(y) == 0:
        return {"macro_f1": None, "accuracy": None, "n": 0, "support": {}}
    return {
        "macro_f1": 100.0 * float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)),
        "accuracy": 100.0 * float(accuracy_score(y, pred)),
        "n": int(len(y)),
        "support": {str(int(label)): int((y == label).sum()) for label in labels},
    }


def discover_models(runs: list[Path], arms: set[str] | None) -> list[dict]:
    found = []
    for run in runs:
        if not (run / "run_config.json").exists():
            raise FileNotFoundError(f"missing run_config.json: {run}")
        for path in sorted(run.glob("*_model.pt")):
            if path.name.startswith("teacher_"):
                continue
            match = MODEL_RE.match(path.name)
            if not match:
                continue
            item = {"run": run, "path": path, "arm": match.group("arm"),
                    "stage": int(match.group("stage")), "seed": int(match.group("seed"))}
            if arms is None or item["arm"] in arms:
                found.append(item)
    if not found:
        raise FileNotFoundError("no matching saved student models")
    return found


def load_prepared(prepared: Path) -> tuple[dict, torch.Tensor, list[dict]]:
    manifest_path = prepared / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for filename, metadata in manifest["files"].items():
        path = prepared / filename
        if file_sha(path) != metadata["sha256"]:
            raise ValueError(f"prepared input hash mismatch: {filename}")
    classes = torch.load(prepared / "class_text_emb.pt", map_location="cpu", weights_only=True)
    stages = []
    stage_count = len(manifest["stage_classes"])
    for stage in range(stage_count):
        train = torch.load(prepared / f"stage_{stage}_train.pt", map_location="cpu", weights_only=False)
        evaluation = torch.load(prepared / f"stage_{stage}_eval.pt", map_location="cpu", weights_only=False)
        if "y" in train or "labels" in train:
            raise ValueError(f"stage {stage} training graph contains labels")
        eval_lookup = {int(origin): int(label) for origin, label in
                       zip(evaluation["origin_ids"].tolist(), evaluation["y"].tolist())}
        train_y = np.asarray([eval_lookup[int(origin)] for origin in train["origin_ids"].tolist()], dtype=np.int64)
        stages.append({"train": train, "train_y": train_y, "evaluation": evaluation})
    return manifest, classes, stages


def evaluate_model(model, classes: torch.Tensor, stages: list[dict], model_stage: int,
                   stage_classes: list[list[int]]) -> dict:
    cells, seen_parts = [], {name: {"y": [], "pred": []} for name in ("train", "checkpoint", "test")}
    future = {"y": [], "pred": []}
    for eval_stage, item in enumerate(stages):
        cell_labels = [int(label) for label in stage_classes[eval_stage]]
        if eval_stage <= model_stage:
            train_pred = evaluate_logits(model, item["train"], classes).argmax(1).cpu().numpy()
            train_y = item["train_y"]
            metric = score(train_y, train_pred, cell_labels)
            cells.append({"eval_stage": eval_stage, "partition": "train", "role": "arrived_train_fit",
                          "message_graph": "train-induced", "metric": metric})
            seen_parts["train"]["y"].append(train_y)
            seen_parts["train"]["pred"].append(train_pred)

            evaluation = item["evaluation"]
            full_pred = evaluate_logits(model, evaluation["graph"], classes).argmax(1).cpu().numpy()
            all_y = evaluation["y"].cpu().numpy()
            for partition in ("checkpoint", "test"):
                mask = evaluation["masks"][partition].cpu().numpy().astype(bool)
                y, pred = all_y[mask], full_pred[mask]
                metric = score(y, pred, cell_labels)
                cells.append({"eval_stage": eval_stage, "partition": partition,
                              "role": "arrived_dev" if partition == "checkpoint" else "arrived_test",
                              "message_graph": "whole-stage-transductive", "metric": metric})
                seen_parts[partition]["y"].append(y)
                seen_parts[partition]["pred"].append(pred)
        else:
            evaluation = item["evaluation"]
            full_pred = evaluate_logits(model, evaluation["graph"], classes).argmax(1).cpu().numpy()
            mask = evaluation["masks"]["test"].cpu().numpy().astype(bool)
            y, pred = evaluation["y"].cpu().numpy()[mask], full_pred[mask]
            metric = score(y, pred, cell_labels)
            cells.append({"eval_stage": eval_stage, "partition": "future", "role": "future_test_probe",
                          "message_graph": "whole-future-stage-transductive", "metric": metric})
            future["y"].append(y)
            future["pred"].append(pred)

    seen_labels = [int(label) for labels in stage_classes[:model_stage + 1] for label in labels]
    future_labels = [int(label) for labels in stage_classes[model_stage + 1:] for label in labels]
    cumulative_seen = {}
    for partition, values in seen_parts.items():
        cumulative_seen[partition] = score(
            np.concatenate(values["y"]), np.concatenate(values["pred"]), seen_labels)
    cumulative_future = (score(np.concatenate(future["y"]), np.concatenate(future["pred"]), future_labels)
                         if future["y"] else None)
    return {"cells": cells, "cumulative_seen": cumulative_seen,
            "cumulative_future_test": cumulative_future}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", default=None)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    manifest, classes, stages = load_prepared(args.prepared)
    models = discover_models(args.runs, set(args.arms) if args.arms else None)
    rows = []
    for item in models:
        saved = torch.load(item["path"], map_location="cpu", weights_only=True)
        if saved.get("training_labels_used"):
            raise ValueError(f"saved model claims labels were used: {item['path']}")
        if int(saved["stage"]) != item["stage"]:
            raise ValueError(f"stage mismatch in saved model: {item['path']}")
        model = create_model(**saved["model_config"]).to(args.device)
        model.load_state_dict(saved["state_dict"])
        model.eval()
        matrix = evaluate_model(model, classes, stages, item["stage"], manifest["stage_classes"])
        row = {
            "run": str(item["run"]), "run_config_sha256": file_sha(item["run"] / "run_config.json"),
            "model": item["path"].name, "model_sha256": file_sha(item["path"]),
            "arm": item["arm"], "model_stage": item["stage"], "student_seed": item["seed"],
            **matrix,
        }
        rows.append(row)
        print(json.dumps({"arm": item["arm"], "stage": item["stage"], "seed": item["seed"],
                          "seen_test_f1": matrix["cumulative_seen"]["test"]["macro_f1"],
                          "future_test_f1": (matrix["cumulative_future_test"] or {}).get("macro_f1")}), flush=True)
        del model

    output = {
        "protocol": "native_tag_v1_stage_partition_matrix_v1",
        "prepared": str(args.prepared),
        "prepared_manifest_sha256": file_sha(args.prepared / "manifest.json"),
        "global_class_ontology_visible": True,
        "checkpoint_selection_performed": False,
        "labels_used_for_evaluation_only": True,
        "future_inputs_used_for_training_or_condensation": False,
        "confirmation_status": "Cora test partitions were previously accessed; development evidence only",
        "train_context": "stage-train induced graph",
        "holdout_context": "whole-stage transductive graph; labels masked by partition",
        "future_context": "whole future-stage transductive graph; future data loaded only after frozen model",
        "rows": rows,
    }
    dump(args.output, output)


if __name__ == "__main__":
    main()
