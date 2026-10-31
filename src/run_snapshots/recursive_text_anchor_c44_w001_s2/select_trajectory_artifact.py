"""Select and export one fixed-outer trajectory artifact using development data only.

`run_outer_trajectory.py` deliberately writes every fixed outer snapshot before
model selection.  This helper performs the separate, auditable selection step
and restores the prepared-manifest provenance required by the next recursive
stage.  Test and future metrics are never read.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from run_native_experiments import file_sha


ALLOWED_METRICS = {
    "selected_student_mean_macro_f1",
    "endpoint_student_mean_macro_f1",
}


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def select_row(trajectory: dict[str, Any], metric: str) -> dict[str, Any]:
    if metric not in ALLOWED_METRICS:
        raise ValueError(f"selection metric must be one of {sorted(ALLOWED_METRICS)}")
    rows = trajectory.get("trajectory")
    if not isinstance(rows, list) or not rows:
        raise ValueError("trajectory contains no candidate rows")
    for row in rows:
        if metric not in row or "outer_step" not in row or "artifact" not in row:
            raise ValueError("trajectory row lacks selection fields")
    # A lower outer step wins an exact tie.  It is the less transformed memory.
    return max(rows, key=lambda row: (float(row[metric]), -int(row["outer_step"])))


def export_selected(run_dir: Path, output: Path, metric: str) -> dict[str, Any]:
    trajectory_path = run_dir / "trajectory.json"
    diagnostic_path = run_dir / "diagnostic_inputs.json"
    completed_path = run_dir / "completed.json"
    trajectory = load_json(trajectory_path)
    diagnostic = load_json(diagnostic_path)
    completed = load_json(completed_path)

    if completed.get("status") != "complete":
        raise ValueError("trajectory run is not complete")
    if diagnostic.get("confirmation_accessed") is not False:
        raise ValueError("selection input is not marked development-only")
    if diagnostic.get("training_labels_used") is not False:
        raise ValueError("trajectory claims labels were used for training")
    if diagnostic.get("outer_selector") != "disabled; fixed pre-registered snapshots":
        raise ValueError("trajectory was not generated with the fixed-snapshot protocol")

    selected = select_row(trajectory, metric)
    artifact_path = run_dir / str(selected["artifact"])
    expected_sha = str(selected.get("artifact_sha256", ""))
    if not artifact_path.is_file():
        raise FileNotFoundError(f"missing selected artifact: {artifact_path}")
    if expected_sha and file_sha(artifact_path) != expected_sha:
        raise ValueError("selected artifact hash differs from trajectory record")

    artifact = torch.load(artifact_path, map_location="cpu", weights_only=True)
    if not isinstance(artifact, dict):
        raise TypeError("selected artifact must be a graph dictionary")
    if {"y", "labels"}.intersection(artifact):
        raise ValueError("selected artifact contains forbidden labels")
    stage = int(diagnostic["stage"])
    observed_stage = artifact.get("stage")
    if not isinstance(observed_stage, torch.Tensor) or observed_stage.numel() == 0:
        raise ValueError("selected artifact has no stage tensor")
    if int(observed_stage.min()) < 0 or int(observed_stage.max()) != stage:
        raise ValueError("selected artifact does not end at the requested stage")

    metadata = dict(artifact.get("metadata", {}))
    if metadata.get("training_labels_used") not in (None, False):
        raise ValueError("selected artifact metadata claims labels were used")
    metadata.update({
        "source_manifest_sha256": str(diagnostic["prepared_manifest_sha256"]),
        "stage": stage,
        "training_labels_used": False,
        "selection": {
            "protocol": "fixed_outer_development_selection_v1",
            "metric": metric,
            "value": float(selected[metric]),
            "outer_step": int(selected["outer_step"]),
            "candidate_outer_steps": [int(value) for value in trajectory["outer_steps"]],
            "student_seeds": [int(value) for value in diagnostic["model_seeds"]],
            "test_or_future_used": False,
            "source_artifact": artifact_path.name,
            "source_artifact_sha256": file_sha(artifact_path),
            "trajectory_sha256": file_sha(trajectory_path),
        },
    })
    exported = dict(artifact)
    exported["metadata"] = metadata
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"selected export already exists: {output}")
    torch.save(exported, output)

    selection = {
        "protocol": "fixed_outer_development_selection_v1",
        "run_dir": str(run_dir.resolve()),
        "trajectory_sha256": file_sha(trajectory_path),
        "diagnostic_inputs_sha256": file_sha(diagnostic_path),
        "selection_metric": metric,
        "test_or_future_used_for_selection": False,
        "selected_outer_step": int(selected["outer_step"]),
        "selected_metric_value": float(selected[metric]),
        "source_artifact": str(artifact_path.resolve()),
        "source_artifact_sha256": file_sha(artifact_path),
        "exported_artifact": str(output.resolve()),
        "exported_artifact_sha256": file_sha(output),
        "stage": stage,
        "student_seeds": [int(value) for value in diagnostic["model_seeds"]],
    }
    selection_path = output.with_suffix(".selection.json")
    selection_path.write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    return selection


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--metric", choices=sorted(ALLOWED_METRICS),
                        default="selected_student_mean_macro_f1")
    args = parser.parse_args()
    run_dir = Path(args.run_dir).resolve()
    output = Path(args.output).resolve() if args.output else run_dir / "selected_dev_artifact.pt"
    result = export_selected(run_dir, output, args.metric)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
