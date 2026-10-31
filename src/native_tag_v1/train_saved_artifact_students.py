"""Retrain independent native TAG students from one saved synthetic artifact.

This runner performs no condensation and never selects an artifact.  It loads
one already exported, label-free synthetic TAG, validates its provenance, and
calls :meth:`Experiment.fit` once per requested model seed.  The selected
student checkpoints use the filename convention consumed by
``evaluate_protocol_matrix.py``.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from collections.abc import Mapping
from pathlib import Path

import torch

from native_core import DATA_KEYS, clean_data
from run_native_experiments import Experiment, dump, file_sha


LABEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
REQUIRED_FIT_CONFIG = (
    "threads", "student_steps", "rate", "batch_size", "lr", "hidden",
)


def load_json_object(path: Path, description: str) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{description} must contain a JSON object")
    return value


def validate_label(label: str) -> str:
    """Keep saved model names unambiguous and matrix-discoverable."""
    if not LABEL_RE.fullmatch(label) or label.startswith("teacher_"):
        raise ValueError(
            "label must be a filesystem-safe matrix arm name and must not start with teacher_"
        )
    return label


def validate_artifact(artifact: object, manifest_sha: str, stage: int) -> dict:
    """Validate a serialized graph before any output directory is created."""
    if not isinstance(artifact, Mapping):
        raise TypeError("artifact must be a mapping")
    forbidden = {"y", "labels"}.intersection(artifact)
    if forbidden:
        raise ValueError(f"artifact contains forbidden label fields: {sorted(forbidden)}")
    unexpected = set(artifact).difference((*DATA_KEYS, "metadata"))
    if unexpected:
        raise ValueError(f"artifact contains undeclared fields: {sorted(unexpected)}")

    metadata = artifact.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("artifact metadata is required for provenance validation")
    if metadata.get("training_labels_used") is not False:
        raise ValueError("artifact must explicitly state training_labels_used=false")
    artifact_manifest_sha = metadata.get("source_manifest_sha256")
    if artifact_manifest_sha is not None and artifact_manifest_sha != manifest_sha:
        raise ValueError("artifact belongs to a different prepared manifest")
    if "stage" in metadata and int(metadata["stage"]) != stage:
        raise ValueError("artifact metadata stage differs from the requested stage")

    # clean_data validates every declared tensor/string field without carrying
    # metadata or any unregistered payload into training.
    clean = clean_data(artifact, "cpu")
    observed_stages = clean["stage"]
    if bool((observed_stages < 0).any()) or int(observed_stages.max()) != stage:
        raise ValueError("artifact stage rows must be non-negative and end at the requested stage")
    if len(clean["x"]) < 2:
        raise ValueError("artifact needs at least two roots for InfoNCE training")
    return clean


def build_args(base: dict, *, output: Path, prepared: Path, artifact: Path,
               stage: int, label: str, device: str | None,
               model_seeds: list[int], check_every: int,
               source_config: Path, source_config_sha: str,
               artifact_sha: str) -> argparse.Namespace:
    missing = [key for key in REQUIRED_FIT_CONFIG if key not in base]
    if missing:
        raise ValueError(f"base config lacks Experiment.fit settings: {missing}")
    if check_every < 1:
        raise ValueError("check-every must be positive")
    seeds = [int(seed) for seed in model_seeds]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("model-seeds must be a nonempty list of unique integers")

    config = dict(base)
    config.update({
        "output": str(output),
        "prepared": str(prepared),
        "artifact": str(artifact),
        "artifact_sha256": artifact_sha,
        "artifact_stage": int(stage),
        "student_label": label,
        "source_config": str(source_config),
        "source_config_sha256": source_config_sha,
        "model_seeds": seeds,
        "device": device or base.get("device", "cuda:0"),
        "check_every": int(check_every),
        "confirm": False,
        "phase": "saved_artifact_student_retrain",
        "stages": int(stage) + 1,
        "arms": [],
        "diagnostic_mode": "fixed_saved_artifact_independent_students",
        "weighted": False,
    })
    return argparse.Namespace(**config)


def train_saved_artifact_students(args: argparse.Namespace) -> dict:
    """Validate inputs, reserve a new run directory, and train saved students."""
    base_path = Path(args.base_config).resolve()
    prepared = Path(args.prepared).resolve()
    artifact_path = Path(args.artifact).resolve()
    output = Path(args.output).resolve()
    validate_label(args.label)

    if not base_path.is_file():
        raise FileNotFoundError(f"missing base config: {base_path}")
    if not artifact_path.is_file():
        raise FileNotFoundError(f"missing artifact: {artifact_path}")
    manifest_path = prepared / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing prepared manifest: {manifest_path}")
    if output.exists():
        raise FileExistsError("choose a new output path; saved-artifact runs never overwrite")

    base = load_json_object(base_path, "base config")
    manifest = load_json_object(manifest_path, "prepared manifest")
    stage_count = len(manifest.get("stage_classes", []))
    if args.stage < 0 or args.stage >= stage_count:
        raise ValueError(f"stage must be in [0, {stage_count - 1}]")

    manifest_sha = file_sha(manifest_path)
    if base.get("source_manifest_sha256") != manifest_sha:
        raise ValueError("base config belongs to a different prepared manifest")
    if "stages" in base and int(base["stages"]) <= args.stage:
        raise ValueError("base config did not cover the requested stage")

    source_config_sha = file_sha(base_path)
    artifact_sha = file_sha(artifact_path)
    loaded = torch.load(artifact_path, map_location="cpu", weights_only=True)
    artifact = validate_artifact(loaded, manifest_sha, int(args.stage))

    fit_args = build_args(
        base,
        output=output,
        prepared=prepared,
        artifact=artifact_path,
        stage=int(args.stage),
        label=args.label,
        device=args.device,
        model_seeds=args.model_seeds,
        check_every=args.check_every,
        source_config=base_path,
        source_config_sha=source_config_sha,
        artifact_sha=artifact_sha,
    )

    # Reserve the exact destination atomically. Experiment may populate this
    # directory but cannot silently reuse an existing run.
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    exp = Experiment(fit_args)
    for stage in range(args.stage + 1):
        exp.load_stage(stage)

    inputs = {
        "mode": "fixed_saved_artifact_independent_students",
        "base_config": str(base_path),
        "base_config_sha256": source_config_sha,
        "source_config": str(base_path),
        "source_config_sha256": source_config_sha,
        "prepared": str(prepared),
        "prepared_manifest_sha256": manifest_sha,
        "artifact": str(artifact_path),
        "artifact_sha256": artifact_sha,
        "artifact_stage": int(args.stage),
        "artifact_roots": len(artifact["x"]),
        "artifact_stage_values": sorted(int(value) for value in artifact["stage"].unique()),
        "label": args.label,
        "model_seeds": [int(seed) for seed in args.model_seeds],
        "weighted": False,
        "training_labels_used": False,
        "confirmation_accessed": False,
    }
    dump(output / "artifact_inputs.json", inputs)

    runs = []
    for model_seed in args.model_seeds:
        seed = int(model_seed)
        model, info = exp.fit(
            artifact, int(args.stage), seed, label=args.label, weighted=False, save=True
        )
        model_path = output / f"{args.label}_s{args.stage}_m{seed}_model.pt"
        train_path = output / f"{args.label}_s{args.stage}_m{seed}_train.json"
        if not model_path.is_file() or not train_path.is_file():
            raise RuntimeError("Experiment.fit did not save the expected model and training log")
        runs.append({
            **info,
            "model": model_path.name,
            "model_sha256": file_sha(model_path),
            "train_log": train_path.name,
            "train_log_sha256": file_sha(train_path),
        })
        del model

    if file_sha(artifact_path) != artifact_sha:
        raise RuntimeError("input artifact changed while students were being trained")
    if file_sha(manifest_path) != manifest_sha:
        raise RuntimeError("prepared manifest changed while students were being trained")
    if file_sha(base_path) != source_config_sha:
        raise RuntimeError("base config changed while students were being trained")

    aggregate = {**inputs, "student_runs": runs, "elapsed_seconds": time.time() - started}
    dump(output / "student_runs.json", aggregate)
    completed = {
        "status": "complete",
        "mode": inputs["mode"],
        "stage": int(args.stage),
        "label": args.label,
        "model_seeds": inputs["model_seeds"],
        "models": [{key: run[key] for key in (
            "seed", "model", "model_sha256", "train_log", "train_log_sha256"
        )} for run in runs],
        "artifact_sha256": artifact_sha,
        "base_config_sha256": source_config_sha,
        "source_config_sha256": source_config_sha,
        "prepared_manifest_sha256": manifest_sha,
        "run_config_sha256": file_sha(output / "run_config.json"),
        "weighted": False,
        "training_labels_used": False,
        "confirmation_accessed": False,
        "elapsed_seconds": time.time() - started,
    }
    dump(output / "completed.json", completed)
    return completed


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", required=True,
                        help="run_config.json that supplies the native student hyperparameters")
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stage", type=int, required=True)
    parser.add_argument("--label", required=True,
                        help="matrix arm label used in saved model filenames")
    parser.add_argument("--device", default=None)
    parser.add_argument("--model-seeds", type=int, nargs="+", default=[100, 101])
    parser.add_argument("--check-every", type=int, default=32)
    return parser


def main(argv: list[str] | None = None) -> None:
    train_saved_artifact_students(make_parser().parse_args(argv))


if __name__ == "__main__":
    main()
