"""One-time confirmation of already selected, hash-frozen student models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from native_core import create_model
from run_native_experiments import Experiment, dump, file_sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    frozen = []
    for run in args.runs:
        if not (run / "completed.json").exists():
            raise ValueError(f"incomplete development run: {run}")
        config = json.loads((run / "run_config.json").read_text())
        results = json.loads((run / "results.json").read_text())
        for row in results["results"]:
            artifact_path = run / f"{row['arm']}_s{row['stage']}_artifact.pt"
            if file_sha(artifact_path) != row["artifact_sha256"]:
                raise ValueError(f"artifact registration mismatch: {artifact_path}")
            for model_run in row["model_runs"]:
                model_path = run / f"{row['arm']}_s{row['stage']}_m{model_run['seed']}_model.pt"
                frozen.append({"run": str(run), "rate": config["rate"], "condensation_seed": config["seed"],
                               "stage": row["stage"], "arm": row["arm"], "student_seed": model_run["seed"],
                               "artifact_path": str(artifact_path), "artifact_sha256": row["artifact_sha256"],
                               "model_path": str(model_path), "model_sha256": file_sha(model_path),
                               "selected_step": model_run["selected_step"],
                               "checkpoint_f1": model_run["checkpoint"]["macro_f1"],
                               "prepared": config["prepared"],
                               "source_manifest_sha256": config["source_manifest_sha256"],
                               "run_config_sha256": file_sha(run / "run_config.json")})
    # Persist the entire candidate list before computing any confirmation metric.
    dump(args.output / "frozen_candidates.json",
         {"purpose": "development confirmation; Cora is previously used, not a pristine final test",
          "no_retraining_or_reselection": True,
          "evaluation_source_sha256": {p.name: file_sha(p) for p in Path(__file__).parent.glob("*.py")},
          "models": frozen})
    evaluations = {}
    rows = []
    for item in frozen:
        prepared = Path(item["prepared"])
        if item["prepared"] not in evaluations:
            exp = Experiment.__new__(Experiment)
            exp.prepared = prepared
            exp.device = args.device
            exp.evals, exp.counts = {}, []
            if file_sha(prepared / "manifest.json") != item["source_manifest_sha256"]:
                raise ValueError("prepared manifest changed after development")
            exp.manifest = json.loads((prepared / "manifest.json").read_text())
            exp.verify_input("class_text_emb.pt")
            exp.classes = torch.load(prepared / "class_text_emb.pt", weights_only=True, map_location="cpu")
            evaluations[item["prepared"]] = exp
        exp = evaluations[item["prepared"]]
        for stage in range(item["stage"] + 1):
            if stage not in exp.evals:
                exp.load_stage(stage)
        if file_sha(item["model_path"]) != item["model_sha256"]:
            raise ValueError("model changed after confirmation freeze")
        checkpoint = torch.load(item["model_path"], weights_only=True, map_location="cpu")
        artifact = torch.load(item["artifact_path"], weights_only=True, map_location="cpu")
        if not torch.equal(checkpoint["train_origin_ids"], artifact["origin_ids"]):
            raise ValueError("model and training artifact identity differs")
        if checkpoint["training_labels_used"] or checkpoint["stage"] != item["stage"]:
            raise ValueError("model provenance contradicts protocol")
        model = create_model(**checkpoint["model_config"]).to(args.device)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.eval()
        replay = exp.metric(model, item["stage"], "checkpoint")
        if abs(replay["macro_f1"] - item["checkpoint_f1"]) > 1e-6:
            raise ValueError(f"saved model does not reproduce checkpoint score: {item['model_path']}")
        rows.append({**item, "checkpoint_replay_f1": replay["macro_f1"],
                     "confirmation": exp.metric(model, item["stage"], "test")})
        del model
    grouped = []
    keys = sorted({(r["run"], r["stage"], r["arm"]) for r in rows})
    for run, stage, arm in keys:
        items = [r for r in rows if (r["run"], r["stage"], r["arm"]) == (run, stage, arm)]
        grouped.append({"run": Path(run).name, "stage": stage, "arm": arm,
                        "rate": items[0]["rate"], "condensation_seed": items[0]["condensation_seed"],
                        "checkpoint_f1": float(np.mean([r["checkpoint_f1"] for r in items])),
                        "confirmation_f1": float(np.mean([r["confirmation"]["macro_f1"] for r in items])),
                        "confirmation_accuracy": float(np.mean([r["confirmation"]["accuracy"] for r in items])),
                        "per_stage_f1": [float(np.mean([r["confirmation"]["per_stage"][j]["macro_f1"] for r in items]))
                                         for j in range(stage + 1)]})
    dump(args.output / "confirmation.json", {"development_only": True, "confirmation_accessed": True,
                                             "no_retraining_or_reselection": True, "models": rows, "summary": grouped})
    dump(args.output / "completed.json", {"status": "complete", "models": len(rows), "comparisons": len(grouped)})
    print(json.dumps({"models": len(rows), "comparisons": len(grouped),
                      "frozen_candidates_sha256": file_sha(args.output / "frozen_candidates.json")}))


if __name__ == "__main__":
    main()
