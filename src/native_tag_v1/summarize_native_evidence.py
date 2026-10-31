"""Export compact, hash-checked summaries from immutable native runs."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch

from native_core import _fingerprint
from run_native_experiments import dump, file_sha


def mean(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def summarize(root, output):
    output.mkdir(parents=True, exist_ok=True)
    if (output / "evidence.json").exists():
        raise FileExistsError("use a new evidence output directory")
    all_rows, audits, inputs = [], [], []
    for run in sorted(root.iterdir()):
        if not (run / "completed.json").exists() or not (run / "results.json").exists():
            continue
        config = json.loads((run / "run_config.json").read_text())
        results = json.loads((run / "results.json").read_text())
        if results["confirmation_accessed"]:
            raise ValueError(f"confirmation was accessed: {run.name}")
        actual_arms = sorted({r["arm"] for r in results["results"]})
        audit = {"run": run.name, "results_sha256": file_sha(run / "results.json"),
                 "source_manifest_sha256": config["source_manifest_sha256"],
                 "configured_arms": config["arms"], "actual_arms": actual_arms,
                 "arms_metadata_matches": set(config["arms"]) == set(actual_arms),
                 "source_code_hashes_match": all(file_sha(run / "code" / name) == sha
                                                 for name, sha in config["source_sha256"].items()),
                 "artifacts": [], "condensation": []}
        if not audit["source_code_hashes_match"]:
            raise ValueError(f"code snapshot hash mismatch: {run}")
        for row in results["results"]:
            path = run / f"{row['arm']}_s{row['stage']}_artifact.pt"
            artifact = torch.load(path, map_location="cpu", weights_only=True)
            checked = file_sha(path) == row["artifact_sha256"]
            if not checked:
                raise ValueError(f"artifact hash mismatch: {path}")
            meta = artifact.get("metadata", {})
            artifact_audit = {"arm": row["arm"], "stage": row["stage"], "sha256_matches": checked,
                              "labels_absent": "y" not in artifact and "labels" not in artifact,
                              "selected_outer_step": meta.get("selected_outer_step"),
                              "reference_audit": meta.get("reference_audit"),
                              "fixed_fields_match": all(_fingerprint(artifact[k]) == h
                                                        for k, h in meta.get("fixed_field_sha256", {}).items())}
            if "optimized_x_rows" in meta:
                mask = torch.tensor(meta["optimized_x_rows"], dtype=torch.bool)
                artifact_audit["protected_feature_hash_matches"] = _fingerprint(artifact["x"][~mask]) == meta["frozen_x_sha256"]
            audit["artifacts"].append(artifact_audit)
            models = row["model_runs"]
            compact = {"run": run.name, "rate": config["rate"], "condensation_seed": config["seed"],
                       "stage": row["stage"], "arm": row["arm"], "roots": row["roots"],
                       "artifact_bytes": row["artifact_bytes"], "train_denominator": row["train_denominator"],
                       "access": row["access"], "macro_f1": row["mean_checkpoint_f1"],
                       "accuracy": mean([m["checkpoint"]["accuracy"] for m in models]),
                       "old_mean_f1": mean([m["checkpoint"]["old_mean_f1"] for m in models]),
                       "current_f1": mean([m["checkpoint"]["current_f1"] for m in models]),
                       "student_seeds": [m["seed"] for m in models],
                       "student_f1": [m["checkpoint"]["macro_f1"] for m in models],
                       "selected_student_steps": [m["selected_step"] for m in models],
                       "endpoint_f1": [m["endpoint_checkpoint"]["macro_f1"] for m in models],
                       "step0_f1": [next((v["f1"] for v in m["curve"] if v["step"] == 0), None) for m in models],
                       "selected_outer_step": meta.get("selected_outer_step"),
                       "per_stage_f1": [mean([m["checkpoint"]["per_stage"][j]["macro_f1"] for m in models])
                                        for j in range(row["stage"] + 1)]}
            all_rows.append(compact)
        for path in sorted(run.glob("*_condense.json")):
            log = json.loads(path.read_text())
            outer = [r for r in log["logs"] if r["event"] == "outer"]
            selected = next(r for r in log["logs"] if r["event"] == "selected")
            audit["condensation"].append({"file": path.name, "selected_outer_step": selected["outer_step"],
                                          "status": selected["status"], "outer_updates": len(outer),
                                          "finite_nonzero_x_gradients": all(math.isfinite(r["x_grad_l2"]) and r["x_grad_l2"] > 0 for r in outer),
                                          "validation_outer_f1": [(v["outer_step"], v["best_checkpoint_f1"], v["selected_step"])
                                                                  for v in log["validations"]]})
        source_path = run / "diagnostic_inputs.json"
        if source_path.exists():
            source = json.loads(source_path.read_text())
            anchor = Path(source["anchor_artifact"])
            registration = json.loads((anchor.parent / "results.json").read_text())
            original = next(r for r in registration["results"] if r["arm"] == "offline_asm" and r["stage"] == 0)
            matches = file_sha(anchor) == source["anchor_sha256"] == original["artifact_sha256"]
            if not matches:
                raise ValueError(f"anchor hash differs from registered artifact: {run}")
            inputs.append({"run": run.name, "anchor": str(anchor), "sha256_matches_current_and_registered": matches})
        audits.append(audit)
    evidence = {"development_only": True, "confirmation_accessed": False,
                "independent_student_seeds_are_not_independent_condensation_seeds": True,
                "rows": all_rows, "run_audits": audits, "anchor_audits": inputs}
    dump(output / "evidence.json", evidence)
    fields = ["run", "rate", "condensation_seed", "stage", "arm", "roots", "macro_f1", "accuracy",
              "old_mean_f1", "current_f1", "artifact_bytes", "selected_outer_step", "selected_student_steps"]
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: r[k] for k in fields} for r in all_rows)
    print(json.dumps({"runs": len(audits), "result_rows": len(all_rows), "anchor_inputs_checked": len(inputs),
                      "code_hashes_valid": all(a["source_code_hashes_match"] for a in audits),
                      "artifact_hashes_valid": all(a["sha256_matches"] for r in audits for a in r["artifacts"]),
                      "historical_arm_metadata_mismatches": [r["run"] for r in audits if not r["arms_metadata_matches"]]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.root, args.output)
