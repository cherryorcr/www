"""Check completeness, source memory hashes and label-free selection decisions."""
import argparse
import json
import math
from pathlib import Path

import run_small_comparison_audited as C
from tgc_r2_response import select_indices


def audit(root):
    errors = []
    trajectories = 0
    stages = 0
    for dataset in ("cora", "citeseer"):
        summary = json.loads((root / dataset / "summary.json").read_text())
        protocol = summary["protocol"]
        if (protocol["selection_seeds"] != [700, 701]
                or protocol["horizons"] != [32, 64, 128]
                or protocol["labels_used_for_selection"]):
            errors.append(f"selection protocol mismatch: {dataset}")
        observed = {(r["order"], r["stage"]) for r in summary["results"]}
        if observed != {(o, s) for o in ("canonical", "order19101") for s in (1, 2)}:
            errors.append(f"stage set incomplete: {dataset}")
        for result in summary["results"]:
            stages += 1
            path = root / dataset / result["order"] / f"s{result['stage']}" / "label_free_decisions.json"
            saved = json.loads(path.read_text())
            if C.sha(path) != result["decisions_sha256"] or saved["report_labels_used"]:
                errors.append(f"decision evidence mismatch: {path}")
            if not all(p["all_prefixes_exact"] for p in saved["prefix_audits"]):
                errors.append(f"shared-prefix numerical equivalence failed: {path}")
            candidates = saved["candidates"]
            if [c["name"] for c in candidates] != protocol["configs"]:
                errors.append(f"candidate registry mismatch: {path}")
            support_hashes = set()
            initial_hashes = {700: set(), 701: set()}
            for candidate in candidates:
                arm = "c0" if candidate["name"] == "c0" else "tgc_r"
                source = Path(candidate["cell"]) / f"{arm}_s{result['stage']}_construction.json"
                construction = json.loads(source.read_text())
                if (C.sha(source) != candidate["construction_sha256"]
                        or construction["memory_sha256"] != candidate["memory_sha256"]):
                    errors.append(f"source mismatch: {source}")
                if [r["seed"] for r in candidate["responses"]] != [700, 701]:
                    errors.append(f"response seed mismatch: {path}")
                for response in candidate["responses"]:
                    trajectories += 1
                    support_hashes.add(response["probe_sha256"])
                    initial_hashes[response["seed"]].add(response["start_state_sha256"])
                    if (response["memory_sha256"] != candidate["memory_sha256"]
                            or response["report_labels_used"]
                            or not response["training_state_unchanged_by_probe"]):
                        errors.append(f"response contract mismatch: {path}")
                    if [r["step"] for r in response["trajectory"]] != [32, 64, 128]:
                        errors.append(f"response trajectory incomplete: {path}")
                    for row in response["trajectory"]:
                        if not all(math.isfinite(row[k]) for k in ("old_loss", "new_loss", "balanced_loss", "train_loss")):
                            errors.append(f"nonfinite response: {path}")
            if len(support_hashes) != 1 or any(len(v) != 1 for v in initial_hashes.values()):
                errors.append(f"candidates used different probe or initialization: {path}")
            for decision in saved["decisions"]:
                subset = [candidates[i] for i in decision["indices"]]
                horizon = None if decision["horizon"] == "curve" else decision["horizon"]
                expected = select_indices(subset, horizon)
                for policy, selected in decision["chosen"].items():
                    if selected != decision["indices"][expected[policy]]:
                        errors.append(f"choice not reproduced: {path} {policy}")
    return {"passed": not errors and stages == 8 and trajectories == 144,
            "errors": errors, "stage_cells": stages, "selection_trajectories": trajectories,
            "extra_prefix_check_trajectories": stages,
            "scope": "artifact and numerical checks; no sequential selector performance claim"}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    args = ap.parse_args()
    result = audit(args.root)
    (args.root / "audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))
    if not result["passed"]:
        raise SystemExit(1)
