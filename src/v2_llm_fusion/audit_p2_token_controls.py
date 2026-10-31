"""Audit completeness, artifact invariants, and paired-seed P2 effects."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from scipy.stats import t as student_t

from p2_text_operators import aggregate_audit, token_count


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def effect(values, reference):
    delta = np.asarray(values) - np.asarray(reference)
    n = len(delta)
    se = float(np.std(delta, ddof=1) / math.sqrt(n)) if n > 1 else float("inf")
    # Paired initialization uncertainty only, not a class-order interval.
    half = float(student_t.ppf(0.975, n - 1)) * se if n > 1 else float("inf")
    return {"paired_gain": float(delta.mean()), "ci_low": float(delta.mean() - half),
            "ci_high": float(delta.mean() + half), "positive_seeds": int((delta > 0).sum())}


def audit_dataset(root: Path, check_artifacts: bool, tokenizer=None) -> dict:
    runs = {}
    errors = []
    for path in sorted(root.glob("*/results.json")):
        d = json.loads(path.read_text())
        runs[path.parent.name] = (d, path)
        if d.get("test_accessed") is not False or d.get("selector_used") is not False:
            errors.append(f"{path.parent.name}: test access or selector state")
        if len(d["results"]) != len(d["stage_classes"]):
            errors.append(f"{path.parent.name}: incomplete stages")
        if len(d["student_seeds"]) != len(set(d["student_seeds"])):
            errors.append(f"{path.parent.name}: duplicate student seeds")
        if check_artifacts:
            ref_root = Path(d["reference"])
            if digest(ref_root / "results.json") != d["reference_results_sha256"]:
                errors.append(f"{path.parent.name}: reference changed")
            old = None
            for stage, row in enumerate(d["results"]):
                memory = torch.load(path.parent / f"append_fixed_s{stage}_memory.pt",
                                    map_location="cpu", weights_only=False)
                ref_file = ref_root / f"append_fixed_s{stage}_memory.pt"
                ref = torch.load(ref_file, map_location="cpu", weights_only=False)
                if digest(ref_file) != row["reference_memory_sha256"]:
                    errors.append(f"{path.parent.name}/s{stage}: reference hash")
                for key in ("x", "mass", "depth", "stage", "edge_index"):
                    if not torch.equal(memory[key], ref[key]):
                        errors.append(f"{path.parent.name}/s{stage}: changed {key}")
                if old is not None:
                    k = len(old["x"])
                    if not torch.equal(memory["text_emb"][:k], old["text_emb"]):
                        errors.append(f"{path.parent.name}/s{stage}: historical embeddings changed")
                    if memory["texts"][:k] != old["texts"]:
                        errors.append(f"{path.parent.name}/s{stage}: historical text changed")
                    mask = (memory["edge_index"] < k).all(0)
                    if not torch.equal(memory["edge_index"][:, mask], old["edge_index"]):
                        errors.append(f"{path.parent.name}/s{stage}: historical edges changed")
                if row["text_audit"] != aggregate_audit(memory["text_input_audit"]):
                    errors.append(f"{path.parent.name}/s{stage}: audit mismatch")
                if tokenizer is not None:
                    for i, a in enumerate(memory["text_input_audit"]):
                        content = sum(token_count(tokenizer, t) for t in a["inputs"])
                        retained = token_count(tokenizer, memory["texts"][i])
                        with_special = sum(len(tokenizer.encode(t, add_special_tokens=True))
                                           for t in a["inputs"])
                        input_hash = hashlib.sha256("\0".join(a["inputs"]).encode()).hexdigest()
                        if (content != a["encoder_content_tokens"] or retained != a["retained_tokens"]
                                or with_special != a["encoder_tokens_with_special"]
                                or input_hash != a["input_sha256"]
                                or max(content, retained) > a["content_token_cap"]):
                            errors.append(f"{path.parent.name}/s{stage}/row{i}: actual tokenizer audit")
                if not torch.isfinite(memory["text_emb"]).all():
                    errors.append(f"{path.parent.name}/s{stage}: nonfinite target")
                old = memory
        for row in d["results"]:
            if row["text_audit"]["budget_violations"]:
                errors.append(f"{path.parent.name}/s{row['stage']}: token overflow")
            if row["student_seeds"] != d["student_seeds"]:
                errors.append(f"{path.parent.name}/s{row['stage']}: seed mismatch")
    baseline = runs.get("concat_t64_g5_r0.5_a0.5")
    if baseline is None:
        return {"errors": errors + ["missing baseline"], "rows": [], "complete_runs": len(runs)}
    base = baseline[0]
    base_seed = [r["cumulative_macro_f1"] for r in base["results"][-1]["per_student"]]
    rows = []
    for name, (d, _) in runs.items():
        last = d["results"][-1]
        for stage, r in enumerate(d["results"]):
            b = base["results"][stage]
            if any(r[key] != b[key] for key in ("roots", "student_steps", "student_batch",
                                               "keep_old_rows", "new_rows", "reference_memory_sha256")):
                errors.append(f"{name}/s{stage}: paired protocol mismatch")
        per_seed = [r["cumulative_macro_f1"] for r in last["per_student"]]
        forgetting = []
        for seed_i in range(len(per_seed)):
            losses = []
            for old_stage in range(len(d["results"]) - 1):
                best = max(r["per_student"][seed_i]["per_stage"][old_stage]["macro_f1"]
                           for r in d["results"][old_stage:-1])
                final_old = last["per_student"][seed_i]["per_stage"][old_stage]["macro_f1"]
                losses.append(best - final_old)
            forgetting.append(float(np.mean(losses)) if losses else 0.0)
        rows.append({"dataset": d["dataset"], "name": name,
            "final_f1": last["cum_dev_f1_mean"], "final_std": last["cum_dev_f1_std"],
            "stage_average_f1": float(np.mean([r["cum_dev_f1_mean"] for r in d["results"]])),
            "old_f1": last["old_mean_f1_mean"], "new_f1": last["current_f1_mean"],
            "forgetting": float(np.mean(forgetting)),
            "train_seconds": sum(r["student_train_seconds"] for r in d["results"]),
            "build_seconds": sum(r["build_seconds"] for r in d["results"]),
            "content_tokens": last["text_audit"]["encoder_content_token_total"],
            "with_special_tokens": last["text_audit"]["encoder_tokens_with_special_total"],
            "encoder_sequences": last["text_audit"]["encoder_sequences"],
            **effect(per_seed, base_seed)})
    return {"errors": errors, "complete_runs": len(runs), "rows": rows,
            "artifact_checks_executed": check_artifacts,
            "training_runs": sum(sum(r["student_training_runs"] for r in d["results"])
                                  for d, _ in runs.values())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--check-artifacts", action="store_true")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--expected-runs", type=int, default=17)
    args = ap.parse_args()
    root = Path(args.root)
    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    reports = {name: audit_dataset(root / name, args.check_artifacts, tokenizer)
               for name in ("cora", "citeseer")}
    rows = [r for report in reports.values() for r in report["rows"]]
    passed = all(not r["errors"] and r["complete_runs"] == args.expected_runs for r in reports.values())
    names = set(r["name"] for r in rows if "_t64_g5_" in r["name"])
    positive = [name for name in sorted(names)
                if len([r for r in rows if r["name"] == name]) == 2
                and all(r["ci_low"] > 0 for r in rows if r["name"] == name)]
    report = {"integrity_passed": passed, "datasets": reports,
              "actual_tokenizer_checks": tokenizer is not None,
              "candidates_for_independent_confirmation": positive,
              "large_graph_gate_passed": False,
              "interpretation": "Exploratory paired-seed dev results; confirmation required before promotion."}
    (root / "audit.json").write_text(json.dumps(report, indent=2))
    if rows:
        with (root / "summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({"integrity_passed": passed,
        "counts": {k: v["complete_runs"] for k, v in reports.items()},
        "errors": {k: v["errors"] for k, v in reports.items()},
        "candidates_for_independent_confirmation": positive}))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
