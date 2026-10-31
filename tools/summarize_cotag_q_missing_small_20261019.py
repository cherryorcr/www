"""Audit and export the missing small-graph baseline entries from saved runs."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import shutil
from statistics import mean, stdev
import sys


METHODS = {
    "tagsam_native_bounded": "TAGSAM bounded core",
    "stad_native_local": "STAD local-Qwen",
    "dance_adapter": "DANCE-inspired",
    "sr_gm_adapter": "SR-GM-inspired",
    "stad_adapter": "STAD-inspired",
}
INSPIRED = {"dance_adapter", "sr_gm_adapter", "stad_adapter"}
INVARIANTS = (
    "dataset", "groups", "budgets", "stage_raw_rows", "order_seed",
    "student_seeds", "student_batch", "encoder", "memory_rate",
    "actual_final_rate", "pool_mask_hash", "report_mask_hash", "test_mask_hash",
    "stage_id_hashes",
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def close(actual, expected):
    if not math.isclose(actual, expected, abs_tol=1e-6, rel_tol=1e-8):
        raise AssertionError((actual, expected))


def write_csv(path, records):
    if not records:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def verify_memory(cell, payload, validator):
    import torch
    parent = None
    artifacts = []
    for row in payload["rows"]:
        stage = row["stage"]
        block_path = cell / f"s{stage}" / "block.pt.gz"
        audit_path = cell / f"s{stage}" / "construction.json"
        with gzip.open(block_path, "rb") as stream:
            block = torch.load(stream, map_location="cpu", weights_only=False)
        if parent is None:
            memory = block
        else:
            memory = {k: torch.cat((parent[k], block[k])) for k in
                      ("x", "text_emb", "labels", "mass", "depth", "stage")}
            memory["texts"] = list(parent["texts"]) + list(block["texts"])
            memory["edge_index"] = torch.cat((parent["edge_index"],
                block["edge_index"] + len(parent["x"])), dim=1)
        checks = validator.validate_memory(memory, parent, row["memory_rows"])
        actual_hash = validator.memory_hash(memory)
        if not (actual_hash == row["memory_hash"] == read(audit_path)["memory_hash"]):
            raise AssertionError(f"memory/metric mismatch: {block_path}")
        if len(block["x"]) != row["new_rows"]:
            raise AssertionError(f"wrong appended block size: {block_path}")
        artifacts.append(dict(stage=stage, block_sha256=digest(block_path),
            construction_sha256=digest(audit_path), memory_hash=actual_hash, checks=checks))
        parent = memory
    return artifacts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-artifacts", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    original = args.root / "results/cotag_q_missing_baselines_20261019"
    repaired = args.root / "results/cotag_q_missing_baselines_20261019_repair_v2"
    matched = args.root / "results/cotag_q_small_9cell_20261019/cells"
    validator = None
    if args.verify_artifacts:
        sys.path.insert(0, str(args.root / "experiments/uq_final_evidence_20261012/code"))
        sys.path.insert(1, str(args.root / "code/v2_llm_fusion"))
        import run_unified_q_revision_20261007 as validator
    summary, students, stages, audit, missing = [], [], [], [], []
    sources = []
    for dataset in ("cora", "citeseer"):
        reference_path = matched / f"{dataset}_o9101_c271006/results.json"
        reference = read(reference_path)["protocol"]
        for arm, label in METHODS.items():
            cell = (original if arm in INSPIRED else repaired) / dataset / arm
            path = cell / "results.json"
            if not path.exists() or (payload := read(path)).get("status") != "complete":
                missing.append(f"{dataset}/{arm}")
                continue
            p = payload["protocol"]
            for key in INVARIANTS:
                if p[key] != reference[key]:
                    raise AssertionError(f"{dataset}/{arm}: mismatched {key}")
            for name, expected in reference["source_hashes"].items():
                if name in p["source_hashes"] and p["source_hashes"][name] != expected:
                    raise AssertionError(f"{dataset}/{arm}: changed source {name}")
            if p["official_reproduction"] or p["method_selection_from_test"]:
                raise AssertionError("unexpected fidelity or test-selection claim")
            expected_steps = 1024 if arm in INSPIRED else 2048
            if p["student_steps"] != expected_steps:
                raise AssertionError("unexpected Student budget")
            if arm not in INSPIRED:
                for key in ("student", "construction_seed"):
                    if p[key] != reference[key]:
                        raise AssertionError(f"common-Student mismatch: {key}")
            rows = payload["rows"]
            if [r["stage"] for r in rows] != list(range(len(p["groups"]))):
                raise AssertionError("incomplete or duplicated stages")
            for row in rows:
                if row["memory_rows"] != p["budgets"][row["stage"]]:
                    raise AssertionError("wrong memory budget")
                if row["checks"] != dict(budget_ok=True, prefix_ok=True, finite_ok=True):
                    raise AssertionError("recorded memory validation failed")
                for split, score in (row["scores"] or {}).items():
                    per_seed = score["per_seed"] if arm in INSPIRED else score["per_student"]
                    if [v["seed"] for v in per_seed] != p["student_seeds"]:
                        raise AssertionError("wrong or duplicated Student seeds")
                    for metric in ("macro_f1" if arm in INSPIRED else "final_f1", "accuracy"):
                        values = [v[metric] for v in per_seed]
                        close(mean(values), score[metric])
                        close(stdev(values), score[metric + "_std"])
                    for v in per_seed:
                        students.append(dict(dataset=dataset, method=label, arm=arm,
                            stage=row["stage"], split=split, student_seed=v["seed"],
                            macro_f1=v.get("final_f1", v.get("macro_f1")), accuracy=v["accuracy"]))
                    stages.append(dict(dataset=dataset, arm=arm, stage=row["stage"], split=split,
                        f1=score.get("final_f1", score.get("macro_f1")), accuracy=score["accuracy"]))
            split = "report_dev" if arm in INSPIRED else "sealed_test"
            selected = [v for v in students if v["dataset"] == dataset and v["arm"] == arm
                        and v["stage"] == len(rows) - 1 and v["split"] == split]
            summary.append(dict(dataset=dataset, arm=arm, method=label, split=split,
                n_order_construction_pairs=1, n_student_seeds=3, student_steps=expected_steps,
                f1_mean=mean(v["macro_f1"] for v in selected),
                f1_student_sd=stdev(v["macro_f1"] for v in selected),
                accuracy_mean=mean(v["accuracy"] for v in selected),
                accuracy_student_sd=stdev(v["accuracy"] for v in selected),
                construction_seconds=sum(v["construction_seconds"] for v in rows),
                peak_allocated_mib=max(v["construction_peak_mb"] for v in rows),
                rank_with_common_student=arm not in INSPIRED))
            item = dict(dataset=dataset, arm=arm, source=str(path), source_sha256=digest(path),
                matched_protocol_source=str(reference_path), matched_sha256=digest(reference_path),
                protocol_fields_verified=list(INVARIANTS), means_and_sample_sd_recomputed=True)
            if validator is not None:
                item["artifacts"] = verify_memory(cell, payload, validator)
            audit.append(item)
            sources.append((dataset, arm, path))
    if missing and not args.allow_partial:
        raise SystemExit(f"Missing results: {missing}")
    args.output.mkdir(parents=True, exist_ok=False)
    for dataset, arm, path in sources:
        dest = args.output / "sources" / dataset / arm
        dest.mkdir(parents=True)
        shutil.copy2(path, dest / "results.json")
        migration = path.parent / "migration.json"
        if migration.exists():
            shutil.copy2(migration, dest / migration.name)
        for source in sorted(path.parent.glob("s*/construction.json")):
            shutil.copy2(source, dest / f"{source.parent.name}_construction.json")
    shutil.copy2(__file__, args.output / Path(__file__).name)
    write_csv(args.output / "summary.csv", summary)
    write_csv(args.output / "student_scores.csv", students)
    write_csv(args.output / "stage_scores.csv", stages)
    write_json(args.output / "verification.json", dict(passed=True, completed=len(summary),
        expected=10, missing=missing, memory_artifacts_verified=validator is not None,
        summarizer_sha256=digest(Path(__file__)), sources=audit))
    lines = ["# Small-graph baseline completion", "",
        f"Completed: {len(summary)}/10 method-dataset entries.", "",
        "Each entry uses one class order and one construction seed, with Student seeds 150/151/152.",
        "Values are mean +/- sample SD across Students, not nine-order/construction statistics.",
        "Common-Student references use the historical test mask (not a new blind test).",
        "Inspired references use report-dev and a 1024-step legacy Student; do not rank them with the common Student.", "",
        "| Dataset | Method | Split | Macro-F1 | Accuracy | Build seconds | Peak MiB |",
        "|---|---|---|---:|---:|---:|---:|"]
    for v in summary:
        lines.append(f"| {v['dataset']} | {v['method']} | {v['split']} "
            f"| {v['f1_mean']:.2f} +/- {v['f1_student_sd']:.2f} "
            f"| {v['accuracy_mean']:.2f} +/- {v['accuracy_student_sd']:.2f} "
            f"| {v['construction_seconds']:.2f} | {v['peak_allocated_mib']:.1f} |")
    lines += ["", "TAGSAM keeps the native PMI selector with a bounded 100-by-10 response loop.",
        "STAD uses the pinned native core with local Qwen2.5-1.5B and internal candidate validation.",
        "Both change the original access/training interface and are qualified adaptations, not official reproductions.",
        "SR-GM-inspired projects conflicts in shared 128-D latent space because raw feature/text dimensions differ.",
        "Budget, stream and mask hashes match the order-9101 primary records. Failed outputs are retained.",
        "TAGSAM stage zero was migrated only after checking its protocol, memory hash and completed Student scores.",
        "Construction times include the reused original stage zero; peak is max allocated CUDA memory across stages."]
    if missing:
        lines += ["", "Pending: " + ", ".join(missing)]
    (args.output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
