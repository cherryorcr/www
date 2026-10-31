"""Audit formal CaT public-baseline construction and Student aggregates."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


DATASETS = ("cora", "citeseer")
ORDERS = ("order19611", "order19612", "order19613", "order19614")
ARMS = ("cat_payload_cctp", "cat_payload_class_shared_tagsam")
SEEDS = (150, 151, 152, 153, 154, 155)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    protocol = read(args.root / "protocol.json")
    errors = []
    checks = 0
    if tuple(protocol["datasets"]) != DATASETS:
        errors.append("dataset coverage mismatch")
    if tuple(protocol["orders"]) != ORDERS:
        errors.append("order coverage mismatch")
    if tuple(protocol["arms"]) != ARMS:
        errors.append("arm coverage mismatch")
    if tuple(protocol["report_seeds"]) != SEEDS:
        errors.append("Student seed mismatch")
    if protocol["config"]["multimodal"]["text_sources"] != 5:
        errors.append("formal text_sources is not five")
    if protocol["config"]["multimodal"]["payload_max_sources"] != 5:
        errors.append("formal payload_max_sources is not five")

    result = read(args.root / "results.json")
    result_rows = {(r["dataset"], r["order"], r["arm"], int(r["stage"])): r
                   for r in result["rows"]}
    expected = {(d, o, a, s) for d in DATASETS for o in ORDERS
                for a in ARMS for s in range(3)}
    if set(result_rows) != expected:
        errors.append("results coverage mismatch")

    for dataset in DATASETS:
        for order in ORDERS:
            construction = read(args.root / dataset / order / "construction.json")
            records = {(r["arm"], int(r["stage"])): r
                       for r in construction["records"]}
            if set(records) != {(a, s) for a in ARMS for s in range(3)}:
                errors.append(f"construction coverage mismatch: {dataset}/{order}")
                continue
            for arm in ARMS:
                parent_rows = 0
                for stage in range(3):
                    key = arm, stage
                    record = records[key]
                    row = result_rows[key := (dataset, order, arm, stage)]
                    checks += 1
                    if record["rows"] != row["memory_rows"]:
                        errors.append(f"row mismatch: {key}")
                    student_memory_hashes = {
                        s.get("memory_sha256") for s in row["per_student"]
                    }
                    if len(student_memory_hashes) != 1 or record["memory_sha256"] not in student_memory_hashes:
                        errors.append(f"memory hash mismatch: {key}")
                    if int(record["rows"]) <= parent_rows:
                        errors.append(f"non-increasing budget: {key}")
                    parent_rows = int(record["rows"])
                    seeds = tuple(int(s["seed"]) for s in row["per_student"])
                    if seeds != SEEDS:
                        errors.append(f"seed order mismatch: {key}")
                    for metric, aggregate in (("cumulative_f1", "mean_f1"),
                                               ("accuracy", "mean_accuracy")):
                        values = np.asarray([s[metric] for s in row["per_student"]], dtype=float)
                        if not np.isfinite(values).all():
                            errors.append(f"nonfinite metric: {key}/{metric}")
                        if abs(float(values.mean()) - float(row["aggregate"][aggregate])) > 1e-8:
                            errors.append(f"aggregate mismatch: {key}/{metric}")
                    audit = record["audit"]
                    if audit.get("current_delta_only") is not True:
                        errors.append(f"not current-delta-only: {key}")
                    if audit.get("historical_raw_reads") != 0:
                        errors.append(f"historical raw read: {key}")
                    for flag in ("old_memory_updated", "topology_updated", "x_updated"):
                        if audit.get(flag) is not False:
                            errors.append(f"unexpected {flag}: {key}")
                    delta_ids = set(int(v) for v in record["delta_ids"])
                    for source_ids in audit.get("text_source_ids", []):
                        if not set(int(v) for v in source_ids).issubset(delta_ids):
                            errors.append(f"text source outside delta: {key}")
                    # Check the JSON artifact hashes whenever the memory file
                    # is present in the fetched evidence package.
                    memory_path = args.root / dataset / order / record["memory_file"]
                    if memory_path.exists():
                        if sha256(memory_path) != record["file_sha256"]:
                            errors.append(f"file hash mismatch: {memory_path}")
    payload = {
        "passed": not errors,
        "checks": checks,
        "errors": errors,
        "scope": "formal two-arm construction, source boundary, budgets, seeds, aggregates; file hashes checked when memory artifacts are present",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
