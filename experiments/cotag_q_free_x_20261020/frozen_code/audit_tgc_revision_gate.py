"""Independent audit for the TGC-R Phase A six-arm matrix."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

import run_small_comparison_audited as C
import tgc_revision_gate as TGC


ARMS = ("c0", "naive_refresh", "c0m", "tgc_r", "tgc_no_preserve", "tgc_shuffled")
TGC_ARMS = {"tgc_r", "tgc_no_preserve", "tgc_shuffled"}
IMMUTABLE = ("text_emb", "mass", "depth", "stage")


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def fail(errors: list[str], message: str) -> None:
    errors.append(message)


def audit_cell(cell: Path) -> dict:
    errors: list[str] = []
    manifest = read(cell / "manifest.json")
    arms = manifest.get("arms", [])
    if (not arms or arms[0] != "c0" or len(arms) != len(set(arms))
            or any(a not in ARMS for a in arms)
            or arms != manifest["protocol"]["arms"]):
        fail(errors, "arm registration mismatch")
    stages = len(manifest["stage_classes"])
    seeds = [int(v) for v in manifest["student_seeds"]]
    results = read(cell / "results.json")
    if results.get("manifest") != manifest:
        fail(errors, "results manifest mismatch")
    completion = read(cell / "completion.json")
    expected_fits = len(arms) * stages * len(seeds)
    if completion.get("status") != "complete":
        fail(errors, "incomplete cell")
    if int(completion.get("fits", -1)) != expected_fits:
        fail(errors, f"fit count mismatch: {completion.get('fits')} != {expected_fits}")
    if int(completion.get("expected_fits", -1)) != expected_fits:
        fail(errors, "declared expected fit count mismatch")

    memories = {arm: [] for arm in arms}
    stage_audits = []
    for stage in range(stages):
        loaded = {}
        for arm in arms:
            memory_path = cell / f"{arm}_s{stage}_memory.pt"
            audit_path = cell / f"{arm}_s{stage}_construction.json"
            if not memory_path.exists() or not audit_path.exists():
                fail(errors, f"missing artifact: {arm} s{stage}")
                continue
            memory = torch.load(memory_path, map_location="cpu", weights_only=False)
            audit = read(audit_path)
            loaded[arm] = memory
            memories[arm].append(memory)
            if audit.get("memory_sha256") != TGC.memory_hash(memory):
                fail(errors, f"memory hash mismatch: {arm} s{stage}")
            if audit.get("tensor_sha256") != C.tensor_hash(memory):
                fail(errors, f"tensor hash mismatch: {arm} s{stage}")
            if int(audit.get("payload", {}).get("rows", -1)) != len(memory["x"]):
                fail(errors, f"payload rows mismatch: {arm} s{stage}")
            if len(memory["x"]) != int(manifest["budgets"][stage]):
                fail(errors, f"budget mismatch: {arm} s{stage}")
            if stage > 0:
                previous = memories[arm][stage - 1]
                n = len(previous["x"])
                old_edges = memory["edge_index"][:, (memory["edge_index"] < n).all(0)]
                if not torch.equal(old_edges, previous["edge_index"]):
                    fail(errors, f"old-old edges changed: {arm} s{stage}")
        if len(loaded) != len(arms):
            continue
        c0 = loaded["c0"]
        new_rows = int(manifest["budgets"][stage]) - (
            int(manifest["budgets"][stage - 1]) if stage else 0)
        old_n = len(c0["x"]) - new_rows
        for arm in arms[1:]:
            memory = loaded[arm]
            for key in ("x", "text_emb", "mass", "depth", "stage"):
                if not torch.equal(c0[key][old_n:], memory[key][old_n:]):
                    fail(errors, f"incoming block changed: {arm} s{stage} {key}")
            if c0["texts"][old_n:] != memory["texts"][old_n:]:
                fail(errors, f"incoming block changed: {arm} s{stage} texts")
        for arm in TGC_ARMS.intersection(arms):
            if not torch.equal(c0["edge_index"], loaded[arm]["edge_index"]):
                fail(errors, f"TGC topology differs from C0: {arm} s{stage}")
            audit = read(cell / f"{arm}_s{stage}_construction.json")["build"]
            if stage == 0:
                continue
            revision = audit.get("revision", {})
            if revision.get("teacher_access") != [
                    "previous_synthetic_memory", "current_raw_delta"]:
                fail(errors, f"teacher access mismatch: {arm} s{stage}")
            updated = set(int(v) for v in revision.get("updated_rows", []))
            if len(updated) > int(revision.get("revision_quota", 0)):
                fail(errors, f"revision quota exceeded: {arm} s{stage}")
            previous = memories[arm][stage - 1]
            cfg = TGC.RevisionConfig(**manifest["protocol"]["teacher"])
            if revision.get("config") != cfg.dict():
                fail(errors, f"revision configuration mismatch: {arm} s{stage}")
            previous_n = len(previous["x"])
            revised = {k: loaded[arm][k][:previous_n] for k in
                       ("x", "text_emb", "mass", "depth", "stage", "texts")}
            revised["edge_index"] = loaded[arm]["edge_index"][
                :, (loaded[arm]["edge_index"] < previous_n).all(0)]
            try:
                TGC.assert_revision_invariants(previous, revised, dict(revision), cfg)
            except AssertionError as exc:
                fail(errors, f"revision invariant: {arm} s{stage}: {exc}")
            expected_core = TGC.protected_rows(previous, cfg.protected_fraction).tolist()
            if revision.get("protected_rows") != expected_core:
                fail(errors, f"protected core mismatch: {arm} s{stage}")
            quota = min(previous_n - len(expected_core),
                        math.floor(previous_n * cfg.revision_fraction_of_old))
            if revision.get("revision_quota") != quota:
                fail(errors, f"effective quota mismatch: {arm} s{stage}")
            certificate = cell / audit["certificate_file"]
            if not certificate.exists() or C.sha(certificate) != audit["certificate_sha256"]:
                fail(errors, f"certificate hash mismatch: {arm} s{stage}")
            else:
                cert = torch.load(certificate, map_location="cpu", weights_only=False)
                ids = torch.tensor(sorted(manifest["stage_source_ids"][stage]), dtype=torch.long)
                if quota and not torch.equal(cert["delta_ids"], ids):
                    fail(errors, f"teacher accessed wrong delta: {arm} s{stage}")
            protected = set(int(v) for v in revision.get("protected_rows", []))
            for row in protected:
                if not torch.equal(previous["x"][row], loaded[arm]["x"][row]):
                    fail(errors, f"protected row changed: {arm} s{stage} row {row}")
            for key in IMMUTABLE:
                if not torch.equal(previous[key], loaded[arm][key][:len(previous["x"])]):
                    fail(errors, f"immutable old field changed: {arm} s{stage} {key}")
            if previous["texts"] != loaded[arm]["texts"][:len(previous["x"])]:
                fail(errors, f"old text changed: {arm} s{stage}")
        stage_audits.append({"stage": stage, "rows": len(c0["x"]),
                             "new_rows": new_rows})

    expected_keys = {(arm, stage, seed) for arm in arms
                     for stage in range(stages) for seed in seeds}
    observed = set()
    for entry in results.get("results", []):
        arm, stage = entry["arm"], int(entry["stage"])
        for fit in entry.get("per_seed", []):
            seed = int(fit["seed"])
            key = (arm, stage, seed)
            if key in observed:
                fail(errors, f"duplicate fit: {key}")
            observed.add(key)
            row_path = cell / f"{arm}_s{stage}_seed{seed}.json"
            checkpoint = cell / f"{arm}_s{stage}_seed{seed}.pt"
            if not row_path.exists() or not checkpoint.exists():
                fail(errors, f"missing fit artifact: {key}")
                continue
            if read(row_path) != fit:
                fail(errors, f"fit JSON mismatch: {key}")
            memory = memories[arm][stage]
            if fit.get("steps") != C.P.student_steps(len(memory["x"]), override=
                    manifest["protocol"]["student"]["steps_override"]):
                fail(errors, f"student steps mismatch: {key}")
            if fit.get("support_sha256") != C.tensor_hash(memory):
                fail(errors, f"support hash mismatch: {key}")
            if fit.get("support_memory_sha256") != TGC.memory_hash(memory):
                fail(errors, f"support memory hash mismatch: {key}")
            if C.sha(checkpoint) != fit.get("checkpoint_sha256"):
                fail(errors, f"checkpoint hash mismatch: {key}")
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if C.tensor_hash(saved["state"]) != fit.get("end_state_sha256"):
                fail(errors, f"checkpoint state mismatch: {key}")
    if observed != expected_keys:
        fail(errors, f"fit key set mismatch: observed={len(observed)} expected={len(expected_keys)}")
    return {"cell": str(cell), "dataset": manifest["dataset"],
            "order": manifest["order"], "passed": not errors,
            "errors": errors, "stage_audits": stage_audits,
            "fit_count": len(observed)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    root = Path(args.root)
    cells = sorted(p for p in root.glob("*/*") if (p / "manifest.json").exists())
    reports = [audit_cell(cell) for cell in cells]
    protocol = read(root / "protocol.json")
    expected = {(d, o) for d in protocol.get("datasets", ["cora", "citeseer"])
                for o in protocol["orders"]}
    observed = {(r["dataset"], r["order"]) for r in reports}
    complete = observed == expected if "datasets" in protocol else bool(reports)
    output = {"root": str(root), "cells": reports,
              "matrix_complete": complete,
              "fits": sum(r["fit_count"] for r in reports),
              "passed": complete and bool(reports) and all(r["passed"] for r in reports)}
    Path(args.output).write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(json.dumps({"cells": len(reports), "fits": output["fits"],
                      "passed": output["passed"],
                      "errors": sum(len(r["errors"]) for r in reports)}))
    if not output["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
