"""Independent reload audit of sealed R3 decisions, trajectories and predictions."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

import run_tgc_selector_confirmation as B
from audit_tgc_r2b_student_revision import independent_scores, require
from summarize_tgc_selector_confirmation import matrix, SLOTS

M, R, T, S, A = B.M, B.R, B.T, B.S, B.A


def audit_cell(data, order, root, device, memo):
    out = root / data["dataset"] / order
    manifest = B.read(out / "manifest.json")
    ids, budgets = B.cell_data(data, order)
    require(manifest["stage_source_ids"] == [x.tolist() for x in ids], "stage ids changed")
    require(manifest["budgets"] == budgets and manifest["stage_classes"] == M.GROUPS[data["dataset"]][order], "stream changed")
    require(manifest["source_dataset_sha256"] == R.C.sha(R.U.DATA_CFG[data["dataset"]]["path"]), "dataset changed")
    require(manifest["pool_sha256"] == R.C.tensor_hash({"pool": data["pool_mask"]}), "pool changed")
    require(manifest["report_mask_sha256"] == R.C.tensor_hash({"report": data["report_dev_mask"]}), "report split changed")
    results = B.read(out / "results.json")
    require(results["protocol"] == B.read(root / "protocol.json"), "result protocol changed")
    records = matrix(results)
    seal_hash = B.verify_seal(root)
    count, executed, maximum, trajectory_checks = 0, 0, 0., 0
    for stage in (0, 1, 2):
        decision = B.validate_decision(out, stage)
        memories = {arm: B.load(out / f"{arm}_s{stage}_memory.pt") for arm in B.ARMS}
        base = memories["c0"]
        require(len(base["x"]) == budgets[stage], "row budget changed")
        require(not decision["selector_labels_used"] and decision["historical_raw_reads"] == 0, "access flags")
        require(T.memory_hash(memories["stop"]) == T.memory_hash(base), "stop is not no-op")
        if stage:
            old = B.load(out / f"c0_s{stage-1}_memory.pt")
            R.C.assert_frozen(old, base)
            support = T._concat_support(old, T.raw_delta_block(data, ids[stage], stage))
            require(decision["support_hash"] == T.memory_hash(support) and decision["old_hash"] == T.memory_hash(old), "illegal/mismatched probe")
            opt = decision["optimization"]
            require(opt["config"] == B.read(root / "protocol.json")["meta_config"], "meta config changed")
            require(opt["warmup_steps"] == 64 and opt["horizon"] == 1, "generator changed")
            require(opt["nonzero_gradient_steps"] == 16 and decision["stop_audit"]["nonzero_gradient_steps"] == 0, "hypergradient/stop")
            require(opt["input_memory_sha256"] == T.memory_hash(base) and opt["probe_sha256"] == T.memory_hash(support), "optimizer input changed")
            require(not opt["report_labels_used"] and opt["historical_raw_reads"] == 0, "optimizer labels")
            require(S.acceptance(opt["selected_validation"], opt["baseline_probe"]["validation"], M.DEFAULT_CFG) == opt["accepted"], "old gate recomputation")
            best = min((r for r in opt["curve"] if "validation_objective" in r), key=lambda r: r["validation_objective"])
            require(best["step"] == opt["selected_outer_step"] and best["validation_probe"] == opt["selected_validation"], "outer checkpoint selection")
            prefix = decision["prefix_audit"]
            require(R.C.sha(out / prefix["checkpoint_file"]) == prefix["checkpoint_sha256"], "prefix artifact changed")
            prefixes = B.load(out / prefix["checkpoint_file"])
            require(set(prefixes) == set((*M.DEFAULT_CFG.train_seeds, *M.DEFAULT_CFG.validation_seeds)), "meta seeds changed")
            require({str(s): A.prefix_hash(p) for s, p in prefixes.items()} == opt["prefix_hashes"], "prefix optimizer input mismatch")
            for seed, saved in prefixes.items():
                ref = A.make_prefix(base["x"].to(device), R.C.training_adjacency(base, device), base["text_emb"].to(device), seed, 64)
                require(torch.equal(saved["rng"], ref["rng"]), "prefix sequence changed")
                for group in ("params", "first", "second"):
                    for key, value in saved[group].items():
                        torch.testing.assert_close(value, ref[group][key].cpu(), atol=2e-5, rtol=2e-4)
            recomputed = M.decision_summary(opt, decision["curve"])
            require(recomputed["policies"] == decision["policies"], "selector decision changed")
            batches = M.probe_batches(len(old["x"]), len(support["x"]), stage)
            require({k: [b.tolist() for b in v] for k, v in batches.items()} == decision["curve"]["probe_batches"], "probe plan changed")
            for row in decision["curve"]["per_seed"]:
                path = out / row["checkpoint_file"]
                require(R.C.sha(path) == row["checkpoint_sha256"], "selector checkpoint changed")
                trajectories = B.load(path)
                for arm in ("c0", "candidate"):
                    saved = trajectories[arm]
                    reference = M.scratch_trajectory(memories[arm], support, batches, row["seed"], device)
                    require(saved["batch_sequence"] == reference["batch_sequence"] and saved["start_hash"] == reference["start_hash"], "selector scratch RNG changed")
                    require(saved["curve"] == row[arm], "stored response curve differs")
                    for point, refpoint in zip(saved["curve"], reference["curve"]):
                        step = point["step"]
                        checkpoint = saved["states"][step]["prefix"]
                        require(point["prefix_hash"] == A.prefix_hash(checkpoint), "selector state hash")
                        for group in ("params", "first", "second"):
                            for key, value in checkpoint[group].items():
                                torch.testing.assert_close(value, reference["states"][step]["prefix"][group][key], atol=2e-5, rtol=2e-4)
                        for key in ("old", "new", "mixed"):
                            require(abs(point[key] - refpoint[key]) < 2e-5, "response loss replay mismatch")
                        trajectory_checks += 1
            for arm, memory in memories.items():
                inv = S.invariants(base, memory, decision["editable_rows"], len(old["x"]), decision["protected_rows"], .1)
                require(inv == decision["invariants"][arm], "memory bound mismatch")
            torch.testing.assert_close((memories["candidate"]["x"] - base["x"]).norm(dim=1),
                                       (memories["random"]["x"] - base["x"]).norm(dim=1), atol=1e-5, rtol=1e-5)
        for policy in M.POLICIES:
            require(T.memory_hash(memories[policy]) == T.memory_hash(memories[decision["policies"][policy]["selected_arm"]]), "policy memory mismatch")
        for arm in B.ARMS:
            actual = []
            for seed in B.REPORT_SEEDS:
                path = out / f"{arm}_s{stage}_seed{seed}.json"
                row, checkpoint = B.read(path), B.load(path.with_suffix(".pt"))
                require(row["seed"] == seed and row["stage"] == stage and row["arm"] == arm, "report key")
                require(row["selection_seal_sha256"] == seal_hash, "report preceded seal")
                require(row["support_memory_sha256"] == T.memory_hash(memories[arm]), "report memory mismatch")
                require(row["support_sha256"] == R.C.tensor_hash(memories[arm]), "report tensor hash")
                require(row["steps"] == R.P.student_steps(budgets[stage]) and row["effective_batch"] == min(16, budgets[stage]), "training schedule")
                require(row["checkpoint_sha256"] == R.C.sha(path.with_suffix(".pt")), "report checkpoint mismatch")
                require(row["end_state_sha256"] == R.C.tensor_hash(checkpoint["state"]), "report state mismatch")
                require(row["start_state_sha256"] == B.read(out / f"c0_s{stage}_seed{seed}.json")["start_state_sha256"], "unpaired report initialization")
                if row["executed_here"]:
                    require(path.with_suffix(".pt").stat().st_mtime >= (root / "selection_seal.json").stat().st_mtime, "fit before all decisions sealed")
                else:
                    source = Path(row["reused_from"])
                    require(row["reused_source_sha256"] == R.C.sha(source), "alias provenance mismatch")
                    original = B.read(source)
                    for key in ("metrics", "support_memory_sha256", "checkpoint_sha256", "seed", "stage", "steps"):
                        require(original[key] == row[key], "alias refit or modified result")
                key = (data["dataset"], order, stage, row["end_state_sha256"])
                if key not in memo:
                    model = R.G.CLIPStudent(data["x"].shape[1]).to(device)
                    model.load_state_dict(checkpoint["state"])
                    memo[key] = independent_scores(model, data, stage)
                    del model
                for scope, metrics in memo[key].items():
                    for metric, value in metrics.items():
                        expected = row["metrics"][scope][metric]
                        diff = 0. if value is None and expected is None else abs(value - expected)
                        require(diff <= 1e-7, f"prediction mismatch {arm}/{stage}/{seed}")
                        maximum = max(maximum, diff)
                count += 1
                executed += int(row["executed_here"])
                actual.append(row)
            require(actual == records[stage, arm]["per_seed"] and R.C.aggregate(actual) == records[stage, arm]["metrics"], "duplicate/stale aggregate")
    completion = B.read(out / "completion.json")
    require(count == completion["fit_slots"] == 162 and executed == completion["executed_fits"], "completion counts")
    return {"dataset": data["dataset"], "order": order, "records_checked": count,
            "executed_fits": executed, "max_metric_difference_pp": maximum,
            "selector_checkpoints_replayed": trajectory_checks}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--minilm", default="/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip")
    ap.add_argument("--text-cache-root", type=Path, default=Path("/home/pc/research/TAG_condensation/results/p5_rate_curve_small_20260919"))
    args = ap.parse_args()
    torch.set_num_threads(2)
    require(B.read(args.root / "protocol.json") == B.protocol(), "frozen source/protocol mismatch")
    B.verify_seal(args.root)
    cells, errors, memo = [], [], {}
    for dataset in B.DATASETS:
        data = R.R.load_data(dataset, args.minilm, args.device, args.text_cache_root / f"{dataset}_r005" / "text_emb_pool_only.pt")
        for order in B.ORDERS:
            try:
                cell = audit_cell(data, order, args.root, args.device, memo)
                cells.append(cell)
                print(cell, flush=True)
            except Exception as error:
                errors.append({"dataset": dataset, "order": order, "error": repr(error)})
    count = sum(c["records_checked"] for c in cells)
    result = {"passed": not errors and count == SLOTS, "records_checked": count,
              "unique_state_evaluations": len(memo), "cells": cells, "errors": errors,
              "test_labels_evaluated": False, "metric_tolerance_pp": 1e-7}
    B.save(args.root / "audit.json", result)
    print(result, flush=True)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
