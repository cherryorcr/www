"""Verify R2B2 prefixes, unchanged memory fields, reuse provenance and predictions."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

import run_tgc_r2b_response_alignment as B
from audit_tgc_r2b_student_revision import independent_scores, require, read, load

R, T, S, A = B.R, B.T, B.S, B.A


def audit_prefixes(base, decision, device):
    results = []
    for warmup, entry in decision["prefix_audits"].items():
        path = Path(entry["checkpoint_path"])
        require(R.C.sha(path) == entry["checkpoint_sha256"], "prefix checkpoint changed")
        prefixes = load(path)
        require(set(prefixes) == {*S.Config().train_seeds, *S.Config().validation_seeds}, "prefix seeds changed")
        require(entry["input_memory_sha256"] == T.memory_hash(base), "prefix parent changed")
        require(not entry["labels_used"] and entry["historical_raw_reads"] == 0, "prefix access flags")
        x, text = base["x"].to(device), base["text_emb"].to(device)
        adj = R.C.training_adjacency(base, device)
        for seed, saved in prefixes.items():
            require(saved["steps"] == int(warmup) and saved["seed"] == seed, "prefix seed/step mismatch")
            require(A.prefix_hash(saved) == entry["prefix_hashes"][str(seed)], "prefix state hash mismatch")
            rebuilt = A.make_prefix(x, adj, text, seed, int(warmup))
            require(torch.equal(saved["rng"], rebuilt["rng"]), "prefix batch sequence changed")
            worst = 0.
            for group in ("params", "first", "second"):
                require(saved[group].keys() == rebuilt[group].keys(), "prefix parameter schema")
                for name, value in saved[group].items():
                    target = rebuilt[group][name].cpu()
                    torch.testing.assert_close(value, target, atol=2e-5, rtol=2e-4)
                    worst = max(worst, float((value - target).abs().max()))
            results.append({"warmup": int(warmup), "seed": seed, "max_state_difference": worst})
    require(len(results) == 8, "incomplete prefix bundle")
    return results


def audit_cell(data, order, args, memo):
    cell = args.root / data["dataset"] / order
    reference = args.reference_root / data["dataset"] / order
    previous = args.previous_root / data["dataset"] / order
    manifest, protocol = read(reference / "manifest.json"), read(cell / "protocol.json")
    B.validate_data(data, manifest)
    require(protocol["source_hashes"] == B.source_hashes(), "frozen source hashes changed")
    require(protocol["reference_manifest_sha256"] == R.C.sha(reference / "manifest.json"), "manifest changed")
    require(protocol["previous_audit_sha256"] == R.C.sha(args.previous_root / "audit.json"), "previous audit changed")
    require(read(args.previous_root / "audit.json")["passed"], "previous audit did not pass")
    require(protocol["outer"] == read(previous / "protocol.json")["outer"], "outer configuration changed")
    require(protocol["report_seeds"] == list(B.B.REPORT_SEEDS), "report seeds changed")
    require(not protocol["selector_labels_used"] and not protocol["test_labels_evaluated"], "label access")
    require(protocol["warmups"] == list(B.WARMUPS) and protocol["horizons"] == list(B.HORIZONS), "matrix changed")
    require(protocol["arms"] == list(B.ARMS), "protocol arms changed")
    ids, budgets = R.A.set_stream(data, manifest["stage_classes"], .05)
    require([i.tolist() for i in ids] == manifest["stage_source_ids"] and list(budgets) == manifest["budgets"], "stream changed")
    results = read(cell / "results.json")
    require(results["protocol"] == protocol, "result protocol differs")
    result_map = {(r["stage"], r["arm"]): r for r in results["results"]}
    expected = {(s, a) for s in (1, 2) for a in B.ARMS}
    require(len(results["results"]) == len(expected) and set(result_map) == expected, "incomplete result matrix")
    completion = read(cell / "completion.json")
    require(completion["status"] == "complete" and completion["fit_slots"] == 312, "completion count")
    count, executed, maximum = 0, 0, 0.
    prefix_checks, invariants, preflights = [], [], []
    for stage in (1, 2):
        base = load(reference / f"c0_s{stage}_memory.pt")
        old = load(reference / f"c0_s{stage-1}_memory.pt")
        support = T._concat_support(old, T.raw_delta_block(data, ids[stage], stage))
        decision_path = cell / f"s{stage}_decisions.json"
        decision, prior = read(decision_path), read(previous / f"s{stage}_decisions.json")
        require(decision["previous_decision_sha256"] == R.C.sha(previous / f"s{stage}_decisions.json"), "previous decision changed")
        require(decision["base_hash"] == T.memory_hash(base) and decision["old_memory_hash"] == T.memory_hash(old), "parent mismatch")
        require(decision["support_hash"] == T.memory_hash(support), "probe access mismatch")
        require(not decision["report_labels_used"] and decision["historical_raw_reads"] == 0, "revision access flags")
        for key in ("editable_rows", "protected_rows"):
            require(decision[key] == prior[key], "editable/protected row set changed")
        require([a["arm"] for a in decision["arms"]] == list(B.ARMS), "decision arm order changed")
        arms = {a["arm"]: a for a in decision["arms"]}
        memories = {name: load(rec["memory_path"]) for name, rec in arms.items()}
        prefix_checks += [{"stage": stage, **r} for r in audit_prefixes(base, decision, args.device)]
        checks = read(cell / f"s{stage}_preflight.json")["checks"]
        require(len(checks) == 6 and {(c["warmup"], c["horizon"]) for c in checks} == {(w, k) for w in B.WARMUPS for k in B.HORIZONS}, "preflight matrix")
        preflights += [{"stage": stage, **r} for r in checks]
        for w in B.WARMUPS:
            for k in B.HORIZONS:
                suffix = f"w{w}_k{k}"
                rec = arms[f"student_{suffix}"]
                opt = rec["audit"]["optimization"]
                require(opt["input_memory_sha256"] == T.memory_hash(base) and opt["probe_sha256"] == T.memory_hash(support), "optimizer inputs")
                require(opt["config"] == protocol["outer"] and opt["horizon"] == k, "optimizer config")
                require(opt["nonzero_gradient_steps"] == 16, "missing hypergradient")
                require(not opt["report_labels_used"] and opt["historical_raw_reads"] == 0, "optimizer access")
                splits, batches = S.probe_plan(len(old["x"]), len(support["x"]), stage, S.Config())
                require(opt["splits"] == {key: {part: value.tolist() for part, value in split.items()} for key, split in splits.items()}, "probe split changed")
                require(opt["probe_batches"] == {key: {part: [v.tolist() for v in values] for part, values in split.items()} for key, split in batches.items()}, "probe batches changed")
                best = min((v for v in opt["curve"] if "validation_objective" in v), key=lambda v: v["validation_objective"])
                require(best["step"] == opt["selected_outer_step"] and best["validation_probe"] == opt["selected_validation"], "selected checkpoint")
                accepted = S.acceptance(opt["selected_validation"], opt["baseline_probe"]["validation"], S.Config())
                guard = arms[f"guarded_{suffix}"]["audit"]
                source_arm = f"student_{suffix}" if accepted else "c0"
                require(opt["accepted"] == guard["accepted"] == accepted and guard["selected_arm"] == source_arm, "gate mismatch")
                require(not guard["report_labels_used"], "gate report labels")
                require(T.memory_hash(memories[f"guarded_{suffix}"]) == T.memory_hash(memories[source_arm]), "guarded memory")
                stop = arms[f"stop_{suffix}"]["audit"]["optimization"]
                require(stop["stop_gradient"] and stop["nonzero_gradient_steps"] == 0, "stop gradient")
                require(T.memory_hash(memories[f"stop_{suffix}"]) == T.memory_hash(base), "stop memory changed")
                torch.testing.assert_close((memories[f"student_{suffix}"]["x"] - base["x"]).norm(dim=1),
                                           (memories[f"random_{suffix}"]["x"] - base["x"]).norm(dim=1), atol=1e-5, rtol=1e-5)
                if w:
                    for item in (opt, stop):
                        require(item["warmup_steps"] == w and item["prefix_hashes"] == decision["prefix_audits"][str(w)]["prefix_hashes"], "shared prefix changed")
                else:
                    prior_arms = {a["arm"]: a for a in prior["arms"]}
                    require(opt == prior_arms[f"student_k{k}"]["audit"]["optimization"], "w0 optimizer changed")
                    for kind in ("student", "random"):
                        require(T.memory_hash(memories[f"{kind}_{suffix}"]) == T.memory_hash(load(previous / f"{kind}_k{k}_s{stage}_memory.pt")), "w0 memory changed")
        for arm in B.ARMS:
            memory = memories[arm]
            inv = S.invariants(base, memory, decision["editable_rows"], len(old["x"]), decision["protected_rows"], .1)
            require(inv == arms[arm]["invariants"], "saved invariant summary differs")
            full_hash, tensor_hash = T.memory_hash(memory), R.C.tensor_hash(memory)
            require(full_hash == arms[arm]["memory_sha256"] == result_map[stage, arm]["memory_sha256"], "memory hash")
            invariants.append({"stage": stage, "arm": arm, **inv})
            actual = []
            for seed in B.B.REPORT_SEEDS:
                path = cell / f"{arm}_s{stage}_seed{seed}.json"
                row, checkpoint = read(path), path.with_suffix(".pt")
                require(row["arm"] == arm and row["stage"] == stage and row["seed"] == seed, "fit key")
                require(row["support_memory_sha256"] == full_hash and row["support_sha256"] == tensor_hash, "fit memory")
                require(row["steps"] == R.P.student_steps(len(memory["x"])) and row["effective_batch"] == min(R.P.STUDENT.batch, len(memory["x"])), "fit schedule")
                require(R.C.sha(checkpoint) == row["checkpoint_sha256"], "fit checkpoint hash")
                saved = load(checkpoint)
                require(R.C.tensor_hash(saved["state"]) == row["end_state_sha256"], "fit end state")
                require(saved["result"]["start_state_sha256"] == row["start_state_sha256"] == read(cell / f"c0_s{stage}_seed{seed}.json")["start_state_sha256"], "paired scratch state")
                if row["executed_here"]:
                    require(checkpoint.stat().st_mtime >= decision_path.stat().st_mtime, "decision saved after report fit")
                else:
                    source = Path(row["reused_from"])
                    require(R.C.sha(source) == row["reused_source_sha256"], "reuse source hash")
                    original = read(source)
                    for key in ("support_memory_sha256", "support_sha256", "checkpoint_sha256", "end_state_sha256", "start_state_sha256", "metrics", "seed", "stage", "steps", "effective_batch"):
                        require(original[key] == row[key], f"reused field differs: {key}")
                key = (data["dataset"], order, stage, row["end_state_sha256"])
                if key not in memo:
                    model = R.G.CLIPStudent(data["x"].shape[1]).to(args.device)
                    model.load_state_dict(saved["state"])
                    memo[key] = independent_scores(model, data, stage)
                    del model
                for scope, metrics in memo[key].items():
                    for metric, value in metrics.items():
                        expected_value = row["metrics"][scope][metric]
                        diff = 0. if value is None and expected_value is None else abs(value - expected_value)
                        require(diff <= 1e-7, f"prediction mismatch: {path.name}/{scope}/{metric}: {diff}")
                        maximum = max(maximum, diff)
                if arm.startswith(("stop_", "guarded_")):
                    source_arm = "c0" if arm.startswith("stop_") else arms[arm]["audit"]["selected_arm"]
                    original = read(cell / f"{source_arm}_s{stage}_seed{seed}.json")
                    require(row["end_state_sha256"] == original["end_state_sha256"], "policy alias refitted")
                count += 1
                executed += int(row["executed_here"])
                actual.append(row)
            require(actual == result_map[stage, arm]["per_seed"], "result per_seed differs")
            require(R.C.aggregate(actual) == result_map[stage, arm]["metrics"], "result aggregate differs")
    require(executed == completion["executed_fits"], "executed count differs")
    return {"dataset": data["dataset"], "order": order, "records_checked": count,
            "executed_fits": executed, "max_metric_difference_pp": maximum,
            "prefix_checks": prefix_checks, "preflights": preflights, "invariants": invariants}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--reference-root", type=Path, required=True)
    ap.add_argument("--previous-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--minilm", default="/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip")
    ap.add_argument("--text-cache-root", type=Path, default=Path("/home/pc/research/TAG_condensation/results/p5_rate_curve_small_20260919"))
    args = ap.parse_args()
    torch.set_num_threads(2)
    cells, errors, memo = [], [], {}
    for dataset in ("cora", "citeseer"):
        data = R.R.load_data(dataset, args.minilm, args.device, args.text_cache_root / f"{dataset}_r005" / "text_emb_pool_only.pt")
        for order in B.B.ORDERS:
            try:
                result = audit_cell(data, order, args, memo)
                cells.append(result)
                print({"dataset": dataset, "order": order, "records": result["records_checked"]}, flush=True)
            except Exception as error:
                errors.append({"dataset": dataset, "order": order, "error": repr(error)})
    count = sum(c["records_checked"] for c in cells)
    result = {"passed": not errors and count == 1248, "records_checked": count,
              "unique_state_evaluations": len(memo), "cells": cells, "errors": errors,
              "test_labels_evaluated": False, "metric_tolerance_pp": 1e-7,
              "scoring": "Independent report-dev TP/FP/FN; all reused and new checkpoints verified"}
    R.save_json(args.output, result)
    print({k: v for k, v in result.items() if k != "cells"}, flush=True)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
