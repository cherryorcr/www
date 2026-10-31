"""Frozen R4 shared-parent experiment: prepare, seal, evaluate, reload audit."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time

import torch

import run_tgc_selector_confirmation as B
import tgc_relational_response as M

R, T, S = M.R, M.T, M.S
DATASETS = ("cora", "citeseer")
ORDERS = ("order19341", "order19342")
CONFIRM_ORDERS = ("order19441", "order19442")
SEEDS = tuple(range(100, 106))
SUPPORT = "lift_compact"
SPECS = {
    "relation_new": ("new", "relation", True, False),
    "infonce_new": ("new", "infonce", True, False),
    "infonce_old": ("old", "infonce", True, False),
    "relation_old": ("old", "relation", True, False),
    "asm_new": ("new", "asm", True, False),
    "relation_new_free": ("new", "relation", False, False),
    "relation_joint": ("joint", "relation", True, False),
    "relation_shuffled": ("new", "relation", True, True),
}
ARMS = ("c0", "relation_new", "random_new", "stop") + tuple(a for a in SPECS if a != "relation_new")
CFG = M.Config()
read, load, save = B.read, B.load, B.save


def protocol():
    groups = {d: {o: R.A.stage_groups(d, 3, o) for o in ORDERS + CONFIRM_ORDERS} for d in DATASETS}
    for dataset in DATASETS:
        signatures = [json.dumps(v) for v in groups[dataset].values()]
        if len(set(signatures)) != len(signatures):
            raise AssertionError("development and confirmation class groups overlap")
    return json.loads(json.dumps({
        "name": "tgc-r4-relational-shared-parent-v2-foreach", "datasets": DATASETS,
        "orders": ORDERS, "confirmation_orders": CONFIRM_ORDERS, "class_groups": groups,
        "report_seeds": SEEDS, "confirmation_report_seeds": list(range(120, 126)),
        "arms": ARMS, "optimization_specs": SPECS, "config": asdict(CFG),
        "support": SUPPORT, "support_selection": "fixed before any teacher or report scores",
        "rate": .05, "stages": 3, "stage0": "unmodified C0 for every arm",
        "history": "C0 shared parent; no recursive candidate rollout in mechanism matrix",
        "student": {"batch": 16, "lr": .001, "steps": "max(50,min(2048,16*rows))",
                    "model": "canonical two-layer GCN and trainable text projection; frozen MiniLM"},
        "primary_metric": "report-dev all-prompt pooled cumulative Macro-F1 percent",
        "secondary_metrics": "seen-prompt F1; pooled old/new; stage-average; paired differences",
        "labels": "current pool class labels used only by C0 assignment; none in R4 optimization",
        "evaluation_graph": "canonical transductive full graph, including unlabeled future/holdout inputs",
        "physical_loader": "offline full fixture read; only current sliced delta enters R4, no OS-level isolation",
        "test_labels_evaluated": False, "report_before_seal": False,
        "asm_control": "existing bidirectional BCE on new-new relation submatrix; not official TAGSAM replication",
        "ablations": "free/joint/shuffled run as diagnostics even if primary fails; not posthoc main-method selection",
        "promotion": {"primary_arm": "relation_new", "mean_final_min_pp": .5,
                      "cells_ge_minus1_min": 3, "worst_final_min_pp": -2.,
                      "worst_old_min_pp": -2., "mean_new_min_pp": 0.,
                      "mean_vs_random_min_pp": 0., "max_build_seconds_per_stage": 600.,
                      "independent_confirmation_required": True},
        "conditional_followup": "primary passing -> independent shared-parent confirmation; then recursive/rate/base+4; no failed candidate promoted",
        "source_hashes": R.source_hashes(),
    }))


def set_stream(data, order):
    return R.A.set_stream(data, R.A.stage_groups(data["dataset"], 3, order), .05)


def current_assignment(data, ids, base, old_n, stage, device):
    groups = R.A.source_groups_for_c0(data, ids, len(base["x"]) - old_n, stage, device)
    if len(groups) != len(base["x"]) - old_n:
        raise AssertionError("source partition row count changed")
    index = {int(v): i for i, v in enumerate(ids.tolist())}
    assignment = torch.full((len(ids),), -1, dtype=torch.long)
    for row, group in enumerate(groups):
        positions = torch.tensor([index[i] for i in group])
        if (assignment[positions] >= 0).any():
            raise AssertionError("source group overlap")
        assignment[positions] = row
        torch.testing.assert_close(data["x"][group].float().mean(0), base["x"][old_n + row],
                                   rtol=2e-5, atol=2e-6)
        if len(group) != int(base["mass"][old_n + row]):
            raise AssertionError("source mass mismatch")
    if (assignment < 0).any():
        raise AssertionError("source partition incomplete")
    return assignment


def verify_decision(out, stage):
    decision = read(out / f"s{stage}_decision.json")
    if set(decision["memories"]) != set(ARMS):
        raise AssertionError("arm matrix changed")
    for arm, record in decision["memories"].items():
        path = out / record["file"]
        if R.C.sha(path) != record["file_sha256"] or T.memory_hash(load(path)) != record["memory_sha256"]:
            raise AssertionError(f"sealed artifact changed: {path}")
    return decision


def save_memory(out, arm, stage, memory):
    path = out / f"{arm}_s{stage}_memory.pt"
    if not path.exists():
        torch.save(memory, path)
    elif T.memory_hash(load(path)) != T.memory_hash(memory):
        raise AssertionError(f"different existing memory: {path}")
    return {"file": path.name, "file_sha256": R.C.sha(path), "memory_sha256": T.memory_hash(memory),
            "payload": R.payload(memory)}


def prepare(data, order, args):
    ids, budgets = set_stream(data, order)
    out = args.output / data["dataset"] / order
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"dataset": data["dataset"], "order": order, "stage_classes": data["stage_classes"],
                "budgets": budgets, "stage_source_ids": [v.tolist() for v in ids],
                "source_dataset_sha256": R.C.sha(R.U.DATA_CFG[data["dataset"]]["path"]),
                "pool_sha256": R.C.tensor_hash({"pool": data["pool_mask"]}),
                "report_mask_sha256": R.C.tensor_hash({"report": data["report_dev_mask"]})}
    R.stable_lock(out / "manifest.json", manifest)
    for stage in range(3):
        if (out / f"s{stage}_decision.json").exists():
            verify_decision(out, stage)
            continue
        if (args.output / "selection_seal.json").exists():
            raise ValueError("cannot construct after report seal")
        base, c0_audit = B.build_c0(data, ids[stage], budgets, stage, out, args.device)
        decision = {"dataset": data["dataset"], "order": order, "stage": stage,
                    "construction": c0_audit, "base_hash": T.memory_hash(base), "arms": {},
                    "report_labels_used": False, "recursive": False}
        memories = {"c0": base}
        if stage == 0:
            memories = {arm: base for arm in ARMS}
        else:
            old = load(out / f"c0_s{stage-1}_memory.pt")
            delta = T.raw_delta_block(data, ids[stage], stage)
            t0 = time.perf_counter()
            assignment = current_assignment(data, ids[stage], base, len(old["x"]), stage, args.device)
            lifted = M.lifted_support(old, delta, base, assignment)
            R.sync(args.device)
            support_seconds = time.perf_counter() - t0
            context_path, audit_path = out / f"s{stage}_context.pt", out / f"s{stage}_support_diagnostics.json"
            if context_path.exists() and audit_path.exists():
                contexts, context_audit = load(context_path), read(audit_path)
                if (context_audit["base_sha256"] != T.memory_hash(base)
                        or context_audit["context_file_sha256"] != R.C.sha(context_path)):
                    raise AssertionError("cached context mismatch")
            else:
                contexts, context_audit = M.build_context(base, old, delta, lifted, stage, CFG, args.device)
                torch.save(contexts, context_path)
                context_audit.update(base_sha256=T.memory_hash(base), support_materialization_seconds=support_seconds,
                                     context_file_sha256=R.C.sha(context_path),
                                     assignment_sha256=R.C.tensor_hash({"assignment": assignment}),
                                     current_delta_ids=delta["current_delta_ids"].tolist(),
                                     support_rows=len(lifted["x"]), old_rows=len(old["x"]),
                                     current_raw_rows=len(delta["x"]), compact_rows=len(base["x"]))
                save(audit_path, context_audit)
            context = contexts[SUPPORT]
            context["adj"] = context["adj"].to(args.device)
            editable = context_audit["editable_old"]
            decision.update(old_rows=len(old["x"]), editable_old=editable,
                            support_diagnostics_sha256=R.C.sha(audit_path), support=SUPPORT)
            for arm, (location, objective, preserve, shuffled) in SPECS.items():
                path, ap = out / f"{arm}_s{stage}_memory.pt", out / f"{arm}_s{stage}_optimization.json"
                if path.exists() and ap.exists():
                    memory, audit = load(path), read(ap)
                    if audit["output_sha256"] != T.memory_hash(memory) or audit["input_memory_sha256"] != T.memory_hash(base):
                        raise AssertionError("cached optimization mismatch")
                else:
                    memory, audit = M.optimize(base, context, editable, stage, location, objective, CFG,
                                               args.device, preserve=preserve, shuffled=shuffled)
                    audit["output_sha256"] = T.memory_hash(memory)
                    save_memory(out, arm, stage, memory)
                    save(ap, audit)
                memories[arm] = memory
                decision["arms"][arm] = {"optimization_file": ap.name, "optimization_sha256": R.C.sha(ap)}
                print(json.dumps({"event": "optimized", "dataset": data["dataset"], "order": order,
                                  "stage": stage, "arm": arm, "seconds": audit["seconds"],
                                  "new_initial": audit["initial"]["validation"]["new"],
                                  "new_final": audit["final"]["validation"]["new"],
                                  "feasible_scale": audit["feasibility"][-1]["scale"]}), flush=True)
            new_rows = list(range(len(old["x"]), len(base["x"])))
            memories["random_new"] = S.matched_random(base, memories["relation_new"], new_rows, stage)
            M.validate_memory(base, memories["random_new"], new_rows, CFG.radius)
            stop, stop_audit = M.optimize(base, context, editable, stage, "new", "relation", CFG, args.device, stop=True)
            memories["stop"] = stop
            save(out / f"stop_s{stage}_optimization.json", stop_audit)
            if T.memory_hash(stop) != T.memory_hash(base) or stop_audit["nonzero_gradient_steps"]:
                raise AssertionError("stop-gradient control failed")
            decision["stop_audit_sha256"] = R.C.sha(out / f"stop_s{stage}_optimization.json")
            decision["numerical_unroll"] = M.numerical_check(base, 8100, args.device, horizons=(1, 4, 32))
            del contexts, context
        decision["memories"] = {arm: save_memory(out, arm, stage, memories[arm]) for arm in ARMS}
        decision["locked_at"] = datetime.now(timezone.utc).isoformat()
        save(out / f"s{stage}_decision.json", decision)
        print(json.dumps({"event": "decision_locked", "dataset": data["dataset"], "order": order, "stage": stage}), flush=True)
    save(out / "prepared.json", {"status": "prepared", "report_started": False})


def seal(root):
    entries = {}
    for dataset in DATASETS:
        for order in ORDERS:
            out = root / dataset / order
            if read(out / "prepared.json")["status"] != "prepared":
                raise AssertionError("incomplete preparation")
            for stage in range(3):
                verify_decision(out, stage)
                paths = [out / f"s{stage}_decision.json"]
                if stage:
                    paths += [out / f"s{stage}_support_diagnostics.json"]
                    paths += [out / f"{arm}_s{stage}_optimization.json" for arm in (*SPECS, "stop")]
                for path in paths:
                    entries[path.relative_to(root).as_posix()] = R.C.sha(path)
    R.stable_lock(root / "selection_seal.json", {"decisions": entries,
                                                "protocol_sha256": R.C.sha(root / "protocol.json")})


def verify_seal(root):
    record = read(root / "selection_seal.json")
    if record["protocol_sha256"] != R.C.sha(root / "protocol.json"):
        raise AssertionError("protocol changed after seal")
    for path, sha in record["decisions"].items():
        if R.C.sha(root / path) != sha:
            raise AssertionError(f"sealed decision changed: {path}")
    return R.C.sha(root / "selection_seal.json")


def evaluate(data, order, args):
    seal_hash = verify_seal(args.output)
    set_stream(data, order)
    out, results = args.output / data["dataset"] / order, []
    for stage in range(3):
        verify_decision(out, stage)
        sources = {}
        for arm in ARMS:
            memory = load(out / f"{arm}_s{stage}_memory.pt")
            memory_hash = T.memory_hash(memory)
            rows = []
            for seed in SEEDS:
                if memory_hash in sources:
                    row = B.alias_fit(sources[memory_hash], arm, stage, seed, out)
                else:
                    row = R.fit(memory, data, stage, seed, arm, out, args.device, None)
                    row["selection_seal_sha256"] = seal_hash
                    save(out / f"{arm}_s{stage}_seed{seed}.json", row)
                rows.append(row)
            sources.setdefault(memory_hash, arm)
            results.append({"stage": stage, "arm": arm, "metrics": R.C.aggregate(rows),
                            "memory_sha256": memory_hash, "per_seed": rows})
            save(out / "progress.json", {"results": results})
            print(json.dumps({"event": "evaluated", "dataset": data["dataset"], "order": order,
                              "stage": stage, "arm": arm,
                              "f1": results[-1]["metrics"]["all"]["cumulative_f1"]["mean"]}), flush=True)
    save(out / "results.json", {"results": results, "manifest": read(out / "manifest.json")})
    save(out / "completion.json", {"status": "complete", "fit_slots": len(results) * len(SEEDS),
                                    "executed_fits": sum(r["executed_here"] for item in results for r in item["per_seed"]),
                                    "completed_at": datetime.now(timezone.utc).isoformat()})


def audit(data, order, args):
    from audit_tgc_r2b_student_revision import independent_scores
    verify_seal(args.output)
    ids, budgets = set_stream(data, order)
    out = args.output / data["dataset"] / order
    manifest = read(out / "manifest.json")
    assert manifest["stage_source_ids"] == [i.tolist() for i in ids] and manifest["budgets"] == budgets
    assert manifest["source_dataset_sha256"] == R.C.sha(R.U.DATA_CFG[data["dataset"]]["path"])
    maximum, checked, unique, memo = 0., 0, 0, {}
    for stage in range(3):
        decision = verify_decision(out, stage)
        base = load(out / f"c0_s{stage}_memory.pt")
        for arm in ARMS:
            memory = load(out / f"{arm}_s{stage}_memory.pt")
            if stage and arm in SPECS:
                opt = read(out / f"{arm}_s{stage}_optimization.json")
                M.validate_memory(base, memory, opt["rows"], CFG.radius)
                assert opt["nonzero_gradient_steps"] == CFG.outer_steps
                if opt["preserve"]:
                    assert opt["final"]["train"]["old_kl"] <= CFG.old_epsilon + 1e-7
            assert len(memory["x"]) == budgets[stage]
            for seed in SEEDS:
                row = read(out / f"{arm}_s{stage}_seed{seed}.json")
                checkpoint = out / f"{arm}_s{stage}_seed{seed}.pt"
                assert row["checkpoint_sha256"] == R.C.sha(checkpoint)
                assert row["support_memory_sha256"] == T.memory_hash(memory)
                assert row["steps"] == R.P.student_steps(len(memory["x"])) and row["seed"] == seed
                key = (stage, row["checkpoint_sha256"])
                if key not in memo:
                    model = R.G.CLIPStudent(memory["x"].shape[1]).to(args.device)
                    model.load_state_dict(load(checkpoint)["state"])
                    memo[key] = independent_scores(model, data, stage)
                    unique += 1
                    del model
                scores = memo[key]
                for scope in ("all", "seen"):
                    for metric, value in scores[scope].items():
                        if value is not None:
                            maximum = max(maximum, abs(value - row["metrics"][scope][metric]))
                checked += 1
        if stage:
            stop = read(out / f"stop_s{stage}_optimization.json")
            assert stop["nonzero_gradient_steps"] == 0
            rand = load(out / f"random_new_s{stage}_memory.pt")
            rel = load(out / f"relation_new_s{stage}_memory.pt")
            torch.testing.assert_close((rand["x"] - base["x"]).norm(dim=1),
                                       (rel["x"] - base["x"]).norm(dim=1), atol=2e-6, rtol=2e-5)
            support = read(out / f"s{stage}_support_diagnostics.json")
            assert support["current_delta_ids"] == ids[stage].tolist()
            assert all(support["supports"][name]["old_normalized_block_max_error"] <= 2e-6
                       for name in ("lift_uniform", "lift_compact"))
    assert maximum < 1e-7 and checked == len(ARMS) * 3 * len(SEEDS)
    save(out / "audit.json", {"status": "passed", "fit_slots": checked, "unique_reload_scores": unique,
                               "maximum_metric_error_pp": maximum, "test_labels_evaluated": False,
                               "physical_access_scope": "offline fixture loaded; logical delta slices checked, no OS sandbox"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--phase", choices=("prepare", "seal", "evaluate", "audit"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--minilm", default="/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip")
    parser.add_argument("--text-cache-root", type=Path, default=Path("/home/pc/research/TAG_condensation/results/p5_rate_curve_small_20260919"))
    args = parser.parse_args()
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    R.stable_lock(args.output / "protocol.json", protocol())
    if args.phase == "seal":
        seal(args.output)
    else:
        if not args.dataset:
            parser.error("dataset required")
        if args.phase in ("evaluate", "audit"):
            verify_seal(args.output)
        data = R.R.load_data(args.dataset, args.minilm, args.device,
                             args.text_cache_root / f"{args.dataset}_r005" / "text_emb_pool_only.pt")
        for order in ORDERS:
            {"prepare": prepare, "evaluate": evaluate, "audit": audit}[args.phase](data, order, args)
    print(f"TGC_R4_{args.phase.upper()}_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
