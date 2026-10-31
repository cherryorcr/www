"""Independent old-row relational revision with displacement-matched controls."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path

import torch

import run_tgc_r4_relational as D

R, T, M, S, B = D.R, D.T, D.M, D.S, D.B
read, load, save = D.read, D.load, D.save
DATASETS = D.DATASETS
ORDERS = D.CONFIRM_ORDERS
SEEDS = tuple(range(120, 126))
ARMS = ("c0", "relation_old", "random_old", "shuffled_old")
CFG = replace(M.Config(), teacher_seeds=(8200, 8201), train_seeds=(8300, 8301),
              validation_seeds=(8310, 8311), probe_seed=8290)


def configure_shared_runner():
    D.ARMS, D.ORDERS, D.SEEDS = ARMS, ORDERS, SEEDS


def protocol():
    return json.loads(json.dumps({
        "name": "tgc-r5-old-confirmation-v1", "datasets": DATASETS, "orders": ORDERS,
        "class_groups": {d: {o: R.A.stage_groups(d, 3, o) for o in ORDERS} for d in DATASETS},
        "report_seeds": SEEDS, "arms": ARMS, "config": asdict(CFG),
        "support": "lift_compact", "mask": "unchanged R4 legacy-uniform teacher utility; 80% protected",
        "history": "shared C0 parents, not recursive", "rate": .05, "stages": 3,
        "primary": "relation_old vs c0 and old-row matched-random; final all-prompt pooled report-dev F1",
        "secondary": "all three stages; pooled old/new; seen prompts; construction and student cost",
        "random": "same editable old rows; exact per-row Euclidean displacement of relation_old",
        "shuffled": "R4 target correspondence shuffle; displacement may differ and is reported",
        "student": {"batch": 16, "lr": .001, "steps": "max(50,min(2048,16*rows))"},
        "access": "old synthetic + current pool delta; no historical raw in optimization; offline loader is not OS isolation",
        "test_labels_used": False, "report_before_seal": False,
        "progression": {"mean_final_min_pp": .5, "cells_ge_minus1_min": 3,
                        "worst_final_min_pp": -2., "worst_old_min_pp": -2.,
                        "mean_new_min_pp": 0., "mean_vs_random_min_pp": 0.},
        "joint_experiment": "runs independently of old-only success; no individual-win prerequisite",
        "source_hashes": R.source_hashes(),
    }))


def prepare(data, order, args):
    ids, budgets = D.set_stream(data, order)
    out = args.output / data["dataset"] / order
    out.mkdir(parents=True, exist_ok=True)
    R.stable_lock(out / "manifest.json", {
        "dataset": data["dataset"], "order": order, "stage_classes": data["stage_classes"],
        "budgets": budgets, "stage_source_ids": [v.tolist() for v in ids],
        "source_dataset_sha256": R.C.sha(R.U.DATA_CFG[data["dataset"]]["path"]),
        "pool_sha256": R.C.tensor_hash({"pool": data["pool_mask"]}),
        "report_mask_sha256": R.C.tensor_hash({"report": data["report_dev_mask"]})})
    for stage in range(3):
        if (out / f"s{stage}_decision.json").exists():
            D.verify_decision(out, stage)
            continue
        if (args.output / "selection_seal.json").exists():
            raise ValueError("construction attempted after report seal")
        base, build = B.build_c0(data, ids[stage], budgets, stage, out, args.device)
        decision = {"stage": stage, "construction": build, "report_labels_used": False,
                    "recursive": False, "arms": {}}
        memories = {arm: base for arm in ARMS}
        if stage:
            old = load(out / f"c0_s{stage-1}_memory.pt")
            delta = T.raw_delta_block(data, ids[stage], stage)
            assignment = D.current_assignment(data, ids[stage], base, len(old["x"]), stage, args.device)
            support = M.lifted_support(old, delta, base, assignment)
            contexts, audit = M.build_context(base, old, delta, support, stage, CFG, args.device,
                                               diagnostics=False)
            save(out / f"s{stage}_support_diagnostics.json", {
                **audit, "current_delta_ids": ids[stage].tolist(),
                "assignment_sha256": R.C.tensor_hash({"q": assignment})})
            editable = audit["editable_old"]
            for arm in ("relation_old", "shuffled_old"):
                memory, opt = M.optimize(base, contexts["lift_compact"], editable, stage,
                                          "old", "relation", CFG, args.device,
                                          shuffled=arm == "shuffled_old")
                memories[arm] = memory
                save(out / f"{arm}_s{stage}_optimization.json", opt)
                decision["arms"][arm] = {"rows": editable, "seconds": opt["seconds"]}
            memories["random_old"] = S.matched_random(base, memories["relation_old"], editable, stage)
            M.validate_memory(base, memories["random_old"], editable, CFG.radius)
            decision["numerical_check"] = M.numerical_check(base, CFG.train_seeds[0], args.device)
            del contexts
        decision["memories"] = {a: D.save_memory(out, a, stage, memories[a]) for a in ARMS}
        decision["locked_at"] = datetime.now(timezone.utc).isoformat()
        save(out / f"s{stage}_decision.json", decision)
        print(json.dumps({"event": "prepared", "dataset": data["dataset"], "order": order,
                          "stage": stage}), flush=True)
    save(out / "prepared.json", {"status": "prepared"})


def seal(root):
    records = {}
    for dataset in DATASETS:
        for order in ORDERS:
            out = root / dataset / order
            assert read(out / "prepared.json")["status"] == "prepared"
            for stage in range(3):
                D.verify_decision(out, stage)
            for path in sorted(out.glob("*.json")):
                records[path.relative_to(root).as_posix()] = R.C.sha(path)
    R.stable_lock(root / "selection_seal.json", {
        "decisions": records, "protocol_sha256": R.C.sha(root / "protocol.json")})


def audit_scores(data, order, args, validate):
    from audit_tgc_r2b_student_revision import independent_scores
    D.verify_seal(args.output)
    ids, budgets = D.set_stream(data, order)
    out = args.output / data["dataset"] / order
    manifest = read(out / "manifest.json")
    assert manifest["stage_source_ids"] == [v.tolist() for v in ids]
    assert manifest["budgets"] == budgets
    assert manifest["source_dataset_sha256"] == R.C.sha(R.U.DATA_CFG[data["dataset"]]["path"])
    maximum, checked, cache = 0., 0, {}
    for stage in range(3):
        D.verify_decision(out, stage)
        validate(out, stage, budgets)
        for arm in ARMS:
            memory = load(out / f"{arm}_s{stage}_memory.pt")
            for seed in SEEDS:
                row = read(out / f"{arm}_s{stage}_seed{seed}.json")
                checkpoint = out / f"{arm}_s{stage}_seed{seed}.pt"
                assert row["checkpoint_sha256"] == R.C.sha(checkpoint)
                assert row["support_memory_sha256"] == T.memory_hash(memory)
                assert row["steps"] == R.P.student_steps(budgets[stage]) and row["seed"] == seed
                key = (stage, row["checkpoint_sha256"])
                if key not in cache:
                    model = R.G.CLIPStudent(memory["x"].shape[1]).to(args.device)
                    model.load_state_dict(load(checkpoint)["state"])
                    cache[key] = independent_scores(model, data, stage)
                    del model
                for scope in ("all", "seen"):
                    for metric, value in cache[key][scope].items():
                        if value is not None:
                            maximum = max(maximum, abs(value - row["metrics"][scope][metric]))
                checked += 1
    assert maximum < 1e-7 and checked == len(ARMS) * 3 * len(SEEDS)
    save(out / "audit.json", {"status": "passed", "fit_slots": checked,
                                "unique_reload_scores": len(cache), "maximum_metric_error_pp": maximum,
                                "test_labels_evaluated": False})


def validate(out, stage, budgets):
    base = load(out / f"c0_s{stage}_memory.pt")
    for arm in ARMS:
        memory = load(out / f"{arm}_s{stage}_memory.pt")
        assert len(memory["x"]) == budgets[stage]
        if not stage:
            assert T.memory_hash(memory) == T.memory_hash(base)
        elif arm != "c0":
            opt = read(out / f"{'relation_old' if arm == 'random_old' else arm}_s{stage}_optimization.json")
            M.validate_memory(base, memory, opt["rows"], CFG.radius)
            assert opt["nonzero_gradient_steps"] == CFG.outer_steps
            assert opt["final"]["train"]["old_kl"] <= CFG.old_epsilon + 1e-7
    if stage:
        random = load(out / f"random_old_s{stage}_memory.pt")
        rel = load(out / f"relation_old_s{stage}_memory.pt")
        torch.testing.assert_close((random["x"] - base["x"]).norm(dim=1),
                                   (rel["x"] - base["x"]).norm(dim=1), atol=2e-6, rtol=2e-5)


def main():
    configure_shared_runner()
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--phase", choices=("prepare", "seal", "evaluate", "audit"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--minilm", default="/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip")
    parser.add_argument("--text-cache-root", type=Path,
                        default=Path("/home/pc/research/TAG_condensation/results/p5_rate_curve_small_20260919"))
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
            D.verify_seal(args.output)
        data = R.R.load_data(args.dataset, args.minilm, args.device,
                             args.text_cache_root / f"{args.dataset}_r005" / "text_emb_pool_only.pt")
        for order in ORDERS:
            if args.phase == "prepare":
                prepare(data, order, args)
            elif args.phase == "evaluate":
                D.evaluate(data, order, args)
            else:
                audit_scores(data, order, args, validate)
    print(f"TGC_R5_OLD_{args.phase.upper()}_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
