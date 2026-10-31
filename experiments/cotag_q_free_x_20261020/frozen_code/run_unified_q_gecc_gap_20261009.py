"""Gated development experiment: fixed-X text controls, then one pooling change."""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import time

import torch

import run_unified_q_text_confirmation_20261008 as C
import unified_q_gap_diagnostics as D

A, R, L, V, M, P, U, S = C.A, C.R, C.L, C.V, C.M, C.P, C.U, C.S
BASELINES = ("uq_dm_edgefree", "cat_cctp", "gecc_hard")
CONTROLS = ("uq_nearest1", "uq_nearest5", "gecc_nearest1", "gecc_nearest5")
PHASE_ARMS = {"p1": "uq_message", "p1-no-prop": "uq_message_no_prop",
              "p2-paired": "uq_pair_mmd", "p2-marginal": "uq_marginal_mmd", "p2-shuffled": "uq_shuffled_mmd",
              "p3-student-mass": "uq_student_mass"}


def candidate_config(cfg, phase):
    if phase == "p0":
        return cfg, set()
    if phase in ("p1", "p1-no-prop"):
        candidate = replace(cfg, feature_pooling="message")
        fields = {"feature_pooling"}
        if phase == "p1-no-prop":
            candidate = replace(candidate, lambda_propagation=0.)
            fields.add("lambda_propagation")
        return candidate, fields
    if phase.startswith("p2-"):
        mode = phase.split("-", 1)[1]
        fields = {"lambda_joint_distribution"}
        if mode != cfg.joint_distribution_mode:
            fields.add("joint_distribution_mode")
        return replace(cfg, lambda_joint_distribution=1., joint_distribution_mode=mode), fields
    if phase == "p3-student-mass":
        return replace(cfg, loss_weighting="student_mass"), {"loss_weighting"}
    raise ValueError("unknown phase")


def check_source(protocol, args, data, ids, budgets, cfg):
    expected = dict(dataset=args.dataset, order_seed=args.order_seed,
        construction_seed=args.construction_seed, budgets=budgets,
        student_steps=args.student_steps, student_seeds=args.student_seeds,
        pool_mask_hash=V.hash_tensor(data["pool_mask"]),
        report_mask_hash=V.hash_tensor(data["report_dev_mask"]),
        stage_id_hashes=[V.hash_tensor(i) for i in ids],
        students=dict(prompt_gcn=P.model_config(), prompt_sgc=S.model_config()))
    for key, value in expected.items():
        if protocol[key] != value:
            raise ValueError(f"archive protocol mismatch: {key}")
    current = json.loads(json.dumps(asdict(cfg)))
    for key, value in protocol["q_config"].items():
        if current[key] != value:
            raise ValueError(f"archive constructor config mismatch: {key}")
    for module in (M, P, S, L, V, A):
        path = Path(module.__file__)
        if protocol["source_hashes"][path.name] != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError(f"archive dependency changed: {path.name}")
    return expected


def evaluate(memory, arm, data, args, result, path):
    digest = V.memory_hash(memory)
    for name, trainer in (("prompt_gcn", None), ("prompt_sgc", S.train_sgc_student)):
        if any(r["arm"] == arm and r["student"] == name for r in result["rows"]):
            continue
        V.sync(args.device)
        start = time.perf_counter()
        with A.preserve_rng():
            score = L.score(memory, data, 4, args.student_seeds, args.device, args.student_steps, 24, trainer=trainer)
        V.sync(args.device)
        assert digest == V.memory_hash(memory)
        result["rows"].append(dict(arm=arm, student=name, score=score, memory_hash=digest,
            memory_rows=len(memory["x"]), unique_texts=len(set(memory["texts"])),
            text_chars=sum(map(len, memory["texts"])), student_seconds=time.perf_counter() - start,
            reused=False))
        V.write_json(path, result)
        print(json.dumps(dict(event="evaluated", dataset=args.dataset, seed=args.construction_seed,
                             arm=arm, student=name, f1=score["final_f1"], accuracy=score["accuracy"])), flush=True)


def run(args):
    reference = L.read(args.reference)
    groups = V.choose_groups(reference, args.order_seed)
    data = L.load_large(args.dataset, args.minilm, args.text_cache, args.device,
        dict(groups=groups, report_dev_mod=reference.get("report_dev_mod", 3)))
    ids = L.stage_ids(data)
    budgets, _ = V.choose_budgets(reference, list(map(len, ids)), groups, "fixed_schedule", .05)
    cfg = R.q_config("uq_dm_edgefree", args.construction_seed, 30, 2048)
    source = json.loads((args.source / "results.json").read_text())
    if source["status"] != "complete":
        raise ValueError("cannot reuse incomplete source")
    common = check_source(source["protocol"], args, data, ids, budgets, cfg)
    for stage, raw in enumerate(ids):
        M.ratio_allocation(data["y"][raw], budgets[stage] - (budgets[stage-1] if stage else 0))
    archives = {arm: C.load_memory(args.source / f"final_{arm}.pt.gz") for arm in BASELINES}
    for arm, memory in archives.items():
        record = [s for s in source["stages"] if s["arm"] == arm][-1]
        if record["memory_hash"] != V.memory_hash(memory):
            raise ValueError("source archive hash mismatch")
    candidate_cfg, expected_change = candidate_config(cfg, args.phase)
    changed = {k for k in asdict(cfg) if asdict(cfg)[k] != asdict(candidate_cfg)[k]}
    if changed != expected_change:
        raise AssertionError("candidate changed more than the predeclared factor")
    protocol = dict(**common, name="uq-gecc-gap-development-v1", phase=args.phase,
        q_config=asdict(candidate_cfg), source=str(args.source), groups=groups,
        changed_constructor_fields=sorted(changed),
        source_results_sha256=hashlib.sha256((args.source / "results.json").read_bytes()).hexdigest(),
        code_hashes={Path(m.__file__).name: hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
                     for m in (U, D, R, A, P, M, S, L, V, C, U.J)},
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        access="stage-local current delta plus own frozen prefix; interface audit, not OS sandbox",
        evaluation="report-dev, final memory only, transductive graph; no sealed test",
        development_not_independent_confirmation=True,
        text_controls="same batched float64 nearest policy and GECC source feature space (0,.3,.9); not identical to native float32 near-tie choices",
        text_preprocessing="native adapter raw-string truncation, 320 chars, equal source quota",
        reuses_final_archive="verified frozen-prefix archives sliced at stage boundaries, never constructs from future blocks",
        residual_diagnostics="auxiliary replay, not assumed byte-identical under nondeterministic CUDA reductions; fixed-X controls use archived X exactly")
    protocol = json.loads(json.dumps(protocol))
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "results.json"
    if path.exists():
        result = json.loads(path.read_text())
        if result["protocol"] != protocol:
            raise ValueError("refusing changed protocol resume")
        if result["status"] == "complete":
            return
        raise RuntimeError("partial run retained; inspect before resuming in a new namespace")
    result = dict(protocol=protocol, stages=[], rows=[], status="running")
    for row in source["rows"]:
        if row["arm"] in BASELINES:
            result["rows"].append(dict(row, reused=True, source_results=str(args.source / "results.json")))
    V.write_json(args.output / "protocol.json", protocol)
    V.write_json(path, result)
    parents = {arm: None for arm in CONTROLS}
    candidate_parent = None
    for stage, raw in enumerate(ids):
        delta = L.delta_view(data, raw, stage)
        start = budgets[stage-1] if stage else 0
        end = budgets[stage]
        base_block = D.edgefree_slice(archives["uq_dm_edgefree"], start, end)
        original_parent = D.edgefree_slice(archives["uq_dm_edgefree"], 0, start) if start else None
        labels = base_block["labels"]
        diagnostics = []
        V.sync(args.device)
        torch.cuda.reset_peak_memory_stats(args.device)
        began = time.perf_counter()
        block, audit = U.optimize_stage(delta, original_parent if args.phase == "p0" else candidate_parent,
            labels, data["encoder"], data["prompt_emb"], candidate_cfg, args.device,
            use_boundary=False, use_topology=False,
            diagnostic_callback=lambda state: diagnostics.append(D.representation_diagnostic(state)))
        V.sync(args.device)
        record = dict(stage=stage, construction_seconds=time.perf_counter() - began,
            construction_peak_mb=torch.cuda.max_memory_allocated(args.device) / 2**20,
            diagnostic=diagnostics[0], audit=R.compact_audit(audit), checks=R.assignment_checks(audit))
        if not record["checks"]["sinkhorn_feasible"] or audit["text_hard_forward_max_error"] > 1e-4:
            raise AssertionError("Q or hard text invariant failed")
        with A.preserve_rng():
            record["checks"].update(A.validate_payload(block, delta, data["encoder"]))
        if args.phase == "p0":
            record["replay_x_max_error"] = float((block["x"] - base_block["x"]).abs().max())
            record["replay_x_relative_error"] = float((block["x"] - base_block["x"]).norm() / base_block["x"].norm().clamp_min(1e-12))
            record["replay_changed_texts"] = sum(a != b for a, b in zip(block["texts"], base_block["texts"]))
            record["replay_hash_exact"] = V.memory_hash(block) == V.memory_hash(base_block)
            archived_audit = next(s["audit"] for s in source["stages"] if s["arm"] == "uq_dm_edgefree" and s["stage"] == stage)
            record["initial_prototypes_match_archive"] = (audit["q_cost"]["prototype_source_ids"] == archived_audit["q_cost"]["prototype_source_ids"])
            features = M.propagated_features(delta, M.Config().gecc_weights, args.device).cpu()
            message = M.propagated_features(delta, (0., 1., 0.), args.device)
            record["reference_constructors"] = {}
            for base_arm in BASELINES:
                ref_block = D.edgefree_slice(archives[base_arm], start, end)
                with A.preserve_rng(), torch.no_grad():
                    geometry = D.class_geometry(delta["x"].to(args.device), message,
                        delta["y"].to(args.device), ref_block["x"].to(args.device),
                        ref_block["labels"].to(args.device), cfg.trust_radius)
                    geometry["holdout_dm"] = D.holdout_dm(message, delta["y"].to(args.device),
                        ref_block["x"].to(args.device), ref_block["labels"].to(args.device), cfg, stage)
                record["reference_constructors"][base_arm] = geometry
            record["text_checks"] = {}
            variants = {}
            with A.preserve_rng():
                for prefix, ref_arm in (("uq", "uq_dm_edgefree"), ("gecc", "gecc_hard")):
                    ref_block = D.edgefree_slice(archives[ref_arm], start, end)
                    for k, variant in D.common_nearest_payloads(delta, features, ref_block, data["encoder"]).items():
                        variants[f"{prefix}_nearest{k}"] = variant
            for arm in CONTROLS:
                ref_arm = "uq_dm_edgefree" if arm.startswith("uq_") else "gecc_hard"
                ref_block = D.edgefree_slice(archives[ref_arm], start, end)
                variant = variants[arm]
                with A.preserve_rng():
                    checks = A.validate_payload(variant, delta, data["encoder"])
                parent = parents[arm]
                memory = A.append_block(parent, variant)
                checks.update(V.validate_memory(memory, parent, end))
                checks.update(A.assert_fixed_content(ref_block, variant))
                if arm == "gecc_nearest5":
                    checks.update(native_strings_identical=variant["texts"] == ref_block["texts"],
                        native_source_ids_identical=variant["text_source_ids"] == ref_block["text_source_ids"],
                        native_changed_strings=sum(a != b for a,b in zip(variant["texts"],ref_block["texts"])),
                        native_embedding_max_error=float((variant["text_emb"] - ref_block["text_emb"]).abs().max()))
                parents[arm] = memory
                record["text_checks"][arm] = checks
        else:
            memory = A.append_block(candidate_parent, block)
            record["checks"].update(V.validate_memory(memory, candidate_parent, end))
            record["memory_hash"] = V.memory_hash(memory)
            candidate_parent = memory
        result["stages"].append(record)
        V.write_json(path, result)
        print(json.dumps(dict(event="stage", phase=args.phase, dataset=args.dataset,
                             seed=args.construction_seed, stage=stage, seconds=record["construction_seconds"])), flush=True)
    if args.phase == "p0":
        for arm, memory in parents.items():
            if arm == "gecc_nearest5" and V.memory_hash(memory) == V.memory_hash(archives["gecc_hard"]):
                for row in source["rows"]:
                    if row["arm"] == "gecc_hard":
                        result["rows"].append(dict(row, arm=arm, reused=True, exact_native_gecc_reuse=True))
                continue
            # Save only changed payload, not another copy of fixed X.
            C.save_memory(args.output / f"payload_{arm}.pt.gz", {k: memory[k] for k in A.TEXT_KEYS})
            evaluate(memory, arm, data, args, result, path)
    else:
        arm = PHASE_ARMS[args.phase]
        C.save_memory(args.output / f"final_{arm}.pt.gz", candidate_parent)
        evaluate(candidate_parent, arm, data, args, result, path)
    result["status"] = "complete"
    V.write_json(path, result)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=("p0", *PHASE_ARMS), required=True)
    p.add_argument("--dataset", choices=("wikics", "sportsfit", "bookhis"), required=True)
    for name in ("reference", "text-cache", "output", "source"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--minilm", required=True)
    p.add_argument("--order-seed", type=int, required=True)
    p.add_argument("--construction-seed", type=int, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--student-steps", type=int, default=2048)
    p.add_argument("--student-seeds", type=int, nargs="+", default=[150, 151, 152])
    torch.set_num_threads(2)
    run(p.parse_args())


if __name__ == "__main__":
    main()
